//! Minimal pickle-stack interpreter for `torch.save` `data.pkl` streams.
//!
//! Implements the opcode subset PyTorch emits (protocols 2–5): scalars,
//! unicode strings, lists/dicts/tuples (both MARK-based and TUPLE1/2/3),
//! memo operations, FRAME/PROTO skips, GLOBAL/STACK_GLOBAL callables,
//! REDUCE restricted to a whitelist (`torch._utils._rebuild_tensor_v2` /
//! `_rebuild_tensor`), BUILD, and BINPERSID for storage persistent IDs.
//!
//! Everything executes eagerly into Values; nothing from the stream is
//! ever executed as code — unknown globals fail loudly instead.

use std::collections::HashMap;

use tpt_crucible_common::error::{Error, Result};

/// A decoded Python value.
///
/// Some variants (`Bool`, `Float`, `Bytes`) exist so every scalar opcode can
/// round-trip; torch state-dicts never read their payloads, hence the allow.
#[derive(Debug, Clone)]
#[allow(dead_code)] // payload-carrying variants kept for opcode completeness
pub(crate) enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// Result of `GLOBAL`/`STACK_GLOBAL` (a reference to a Python global).
    Callable(String),
    /// Materialized `_rebuild_tensor_v2(...)` result.
    Tensor(TensorMeta),
}

/// Metadata carried by a rebuilt torch tensor.
#[derive(Debug, Clone)]
pub(crate) struct TensorMeta {
    /// Storage key (the zip member under `<prefix>/data/<key>`).
    pub storage_key: String,
    /// Element offset into the storage.
    pub offset: i64,
    /// Logical shape.
    pub size: Vec<i64>,
    /// Row-major strides in elements.
    pub stride: Vec<i64>,
}

fn err(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<data.pkl>".into(),
        format: "pytorch-pickle".into(),
        reason: reason.into(),
    }
}

/// Root value plus storage table (`key -> (storage type, numel)`).
pub(crate) type Picked = (Value, HashMap<String, (String, i64)>);

/// Pickle bytecode cursor + machine state.
pub(crate) struct Vm<'a> {
    b: &'a [u8],
    p: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: HashMap<u32, Value>,
    memo_next: u32,
    /// Storage persistent-IDs seen: key -> (storage type name, numel).
    pub storages: HashMap<String, (String, i64)>,
}

impl<'a> Vm<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            b: data,
            p: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: HashMap::new(),
            memo_next: 0,
            storages: HashMap::new(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).ok_or_else(|| err("overflow"))?;
        if end > self.b.len() {
            return Err(err("truncated pickle stream"));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn varint_le(&mut self, n: usize) -> Result<u64> {
        let b = self.take(n)?;
        let mut v = 0u64;
        for (i, &byte) in b.iter().enumerate() {
            v |= (byte as u64) << (8 * i);
        }
        Ok(v)
    }

    fn line(&mut self) -> Result<Vec<u8>> {
        let start = self.p;
        while self.p < self.b.len() && self.b[self.p] != b'\n' {
            self.p += 1;
        }
        if self.p >= self.b.len() {
            return Err(err("unterminated line"));
        }
        let s = self.b[start..self.p].to_vec();
        self.p += 1; // consume \n
        Ok(s)
    }

    fn mark_pos(&self) -> Option<usize> {
        self.marks.last().copied()
    }

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().ok_or_else(|| err("stack underflow"))
    }

    fn pop_marked(&mut self) -> Result<Vec<Value>> {
        let mark = self.mark_pos().ok_or_else(|| err("no mark"))?;
        let items = self.stack.split_off(mark);
        self.marks.pop();
        Ok(items)
    }

    /// Drain values down to the nearest container (markless `SETITEMS` /
    /// `APPENDS` style used by modern picklers).
    fn drain_to_container(&mut self, kind: &str) -> Result<Vec<Value>> {
        let mut buf = Vec::new();
        loop {
            match self.stack.last() {
                Some(Value::Dict(_)) if kind == "dict" => break,
                Some(Value::List(_)) if kind == "list" => break,
                None => return Err(err(format!("no {kind} below the item run"))),
                _ => {}
            }
            buf.push(self.pop()?);
        }
        buf.reverse();
        Ok(buf)
    }

    fn memo_put(&mut self, v: Value) {
        self.memo.insert(self.memo_next, v.clone());
        self.memo_next += 1;
        self.stack.push(v);
    }
}

/// Whitelisted `REDUCE` targets (everything else is rejected by name).
const REBUILD_TARGETS: [&str; 2] = [
    "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_tensor",
];

impl<'a> Vm<'a> {
    /// Run the stream to `STOP`, returning the root value plus every storage
    /// persistent-ID seen (`key -> (storage type, numel)`).
    pub fn run(mut self) -> Result<Picked> {
        loop {
            let op = self.byte()?;
            match op {
                // --- framing / no-ops ---
                0x80 => {
                    self.byte()?; // PROTO <n>
                }
                0x95 => {
                    self.take(8)?; // FRAME <u64>
                }

                // --- scalars ---
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)),
                0x89 => self.stack.push(Value::Bool(false)),
                b'K' => {
                    let v = self.byte()? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'M' => {
                    let v = self.varint_le(2)? as i64;
                    self.stack.push(Value::Int(v));
                }
                b'J' => {
                    let v = self.varint_le(4)? as u32 as i64;
                    self.stack.push(Value::Int(v));
                }
                0x8a => {
                    // LONG1: u8 length, signed little-endian payload.
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    let mut v = 0i64;
                    for (i, &byte) in b.iter().enumerate() {
                        v |= (byte as i64) << (8 * i);
                    }
                    if n > 0 && b[n - 1] & 0x80 != 0 {
                        v |= !0i64 << (8 * n);
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    // BINFLOAT: big-endian f64.
                    let b = self.take(8)?;
                    let mut be = [0u8; 8];
                    be.copy_from_slice(b);
                    be.reverse();
                    self.stack.push(Value::Float(f64::from_be_bytes(be)));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let n = self.byte()? as usize;
                    let s = String::from_utf8(self.take(n)?.to_vec())
                        .map_err(|_| err("invalid unicode string"))?;
                    self.memo_put(Value::Str(s));
                }
                b'X' => {
                    // BINUNICODE
                    let n = self.varint_le(4)? as usize;
                    let s = String::from_utf8(self.take(n)?.to_vec())
                        .map_err(|_| err("invalid unicode string"))?;
                    self.memo_put(Value::Str(s));
                }
                b'C' => {
                    // SHORT_BINBYTES
                    let n = self.byte()? as usize;
                    let data = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(data));
                }
                b'B' => {
                    // BINBYTES
                    let n = self.varint_le(4)? as usize;
                    let data = self.take(n)?.to_vec();
                    self.stack.push(Value::Bytes(data));
                }

                // --- containers ---
                0x5d => self.memo_put(Value::List(Vec::new())), // EMPTY_LIST
                b'}' => self.memo_put(Value::Dict(Vec::new())), // EMPTY_DICT
                b')' => self.stack.push(Value::Tuple(Vec::new())), // EMPTY_TUPLE
                b'l' => {
                    // LIST (mark-based)
                    let items = self.pop_marked()?;
                    self.memo_put(Value::List(items));
                }
                b'd' => {
                    // DICT (mark-based): pairs from mark.
                    let items = self.pop_marked()?;
                    let pairs = to_pairs(items)?;
                    self.memo_put(Value::Dict(pairs));
                }
                b't' => {
                    // TUPLE (mark-based)
                    let items = self.pop_marked()?;
                    self.stack.push(Value::Tuple(items));
                }
                0x85..=0x87 => {
                    // TUPLE1 / TUPLE2 / TUPLE3
                    let n = op - 0x84;
                    if self.stack.len() < n as usize {
                        return Err(err("tuple underflow"));
                    }
                    let items = self.stack.split_off(self.stack.len() - n as usize);
                    self.stack.push(Value::Tuple(items));
                }
                b'a' => {
                    // APPEND: list.append(top)
                    let v = self.pop()?;
                    let target = last_list(&mut self.stack)?;
                    target.push(v);
                }
                b'e' => {
                    // APPENDS: mark-based, or drain to the nearest list.
                    let items = if self.marks.is_empty() {
                        self.drain_to_container("list")?
                    } else {
                        self.pop_marked()?
                    };
                    let target = last_list(&mut self.stack)?;
                    target.extend(items);
                }
                b's' => {
                    // SETITEM: single key/value pair onto the dict below.
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let target = last_dict(&mut self.stack)?;
                    target.push((k, v));
                }
                b'u' => {
                    // SETITEMS: mark-based, or drain pairs to the nearest dict.
                    let items = if self.marks.is_empty() {
                        self.drain_to_container("dict")?
                    } else {
                        self.pop_marked()?
                    };
                    let target = last_dict(&mut self.stack)?;
                    target.extend(to_pairs(items)?);
                }

                // --- memo ---
                0x94 => {
                    // MEMOIZE
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| err("memoize underflow"))?;
                    self.memo.insert(self.memo_next, v);
                    self.memo_next += 1;
                }
                b'q' => {
                    // BINPUT
                    let idx = self.byte()? as u32;
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| err("binput underflow"))?;
                    self.memo.insert(idx, v);
                }
                b'r' => {
                    // LONG_BINPUT
                    let idx = self.varint_le(4)? as u32;
                    let v = self
                        .stack
                        .last()
                        .cloned()
                        .ok_or_else(|| err("binput underflow"))?;
                    self.memo.insert(idx, v);
                }
                b'h' => {
                    // BINGET
                    let idx = self.byte()? as u32;
                    let v = self
                        .memo
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| err("binget miss"))?;
                    self.stack.push(v);
                }
                b'j' => {
                    // LONG_BINGET
                    let idx = self.varint_le(4)? as u32;
                    let v = self
                        .memo
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| err("binget miss"))?;
                    self.stack.push(v);
                }

                // --- marks / stack hygiene ---
                0x28 => self.marks.push(self.stack.len()), // MARK
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_marked()?;
                }

                // --- globals / reduction ---
                b'c' => {
                    // GLOBAL: module\n name\n
                    let module =
                        String::from_utf8(self.line()?).map_err(|_| err("bad global module"))?;
                    let name =
                        String::from_utf8(self.line()?).map_err(|_| err("bad global name"))?;
                    self.stack.push(Value::Callable(format!("{module}.{name}")));
                }
                0x93 => {
                    // STACK_GLOBAL: name str on top, module str below.
                    let name = match self.pop()? {
                        Value::Str(s) => s,
                        _ => return Err(err("stack_global name not a string")),
                    };
                    let module = match self.pop()? {
                        Value::Str(s) => s,
                        _ => return Err(err("stack_global module not a string")),
                    };
                    self.stack.push(Value::Callable(format!("{module}.{name}")));
                }
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let built = apply_reduce(&callable, &args)?;
                    if matches!(built, Value::Tensor(_) | Value::Dict(_) | Value::List(_)) {
                        self.memo_put(built);
                    } else {
                        self.stack.push(built);
                    }
                }
                b'b' => {
                    // BUILD: state ignored (torch tensors carry none here).
                    let _state = self.pop()?;
                    let obj = self.pop()?;
                    self.stack.push(obj);
                }
                b'Q' | b'P' => {
                    // BINPERSID / PERSID
                    let pid = if op == b'P' {
                        Value::Str(String::from_utf8(self.line()?).map_err(|_| err("bad persid"))?)
                    } else {
                        self.pop()?
                    };
                    self.register_persid(pid)?;
                }

                b'.' => {
                    // STOP
                    let top = self.pop()?;
                    return Ok((top, self.storages));
                }
                other => {
                    return Err(err(format!("unsupported pickle opcode {other:#04x}")));
                }
            }
        }
    }

    /// Handle a `('storage', storage_type, key, location, numel)` pid.
    fn register_persid(&mut self, pid: Value) -> Result<()> {
        let items = match pid {
            Value::Tuple(items) => items,
            _ => return Err(err("persistent id is not a tuple")),
        };
        if items.len() != 5 {
            return Err(err("storage pid must have 5 elements"));
        }
        let tag = as_str(&items[0]).ok_or_else(|| err("pid tag not a string"))?;
        if tag != "storage" {
            return Err(Error::UnsupportedOperation(format!(
                "unknown persistent-id tag `{tag}`"
            )));
        }
        let type_name = match &items[1] {
            Value::Callable(c) => c.clone(),
            _ => return Err(err("pid storage type is not a global")),
        };
        let key = as_str(&items[2])
            .ok_or_else(|| err("pid storage key not a string"))?
            .to_string();
        let numel = match &items[4] {
            Value::Int(n) => *n,
            _ => return Err(err("pid numel not an int")),
        };
        self.storages.insert(key.clone(), (type_name, numel));
        self.stack.push(Value::Tensor(TensorMeta {
            storage_key: key,
            offset: 0,
            size: vec![numel],
            stride: vec![1],
        }));
        Ok(())
    }
}

/// Interpret one whitelisted `REDUCE`.
fn apply_reduce(callable: &Value, args: &Value) -> Result<Value> {
    let name = match callable {
        Value::Callable(n) => n.as_str(),
        _ => return Err(err("reduce target is not a callable")),
    };
    let argv = match args {
        Value::Tuple(t) => t,
        _ => return Err(err("reduce args not a tuple")),
    };
    match name {
        t if REBUILD_TARGETS.contains(&t) => {
            if argv.len() < 4 {
                return Err(err("rebuild_tensor expects >= 4 args"));
            }
            let storage_key = match &argv[0] {
                Value::Tensor(m) => m.storage_key.clone(),
                _ => return Err(err("rebuild arg0 is not a storage placeholder")),
            };
            let offset = match &argv[1] {
                Value::Int(i) => *i,
                _ => return Err(err("rebuild arg1 (offset) not an int")),
            };
            let size = as_i64_list(&argv[2]).ok_or_else(|| err("rebuild size not ints"))?;
            let stride = as_i64_list(&argv[3]).ok_or_else(|| err("rebuild stride not ints"))?;
            Ok(Value::Tensor(TensorMeta {
                storage_key,
                offset,
                size,
                stride,
            }))
        }
        "collections.OrderedDict" => Ok(Value::Dict(Vec::new())),
        other => Err(Error::UnsupportedOperation(format!(
            "pickle reduce of `{other}`"
        ))),
    }
}

fn as_str(v: &Value) -> Option<&str> {
    match v {
        Value::Str(s) => Some(s),
        _ => None,
    }
}

fn as_i64_list(v: &Value) -> Option<Vec<i64>> {
    match v {
        Value::List(items) | Value::Tuple(items) => items
            .iter()
            .map(|x| match x {
                Value::Int(i) => Some(*i),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn to_pairs(items: Vec<Value>) -> Result<Vec<(Value, Value)>> {
    if items.len() % 2 != 0 {
        return Err(err("odd number of dict items"));
    }
    Ok(items
        .chunks(2)
        .map(|c| (c[0].clone(), c[1].clone()))
        .collect())
}

fn last_list(stack: &mut [Value]) -> Result<&mut Vec<Value>> {
    match stack.last_mut() {
        Some(Value::List(l)) => Ok(l),
        _ => Err(err("append target is not a list")),
    }
}

fn last_dict(stack: &mut [Value]) -> Result<&mut Vec<(Value, Value)>> {
    match stack.last_mut() {
        Some(Value::Dict(d)) => Ok(d),
        _ => Err(err("setitem target is not a dict")),
    }
}
