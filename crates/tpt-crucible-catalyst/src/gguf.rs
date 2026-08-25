//! GGUF ingestion — the llama.cpp single-file container.
//!
//! A native, dependency-free parser for **GGUF v2/v3**: magic + version,
//! key/value metadata tree, tensor directory (`name`, dims, ggml type,
//! offset), then an aligned tensor-data region. Legacy v1 files are rejected
//! with a clear error.
//!
//! Supported ggml element types map straight onto TPT-IR dtypes: `F32`,
//! `F16`, `BF16`, and the legacy block quants `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/
//! `Q8_0`. K-quants and imatrix formats surface as
//! [`Error::UnsupportedDType`] until dequantization lands (see `todo.md`).

use std::collections::BTreeMap;
use std::path::Path;

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::ops::Op;
use tpt_crucible_common::{Graph, Tensor, TensorData, TensorDesc};

/// A parsed GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    /// Unsigned byte.
    U8(u8),
    /// Unsigned 16-bit.
    U16(u16),
    /// Unsigned 32-bit.
    U32(u32),
    /// Unsigned 64-bit.
    U64(u64),
    /// Signed byte.
    I8(i8),
    /// Signed 16-bit.
    I16(i16),
    /// Signed 32-bit.
    I32(i32),
    /// Signed 64-bit.
    I64(i64),
    /// Float32.
    F32(f32),
    /// Float64.
    F64(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Homogeneous array.
    Array(Vec<GgufValue>),
    /// Nested map.
    Map(BTreeMap<String, GgufValue>),
}

macro_rules! as_int {
    ($name:ident, $ty:ty) => {
        /// Numeric view; returns `Some` for any integer/bool value that fits.
        pub fn $name(&self) -> Option<$ty> {
            match *self {
                Self::U8(v) => <$ty>::try_from(v).ok(),
                Self::U16(v) => <$ty>::try_from(v).ok(),
                Self::U32(v) => <$ty>::try_from(v).ok(),
                Self::U64(v) => <$ty>::try_from(v).ok(),
                Self::I8(v) => <$ty>::try_from(v).ok(),
                Self::I16(v) => <$ty>::try_from(v).ok(),
                Self::I32(v) => <$ty>::try_from(v).ok(),
                Self::I64(v) => <$ty>::try_from(v).ok(),
                Self::Bool(v) => Some(<$ty>::from(v)),
                _ => None,
            }
        }
    };
}

impl GgufValue {
    as_int!(as_u8, u8);
    as_int!(as_u16, u16);
    as_int!(as_u32, u32);
    as_int!(as_u64, u64);
    as_int!(as_i32, i32);

    /// Borrow the string value if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Float view covering every numeric type.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F32(v) => Some(*v as f64),
            Self::F64(v) => Some(*v),
            _ => self.as_i32().map(|i| i as f64),
        }
    }
}

impl core::fmt::Display for GgufValue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Array(items) => {
                let joined: Vec<String> = items.iter().map(ToString::to_string).collect();
                write!(f, "[{}]", joined.join(", "))
            }
            Self::Map(map) => {
                let joined: Vec<String> = map.iter().map(|(k, v)| format!("{k}={v}")).collect();
                write!(f, "{{{}}}", joined.join(", "))
            }
            other => write!(f, "{}", scalar_display(other)),
        }
    }
}

fn scalar_display(v: &GgufValue) -> String {
    match v {
        GgufValue::U8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::U64(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::F32(x) => x.to_string(),
        GgufValue::F64(x) => x.to_string(),
        GgufValue::Bool(x) => x.to_string(),
        _ => String::new(),
    }
}

/// Cursor over a GGUF byte slice.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(truncated)?;
        if end > self.data.len() {
            return Err(truncated());
        }
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| Error::ParseFormat {
            path: "<gguf>".into(),
            format: "gguf".into(),
            reason: "metadata string is not valid utf-8".into(),
        })
    }

    fn value(&mut self, depth: u32) -> Result<GgufValue> {
        const MAX_DEPTH: u32 = 8;
        if depth > MAX_DEPTH {
            return Err(Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: "metadata nesting deeper than 8 levels".into(),
            });
        }
        Ok(match self.u32()? {
            0 => GgufValue::U8(self.u8()?),
            1 => GgufValue::U16(self.u16()?),
            2 => GgufValue::U32(self.u32()?),
            3 => GgufValue::U64(self.u64()?),
            4 => GgufValue::I8(self.u8()? as i8),
            5 => GgufValue::I16(self.u16()? as i16),
            6 => GgufValue::I32(self.u32()? as i32),
            7 => GgufValue::I64(self.u64()? as i64),
            8 => GgufValue::F32(f32::from_bits(self.u32()?)),
            9 => GgufValue::F64(f64::from_bits(self.u64()?)),
            10 => GgufValue::Bool(self.u8()? != 0),
            11 => GgufValue::String(self.string()?),
            12 => {
                let _elem_type = self.u32()?;
                let count = self.u64()? as usize;
                if count > 1 << 24 {
                    return Err(Error::ParseFormat {
                        path: "<gguf>".into(),
                        format: "gguf".into(),
                        reason: format!("array of {count} entries exceeds sanity limit"),
                    });
                }
                let mut items = Vec::with_capacity(count.min(4096));
                for _ in 0..count {
                    items.push(self.value(depth + 1)?);
                }
                GgufValue::Array(items)
            }
            13 => {
                // MAP: u32 entry count, then key/value pairs.
                let count = self.u32()?;
                let mut map = BTreeMap::new();
                for _ in 0..count {
                    let key = self.string()?;
                    let v = self.value(depth + 1)?;
                    map.insert(key, v);
                }
                GgufValue::Map(map)
            }
            other => {
                return Err(Error::ParseFormat {
                    path: "<gguf>".into(),
                    format: "gguf".into(),
                    reason: format!("unknown gguf value type {other}"),
                });
            }
        })
    }

    /// Position after `align_up(pos, alignment)`.
    fn align(&mut self, alignment: u64) -> u64 {
        let rem = (self.pos as u64) % alignment.max(1);
        if rem == 0 {
            self.pos as u64
        } else {
            self.pos as u64 + alignment - rem
        }
    }
}

fn truncated() -> Error {
    Error::ParseFormat {
        path: "<gguf>".into(),
        format: "gguf".into(),
        reason: "file truncated mid-record".into(),
    }
}

/// One tensor directory entry.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name, e.g. `blk.0.attn_q.weight`.
    pub name: String,
    /// Dimensions (ne[]), fastest-varying first.
    pub dims: Vec<u64>,
    /// Raw ggml type code.
    pub ggml_type: u32,
    /// Mapped TPT-IR dtype.
    pub dtype: DType,
    /// Byte offset relative to the start of the data section.
    pub offset: u64,
    /// Payload size in bytes.
    pub size_bytes: u64,
}

/// Parsed GGUF header: metadata, tensor directory, and data-section offset.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Container version (2 or 3).
    pub version: u32,
    /// Metadata key/value tree.
    pub kv: BTreeMap<String, GgufValue>,
    /// Tensor directory in file order.
    pub tensors: Vec<GgufTensorInfo>,
    /// Absolute offset where the aligned tensor-data region starts.
    pub data_offset: u64,
}

/// Map a ggml type code onto a TPT-IR dtype.
fn ggml_type_to_dtype(code: u32) -> Result<DType> {
    Ok(match code {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::Q4_0,
        3 => DType::Q4_1,
        6 => DType::Q5_0,
        7 => DType::Q5_1,
        8 => DType::Q8_0,
        30 => DType::Bf16,
        other => {
            return Err(Error::UnsupportedDType(format!(
                "ggml type {other} (k-quants and imatrix formats are not yet supported)"
            )));
        }
    })
}

fn bad_magic() -> Error {
    Error::ParseFormat {
        path: "<gguf>".into(),
        format: "gguf".into(),
        reason: "missing GGUF magic".into(),
    }
}

/// Parse a full GGUF v2/v3 header from `data`.
pub fn parse_header(data: &[u8]) -> Result<GgufHeader> {
    let mut r = Reader::new(data);
    let magic = r.take(4).map_err(|_| bad_magic())?;
    if magic != b"GGUF" {
        return Err(bad_magic());
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(Error::ParseFormat {
            path: "<gguf>".into(),
            format: "gguf".into(),
            reason: format!("gguf v{version} is not supported (v2 and v3 accepted)"),
        });
    }

    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    const MAX_ENTRIES: u64 = 1 << 20;
    if tensor_count > MAX_ENTRIES || kv_count > MAX_ENTRIES {
        return Err(Error::ParseFormat {
            path: "<gguf>".into(),
            format: "gguf".into(),
            reason: "entry counts exceed sanity limit (2^20)".into(),
        });
    }

    let mut kv = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let value = r.value(0)?;
        kv.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims == 0 || n_dims > 8 {
            return Err(Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: format!("tensor `{name}` has implausible rank {n_dims}"),
            });
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        let dtype = ggml_type_to_dtype(ggml_type)?;
        let elements = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: format!("tensor `{name}` dimension product overflows"),
            })?;
        let size_bytes = dtype.byte_size(elements as usize) as u64;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            dtype,
            offset,
            size_bytes,
        });
    }

    let alignment = kv
        .get("general.alignment")
        .and_then(GgufValue::as_u64)
        .unwrap_or(32);
    let data_offset = r.align(alignment);

    // Bounds-check every tensor against the actual file length.
    for t in &tensors {
        let start = data_offset
            .checked_add(t.offset)
            .ok_or_else(|| Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: format!("tensor `{}` offset overflows", t.name),
            })?;
        let end = start
            .checked_add(t.size_bytes)
            .ok_or_else(|| Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: format!("tensor `{}` extent overflows", t.name),
            })?;
        if end > data.len() as u64 {
            return Err(Error::ParseFormat {
                path: "<gguf>".into(),
                format: "gguf".into(),
                reason: format!(
                    "tensor `{}` needs bytes [{start}, {end}) but file is {} bytes",
                    t.name,
                    data.len()
                ),
            });
        }
    }

    Ok(GgufHeader {
        version,
        kv,
        tensors,
        data_offset,
    })
}

/// Extract the standard Llama-family hyperparameters into graph metadata.
///
/// Keys mirror GGUF names with dots replaced by underscores
/// (`general_architecture`, `<arch>_block_count`, …) so downstream passes can
/// rely on them without re-parsing the file.
fn copy_llm_metadata(kv: &BTreeMap<String, GgufValue>, graph: &mut Graph) {
    for key in [
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
    ] {
        if let Some(v) = kv.get(key) {
            graph.set_metadata(key.replace('.', "_"), v.to_string());
        }
    }
    let arch = kv
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .map(str::to_string);
    if let Some(arch) = arch {
        for suffix in [
            "block_count",
            "context_length",
            "embedding_length",
            "attention.head_count",
            "attention.head_count_kv",
            "attention.key_length",
            "attention.value_length",
            "rope.freq_base",
        ] {
            let k = format!("{arch}.{suffix}");
            if let Some(v) = kv.get(&k) {
                graph.set_metadata(k.replace('.', "_"), v.to_string());
            }
        }
    }
}

/// Build a weight-only TPT-IR graph from GGUF bytes.
///
/// Every tensor becomes an immutable `constant` node carrying its (possibly
/// block-quantized) payload verbatim; dequantization happens in later passes.
pub fn build_graph(data: &[u8], source_name: &str) -> Result<Graph> {
    let header = parse_header(data)?;
    let mut graph = Graph::new(source_name);
    graph.set_metadata("source_format", "gguf");
    graph.set_metadata("gguf_version", header.version.to_string());
    copy_llm_metadata(&header.kv, &mut graph);

    for t in &header.tensors {
        let start = (header.data_offset + t.offset) as usize;
        let end = start + t.size_bytes as usize;
        // GGUF stores ne[] fastest-varying first; row-major IR wants the
        // reverse order.
        let desc = TensorDesc::new(
            t.dims.iter().rev().map(|&d| d as usize).collect::<Vec<_>>(),
            t.dtype,
        );
        let tensor = Tensor {
            desc,
            data: TensorData::from_vec(data[start..end].to_vec()),
        };
        graph.push(t.name.clone(), Op::Constant { tensor }, Vec::<_>::new());
    }

    graph.validate()?;
    Ok(graph)
}

/// Ingest a `.gguf` file.
pub fn ingest(path: &Path) -> Result<Graph> {
    let data = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    build_graph(&data, &name)
}

#[cfg(test)]
/// Minimal GGUF v3 fixture shared with sibling-module tests (e.g. llamafile).
pub(crate) fn fixture_gguf_v3() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&0u64.to_le_bytes()); // kv count

    // one f32 [2] tensor
    let name = b"t";
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name);
    b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    b.extend_from_slice(&2u64.to_le_bytes()); // ne[0]
    b.extend_from_slice(&0u32.to_le_bytes()); // ggml F32
    b.extend_from_slice(&0u64.to_le_bytes()); // offset

    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend_from_slice(&1.0f32.to_le_bytes());
    b.extend_from_slice(&2.0f32.to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::NodeId;

    /// Minimal but valid GGUF v3 file builder.
    ///
    /// `ggml_type` selects the element type written into the tensor directory
    /// (data bytes are always emitted for an f32-sized payload).
    fn fixture_at(version: u32, ggml_type: u32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&version.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv count

        // kv #1: general.architecture = "llama" (string)
        let key = b"general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&11u32.to_le_bytes()); // type STRING
        b.extend_from_slice(&5u64.to_le_bytes());
        b.extend_from_slice(b"llama");

        // kv #2: llama.block_count = 22 (u32)
        let key = b"llama.block_count";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&2u32.to_le_bytes()); // type U32
        b.extend_from_slice(&22u32.to_le_bytes());

        // tensor dir: one f32 [2 x 4] tensor
        let name = b"blk.0.attn_q.weight";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&2u64.to_le_bytes()); // ne[0]
        b.extend_from_slice(&4u64.to_le_bytes()); // ne[1]
        b.extend_from_slice(&ggml_type.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // pad to alignment 32 before data
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for i in 0..8u32 {
            b.extend_from_slice(&(i as f32).to_le_bytes());
        }
        b
    }

    #[test]
    fn parses_v3_fixture() {
        let data = fixture_at(3, 0);
        let header = parse_header(&data).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(
            header.kv.get("general.architecture").unwrap().as_str(),
            Some("llama")
        );
        assert_eq!(header.tensors.len(), 1);

        let t = &header.tensors[0];
        assert_eq!(t.name, "blk.0.attn_q.weight");
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.size_bytes, 32);
        assert_eq!(
            header.data_offset as usize % 32,
            0,
            "data region must be 32-byte aligned"
        );

        let g = build_graph(&data, "tiny").unwrap();
        assert_eq!(g.metadata("llama_block_count"), Some("22"));
        let node = g.get_node(NodeId(0)).unwrap();
        match &node.op {
            Op::Constant { tensor } => {
                let vals = tensor.data.to_f32(&tensor.desc).unwrap();
                assert_eq!(vals, vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
                // Row-major shape is the reverse of ne[].
                assert_eq!(tensor.desc.shape, vec![4, 2]);
            }
            other => panic!("expected constant, got {other:?}"),
        }
    }

    #[test]
    fn v2_also_supported() {
        let data = fixture_at(2, 0);
        assert_eq!(parse_header(&data).unwrap().version, 2);
    }

    #[test]
    fn v1_rejected_with_clear_error() {
        let err = parse_header(&fixture_at(1, 0)).unwrap_err();
        assert!(err.to_string().contains("v1"));
    }

    #[test]
    fn wrong_magic_rejected() {
        let mut data = fixture_at(3, 0);
        data[0] = b'X';
        let err = parse_header(&data).unwrap_err();
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn k_quant_dtype_rejected_by_name() {
        // ggml type code 12 == Q4_K (k-quant family).
        let err = parse_header(&fixture_at(3, 12)).unwrap_err();
        assert!(err.to_string().contains("k-quants"));
    }

    #[test]
    fn truncated_data_rejected() {
        let mut data = fixture_at(3, 0);
        data.truncate(data.len() - 8);
        assert!(build_graph(&data, "cut").is_err());
    }

    #[test]
    fn value_display_helpers() {
        assert_eq!(GgufValue::U32(7).as_u64(), Some(7));
        assert_eq!(GgufValue::Bool(true).as_u8(), Some(1));
        assert_eq!(GgufValue::String("hi".into()).as_str(), Some("hi"));
        assert_eq!(GgufValue::F32(0.5).as_f64(), Some(0.5));
        assert_eq!(GgufValue::U64(9).to_string(), "9");
    }
}
