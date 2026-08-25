//! Miniature protobuf wire-format reader shared by protobuf-based formats.
//!
//! Used by [`crate::onnx`] (ONNX `ModelProto`) and [`crate::tensorflow`]
//! (TensorFlow `SavedModel`). Carries a `(path, format)` diagnostic context so
//! decoding failures surface as [`Error::ParseFormat`] naming the container,
//! indistinguishable from the per-format parsers' own errors.

use tpt_crucible_common::error::{Error, Result};

/// Cursor over a protobuf message.
pub(crate) struct Rd<'a> {
    b: &'a [u8],
    p: usize,
    /// `(path, format)` used in [`Error::ParseFormat`] diagnostics.
    ctx: (&'static str, &'static str),
}

impl<'a> Rd<'a> {
    pub(crate) fn new(b: &'a [u8], path: &'static str, format: &'static str) -> Self {
        Self {
            b,
            p: 0,
            ctx: (path, format),
        }
    }

    /// A `ParseFormat` error bound to this cursor's container context.
    pub(crate) fn malformed(&self, reason: &str) -> Error {
        Error::ParseFormat {
            path: self.ctx.0.into(),
            format: self.ctx.1.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn eof(&self) -> bool {
        self.p >= self.b.len()
    }

    pub(crate) fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            if self.p >= self.b.len() {
                return Err(self.malformed("truncated varint"));
            }
            let byte = self.b[self.p];
            self.p += 1;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift >= 70 {
                return Err(self.malformed("varint longer than 10 bytes"));
            }
        }
    }

    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| self.malformed("length overflow"))?;
        if end > self.b.len() {
            return Err(self.malformed("truncated field"));
        }
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }

    /// Length-delimited payload (wire type 2).
    pub(crate) fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        self.take(len)
    }

    pub(crate) fn fixed32(&mut self) -> Result<f32> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn fixed64(&mut self) -> Result<f64> {
        let b = self.take(8)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn skip(&mut self, wt: u8) -> Result<()> {
        match wt {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.take(4)?;
            }
            other => return Err(self.malformed(&format!("unsupported wire type {other}"))),
        }
        Ok(())
    }
}

/// Walk every field of a protobuf message, dispatching on `(field, wire_type)`.
///
/// Handlers consume their own payloads from the cursor.
pub(crate) fn for_fields<F>(
    buf: &[u8],
    path: &'static str,
    format: &'static str,
    mut on: F,
) -> Result<()>
where
    F: FnMut(u32, u8, &mut Rd) -> Result<()>,
{
    let mut r = Rd::new(buf, path, format);
    while !r.eof() {
        let key = r.varint()?;
        let field = (key >> 3) as u32;
        let wt = (key & 7) as u8;
        if field == 0 {
            return Err(r.malformed("field number 0 is reserved"));
        }
        on(field, wt, &mut r)?;
    }
    Ok(())
}

/// Read a `string` field (wire type 2).
pub(crate) fn str_field(r: &mut Rd) -> Result<String> {
    let b = r.bytes()?;
    String::from_utf8(b.to_vec()).map_err(|_| r.malformed("field is not valid utf-8"))
}

/// Read a packed-or-unpacked `repeated int64` field.
pub(crate) fn repeated_i64(wt: u8, r: &mut Rd, out: &mut Vec<i64>) -> Result<()> {
    match wt {
        2 => {
            let b = r.bytes()?;
            let mut inner = Rd::new(b, r.ctx.0, r.ctx.1);
            while !inner.eof() {
                out.push(inner.varint()? as i64);
            }
        }
        0 => out.push(r.varint()? as i64),
        _ => return Err(r.malformed("bad wire type for int64 field")),
    }
    Ok(())
}

/// Read a packed-or-unpacked `repeated float` field.
pub(crate) fn repeated_f32(wt: u8, r: &mut Rd, out: &mut Vec<f32>) -> Result<()> {
    match wt {
        2 => {
            let b = r.bytes()?;
            if b.len() % 4 != 0 {
                return Err(r.malformed("packed float length not a multiple of 4"));
            }
            for c in b.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        5 => out.push(r.fixed32()?),
        _ => return Err(r.malformed("bad wire type for float field")),
    }
    Ok(())
}

/// Read a packed-or-unpacked `repeated double` field.
pub(crate) fn repeated_f64(wt: u8, r: &mut Rd, out: &mut Vec<f64>) -> Result<()> {
    match wt {
        2 => {
            let b = r.bytes()?;
            if b.len() % 8 != 0 {
                return Err(r.malformed("packed double length not a multiple of 8"));
            }
            for c in b.chunks_exact(8) {
                out.push(f64::from_le_bytes(c.try_into().unwrap()));
            }
        }
        1 => out.push(r.fixed64()?),
        _ => return Err(r.malformed("bad wire type for double field")),
    }
    Ok(())
}

/// Read a packed-or-unpacked `repeated bool` field (varints; TF `bool_val`).
pub(crate) fn repeated_bool(wt: u8, r: &mut Rd, out: &mut Vec<bool>) -> Result<()> {
    match wt {
        2 => {
            let b = r.bytes()?;
            out.extend(b.iter().map(|&byte| byte != 0));
        }
        0 => out.push(r.varint()? != 0),
        _ => return Err(r.malformed("bad wire type for bool field")),
    }
    Ok(())
}