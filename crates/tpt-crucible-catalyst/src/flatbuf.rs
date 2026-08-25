//! Minimal read-only FlatBuffers cursor (shared by `.tflite` ingestion).
//!
//! Implements just enough of the wire format to walk tables, vectors and
//! strings, with every read bounds-checked so malformed files surface as
//! [`Error::ParseFormat`] rather than panics. Little-endian only, matching
//! the FlatBuffers spec default and every LiteRT producer.
//!
//! Wire-format reminders baked into these helpers:
//!
//! * a table starts with an `i32` `soffset`; the vtable lives at
//!   `table_pos - soffset` (signed, so vtables may sit before *or* after the
//!   table);
//! * the vtable is `u16` length, `u16` table length, then one `u16` per field
//!   holding the field's offset relative to the table start (`0` = absent);
//! * an "offset" field stores a `u32` relative to the offset slot itself;
//! * vectors/strings are prefixed with a `u32` element count.

use tpt_crucible_common::error::{Error, Result};

/// Diagnostic context shared by every malformed-file error from this module.
pub(crate) const PATH: &str = "<tflite>";
pub(crate) const FORMAT: &str = "tflite";

pub(crate) fn malformed(reason: &str) -> Error {
    Error::ParseFormat {
        path: PATH.into(),
        format: FORMAT.into(),
        reason: reason.into(),
    }
}

/// Read a `u16` at `pos`.
fn u16_at(b: &[u8], pos: usize) -> Result<u16> {
    let end = pos
        .checked_add(2)
        .ok_or_else(|| malformed("position overflow"))?;
    b.get(pos..end)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| malformed("truncated u16"))
}

/// Read a `u32` at `pos`.
fn u32_at(b: &[u8], pos: usize) -> Result<u32> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| malformed("position overflow"))?;
    b.get(pos..end)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| malformed("truncated u32"))
}

/// Read an `i32` at `pos`.
fn i32_at(b: &[u8], pos: usize) -> Result<i32> {
    Ok(u32_at(b, pos)? as i32)
}

/// Validate the minimum viable layout (`u32` root offset + `TFL3` file
/// identifier) and return a cursor at the root table.
pub(crate) fn root(bytes: &[u8]) -> Result<Tbl<'_>> {
    if bytes.len() < 8 {
        return Err(malformed("file shorter than the 8-byte header"));
    }
    if &bytes[4..8] != b"TFL3" {
        return Err(malformed(
            "missing `TFL3` file identifier (not a .tflite model?)",
        ));
    }
    let root_rel = u32_at(bytes, 0)? as usize;
    if root_rel < 4 || root_rel + 4 > bytes.len() {
        return Err(malformed("root table out of bounds"));
    }
    Ok(Tbl {
        b: bytes,
        pos: root_rel,
    })
}

/// Absolute position of a vector's payload start plus element count.
fn vec_at(b: &[u8], pos: usize) -> Result<(usize, usize)> {
    let len = u32_at(b, pos)? as usize;
    let start = pos
        .checked_add(4)
        .ok_or_else(|| malformed("vector position overflow"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| malformed("vector length overflow"))?;
    if end > b.len() {
        return Err(malformed("vector out of bounds"));
    }
    Ok((start, len))
}

/// A table cursor at absolute position `pos`.
#[derive(Clone, Copy)]
pub(crate) struct Tbl<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Tbl<'a> {
    /// Absolute position of field `id`'s storage slot, or `None` when absent.
    fn slot(&self, id: u16) -> Result<Option<usize>> {
        let soffset = i32_at(self.b, self.pos)?;
        let vtable = (self.pos as i64) - (soffset as i64);
        if vtable < 0 || vtable > self.b.len() as i64 {
            return Err(malformed("vtable out of bounds"));
        }
        let vtable = vtable as usize;
        let vtable_len = u16_at(self.b, vtable)? as usize;
        let entry = 4 + id as usize * 2;
        if entry + 2 > vtable_len {
            return Ok(None); // field beyond the vtable => absent (default)
        }
        let rel = u16_at(self.b, vtable + entry)? as usize;
        if rel == 0 {
            return Ok(None);
        }
        let slot = self
            .pos
            .checked_add(rel)
            .ok_or_else(|| malformed("field overflow"))?;
        if slot + 4 > self.b.len() {
            return Err(malformed("field slot out of bounds"));
        }
        Ok(Some(slot))
    }

    /// Follow an offset-type field to its target position.
    ///
    /// FlatBuffers offsets are stored relative to the offset slot itself;
    /// they are read as **signed** here so hand-assembled (forward-built)
    /// buffers whose targets precede their referents decode identically to
    /// canonical back-to-front encodings.
    fn indirect(&self, slot: usize) -> Result<usize> {
        let rel = i32_at(self.b, slot)?;
        let target = slot as i64 + rel as i64;
        if target < 0 || target >= self.b.len() as i64 {
            return Err(malformed("offset target out of bounds"));
        }
        Ok(target as usize)
    }

    /// `u8` scalar field (bools, byte enums).
    pub(crate) fn u8_field(&self, id: u16) -> Result<Option<u8>> {
        Ok(self.slot(id)?.map(|s| self.b[s]))
    }

    /// `i32`/`u32`/enum scalar field.
    pub(crate) fn i32_field(&self, id: u16) -> Result<Option<i32>> {
        match self.slot(id)? {
            Some(s) => Ok(Some(i32_at(self.b, s)?)),
            None => Ok(None),
        }
    }

    /// `f32` scalar field.
    #[allow(dead_code)]
    pub(crate) fn f32_field(&self, id: u16) -> Result<Option<f32>> {
        match self.slot(id)? {
            Some(s) => Ok(Some(f32::from_le_bytes(
                self.b[s..s + 4].try_into().unwrap(),
            ))),
            None => Ok(None),
        }
    }

    /// Nested-table field.
    pub(crate) fn table_field(&self, id: u16) -> Result<Option<Tbl<'a>>> {
        match self.slot(id)? {
            Some(s) => Ok(Some(Tbl {
                b: self.b,
                pos: self.indirect(s)?,
            })),
            None => Ok(None),
        }
    }

    /// `string` field (UTF-8 validated).
    pub(crate) fn str_field(&self, id: u16) -> Result<Option<String>> {
        match self.slot(id)? {
            Some(s) => {
                let target = self.indirect(s)?;
                let (start, len) = vec_at(self.b, target)?;
                let raw = self
                    .b
                    .get(start..start + len)
                    .ok_or_else(|| malformed("string out of bounds"))?;
                String::from_utf8(raw.to_vec())
                    .map(Some)
                    .map_err(|_| malformed("string is not valid utf-8"))
            }
            None => Ok(None),
        }
    }

    /// `[ubyte]` field returned as a slice.
    pub(crate) fn bytes_field(&self, id: u16) -> Result<Option<&'a [u8]>> {
        match self.slot(id)? {
            Some(s) => {
                let target = self.indirect(s)?;
                let (start, len) = vec_at(self.b, target)?;
                self.b
                    .get(start..start + len)
                    .map(Some)
                    .ok_or_else(|| malformed("byte vector out of bounds"))
            }
            None => Ok(None),
        }
    }

    /// `[int]` field returned as owned `i32`s.
    pub(crate) fn i32_vec_field(&self, id: u16) -> Result<Option<Vec<i32>>> {
        match self.slot(id)? {
            Some(s) => {
                let target = self.indirect(s)?;
                let (start, len) = vec_at(self.b, target)?;
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let p = start + i * 4;
                    if p + 4 > self.b.len() {
                        return Err(malformed("int vector out of bounds"));
                    }
                    out.push(i32_at(self.b, p)?);
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    /// Vector-of-tables field, mapped element-wise through `f`.
    pub(crate) fn table_vec_field<T>(
        &self,
        id: u16,
        mut f: impl FnMut(Tbl<'a>) -> Result<T>,
    ) -> Result<Option<Vec<T>>> {
        let Some(s) = self.slot(id)? else {
            return Ok(None);
        };
        let target = self.indirect(s)?;
        let (start, len) = vec_at(self.b, target)?;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let p = start + i * 4;
            if p + 4 > self.b.len() {
                return Err(malformed("table vector out of bounds"));
            }
            let elem = self.indirect(p)?;
            if elem >= self.b.len() {
                return Err(malformed("table element out of bounds"));
            }
            out.push(f(Tbl {
                b: self.b,
                pos: elem,
            })?);
        }
        Ok(Some(out))
    }

    /// Number of elements in a vector-of-tables field.
    pub(crate) fn table_vec_len(&self, id: u16) -> Result<usize> {
        match self.slot(id)? {
            Some(s) => {
                let target = self.indirect(s)?;
                Ok(vec_at(self.b, target)?.1)
            }
            None => Ok(0),
        }
    }

    /// Element `idx` of a vector-of-tables field.
    pub(crate) fn table_vec_get(&self, id: u16, idx: usize) -> Result<Option<Tbl<'a>>> {
        let Some(s) = self.slot(id)? else {
            return Ok(None);
        };
        let target = self.indirect(s)?;
        let (start, len) = vec_at(self.b, target)?;
        if idx >= len {
            return Ok(None);
        }
        let p = start + idx * 4;
        if p + 4 > self.b.len() {
            return Err(malformed("table vector out of bounds"));
        }
        let elem = self.indirect(p)?;
        if elem >= self.b.len() {
            return Err(malformed("table element out of bounds"));
        }
        Ok(Some(Tbl {
            b: self.b,
            pos: elem,
        }))
    }
}
