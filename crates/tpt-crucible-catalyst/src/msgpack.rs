//! Minimal read-only MessagePack decoder (shared by JAX/Flax ingestion).
//!
//! Covers the full core format (`fixint`, `fixstr`/`str8/16/32`,
//! `fixmap`/`map16/32`, `fixarray`/`array16/32`, `bin8/16/32`, `ext8/16/32`,
//! `fixext1/2/4/8/16`, floats, signed/unsigned ints, bool, nil) with every
//! read bounds-checked so malformed bundles surface as
//! [`Error::ParseFormat`] rather than panics. Extension payloads are kept
//! raw — interpretation belongs to the caller ([`crate::flax`]).

use tpt_crucible_common::error::{Error, Result};

/// Diagnostic context shared by malformed-bundle errors.
pub(crate) const PATH: &str = "<jax-flax>";
pub(crate) const FORMAT: &str = "jax-flax";

pub(crate) fn malformed(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: PATH.into(),
        format: FORMAT.into(),
        reason: reason.into(),
    }
}

/// A decoded MessagePack value.
#[derive(Debug, Clone)]
pub(crate) enum Mp {
    Nil,
    /// Decoded for completeness; Flax trees carry bools only inside chunked
    /// wrappers where the value itself is ignored.
    Bool(#[allow(dead_code)] bool),
    /// Signed integer (both unsigned and negative encodings).
    Int(i64),
    /// Decoded for completeness; float leaves carry no weights.
    F64(#[allow(dead_code)] f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Mp>),
    Map(Vec<(Mp, Mp)>),
    /// `(type code, raw payload)`; codes ≤ 127 per the spec.
    Ext(i8, Vec<u8>),
}

impl Mp {
    /// Borrow as a map's entries (the only container Flax uses for trees).
    pub(crate) fn as_map(&self) -> Option<&[(Mp, Mp)]> {
        match self {
            Mp::Map(entries) => Some(entries),
            _ => None,
        }
    }

    /// String value of a map key.
    #[allow(dead_code)]
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Mp::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Integer value of a numeric leaf.
    #[allow(dead_code)]
    pub(crate) fn as_int(&self) -> Option<i64> {
        match self {
            Mp::Int(v) => Some(*v),
            _ => None,
        }
    }
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .ok_or_else(|| malformed("length overflow"))?;
        let s = self
            .b
            .get(self.p..end)
            .ok_or_else(|| malformed("truncated message"))?;
        self.p = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_be_bytes([s[0], s[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes(s.try_into().unwrap()))
    }

    #[allow(dead_code)]
    fn u64(&mut self) -> Result<u64> {
        let s = self.take(8)?;
        Ok(u64::from_be_bytes(s.try_into().unwrap()))
    }
}

/// Decode exactly one MessagePack value covering the whole buffer (trailing
/// bytes are rejected).
pub(crate) fn parse(bytes: &[u8]) -> Result<Mp> {
    let mut r = Rd { b: bytes, p: 0 };
    let v = read_value(&mut r)?;
    if r.p != bytes.len() {
        return Err(malformed(format!(
            "{} trailing bytes after top-level value",
            bytes.len() - r.p
        )));
    }
    Ok(v)
}

fn read_value(r: &mut Rd<'_>) -> Result<Mp> {
    let marker = r.u8()?;
    Ok(match marker {
        // positive fixint
        0x00..=0x7f => Mp::Int(i64::from(marker)),
        // fixmap
        0x80..=0x8f => Mp::Map(read_pairs(r, usize::from(marker & 0x0f))?),
        // fixarray
        0x90..=0x9f => Mp::Array(read_array(r, usize::from(marker & 0x0f))?),
        // fixstr
        0xa0..=0xbf => Mp::Str(read_str(r, usize::from(marker & 0x1f))?),
        0xc0 => Mp::Nil,
        0xc1 => return Err(malformed("reserved marker 0xc1")),
        0xc2 => Mp::Bool(false),
        0xc3 => Mp::Bool(true),
        0xc4..=0xc6 => Mp::Bin(take_len_payload(r, marker - 0xc4)?),
        0xc7..=0xc9 => {
            let n = len_header(marker - 0xc7, r)?;
            let (code, data) = take_ext(r, n)?;
            Mp::Ext(code, data)
        }
        0xca => {
            let s = r.take(4)?;
            Mp::F64(f32::from_le_bytes(s.try_into().unwrap()) as f64)
        }
        0xcb => {
            let s = r.take(8)?;
            Mp::F64(f64::from_le_bytes(s.try_into().unwrap()))
        }
        0xcc => Mp::Int(i64::from(r.u8()?)),
        0xcd => Mp::Int(i64::from(r.u16()?)),
        0xce => Mp::Int(i64::from(r.u32()?)),
        0xcf => Mp::Int(u64::from_le_bytes(r.take(8)?.try_into().unwrap()) as i64),
        0xd0 => Mp::Int(i64::from(r.u8()? as i8)),
        0xd1 => Mp::Int(i64::from(r.u16()? as i16)),
        0xd2 => Mp::Int(i64::from(i32::from_le_bytes(
            r.take(4)?.try_into().unwrap(),
        ))),
        0xd3 => Mp::Int(i64::from_le_bytes(r.take(8)?.try_into().unwrap())),
        0xd4..=0xd8 => {
            let n = 1usize << (marker - 0xd4);
            let (code, data) = take_ext(r, n)?;
            Mp::Ext(code, data)
        }
        0xd9..=0xdb => {
            let n = len_header(marker - 0xd9, r)?;
            Mp::Str(read_str(r, n)?)
        }
        0xdc => {
            let n = usize::from(r.u16()?);
            Mp::Array(read_array(r, n)?)
        }
        0xdd => {
            let n = r.u32()? as usize;
            Mp::Array(read_array(r, n)?)
        }
        0xde => {
            let n = usize::from(r.u16()?);
            Mp::Map(read_pairs(r, n)?)
        }
        0xdf => {
            let n = r.u32()? as usize;
            Mp::Map(read_pairs(r, n)?)
        }
        // negative fixint
        0xe0..=0xff => Mp::Int(i64::from(marker as i8)),
    })
}

/// Element/byte length for the `bin8/16/32`, `ext8/16/32`, `str8/16/32`
/// families (`which`: 0 → 8-bit header, 1 → 16-bit, 2 → 32-bit).
fn len_header(which: u8, r: &mut Rd<'_>) -> Result<usize> {
    Ok(match which {
        0 => usize::from(r.u8()?),
        1 => usize::from(r.u16()?),
        2 => r.u32()? as usize,
        _ => unreachable!("len_header called with which > 2"),
    })
}

fn take_len_payload(r: &mut Rd<'_>, which: u8) -> Result<Vec<u8>> {
    let n = len_header(which, r)?;
    Ok(r.take(n)?.to_vec())
}

fn take_ext(r: &mut Rd<'_>, n: usize) -> Result<(i8, Vec<u8>)> {
    let code = r.u8()? as i8;
    Ok((code, r.take(n)?.to_vec()))
}

fn read_str(r: &mut Rd<'_>, n: usize) -> Result<String> {
    let s = r.take(n)?;
    String::from_utf8(s.to_vec()).map_err(|_| malformed("string is not valid utf-8"))
}

fn read_array(r: &mut Rd<'_>, n: usize) -> Result<Vec<Mp>> {
    (0..n).map(|_| read_value(r)).collect()
}

fn read_pairs(r: &mut Rd<'_>, n: usize) -> Result<Vec<(Mp, Mp)>> {
    (0..n)
        .map(|_| Ok((read_value(r)?, read_value(r)?)))
        .collect()
}
