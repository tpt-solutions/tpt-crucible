//! NumPy `.npy` array parsing (used by Keras v3 NPZ weight stores).
//!
//! Supports little-endian numeric dtypes in C-order; Fortran-ordered arrays
//! are rejected with a structured error. Header dictionaries are parsed
//! leniently (quoted-key extraction) rather than via a Python-literal parser.

use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// A decoded `.npy` payload.
#[derive(Debug)]
pub(crate) struct NpyArray {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Little-endian C-order payload.
    pub data: Vec<u8>,
}

fn bad(reason: impl Into<String>) -> Error {
    Error::ParseFormat {
        path: "<npy>".into(),
        format: "numpy".into(),
        reason: reason.into(),
    }
}

/// Map a NumPy `descr` string onto a TPT-IR dtype.
fn descr_to_dtype(descr: &str) -> Result<DType> {
    Ok(match descr {
        "|b1" => DType::Bool,
        "|u1" => DType::U8,
        "<u2" => DType::U16,
        "<u4" => DType::U32,
        "<i1" | "|i1" => DType::I8,
        "<i2" => DType::I16,
        "<i4" => DType::I32,
        "<i8" => DType::I64,
        "<f2" => DType::F16,
        "<f4" => DType::F32,
        "<f8" => DType::F64,
        other => {
            return Err(Error::UnsupportedDType(format!("numpy descr `{other}`")));
        }
    })
}

/// Extract a single-quoted value for `"key"` out of the header dict literal.
fn header_value<'h>(header: &'h str, key: &str) -> Option<&'h str> {
    let needle = format!("'{key}'");
    let start = header.find(&needle)? + needle.len();
    let rest = &header[start..];
    let q1 = rest.find('\'')?;
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('\'')?;
    Some(&rest[..q2])
}

/// Parse an `.npy` byte stream.
/// Parse the shape tuple out of the header dict (unquoted, unlike string
/// values, so it needs its own extraction pass).
fn header_shape(header: &str) -> Option<Vec<i64>> {
    let start = header.find("'shape'")? + "'shape':".len();
    let rest = &header[start..];
    let open = rest.find('(')?;
    let close = rest.find(')')?;
    let inner = &rest[open + 1..close];
    Some(
        inner
            .split(',')
            .filter_map(|p| p.trim().parse::<i64>().ok())
            .collect(),
    )
}
pub(crate) fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(bad("missing \\x93NUMPY magic"));
    }
    let major = bytes[6];
    let (header_len, mut pos): (usize, usize) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        other => return Err(bad(format!("unknown npy version {other}"))),
    };
    if pos + header_len > bytes.len() {
        return Err(bad("truncated npy header"));
    }
    let header = String::from_utf8_lossy(&bytes[pos..pos + header_len]).into_owned();
    pos += header_len;

    let descr = header_value(&header, "descr")
        .ok_or_else(|| bad("header missing 'descr'"))?
        .to_string();
    let dtype = descr_to_dtype(&descr)?;
    if header.contains("'fortran_order': True") {
        return Err(Error::UnsupportedOperation(
            "fortran-order arrays are not supported yet".into(),
        ));
    }

    let shape = header_shape(&header).ok_or_else(|| bad("header missing shape tuple"))?;
    let shape: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();

    let n_elems: usize = shape.iter().product();
    let expected = dtype.byte_size(n_elems);
    let data = bytes[pos..].to_vec();
    if data.len() != expected {
        return Err(bad(format!(
            "data region is {} bytes, descriptor implies {expected}",
            data.len()
        )));
    }
    Ok(NpyArray { dtype, shape, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_bytes(fortran: bool, descr: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
        // Python literals are capitalized.
        let fortran = if fortran { "True" } else { "False" };
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let mut h = format!(
            "{{'descr': '{descr}', 'fortran_order': {fortran}, 'shape': ({}, ), }}",
            shape_s.join(", ")
        )
        .into_bytes();
        h.push(b'\n');
        // Pad so that (magic + version + hlen + header) is a multiple of 64,
        // keeping the newline as the final header byte (numpy convention).
        while (h.len() + 10) % 64 != 0 {
            h.insert(h.len() - 1, b' ');
        }

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(1); // major
        out.push(0); // minor
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn parses_f32_matrix() {
        let mut data = Vec::new();
        for v in [1.0f32, 2.0, 3.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let arr = parse_npy(&npy_bytes(false, "<f4", &[1, 3], &data)).unwrap();
        assert_eq!(arr.dtype, DType::F32);
        assert_eq!(arr.shape, vec![1, 3]);
        assert_eq!(arr.data, data);
    }

    #[test]
    fn rejects_fortran_order() {
        let err = parse_npy(&npy_bytes(true, "<f4", &[2], &[0; 8])).unwrap_err();
        assert!(err.to_string().contains("fortran"));
    }

    #[test]
    fn bad_magic_rejected() {
        assert!(parse_npy(b"nope").is_err());
    }
}
