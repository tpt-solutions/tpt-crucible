//! Tensor descriptors, raw data buffers, and typed accessors.

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Shape + dtype metadata for a tensor (no payload).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDesc {
    /// Dimensions, row-major. An empty shape denotes a scalar.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
}

impl TensorDesc {
    /// Build a descriptor from a shape and dtype.
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
        }
    }

    /// Total element count (`1` for scalars).
    pub fn num_elements(&self) -> usize {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .unwrap_or(usize::MAX)
    }

    /// Payload size in bytes implied by this descriptor.
    ///
    /// Block-quantized dtypes round up to whole blocks.
    pub fn byte_size(&self) -> usize {
        self.dtype.byte_size(self.num_elements())
    }
}

impl core::fmt::Display for TensorDesc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.shape.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "] {}", self.dtype)
    }
}

/// Raw little-endian tensor payload.
///
/// Data is always stored little-endian regardless of host byte order, which
/// keeps TPT-IR artifacts portable across targets (x86, ARM, wasm).
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorData(Vec<u8>);

impl TensorData {
    /// Wrap an existing byte buffer.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self(data)
    }

    /// Borrow the raw bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Buffer length in bytes.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// View the payload as f32 values.
    ///
    /// Supports uniform float dtypes (`f16`, `bf16`, `f32`) reinterpreted as
    /// single precision; anything else (integers, quants) is rejected — use a
    /// dequantization pass instead.
    pub fn to_f32(&self, desc: &TensorDesc) -> Result<Vec<f32>> {
        match desc.dtype {
            DType::F32 => Ok(self
                .0
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            DType::F16 => Ok(self
                .0
                .chunks_exact(2)
                .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            DType::Bf16 => Ok(self
                .0
                .chunks_exact(2)
                .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            other => Err(Error::UnsupportedOperation(format!(
                "`{other}` tensors cannot be viewed as f32; dequantize first"
            ))),
        }
    }
}

impl core::fmt::Debug for TensorData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const PREVIEW: usize = 8;
        write!(f, "TensorData({} bytes", self.0.len())?;
        if !self.0.is_empty() {
            let n = PREVIEW.min(self.0.len());
            write!(f, ", head={:02x?}", &self.0[..n])?;
            if self.0.len() > n {
                write!(f, ", …")?;
            }
        }
        write!(f, ")")
    }
}

/// A dense tensor: descriptor plus raw payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensor {
    /// Shape and dtype.
    pub desc: TensorDesc,
    /// Little-endian payload; length must equal [`TensorDesc::byte_size`].
    pub data: TensorData,
}

impl Tensor {
    /// Assemble a tensor, validating that the buffer matches the descriptor.
    pub fn new(desc: TensorDesc, data: TensorData) -> Result<Self> {
        let expected = desc.byte_size();
        if data.len() != expected {
            return Err(Error::TensorSizeMismatch {
                name: "<tensor>".into(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { desc, data })
    }

    /// Build an f32 tensor from host-order values.
    pub fn from_f32(shape: impl Into<Vec<usize>>, values: &[f32]) -> Self {
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            desc: TensorDesc::new(shape, DType::F32),
            data: TensorData(data),
        }
    }

    /// Build an all-zero tensor of the given descriptor.
    pub fn zeros(desc: TensorDesc) -> Self {
        Self {
            data: TensorData(vec![0; desc.byte_size()]),
            desc,
        }
    }

    /// Element count (convenience).
    pub fn num_elements(&self) -> usize {
        self.desc.num_elements()
    }
}

/// Convert IEEE 754 half-precision bits to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    let f32_bits = match exp {
        // Zero / denormal: value = frac * 2^-24.
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // Shift the implicit leading 1 up to bit 10; `e` counts shifts
                // so the leading bit originally sat at position 10 - e.
                let mut e = 0u32;
                let mut m = frac;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    e += 1;
                }
                let l = 10 - e; // original leading-bit index
                                // value = (1 + mant/2^10) * 2^(l - 24)
                (sign << 31) | ((l + 103) << 23) | ((m & 0x03ff) << 13)
            }
        }
        // Inf / NaN (payload shifted into f32 mantissa).
        0x1f => (sign << 31) | (0xff << 23) | (frac << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(f32_bits)
}

/// Convert brain-float bits to f32 (exact, truncating extension).
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);
        assert_eq!(f16_bits_to_f32(0x4000), 2.0);
        assert!(f16_bits_to_f32(0x7c00).is_infinite());
        assert!(f16_bits_to_f32(0x7e00).is_nan());
        // Smallest normal f16 = 2^-14.
        assert_eq!(f16_bits_to_f32(0x0400), 2.0f32.powi(-14));
    }

    #[test]
    fn bf16_conversion() {
        let bits = (1.0f32.to_bits() >> 16) as u16;
        assert_eq!(bf16_bits_to_f32(bits), 1.0);
        assert!(bf16_bits_to_f32(0x7f80).is_infinite());
        assert!(bf16_bits_to_f32(0xffc1).is_nan());
    }

    #[test]
    fn tensor_roundtrip_f32() {
        let t = Tensor::from_f32(vec![2, 3], &[1.0, -2.0, 3.5, 0.25, 8.0, -0.125]);
        assert_eq!(t.desc.byte_size(), 24);
        let back = t.data.to_f32(&t.desc).unwrap();
        assert_eq!(back, vec![1.0, -2.0, 3.5, 0.25, 8.0, -0.125]);
    }

    #[test]
    fn tensor_size_validation() {
        let desc = TensorDesc::new(vec![4], DType::F32);
        let err = Tensor::new(desc.clone(), TensorData(vec![0; 3])).unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
        assert!(Tensor::new(desc, TensorData(vec![0; 16])).is_ok());
    }

    #[test]
    fn integer_view_rejected() {
        let mut t = Tensor::from_f32(vec![2], &[1.0, 2.0]);
        t.desc.dtype = DType::Q8_0;
        assert!(t.data.to_f32(&t.desc).is_err());
    }

    #[test]
    fn display_desc() {
        let d = TensorDesc::new(vec![7, 512], DType::F16);
        assert_eq!(d.to_string(), "[7, 512] f16");
        assert_eq!(d.num_elements(), 3584);
    }
}
