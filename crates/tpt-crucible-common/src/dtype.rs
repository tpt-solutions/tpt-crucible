//! Element data types supported by TPT-IR, including block-quantized formats.

use core::fmt;

use serde::{Deserialize, Serialize};

/// Element type of a [`crate::tensor::Tensor`].
///
/// Includes the legacy GGML block-quantized formats (`Q4_0` … `Q8_0`) used by
/// GGUF model files. Block-quantized dtypes pack a fixed number of elements
/// into fixed-size blocks (see [`DType::block_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    /// Boolean, one byte per element.
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    /// IEEE 754 half precision.
    F16,
    /// Brain floating point (1 sign, 8 exp, 7 mantissa).
    Bf16,
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 double precision.
    F64,
    /// 4-bit quantization with fp16 scale blocks (32 elems / 18 bytes).
    Q4_0,
    /// 4-bit quantization with scale + min blocks (32 elems / 20 bytes).
    Q4_1,
    /// 5-bit quantization (32 elems / 22 bytes).
    Q5_0,
    /// 5-bit quantization with scale + min (32 elems / 24 bytes).
    Q5_1,
    /// 8-bit quantization with fp16 scale (32 elems / 34 bytes).
    Q8_0,
}

impl DType {
    /// Canonical snake_case name (also the serde representation).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
        }
    }

    /// True for the block-quantized GGML formats.
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0
        )
    }

    /// True for floating-point types (including brain floats).
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32 | Self::F64)
    }

    /// True for signed/unsigned integers (excludes bool).
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    /// Byte size of one element for uniform dtypes; `None` for block-quantized
    /// dtypes (use [`DType::byte_size`] instead).
    pub const fn item_size(self) -> Option<usize> {
        Some(match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            _ => return None,
        })
    }

    /// `(elements_per_block, bytes_per_block)` for block-quantized dtypes.
    pub const fn block_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::Q4_0 => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            _ => None,
        }
    }

    /// Size in bytes occupied by `num_elements` elements of this dtype.
    ///
    /// Block-quantized dtypes round up to whole blocks.
    pub fn byte_size(self, num_elements: usize) -> usize {
        match self.block_layout() {
            Some((elems_per_block, bytes_per_block)) => {
                num_elements.div_ceil(elems_per_block) * bytes_per_block
            }
            None => num_elements * self.item_size().unwrap_or(1),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sizes() {
        assert_eq!(DType::F32.byte_size(100), 400);
        assert_eq!(DType::F16.byte_size(3), 6);
        assert_eq!(DType::Bool.byte_size(9), 9);
        assert_eq!(DType::U64.item_size(), Some(8));
    }

    #[test]
    fn block_quant_rounding() {
        // Q4_0: 32 elements per 18-byte block; partial blocks round up.
        assert_eq!(DType::Q4_0.byte_size(32), 18);
        assert_eq!(DType::Q4_0.byte_size(33), 36);
        assert_eq!(DType::Q4_0.byte_size(1), 18);
        assert_eq!(DType::Q8_0.byte_size(4096), 4096 / 32 * 34);
        assert_eq!(DType::Q4_0.block_layout(), Some((32, 18)));
    }

    #[test]
    fn predicates_and_names() {
        assert!(DType::Bf16.is_float());
        assert!(!DType::Bf16.is_integer());
        assert!(DType::I8.is_integer());
        assert!(DType::Q8_0.is_quantized());
        assert_eq!(DType::Q4_1.to_string(), "q4_1");
        assert_eq!(DType::F32.to_string(), "f32");
    }

    #[test]
    fn serde_snake_case_roundtrip() {
        let json = serde_json::to_string(&DType::Q8_0).unwrap();
        assert_eq!(json, "\"q8_0\"");
        let back: DType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, DType::Q8_0);
    }
}
