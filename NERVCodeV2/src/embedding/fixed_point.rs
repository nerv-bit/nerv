use serde::{Serialize, Deserialize};

/// Fixed-point 32.16 representation
/// Range: ±2^15, precision ≈ 1.5e-5
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fixed32_16(i64);

impl Fixed32_16 {
    pub const SCALE: i64 = 1 << 16;
    pub const EPSILON: i64 = 1; // ≈ 1.5e-5

    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Fixed32_16((x * Self::SCALE as f64).round() as i64)
    }

    #[inline]
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / Self::SCALE as f64
    }

    #[inline]
   pub fn checked_add(self, other: Self) -> Option<Self> {
    self.0.checked_add(other.0).map(FixedPoint32_16)
}

    #[inline]
    pub fn abs_diff(self, rhs: Self) -> i64 {
        (self.0 - rhs.0).abs()
    }
}