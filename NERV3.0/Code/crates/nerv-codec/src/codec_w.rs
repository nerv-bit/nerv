//! The frozen linear codec `W` and the per-leg delta computation (WP §7.2).
//!
//! Semantics frozen here (the native reference, DSR-7 — the encoder AIR must
//! agree bit-for-bit):
//!   * Weights are signed 1.15 fixed point: `w ∈ [−2^15, 2^15)`,
//!     interpreted as `w · 2^-15`. Zero bias by construction (a bare
//!     matrix; WP §7.2), so embedding accumulation is exact group
//!     arithmetic.
//!   * Per coordinate, the accumulator is exact i128 integer arithmetic
//!     over the active features only.
//!   * One named rounding point: `δ_j = round½even(Σ_i w_ji·ΔS_i / 2^15)`.
//!   * The result wraps into `ℤ/2^64` (two's complement) — the closed
//!     group of the embedding accumulator (WP §7.5).

use std::fmt;

use nerv_core::constants::W_COMMIT;
use nerv_core::fixed_point::Q15_FRAC_BITS;
use nerv_core::hash::Hash256;
use nerv_core::{round_half_even_pow2, wrap_mod_2_64};


use crate::features::{FeatureVector, FEATURE_COUNT};

/// Embedding dimension (WP §7.2/App. B: 64).
pub const EMBEDDING_DIM: usize = 64;
/// 1.15 format: 15 fractional bits (nerv-core `Q15_FRAC_BITS` — one source).
pub const WEIGHT_FRACTIONAL_BITS: u32 = Q15_FRAC_BITS;

/// Canonical serialized size: 8-byte version tag + 64 × 256 × 2-byte weights.
pub const CANONICAL_SIZE: usize = 8 + EMBEDDING_DIM * FEATURE_COUNT * 2;

// ---------------------------------------------------------------------------
// Version tag and commitment
// ---------------------------------------------------------------------------

/// The governance-epoch version tag of a frozen `W` (WP §7.6). Every leg
/// carries it ("weight-version tag"), so each delta proof pins its own `W`
/// and the Amnesia Rule holds without re-encoding state.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct WeightVersion(pub u64);

impl WeightVersion {
    pub const GENESIS: WeightVersion = WeightVersion(0);
}

impl fmt::Display for WeightVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wv{}", self.0)
    }
}

/// `Hash(W ‖ version)` (WP §7.6) — the value `params_root` commits at codec
/// adoption.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WCommitment(pub [u8; 32]);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WError {
    #[error("serialized codec has length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
}

// ---------------------------------------------------------------------------
// The codec
// ---------------------------------------------------------------------------

/// The codec `W`: a 64 × 256 matrix of signed 1.15 fixed-point weights,
/// frozen and version-tagged per governance epoch (WP §7.2, §7.6).
#[derive(Clone, PartialEq, Eq)]
pub struct CodecW {
    version: WeightVersion,
    weights: Box<[[i16; FEATURE_COUNT]; EMBEDDING_DIM]>,
}

impl fmt::Debug for CodecW {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CodecW")
            .field("version", &self.version)
            .field("commitment", &self.commitment())
            .finish()
    }
}

impl CodecW {
    /// Builds a codec from the full row-major weight matrix. Every `i16`
    /// is a valid 1.15 weight by type — the "quantization format" check of
    /// the epoch certificate (WP §7.6) is satisfied by construction here
    /// and re-checked on `parse` for ceremony inputs arriving as bytes.
    pub fn new(version: WeightVersion, weights: [[i16; FEATURE_COUNT]; EMBEDDING_DIM]) -> Self {
        Self { version, weights: Box::new(weights) }
    }

    pub fn version(&self) -> WeightVersion {
        self.version
    }

    pub fn weight(&self, row: usize, col: usize) -> i16 {
        self.weights[row][col]
    }

    /// Panics if `row >= 64`, which cannot arise from `EMBEDDING_DIM`-bounded
    /// callers.
    pub fn row(&self, row: usize) -> &[i16; FEATURE_COUNT] {
        &self.weights[row]
    }

    /// The full row-major matrix (consumed by the epoch certification in
    /// `weight_gen`).
    pub fn weights(&self) -> &[[i16; FEATURE_COUNT]; EMBEDDING_DIM] {
        &self.weights
    }

    /// Canonical serialization: version (u64 LE) followed by the 16,384
    /// weights (i16 LE, row-major) — 32,772 bytes, fixed width, no length
    /// prefix. This is the preimage hashed into the `params_root`
    /// commitment (WP §7.6).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(CANONICAL_SIZE);
        out.extend_from_slice(&self.version.0.to_le_bytes());
        for row in self.weights.iter() {
            for &w in row.iter() {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    /// Parses the canonical serialization. Fails on any length other than
    /// [`CANONICAL_SIZE`]; every parsed weight is a valid 1.15 value.
    pub fn parse(bytes: &[u8]) -> Result<Self, WError> {
        if bytes.len() != CANONICAL_SIZE {
            return Err(WError::BadLength { len: bytes.len(), expected: CANONICAL_SIZE });
        }
        let mut vb = [0u8; 8];
        vb.copy_from_slice(&bytes[..8]);
        let version = WeightVersion(u64::from_le_bytes(vb));
        let mut weights = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
        let mut off = 8;
        for row in weights.iter_mut() {
            for w in row.iter_mut() {
                let mut wb = [0u8; 2];
                wb.copy_from_slice(&bytes[off..off + 2]);
                *w = i16::from_le_bytes(wb);
                off += 2;
            }
        }
        Ok(Self { version, weights: Box::new(weights) })
    }

    /// `Hash(W ‖ version)` (WP §7.6), domain `nerv.w.commit`.
    pub fn commitment(&self) -> WCommitment {
        WCommitment(*Hash256::concat(&W_COMMIT, &self.canonical_bytes()).as_bytes())
    }


    /// `δ_leg = W · ΔS_leg ∈ (ℤ/2^64)^64` (WP §7.2) — the native reference
    /// (DSR-7). The computation is total and infallible: with `|w| ≤ 2^15`
    /// and `|ΔS| < 2^63`, even a dense 256-term accumulator is bounded by
    /// `2^86 ≪ 2^127`, so no i128 overflow is reachable on any input
    /// vector; the final wrap_mod_2_64 is the ℤ/2^64 wrap — nerv-core's frozen canonical delta rule, not an error path." 
    ///  Admissibility of `features` is the caller's (and the
    /// circuit's, statement 6) precondition; `apply` is well-defined
    /// regardless.
    pub fn apply(&self, features: &FeatureVector) -> Delta {
        let active: Vec<(usize, i64)> = features.active_features().collect();
        let mut out = [0u64; EMBEDDING_DIM];
        for (r, row) in self.weights.iter().enumerate() {
            let mut acc: i128 = 0;
            for &(i, x) in active.iter() {
                acc += i128::from(row[i]) * i128::from(x);
            }
            out[r] = wrap_mod_2_64(round_half_even_pow2(acc, WEIGHT_FRACTIONAL_BITS));
        }
        Delta(out)
    }
}

// ---------------------------------------------------------------------------
// Delta
// ---------------------------------------------------------------------------

/// `δ_leg ∈ (ℤ/2^64)^64` (WP §7.2) — the leg's sealed, digitized delta;
/// also the type of block aggregates `Δ_B` and of embedding updates.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Delta(pub [u64; EMBEDDING_DIM]);

impl Default for Delta {
    fn default() -> Self {
        Delta([0; EMBEDDING_DIM])
    }
}

impl nerv_core::codec::Encode for Delta {
    fn encode_into(&self, out: &mut Vec<u8>) {
        for v in &self.0 {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        EMBEDDING_DIM * 8
    }
}

impl nerv_core::codec::Decode for Delta {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let mut arr = [0u64; EMBEDDING_DIM];
        for slot in arr.iter_mut() {
            *slot = r.read_u64()?;
        }
        Ok(Delta(arr))
    }
}

impl Delta {
    /// WP §5.1 statement 8: `δ_leg ≠ 0` — every provable transaction moves
    /// the public index. A sanity check, explicitly not collision
    /// resistance (WP §7.4).
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&x| x == 0)
    }

    /// Exact closed-group addition (WP §7.5: `e_{t+1} = e_t + Δ_B mod 2^64`).
    pub fn wrapping_add(&self, rhs: &Delta) -> Delta {
        let mut out = [0u64; EMBEDDING_DIM];
        for (o, (a, b)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = a.wrapping_add(*b);
        }
        Delta(out)
    }

    /// Canonical serialization: 64 little-endian u64 coordinates (512 bytes).
    pub fn canonical_bytes(&self) -> [u8; EMBEDDING_DIM * 8] {
        let mut out = [0u8; EMBEDDING_DIM * 8];
        for (i, &v) in self.0.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn from_canonical_bytes(bytes: &[u8; EMBEDDING_DIM * 8]) -> Self {
        let mut out = [0u64; EMBEDDING_DIM];
        for (i, v) in out.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            *v = u64::from_le_bytes(b);
        }
        Delta(out)
    }
}



// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::features::{RAIL_COUNT, RAIL_FEE, RAIL_VOLUME};

    fn codec_with(row0: &[(usize, i16)]) -> CodecW {
        let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
        for &(col, w) in row0 {
            rows[0][col] = w;
        }
        CodecW::new(WeightVersion(1), rows)
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn round_half_even_ties_to_even() {
        assert_eq!(round_half_even_shift(5, 1), 2); // 2.5 → 2
        assert_eq!(round_half_even_shift(7, 1), 4); // 3.5 → 4
        assert_eq!(round_half_even_shift(2, 1), 1);
        assert_eq!(round_half_even_shift(1, 1), 0); // 0.5 → 0
        assert_eq!(round_half_even_shift(3, 1), 2); // 1.5 → 2
        assert_eq!(round_half_even_shift(-5, 1), -2); // −2.5 → −2
        assert_eq!(round_half_even_shift(-7, 1), -4); // −3.5 → −4
        assert_eq!(round_half_even_shift(-1, 1), 0);
        assert_eq!(round_half_even_shift(-3, 1), -2);
        assert_eq!(round_half_even_shift(1, 0), 1);
        assert_eq!(round_half_even_shift(0, 15), 0);
        assert_eq!(round_half_even_shift(32_768, 15), 1); // exact
        assert_eq!(round_half_even_shift(49_152, 15), 2); // 1.5 → 2
        assert_eq!(round_half_even_shift(81_920, 15), 2); // 2.5 → 2
        assert_eq!(round_half_even_shift(114_688, 15), 4); // 3.5 → 4
        assert_eq!(round_half_even_shift(-49_152, 15), -2);
    }

    #[test]
    fn apply_half_unit_weight_rounds_half_even() {
        // w = 0.5 on the count rail: δ = round½even(x / 2).
        let w = codec_with(&[(RAIL_COUNT, 16_384)]);
        let cases = [
            (0i64, 0u64),
            (1, 0),
            (2, 1),
            (3, 2),
            (5, 2),
            (-1, 0),
            (-2, (-1i64) as u64),
            (-3, (-2i64) as u64),
        ];
        for (x, expect) in cases {
            let mut fv = FeatureVector::new();
            fv.set_value(RAIL_COUNT, x);
            assert_eq!(w.apply(&fv).0[0], expect, "x = {x}");
        }
    }

    #[test]
    fn apply_scales_once_at_the_named_point() {
        // Two half-unit contributions sum to exactly one unit before the
        // single rounding point: δ = 1, whereas per-term rounding would
        // give 0. Pins the frozen rounding semantics.
        let w = codec_with(&[(RAIL_VOLUME, 16_384), (RAIL_FEE, 16_384)]);
        let mut fv = FeatureVector::new();
        fv.set_value(RAIL_VOLUME, 1);
        fv.set_value(RAIL_FEE, 1);
        assert_eq!(w.apply(&fv).0[0], 1);
    }

    #[test]
    fn apply_wraps_into_zmod2p64() {
        // w = −2^15, feature = −2^63: product = +2^78, δ = 2^63.
        let w = codec_with(&[(0, i16::MIN)]);
        let mut fv = FeatureVector::new();
        fv.set_value(0, i64::MIN);
        assert_eq!(w.apply(&fv).0[0], 1u64 << 63);
        // Two such features: 2^79 >> 15 = 2^64 wraps to 0.
        let w2 = codec_with(&[(0, i16::MIN), (1, i16::MIN)]);
        let mut fv2 = FeatureVector::new();
        fv2.set_value(0, i64::MIN);
        fv2.set_value(1, i64::MIN);
        let d = w2.apply(&fv2);
        assert_eq!(d.0[0], 0);
        assert!(d.is_zero());
    }

    #[test]
    fn apply_statement8_nonzero_delta() {
        // w = 1 − 2^-15 on the count rail (always 1): δ = 1 ≠ 0.
        let w = codec_with(&[(RAIL_COUNT, i16::MAX)]);
        let mut fv = FeatureVector::new();
        fv.set_value(RAIL_COUNT, 1);
        let d = w.apply(&fv);
        assert!(!d.is_zero());
        assert_eq!(d.0[0], 1);
    }

    #[test]
    fn zero_weights_give_zero_delta() {
        let w = CodecW::new(WeightVersion(0), [[0i16; FEATURE_COUNT]; EMBEDDING_DIM]);
        let mut fv = FeatureVector::new();
        fv.set_value(RAIL_VOLUME, 1_000_000);
        fv.set_value(RAIL_COUNT, 1);
        assert!(w.apply(&fv).is_zero());
    }

    #[test]
    fn canonical_roundtrip_and_parse_length() {
        let w = codec_with(&[(RAIL_VOLUME, 12_345), (RAIL_COUNT, -12_345)]);
        let bytes = w.canonical_bytes();
        assert_eq!(bytes.len(), CANONICAL_SIZE);
        assert_eq!(CodecW::parse(&bytes).unwrap(), w);
        assert!(matches!(
            CodecW::parse(&bytes[..CANONICAL_SIZE - 1]),
            Err(WError::BadLength { .. })
        ));
        assert_eq!(bytes[0..8], 1u64.to_le_bytes());
    }

    #[test]
    fn commitment_is_version_sensitive_and_pinned() {
        let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
        rows[0][RAIL_COUNT] = i16::MAX;
        rows[1][RAIL_VOLUME] = -16_384;
        let a = CodecW::new(WeightVersion(7), rows);
        let b = CodecW::new(WeightVersion(8), rows);
        assert_eq!(a.commitment(), a.commitment());
        assert_ne!(a.commitment(), b.commitment());
        assert_ne!(a, b); // version participates in equality
        // Differential pin: nerv-core's wrapper must equal
        // BLAKE3("nerv.w.commit" ‖ canonical_bytes). A failure here means
        // the wrapper convention differs — reconcile the chunk's
        // integration contract for nerv_core::hash.
        let mut msg = Vec::new();
        msg.extend_from_slice(W_COMMIT.as_bytes());
        msg.extend_from_slice(&a.canonical_bytes());
        assert_eq!(a.commitment().0, *blake3::hash(&msg).as_bytes());
    }

    #[test]
    fn delta_serialization_roundtrip() {
        let mut d = Delta::default();
        d.0[0] = u64::MAX;
        d.0[63] = 1;
        let bytes = d.canonical_bytes();
        assert_eq!(bytes.len(), 512);
        assert_eq!(Delta::from_canonical_bytes(&bytes), d);
    }

    fn delta_from_seed(mut s: u64) -> Delta {
        let mut out = [0u64; EMBEDDING_DIM];
        for v in out.iter_mut() {
            *v = splitmix64(&mut s);
        }
        Delta(out)
    }

    fn codec_from_seed(seed: u64) -> CodecW {
        let mut s = seed;
        let mut rows = [[0i16; FEATURE_COUNT]; EMBEDDING_DIM];
        for row in rows.iter_mut() {
            for w in row.iter_mut() {
                *w = splitmix64(&mut s) as i16;
            }
        }
        CodecW::new(WeightVersion(0), rows)
    }

    proptest! {
        #[test]
        fn prop_delta_group_laws(
            s1 in any::<u64>(),
            s2 in any::<u64>(),
            s3 in any::<u64>(),
        ) {
            let (a, b, c) = (delta_from_seed(s1), delta_from_seed(s2), delta_from_seed(s3));
            assert_eq!(a.wrapping_add(&b).wrapping_add(&c), a.wrapping_add(&b.wrapping_add(&c)));
            assert_eq!(a.wrapping_add(&b), b.wrapping_add(&a));
            assert_eq!(a.wrapping_add(&Delta::default()), a);
        }

        #[test]
        fn prop_delta_bytes_roundtrip(seed in any::<u64>()) {
            let d = delta_from_seed(seed);
            assert_eq!(Delta::from_canonical_bytes(&d.canonical_bytes()), d);
        }

        #[test]
        fn prop_apply_rounding_error_bounded(
            seed in any::<u64>(),
            f1 in prop::collection::vec(
                (0usize..FEATURE_COUNT, -(2i64.pow(40))..=2i64.pow(40)), 0..=6),
            f2 in prop::collection::vec(
                (0usize..FEATURE_COUNT, -(2i64.pow(40))..=2i64.pow(40)), 0..=6),
        ) {
            let w = codec_from_seed(seed);
            let v1 = sparse(&f1);
            let v2 = sparse(&f2);
            let mut v12 = v1.clone();
            for &(i, val) in &f2 {
                v12.set_value(i, v12.value(i) + val);
            }
            let d1 = w.apply(&v1);
            let d2 = w.apply(&v2);
            let d12 = w.apply(&v12);
            let sum = d1.wrapping_add(&d2);
            // Small values ⇒ no wrap: u64 ↔ i64 reinterpretation is exact,
            // and the two roundings differ from the single rounding by at
            // most 1 per coordinate (each rounding error ≤ 1/2).
            for j in 0..EMBEDDING_DIM {
                let diff = (d12.0[j] as i64 as i128 - sum.0[j] as i64 as i128).abs();
                assert!(diff <= 1, "coordinate {j}: |{diff}| > 1");
            }
            assert!(w.apply(&FeatureVector::new()).is_zero());
        }

        #[test]
        fn prop_codec_bytes_roundtrip(seed in any::<u64>()) {
            let w = codec_from_seed(seed);
            assert_eq!(CodecW::parse(&w.canonical_bytes()).unwrap(), w);
        }
    }

    fn sparse(pairs: &[(usize, i64)]) -> FeatureVector {
        let mut fv = FeatureVector::new();
        for &(i, v) in pairs {
            fv.set_value(i, v);
        }
        fv
    }
}

