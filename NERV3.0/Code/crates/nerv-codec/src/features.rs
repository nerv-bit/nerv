//! Per-leg feature vectors `ΔS_leg` (WP §7.2): construction and admissibility.
//!
//! The feature vector is the encoder's input: a sparse, bounded, signed
//! 256-dimensional summary of one leg's settled activity, constructed
//! client-side from the leg's own data. The admissible support (≤ 6 active
//! account slots; well-formed rails) is what the whole-transaction proof
//! enforces as statement 6 (WP §5.1). This module is the native reference
//! implementation (DSR-7): the encoder AIR in nerv-proofs must agree with
//! these functions bit-for-bit.
//!
//! Layout (frozen; WP §7.2 feature table):
//!   0..=223   account slots — signed per-address value deltas
//!   224       volume rail — total output value of the leg (nano-NERV)
//!   225       fee rail — the leg's declared fee share (nano-NERV)
//!   226       count rail — 1 (marks one leg event)
//!   227       log-magnitude rail — Σ ⌊log₂(1+v)⌋ over the leg's values
//!   228..=235 type rails — one-hot over the frozen leg kinds (233..=235 reserved)
//!   236..=239 reserved — required zero
//!   240..=255 time rails — one-hot coarse time-of-epoch bucket

use std::fmt;
use nerv_core::constants::ACCOUNT_SLOT;
use nerv_core::hash::Hash256;

// ---------------------------------------------------------------------------
// Frozen layout constants (WP §7.2; App. B "Overlay")
// ---------------------------------------------------------------------------

pub const FEATURE_COUNT: usize = 256;
/// Account-slot rails, indices `0..=223` (WP §7.2: 224 slots).
pub const ACCOUNT_SLOT_COUNT: usize = 224;
/// Volume rail: total output value of the leg (nano-NERV).
pub const RAIL_VOLUME: usize = 224;
/// Fee rail: the leg's declared fee share (nano-NERV).
pub const RAIL_FEE: usize = 225;
/// Count rail: `1` for every leg event.
pub const RAIL_COUNT: usize = 226;
/// Log-magnitude rail: `Σ ⌊log₂(1+v)⌋` over the leg's input and output values.
pub const RAIL_LOG_MAGNITUDE: usize = 227;
/// Type rails `228..=235`: one-hot over the frozen leg kinds; `233..=235`
/// reserved and required zero until governance activates them.
pub const RAIL_TYPE_BASE: usize = 228;
pub const RAIL_TYPE_COUNT: usize = 8;
/// Kinds constructible today (`SingleShard..=Burn`); rails beyond these must
/// be zero.
pub const RAIL_VALID_TYPE_COUNT: usize = 5;
/// Reserved rails `236..=239`: required zero.
pub const RAIL_RESERVED_BASE: usize = 236;
pub const RAIL_RESERVED_COUNT: usize = 4;
/// Time rails `240..=255`: one-hot coarse time-of-epoch bucket.
pub const RAIL_TIME_BASE: usize = 240;
pub const RAIL_TIME_COUNT: usize = 16;

/// WP §7.4: at most 6 active account slots per admissible leg.
pub const MAX_ACTIVE_ACCOUNT_SLOTS: usize = 6;
/// Active rails at most: volume, fee, count, log, one type, one time.
pub const MAX_ACTIVE_RAILS: usize = 6;
/// Circuit-facing sparsity bound. WP §7.2 quotes "at most 11 of W's 256
/// columns" for the typical leg; the frozen layout admits 6 + 6 = 12 — the
/// encoder AIR provisions 12 column selections (spec erratum, recorded here
/// so document and code never diverge silently).
pub const MAX_ACTIVE_FEATURES: usize = MAX_ACTIVE_ACCOUNT_SLOTS + MAX_ACTIVE_RAILS;

/// WP §3.2: `0 < v ≤ 2^60` nano-NERV per note value (mirrors
/// `params.toml [custody] value_max_nano`).
pub const VALUE_MAX_NANO: u64 = 1 << 60;
/// Uniform per-coordinate magnitude bound — the AIR's range check.
pub const FEATURE_ABS_MAX: i64 = (1 << 62) - 1;
/// Log-magnitude rail bound (frozen; ≤ 16 max-magnitude values plus margin).
pub const LOG_RAIL_MAX: i64 = 1024;

// ---------------------------------------------------------------------------
// Leg classification (type rails)
// ---------------------------------------------------------------------------

/// Leg classification for the one-hot type rails — the frozen instantiation
/// of WP §7.2's "transaction-type flags", which the WP leaves as an
/// unspecified flag set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum LegKind {
    /// Consumes inputs and creates outputs; the whole transaction touches
    /// one shard.
    SingleShard = 0,
    /// Cross-shard leg consuming inputs in this shard (settlement
    /// unconditional, WP §4.5).
    CrossShardSpend = 1,
    /// Cross-shard leg creating outputs only (settlement gated on sibling
    /// spend legs, WP §4.5).
    CrossShardIssue = 2,
    /// Emission claim leg (WP §12.3).
    Claim = 3,
    /// Shielded→transparent burn (WP §3.8).
    Burn = 4,
}

impl LegKind {
    /// Rail codes `5..=7` are reserved (zero) until governance activates
    /// them; they are not constructible.
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::SingleShard),
            1 => Some(Self::CrossShardSpend),
            2 => Some(Self::CrossShardIssue),
            3 => Some(Self::Claim),
            4 => Some(Self::Burn),
            _ => None,
        }
    }

    pub fn code(self) -> u8 {
        self as u8
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FeatureError {
    #[error("note value {value} nano-NERV outside (0, 2^60] (WP §3.2)")]
    ValueOutOfRange { value: u64 },
    #[error("account slot {slot} magnitude {value} exceeds ±{bound}")]
    SlotMagnitude { slot: usize, value: i64, bound: i64 },
    #[error("volume rail {value} exceeds {bound}")]
    VolumeOverflow { value: u64, bound: u64 },
    #[error("fee {value} exceeds {bound}")]
    FeeOverflow { value: u64, bound: u64 },
    #[error("log-magnitude rail {value} exceeds {max}")]
    LogRailOverflow { value: u64, max: i64 },
    #[error("epoch length must be nonzero")]
    ZeroEpochLength,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AdmissibilityViolation {
    #[error("{count} active account slots exceeds the admissible maximum {max} (WP §7.4)")]
    TooManyActiveSlots { count: usize, max: usize },
    #[error("account slot {index} magnitude {value} outside ±{bound}")]
    SlotRange { index: usize, value: i64, bound: i64 },
    #[error("{count} active features exceeds the encoder sparsity bound {max}")]
    TooManyActiveFeatures { count: usize, max: usize },
    #[error("volume rail {value} outside [0, {bound}]")]
    VolumeRange { value: i64, bound: i64 },
    #[error("fee rail {value} outside [0, {bound}]")]
    FeeRange { value: i64, bound: i64 },
    #[error("count rail must be exactly 1, found {value}")]
    CountRail { value: i64 },
    #[error("log-magnitude rail {value} outside [0, {max}]")]
    LogRailRange { value: i64, max: i64 },
    #[error("type rails not one-hot over the {valid} frozen kinds (pattern {pattern:#04x})")]
    TypeRailsNotOneHot { pattern: u8, valid: usize },
    #[error("time rails not one-hot (pattern {pattern:#06x})")]
    TimeRailsNotOneHot { pattern: u16 },
    #[error("reserved rail {index} must be zero, found {value}")]
    ReservedRail { index: usize, value: i64 },
}

// ---------------------------------------------------------------------------
// The feature vector
// ---------------------------------------------------------------------------

/// A per-leg feature vector `ΔS_leg` (WP §7.2): 256 signed coordinates.
/// Slots carry per-address value deltas (negative for inputs, positive for
/// outputs); rails carry the leg's public scalars. Construction is pure and
/// checked; admissibility (WP §5.1 statement 6) is validated separately so
/// vectors arriving from parsing or from circuit witnesses can be
/// re-checked against the same constraints.
#[derive(Clone, PartialEq, Eq)]
pub struct FeatureVector {
    values: [i64; FEATURE_COUNT],
}

impl Default for FeatureVector {
    fn default() -> Self {
        Self { values: [0; FEATURE_COUNT] }
    }
}

impl nerv_core::codec::Encode for FeatureVector {
    fn encode_into(&self, out: &mut Vec<u8>) {
        for v in self.values() {
            // i64 → two's-complement little-endian 8 bytes. Matches the
            // wire contract for signed-integer fields in
            // `nerv_core::codec` (LE fixed-width).
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        FEATURE_COUNT * 8
    }
}

impl nerv_core::codec::Decode for FeatureVector {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let mut arr = [0i64; FEATURE_COUNT];
        for slot in arr.iter_mut() {
            let bytes: [u8; 8] = r.take_array::<8>()?;
            *slot = i64::from_le_bytes(bytes);
        }
        Ok(FeatureVector { values: arr })
    }
}

impl fmt::Debug for FeatureVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sparse form: a full 256-coordinate dump is noise in diagnostics.
        let mut map = f.debug_map();
        for (i, v) in self.active_features() {
            map.entry(&i, &v);
        }
        map.finish()
    }
}

impl FeatureVector {
    pub fn new() -> Self {
        Self::default()
    }

    /// The coordinate at `index` (a frozen layout constant).
    /// Panics if `index >= 256`, which cannot arise from the layout constants.
    pub fn value(&self, index: usize) -> i64 {
        self.values[index]
    }

    /// Sets the coordinate at `index`. Callers assembling vectors by hand
    /// must run [`FeatureVector::check_admissible`] afterwards; the circuit
    /// enforces the same constraints (WP §5.1 statement 6).
    pub fn set_value(&mut self, index: usize, value: i64) {
        self.values[index] = value;
    }

    /// All 256 coordinates in layout order (circuit witness input).
    pub fn values(&self) -> &[i64; FEATURE_COUNT] {
        &self.values
    }

    /// The `(index, value)` pairs of non-zero coordinates — the sparse set
    /// the encoder scales (`δ = W · ΔS` touches only these, WP §7.2).
    pub fn active_features(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, &v)| (i, v))
    }

    pub fn active_feature_count(&self) -> usize {
        self.active_features().count()
    }

    pub fn active_account_slot_count(&self) -> usize {
        self.values[..ACCOUNT_SLOT_COUNT]
            .iter()
            .filter(|&&v| v != 0)
            .count()
    }

    /// Canonical serialization: 256 little-endian 64-bit coordinates —
    /// 2048 bytes, fixed width, no length prefix.
    pub fn canonical_bytes(&self) -> [u8; FEATURE_COUNT * 8] {
        let mut out = [0u8; FEATURE_COUNT * 8];
        for (i, &v) in self.values.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Infallible inverse of [`FeatureVector::canonical_bytes`]: every bit
    /// pattern is a valid coordinate pattern; admissibility is a separate
    /// check.
    pub fn from_canonical_bytes(bytes: &[u8; FEATURE_COUNT * 8]) -> Self {
        let mut values = [0i64; FEATURE_COUNT];
        for (i, v) in values.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
            *v = i64::from_le_bytes(b);
        }
        Self { values }
    }

    /// The admissible support (WP §7.2/§7.4; WP §5.1 statement 6): ≤ 6
    /// active account slots, bounded and well-formed rails, one-hot type
    /// and time blocks, zero reserved rails, ≤ 12 active features.
    pub fn check_admissible(&self) -> Result<(), AdmissibilityViolation> {
        let mut active_slots = 0usize;
        for (i, &v) in self.values[..ACCOUNT_SLOT_COUNT].iter().enumerate() {
            if v != 0 {
                active_slots += 1;
            }
            if !(-FEATURE_ABS_MAX..=FEATURE_ABS_MAX).contains(&v) {
                return Err(AdmissibilityViolation::SlotRange {
                    index: i,
                    value: v,
                    bound: FEATURE_ABS_MAX,
                });
            }
        }
        if active_slots > MAX_ACTIVE_ACCOUNT_SLOTS {
            return Err(AdmissibilityViolation::TooManyActiveSlots {
                count: active_slots,
                max: MAX_ACTIVE_ACCOUNT_SLOTS,
            });
        }

        let volume = self.values[RAIL_VOLUME];
        if !(0..=FEATURE_ABS_MAX).contains(&volume) {
            return Err(AdmissibilityViolation::VolumeRange { value: volume, bound: FEATURE_ABS_MAX });
        }
        let fee = self.values[RAIL_FEE];
        if !(0..=FEATURE_ABS_MAX).contains(&fee) {
            return Err(AdmissibilityViolation::FeeRange { value: fee, bound: FEATURE_ABS_MAX });
        }
        if self.values[RAIL_COUNT] != 1 {
            return Err(AdmissibilityViolation::CountRail { value: self.values[RAIL_COUNT] });
        }
        let log = self.values[RAIL_LOG_MAGNITUDE];
        if !(0..=LOG_RAIL_MAX).contains(&log) {
            return Err(AdmissibilityViolation::LogRailRange { value: log, max: LOG_RAIL_MAX });
        }

        let mut pattern: u8 = 0;
        for (k, &v) in self.values[RAIL_TYPE_BASE..RAIL_TYPE_BASE + RAIL_TYPE_COUNT]
            .iter()
            .enumerate()
        {
            if v != 0 && v != 1 {
                return Err(AdmissibilityViolation::TypeRailsNotOneHot {
                    pattern,
                    valid: RAIL_VALID_TYPE_COUNT,
                });
            }
            if v == 1 {
                pattern |= 1 << k;
            }
        }
        if pattern.count_ones() != 1 || pattern >= (1u8 << RAIL_VALID_TYPE_COUNT) {
            return Err(AdmissibilityViolation::TypeRailsNotOneHot {
                pattern,
                valid: RAIL_VALID_TYPE_COUNT,
            });
        }

        for (k, &v) in self.values[RAIL_RESERVED_BASE..RAIL_RESERVED_BASE + RAIL_RESERVED_COUNT]
            .iter()
            .enumerate()
        {
            if v != 0 {
                return Err(AdmissibilityViolation::ReservedRail {
                    index: RAIL_RESERVED_BASE + k,
                    value: v,
                });
            }
        }

        let mut time_pattern: u16 = 0;
        for (k, &v) in self.values[RAIL_TIME_BASE..RAIL_TIME_BASE + RAIL_TIME_COUNT]
            .iter()
            .enumerate()
        {
            if v != 0 && v != 1 {
                return Err(AdmissibilityViolation::TimeRailsNotOneHot { pattern: time_pattern });
            }
            if v == 1 {
                time_pattern |= 1 << k;
            }
        }
        if time_pattern.count_ones() != 1 {
            return Err(AdmissibilityViolation::TimeRailsNotOneHot { pattern: time_pattern });
        }

        let active = self.active_feature_count();
        if active > MAX_ACTIVE_FEATURES {
            return Err(AdmissibilityViolation::TooManyActiveFeatures {
                count: active,
                max: MAX_ACTIVE_FEATURES,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure reference functions
// ---------------------------------------------------------------------------


/// `slot(addr) = BLAKE3("nerv.slot" ‖ addr) mod 224` (WP §7.2): the full
/// 256-bit little-endian digest value reduced mod 224 via
/// `Hash256::reduce_mod` — nerv-core's ToField convention for "hash mod m".
/// The AIR performs the identical computation. Address bytes are the
/// caller's canonical encoding.
pub fn slot_index(addr: &[u8]) -> u16 {
    Hash256::concat(&ACCOUNT_SLOT, addr).reduce_mod(ACCOUNT_SLOT_COUNT as u64) as u16
}


/// `⌊log₂(1 + v)⌋` for `v ≤ 2^60` (WP §7.2 log-magnitude rail).
/// Integer-exact. `v = 0` gives 0. Panics only if `v = u64::MAX`, which
/// violates the value contract enforced by every caller.
pub fn floor_log2_1p(v: u64) -> u32 {
    debug_assert!(v <= VALUE_MAX_NANO);
    let t = v + 1;
    63 - t.leading_zeros()
}

/// Coarse time-of-epoch bucket: `⌊16 · (h mod E) / E⌋` where `h` is the
/// leg's expiry height (a public shell field) and `E` the epoch length in
/// blocks [frozen semantics: the WP's "coarse time-of-epoch buckets
/// (public, not private timing)" instantiated on the only per-leg public
/// time coordinate. Statement 6 checks volume/fee/count/log correctness;
/// the type and time blocks are constrained to well-formed one-hots, and
/// the expiry-derived bucket is the wallet-side convention.]
pub fn time_bucket(expiry_height: u64, epoch_length_blocks: u64) -> Result<usize, FeatureError> {
    if epoch_length_blocks == 0 {
        return Err(FeatureError::ZeroEpochLength);
    }
    let phase = expiry_height % epoch_length_blocks;
    let b = (u128::from(phase) * RAIL_TIME_COUNT as u128) / u128::from(epoch_length_blocks);
    Ok((b as usize).min(RAIL_TIME_COUNT - 1))
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// The leg's raw movement data supplied by the wallet (WP §7.2: features
/// are constructed client-side, per leg, from that leg's own data).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegMovement {
    /// `(spending address bytes, value)` — one entry per input note consumed
    /// by this leg; the address is homed in this shard.
    pub inputs: Vec<(Vec<u8>, u64)>,
    /// `(recipient delivery address bytes, value)` — one entry per output
    /// note created by this leg; the address is homed in this shard.
    pub outputs: Vec<(Vec<u8>, u64)>,
    /// The leg's declared fee share (nano-NERV).
    pub fee_nano: u64,
    /// The leg's classification (one-hot type rails).
    pub kind: LegKind,
    /// The leg's declared expiry height (public shell field; feeds the time
    /// bucket).
    pub expiry_height: u64,
    /// Epoch length in blocks (governance parameter).
    pub epoch_length_blocks: u64,
}

/// Constructs `ΔS_leg` from the leg's movement data (WP §7.2). Hard input
/// errors (value range, rail bounds) are returned here; admissibility
/// (≤ 6 active slots) is a separate check — a wallet whose construction is
/// inadmissible must re-shape the transaction (e.g., split the leg).
pub fn build_leg_features(m: &LegMovement) -> Result<FeatureVector, FeatureError> {
    let mut fv = FeatureVector::new();

    // Account slots: −v per input, +v per output, at the address's slot.
    // Slot collisions are harmless (WP §7.2): they sum. The running slot
    // magnitude is bounded by ±FEATURE_ABS_MAX, so plain i64 arithmetic
    // cannot overflow (|slot| ≤ 2^62−1, v ≤ 2^60 ⇒ |next| < 2^63).
    for (addr, v) in &m.inputs {
        validate_value(*v)?;
        let s = slot_index(addr) as usize;
        let next = fv.values[s] - *v as i64;
        if next < -FEATURE_ABS_MAX {
            return Err(FeatureError::SlotMagnitude { slot: s, value: next, bound: FEATURE_ABS_MAX });
        }
        fv.values[s] = next;
    }
    for (addr, v) in &m.outputs {
        validate_value(*v)?;
        let s = slot_index(addr) as usize;
        let next = fv.values[s] + *v as i64;
        if next > FEATURE_ABS_MAX {
            return Err(FeatureError::SlotMagnitude { slot: s, value: next, bound: FEATURE_ABS_MAX });
        }
        fv.values[s] = next;
    }

    // Volume rail: total output value of this leg.
    let cap = FEATURE_ABS_MAX as u64;
    let mut volume: u64 = 0;
    for (_, v) in &m.outputs {
        if volume > cap - v {
            return Err(FeatureError::VolumeOverflow { value: volume.saturating_add(*v), bound: cap });
        }
        volume += v;
    }
    fv.values[RAIL_VOLUME] = volume as i64;

    // Fee rail.
    if m.fee_nano > cap {
        return Err(FeatureError::FeeOverflow { value: m.fee_nano, bound: cap });
    }
    fv.values[RAIL_FEE] = m.fee_nano as i64;

    // Count rail: one leg event.
    fv.values[RAIL_COUNT] = 1;

    // Log-magnitude rail: Σ ⌊log₂(1+v)⌋ over the leg's input and output
    // values [frozen aggregation of "⌊log₂(1 + v)⌋ per value"].
    let mut log: u64 = 0;
    let max_log = LOG_RAIL_MAX as u64;
    for (_, v) in m.inputs.iter().chain(m.outputs.iter()) {
        log += u64::from(floor_log2_1p(*v));
        if log > max_log {
            return Err(FeatureError::LogRailOverflow { value: log, max: LOG_RAIL_MAX });
        }
    }
    fv.values[RAIL_LOG_MAGNITUDE] = log as i64;

    // Type rail: one-hot over the frozen kinds.
    fv.values[RAIL_TYPE_BASE + m.kind.code() as usize] = 1;

    // Time rail: one-hot coarse bucket.
    let bucket = time_bucket(m.expiry_height, m.epoch_length_blocks)?;
    fv.values[RAIL_TIME_BASE + bucket] = 1;

    Ok(fv)
}

fn validate_value(v: u64) -> Result<(), FeatureError> {
    if v == 0 || v > VALUE_MAX_NANO {
        return Err(FeatureError::ValueOutOfRange { value: v });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn floor_log2_1p_reference_table() {
        assert_eq!(floor_log2_1p(0), 0);
        assert_eq!(floor_log2_1p(1), 1); // log₂ 2
        assert_eq!(floor_log2_1p(2), 1); // log₂ 3 = 1.58…
        assert_eq!(floor_log2_1p(3), 2); // log₂ 4
        assert_eq!(floor_log2_1p(7), 3); // log₂ 8
        assert_eq!(floor_log2_1p(8), 3); // log₂ 9 = 3.17…
        assert_eq!(floor_log2_1p(1 << 60), 60);
        assert_eq!(floor_log2_1p((1 << 60) - 1), 60);
    }

   #[test]
    fn slot_index_matches_blake3_reference() {
        // Differential pin (DSR-7): equals BLAKE3("nerv.slot" ‖ addr) with
        // the FULL 256-bit LE digest value reduced mod 224 — the reduce_mod
        // limb loop reproduced independently.
        let cases: &[&[u8]] = &[&[], b"a", &[0u8; 32], &[0xff; 17]];
        for addr in cases {
            let mut msg = Vec::new();
            msg.extend_from_slice(ACCOUNT_SLOT.as_bytes());
            msg.extend_from_slice(addr);
            let d = blake3::hash(&msg);
            let mut r: u128 = 0;
            for limb in d.as_bytes().chunks_exact(8) {
                let mut b = [0u8; 8];
                b.copy_from_slice(limb);
                r = ((r << 64) + u64::from_le_bytes(b) as u128) % ACCOUNT_SLOT_COUNT as u128;
            }
            assert_eq!(u64::from(slot_index(addr)), r as u64, "addr len {}", addr.len());
        }
    }


    #[test]
    fn slot_index_deterministic_and_in_range() {
        for k in 0..512u64 {
            let addr = k.to_le_bytes();
            let s = slot_index(&addr);
            assert!((s as usize) < ACCOUNT_SLOT_COUNT);
            assert_eq!(s, slot_index(&addr));
        }
    }

    #[test]
    fn time_bucket_partitions_epoch() {
        let e = 43_200u64;
        assert_eq!(time_bucket(0, e).unwrap(), 0);
        assert_eq!(time_bucket(6, e).unwrap(), 0);
        assert_eq!(time_bucket(2_700, e).unwrap(), 1);
        assert_eq!(time_bucket(43_199, e).unwrap(), 15);
        assert_eq!(time_bucket(43_200, e).unwrap(), 0);
        assert_eq!(time_bucket(86_399, e).unwrap(), 15);
        assert!(matches!(time_bucket(0, 0), Err(FeatureError::ZeroEpochLength)));
    }

    #[test]
    fn build_rejects_value_out_of_range() {
        let base = LegMovement {
            inputs: vec![],
            outputs: vec![],
            fee_nano: 0,
            kind: LegKind::SingleShard,
            expiry_height: 1,
            epoch_length_blocks: 100,
        };
        let mut m = base.clone();
        m.inputs = vec![(vec![1u8; 32], 0)];
        assert!(matches!(build_leg_features(&m), Err(FeatureError::ValueOutOfRange { value: 0 })));
        m.inputs = vec![(vec![1u8; 32], VALUE_MAX_NANO)];
        assert!(build_leg_features(&m).is_ok());
        m.inputs = vec![(vec![1u8; 32], VALUE_MAX_NANO + 1)];
        assert!(matches!(build_leg_features(&m), Err(FeatureError::ValueOutOfRange { .. })));
    }

    fn addr(k: u32) -> Vec<u8> {
        let mut a = vec![0xA5u8; 32];
        a[0..4].copy_from_slice(&k.to_le_bytes());
        a
    }

    /// Addresses with pairwise-distinct slots, generated adaptively so the
    /// test is deterministic regardless of BLAKE3's (fixed) outputs.
    fn distinct_slot_addrs(n: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(n);
        let mut slots = std::collections::HashSet::new();
        let mut k = 0u32;
        while out.len() < n {
            let a = addr(k);
            if slots.insert(slot_index(&a)) {
                out.push(a);
            }
            k += 1;
        }
        out
    }

    #[test]
    fn build_wires_every_rail() {
        let a = distinct_slot_addrs(3);
        let (alice, carol, change) = (a[0].clone(), a[1].clone(), a[2].clone());
        let m = LegMovement {
            inputs: vec![(alice.clone(), 35_000_000_000), (alice.clone(), 25_000_000_000)],
            outputs: vec![(carol.clone(), 25_000_000_000), (change.clone(), 9_999_000_000)],
            fee_nano: 600_000,
            kind: LegKind::SingleShard,
            expiry_height: 7_000,
            epoch_length_blocks: 43_200,
        };
        let fv = build_leg_features(&m).unwrap();
        assert_eq!(fv.value(slot_index(&alice) as usize), -60_000_000_000);
        assert_eq!(fv.value(slot_index(&carol) as usize), 25_000_000_000);
        assert_eq!(fv.value(slot_index(&change) as usize), 9_999_000_000);
        assert_eq!(fv.value(RAIL_VOLUME), 34_999_000_000);
        assert_eq!(fv.value(RAIL_FEE), 600_000);
        assert_eq!(fv.value(RAIL_COUNT), 1);
        assert_eq!(
            fv.value(RAIL_LOG_MAGNITUDE),
            (floor_log2_1p(35_000_000_000)
                + floor_log2_1p(25_000_000_000)
                + floor_log2_1p(25_000_000_000)
                + floor_log2_1p(9_999_000_000)) as i64
        );
        assert_eq!(fv.value(RAIL_TYPE_BASE), 1);
        for k in 1..RAIL_TYPE_COUNT {
            assert_eq!(fv.value(RAIL_TYPE_BASE + k), 0);
        }
        let bucket = time_bucket(7_000, 43_200).unwrap();
        assert_eq!(fv.value(RAIL_TIME_BASE + bucket), 1);
        assert_eq!(fv.active_account_slot_count(), 3);
        assert!(fv.check_admissible().is_ok());

        // Slot conservation: Σ slots = Σ outputs − Σ inputs.
        let net: i128 = (0..ACCOUNT_SLOT_COUNT).map(|i| i128::from(fv.value(i))).sum();
        assert_eq!(net, 34_999_000_000 - 60_000_000_000);
    }

    #[test]
    fn admissibility_rejects_excess_active_slots() {
        // 24 distinct-slot outputs: the active-slot count is ≥ 7 with
        // certainty (all 24 drawing into ≤ 6 of 224 slots is ~2^-88).
        let outputs: Vec<(Vec<u8>, u64)> =
            distinct_slot_addrs(24).into_iter().map(|a| (a, 1)).collect();
        let m = LegMovement {
            inputs: vec![],
            outputs,
            fee_nano: 1,
            kind: LegKind::CrossShardIssue,
            expiry_height: 5,
            epoch_length_blocks: 100,
        };
        let fv = build_leg_features(&m).unwrap();
        assert!(matches!(
            fv.check_admissible(),
            Err(AdmissibilityViolation::TooManyActiveSlots { count, .. }) if count > MAX_ACTIVE_ACCOUNT_SLOTS
        ));
    }

    #[test]
    fn admissibility_rejects_malformed_rails() {
        let mut fv = FeatureVector::new();
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::CountRail { .. })));
        fv.set_value(RAIL_COUNT, 1);
        fv.set_value(RAIL_VOLUME, -1);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::VolumeRange { .. })));
        fv.set_value(RAIL_VOLUME, 0);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::TypeRailsNotOneHot { .. })));
        fv.set_value(RAIL_TYPE_BASE, 1);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::TimeRailsNotOneHot { .. })));
        fv.set_value(RAIL_TIME_BASE + 3, 1);
        fv.set_value(RAIL_RESERVED_BASE, 5);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::ReservedRail { .. })));
        fv.set_value(RAIL_RESERVED_BASE, 0);
        fv.set_value(RAIL_LOG_MAGNITUDE, 1);
        assert!(fv.check_admissible().is_ok());
        fv.set_value(RAIL_LOG_MAGNITUDE, 1025);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::LogRailRange { .. })));
        fv.set_value(RAIL_LOG_MAGNITUDE, 1);

        // Two type rails set; a non-0/1 type value; a reserved kind bit.
        fv.set_value(RAIL_TYPE_BASE + 1, 1);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::TypeRailsNotOneHot { .. })));
        fv.set_value(RAIL_TYPE_BASE + 1, 0);
        fv.set_value(RAIL_TYPE_BASE, 2);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::TypeRailsNotOneHot { .. })));
        fv.set_value(RAIL_TYPE_BASE, 0);
        fv.set_value(RAIL_TYPE_BASE + 5, 1);
        assert!(matches!(fv.check_admissible(), Err(AdmissibilityViolation::TypeRailsNotOneHot { .. })));
        fv.set_value(RAIL_TYPE_BASE + 5, 0);
        fv.set_value(RAIL_TYPE_BASE, 1);
        assert!(fv.check_admissible().is_ok());
    }

    #[test]
    fn log_rail_bound_rejects_many_tiny_values() {
        let outputs: Vec<(Vec<u8>, u64)> = (0..2000).map(|k| (addr(k), 1)).collect();
        let m = LegMovement {
            inputs: vec![],
            outputs,
            fee_nano: 1,
            kind: LegKind::CrossShardIssue,
            expiry_height: 5,
            epoch_length_blocks: 100,
        };
        assert!(matches!(
            build_leg_features(&m),
            Err(FeatureError::LogRailOverflow { .. })
        ));
    }

    #[test]
    fn canonical_roundtrip_is_total() {
        let a = distinct_slot_addrs(3);
        let m = LegMovement {
            inputs: vec![(a[0].clone(), 5)],
            outputs: vec![(a[1].clone(), 4)],
            fee_nano: 1,
            kind: LegKind::SingleShard,
            expiry_height: 9,
            epoch_length_blocks: 100,
        };
        let fv = build_leg_features(&m).unwrap();
        let bytes = fv.canonical_bytes();
        assert_eq!(bytes.len(), 2048);
        assert_eq!(FeatureVector::from_canonical_bytes(&bytes), fv);
        // Total on arbitrary bit patterns.
        let raw = [0xA5u8; 2048];
        assert_eq!(FeatureVector::from_canonical_bytes(&raw).canonical_bytes(), raw);
        let _ = format!("{fv:?}"); // sparse Debug does not panic
    }

    fn any_addr() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(any::<u8>(), 16..=64)
    }

    proptest! {
        #[test]
        fn prop_small_legs_build_admissible(
            inputs in prop::collection::vec((any_addr(), 1u64..=VALUE_MAX_NANO / 8), 0..=3),
            outputs in prop::collection::vec((any_addr(), 1u64..=VALUE_MAX_NANO / 8), 0..=3),
            fee in 0u64..=1_000_000_000,
            kind in 0u8..=4,
            expiry in any::<u64>(),
            epoch in 1u64..=1_000_000,
        ) {
            let m = LegMovement {
                inputs,
                outputs,
                fee_nano: fee,
                kind: LegKind::from_code(kind).unwrap(),
                expiry_height: expiry,
                epoch_length_blocks: epoch,
            };
            let fv = build_leg_features(&m).unwrap();
            // ≤ 3 + 3 distinct slots ⇒ always on the admissible support.
            fv.check_admissible().unwrap();
            // Slot conservation invariant.
            let net: i128 = (0..ACCOUNT_SLOT_COUNT).map(|i| i128::from(fv.value(i))).sum();
            assert_eq!(
                net,
                i128::from(m.outputs.iter().map(|(_, v)| *v).sum::<u64>())
                    - i128::from(m.inputs.iter().map(|(_, v)| *v).sum::<u64>())
            );
        }

        #[test]
        fn prop_feature_bytes_roundtrip(bytes in prop::collection::vec(any::<u8>(), 2048..=2048)) {
            let mut b = [0u8; FEATURE_COUNT * 8];
            b.copy_from_slice(&bytes);
            let fv = FeatureVector::from_canonical_bytes(&b);
            assert_eq!(fv.canonical_bytes(), b);
        }
    }
}

