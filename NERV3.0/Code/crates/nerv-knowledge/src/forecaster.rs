//! The per-shard forecaster (WP §10.2; errata 166–167): the linear
//! AR(1024) over the last 1,024 (Δ_B, fee-sum, time-bucket)
//! observations. This module owns the state (weights, window, optimizer
//! moments) and the deterministic inference; the Adam transition is
//! adam.rs's, the scoring huber.rs's, the block loop the loop module's.


use std::collections::VecDeque;


use nerv_codec::codec_w::{Delta, EMBEDDING_DIM};
use nerv_core::constants::DERIVED_FS;
use nerv_core::fixed_point::Q15_FRAC_BITS;
use nerv_core::hash::Hash256;
use nerv_core::{round_half_even_pow2, wrap_mod_2_64, Q15};


/// The AR order and window capacity (WP §10.2's "1,024 triples").
pub const AR_ORDER: usize = 1024;
/// The embedding dimension (the codec's own).
pub const DIMS: usize = EMBEDDING_DIM;
/// Per-channel weight count.
pub const PER_CHANNEL: usize = DIMS * AR_ORDER;
/// The canonical Adam-trained count: ar ‖ fee ‖ bucket ‖ bias (the
/// scale updates by huber's robust EMA, not Adam — erratum 169).
pub const LEARNABLE: usize = 3 * PER_CHANNEL + DIMS;

/// The published default Huber scale (§10.3's per-dimension scales).
pub const SCALE_DEFAULT: u64 = 1 << 20;


pub const WEIGHTS_CANONICAL_LEN: usize = 3 * PER_CHANNEL * 2 + 2 * DIMS * 8;
pub const MOMENTS_CANONICAL_LEN: usize = 2 * LEARNABLE * 16 + 8;


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ForecasterError {
    #[error("serialized length {len}, expected {expected}")]
    BadLength { len: usize, expected: usize },
}


/// One observed block: the revealed aggregate, the fee total, and the
/// 16-bucket time-of-epoch index (erratum 166).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub delta: Delta,
    pub fee_sum: u64,
    pub bucket: u16,
}


/// The last 1,024 observations; `lag(1)` is the most recent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Window {
    obs: VecDeque<Observation>,
}


impl Window {
    pub fn new() -> Window {
        Window::default()
    }


    pub fn push(&mut self, obs: Observation) {
        debug_assert!(obs.bucket < 16, "bucket is the 16-bucket time-of-epoch index");
        if self.obs.len() == AR_ORDER {
            self.obs.pop_front();
        }
        self.obs.push_back(obs);
    }


    pub fn len(&self) -> usize {
        self.obs.len()
    }


    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }


    pub fn is_full(&self) -> bool {
        self.obs.len() == AR_ORDER
    }


    /// The observation `k` blocks back (lag 1 = the most recent).
    pub fn lag(&self, k: usize) -> Option<&Observation> {
        if k == 0 || k > self.obs.len() {
            None
        } else {
            self.obs.get(self.obs.len() - k)
        }
    }


    /// Oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = &Observation> {
        self.obs.iter()
    }


    pub fn clear(&mut self) {
        self.obs.clear();
    }
}


/// The parameterization: three Q15 weight channels (AR on the dimension's
/// own lagged centered deltas; exogenous fee and bucket channels), the
/// i64 bias, and the u64 Huber scales. Canonical parameter order:
/// ar ‖ fee ‖ bucket ‖ bias ‖ scale (erratum 166).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Weights {
    ar: Vec<Q15>,
    fee: Vec<Q15>,
    bucket: Vec<Q15>,
    bias: Vec<i64>,
    scale: Vec<u64>,
}


impl Weights {
    fn check(j: usize, k: usize) {
        debug_assert!(j < DIMS, "dimension index");
        debug_assert!(k >= 1 && k <= AR_ORDER, "lag index is 1-based");
    }


    pub fn zero() -> Weights {
        Weights {
            ar: vec![Q15::ZERO; PER_CHANNEL],
            fee: vec![Q15::ZERO; PER_CHANNEL],
            bucket: vec![Q15::ZERO; PER_CHANNEL],
            bias: vec![0; DIMS],
            scale: vec![SCALE_DEFAULT; DIMS],
        }
    }


    /// The reference parameterization (§7.6's revert target): the
    /// decaying AR core {1/2, 1/4, 1/8, 1/16} on lags 1–4, everything
    /// else zero, scales at the published default.
    pub fn reference() -> Weights {
        let mut w = Weights::zero();
        for j in 0..DIMS {
            for (k, bits) in [(1usize, 16384i16), (2, 8192), (3, 4096), (4, 2048)] {
                w.ar[j * AR_ORDER + (k - 1)] = Q15::from_bits(bits);
            }
        }
        w
    }


    pub fn ar(&self, j: usize, k: usize) -> Q15 {
        Self::check(j, k);
        self.ar[j * AR_ORDER + (k - 1)]
    }


    pub fn set_ar(&mut self, j: usize, k: usize, v: Q15) {
        Self::check(j, k);
        self.ar[j * AR_ORDER + (k - 1)] = v;
    }


    pub fn fee(&self, j: usize, k: usize) -> Q15 {
        Self::check(j, k);
        self.fee[j * AR_ORDER + (k - 1)]
    }


    pub fn set_fee(&mut self, j: usize, k: usize, v: Q15) {
        Self::check(j, k);
        self.fee[j * AR_ORDER + (k - 1)] = v;
    }


    pub fn bucket(&self, j: usize, k: usize) -> Q15 {
        Self::check(j, k);
        self.bucket[j * AR_ORDER + (k - 1)]
    }


    pub fn set_bucket(&mut self, j: usize, k: usize, v: Q15) {
        Self::check(j, k);
        self.bucket[j * AR_ORDER + (k - 1)] = v;
    }


    pub fn bias(&self, j: usize) -> i64 {
        debug_assert!(j < DIMS);
        self.bias[j]
    }


    pub fn set_bias(&mut self, j: usize, v: i64) {
        debug_assert!(j < DIMS);
        self.bias[j] = v;
    }


    pub fn scale(&self, j: usize) -> u64 {
        debug_assert!(j < DIMS);
        self.scale[j]
    }


    pub fn set_scale(&mut self, j: usize, v: u64) {
        debug_assert!(j < DIMS);
        self.scale[j] = v;
    }


    pub fn ar_slice_mut(&mut self) -> &mut [Q15] {
        &mut self.ar
    }


    pub fn fee_slice_mut(&mut self) -> &mut [Q15] {
        &mut self.fee
    }


    pub fn bucket_slice_mut(&mut self) -> &mut [Q15] {
        &mut self.bucket
    }


    pub fn bias_slice_mut(&mut self) -> &mut [i64] {
        &mut self.bias
    }


    pub fn scale_slice_mut(&mut self) -> &mut [u64] {
        &mut self.scale
    }


    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(WEIGHTS_CANONICAL_LEN);
        for ch in [&self.ar, &self.fee, &self.bucket] {
            for w in ch {
                out.extend_from_slice(&w.to_bits().to_le_bytes());
            }
        }
        for b in &self.bias {
            out.extend_from_slice(&b.to_le_bytes());
        }
        for s in &self.scale {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }


    pub fn from_canonical(bytes: &[u8]) -> Result<Weights, ForecasterError> {
        if bytes.len() != WEIGHTS_CANONICAL_LEN {
            return Err(ForecasterError::BadLength {
                len: bytes.len(),
                expected: WEIGHTS_CANONICAL_LEN,
            });
        }
        let mut at = 0usize;
        let mut channels = Vec::with_capacity(3);
        for _ in 0..3 {
            let mut v = Vec::with_capacity(PER_CHANNEL);
            for _ in 0..PER_CHANNEL {
                v.push(Q15::from_bits(i16::from_le_bytes([bytes[at], bytes[at + 1]])));
                at += 2;
            }
            channels.push(v);
        }
        let mut bias = Vec::with_capacity(DIMS);
        for _ in 0..DIMS {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[at..at + 8]);
            bias.push(i64::from_le_bytes(b));
            at += 8;
        }
        let mut scale = Vec::with_capacity(DIMS);
        for _ in 0..DIMS {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[at..at + 8]);
            scale.push(u64::from_le_bytes(b));
            at += 8;
        }
        Ok(Weights {
            ar: channels.remove(0),
            fee: channels.remove(0),
            bucket: channels.remove(0),
            bias,
            scale,
        })
    }
}


/// The optimizer moments (storage; the transition is adam.rs's): first
/// and second moments per canonical parameter, plus the step counter
/// (the β^t powers are replay-local, erratum 167).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Moments {
    pub m: Vec<i128>,
    pub v: Vec<u128>,
    pub step: u64,
}


impl Moments {
    pub fn zero() -> Moments {
        Moments { m: vec![0i128; LEARNABLE], v: vec![0u128; LEARNABLE], step: 0 }
    }


    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(MOMENTS_CANONICAL_LEN);
        for x in &self.m {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for x in &self.v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&self.step.to_le_bytes());
        out
    }


    pub fn from_canonical(bytes: &[u8]) -> Result<Moments, ForecasterError> {
        if bytes.len() != MOMENTS_CANONICAL_LEN {
            return Err(ForecasterError::BadLength {
                len: bytes.len(),
                expected: MOMENTS_CANONICAL_LEN,
            });
        }
        let mut at = 0usize;
        let mut m = Vec::with_capacity(LEARNABLE);
        for _ in 0..LEARNABLE {
            let mut b = [0u8; 16];
            b.copy_from_slice(&bytes[at..at + 16]);
            m.push(i128::from_le_bytes(b));
            at += 16;
        }
        let mut v = Vec::with_capacity(LEARNABLE);
        for _ in 0..LEARNABLE {
            let mut b = [0u8; 16];
            b.copy_from_slice(&bytes[at..at + 16]);
            v.push(u128::from_le_bytes(b));
            at += 16;
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[at..at + 8]);
        Ok(Moments { m, v, step: u64::from_le_bytes(b) })
    }
}


/// The full forecaster state. The window is re-derivable from the public
/// reveal stream and is excluded from the state root (erratum 167).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forecaster {
    weights: Weights,
    window: Window,
    moments: Moments,
}


impl Forecaster {
    pub fn new(weights: Weights) -> Forecaster {
        Forecaster { weights, window: Window::new(), moments: Moments::zero() }
    }


    /// The W-epoch-boundary state: the reference parameterization, an
    /// empty window, zeroed optimizer (§7.6).
    pub fn reference() -> Forecaster {
        Forecaster::new(Weights::reference())
    }


    pub fn weights(&self) -> &Weights {
        &self.weights
    }


    pub fn weights_mut(&mut self) -> &mut Weights {
        &mut self.weights
    }


    pub fn window(&self) -> &Window {
        &self.window
    }


    pub fn push_observation(&mut self, obs: Observation) {
        self.window.push(obs);
    }


    pub fn moments(&self) -> &Moments {
        &self.moments
    }


    pub fn moments_mut(&mut self) -> &mut Moments {
        &mut self.moments
    }


    /// Run one Adam step on the forecaster's `(weights, moments)` pair
    /// from `grads`. Convenience wrapper that splits the borrow across
    /// `&mut self.weights` and `&mut self.moments` internally, so
    /// callers don't have to thread two simultaneous `&mut` references
    /// through `self`. Errors propagate from [`adam::step`].
    pub fn adam_step(&mut self, grads: &[i128]) -> Result<(), crate::adam::AdamError> {
        crate::adam::step(&mut self.weights, &mut self.moments, grads)
    }


    /// The W-epoch revert (§7.6): weights ← reference, optimizer ← zero,
    /// window ← cleared (the old window's deltas were computed under the
    /// previous W; warm-start is post-boundary aggregates only).
    pub fn reset_to_reference(&mut self) {
        self.weights = Weights::reference();
        self.window = Window::new();
        self.moments = Moments::zero();
    }


     /// Δ̂_B — the shared prediction core (erratum 166), delegating.
    pub fn predict(&self) -> Delta {
        predict(&self.weights, &self.window)
    }



    /// forecaster_state_root (erratum 167): H("nerv.derived.fs" ‖ weights
    /// ‖ m ‖ v ‖ step). The window is excluded. The preimage is ≈ 6.7 MB
    /// — advisory-layer only; light clients never touch it (DSR-4).
    pub fn state_root(&self) -> Hash256 {
        let mut buf = self.weights.canonical_bytes();
        buf.extend_from_slice(&self.moments.canonical_bytes());
        Hash256::concat(&DERIVED_FS, &buf)
    }
}

/// The prediction core (erratum 166): weights ∘ window — the SINGLE
/// implementation for the incumbent and every challenger (erratum 172;
/// no drift). Per dimension, the Q15-weighted i128 sum over the window's
/// lagged triples, one round-half-even at 2⁻¹⁵, plus the bias, wrapped
/// mod 2⁶⁴. Missing lags contribute zero; the adversarial worst case is
/// < 2⁹¹ — total, no panics.
pub fn predict(weights: &Weights, window: &Window) -> Delta {
    let mut out = [0u64; DIMS];
    let n = window.obs.len();
    for j in 0..DIMS {
        let row = j * AR_ORDER;
        let mut acc: i128 = 0;
        for lag in 1..=AR_ORDER.min(n) {
            let obs = &window.obs[n - lag];
            let at = row + lag - 1;
            acc += i128::from(weights.ar[at].to_bits()) * i128::from(obs.delta.0[j] as i64);
            acc += i128::from(weights.fee[at].to_bits()) * i128::from(obs.fee_sum);
            acc += i128::from(weights.bucket[at].to_bits()) * i128::from(u64::from(obs.bucket));
        }
        let centered = i128::from(weights.bias[j]) + round_half_even_pow2(acc, Q15_FRAC_BITS);
        out[j] = wrap_mod_2_64(centered);
    }
    Delta(out)
}




#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use proptest::prelude::*;


    fn obs(delta: u64, fee: u64, bucket: u16) -> Observation {
        Observation { delta: Delta([delta; DIMS]), fee_sum: fee, bucket }
    }


    /// The independent differential: an oldest-first traversal that
    /// re-derives the lag mapping (position i ↔ lag n − i).
    fn predict_reference(f: &Forecaster) -> Delta {
        let w = f.weights();
        let n = f.window().len();
        let mut acc_ar = [0i128; DIMS];
        let mut acc_fee = [0i128; DIMS];
        let mut acc_b = [0i128; DIMS];
        for (i, o) in f.window().iter().enumerate() {
            let lag = n - i;
            for j in 0..DIMS {
                acc_ar[j] += i128::from(w.ar(j, lag).to_bits()) * i128::from(o.delta.0[j] as i64);
                acc_fee[j] += i128::from(w.fee(j, lag).to_bits()) * i128::from(o.fee_sum);
                acc_b[j] +=
                    i128::from(w.bucket(j, lag).to_bits()) * i128::from(u64::from(o.bucket));
            }
        }
        let mut out = [0u64; DIMS];
        for j in 0..DIMS {
            let total = acc_ar[j] + acc_fee[j] + acc_b[j];
            out[j] = wrap_mod_2_64(i128::from(w.bias(j)) + round_half_even_pow2(total, Q15_FRAC_BITS));
        }
        Delta(out)
    }


    #[test]
    fn layout_pins() {
        assert_eq!(AR_ORDER, 1024);
        assert_eq!(DIMS, 64);
        assert_eq!(PER_CHANNEL, 65_536);
        assert_eq!(LEARNABLE, 196_672);
        assert_eq!(WEIGHTS_CANONICAL_LEN, 394_240);
        assert_eq!(MOMENTS_CANONICAL_LEN, 6_293_512);
        assert_eq!(SCALE_DEFAULT, 1 << 20);
    }


    #[test]
    fn window_lag_and_capacity() {
        let mut w = Window::new();
        assert!(w.is_empty());
        assert_eq!(w.lag(0), None);
        assert_eq!(w.lag(1), None);
        for i in 0..5u64 {
            w.push(obs(i, 0, 0));
        }
        assert_eq!(w.len(), 5);
        assert_eq!(w.lag(1).unwrap().delta.0[0], 4);
        assert_eq!(w.lag(5).unwrap().delta.0[0], 0);
        assert_eq!(w.lag(6), None);
        let seq: Vec<u64> = w.iter().map(|o| o.delta.0[0]).collect();
        assert_eq!(seq, vec![0, 1, 2, 3, 4]);
        for i in 5..=(AR_ORDER + 10) as u64 {
            w.push(obs(i, 0, 0));
        }
        assert_eq!(w.len(), AR_ORDER);
        assert!(w.is_full());
        assert_eq!(w.lag(1).unwrap().delta.0[0], (AR_ORDER + 10) as u64);
        assert_eq!(w.lag(AR_ORDER).unwrap().delta.0[0], 11);
        w.clear();
        assert!(w.is_empty());
    }


    #[test]
    fn reference_parameterization_pins() {
        let w = Weights::reference();
        for j in 0..DIMS {
            assert_eq!(w.ar(j, 1), Q15::from_bits(16384));
            assert_eq!(w.ar(j, 2), Q15::from_bits(8192));
            assert_eq!(w.ar(j, 3), Q15::from_bits(4096));
            assert_eq!(w.ar(j, 4), Q15::from_bits(2048));
            for k in 5..=AR_ORDER {
                assert_eq!(w.ar(j, k), Q15::ZERO);
            }
            assert_eq!(w.bias(j), 0);
            assert_eq!(w.scale(j), SCALE_DEFAULT);
        }
        for j in 0..DIMS {
            for k in 1..=AR_ORDER {
                assert_eq!(w.fee(j, k), Q15::ZERO);
                assert_eq!(w.bucket(j, k), Q15::ZERO);
            }
        }
        let sum: i64 = (1..=4).map(|k| w.ar(0, k).to_bits() as i64).sum();
        assert_eq!(sum, 30_720, "the AR core sums to 15/16 — no unit root");
    }


    #[test]
    fn reference_predicts_over_constant_streams() {
        let mut f = Forecaster::reference();
        assert!(f.predict().is_zero(), "empty window → the zero-bias prediction");
        f.push_observation(obs(1_000_000, 0, 0));
        assert_eq!(f.predict().0[0], 500_000, "n=1: 1/2");
        f.push_observation(obs(1_000_000, 0, 0));
        assert_eq!(f.predict().0[0], 750_000, "n=2: +1/4");
        f.push_observation(obs(1_000_000, 0, 0));
        assert_eq!(f.predict().0[0], 875_000, "n=3: +1/8");
        f.push_observation(obs(1_000_000, 0, 0));
        assert_eq!(f.predict().0[0], 937_500, "n≥4: the full 15/16");
        for _ in 0..100 {
            f.push_observation(obs(1_000_000, 0, 0));
        }
        assert_eq!(f.predict().0[0], 937_500);
        assert!(f.predict().0.iter().all(|&v| v == 937_500));
    }


    #[test]
    fn the_named_rounding_point_is_half_even() {
        // Weight 1/2 on lag 1; a single observation x: the scaling point
        // sees exactly x/2.
        for (x, want) in [(1u64, 0u64), (3, 2), (5, 2), (7, 4), (9, 4)] {
            let mut f = Forecaster::reference();
            f.push_observation(obs(x, 0, 0));
            assert_eq!(f.predict().0[0], want, "x={x}");
        }
    }


    #[test]
    fn negative_predictions_wrap() {
        let mut f = Forecaster::reference();
        f.weights_mut().set_ar(0, 1, Q15::from_bits(i16::MAX));
        f.push_observation(obs(u64::MAX, 0, 0));
        // acc = 32767·(−1); /2^15 → −0.99997 → −1; wrap → MAX.
        assert_eq!(f.predict().0[0], u64::MAX);
        assert_eq!(f.predict().0[1], 0, "the untouched dimensions sit at bias 0");
    }


    #[test]
    fn fee_and_bucket_channels() {
        let mut f = Forecaster::reference();
        f.weights_mut().set_fee(0, 1, Q15::from_bits(1 << 14));
        f.weights_mut().set_bucket(1, 1, Q15::from_bits(1 << 14));
        f.push_observation(obs(0, 1000, 9));
        assert_eq!(f.predict().0[0], 500, "0.5 × 1000");
        assert_eq!(f.predict().0[1], 4, "0.5 × 9 = 4.5 → tie to even");


        // Lag-2 sees the OLDER observation.
        let mut g = Forecaster::reference();
        g.weights_mut().set_bucket(0, 2, Q15::from_bits(1 << 14));
        g.push_observation(obs(0, 0, 14));
        g.push_observation(obs(0, 0, 3));
        assert_eq!(g.predict().0[0], 7, "0.5 × lag-2 bucket 14");
    }


    #[test]
    fn bias_carries_the_level() {
        let mut f = Forecaster::new(Weights::zero());
        f.weights_mut().set_bias(5, -7);
        for _ in 0..10 {
            f.push_observation(obs(u64::MAX - 3, 999, 15));
        }
        let p = f.predict();
        assert_eq!(p.0[5], (-7i64) as u64);
        assert!(p.0.iter().enumerate().all(|(j, &v)| j == 5 || v == 0));
        let g = Forecaster::new(Weights::zero());
        assert!(g.predict().is_zero());
    }


    #[test]
    fn prediction_reads_only_the_window_tail() {
        let mut f = Forecaster::reference();
        for _ in 0..(AR_ORDER - 4) {
            f.push_observation(obs(0, 0, 0));
        }
        for _ in 0..4 {
            f.push_observation(obs(1_000_000, 0, 0));
        }
        assert_eq!(f.predict().0[0], 937_500);
        // A full refill evicts the signal entirely.
        for _ in 0..AR_ORDER {
            f.push_observation(obs(0, 0, 0));
        }
        assert_eq!(f.predict().0[0], 0);
    }


    #[test]
    fn adversarial_extremes_are_total() {
        let mut f = Forecaster::reference();
        {
            let w = f.weights_mut();
            for a in w.ar_slice_mut() {
                *a = Q15::from_bits(i16::MIN);
            }
            for a in w.fee_slice_mut() {
                *a = Q15::from_bits(i16::MIN);
            }
            for a in w.bucket_slice_mut() {
                *a = Q15::from_bits(i16::MIN);
            }
            w.set_bias(0, i64::MIN);
        }
        for _ in 0..AR_ORDER {
            f.push_observation(Observation {
                delta: Delta([u64::MAX; DIMS]),
                fee_sum: u64::MAX,
                bucket: 15,
            });
        }
        let p = f.predict();
        assert_eq!(p, f.predict(), "determinism under extremes");
        assert_eq!(p, predict_reference(&f));
        // The independent closed form for dimension 0.
        let per_lag = i128::from(i16::MIN as i64) * i128::from(-1i64)
            + i128::from(i16::MIN as i64) * i128::from(u64::MAX)
            + i128::from(i16::MIN as i64) * i128::from(15u64);
        let acc = per_lag * AR_ORDER as i128;
        let want = wrap_mod_2_64(i128::from(i64::MIN) + round_half_even_pow2(acc, Q15_FRAC_BITS));
        assert_eq!(p.0[0], want);
    }


    #[test]
    fn canonical_layout_pins_and_roundtrips() {
        let w = Weights::reference();
        let b = w.canonical_bytes();
        assert_eq!(b.len(), WEIGHTS_CANONICAL_LEN);
        assert_eq!(i16::from_le_bytes([b[0], b[1]]), 16384);
        assert_eq!(i16::from_le_bytes([b[2], b[3]]), 8192);
        assert_eq!(i16::from_le_bytes([b[4], b[5]]), 4096);
        assert_eq!(i16::from_le_bytes([b[6], b[7]]), 2048);
        assert_eq!(i16::from_le_bytes([b[8], b[9]]), 0);
        let d1 = 1024 * 2;
        assert_eq!(i16::from_le_bytes([b[d1], b[d1 + 1]]), 16384, "dim 1 lag 1");
        let bias_off = 3 * PER_CHANNEL * 2;
        assert_eq!(u64::from_le_bytes(b[bias_off..bias_off + 8].try_into().unwrap()), 0);
        let scale_off = bias_off + DIMS * 8;
        assert_eq!(
            u64::from_le_bytes(b[scale_off..scale_off + 8].try_into().unwrap()),
            SCALE_DEFAULT
        );
        assert_eq!(Weights::from_canonical(&b).unwrap(), w);
        assert!(Weights::from_canonical(&b[..b.len() - 1]).is_err());
        let mut long = b.clone();
        long.push(0);
        assert!(Weights::from_canonical(&long).is_err());
    }


    #[test]
    fn moments_canonical_roundtrip() {
        let mut m = Moments::zero();
        assert_eq!(m.m.len(), LEARNABLE);
        assert_eq!(m.v.len(), LEARNABLE);
        m.m[0] = -1;
        m.m[LEARNABLE - 1] = i128::MIN;
        m.v[0] = u128::MAX;
        m.step = 86_400;
        let b = m.canonical_bytes();
        assert_eq!(b.len(), MOMENTS_CANONICAL_LEN);
        assert_eq!(Moments::from_canonical(&b).unwrap(), m);
        assert!(Moments::from_canonical(&b[..b.len() - 1]).is_err());
        let mut long = b.clone();
        long.push(0);
        assert!(Moments::from_canonical(&long).is_err());
    }


    #[test]
    fn state_root_binds_every_committed_component() {
        let f = Forecaster::reference();
        let r = f.state_root();
        assert_eq!(r, f.state_root());


        let mut g = f.clone();
        g.weights_mut().set_ar(0, 5, Q15::from_bits(1));
        assert_ne!(g.state_root(), r);
        let mut g = f.clone();
        g.weights_mut().set_bias(63, 1);
        assert_ne!(g.state_root(), r);
        let mut g = f.clone();
        g.weights_mut().set_scale(63, 5);
        assert_ne!(g.state_root(), r);
        let mut g = f.clone();
        g.moments_mut().m[0] = 1;
        assert_ne!(g.state_root(), r);
        let mut g = f.clone();
        g.moments_mut().v[LEARNABLE - 1] = 1;
        assert_ne!(g.state_root(), r);
        let mut g = f.clone();
        g.moments_mut().step = 1;
        assert_ne!(g.state_root(), r);


        // The window is NOT committed (erratum 167) — re-derivable.
        let mut g = f.clone();
        g.push_observation(obs(5, 0, 0));
        assert_eq!(g.state_root(), r);


        // The literal formula.
        let mut pre = Vec::new();
        pre.extend_from_slice(DERIVED_FS.as_bytes());
        pre.extend_from_slice(&f.weights().canonical_bytes());
        pre.extend_from_slice(&f.moments().canonical_bytes());
        assert_eq!(f.state_root().as_bytes(), blake3::hash(&pre).as_bytes());
    }


    #[test]
    fn reset_to_reference() {
        let mut f = Forecaster::reference();
        for i in 0..10u64 {
            f.push_observation(obs(i, i, (i % 16) as u16));
        }
        f.weights_mut().set_ar(0, 1, Q15::MIN);
        f.moments_mut().m[7] = -42;
        f.moments_mut().v[9] = 99;
        f.moments_mut().step = 12;
        f.reset_to_reference();
        assert_eq!(f, Forecaster::reference());
    }


    fn random_forecaster(rng: &mut SplitMix64) -> Forecaster {
        let mut w = Weights::zero();
        for a in w.ar_slice_mut() {
            *a = Q15::from_bits(rng.next_u64() as i16);
        }
        for a in w.fee_slice_mut() {
            *a = Q15::from_bits(rng.next_u64() as i16);
        }
        for a in w.bucket_slice_mut() {
            *a = Q15::from_bits(rng.next_u64() as i16);
        }
        for b in w.bias_slice_mut() {
            *b = rng.next_u64() as i64;
        }
        for s in w.scale_slice_mut() {
            *s = rng.next_u64();
        }
        let mut f = Forecaster::new(w);
        let n = (rng.next_u64() % 48) as usize;
        for _ in 0..n {
            f.push_observation(Observation {
                delta: Delta(std::array::from_fn(|_| rng.next_u64())),
                fee_sum: rng.next_u64(),
                bucket: (rng.next_u64() % 16) as u16,
            });
        }
        let mom = f.moments_mut();
        for _ in 0..8 {
            let i = (rng.next_u64() as usize) % LEARNABLE;
            mom.m[i] = (rng.next_u64() as i128) - (1i128 << 100);
            mom.v[i] = u128::from(rng.next_u64()) << 64;
        }
        mom.step = rng.next_u64();
        f
    }


    proptest! {
        #[test]
        fn prop_predict_matches_the_reference_traversal(seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed);
            for _ in 0..4 {
                let f = random_forecaster(&mut rng);
                prop_assert_eq!(f.predict(), predict_reference(&f));
            }
        }


        #[test]
        fn prop_canonical_roundtrips(seed in any::<u64>()) {
            let mut rng = SplitMix64::new(seed);
            let f = random_forecaster(&mut rng);
            let w = Weights::from_canonical(&f.weights().canonical_bytes()).unwrap();
            prop_assert_eq!(w, *f.weights());
            let m = Moments::from_canonical(&f.moments().canonical_bytes()).unwrap();
            prop_assert_eq!(m, *f.moments());
        }
    }
}
