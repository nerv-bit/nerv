//! Deterministic integer Adam (WP §10.2; erratum 168): Newton √,
//! iterative β^t, gradient and update clipping, saturating parameter
//! bounds. Every operation is integer-exact and platform-stable.


use nerv_core::fixed_point::round_half_even_pow2;
use nerv_core::Q15;


use crate::forecaster::{LEARNABLE, PER_CHANNEL, Moments, Weights};


/// β1 = 1 − 2^-4.
pub const K1: u32 = 4;
/// β2 = 1 − 2^-8.
pub const K2: u32 = 8;
/// The published gradient clip (true units; raw Q40 = 2^60).
pub const GRAD_CLIP_TRUE: i128 = 1 << 20;
pub const GRAD_CLIP_RAW: i128 = GRAD_CLIP_TRUE << 40;
/// The update-ratio clip: |m̂/(√v̂+ε)| ≤ 2.
pub const RATIO_CLIP_RAW: i128 = 2 << 40;
/// The per-type step sizes (genesis-config): Q15 units and bias units.
pub const ALPHA_W: i64 = 16;
pub const ALPHA_B: i64 = 1 << 20;


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AdamError {
    #[error("gradient vector has {found} entries, expected {expected}")]
    GradLen { found: usize, expected: usize },
}


/// Newton integer square root: bit-length initial guess (≥ √n), the
/// monotone descent y = (x + n/x)/2, adjust-down termination.
pub fn isqrt(n: u128) -> u64 {
    if n == 0 {
        return 0;
    }
    let bits = 128 - n.leading_zeros();
    let mut x: u128 = 1u128 << ((bits + 1) / 2);
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            break;
        }
        x = y;
    }
    while x > 0 && x.checked_mul(x).is_some_and(|s| s > n) {
        x -= 1;
    }
    x as u64
}


const fn q64_mul(a: u128, b: u128) -> u128 {
    (a * b) >> 64
}


/// β^t in Q64 by binary exponentiation with floor-multiplies; β =
/// 1 − 2^-k has an exact Q64 form.
pub fn beta_pow_q64(k: u32, t: u64) -> u128 {
    debug_assert!(k >= 1 && k <= 63);
    let base = ((1u128 << k) - 1) << (64 - k as u32);
    let mut result: u128 = 1u128 << 64;
    let mut exp = t;
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            result = q64_mul(result, b);
        }
        b = q64_mul(b, b);
        exp >>= 1;
    }
    result
}


/// 1 − β^t in Q64. Positive for every t ≥ 1 (tested).
pub fn bias_correction_q64(k: u32, t: u64) -> u128 {
    (1u128 << 64) - beta_pow_q64(k, t)
}


/// Floor division (divisor > 0).
fn floor_div(a: i128, b: i128) -> i128 {
    debug_assert!(b > 0);
    let q = a / b;
    if a % b != 0 && a < 0 {
        q - 1
    } else {
        q
    }
}


/// One Adam step over the canonical parameter order (ar ‖ fee ‖ bucket
/// ‖ bias). Total on corrupt moments (saturating multiplies); exact on
/// the reachable set.
#[allow(clippy::too_many_lines)]
pub fn step(
    weights: &mut Weights,
    moments: &mut Moments,
    grads: &[i128],
) -> Result<(), AdamError> {
    if grads.len() != LEARNABLE {
        return Err(AdamError::GradLen { found: grads.len(), expected: LEARNABLE });
    }
    let t = moments.step.saturating_add(1);
    moments.step = t;
    let bc1 = bias_correction_q64(K1, t) as i128;
    let s_bc2 = u128::from(isqrt(bias_correction_q64(K2, t)));


    for i in 0..LEARNABLE {
        let g = grads[i].clamp(-GRAD_CLIP_RAW, GRAD_CLIP_RAW);


        let d = g.saturating_sub(moments.m[i]);
        moments.m[i] = moments.m[i].saturating_add(d >> K1);


        let ga = g.unsigned_abs() as u128;
        let g2 = ga * ga;
        let v = moments.v[i];
        moments.v[i] = if g2 >= v {
            v + ((g2 - v) >> K2)
        } else {
            v - ((v - g2) >> K2)
        };


        let m_hat = floor_div(moments.m[i].saturating_mul(1i128 << 64), bc1);
        let sv = (u128::from(isqrt(moments.v[i]))) << 32;
        let vhat = (sv / s_bc2) as i128;
        let denom = vhat + 1;
        let ratio = floor_div(m_hat.saturating_mul(1i128 << 40), denom)
            .clamp(-RATIO_CLIP_RAW, RATIO_CLIP_RAW);


        let (alpha, is_q15) = if i < 3 * PER_CHANNEL {
            (ALPHA_W, true)
        } else {
            (ALPHA_B, false)
        };
        let delta = -round_half_even_pow2(i128::from(alpha) * ratio, 40);


        if is_q15 {
            let idx = i % PER_CHANNEL;
            let channel = i / PER_CHANNEL;
            let old = match channel {
                0 => weights.ar_slice_mut()[idx].to_bits(),
                1 => weights.fee_slice_mut()[idx].to_bits(),
                _ => weights.bucket_slice_mut()[idx].to_bits(),
            } as i128;
            let new =
                (old + delta).clamp(i16::MIN as i128, i16::MAX as i128) as i16;
            match channel {
                0 => weights.ar_slice_mut()[idx] = Q15::from_bits(new),
                1 => weights.fee_slice_mut()[idx] = Q15::from_bits(new),
                _ => weights.bucket_slice_mut()[idx] = Q15::from_bits(new),
            }
        } else {
            let j = i - 3 * PER_CHANNEL;
            let d64 = delta.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            weights.bias_slice_mut()[j] = weights.bias_slice_mut()[j].saturating_add(d64);
        }
    }
    Ok(())
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::forecaster::{DIMS, Moments, Weights};
    use crate::testutil::SplitMix64;


    fn isqrt_reference(n: u128) -> u64 {
        let (mut lo, mut hi) = (0u128, 1u128 << 64);
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if mid * mid <= n {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo as u64
    }


    #[test]
    fn isqrt_matches_the_binary_search_reference() {
        for n in 0..=65_536u128 {
            assert_eq!(u128::from(isqrt(n)), isqrt_reference(n), "n={n}");
        }
        let mut rng = SplitMix64::new(0xADE1);
        for _ in 0..2_000 {
            let n = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;
            assert_eq!(u128::from(isqrt(n)), isqrt_reference(n), "n={n}");
        }
        for x in [0u64, 1, 2, 3, 255, 256, 65535, 65536] {
            let sq = u128::from(x) * u128::from(x);
            assert_eq!(u128::from(isqrt(sq)), isqrt_reference(sq));
            if sq > 0 {
                assert_eq!(u128::from(isqrt(sq - 1)), isqrt_reference(sq - 1));
            }
            assert_eq!(u128::from(isqrt(sq + 1)), isqrt_reference(sq + 1));
        }
        for n in [u128::MAX, u128::MAX - 1, 1u128 << 127, (1u128 << 126) + 12345] {
            assert_eq!(u128::from(isqrt(n)), isqrt_reference(n));
        }
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(4), 2);
    }


    #[test]
    fn bias_correction_pins_and_positivity() {
        assert_eq!(bias_correction_q64(4, 1), 1u128 << 60);
        assert_eq!(bias_correction_q64(8, 1), 1u128 << 56);
        assert_eq!(bias_correction_q64(4, 2), 31 * (1u128 << 56));
        assert_eq!(beta_pow_q64(4, 0), 1u128 << 64);
        assert_eq!(bias_correction_q64(4, 0), 0, "t = 0 — callers use t ≥ 1");
        for k in [K1, K2] {
            let mut prev = 0u128;
            for t in 1..=2_000u64 {
                let bc = bias_correction_q64(k, t);
                assert!(bc > 0, "k={k} t={t}");
                assert!(bc <= 1u128 << 64);
                assert!(bc >= prev, "monotone in t: k={k} t={t}");
                prev = bc;
            }
            assert!(prev > (1u128 << 64) / 2, "converging to 1");
        }
    }


    fn zero_state() -> (Weights, Moments) {
        (Weights::zero(), Moments::zero())
    }


    #[test]
    fn the_t1_spike_is_the_classic_step() {
        // g = 2^60 (the clip) on every parameter from a zero state:
        // m = 2^56, v = 2^112, m̂ = 2^60, √v̂ = 2^60 → ratio = 2^40−1,
        // δ_w = −16, δ_b = −2^20.
        let (mut w, mut m) = zero_state();
        let grads = vec![GRAD_CLIP_RAW; LEARNABLE];
        step(&mut w, &mut m, &grads).unwrap();
        assert_eq!(m.step, 1);
        assert_eq!(m.m[0], 1i128 << 56);
        assert_eq!(m.v[0], 1u128 << 112);
        assert_eq!(w.ar(0, 1).to_bits(), -16);
        assert_eq!(w.fee(0, 1).to_bits(), -16);
        assert_eq!(w.bucket(0, 1).to_bits(), -16);
        assert_eq!(w.bias(0), -ALPHA_B);
        assert_eq!(w.bias(DIMS - 1), -ALPHA_B);


        // The negative spike: ratio = −2^40 exactly (floor), δ_w = +16.
        let (mut w2, mut m2) = zero_state();
        let grads = vec![-GRAD_CLIP_RAW; LEARNABLE];
        step(&mut w2, &mut m2, &grads).unwrap();
        assert_eq!(w2.ar(0, 1).to_bits(), 16);
        assert_eq!(w2.bias(0), ALPHA_B);
    }


    #[test]
    fn gradient_clipping_binds() {
        let (mut w, mut m) = zero_state();
        step(&mut w, &mut m, &vec![GRAD_CLIP_RAW; LEARNABLE]).unwrap();
        let (mut w2, mut m2) = zero_state();
        let huge = vec![1i128 << 100; LEARNABLE];
        step(&mut w2, &mut m2, &huge).unwrap();
        assert_eq!(w.ar(0, 1), w2.ar(0, 1));
        assert_eq!(w2.ar(0, 1).to_bits(), -16);
        assert_eq!(m2.m[0], 1i128 << 56, "the clipped gradient is what enters m");
    }


    #[test]
    fn ratio_clip_pins_the_step() {
        // m = 2^60, v = 0, zero gradient: the decayed m̂ over ε = 1
        // overflows the ratio, which clips to ±2 → δ_w = ∓32.
        let (mut w, mut m) = zero_state();
        m.m[0] = 1i128 << 60;
        m.v[0] = 0;
        step(&mut w, &mut m, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(m.m[0], 15 * (1i128 << 52));
        assert_eq!(w.ar(0, 1).to_bits(), -32, "ratio clipped to −2");
        assert_eq!(w.bias(0), 0, "bias gradient was zero: ratio 0 → no move");


        let (mut w2, mut m2) = zero_state();
        m2.m[LEARNABLE - 1] = -(1i128 << 60);
        m2.v[LEARNABLE - 1] = 0;
        step(&mut w2, &mut m2, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(w2.bias(DIMS - 1), 2 * ALPHA_B, "ratio clipped to +2");
    }


    #[test]
    fn moment_decay_is_the_exact_ema() {
        let (mut w, mut m) = zero_state();
        step(&mut w, &mut m, &vec![GRAD_CLIP_RAW; LEARNABLE]).unwrap();
        assert_eq!(m.m[0], 1i128 << 56);
        assert_eq!(m.v[0], 1u128 << 112);
        step(&mut w, &mut m, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(m.m[0], 15 * (1i128 << 52), "m × 15/16");
        assert_eq!(m.v[0], 255 * (1u128 << 104), "v × 255/256");
        assert_eq!(m.step, 2);
    }


    #[test]
    fn zero_gradients_from_a_zero_state_move_nothing() {
        let (mut w, mut m) = zero_state();
        step(&mut w, &mut m, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(m.step, 1);
        assert_eq!(m.m.iter().sum::<i128>(), 0);
        assert!(m.v.iter().all(|&x| x == 0));
        assert_eq!(w, Weights::zero());
    }


    #[test]
    fn saturating_parameter_bounds() {
        let (mut w, mut m) = zero_state();
        w.set_ar(0, 1, nerv_core::Q15::from_bits(i16::MAX));
        w.set_bias(0, i64::MAX);
        // A negative gradient moves parameters UP (descent).
        step(&mut w, &mut m, &vec![-GRAD_CLIP_RAW; LEARNABLE]).unwrap();
        assert_eq!(w.ar(0, 1).to_bits(), i16::MAX, "saturated at +MAX");
        assert_eq!(w.bias(0), i64::MAX, "saturated at +MAX");


        let (mut w, mut m) = zero_state();
        w.set_ar(0, 1, nerv_core::Q15::from_bits(i16::MIN));
        w.set_bias(0, i64::MIN);
        step(&mut w, &mut m, &vec![GRAD_CLIP_RAW; LEARNABLE]).unwrap();
        assert_eq!(w.ar(0, 1).to_bits(), i16::MIN, "saturated at −MAX");
        assert_eq!(w.bias(0), i64::MIN);
    }


    #[test]
    fn determinism_and_length_error() {
        let (mut a, mut ma) = zero_state();
        let (mut b, mut mb) = zero_state();
        let grads: Vec<i128> = (0..LEARNABLE as i128).map(|i| (i * 7919) % (1 << 50)).collect();
        step(&mut a, &mut ma, &grads).unwrap();
        step(&mut b, &mut mb, &grads).unwrap();
        assert_eq!(a, b);
        assert_eq!(ma, mb);


        let (mut w, mut m) = zero_state();
        assert!(matches!(
            step(&mut w, &mut m, &vec![0i128; LEARNABLE - 1]),
            Err(AdamError::GradLen { found: LEARNABLE - 1, expected: LEARNABLE })
        ));
    }


    #[test]
    fn corrupt_moments_are_total() {
        let (mut w, mut m) = zero_state();
        m.m[0] = i128::MIN;
        m.m[1] = i128::MAX;
        m.v[0] = u128::MAX;
        m.v[1] = u128::MAX - 1;
        step(&mut w, &mut m, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(m.step, 1);
        let again = m.clone();
        step(&mut w, &mut m, &vec![0i128; LEARNABLE]).unwrap();
        assert_eq!(m.step, 2);
        assert_ne!(m, again, "deterministic, not frozen");
    }
}
