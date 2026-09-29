//! Componentwise Huber scoring and the robust scale estimate (WP §10.3;
//! erratum 169): the centered residual, the bounded blame, the Q32 loss,
//! and the scale EMA.


use nerv_codec::codec_w::Delta;


use crate::forecaster::DIMS;


pub const S_MIN: u64 = 1;
pub const S_MAX: u64 = 1 << 40;
pub const SCALE_EMA_SHIFT: u32 = 4;
/// |b| = 1 in Q40.
pub const BLAME_ONE: i128 = 1 << 40;


/// r_j = centerlift(Δ_B[j] − Δ̂[j]) — the wrapping subtraction's signed
/// reading is the centered representative.
pub fn residual(delta_b: &Delta, pred: &Delta) -> [i64; DIMS] {
    let mut r = [0i64; DIMS];
    for (j, o) in r.iter_mut().enumerate() {
        *o = delta_b.0[j].wrapping_sub(pred.0[j]) as i64;
    }
    r
}


fn normalized(abs_r: u64, s: u64) -> u128 {
    let s = u128::from(s.max(S_MIN));
    (u128::from(abs_r) << 40) / s
}


/// The blame b_j = ρ′(r_j/s_j) in Q40: u within the band, ±1 beyond —
/// |b| ≤ 1 by construction (bounded blame, P8).
pub fn blame(r: &[i64; DIMS], scales: &[u64; DIMS]) -> [i128; DIMS] {
    let mut b = [0i128; DIMS];
    for j in 0..DIMS {
        let u = normalized(r[j].unsigned_abs(), scales[j]);
        let mag = u.min(1u128 << 40) as i128;
        b[j] = if r[j] < 0 { -mag } else { mag };
    }
    b
}


/// L_j = s_j·ρ(u_j) in Q32 (delta units × 2^32).
pub fn loss_per_dim(r: &[i64; DIMS], scales: &[u64; DIMS]) -> [u128; DIMS] {
    let mut l = [0u128; DIMS];
    for j in 0..DIMS {
        let s = u128::from(scales[j].max(S_MIN));
        let abs_r = u128::from(r[j].unsigned_abs());
        let u = normalized(r[j].unsigned_abs(), scales[j]);
        l[j] = if u <= 1u128 << 40 {
            (s * u * u) >> 49
        } else {
            (abs_r << 32) - (s << 31)
        };
    }
    l
}


pub fn total_loss(r: &[i64; DIMS], scales: &[u64; DIMS]) -> u128 {
    loss_per_dim(r, scales).iter().fold(0u128, |a, &x| a + x)
}


/// The robust scale EMA: s ← clamp(s + ((|r|−s) ≫ 4), S_MIN, S_MAX).
pub fn scale_ema(s: u64, abs_r: u64) -> u64 {
    let s_c = i128::from(s.max(S_MIN));
    let d = (i128::from(abs_r) - s_c) >> SCALE_EMA_SHIFT;
    (s_c + d).clamp(S_MIN as i128, S_MAX as i128) as u64
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;


    fn d(v: u64) -> Delta {
        Delta([v; DIMS])
    }


    #[test]
    fn residual_is_the_centered_difference() {
        assert_eq!(residual(&d(100), &d(100)), [0i64; DIMS]);
        assert_eq!(residual(&d(100), &d(99))[0], 1);
        assert_eq!(residual(&d(99), &d(100))[0], -1);
        // Wraparound: 0 − 1 ≡ −1.
        assert_eq!(residual(&d(0), &d(1))[0], -1);
        assert_eq!(residual(&d(1), &d(0))[0], 1);
        assert_eq!(residual(&d(0), &d(u64::MAX))[0], 1);
        assert_eq!(residual(&d(u64::MAX), &d(0))[0], -1);
        assert_eq!(residual(&d(0), &d(1u64 << 63))[0], i64::MIN);
    }


    #[test]
    fn blame_band_and_cap() {
        let r = [25i64; DIMS];
        let s = [50u64; DIMS];
        assert_eq!(blame(&r, &s)[0], 1i128 << 39, "u = 1/2 → b = 1/2");


        let r = [50i64; DIMS];
        assert_eq!(blame(&r, &s)[0], BLAME_ONE, "the band edge is exactly 1");


        let r = [100i64; DIMS];
        assert_eq!(blame(&r, &s)[0], BLAME_ONE, "capped beyond the band");


        let r = [-100i64; DIMS];
        assert_eq!(blame(&r, &s)[0], -BLAME_ONE);


        let r = [0i64; DIMS];
        assert_eq!(blame(&r, &s)[0], 0);


        // The scale floor guards a corrupt zero scale.
        let r = [5i64; DIMS];
        let s0 = [0u64; DIMS];
        assert_eq!(blame(&r, &s0)[0], BLAME_ONE);
    }


    #[test]
    fn loss_continuity_and_pins() {
        let s = [50u64; DIMS];
        let r = [0i64; DIMS];
        assert_eq!(loss_per_dim(&r, &s)[0], 0);


        // The band boundary: quadratic and linear agree exactly.
        let r = [50i64; DIMS];
        let quad = (50u128 * (1u128 << 40) * (1u128 << 40)) >> 49;
        let lin = (50u128 << 32) - (50u128 << 31);
        assert_eq!(quad, lin, "continuity at |r| = s");
        assert_eq!(loss_per_dim(&r, &s)[0], quad);
        assert_eq!(loss_per_dim(&r, &s)[0], 50u128 << 31, "L = s/2 in Q32");


        // Quadratic: |r| = s/2 → L = s/8.
        let r = [25i64; DIMS];
        assert_eq!(loss_per_dim(&r, &s)[0], 50u128 << 29);


        // Linear: |r| = 2s → L = 3s/2.
        let r = [100i64; DIMS];
        assert_eq!(loss_per_dim(&r, &s)[0], 3 * (50u128 << 31));


        // The total sums.
        let r = [25i64; DIMS];
        assert_eq!(total_loss(&r, &s), u128::from(DIMS as u64) * (50u128 << 29));
    }


    #[test]
    fn loss_monotone_in_magnitude() {
        let s = [1000u64; DIMS];
        let mut prev = [0u128; DIMS];
        for m in 0..=3000u64 {
            let r = [m as i64; DIMS];
            let l = loss_per_dim(&r, &s);
            for j in 0..DIMS {
                assert!(l[j] >= prev[j], "monotone at m={m}");
                prev[j] = l[j];
            }
        }
        // Extremes are total.
        let r = [i64::MIN; DIMS];
        let _ = total_loss(&r, &s);
        let r = [i64::MAX; DIMS];
        let _ = total_loss(&r, &s);
        let s_max = [S_MAX; DIMS];
        let _ = total_loss(&r, &s_max);
    }


    #[test]
    fn scale_ema_tracks_and_clamps() {
        assert_eq!(scale_ema(100, 0), 93, "floor(−100/16) = −7");
        assert_eq!(scale_ema(100, 200), 106);
        assert_eq!(scale_ema(16, 0), 15);
        assert_eq!(scale_ema(1, 0), 1, "clamped at S_MIN");
        assert_eq!(scale_ema(0, 0), 1, "a corrupt zero clamps up");
        assert_eq!(scale_ema(S_MAX, u64::MAX), S_MAX, "clamped at S_MAX");
        assert_eq!(scale_ema(S_MAX, 0), S_MAX - (S_MAX >> 4));
        // Convergence toward |r|.
        let mut s = 1u64;
        for _ in 0..200 {
            s = scale_ema(s, 10_000);
        }
        assert!((9_900..=10_100).contains(&s), "tracked to {s}");
        // Determinism.
        assert_eq!(scale_ema(123, 456), scale_ema(123, 456));
    }
}
