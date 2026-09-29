//! The relay-local traffic shaper (WP §6.2, §10.5; erratum 155): an
//! order-1 linear learner over per-bucket packet counts. Knowledge-layer
//! class — local, non-canonical, never a consensus input.

pub const SMOOTH_SHIFT: u32 = 4;
pub const DEFAULT_FLOOR: u64 = 1;
pub const DEFAULT_CAP: u64 = 16;

/// Level tracker: a Q32 EMA (rate 1/16) over observed bucket counts, a
/// floor below which cover is always emitted, and a per-call emission
/// cap. `dummies_for(observed)` returns how many dummy packets to emit
/// now so the bucket reaches at least the predicted level (or the floor
/// during warmup). Deterministic in its observation sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoverModel {
    ema_q32: u64,
    floor: u64,
    cap: u64,
}

impl CoverModel {
    pub fn new(floor: u64, cap: u64) -> CoverModel {
        assert!(floor <= cap, "cover floor exceeds cap");
        CoverModel { ema_q32: 0, floor, cap }
    }

    pub fn genesis() -> CoverModel {
        CoverModel::new(DEFAULT_FLOOR, DEFAULT_CAP)
    }

    pub fn floor(&self) -> u64 {
        self.floor
    }

    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// Record a completed bucket's observed count; advances the level.
    pub fn observe(&mut self, observed: u64) {
        let obs_term = observed.saturating_mul(1u64 << (32 - SMOOTH_SHIFT));
        self.ema_q32 = self
            .ema_q32
            .saturating_sub(self.ema_q32 >> SMOOTH_SHIFT)
            .saturating_add(obs_term);
    }

    /// The predicted per-bucket level, rounded.
    pub fn predict(&self) -> u64 {
        ((self.ema_q32 as u128 + (1u128 << 31)) >> 32) as u64
    }

    /// Cover packets to emit now for a bucket with `observed_so_far`
    /// real packets: the shortfall to the predicted level (or the floor),
    /// capped.
    pub fn dummies_for(&self, observed_so_far: u64) -> u64 {
        let target = self.predict().max(self.floor);
        target.saturating_sub(observed_so_far).min(self.cap)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn warmup_floor() {
        let m = CoverModel::genesis();
        assert_eq!(m.floor(), 1);
        assert_eq!(m.cap(), 16);
        assert_eq!(m.predict(), 0);
        assert_eq!(m.dummies_for(0), 1, "the floor carries warmup");
        assert_eq!(m.dummies_for(1), 0);
        assert_eq!(m.dummies_for(5), 0, "above the floor, nothing");
    }

    #[test]
    fn tracks_a_constant_level() {
        let mut m = CoverModel::new(1, 16);
        for _ in 0..64 {
            m.observe(10);
        }
        assert!((9..=11).contains(&m.predict()), "converged to 10, got {}", m.predict());
        assert_eq!(m.dummies_for(10), 0);
        assert_eq!(m.dummies_for(7), 3);
        assert_eq!(m.dummies_for(0), 10);
    }

    #[test]
    fn rises_conservatively_and_falls_slowly() {
        let mut m = CoverModel::new(1, 16);
        for _ in 0..32 {
            m.observe(0);
        }
        assert_eq!(m.predict(), 0);
        m.observe(100);
        assert!(m.predict() < 100, "one bucket moves the EMA by 1/16");
        for _ in 0..200 {
            m.observe(100);
        }
        assert!(m.predict() >= 90);
        for _ in 0..32 {
            m.observe(0);
        }
        assert!(m.predict() > 0, "the level decays, it does not collapse");
        assert!(m.predict() < 90);
    }

    #[test]
    fn cap_binds_and_floor_survives_high_traffic() {
        let mut m = CoverModel::new(1, 4);
        for _ in 0..64 {
            m.observe(1_000);
        }
        assert_eq!(m.dummies_for(0), 4, "the cap binds");
        let mut quiet = CoverModel::new(1, 4);
        for _ in 0..64 {
            quiet.observe(0);
        }
        assert_eq!(quiet.dummies_for(0), 1, "the floor persists under silence");
    }

    #[test]
    fn determinism_and_value_semantics() {
        let mut a = CoverModel::genesis();
        let mut b = CoverModel::genesis();
        for i in 0..20u64 {
            a.observe(i);
            b.observe(i);
        }
        assert_eq!(a, b);
        assert_eq!(a.dummies_for(3), b.dummies_for(3));
        let snap = a.clone();
        a.observe(5);
        assert_ne!(a, snap);
        assert_eq!(b, snap, "b follows the same history");
    }

    #[test]
    fn overflow_safety() {
        let mut m = CoverModel::new(1, 16);
        m.observe(u64::MAX);
        let _ = m.predict();
        let _ = m.dummies_for(u64::MAX);
        m.observe(0);
        let _ = m.predict();
        assert!(CoverModel::new(0, 0).dummies_for(0) == 0);
        assert!(CoverModel::new(5, 5).dummies_for(0) == 5);
    }
}
