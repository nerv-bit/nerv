//! Tail-residual advisory flags (WP §10.5, §2.5; erratum 173): early,
//! public signals from the forecaster's residuals — consumed by
//! operators, researchers, and governance review. Never consensus,
//! never slashing; the ledger itself never moves.

use std::collections::VecDeque;

use crate::forecaster::DIMS;

/// The tail threshold: |r_j| ≥ TAIL_MULT · s_j (genesis-config).
pub const TAIL_MULT: u128 = 4;
/// The cascade threshold: this many tailed dimensions at once (half).
pub const CASCADE_DIMS: usize = 32;
/// The advisory ring's capacity.
pub const HISTORY: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnomalyFlags {
    pub height: u64,
    pub tail_dims: u32,
    pub cascade: bool,
    /// max_j |r_j|/s_j in Q40.
    pub max_normalized_q40: u128,
}

#[derive(Clone, Debug, Default)]
pub struct AnomalyTracker {
    history: VecDeque<AnomalyFlags>,
}

impl AnomalyTracker {
    pub fn new() -> AnomalyTracker {
        AnomalyTracker::default()
    }

    /// One block's advisory flags from the residual and the scales.
    pub fn observe(
        &mut self,
        height: u64,
        residual: &[i64; DIMS],
        scales: &[u64; DIMS],
    ) -> AnomalyFlags {
        let mut tail_dims = 0u32;
        let mut max_u = 0u128;
        for j in 0..DIMS {
            let s = u128::from(scales[j].max(1));
            let u = (u128::from(residual[j].unsigned_abs()) << 40) / s;
            if u >= TAIL_MULT << 40 {
                tail_dims += 1;
            }
            max_u = max_u.max(u);
        }
        let flags = AnomalyFlags {
            height,
            tail_dims,
            cascade: tail_dims as usize >= CASCADE_DIMS,
            max_normalized_q40: max_u,
        };
        self.history.push_back(flags);
        while self.history.len() > HISTORY {
            self.history.pop_front();
        }
        flags
    }

    pub fn history(&self) -> &VecDeque<AnomalyFlags> {
        &self.history
    }

    pub fn last(&self) -> Option<AnomalyFlags> {
        self.history.back().copied()
    }

    /// The trailing cascade frequency in permille (advisory).
    pub fn cascade_rate_permille(&self) -> u64 {
        let n = self.history.len();
        if n == 0 {
            return 0;
        }
        let cascades = self.history.iter().filter(|f| f.cascade).count() as u64;
        cascades * 1000 / n as u64
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn pins() {
        assert_eq!(TAIL_MULT, 4);
        assert_eq!(CASCADE_DIMS, 32);
        assert_eq!(HISTORY, 256);
        assert_eq!(DIMS, 64);
    }

    #[test]
    fn tail_threshold_is_exact() {
        let s = [100u64; DIMS];
        let mut t = AnomalyTracker::new();
        let f = t.observe(1, &[400i64; DIMS], &s);
        assert_eq!(f.tail_dims, DIMS as u32, "|r| = 4s is in the tail");
        assert!(f.cascade);
        assert_eq!(f.max_normalized_q40, 4 * (1u128 << 40));
        let f = t.observe(2, &[399i64; DIMS], &s);
        assert_eq!(f.tail_dims, 0);
        assert!(!f.cascade);
        assert_eq!(f.max_normalized_q40, 399 * (1u128 << 40) / 100);
    }

    #[test]
    fn cascade_threshold_and_negatives() {
        let s = [1u64; DIMS];
        let mut t = AnomalyTracker::new();
        let mut r = [0i64; DIMS];
        for v in r.iter_mut().take(31) {
            *v = 4;
        }
        let f = t.observe(1, &r, &s);
        assert_eq!(f.tail_dims, 31);
        assert!(!f.cascade);
        r[31] = 4;
        let f = t.observe(2, &r, &s);
        assert_eq!(f.tail_dims, 32);
        assert!(f.cascade);
        let f = t.observe(3, &[-4i64; DIMS], &s);
        assert_eq!(f.tail_dims, DIMS as u32, "negatives tail by magnitude");
        assert!(f.cascade);
        // Sparse tails don't cascade.
        let mut sparse = [0i64; DIMS];
        sparse[0] = 1_000_000;
        let f = t.observe(4, &sparse, &s);
        assert_eq!(f.tail_dims, 1);
        assert!(!f.cascade);
        assert!(f.max_normalized_q40 > (1u128 << 40));
    }

    #[test]
    fn scale_floor_and_extremes() {
        let mut t = AnomalyTracker::new();
        let f = t.observe(1, &[5i64; DIMS], &[0u64; DIMS]);
        assert_eq!(f.tail_dims, DIMS as u32, "a corrupt zero scale floors to 1");
        let f = t.observe(2, &[i64::MIN; DIMS], &[1u64; DIMS]);
        assert!(f.cascade);
        assert_eq!(f.max_normalized_q40, 1u128 << 103);
        let f = t.observe(3, &[0i64; DIMS], &[1u64; DIMS]);
        assert_eq!(f.max_normalized_q40, 0);
        assert_eq!(f.tail_dims, 0);
    }

    #[test]
    fn history_bounded_and_rate() {
        let mut t = AnomalyTracker::new();
        for h in 1..=300u64 {
            t.observe(h, &[4i64; DIMS], &[1u64; DIMS]);
        }
        assert_eq!(t.history().len(), HISTORY);
        assert_eq!(t.history().first().unwrap().height, 300 - HISTORY as u64 + 1);
        assert_eq!(t.last().unwrap().height, 300);
        assert_eq!(t.cascade_rate_permille(), 1000);
        for h in 301..=356u64 {
            t.observe(h, &[0i64; DIMS], &[1u64; DIMS]);
        }
        assert_eq!(t.cascade_rate_permille(), 200 * 1000 / 256);
        assert_eq!(AnomalyTracker::new().cascade_rate_permille(), 0);
        assert!(AnomalyTracker::new().last().is_none());
        assert!(AnomalyTracker::new().history().is_empty());
    }
}
