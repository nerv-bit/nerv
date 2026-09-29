//! Constraint evaluation at a point and the true-degree model (WP §5.3).
//!
//! `PointCollector` runs the AIR's own `eval` at one evaluation point — a
//! trace row pair over the LDE domain (prover, Goldilocks) or the opened
//! values at the out-of-domain point ζ (verifier, lifted) — folding every
//! asserted constraint into one challenge-field value
//!     F(x) = Σ_i α^i · C_i(x),
//! assertion order fixed by `eval` itself, so prover and verifier fold
//! identically. An honest trace makes F vanish on the whole trace domain
//! H — the divisibility by Z_H that the quotient polynomial proves; a
//! tampered row leaves F(h^i) ≠ 0 up to the α-fold's n/|EF| binding
//! error (security's α-fold term).
//!
//! The row selectors are collector INPUTS, instantiated as the selector
//! polynomials: is_first_row and is_last_row are the normalized Lagrange
//! polynomials of H's endpoints (degree N−1); is_transition is degree 1
//! (x − h^{N−1}). `NativeEval`'s 0/1 instantiation agrees with these in
//! exactly what matters: their zero sets on the rows.
//!
//! `measure_true` is the TRUE constraint-degree model for quotient
//! sizing: witness, preprocessed, and boundary-selector leaves each scale
//! a product's degree by N−1; a transition leaf adds 1. The witness-only
//! `sym::measure` degree is NOT the quotient degree.

use crate::air::builder::{Air, AirBuilder, AirExpr};
use crate::air::sym::SymExpr;
use crate::stark::domain::two_adic_generator;
use crate::stark::ext_field::ExtF;
use nerv_core::field::Goldilocks;

impl From<Goldilocks> for ExtF {
    fn from(g: Goldilocks) -> ExtF {
        ExtF::from_base(g)
    }
}

impl AirExpr for ExtF {
    fn zero() -> Self {
        ExtF::ZERO
    }
    fn one() -> Self {
        ExtF::ONE
    }
}

/// Lift one leaf value into the challenge field for folding.
pub trait LiftExtF {
    fn lift(self) -> ExtF;
}

impl LiftExtF for Goldilocks {
    fn lift(self) -> ExtF {
        ExtF::from_base(self)
    }
}

impl LiftExtF for ExtF {
    fn lift(self) -> ExtF {
        self
    }
}

/// The three row selectors at one point.
#[derive(Clone, Copy, Debug)]
pub struct SelectorVals<V> {
    pub first: V,
    pub last: V,
    pub transition: V,
}

/// The selector polynomials at an out-of-domain point: first =
/// Z_H(x)/(N·(x−1)), last = Z_H(x)/(N·(x−h^{N−1})) — the normalized
/// Lagrange polynomials of H's endpoints; transition = x − h^{N−1}.
/// Contract: x ∉ H (the driver rejects ζ with Z_H(ζ) = 0).
pub fn selector_vals_ext(log_n: usize, x: ExtF) -> SelectorVals<ExtF> {
    let n = 1u64 << log_n;
    let h_last = two_adic_generator(log_n).inverse(); // h^{N−1}
    let zh = x.pow(n) - ExtF::ONE;
    let n_inv = ExtF::from_base(Goldilocks::from_u64_reduce(n as u64).inverse());
    SelectorVals {
        first: zh * (x - ExtF::ONE).inverse() * n_inv,
        last: zh * (x - ExtF::from_base(h_last)).inverse() * n_inv,
        transition: x - ExtF::from_base(h_last),
    }
}

/// The single-point constraint folder.
pub struct PointCollector<'a, V> {
    cur: &'a [V],
    nxt: &'a [V],
    prep: &'a [V],
    pubs: &'a [V],
    sels: SelectorVals<V>,
    alpha: ExtF,
    alpha_pow: ExtF,
    fold: ExtF,
    count: usize,
}

impl<'a, V: AirExpr + LiftExtF + Copy> PointCollector<'a, V> {
    pub fn new(
        cur: &'a [V],
        nxt: &'a [V],
        prep: &'a [V],
        pubs: &'a [V],
        sels: SelectorVals<V>,
        alpha: ExtF,
    ) -> PointCollector<'a, V> {
        PointCollector {
            cur,
            nxt,
            prep,
            pubs,
            sels,
            alpha,
            alpha_pow: ExtF::ONE,
            fold: ExtF::ZERO,
            count: 0,
        }
    }

    /// Run the AIR's `eval` at this point. Returns (F(x), assertion count)
    /// — the count is the prover/verifier consistency guard.
    pub fn evaluate<A: Air<PointCollector<'a, V>>>(mut self, air: &A) -> (ExtF, usize) {
        air.eval(&mut self);
        (self.fold, self.count)
    }
}

impl<V: AirExpr + LiftExtF + Copy> AirBuilder for PointCollector<'_, V> {
    type Expr = V;

    fn witness(&self, offset: usize, col: usize) -> V {
        match offset {
            0 => self.cur[col],
            1 => self.nxt[col],
            _ => panic!("witness offset {offset}: the builder frame is 2 rows"),
        }
    }

    fn public(&self, idx: usize) -> V {
        self.pubs[idx]
    }

    fn preprocessed(&self, col: usize) -> V {
        self.prep[col]
    }

    fn is_first_row(&self) -> V {
        self.sels.first
    }

    fn is_last_row(&self) -> V {
        self.sels.last
    }

    fn is_transition(&self) -> V {
        self.sels.transition
    }

    fn assert_zero(&mut self, expr: V, _name: &'static str) {
        self.fold = self.fold + self.alpha_pow * expr.lift();
        self.alpha_pow = self.alpha_pow * self.alpha;
        self.count += 1;
    }
}

// ---------------------------------------------------------------------------
// The true-degree model.
// ---------------------------------------------------------------------------

/// (heavy leaves multiplied, uses the degree-1 transition selector).
fn true_degree_of(e: &SymExpr) -> (usize, bool) {
    match e {
        SymExpr::Const(_) | SymExpr::Pub(_) => (0, false),
        SymExpr::Wit { .. } | SymExpr::Prep(_) | SymExpr::First | SymExpr::Last => (1, false),
        SymExpr::Transition => (0, true),
        SymExpr::Add(a, b) | SymExpr::Sub(a, b) => {
            let (ka, ta) = true_degree_of(a);
            let (kb, tb) = true_degree_of(b);
            (ka.max(kb), ta || tb)
        }
        SymExpr::Mul(a, b) => {
            let (ka, ta) = true_degree_of(a);
            let (kb, tb) = true_degree_of(b);
            (ka + kb, ta || tb)
        }
    }
}

/// An AIR's true-degree profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrueDegree {
    pub max_heavy: usize,
    pub uses_transition: bool,
}

pub struct TrueDegreeBuilder {
    max_heavy: usize,
    uses_transition: bool,
}

impl AirBuilder for TrueDegreeBuilder {
    type Expr = SymExpr;

    fn witness(&self, offset: usize, col: usize) -> SymExpr {
        SymExpr::Wit { offset, col }
    }
    fn public(&self, idx: usize) -> SymExpr {
        SymExpr::Pub(idx)
    }
    fn preprocessed(&self, col: usize) -> SymExpr {
        SymExpr::Prep(col)
    }
    fn is_first_row(&self) -> SymExpr {
        SymExpr::First
    }
    fn is_last_row(&self) -> SymExpr {
        SymExpr::Last
    }
    fn is_transition(&self) -> SymExpr {
        SymExpr::Transition
    }
    fn assert_zero(&mut self, expr: SymExpr, _name: &'static str) {
        let (k, t) = true_degree_of(&expr);
        self.max_heavy = self.max_heavy.max(k);
        self.uses_transition |= t;
    }
}

/// Measure an AIR's true degree. Same walk discipline as `sym::measure`
/// (expression trees built and discarded — safe at seal-chip scale).
pub fn measure_true<A: Air<TrueDegreeBuilder>>(air: &A) -> TrueDegree {
    let mut b = TrueDegreeBuilder { max_heavy: 0, uses_transition: false };
    air.eval(&mut b);
    TrueDegree { max_heavy: b.max_heavy, uses_transition: b.uses_transition }
}

/// Upper bound on the quotient polynomial's coefficient count at trace
/// size N = 2^log_n: deg C ≤ k(N−1) + t, Q = C/Z_H has at most
/// k(N−1) + t − N + 1 coefficients (clamped to ≥ 1).
pub fn quotient_coeff_bound(td: &TrueDegree, log_n: usize) -> usize {
    let n = 1usize << log_n;
    (td.max_heavy * (n - 1) + usize::from(td.uses_transition) + 1 - n).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::chips::conservation::{
        gen_cons_prep, gen_cons_trace, ConservationChip, ACCP_BASE,
    };
    use crate::air::chips::digit::DigitChip;
    use crate::air::chips::merkle_poseidon2::MerkleChip;
    use crate::air::chips::range::{gen_range_witness, RangeChip};
    use crate::stark::domain::TwoAdicDomain;
    use crate::testutil::SplitMix64;

    struct CounterAir;

    impl<B: AirBuilder> Air<B> for CounterAir {
        fn eval(&self, b: &mut B) {
            let cur = b.witness(0, 0);
            let nxt = b.witness(1, 0);
            let one = B::constant(1);
            let step = b.is_transition();
            b.assert_zero(step * (nxt - cur - one), "counter_step");
            let first = b.is_first_row();
            let want = b.public(0);
            b.assert_zero(first * (cur - want), "counter_init");
        }
    }

    fn gf(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    fn extf(rng: &mut SplitMix64) -> ExtF {
        ExtF::new(gf(rng), gf(rng))
    }

    /// The 0/1 row-selector semantics at row i of an N-row trace — the
    /// NativeEval instantiation, valid for the H-run differential.
    fn bool_sels(i: usize, n: usize) -> SelectorVals<Goldilocks> {
        SelectorVals {
            first: Goldilocks::from_u32(u32::from(i == 0)),
            last: Goldilocks::from_u32(u32::from(i + 1 == n)),
            transition: Goldilocks::from_u32(u32::from(i + 1 < n)),
        }
    }

    fn all_folds<'a, A>(
        air: &A,
        rows: &'a [Vec<Goldilocks>],
        prep_rows: &'a [Vec<Goldilocks>],
        pubs: &'a [Goldilocks],
        alpha: ExtF,
    ) -> Vec<ExtF>
    where
        A: Air<PointCollector<'a, Goldilocks>>,
    {
        let n = rows.len();
        (0..n)
            .map(|i| {
                let cur = rows[i].as_slice();
                let nxt = rows[(i + 1) % n].as_slice();
                let prep: &[Goldilocks] =
                    if prep_rows.is_empty() { &[] } else { prep_rows[i].as_slice() };
                PointCollector::new(cur, nxt, prep, pubs, bool_sels(i, n), alpha)
                    .evaluate(air)
                    .0
            })
            .collect()
    }

    #[test]
    fn counter_folds_vanish_on_h_and_bind_tampering() {
        let mut rng = SplitMix64::new(0xC01);
        let alpha = extf(&mut rng);
        let n = 8usize;
        let honest: Vec<Goldilocks> =
            (0..n).map(|i| Goldilocks::from_u64_reduce(i as u64)).collect();
        let empty: [Vec<Goldilocks>; 0] = [];
        let pubs = [Goldilocks::ZERO];

        assert!(
            all_folds(&CounterAir, &honest, &empty, &pubs, alpha)
                .iter()
                .all(|f| f.is_zero())
        );
        // Honesty is α-independent.
        let alpha2 = extf(&mut rng);
        assert!(
            all_folds(&CounterAir, &honest, &empty, &pubs, alpha2)
                .iter()
                .all(|f| f.is_zero())
        );

        let mut bad = honest.clone();
        bad[2] = Goldilocks::from_u64_reduce(5);
        let folds = all_folds(&CounterAir, &bad, &empty, &pubs, alpha);
        assert!(!folds[1].is_zero(), "broken step constraint must bind");
        assert!(folds[0].is_zero());

        let folds2 = all_folds(&CounterAir, &honest, &empty, &[Goldilocks::ONE], alpha);
        assert!(!folds2[0].is_zero(), "broken boundary constraint must bind");
    }

    #[test]
    fn folds_match_native_eval_verdicts() {
        let mut rng = SplitMix64::new(0xD1F);
        let alpha = extf(&mut rng);

        // RangeChip (no preprocessed columns).
        let rows: Vec<Vec<Goldilocks>> = (0..12)
            .map(|_| gen_range_witness(rng.next_u64() % (1 << 20), 32))
            .collect();
        let empty: [Vec<Goldilocks>; 0] = [];
        let pubs: [Goldilocks; 0] = [];
        let chip = RangeChip::new(0, 1, 32);
        assert!(all_folds(&chip, &rows, &empty, &pubs, alpha).iter().all(|f| f.is_zero()));
        assert!(NativeEval::check(rows.clone(), vec![], &chip, 16).is_ok());

        let mut bad = rows.clone();
        bad[3][0] = bad[3][0] + Goldilocks::ONE;
        assert!(all_folds(&chip, &bad, &empty, &pubs, alpha).iter().any(|f| !f.is_zero()));
        assert!(NativeEval::check(bad, vec![], &chip, 16).is_err());

        // ConservationChip (preprocessed-gated).
        let entries = vec![(100u64, true), (60, false), (40, false)];
        let cons_rows = gen_cons_trace(&entries);
        let cons_prep = gen_cons_prep(1, 2, 0);
        let cons = ConservationChip::new();
        assert!(
            all_folds(&cons, &cons_rows, &cons_prep, &pubs, alpha)
                .iter()
                .all(|f| f.is_zero())
        );
        assert!(
            NativeEval::check_with_prep(cons_rows.clone(), cons_prep.clone(), vec![], &cons, 16)
                .is_ok()
        );

        let mut bad_cons = cons_rows;
        bad_cons[1][ACCP_BASE] = bad_cons[1][ACCP_BASE] + Goldilocks::ONE;
        assert!(all_folds(&cons, &bad_cons, &cons_prep, &pubs, alpha).iter().any(|f| !f.is_zero()));
        assert!(
            NativeEval::check_with_prep(bad_cons, cons_prep, vec![], &cons, 16).is_err()
        );
    }

    #[test]
    fn extf_point_evaluation_runs_the_same_air() {
        // The ζ-point shape the verifier runs: ExtF leaves throughout.
        let cur = [ExtF::from_base(Goldilocks::ONE)];
        let nxt = [ExtF::from_base(Goldilocks::from_u32(2))];
        let prep: [ExtF; 0] = [];
        let pubs = [ExtF::ZERO];
        let sels = SelectorVals {
            first: ExtF::ZERO,
            last: ExtF::ZERO,
            transition: ExtF::ONE,
        };
        let alpha = extf(&mut SplitMix64::new(0xE7));

        let (fold, count) =
            PointCollector::new(&cur, &nxt, &prep, &pubs, sels, alpha).evaluate(&CounterAir);
        assert_eq!(count, 2);
        assert!(fold.is_zero());

        let bad = [ExtF::from_base(Goldilocks::from_u32(5))];
        let (fold2, _) =
            PointCollector::new(&bad, &nxt, &prep, &pubs, sels, alpha).evaluate(&CounterAir);
        let want = ExtF::from_base(
            Goldilocks::from_u32(2) - Goldilocks::from_u32(5) - Goldilocks::ONE,
        );
        assert_eq!(fold2, want);
    }

    #[test]
    fn assertion_count_is_deterministic() {
        let row = gen_range_witness(7, 32);
        let prep: [Goldilocks; 0] = [];
        let pubs: [Goldilocks; 0] = [];
        let c = PointCollector::new(
            &row,
            &row,
            &prep,
            &pubs,
            bool_sels(0, 1),
            ExtF::ONE,
        );
        assert_eq!(c.evaluate(&RangeChip::new(0, 1, 32)).1, 33);
    }

    #[test]
    fn selector_values_at_out_of_domain_points() {
        for log_n in 1..=6usize {
            let h = two_adic_generator(log_n);
            let h_last = h.inverse();
            let n = 1usize << log_n;

            // Zero-set structure on H (poles skipped).
            for i in 1..n {
                let s = selector_vals_ext(log_n, ExtF::from_base(h.pow(i as u64)));
                assert!(s.first.is_zero(), "log={log_n} i={i}");
                if i + 1 != n {
                    assert!(s.last.is_zero());
                }
                if i + 1 == n {
                    assert!(s.transition.is_zero());
                } else {
                    assert!(!s.transition.is_zero());
                }
            }

            // The defining identities at coset points, non-circularly:
            // L₁·N·(x−1) = Z_H, L_N·N·(x−h^{N−1}) = Z_H, T = x − h^{N−1}.
            let dom = TwoAdicDomain::standard_coset(log_n + 1);
            let n_f = ExtF::from_base(Goldilocks::from_u64_reduce(n as u64));
            let h_last_f = ExtF::from_base(h_last);
            for j in 0..dom.size() {
                let x = ExtF::from_base(dom.point(j));
                let s = selector_vals_ext(log_n, x);
                let zh = x.pow(n as u64) - ExtF::ONE;
                assert_eq!(s.first * n_f * (x - ExtF::ONE), zh, "log={log_n} j={j}");
                assert_eq!(s.last * n_f * (x - h_last_f), zh, "log={log_n} j={j}");
                assert_eq!(s.transition, x - h_last_f);
            }
        }
    }

    struct DegreeAir;

    impl<B: AirBuilder> Air<B> for DegreeAir {
        fn eval(&self, b: &mut B) {
            let p = b.preprocessed(0);
            let w0 = b.witness(0, 0);
            let w1 = b.witness(1, 1);
            b.assert_zero(p * w0 * w1, "k3");
            b.assert_zero(b.is_first_row() * w0, "k2");
            b.assert_zero(b.is_transition() * w0, "k1t");
            b.assert_zero(w0 + b.public(0), "k1");
            b.assert_zero(b.is_last_row() * p * w0 * w0, "k4");
        }
    }

    #[test]
    fn true_degree_measured() {
        assert_eq!(
            measure_true(&DegreeAir),
            TrueDegree { max_heavy: 4, uses_transition: true }
        );
        // N = 8: 4·7 + 1 − 8 + 1 = 22.
        assert_eq!(quotient_coeff_bound(&TrueDegree { max_heavy: 4, uses_transition: true }, 3), 22);
        assert_eq!(quotient_coeff_bound(&TrueDegree { max_heavy: 5, uses_transition: false }, 4), 17);
        assert_eq!(quotient_coeff_bound(&TrueDegree { max_heavy: 0, uses_transition: false }, 5), 1);
    }

    #[test]
    fn chip_true_degree_pins() {
        assert_eq!(
            measure_true(&RangeChip::new(0, 1, 32)),
            TrueDegree { max_heavy: 2, uses_transition: false }
        );
        assert_eq!(
            measure_true(&DigitChip),
            TrueDegree { max_heavy: 2, uses_transition: false }
        );
        assert_eq!(
            measure_true(&ConservationChip::new()),
            TrueDegree { max_heavy: 4, uses_transition: false }
        );
        assert_eq!(
            measure_true(&MerkleChip::new(4)),
            TrueDegree { max_heavy: 5, uses_transition: false }
        );
    }
}
