//! The backend-neutral AIR builder (DSR-7's "one eval, two builders").
//!
//! Every chip and AIR in nerv-proofs is written ONCE against `AirBuilder`
//! and runs against two implementations:
//! * `NativeEval` — evaluates constraints over a concrete witness trace,
//!   reporting failures: the differential oracle, the conformance-vector
//!   checker, and the correctness half of DSR-7;
//! * the plonky3 adapter (chunk 11) — records the same constraints
//!   symbolically for the prover.
//!
//! The trait is deliberately minimal: witness access at (row offset,
//! column), public inputs, PREPROCESSED columns (row-varying constants —
//! plonky3's preprocessed trace; the native evaluator's per-row table),
//! row-position selectors, zero-assertions, and constants. Everything
//! else — booleans, ranges, decompositions — is a chip built from these
//! primitives (WP §5.3's sub-AIR pattern).
//!
//! Frame model: a frame of 2 rows (current, next). The evaluator slides
//! the frame over the trace; `witness(0, col)` is the current row's value
//! and `witness(1, col)` the next row's (zero past the boundary —
//! unguarded transition constraints fail there, which is correct).
//! Constraints are multiplied by position selectors or preprocessed
//! indicators to scope them.

use std::fmt;
use std::ops::{Add, Mul, Sub};
use nerv_core::field::Goldilocks;

// ---------------------------------------------------------------------------
// Expression
// ---------------------------------------------------------------------------

/// A field-element expression. Native: `Goldilocks`. Symbolic (chunk 11):
/// an expression tree. The ops are field ops — degree accounting is the
/// prover's concern, not the builder's.
pub trait AirExpr:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + From<Goldilocks>
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl AirExpr for Goldilocks {
    fn zero() -> Self {
        Goldilocks::ZERO
    }
    fn one() -> Self {
        Goldilocks::ONE
    }
}

// ---------------------------------------------------------------------------
// The builder
// ---------------------------------------------------------------------------

/// The interface every chip and AIR is written against.
pub trait AirBuilder {
    type Expr: AirExpr;

    /// Witness value at (current position + offset, column).
    /// Offset 0 = current row, 1 = next row. Past the trace boundary:
    /// zero (unguarded constraints fail there — by design).
    fn witness(&self, offset: usize, col: usize) -> Self::Expr;

    /// Public input at index (zero past the end).
    fn public(&self, idx: usize) -> Self::Expr;

    /// Preprocessed ("fixed") column value at the current row — the
    /// row-varying constants: round tables, phase indicators. Backend
    /// mapping: plonky3 preprocessed trace; native a per-row table.
    /// Zero if the table has no such row/column.
    fn preprocessed(&self, col: usize) -> Self::Expr;

    /// 1 at row 0, 0 elsewhere.
    fn is_first_row(&self) -> Self::Expr;

    /// 1 at the last row, 0 elsewhere.
    fn is_last_row(&self) -> Self::Expr;

    /// 1 at every row except the last, 0 at the last.
    fn is_transition(&self) -> Self::Expr;

    /// Assert `expr = 0` at the current evaluation position.
    fn assert_zero(&mut self, expr: Self::Expr, name: &'static str);

    // -- provided -----------------------------------------------------------

    fn constant(v: u64) -> Self::Expr {
        <Self::Expr as From<Goldilocks>>::from(Goldilocks::from_u64_reduce(v))
    }

    fn assert_eq(&mut self, lhs: Self::Expr, rhs: Self::Expr, name: &'static str) {
        self.assert_zero(lhs - rhs, name);
    }

    fn assert_bool(&mut self, x: Self::Expr, name: &'static str) {
        let one = Self::constant(1);
        self.assert_zero(x.clone() * (x - one), name);
    }
}

// ---------------------------------------------------------------------------
// The Air trait
// ---------------------------------------------------------------------------

/// An AIR: constraints over a witness trace, generated once and evaluated
/// by any builder.
pub trait Air<B: AirBuilder> {
    fn eval(&self, builder: &mut B);
}

// ---------------------------------------------------------------------------
// The native evaluator
// ---------------------------------------------------------------------------

/// One constraint failure: where, what, and the offending value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintFailure {
    pub row: usize,
    pub name: &'static str,
    pub value: u64,
}

impl fmt::Display for ConstraintFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row {}: {} (value {})", self.row, self.name, self.value)
    }
}

/// The native evaluator: runs `eval` at each trace position, checking
/// assertions immediately. The differential oracle.
pub struct NativeEval {
    trace: Vec<Vec<Goldilocks>>,
    public_values: Vec<Goldilocks>,
    prep: Vec<Vec<Goldilocks>>,
    position: usize,
    failures: Vec<ConstraintFailure>,
}

impl AirBuilder for NativeEval {
    type Expr = Goldilocks;

    fn witness(&self, offset: usize, col: usize) -> Goldilocks {
        let row = self.position + offset;
        if row < self.trace.len() && col < self.trace[row].len() {
            self.trace[row][col]
        } else {
            Goldilocks::ZERO
        }
    }

    fn public(&self, idx: usize) -> Goldilocks {
        self.public_values.get(idx).copied().unwrap_or(Goldilocks::ZERO)
    }

    fn preprocessed(&self, col: usize) -> Goldilocks {
        self.prep
            .get(self.position)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(Goldilocks::ZERO)
    }

    fn is_first_row(&self) -> Goldilocks {
        if self.position == 0 {
            Goldilocks::ONE
        } else {
            Goldilocks::ZERO
        }
    }

    fn is_last_row(&self) -> Goldilocks {
        if self.position + 1 >= self.trace.len() {
            Goldilocks::ONE
        } else {
            Goldilocks::ZERO
        }
    }

    fn is_transition(&self) -> Goldilocks {
        if self.position + 1 < self.trace.len() {
            Goldilocks::ONE
        } else {
            Goldilocks::ZERO
        }
    }

    fn assert_zero(&mut self, expr: Goldilocks, name: &'static str) {
        if !expr.is_zero() {
            self.failures.push(ConstraintFailure {
                row: self.position,
                name,
                value: expr.as_u64(),
            });
        }
    }
}

impl NativeEval {
    /// Checks all constraints of `air` over the given trace (no
    /// preprocessed table). Delegates to [`NativeEval::check_with_prep`].
    pub fn check<A: Air<NativeEval>>(
        trace: Vec<Vec<Goldilocks>>,
        public_values: Vec<Goldilocks>,
        air: &A,
        max_failures: usize,
    ) -> Result<(), Vec<ConstraintFailure>> {
        NativeEval::check_with_prep(trace, Vec::new(), public_values, air, max_failures)
    }

    /// The full check: trace, preprocessed table (one row per trace row),
    /// public inputs. Returns the first `max_failures` failures.
    pub fn check_with_prep<A: Air<NativeEval>>(
        trace: Vec<Vec<Goldilocks>>,
        prep: Vec<Vec<Goldilocks>>,
        public_values: Vec<Goldilocks>,
        air: &A,
        max_failures: usize,
    ) -> Result<(), Vec<ConstraintFailure>> {
        debug_assert!(prep.is_empty() || prep.len() == trace.len());
        let mut eval = NativeEval {
            trace,
            public_values,
            prep,
            position: 0,
            failures: Vec::new(),
        };
        for pos in 0..eval.trace.len() {
            if eval.failures.len() >= max_failures {
                break;
            }
            eval.position = pos;
            air.eval(&mut eval);
        }
        if eval.failures.is_empty() {
            Ok(())
        } else {
            Err(eval.failures)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// A simple AIR: column 1 = 2 × column 0 at every row.
    struct DoubleAir;

    impl<B: AirBuilder> Air<B> for DoubleAir {
        fn eval(&self, builder: &mut B) {
            let a = builder.witness(0, 0);
            let b = builder.witness(0, 1);
            let two = B::constant(2);
            builder.assert_eq(b, a * two, "double");
        }
    }

    /// A transition AIR: next row's column 0 = current + 1 (guarded).
    struct CounterAir;

    impl<B: AirBuilder> Air<B> for CounterAir {
        fn eval(&self, builder: &mut B) {
            let a = builder.witness(0, 0);
            let a_next = builder.witness(1, 0);
            let one = B::constant(1);
            let sel = builder.is_transition();
            builder.assert_zero(sel * (a_next - a - one), "counter_step");
        }
    }

    /// A boundary AIR: column 0 = 42 at the first row only.
    struct BoundaryAir;

    impl<B: AirBuilder> Air<B> for BoundaryAir {
        fn eval(&self, builder: &mut B) {
            let a = builder.witness(0, 0);
            let sel = builder.is_first_row();
            let want = B::constant(42);
            builder.assert_zero(sel * (a - want), "boundary");
        }
    }

    /// A preprocessed AIR: column 0 equals the per-row preprocessed value.
    struct PrepAir;

    impl<B: AirBuilder> Air<B> for PrepAir {
        fn eval(&self, builder: &mut B) {
            let x = builder.witness(0, 0);
            let k = builder.preprocessed(0);
            builder.assert_eq(x, k, "match_prep");
        }
    }

    fn row(vals: &[u64]) -> Vec<Goldilocks> {
        vals.iter().map(|&v| Goldilocks::from_u64_reduce(v)).collect()
    }

    #[test]
    fn double_air_accepts_and_rejects() {
        let good = vec![row(&[1, 2]), row(&[5, 10]), row(&[100, 200])];
        assert!(NativeEval::check(good, vec![], &DoubleAir, 16).is_ok());

        let bad = vec![row(&[1, 3])];
        let errs = NativeEval::check(bad, vec![], &DoubleAir, 16).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].name, "double");
        assert_eq!(errs[0].row, 0);
    }

    #[test]
    fn counter_air_transition_semantics() {
        let good = vec![row(&[0]), row(&[1]), row(&[2]), row(&[3])];
        assert!(NativeEval::check(good, vec![], &CounterAir, 16).is_ok());

        let bad = vec![row(&[0]), row(&[1]), row(&[5])];
        let errs = NativeEval::check(bad, vec![], &CounterAir, 16).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].row, 1);
        assert_eq!(errs[0].name, "counter_step");

        let single = vec![row(&[0])];
        assert!(NativeEval::check(single, vec![], &CounterAir, 16).is_ok());
    }

    #[test]
    fn boundary_air_first_row_only() {
        let good = vec![row(&[42]), row(&[0]), row(&[7])];
        assert!(NativeEval::check(good, vec![], &BoundaryAir, 16).is_ok());

        let bad = vec![row(&[41]), row(&[0])];
        let errs = NativeEval::check(bad, vec![], &BoundaryAir, 16).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].row, 0);
    }

    #[test]
    fn selectors_are_correct() {
        let mut eval = NativeEval {
            trace: vec![row(&[1]), row(&[2]), row(&[3])],
            public_values: vec![],
            prep: vec![],
            position: 0,
            failures: vec![],
        };
        assert_eq!(eval.is_first_row(), Goldilocks::ONE);
        assert_eq!(eval.is_last_row(), Goldilocks::ZERO);
        assert_eq!(eval.is_transition(), Goldilocks::ONE);
        eval.position = 1;
        assert_eq!(eval.is_first_row(), Goldilocks::ZERO);
        assert_eq!(eval.is_last_row(), Goldilocks::ZERO);
        assert_eq!(eval.is_transition(), Goldilocks::ONE);
        eval.position = 2;
        assert_eq!(eval.is_first_row(), Goldilocks::ZERO);
        assert_eq!(eval.is_last_row(), Goldilocks::ONE);
        assert_eq!(eval.is_transition(), Goldilocks::ZERO);
        assert_eq!(eval.witness(1, 0), Goldilocks::ZERO);
        assert_eq!(eval.witness(0, 0), Goldilocks::from_u64_reduce(3));
    }

    #[test]
    fn public_values_and_out_of_bounds() {
        let mut eval = NativeEval {
            trace: vec![row(&[0])],
            public_values: vec![Goldilocks::from_u32(9)],
            prep: vec![],
            position: 0,
            failures: vec![],
        };
        assert_eq!(eval.public(0), Goldilocks::from_u32(9));
        assert_eq!(eval.public(1), Goldilocks::ZERO);
        assert_eq!(eval.witness(0, 99), Goldilocks::ZERO);
    }

    #[test]
    fn preprocessed_is_per_row() {
        let prep = vec![vec![Goldilocks::from_u32(7)], vec![Goldilocks::from_u32(8)]];
        let mut eval = NativeEval {
            trace: vec![row(&[0]), row(&[0])],
            public_values: vec![],
            prep,
            position: 0,
            failures: vec![],
        };
        assert_eq!(eval.preprocessed(0), Goldilocks::from_u32(7));
        eval.position = 1;
        assert_eq!(eval.preprocessed(0), Goldilocks::from_u32(8));
        assert_eq!(eval.preprocessed(5), Goldilocks::ZERO);
        let mut empty = NativeEval {
            trace: vec![row(&[0])],
            public_values: vec![],
            prep: vec![],
            position: 0,
            failures: vec![],
        };
        assert_eq!(empty.preprocessed(0), Goldilocks::ZERO);
    }

    #[test]
    fn check_with_prep_end_to_end() {
        let trace = vec![row(&[7]), row(&[8])];
        let prep = vec![vec![Goldilocks::from_u32(7)], vec![Goldilocks::from_u32(8)]];
        assert!(NativeEval::check_with_prep(trace, prep, vec![], &PrepAir, 4).is_ok());

        let bad = vec![row(&[7]), row(&[9])];
        let errs = NativeEval::check_with_prep(bad, prep, vec![], &PrepAir, 4).unwrap_err();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0].row, 1);
        assert_eq!(errs[0].name, "match_prep");
    }

    #[test]
    fn max_failures_limits_output() {
        let bad = vec![row(&[1, 1]), row(&[2, 2]), row(&[3, 3])];
        let errs = NativeEval::check(bad, vec![], &DoubleAir, 2).unwrap_err();
        assert_eq!(errs.len(), 2);
    }
}

