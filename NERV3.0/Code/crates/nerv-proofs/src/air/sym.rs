//! The symbolic backend: `AirBuilder` over expression trees. Recording an
//! AIR symbolically turns every structural claim into a measured fact —
//! degree and constraint-family counts (D.2's gate, per errata 68/77) —
//! and yields an interpreter whose failure sets must match `NativeEval`
//! exactly (the "one eval, two builders" differential, DSR-7). The
//! plonky3 adapter consumes `ConstraintSet` as a fully-enumerated input,
//! reducing the backend binding to mechanical translation.

use std::ops::{Add, Mul, Sub};
use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder, AirExpr, ConstraintFailure};

#[derive(Clone, PartialEq, Debug)]
pub enum SymExpr {
    Const(Goldilocks),
    Wit { offset: usize, col: usize },
    Pub(usize),
    Prep(usize),
    First,
    Last,
    Transition,
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
}

impl SymExpr {
    /// Witness degree 1, constants/preprocessed/publics 0.
    pub fn degree(&self) -> u32 {
        match self {
            SymExpr::Const(_) | SymExpr::Pub(_) | SymExpr::Prep(_)
            | SymExpr::First | SymExpr::Last | SymExpr::Transition => 0,
            SymExpr::Wit { .. } => 1,
            SymExpr::Add(a, b) | SymExpr::Sub(a, b) => a.degree().max(b.degree()),
            SymExpr::Mul(a, b) => a.degree() + b.degree(),
        }
    }
}

impl From<Goldilocks> for SymExpr {
    fn from(c: Goldilocks) -> SymExpr {
        SymExpr::Const(c)
    }
}

impl AirExpr for SymExpr {
    fn zero() -> Self {
        SymExpr::Const(Goldilocks::ZERO)
    }
    fn one() -> Self {
        SymExpr::Const(Goldilocks::ONE)
    }
}

impl Add for SymExpr {
    type Output = SymExpr;
    fn add(self, rhs: SymExpr) -> SymExpr {
        SymExpr::Add(Box::new(self), Box::new(rhs))
    }
}
impl Sub for SymExpr {
    type Output = SymExpr;
    fn sub(self, rhs: SymExpr) -> SymExpr {
        SymExpr::Sub(Box::new(self), Box::new(rhs))
    }
}
impl Mul for SymExpr {
    type Output = SymExpr;
    fn mul(self, rhs: SymExpr) -> SymExpr {
        SymExpr::Mul(Box::new(self), Box::new(rhs))
    }
}

/// The recording builder.
#[derive(Default)]
pub struct SymBuilder {
    constraints: Vec<(SymExpr, &'static str)>,
}

impl SymBuilder {
    pub fn new() -> SymBuilder {
        SymBuilder::default()
    }
}

impl AirBuilder for SymBuilder {
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

    fn assert_zero(&mut self, expr: SymExpr, name: &'static str) {
        self.constraints.push((expr, name));
    }
}

/// A recorded AIR: the constraint families, ready for degree/count
/// reporting, interpretation, and backend translation.
pub struct ConstraintSet {
    constraints: Vec<(SymExpr, &'static str)>,
}

impl ConstraintSet {
    pub fn record<A: Air<SymBuilder>>(air: &A) -> ConstraintSet {
        let mut b = SymBuilder::new();
        air.eval(&mut b);
        ConstraintSet { constraints: b.constraints }
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn constraints(&self) -> &[(SymExpr, &'static str)] {
        &self.constraints
    }

    pub fn max_degree(&self) -> u32 {
        self.constraints.iter().map(|(e, _)| e.degree()).max().unwrap_or(0)
    }

    /// D.2's gate input: (family count, max degree).
    pub fn report(&self) -> (usize, u32) {
        (self.len(), self.max_degree())
    }

    pub fn evaluate(
        &self,
        e: &SymExpr,
        row: usize,
        trace: &[Vec<Goldilocks>],
        prep: &[Vec<Goldilocks>],
        publics: &[Goldilocks],
    ) -> Goldilocks {
        match e {
            SymExpr::Const(c) => *c,
            SymExpr::Wit { offset, col } => trace
                .get(row + offset)
                .and_then(|t| t.get(*col))
                .copied()
                .unwrap_or(Goldilocks::ZERO),
            SymExpr::Pub(i) => publics.get(*i).copied().unwrap_or(Goldilocks::ZERO),
            SymExpr::Prep(c) => prep
                .get(row)
                .and_then(|p| p.get(*c))
                .copied()
                .unwrap_or(Goldilocks::ZERO),
            SymExpr::First => {
                if row == 0 { Goldilocks::ONE } else { Goldilocks::ZERO }
            }
            SymExpr::Last => {
                if row + 1 >= trace.len() { Goldilocks::ONE } else { Goldilocks::ZERO }
            }
            SymExpr::Transition => {
                if row + 1 < trace.len() { Goldilocks::ONE } else { Goldilocks::ZERO }
            }
            SymExpr::Add(a, b) => {
                self.evaluate(a, row, trace, prep, publics) + self.evaluate(b, row, trace, prep, publics)
            }
            SymExpr::Sub(a, b) => {
                self.evaluate(a, row, trace, prep, publics) - self.evaluate(b, row, trace, prep, publics)
            }
            SymExpr::Mul(a, b) => {
                self.evaluate(a, row, trace, prep, publics) * self.evaluate(b, row, trace, prep, publics)
            }
        }
    }
}


/// Counting/degree-measuring builder: builds each expression tree,
/// measures, and discards it — O(1) retained memory. For large AIRs
/// (the seal chip: ~10⁶ families) where `ConstraintSet::record`'s
/// tree retention is infeasible (erratum 83).
#[derive(Default)]
pub struct MeasureBuilder {
    count: usize,
    max_degree: u32,
}

impl MeasureBuilder {
    pub fn new() -> MeasureBuilder {
        MeasureBuilder::default()
    }

    pub fn report(&self) -> (usize, u32) {
        (self.count, self.max_degree)
    }
}

impl AirBuilder for MeasureBuilder {
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
        self.count += 1;
        let d = expr.degree();
        if d > self.max_degree {
            self.max_degree = d;
        }
    }
}

/// Measures (constraint-family count, max witness degree) without
/// retaining expression trees.
pub fn measure<A: Air<MeasureBuilder>>(air: &A) -> (usize, u32) {
    let mut b = MeasureBuilder::new();
    air.eval(&mut b);
    b.report()
}


impl ConstraintSet {
    /// Mirrors `NativeEval::check_with_prep` semantics exactly: full
    /// per-row scans, failure cap checked between rows.
    pub fn check(
        &self,
        trace: &[Vec<Goldilocks>],
        prep: &[Vec<Goldilocks>],
        publics: &[Goldilocks],
        max_failures: usize,
    ) -> Result<(), Vec<ConstraintFailure>> {
        let mut failures = Vec::new();
        'rows: for row in 0..trace.len() {
            if failures.len() >= max_failures {
                break 'rows;
            }
            for (e, name) in &self.constraints {
                let v = self.evaluate(e, row, trace, prep, publics);
                if !v.is_zero() {
                    failures.push(ConstraintFailure { row, name: *name, value: v.as_u64() });
                }
            }
        }
        if failures.is_empty() { Ok(()) } else { Err(failures) }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::chips::blake3::{gen_prep, gen_trace, Blake3Compression};
    use crate::air::chips::conservation::{gen_cons_prep, gen_cons_trace, ConservationChip};
    use crate::air::chips::digit::{gen_digit_witness, DigitChip};
    use crate::air::chips::merkle_poseidon2::{gen_merkle_trace, MerkleChip};
    use crate::air::chips::range::{gen_range_witness, RangeChip};
    use crate::testutil::SplitMix64;

    fn differential<A>(air: &A, trace: Vec<Vec<Goldilocks>>, prep: Vec<Vec<Goldilocks>>, publics: Vec<Goldilocks>)
    where
        A: Air<NativeEval> + Air<SymBuilder>,
    {
        let native = NativeEval::check_with_prep(trace.clone(), prep.clone(), publics.clone(), air, usize::MAX);
        let set = ConstraintSet::record(air);
        let sym = set.check(&trace, &prep, &publics, usize::MAX);
        assert_eq!(native.is_ok(), sym.is_ok());
        if let (Err(a), Err(b)) = (native, sym) {
            assert_eq!(a, b, "native and symbolic failure sets must match exactly");
        }
    }

    #[test]
    fn degrees_and_counts_measured() {
        let range = ConstraintSet::record(&RangeChip::new(0, 1, 60));
        assert_eq!(range.max_degree(), 2);
        assert_eq!(range.len(), 61);

        let digit = ConstraintSet::record(&DigitChip);
        assert_eq!(digit.max_degree(), 2);
        assert_eq!(digit.len(), 74);

        let cons = ConstraintSet::record(&ConservationChip::new());
        assert_eq!(cons.max_degree(), 3);
        assert!(cons.len() > 200 && cons.len() < 400, "conservation families: {}", cons.len());

        let merkle = ConstraintSet::record(&MerkleChip::new(32));
        assert_eq!(merkle.max_degree(), 3);
        assert!(merkle.len() > 80 && merkle.len() < 200, "merkle families: {}", merkle.len());

        let bl3 = ConstraintSet::record(&Blake3Compression::new());
        assert_eq!(bl3.max_degree(), 3);
        assert!(bl3.len() > 500 && bl3.len() < 2_000, "blake3 families: {}", bl3.len());

        let custody = ConstraintSet::record(&crate::air::custody_air::CustodyAir::new(2, 3, vec![2], 1, 32));
        assert_eq!(custody.max_degree(), 3);
        assert!(custody.len() > 1_000 && custody.len() < 50_000, "custody families: {}", custody.len());
    }

    #[test]
    fn differential_range_and_digit() {
        let chip = RangeChip::new(0, 1, 32);
        let good: Vec<_> = [0u64, 1, 255, 65535, 12345678].map(|v| gen_range_witness(v, 32)).to_vec();
        differential(&chip, good, vec![], vec![]);
        let mut bad = gen_range_witness(300, 32);
        bad[0] = nerv_core::field::Goldilocks::from_u64_reduce(300);
        differential(&chip, vec![bad], vec![], vec![]);

        differential(&DigitChip, (0..8).map(|_| gen_digit_witness(0xDEAD)).collect(), vec![], vec![]);
        let mut bad = gen_digit_witness(7);
        bad[2] = bad[2] + nerv_core::field::Goldilocks::ONE;
        differential(&DigitChip, vec![bad], vec![], vec![]);
    }

    #[test]
    fn differential_conservation() {
        let chip = ConservationChip::new();
        let entries = vec![(100u64, true), (60u64, false), (40u64, false)];
        let trace = gen_cons_trace(&entries);
        let prep = gen_cons_prep(1, 2, 0);
        differential(&chip, trace, prep, vec![]);
        let entries = vec![(100u64, true), (61u64, false), (40u64, false)];
        differential(&chip, gen_cons_trace(&entries), gen_cons_prep(1, 2, 0), vec![]);
    }

    #[test]
    fn differential_merkle_and_blake3() {
        let mut rng = SplitMix64::new(0x5A1);
        let cm = nerv_core::hash::Hash256::from_bytes(rng.bytes32());
        let sibs = vec![[nerv_core::field::Goldilocks::from_u32(11); 4]];
        let chip = MerkleChip::new(1);
        let trace = gen_merkle_trace(1, &cm, 0b01, &sibs);
        let prep = chip.gen_prep();
        let root: Vec<nerv_core::field::Goldilocks> =
            (0..4).map(|i| trace[chip.rows() - 1][i]).collect();
        differential(&chip, trace.clone(), prep.clone(), root.clone());
        let mut bad = trace;
        bad[10][3] = bad[10][3] + nerv_core::field::Goldilocks::ONE;
        differential(&chip, bad, prep, root);

        let cv = [0x11223344u32; 8];
        let block = core::array::from_fn(|_| (rng.next_u64() >> 32) as u32);
        let bl3 = Blake3Compression::new();
        let trace = gen_trace(&cv, &block, 64, 0, 1);
        let prep = gen_prep(64, 0, 1);
        differential(&bl3, trace.clone(), prep.clone(), vec![]);
        let mut bad = trace;
        bad[20][17] = bad[20][17] + nerv_core::field::Goldilocks::ONE;
        differential(&bl3, bad, prep, vec![]);
    }

    #[test]
    fn differential_custody_v3() {
        use crate::air::custody_air::{
            gen_custody_trace, BurnWitness, CustodyAir, CustodyWitness, InputWitness,
            OutputWitness, RevertWitness,
        };
        use nerv_custody::burn::BurnCommitment;
        use nerv_custody::commitment::NoteOpening;
        use nerv_custody::nct::NoteCommitmentTree;
        use nerv_custody::nullifier::derive_nullifier;
        use nerv_custody::tx::{InputSet, LegShell, Output};
        use nerv_custody::{Address, MasterSeed, WalletKeys};
        use nerv_core::types::{FeeSats, Height, LegIndex, ShardSet};

        let mut rng = SplitMix64::new(0x5A2);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| Address::generate(det, wk.nullifier_key(), i, &g).unwrap();
        let opening = |v: u64, i: u64| NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let input = opening(1000, 0);
        let out = opening(500, 1);
        let revert = opening(500, 1);
        let mut tree = NoteCommitmentTree::new();
        let idx = tree.append(&input.commitment().unwrap()).unwrap();
        let root = tree.root();
        let sibs: Vec<[nerv_core::field::Goldilocks; 4]> = tree
            .witness(idx)
            .unwrap()
            .siblings
            .iter()
            .map(|d| d.to_elements().unwrap())
            .collect();
        let nf = derive_nullifier(&wk.nullifier_key_at(0), &input.rho);
        let leg7 = LegShell {
            shard: g.ids()[7],
            inputs: InputSet::new(vec![nf]),
            outputs: vec![],
            fee: FeeSats::from_u64(100),
            anchor: root.as_hash256(),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
        };
        let leg40 = LegShell {
            shard: g.ids()[40],
            inputs: InputSet::new(vec![]),
            outputs: vec![Output {
                cm: out.commitment().unwrap(),
                sealed_note: vec![0xA5; 48],
                value: 500,
                conditional: true,
                revert_cm: Some(revert.commitment().unwrap()),
            }],
            fee: FeeSats::from_u64(0),
            anchor: nerv_core::hash::Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
        };
        let shell = nerv_custody::tx::TransactionShell { legs: vec![leg7, leg40] };
        let txid = shell.txid().unwrap();
        let burn_cm = *BurnCommitment::new(&txid, LegIndex::from_u8(0), 400).unwrap().as_hash();
        let witness = CustodyWitness {
            inputs: vec![InputWitness {
                opening: input,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: idx,
                siblings: sibs,
                anchor: root,
            }],
            outputs: vec![OutputWitness { opening: out }],
            reverts: vec![RevertWitness { opening: revert }],
            burns: vec![BurnWitness { leg: 0, value: 400, commitment: burn_cm }],
            fees: vec![100, 0],
        };
        let air = CustodyAir::new(1, 1, vec![0], 1, nerv_custody::nct::DEPTH);
        let ct = gen_custody_trace(&witness, &shell, nerv_custody::nct::DEPTH).unwrap();
        differential(&air, ct.trace.clone(), ct.prep.clone(), ct.publics.clone());
        let mut bad = ct.trace;
        bad[500][air.rev_reg(0)] = bad[500][air.rev_reg(0)] + nerv_core::field::Goldilocks::ONE;
        differential(&air, bad, ct.prep, ct.publics);
    }
}

