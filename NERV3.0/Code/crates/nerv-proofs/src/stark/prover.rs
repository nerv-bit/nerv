//! The prove driver (WP §5.4–§5.5): pads the trace and preprocessed table
//! to a common power-of-two height, runs the D.2 security gate at the
//! actual trace size, builds the composition plan, and delegates to
//! `compose_prove`. In debug builds the padded trace is additionally
//! checked row-by-row against the AIR (`NativeEval`) — the free
//! differential; in release a pad-incompatible or dishonest input yields
//! a proof that fails verification. Total either way.
//!
//! PADDING CONTRACT: the AIR must be pad-compatible — every constraint
//! holds on the zero-padded extension (all chips are; their zero-fill
//! tests pin it). Row-chained AIRs must be generated at power-of-two
//! heights instead; register entry 61 records the obligation.

use crate::air::builder::{Air, ConstraintFailure, NativeEval};
use crate::air::fs::FsTranscript;
use crate::air::sym::{measure, MeasureBuilder};
use crate::security::{self, AirShape, FriShape, Profile, SecurityGateError};
use crate::stark::collector::{measure_true, PointCollector, TrueDegreeBuilder};
use crate::stark::compose::{compose_prove, ComposeError, Plan};
use crate::stark::ext_field::ExtF;
use nerv_core::field::Goldilocks;

/// A proof plus its statement: the trace height class, the width, and the
/// padded preprocessed table the proof was composed under. The statement
/// fields are transcript-bound (shape absorption), so an edited `Proved`
/// fails verification; the table is the authentic one — a verifier
/// holding a different table rejects (its ζ-check evaluates the AIR
/// against its own Lagrange values).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proved {
    pub proof: crate::stark::compose::ComposedProof,
    pub log_n: usize,
    pub width: usize,
    pub prep: Vec<Vec<Goldilocks>>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProveError {
    #[error(transparent)]
    Compose(#[from] ComposeError),
    #[error(transparent)]
    Security(#[from] SecurityGateError),
    #[error("trace is empty or has zero columns")]
    EmptyTrace,
    #[error("constraints fail on the padded trace (debug check): {first}")]
    ConstraintViolation { first: ConstraintFailure },
}

pub struct Prover {
    fri: FriShape,
}

impl Prover {
    pub const fn new(fri: FriShape) -> Prover {
        Prover { fri }
    }

    pub const fn fri(&self) -> &FriShape {
        &self.fri
    }

    #[allow(clippy::too_many_lines)]
    pub fn prove<A>(
        &self,
        air: &A,
        trace: &[Vec<Goldilocks>],
        prep: &[Vec<Goldilocks>],
        publics: &[Goldilocks],
        t: &mut FsTranscript,
    ) -> Result<Proved, ProveError>
    where
        A: Sync,
        for<'a> A: Air<PointCollector<'a, Goldilocks>>
            + Air<PointCollector<'a, ExtF>>
            + Air<NativeEval>
            + Air<MeasureBuilder>
            + Air<TrueDegreeBuilder>,
    {
        let width = trace.first().map_or(0, |r| r.len());
        if trace.is_empty() || width == 0 {
            return Err(ProveError::EmptyTrace);
        }
        let height = trace.len().max(prep.len());
        let log_n = height.max(2).next_power_of_two().ilog2() as usize;
        let n = 1usize << log_n;
        let padded_trace = pad_rows(trace, n, width);
        let prep_width = prep.first().map_or(0, |r| r.len());
        let padded_prep =
            if prep.is_empty() { Vec::new() } else { pad_rows(prep, n, prep_width) };

        let (count, _) = measure(air);
        let td = measure_true(air);
        let profile = Profile::wallet(
            AirShape {
                num_constraints: count,
                max_constraint_degree: td.max_heavy,
                max_combo: AirShape::NERV_MAX_COMBO,
            },
            2 * width + 1,
        );
        security::validate(&profile, &self.fri, log_n)?;

        #[cfg(debug_assertions)]
        if let Err(failures) = NativeEval::check_with_prep(
            padded_trace.clone(),
            padded_prep.clone(),
            publics.to_vec(),
            air,
            8,
        ) {
            if let Some(first) = failures.first() {
                return Err(ProveError::ConstraintViolation { first: first.clone() });
            }
        }

        let plan = Plan::new(log_n, &self.fri, &td)?;
        let proof = compose_prove(&plan, air, &padded_trace, &padded_prep, publics, t)?;
        Ok(Proved { proof, log_n, width, prep: padded_prep })
    }
}

pub(crate) fn pad_rows(rows: &[Vec<Goldilocks>], n: usize, width: usize) -> Vec<Vec<Goldilocks>> {
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(rows);
    while out.len() < n {
        out.push(vec![Goldilocks::ZERO; width]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::chips::conservation::{
        gen_cons_prep, gen_cons_trace, ConservationChip, PREP_ACTIVE, WIDTH as CONS_W,
    };
    use crate::air::chips::range::{gen_range_witness, RangeChip};
    use crate::stark::compose::ComposedProof;
    use crate::stark::verifier::Verifier;
    use nerv_core::codec::{Decode, Encode};

    fn strong_fri() -> FriShape {
        FriShape {
            log_blowup: 4,
            num_queries: 64,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        }
    }

    fn seeded(seed: u64) -> FsTranscript {
        let mut t = FsTranscript::new();
        t.absorb_bytes(&seed.to_le_bytes());
        t
    }

    fn range_rows(seed: u64, n: usize) -> Vec<Vec<Goldilocks>> {
        let mut r = crate::testutil::SplitMix64::new(seed);
        (0..n).map(|_| gen_range_witness(r.next_u64() % (1 << 20), 32)).collect()
    }

    #[test]
    fn range_end_to_end() {
        let rows = range_rows(0xA1, 4);
        let chip = RangeChip::new(0, 1, 32);
        let prover = Prover::new(strong_fri());
        let mut tp = seeded(0x5EED);
        let proved = prover.prove(&chip, &rows, &[], &[], &mut tp).unwrap();
        assert_eq!(proved.log_n, 2);
        assert_eq!(proved.width, 33);
        assert!(proved.prep.is_empty());

        let verifier = Verifier::new(strong_fri());
        let mut tv = seeded(0x5EED);
        assert!(verifier.verify(&chip, &proved, &[], &[], &mut tv).unwrap());

        let mut tp2 = seeded(0x5EED);
        let proved2 = prover.prove(&chip, &rows, &[], &[], &mut tp2).unwrap();
        assert_eq!(proved, proved2);

        let bytes = proved.proof.encode();
        assert_eq!(bytes.len(), proved.proof.encoded_len());
        let decoded = ComposedProof::decode(&bytes).unwrap();
        assert_eq!(decoded, proved.proof);
        let wire = Proved { proof: decoded, ..proved.clone() };
        let mut tv2 = seeded(0x5EED);
        assert!(verifier.verify(&chip, &wire, &[], &[], &mut tv2).unwrap());

        let mut tv3 = seeded(0xBEEF);
        assert!(!verifier.verify(&chip, &proved, &[], &[], &mut tv3).unwrap());
        let mut tv4 = seeded(0x5EED);
        assert!(!verifier.verify(&chip, &proved, &[], &[Goldilocks::ONE], &mut tv4).unwrap());

        let other = Verifier::new(FriShape { num_queries: 32, ..strong_fri() });
        let mut tv5 = seeded(0x5EED);
        assert!(!other.verify(&chip, &proved, &[], &[], &mut tv5).unwrap());

        let mut lying = proved.clone();
        lying.width = 32;
        let mut tv6 = seeded(0x5EED);
        assert!(!verifier.verify(&chip, &lying, &[], &[], &mut tv6).unwrap());

        let mut lying2 = proved.clone();
        lying2.log_n = 3;
        let mut tv7 = seeded(0x5EED);
        assert!(!verifier.verify(&chip, &lying2, &[], &[], &mut tv7).unwrap());
    }

    #[test]
    fn conservation_end_to_end_padding_and_prep_authenticity() {
        let entries = vec![(100u64, true), (60, false), (40, false)];
        let rows = gen_cons_trace(&entries);
        let prep = gen_cons_prep(1, 2, 0);
        let chip = ConservationChip::new();
        let prover = Prover::new(strong_fri());
        let mut tp = seeded(0xE0);
        let proved = prover.prove(&chip, &rows, &prep, &[], &mut tp).unwrap();

        assert_eq!(proved.log_n, 2);
        assert_eq!(proved.width, CONS_W);
        assert_eq!(proved.prep.len(), 4);
        assert_eq!(proved.prep[3], vec![Goldilocks::ZERO; 6]);

        let verifier = Verifier::new(strong_fri());
        let mut tv = seeded(0xE0);
        assert!(verifier.verify(&chip, &proved, &prep, &[], &mut tv).unwrap());

        let mut bad = prep.clone();
        bad[0][PREP_ACTIVE] = Goldilocks::ZERO;
        let mut tv2 = seeded(0xE0);
        assert!(!verifier.verify(&chip, &proved, &bad, &[], &mut tv2).unwrap());

        let mut long = prep.clone();
        long.push(vec![Goldilocks::ZERO; 6]);
        long.push(vec![Goldilocks::ZERO; 6]);
        let mut tv3 = seeded(0xE0);
        assert!(!verifier.verify(&chip, &proved, &long, &[], &mut tv3).unwrap());
    }

    #[test]
    fn weak_configuration_fails_the_gate() {
        let weak = FriShape {
            log_blowup: 1,
            num_queries: 2,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        };
        let prover = Prover::new(weak);
        let chip = RangeChip::new(0, 1, 32);
        let mut t = seeded(1);
        assert!(matches!(
            prover.prove(&chip, &range_rows(1, 4), &[], &[], &mut t),
            Err(ProveError::Security(_))
        ));
    }

    #[test]
    fn empty_trace_rejected() {
        let prover = Prover::new(strong_fri());
        let chip = RangeChip::new(0, 1, 32);
        let mut t = seeded(2);
        assert!(matches!(
            prover.prove(&chip, &[], &[], &[], &mut t),
            Err(ProveError::EmptyTrace)
        ));
        let mut t2 = seeded(3);
        assert!(matches!(
            prover.prove(&chip, &[Vec::new()], &[], &[], &mut t2),
            Err(ProveError::EmptyTrace)
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn constraint_violation_caught_by_the_debug_differential() {
        let mut rows = range_rows(0xB2, 4);
        rows[1][0] = rows[1][0] + Goldilocks::ONE;
        let chip = RangeChip::new(0, 1, 32);
        let prover = Prover::new(strong_fri());
        let mut t = seeded(0x5EED);
        assert!(matches!(
            prover.prove(&chip, &rows, &[], &[], &mut t),
            Err(ProveError::ConstraintViolation { .. })
        ));
    }
}

