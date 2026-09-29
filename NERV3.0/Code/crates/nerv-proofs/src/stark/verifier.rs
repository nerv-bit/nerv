//! The verification driver (WP §5.5, the B-gate's 0.2–0.5 ms class):
//! rebuilds the composition plan from the statement and the AIR it holds,
//! pads its own preprocessed table to the statement's height, and
//! delegates to `compose_verify`. The table is the verifier's OWN
//! authenticated copy — one that differs from the prover's rejects.
//! Adversarial `Proved` statements are rejected, never panic.

use crate::air::builder::Air;
use crate::air::fs::FsTranscript;
use crate::security::FriShape;
use crate::stark::collector::{measure_true, PointCollector, TrueDegreeBuilder};
use crate::stark::compose::{compose_verify, ComposeError, Plan};
use crate::stark::domain::TWO_ADICITY;
use crate::stark::ext_field::ExtF;
use crate::stark::prover::{pad_rows, Proved};
use nerv_core::field::Goldilocks;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    #[error(transparent)]
    Compose(#[from] ComposeError),
    #[error("verifier's preprocessed rows are ragged")]
    RaggedPrep,
}

pub struct Verifier {
    fri: FriShape,
}

impl Verifier {
    pub const fn new(fri: FriShape) -> Verifier {
        Verifier { fri }
    }

    pub const fn fri(&self) -> &FriShape {
        &self.fri
    }

    pub fn verify<A>(
        &self,
        air: &A,
        proved: &Proved,
        prep: &[Vec<Goldilocks>],
        publics: &[Goldilocks],
        t: &mut FsTranscript,
    ) -> Result<bool, VerifyError>
    where
        A: Sync,
        for<'a> A: Air<PointCollector<'a, Goldilocks>> + Air<PointCollector<'a, ExtF>>
            + Air<TrueDegreeBuilder>,
    {
        if proved.width == 0 || !(1..=TWO_ADICITY).contains(&proved.log_n) {
            return Ok(false);
        }
        let n = 1usize << proved.log_n;
        let prep_width = prep.first().map_or(0, |r| r.len());
        if prep.iter().any(|r| r.len() != prep_width) {
            return Err(VerifyError::RaggedPrep);
        }
        if prep.len() > n {
            return Ok(false);
        }
        let padded_prep =
            if prep.is_empty() { Vec::new() } else { pad_rows(prep, n, prep_width) };

        let td = measure_true(air);
        let plan = Plan::new(proved.log_n, &self.fri, &td)?;
        compose_verify(&plan, air, proved.width, &padded_prep, publics, &proved.proof, t)
            .map_err(VerifyError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::chips::range::{gen_range_witness, RangeChip};
    use crate::air::sym::{measure, MeasureBuilder};
    use crate::stark::prover::Prover;
    use crate::testutil::SplitMix64;

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

    fn proved_range() -> (RangeChip, Proved) {
        let mut r = SplitMix64::new(0x6E);
        let rows: Vec<Vec<Goldilocks>> =
            (0..4).map(|_| gen_range_witness(r.next_u64() % (1 << 20), 32)).collect();
        let chip = RangeChip::new(0, 1, 32);
        let mut t = seeded(0x5EED);
        let proved = Prover::new(strong_fri()).prove(&chip, &rows, &[], &[], &mut t).unwrap();
        (chip, proved)
    }

    #[test]
    fn statement_guards_reject_totally() {
        let (chip, proved) = proved_range();
        let verifier = Verifier::new(strong_fri());

        let mut zero_w = proved.clone();
        zero_w.width = 0;
        let mut t = seeded(1);
        assert_eq!(verifier.verify(&chip, &zero_w, &[], &[], &mut t), Ok(false));

        for bad_log in [0usize, TWO_ADICITY + 1, 64] {
            let mut lying = proved.clone();
            lying.log_n = bad_log;
            let mut t = seeded(1);
            assert_eq!(verifier.verify(&chip, &lying, &[], &[], &mut t), Ok(false), "log={bad_log}");
        }
    }

    #[test]
    fn ragged_prep_is_a_caller_error() {
        let (chip, proved) = proved_range();
        let ragged = vec![vec![Goldilocks::ZERO; 6], vec![Goldilocks::ZERO]];
        let mut t = seeded(1);
        assert_eq!(
            Verifier::new(strong_fri()).verify(&chip, &proved, &ragged, &[], &mut t),
            Err(VerifyError::RaggedPrep)
        );
    }

    #[test]
    fn the_verifier_measures_the_air_it_holds() {
        let (chip, proved) = proved_range();
        assert_eq!(measure(&chip).0, 33);
        let td = measure_true(&chip);
        assert_eq!((td.max_heavy, td.uses_transition), (2, false));
        let mut t = seeded(0x5EED);
        assert!(Verifier::new(strong_fri()).verify(&chip, &proved, &[], &[], &mut t).unwrap());
        let _ = MeasureBuilder::new();
    }
}

