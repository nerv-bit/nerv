//! The genesis-committed emission schedule (WP §12.2; errata 156–157):
//! the pure function of (day, params) whose deterministic replay theorem
//! M1 audits the ledger against. One day = one epoch; day 0 is genesis
//! and emits nothing.

use std::collections::BTreeSet;

use nerv_core::params::{self, EconomyBucket};

use crate::error::ScheduleError;

pub const SUPPLY_NERV: u64 = params::PROTOCOL_SUPPLY_NERV;
pub const DAY_SECS: u64 = params::TIMING_EPOCH_SECS;
pub const NANO_PER_NERV: u64 = params::PROTOCOL_NANO_PER_NERV;

const _: () = assert!(DAY_SECS == 86_400);
const _: () = assert!(NANO_PER_NERV == 1_000_000_000);
const _: () = assert!(SUPPLY_NERV == 10_000_000_000);

const MAX_QUARTERS: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccountKind {
    Signed,
    SignedGrant,
    ProducerPayout,
    CommitmentNote,
    ChallengerMarket,
}

impl AccountKind {
    fn parse(s: &str) -> Option<AccountKind> {
        Some(match s {
            "signed" => AccountKind::Signed,
            "signed-grant" => AccountKind::SignedGrant,
            "producer-payout" => AccountKind::ProducerPayout,
            "commitment-note" => AccountKind::CommitmentNote,
            "challenger-market" => AccountKind::ChallengerMarket,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Curve {
    Linear { cliff_days: u64, term_days: u64 },
    SubsidyDecline { year_one_days: u64, term_days: u64 },
    QuarterlyGeometric { quarters: u64, quarter_days: u64, num: u64, den: u64 },
    ClaimWindow { window_days: u64 },
    MilestoneWindow { window_days: u64 },
}

/// One bucket's frozen schedule. Constructible only through
/// [`EmissionSchedule::from_params`] — every instance is validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketSchedule {
    name: &'static str,
    account: AccountKind,
    total_nerv: u64,
    share_permille: u64,
    curve: Curve,
    burn_unclaimed: bool,
    /// The subsidy/quarterly normalization denominator; 0 otherwise.
    denom: u128,
}

impl BucketSchedule {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn account(&self) -> AccountKind {
        self.account
    }

    pub fn total_nerv(&self) -> u64 {
        self.total_nerv
    }

    pub fn share_permille(&self) -> u64 {
        self.share_permille
    }

    pub fn curve(&self) -> Curve {
        self.curve
    }

    pub fn burn_unclaimed(&self) -> bool {
        self.burn_unclaimed
    }

    pub fn is_time_driven(&self) -> bool {
        !matches!(self.curve, Curve::ClaimWindow { .. } | Curve::MilestoneWindow { .. })
    }

    pub fn term_days(&self) -> u64 {
        match self.curve {
            Curve::Linear { term_days, .. } | Curve::SubsidyDecline { term_days, .. } => term_days,
            Curve::QuarterlyGeometric { quarters, quarter_days, .. } => quarters * quarter_days,
            Curve::ClaimWindow { window_days } | Curve::MilestoneWindow { window_days } => {
                window_days
            }
        }
    }

    pub fn window_days(&self) -> Option<u64> {
        match self.curve {
            Curve::ClaimWindow { window_days } | Curve::MilestoneWindow { window_days } => {
                Some(window_days)
            }
            _ => None,
        }
    }

    /// Claim/milestone windows are open for 1 ≤ day ≤ window_days
    /// (genesis day 0 is closed — nothing exists to claim).
    pub fn window_open(&self, day: u64) -> bool {
        match self.curve {
            Curve::ClaimWindow { window_days } | Curve::MilestoneWindow { window_days } => {
                day >= 1 && day <= window_days
            }
            _ => false,
        }
    }

    /// Cumulative whole-NERV released by the end of `day`. Event-driven
    /// buckets return 0 — their emission lives in the ledger.
    pub fn released_by_day(&self, day: u64) -> u64 {
        if self.total_nerv == 0 {
            return 0;
        }
        match self.curve {
            Curve::Linear { cliff_days, term_days } => {
                if day <= cliff_days {
                    0
                } else if day >= term_days {
                    self.total_nerv
                } else {
                    ((self.total_nerv as u128 * u128::from(day - cliff_days))
                        / u128::from(term_days - cliff_days)) as u64
                }
            }
            Curve::SubsidyDecline { year_one_days, term_days } => {
                if day >= term_days {
                    self.total_nerv
                } else {
                    let w = subsidy_weight(term_days, year_one_days, day);
                    ((self.total_nerv as u128 * w) / self.denom) as u64
                }
            }
            Curve::QuarterlyGeometric { quarters, quarter_days, num, den } => {
                if day >= quarters * quarter_days {
                    self.total_nerv
                } else {
                    let s = quarterly_partial(quarters, quarter_days, num, den, day);
                    ((self.total_nerv as u128 * s) / self.denom) as u64
                }
            }
            Curve::ClaimWindow { .. } | Curve::MilestoneWindow { .. } => 0,
        }
    }

    pub fn day_emission(&self, day: u64) -> u64 {
        self.released_by_day(day).saturating_sub(self.released_by_day(day.saturating_sub(1)))
    }

    pub fn released_nano(&self, day: u64) -> u128 {
        u128::from(self.released_by_day(day)) * u128::from(NANO_PER_NERV)
    }

    pub fn day_emission_nano(&self, day: u64) -> u128 {
        u128::from(self.day_emission(day)) * u128::from(NANO_PER_NERV)
    }
}

// -- the weight/numerator arithmetic (erratum 156) -------------------------

/// W(d) = Σ_{i<d} w(i) with w(i) = 2(T−Y) for i < Y, 2(T−i) for Y ≤ i < T.
fn subsidy_weight(term: u64, year_one: u64, day: u64) -> u128 {
    let d = u128::from(day.min(term));
    if d == 0 {
        return 0;
    }
    let (t, y) = (u128::from(term), u128::from(year_one));
    if d <= y {
        2 * (t - y) * d
    } else {
        let base = 2 * (t - y) * y + (t - y) * (t - y + 1);
        let rem = (t - d) * (t - d + 1);
        base - rem
    }
}

fn subsidy_denominator(term: u64, year_one: u64) -> u128 {
    let (t, y) = (u128::from(term), u128::from(year_one));
    (t - y) * (t + y + 1)
}

/// (den^q − num^q)/(den − num) — the geometric sum, always integral.
fn quarterly_denominator(quarters: u64, num: u64, den: u64) -> Option<u128> {
    let den_q = (u128::from(den)).checked_pow(quarters as u32)?;
    let num_q = (u128::from(num)).checked_pow(quarters as u32)?;
    Some((den_q - num_q) / u128::from(den - num))
}

/// Σ_{k<elapsed} num^k·den^(Q−1−k), elapsed = min(day/quarter_days, Q).
fn quarterly_partial(
    quarters: u64,
    quarter_days: u64,
    num: u64,
    den: u64,
    day: u64,
) -> u128 {
    let elapsed = (day / quarter_days).min(quarters);
    let mut term = u128::from(den).pow((quarters - 1) as u32);
    let mut sum = 0u128;
    for _ in 0..elapsed {
        sum += term;
        term = (term / u128::from(den)) * u128::from(num);
    }
    sum
}

// -- the schedule -----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmissionSchedule {
    buckets: Vec<BucketSchedule>,
}

impl EmissionSchedule {
    pub fn from_params(buckets: &[EconomyBucket]) -> Result<EmissionSchedule, ScheduleError> {
        let mut parsed = Vec::with_capacity(buckets.len());
        for b in buckets {
            let account = AccountKind::parse(b.account).ok_or(ScheduleError::UnknownAccount {
                bucket: b.name,
                account: b.account,
            })?;
            let (curve, denom, burn_unclaimed) = parse_curve(b)?;
            parsed.push(BucketSchedule {
                name: b.name,
                account,
                total_nerv: b.total_nerv,
                share_permille: b.share_permille,
                curve,
                burn_unclaimed,
                denom,
            });
        }
        Ok(EmissionSchedule { buckets: parsed })
    }

    /// The allocation invariants (erratum 157): shares, supply, names.
    pub fn validate_allocation(&self) -> Result<(), ScheduleError> {
        let mut shares = 0u64;
        let mut totals = 0u128;
        let mut names = BTreeSet::new();
        for b in &self.buckets {
            shares += b.share_permille;
            totals += u128::from(b.total_nerv);
            if !names.insert(b.name) {
                return Err(ScheduleError::DuplicateBucket { name: b.name });
            }
        }
        if shares != 1000 {
            return Err(ScheduleError::ShareSum { found: shares });
        }
        if totals != u128::from(SUPPLY_NERV) {
            return Err(ScheduleError::SupplySum { found: totals, expected: SUPPLY_NERV });
        }
        Ok(())
    }

    /// The committed schedule: the compile-time params static. The parse
    /// is total on the committed file; nerv-conformance re-runs both
    /// validations against the live tree.
    pub fn genesis() -> EmissionSchedule {
        let schedule = EmissionSchedule::from_params(&params::ECONOMY_BUCKETS)
            .expect("specs/params.toml [economy] parses (conformance-validated)");
        schedule
            .validate_allocation()
            .expect("specs/params.toml [economy] allocation invariants hold");
        schedule
    }

    pub fn buckets(&self) -> &[BucketSchedule] {
        &self.buckets
    }

    pub fn bucket(&self, name: &str) -> Option<&BucketSchedule> {
        self.buckets.iter().find(|b| b.name == name)
    }

    /// Time-driven cumulative emission (event-driven buckets excluded —
    /// their emission is ledger-recorded).
    pub fn released_by_day(&self, day: u64) -> u64 {
        self.buckets
            .iter()
            .filter(|b| b.is_time_driven())
            .map(|b| b.released_by_day(day))
            .sum()
    }

    /// The §12.6 upper bound: time-driven plus the full claim/milestone
    /// totals while their windows are open. Illustrative only (P9).
    pub fn upper_bound_by_day(&self, day: u64) -> u64 {
        self.buckets
            .iter()
            .map(|b| {
                if b.is_time_driven() {
                    b.released_by_day(day)
                } else if b.window_open(day) {
                    b.total_nerv
                } else {
                    0
                }
            })
            .sum()
    }

    pub fn time_driven_total(&self) -> u64 {
        self.buckets
            .iter()
            .filter(|b| b.is_time_driven())
            .map(|b| b.total_nerv)
            .sum()
    }
}

fn parse_curve(b: &EconomyBucket) -> Result<(Curve, u128, bool), ScheduleError> {
    let missing =
        |field: &'static str| ScheduleError::MissingField { bucket: b.name, field };
    let bad = |detail: &'static str| ScheduleError::Inconsistent { bucket: b.name, detail };
    match b.kind {
        "linear-vesting" => {
            let cliff = b.cliff_days.ok_or_else(|| missing("cliff_days"))?;
            let term = b.term_days;
            if cliff >= term {
                return Err(bad("cliff_days must precede term_days"));
            }
            if let Some(lin) = b.linear_days {
                if lin != term - cliff {
                    return Err(bad("linear_days must equal term_days − cliff_days"));
                }
            }
            Ok((Curve::Linear { cliff_days: cliff, term_days: term }, 0, false))
        }
        "subsidy-decline" => {
            let year_one = b.year_one_days.ok_or_else(|| missing("year_one_days"))?;
            let term = b.term_days;
            if year_one >= term {
                return Err(bad("year_one_days must precede term_days"));
            }
            Ok((
                Curve::SubsidyDecline { year_one_days: year_one, term_days: term },
                subsidy_denominator(term, year_one),
                false,
            ))
        }
        "claim-window" => {
            let window = b.window_days.ok_or_else(|| missing("window_days"))?;
            if window == 0 || window > b.term_days {
                return Err(bad("window_days must be in 1..=term_days"));
            }
            let burn = b.burn_unclaimed.ok_or_else(|| missing("burn_unclaimed"))?;
            if !burn {
                return Err(bad("claim windows burn unclaimed amounts at close (§12.2)"));
            }
            Ok((Curve::ClaimWindow { window_days: window }, 0, true))
        }
        "milestone-window" => {
            let window = b.window_days.ok_or_else(|| missing("window_days"))?;
            if window == 0 || window > b.term_days {
                return Err(bad("window_days must be in 1..=term_days"));
            }
            if b.burn_unclaimed.is_some() {
                return Err(bad("burn_unclaimed does not apply to milestone windows"));
            }
            Ok((Curve::MilestoneWindow { window_days: window }, 0, false))
        }
        "quarterly-geometric" => {
            let quarters = b.quarters.ok_or_else(|| missing("quarters"))?;
            let quarter_days = b.quarter_days.ok_or_else(|| missing("quarter_days"))?;
            let num = b.ratio_num.ok_or_else(|| missing("ratio_num"))?;
            let den = b.ratio_den.ok_or_else(|| missing("ratio_den"))?;
            if quarters == 0 || quarters > MAX_QUARTERS {
                return Err(bad("quarters must be in 1..=64"));
            }
            if quarter_days == 0 {
                return Err(bad("quarter_days must be nonzero"));
            }
            let product = quarters
                .checked_mul(quarter_days)
                .ok_or_else(|| bad("quarters × quarter_days overflows"))?;
            if b.term_days != product {
                return Err(bad("term_days must equal quarters × quarter_days"));
            }
            if num == 0 || num >= den {
                return Err(bad("ratio_num must be in 1..ratio_den"));
            }
            let denom = quarterly_denominator(quarters, num, den)
                .ok_or_else(|| ScheduleError::Overflow { bucket: b.name })?;
            Ok((
                Curve::QuarterlyGeometric { quarters, quarter_days, num, den },
                denom,
                false,
            ))
        }
        other => Err(ScheduleError::UnknownKind { bucket: b.name, kind: other }),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn raw(name: &'static str, kind: &'static str, total: u64) -> EconomyBucket {
        EconomyBucket {
            name,
            kind,
            total_nerv: total,
            share_permille: 0,
            account: "signed",
            term_days: 3600,
            cliff_days: None,
            linear_days: None,
            year_one_days: None,
            quarters: None,
            quarter_days: None,
            ratio_num: None,
            ratio_den: None,
            window_days: None,
            burn_unclaimed: None,
        }
    }

    #[test]
    fn frozen_params_parse_and_validate() {
        let s = EmissionSchedule::from_params(&params::ECONOMY_BUCKETS).unwrap();
        s.validate_allocation().unwrap();
        assert_eq!(s.buckets().len(), params::ECONOMY_BUCKET_COUNT);
        assert_eq!(s.buckets().len(), 7);
        let names: Vec<&str> = s.buckets().iter().map(|b| b.name()).collect();
        assert_eq!(
            names,
            vec![
                "early-contributors",
                "founder",
                "validator-subsidy",
                "community",
                "ecosystem",
                "foundation",
                "useful-work",
            ]
        );
        assert_eq!(s.buckets().iter().filter(|b| b.is_time_driven()).count(), 5);
        assert_eq!(
            s.buckets().iter().filter(|b| !b.is_time_driven()).count(), 2,
            "community + ecosystem are event-driven"
        );
        assert_eq!(EmissionSchedule::genesis(), s);
    }

    #[test]
    fn allocation_invariants() {
        let s = EmissionSchedule::genesis();
        let shares: u64 = s.buckets().iter().map(|b| b.share_permille()).sum();
        assert_eq!(shares, 1000);
        let totals: u128 = s.buckets().iter().map(|b| u128::from(b.total_nerv())).sum();
        assert_eq!(totals, 10_000_000_000);
        assert_eq!(s.time_driven_total(), 6_800_000_000);
        assert_eq!(s.released_by_day(u64::MAX / 4), 6_800_000_000);
        assert_eq!(s.upper_bound_by_day(u64::MAX / 4), 10_000_000_000);
        assert_eq!(s.released_by_day(0), 0);
        assert_eq!(s.upper_bound_by_day(0), 0, "day 0: every window closed");
        assert_eq!(s.bucket("founder").unwrap().account(), AccountKind::Signed);
        assert_eq!(
            s.bucket("community").unwrap().account(),
            AccountKind::CommitmentNote
        );
        assert_eq!(
            s.bucket("validator-subsidy").unwrap().account(),
            AccountKind::ProducerPayout
        );
        assert_eq!(
            s.bucket("ecosystem").unwrap().account(),
            AccountKind::SignedGrant
        );
        assert_eq!(
            s.bucket("useful-work").unwrap().account(),
            AccountKind::ChallengerMarket
        );
        assert!(s.bucket("nonexistent").is_none());
    }

    #[test]
    fn linear_pins() {
        let s = EmissionSchedule::genesis();
        let early = s.bucket("early-contributors").unwrap();
        assert_eq!(early.released_by_day(0), 0);
        assert_eq!(early.released_by_day(180), 0, "the cliff emits nothing");
        assert_eq!(early.released_by_day(181), 1_587_301); // 2e9/1260 floored
        assert_eq!(early.released_by_day(360), 285_714_285); // 2e9/7
        assert_eq!(early.released_by_day(1440), 2_000_000_000);
        assert_eq!(early.released_by_day(5000), 2_000_000_000);
        assert_eq!(early.day_emission(0), 0);
        assert_eq!(early.day_emission(181), 1_587_301);

        let founder = s.bucket("founder").unwrap();
        assert_eq!(founder.released_by_day(360), 0);
        assert_eq!(founder.released_by_day(361), 416_666); // 600M/1440 floored
        assert_eq!(founder.released_by_day(1800), 600_000_000);

        let work = s.bucket("useful-work").unwrap();
        assert_eq!(work.released_by_day(1), 83_333); // 300M/3600 floored
        assert_eq!(work.released_by_day(360), 30_000_000);
        assert_eq!(work.released_by_day(3600), 300_000_000);
        assert_eq!(work.released_nano(360), 30_000_000_000_000_000);
        assert_eq!(work.day_emission_nano(1), 83_333_000_000_000);
    }

    #[test]
    fn linear_telescoping_is_exact() {
        let s = EmissionSchedule::genesis();
        for name in ["early-contributors", "founder", "useful-work"] {
            let b = s.bucket(name).unwrap();
            let term = b.term_days();
            let mut sum = 0u64;
            let mut prev = 0u64;
            for d in 0..=term + 5 {
                let r = b.released_by_day(d);
                assert!(r >= prev, "{name}: monotone at {d}");
                prev = r;
                if d >= 1 {
                    sum += b.day_emission(d);
                }
            }
            assert_eq!(sum, b.total_nerv(), "{name}: day emissions telescope to total");
            assert_eq!(b.released_by_day(term), b.total_nerv());
        }
    }

    /// The independent per-day weight sum (the differential for the
    /// closed-form W(d)).
    fn weight_loop(term: u64, year_one: u64, day: u64) -> u128 {
        (0..day.min(term))
            .map(|i| {
                if i < year_one {
                    2 * u128::from(term - year_one)
                } else {
                    2 * u128::from(term - i)
                }
            })
            .sum()
    }

    #[test]
    fn subsidy_weight_matches_the_loop_differential() {
        for (t, y) in [(3600u64, 360u64), (4, 2), (10, 0), (5, 4), (3600, 3599)] {
            for d in 0..=t + 2 {
                assert_eq!(
                    subsidy_weight(t, y, d),
                    weight_loop(t, y, d),
                    "t={t} y={y} d={d}"
                );
            }
            assert_eq!(
                subsidy_denominator(t, y),
                weight_loop(t, y, t),
                "D is the full weight sum: t={t} y={y}"
            );
        }
    }

    #[test]
    fn subsidy_synthetic_exact() {
        // T=4, Y=2, total=70: weights 4,4,4,2; D=14.
        let b = EconomyBucket {
            year_one_days: Some(2),
            term_days: 4,
            ..raw("syn", "subsidy-decline", 70)
        };
        let s = EmissionSchedule::from_params(&[b]).unwrap();
        let bucket = s.bucket("syn").unwrap();
        let expect = [0u64, 20, 40, 60, 70, 70];
        for (d, &want) in expect.iter().enumerate() {
            assert_eq!(bucket.released_by_day(d as u64), want, "day {d}");
        }
        assert_eq!(bucket.day_emission(1), 20);
        assert_eq!(bucket.day_emission(4), 10);
        assert_eq!(bucket.day_emission(5), 0);
    }

    #[test]
    fn subsidy_genesis_shape() {
        let s = EmissionSchedule::genesis();
        let b = s.bucket("validator-subsidy").unwrap();
        let y1 = b.released_by_day(360);
        assert!(
            (545_000_000..=546_000_000).contains(&y1),
            "year-one ≈ 545M (erratum 156), got {y1}"
        );
        assert_eq!(b.released_by_day(0), 0);
        assert_eq!(b.released_by_day(3600), 3_000_000_000);
        assert!(b.released_by_day(3599) < 3_000_000_000);
        // Year-one days run at the constant rate (the ±1 of flooring).
        for d in 1..=359u64 {
            let e = b.day_emission(d);
            assert!(
                (1_514_000..=1_516_000).contains(&e),
                "year-one daily rate at d={d}: {e}"
            );
        }
        // Continuity at the boundary: the weights at Y−1 and Y are equal.
        let (a, c) = (b.day_emission(360), b.day_emission(361));
        assert!(a.abs_diff(c) <= 2, "decline starts at the year-one rate: {a} vs {c}");
        // The decline is real and monotone.
        assert!(b.day_emission(361) > b.day_emission(1000));
        assert!(b.day_emission(1000) > b.day_emission(3599));
        assert!(b.day_emission(3599) > 0);
        // Telescoping.
        let mut sum = 0u64;
        for d in 1..=3600u64 {
            sum += b.day_emission(d);
        }
        assert_eq!(sum, 3_000_000_000);
    }

    /// The independent numerator sum (the differential for D).
    fn geo_loop(quarters: u64, num: u64, den: u64) -> u128 {
        (0..quarters)
            .map(|k| {
                u128::from(num).pow(k as u32) * u128::from(den).pow((quarters - 1 - k) as u32)
            })
            .sum()
    }

    #[test]
    fn quarterly_denominator_differential() {
        assert_eq!(quarterly_denominator(20, 9, 10), Some(geo_loop(20, 9, 10)));
        assert_eq!(
            quarterly_denominator(20, 9, 10),
            Some(100_000_000_000_000_000_000u128 - 12_157_665_459_056_928_801u128)
        );
        assert_eq!(quarterly_denominator(4, 1, 2), Some(15));
        assert_eq!(quarterly_denominator(1, 1, 2), Some(1));
        assert_eq!(quarterly_denominator(64, 1, 2), Some(u128::from(2u128.pow(63))));
        assert_eq!(quarterly_denominator(64, u64::MAX - 1, u64::MAX), None, "overflow");
    }

    #[test]
    fn quarterly_synthetic_exact() {
        // num/den = 1/2, 4 quarters of 10 days, total 1500:
        // numerators 8,4,2,1 of D=15 → 800, 400, 200, 100.
        let b = EconomyBucket {
            quarters: Some(4),
            quarter_days: Some(10),
            ratio_num: Some(1),
            ratio_den: Some(2),
            term_days: 40,
            ..raw("syn", "quarterly-geometric", 1500)
        };
        let s = EmissionSchedule::from_params(&[b]).unwrap();
        let bucket = s.bucket("syn").unwrap();
        for d in 0..=89 {
            assert_eq!(bucket.released_by_day(d), 0, "day {d}: nothing before Q1 ends");
        }
        assert_eq!(bucket.released_by_day(90), 800);
        assert_eq!(bucket.released_by_day(91), 800);
        assert_eq!(bucket.released_by_day(100), 800, "no double-release inside a quarter");
        assert_eq!(bucket.released_by_day(110), 1200);
        assert_eq!(bucket.released_by_day(130), 1400);
        assert_eq!(bucket.released_by_day(150), 1500);
        assert_eq!(bucket.released_by_day(400), 1500);
        assert_eq!(bucket.day_emission(90), 800);
        assert_eq!(bucket.day_emission(91), 0);
    }

    #[test]
    fn quarterly_genesis_shape() {
        let s = EmissionSchedule::genesis();
        let b = s.bucket("foundation").unwrap();
        assert_eq!(b.term_days(), 1800);
        let q1 = b.released_by_day(90);
        assert!(
            (102_400_000..=102_520_000).contains(&q1),
            "first quarter ≈ 102.46M, got {q1}"
        );
        assert_eq!(b.released_by_day(1800), 900_000_000);
        // Quarter amounts strictly decrease (r = 9/10).
        let mut amounts = Vec::new();
        for k in 0..20u64 {
            let end = b.released_by_day((k + 1) * 90);
            let start = b.released_by_day(k * 90);
            amounts.push(end - start);
        }
        for w in amounts.windows(2) {
            assert!(w[1] < w[0], "quarterly amounts decrease: {:?}", &amounts[..4]);
        }
        assert_eq!(amounts.iter().sum::<u64>(), 900_000_000);
        // Telescoping over the full term.
        let mut sum = 0u64;
        for d in 1..=1800u64 {
            sum += b.day_emission(d);
        }
        assert_eq!(sum, 900_000_000);
    }

    #[test]
    fn windows() {
        let s = EmissionSchedule::genesis();
        let community = s.bucket("community").unwrap();
        assert_eq!(community.window_days(), Some(720));
        assert!(!community.window_open(0));
        assert!(community.window_open(1));
        assert!(community.window_open(720));
        assert!(!community.window_open(721));
        assert!(community.burn_unclaimed());
        assert_eq!(community.released_by_day(500), 0, "event-driven");

        let eco = s.bucket("ecosystem").unwrap();
        assert_eq!(eco.window_days(), Some(1800));
        assert!(eco.window_open(1800));
        assert!(!eco.window_open(1801));
        assert!(!eco.burn_unclaimed());
        assert_eq!(eco.released_by_day(1000), 0);
        // The time-driven buckets have no windows.
        for name in ["early-contributors", "founder", "validator-subsidy", "foundation", "useful-work"] {
            assert!(s.bucket(name).unwrap().window_days().is_none());
            assert!(!s.bucket(name).unwrap().window_open(5));
        }
    }

    #[test]
    fn monotone_everywhere_and_day_zero() {
        let s = EmissionSchedule::genesis();
        for b in s.buckets() {
            assert_eq!(b.released_by_day(0), 0);
            assert_eq!(b.day_emission(0), 0);
            let mut prev = 0u64;
            for d in 1..=3610u64 {
                let r = b.released_by_day(d);
                assert!(r >= prev, "{} at {}", b.name(), d);
                prev = r;
            }
            if b.is_time_driven() {
                assert_eq!(prev, b.total_nerv(), "{} saturates at total", b.name());
            }
        }
        let mut prev = 0u64;
        for d in 0..=4000u64 {
            let r = s.released_by_day(d);
            assert!(r >= prev);
            prev = r;
        }
    }

    /// The §12.6 upper bound at day 360 (full claim rates) — a sanity
    /// band, not a conformance target (erratum 156).
    #[test]
    fn year_one_upper_bound_band() {
        let s = EmissionSchedule::genesis();
        let ub = s.upper_bound_by_day(360);
        assert!(
            (4_380_000_000..=4_450_000_000).contains(&ub),
            "§12.6 full-claim upper bound ≈ 44.1%, got {ub}"
        );
        // The WP's ~25% illustrative line reflects realistic claim rates;
        // the time-driven floor alone:
        assert!(
            (1_200_000_000..=1_230_000_000).contains(&s.released_by_day(360)),
            "time-driven year-one ≈ 1.21B"
        );
    }

    #[test]
    fn positive_synthetic_allocation() {
        let mut a = EconomyBucket { share_permille: 500, ..raw("a", "linear-vesting", 5_000_000_000) };
        a.cliff_days = Some(0);
        a.linear_days = Some(3600);
        let mut b = raw("b", "claim-window", 5_000_000_000);
        b.share_permille = 500;
        b.account = "commitment-note";
        b.term_days = 720;
        b.window_days = Some(720);
        b.burn_unclaimed = Some(true);
        let s = EmissionSchedule::from_params(&[a, b]).unwrap();
        s.validate_allocation().unwrap();
        assert_eq!(s.released_by_day(3600), 5_000_000_000);
        assert_eq!(s.upper_bound_by_day(1), 10_000_000_000);
        assert_eq!(s.upper_bound_by_day(721), 5_000_000_000, "claims closed");
    }

    #[test]
    fn validation_errors() {
        let cases: Vec<(EconomyBucket, ScheduleError)> = vec![
            (
                raw("x", "wtf", 1),
                ScheduleError::UnknownKind { bucket: "x", kind: "wtf" },
            ),
            (
                EconomyBucket { account: "nope", ..raw("x", "linear-vesting", 1) },
                ScheduleError::UnknownAccount { bucket: "x", account: "nope" },
            ),
            (
                raw("x", "linear-vesting", 1),
                ScheduleError::MissingField { bucket: "x", field: "cliff_days" },
            ),
            (
                EconomyBucket { cliff_days: Some(100), term_days: 50, ..raw("x", "linear-vesting", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "cliff_days must precede term_days" },
            ),
            (
                EconomyBucket {
                    cliff_days: Some(10),
                    term_days: 100,
                    linear_days: Some(50),
                    ..raw("x", "linear-vesting", 1)
                },
                ScheduleError::Inconsistent { bucket: "x", detail: "linear_days must equal term_days − cliff_days" },
            ),
            (
                raw("x", "subsidy-decline", 1),
                ScheduleError::MissingField { bucket: "x", field: "year_one_days" },
            ),
            (
                EconomyBucket { year_one_days: Some(100), term_days: 100, ..raw("x", "subsidy-decline", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "year_one_days must precede term_days" },
            ),
            (
                raw("x", "claim-window", 1),
                ScheduleError::MissingField { bucket: "x", field: "window_days" },
            ),
            (
                EconomyBucket { window_days: Some(0), term_days: 720, ..raw("x", "claim-window", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "window_days must be in 1..=term_days" },
            ),
            (
                EconomyBucket { window_days: Some(800), term_days: 720, ..raw("x", "claim-window", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "window_days must be in 1..=term_days" },
            ),
            (
                EconomyBucket { term_days: 720, window_days: Some(720), ..raw("x", "claim-window", 1) },
                ScheduleError::MissingField { bucket: "x", field: "burn_unclaimed" },
            ),
            (
                EconomyBucket {
                    term_days: 720,
                    window_days: Some(720),
                    burn_unclaimed: Some(false),
                    ..raw("x", "claim-window", 1)
                },
                ScheduleError::Inconsistent { bucket: "x", detail: "claim windows burn unclaimed amounts at close (§12.2)" },
            ),
            (
                EconomyBucket {
                    term_days: 720,
                    window_days: Some(720),
                    burn_unclaimed: Some(true),
                    ..raw("x", "milestone-window", 1)
                },
                ScheduleError::Inconsistent { bucket: "x", detail: "burn_unclaimed does not apply to milestone windows" },
            ),
            (
                raw("x", "quarterly-geometric", 1),
                ScheduleError::MissingField { bucket: "x", field: "quarters" },
            ),
            (
                EconomyBucket { quarters: Some(0), quarter_days: Some(90), ratio_num: Some(9), ratio_den: Some(10), term_days: 0, ..raw("x", "quarterly-geometric", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "quarters must be in 1..=64" },
            ),
            (
                EconomyBucket { quarters: Some(20), quarter_days: Some(90), ratio_num: Some(9), ratio_den: Some(10), term_days: 1700, ..raw("x", "quarterly-geometric", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "term_days must equal quarters × quarter_days" },
            ),
            (
                EconomyBucket { quarters: Some(20), quarter_days: Some(90), ratio_num: Some(10), ratio_den: Some(10), term_days: 1800, ..raw("x", "quarterly-geometric", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "ratio_num must be in 1..ratio_den" },
            ),
            (
                EconomyBucket { quarters: Some(20), quarter_days: Some(90), ratio_num: Some(0), ratio_den: Some(10), term_days: 1800, ..raw("x", "quarterly-geometric", 1) },
                ScheduleError::Inconsistent { bucket: "x", detail: "ratio_num must be in 1..ratio_den" },
            ),
            (
                EconomyBucket { quarters: Some(64), quarter_days: Some(90), ratio_num: Some(1), ratio_den: Some(u64::MAX), term_days: 5760, ..raw("x", "quarterly-geometric", 1) },
                ScheduleError::Overflow { bucket: "x" },
            ),
        ];
        for (b, want) in cases {
            assert_eq!(EmissionSchedule::from_params(&[b]), Err(want));
        }
    }

    #[test]
    fn allocation_errors() {
        let mut a = EconomyBucket { share_permille: 500, ..raw("a", "linear-vesting", 5_000_000_000) };
        a.cliff_days = Some(0);
        a.linear_days = Some(3600);
        let mut b = a.clone();
        b.name = "b";
        b.share_permille = 499;
        let s = EmissionSchedule::from_params(&[a, b]).unwrap();
        assert_eq!(
            s.validate_allocation(),
            Err(ScheduleError::ShareSum { found: 999 })
        );

        let mut a = EconomyBucket { share_permille: 500, ..raw("a", "linear-vesting", 5_000_000_000) };
        a.cliff_days = Some(0);
        a.linear_days = Some(3600);
        let mut b = a.clone();
        b.name = "b";
        b.total_nerv = 4_000_000_000;
        let s = EmissionSchedule::from_params(&[a, b]).unwrap();
        assert_eq!(
            s.validate_allocation(),
            Err(ScheduleError::SupplySum { found: 9_000_000_000, expected: 10_000_000_000 })
        );

        let mut a = EconomyBucket { share_permille: 500, ..raw("dup", "linear-vesting", 5_000_000_000) };
        a.cliff_days = Some(0);
        a.linear_days = Some(3600);
        let mut b = a.clone();
        b.share_permille = 500;
        let s = EmissionSchedule::from_params(&[a, b]).unwrap();
        assert_eq!(
            s.validate_allocation(),
            Err(ScheduleError::DuplicateBucket { name: "dup" })
        );

        assert_eq!(
            EmissionSchedule::from_params(&[]).unwrap().validate_allocation(),
            Err(ScheduleError::ShareSum { found: 0 })
        );
    }
}
