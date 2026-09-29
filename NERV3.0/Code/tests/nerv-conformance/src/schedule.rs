//! Genesis emission schedule — the exact integer reference (WP §12.2/§12.3,
//! theorem M1). Deterministic pacing per bucket; claim/milestone buckets are
//! envelopes (availability windows) since their emission is event-driven.
//!
//! Day semantics: cumulative(d) = emitted by the end of day d; day 0 is the
//! first 24 h after genesis. One round-half-even rounding per evaluation.
//! Validator-subsidy shape: constant daily rate r = total/1980 for year one
//! (= 545,454,545.45 NERV ≈ the "~545M" of §12.2), then the daily rate
//! declines linearly to zero at term; the closed form below is exact.

use nerv_core::fixed_point::round_half_even;

use crate::error::ScheduleError;
use crate::spec::EconomySpec;

#[derive(Clone, Debug)]
pub enum Curve {
    LinearVesting { total_nano: u128, cliff_days: u64, linear_days: u64, term_days: u64 },
    SubsidyDecline { total_nano: u128, year_one_days: u64, term_days: u64 },
    QuarterlyGeometric { total_nano: u128, quarters: u64, quarter_days: u64, ratio_num: u64, ratio_den: u64 },
    Envelope { total_nano: u128, opens_day: u64, closes_day: u64, burn_unclaimed: bool },
}

impl Curve {
    pub fn total_nano(&self) -> u128 {
        match self {
            Curve::LinearVesting { total_nano, .. }
            | Curve::SubsidyDecline { total_nano, .. }
            | Curve::QuarterlyGeometric { total_nano, .. }
            | Curve::Envelope { total_nano, .. } => *total_nano,
        }
    }

    pub fn is_deterministic(&self) -> bool {
        !matches!(self, Curve::Envelope { .. })
    }

    pub fn term_day(&self) -> u64 {
        match self {
            Curve::LinearVesting { term_days, .. } | Curve::SubsidyDecline { term_days, .. } => *term_days,
            Curve::QuarterlyGeometric { quarters, quarter_days, .. } => quarters * quarter_days,
            Curve::Envelope { closes_day, .. } => *closes_day,
        }
    }

    /// Cumulative nano-NERV emitted by end of `day`; `None` for envelopes.
    pub fn cumulative_nano(&self, day: u64) -> Result<Option<u128>, ScheduleError> {
        const I128: ScheduleError = ScheduleError::Arithmetic { op: "i128 conversion" };
        match self {
            Curve::Envelope { .. } => Ok(None),
            Curve::LinearVesting { total_nano, cliff_days, linear_days, .. } => {
                let term = cliff_days.saturating_add(*linear_days);
                if day < *cliff_days {
                    Ok(Some(0))
                } else if day >= term {
                    Ok(Some(*total_nano))
                } else {
                    let num = i128::try_from(*total_nano).map_err(|_| I128)?
                        .checked_mul((day - cliff_days) as i128)
                        .ok_or(ScheduleError::Arithmetic { op: "linear numerator" })?;
                    let q = round_half_even(num, *linear_days as u128)
                        .map_err(|_| ScheduleError::Arithmetic { op: "linear rounding" })?;
                    Ok(Some(q as u128))
                }
            }
            Curve::SubsidyDecline { total_nano, year_one_days, term_days } => {
                let (y1, t) = (*year_one_days, *term_days);
                if day >= t {
                    return Ok(Some(*total_nano));
                }
                if day <= y1 {
                    // rate r = 2·total/(y1+t); cumulative = r·day
                    let num = i128::try_from(*total_nano).map_err(|_| I128)?
                        .checked_mul(2).and_then(|v| v.checked_mul(day as i128))
                        .ok_or(ScheduleError::Arithmetic { op: "subsidy year-one numerator" })?;
                    let q = round_half_even(num, (y1 + t) as u128)
                        .map_err(|_| ScheduleError::Arithmetic { op: "subsidy rounding" })?;
                    Ok(Some(q as u128))
                } else {
                    // cumulative = total·(2·y1·l + 2·x·l − x²) / ((y1+t)·l), x = day−y1, l = t−y1
                    let l = t - y1;
                    let x = day - y1;
                    let inner = (2u128 * y1 as u128 * l as u128)
                        .checked_add(2u128 * x as u128 * l as u128)
                        .and_then(|v| v.checked_sub(x as u128 * x as u128))
                        .ok_or(ScheduleError::Arithmetic { op: "subsidy numerator" })?;
                    let num = i128::try_from(total_nano.checked_mul(inner)
                        .ok_or(ScheduleError::Arithmetic { op: "subsidy numerator" })?)
                        .map_err(|_| I128)?;
                    let den = (y1 as u128 + t as u128) * l as u128;
                    let q = round_half_even(num, den)
                        .map_err(|_| ScheduleError::Arithmetic { op: "subsidy rounding" })?;
                    Ok(Some(q as u128))
                }
            }
            Curve::QuarterlyGeometric { total_nano, quarters, quarter_days, ratio_num, ratio_den } => {
                // Paid at quarter START (day 0 is inside quarter 0, already paid).
                let q = (day / quarter_days).saturating_add(1).min(*quarters);
                // cumulative(q) = total · rd^(Q−q) · (rd^q − rn^q) / (rd^Q − rn^Q)
                let rd = *ratio_den as u128;
                let rn = *ratio_num as u128;
                let qn = *quarters as u32;
                let rdp = |e: u32| rd.checked_pow(e).ok_or(ScheduleError::Arithmetic { op: "geometric power" });
                let rnp = |e: u32| rn.checked_pow(e).ok_or(ScheduleError::Arithmetic { op: "geometric power" });
                let den = rdp(qn)?.checked_sub(rnp(qn)?)
                    .ok_or(ScheduleError::Arithmetic { op: "geometric denominator" })?;
                let num = total_nano.checked_mul(rdp(qn - q as u32)?)
                    .and_then(|v| v.checked_mul(rdp(q as u32)?.checked_sub(rnp(q as u32)?)?))
                    .ok_or(ScheduleError::Arithmetic { op: "geometric numerator" })?;
                let num = i128::try_from(num).map_err(|_| I128)?;
                let qv = round_half_even(num, den)
                    .map_err(|_| ScheduleError::Arithmetic { op: "geometric rounding" })?;
                Ok(Some(qv as u128))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BucketSchedule {
    pub name: String,
    pub curve: Curve,
}

#[derive(Clone, Debug)]
pub struct EmissionSchedule {
    buckets: Vec<BucketSchedule>,
    supply_nano: u128,
}

impl EmissionSchedule {
    pub fn from_economy(e: &EconomySpec, supply_nerv: u64, nano_per_nerv: u64) -> Result<EmissionSchedule, ScheduleError> {
        let nano = nano_per_nerv as u128;
        let mut buckets = Vec::with_capacity(e.buckets.len());
        for bk in &e.buckets {
            let total_nano = bk.total_nerv as u128 * nano;
            let curve = match bk.kind.as_str() {
                "linear-vesting" => {
                    let cliff = bk.cliff_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing cliff_days" })?;
                    let linear = bk.linear_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing linear_days" })?;
                    if linear == 0 {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "linear_days must be ≥ 1" });
                    }
                    if cliff.checked_add(linear) != Some(bk.term_days) {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "cliff_days + linear_days != term_days" });
                    }
                    Curve::LinearVesting { total_nano, cliff_days: cliff, linear_days: linear, term_days: bk.term_days }
                }
                "subsidy-decline" => {
                    let y1 = bk.year_one_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing year_one_days" })?;
                    if y1 == 0 || y1 >= bk.term_days {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "requires 1 ≤ year_one_days < term_days" });
                    }
                    Curve::SubsidyDecline { total_nano, year_one_days: y1, term_days: bk.term_days }
                }
                "claim-window" => {
                    let w = bk.window_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing window_days" })?;
                    if w == 0 || w > bk.term_days {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "requires 1 ≤ window_days ≤ term_days" });
                    }
                    Curve::Envelope { total_nano, opens_day: 0, closes_day: w, burn_unclaimed: bk.burn_unclaimed.unwrap_or(false) }
                }
                "milestone-window" => {
                    let w = bk.window_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing window_days" })?;
                    if w == 0 || w > bk.term_days {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "requires 1 ≤ window_days ≤ term_days" });
                    }
                    Curve::Envelope { total_nano, opens_day: 0, closes_day: w, burn_unclaimed: false }
                }
                "quarterly-geometric" => {
                    let q = bk.quarters.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing quarters" })?;
                    let qd = bk.quarter_days.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing quarter_days" })?;
                    let rn = bk.ratio_num.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing ratio_num" })?;
                    let rd = bk.ratio_den.ok_or(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "missing ratio_den" })?;
                    if q == 0 || qd == 0 || rn == 0 || rd <= rn {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "requires quarters ≥ 1, quarter_days ≥ 1, 0 < ratio_num < ratio_den" });
                    }
                    if q.checked_mul(qd) != Some(bk.term_days) {
                        return Err(ScheduleError::Malformed { bucket: bk.name.clone(), reason: "quarters × quarter_days != term_days" });
                    }
                    (rd as u128).checked_pow(q as u32)
                        .ok_or(ScheduleError::Arithmetic { op: "ratio_den^quarters exceeds u128" })?;
                    Curve::QuarterlyGeometric { total_nano, quarters: q, quarter_days: qd, ratio_num: rn, ratio_den: rd }
                }
                other => {
                    return Err(ScheduleError::UnknownKind { bucket: bk.name.clone(), kind: other.to_string() })
                }
            };
            buckets.push(BucketSchedule { name: bk.name.clone(), curve });
        }
        let supply_nano = supply_nerv as u128 * nano;
        let sum: u128 = buckets.iter().map(|b| b.curve.total_nano()).sum();
        if sum != supply_nano {
            return Err(ScheduleError::SupplyMismatch { sum, expected: supply_nano });
        }
        Ok(EmissionSchedule { buckets, supply_nano })
    }

    pub fn buckets(&self) -> &[BucketSchedule] {
        &self.buckets
    }

    pub fn bucket(&self, name: &str) -> Option<&BucketSchedule> {
        self.buckets.iter().find(|b| b.name == name)
    }

    pub fn supply_nano(&self) -> u128 {
        self.supply_nano
    }

    pub fn cumulative_nano(&self, bucket: &str, day: u64) -> Result<Option<u128>, ScheduleError> {
        let b = self.bucket(bucket)
            .ok_or_else(|| ScheduleError::Malformed { bucket: bucket.to_string(), reason: "no such bucket" })?;
        b.curve.cumulative_nano(day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::LoadedSpec;

    fn real() -> EmissionSchedule {
        let path = crate::util::locate(crate::SPEC_FILE).expect("spec file");
        let text = std::fs::read_to_string(path).expect("read spec");
        let s = LoadedSpec::from_str(&text).expect("valid spec");
        EmissionSchedule::from_economy(&s.spec.economy, s.spec.protocol.supply_nerv, s.spec.protocol.nano_per_nerv).expect("schedule")
    }

    fn real_econ() -> crate::spec::EconomySpec {
        let path = crate::util::locate(crate::SPEC_FILE).expect("spec file");
        let text = std::fs::read_to_string(path).expect("read spec");
        LoadedSpec::from_str(&text).expect("valid spec").spec.economy.clone()
    }

    const NANO: u128 = 1_000_000_000;

    #[test]
    fn totals_exact_at_and_beyond_term() {
        let s = real();
        for b in s.buckets() {
            if !b.curve.is_deterministic() {
                continue;
            }
            let total = b.curve.total_nano();
            assert_eq!(b.curve.cumulative_nano(b.curve.term_day()).unwrap(), Some(total), "{}", b.name);
            assert_eq!(b.curve.cumulative_nano(u64::MAX / 2).unwrap(), Some(total), "{}", b.name);
        }
        assert_eq!(s.supply_nano(), 10_000_000_000 * NANO);
    }

    #[test]
    fn deterministic_curves_monotone() {
        let s = real();
        for b in s.buckets() {
            if !b.curve.is_deterministic() {
                continue;
            }
            let mut prev = 0u128;
            for day in 0..=b.curve.term_day() {
                let cur = b.curve.cumulative_nano(day).unwrap().unwrap();
                assert!(cur >= prev, "{} not monotone at day {day}", b.name);
                prev = cur;
            }
            assert_eq!(prev, b.curve.total_nano(), "{}", b.name);
        }
    }

    #[test]
    fn contributor_checkpoints() {
        let s = real();
        let c = |d: u64| s.cumulative_nano("early-contributors", d).unwrap().unwrap();
        assert_eq!(c(179), 0);
        assert_eq!(c(180), 0);
        // 2.0B × 180/1260 nano, remainder 360 < half-divisor → rounds down
        assert_eq!(c(360), 285_714_285_714_285_714);
        assert_eq!(c(1440), 2_000_000_000 * NANO);
    }

    #[test]
    fn founder_checkpoints() {
        let s = real();
        let c = |d: u64| s.cumulative_nano("founder", d).unwrap().unwrap();
        assert_eq!(c(360), 0);
        assert_eq!(c(720), 150_000_000 * NANO); // 0.6B × 360/1440, exact
        assert_eq!(c(1800), 600_000_000 * NANO);
    }

    #[test]
    fn validator_checkpoints() {
        let s = real();
        let c = |d: u64| s.cumulative_nano("validator-subsidy", d).unwrap().unwrap();
        assert_eq!(c(0), 0);
        // 2·3B·360/3960 nano: quotient 545454545454545454, remainder 2160, 2r > den → up
        assert_eq!(c(360), 545_454_545_454_545_455);
        assert_eq!(c(3600), 3_000_000_000 * NANO);
    }

    #[test]
    fn foundation_quarterly_geometric() {
        let s = real();
        let f = s.bucket("foundation").unwrap();
        let c = |d: u64| f.curve.cumulative_nano(d).unwrap().unwrap();
        assert_eq!(c(0), c(89)); // both inside quarter 0
        assert!(c(90) > c(89));
        assert_eq!(c(1800), 900_000_000 * NANO);
        // quarters started by day d: cv[q] = cumulative after q quarters
        let cv: Vec<u128> = (0..=20u64)
            .map(|q| if q == 0 { 0 } else { c(90 * (q - 1)) })
            .collect();
        assert_eq!(cv[20], 900_000_000 * NANO);
        for q in 1..20usize {
            let dq = cv[q + 1] - cv[q];
            let dprev = cv[q] - cv[q - 1];
            // deltas decline by the ratio 9/10, within rounding (≤ 1 nano per step)
            assert!((10 * dq).abs_diff(9 * dprev) <= 20, "quarter {q}: {} vs {}", 10 * dq, 9 * dprev);
        }
    }

    #[test]
    fn useful_work_checkpoints() {
        let s = real();
        let c = |d: u64| s.cumulative_nano("useful-work", d).unwrap().unwrap();
        assert_eq!(c(0), 0);
        assert_eq!(c(1800), 150_000_000 * NANO); // 0.3B × 1800/3600, exact
        assert_eq!(c(3600), 300_000_000 * NANO);
    }

    #[test]
    fn claim_driven_buckets_are_envelopes() {
        let s = real();
        let community = s.bucket("community").unwrap();
        match &community.curve {
            Curve::Envelope { total_nano, opens_day, closes_day, burn_unclaimed } => {
                assert_eq!(*total_nano, 2_200_000_000 * NANO);
                assert_eq!((*opens_day, *closes_day), (0, 720));
                assert!(*burn_unclaimed);
            }
            other => panic!("community must be an envelope, got {other:?}"),
        }
        let eco = s.bucket("ecosystem").unwrap();
        match &eco.curve {
            Curve::Envelope { total_nano, closes_day, burn_unclaimed, .. } => {
                assert_eq!(*total_nano, 1_000_000_000 * NANO);
                assert_eq!(*closes_day, 1800);
                assert!(!*burn_unclaimed);
            }
            other => panic!("ecosystem must be an envelope, got {other:?}"),
        }
        assert_eq!(s.cumulative_nano("community", 100).unwrap(), None);
    }

    #[test]
    fn malformed_economies_rejected() {
        let supply = 10_000_000_000u64;
        let nano = 1_000_000_000u64;

        let mut e = real_econ();
        e.buckets[0].kind = "bogus".into();
        assert!(matches!(
            EmissionSchedule::from_economy(&e, supply, nano),
            Err(ScheduleError::UnknownKind { .. })
        ));

        let mut e = real_econ();
        e.buckets[0].linear_days = None;
        assert!(matches!(
            EmissionSchedule::from_economy(&e, supply, nano),
            Err(ScheduleError::Malformed { .. })
        ));

        let mut e = real_econ();
        e.buckets[5].ratio_den = e.buckets[5].ratio_num;
        assert!(matches!(
            EmissionSchedule::from_economy(&e, supply, nano),
            Err(ScheduleError::Malformed { .. })
        ));

        let mut e = real_econ();
        e.buckets[6].total_nerv += 1;
        assert!(matches!(
            EmissionSchedule::from_economy(&e, supply, nano),
            Err(ScheduleError::SupplyMismatch { .. })
        ));
    }
}

