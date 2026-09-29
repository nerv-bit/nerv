//! Fees (WP §12.5, App D.4; erratum 161).

use std::collections::VecDeque;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::params::{
    FEES_DYNAMIC_FLOOR_BASE_FLOOR_NANO, FEES_DYNAMIC_FLOOR_DECAY_DEN, FEES_DYNAMIC_FLOOR_DECAY_NUM,
    FEES_DYNAMIC_FLOOR_M_MAX,
};

pub const PRODUCER_PERMILLE: u64 = nerv_core::params::FEES_SPLIT_PRODUCER_PERMILLE;
pub const PROVER_PERMILLE: u64 = nerv_core::params::FEES_SPLIT_PROVER_PERMILLE;
pub const DA_PERMILLE: u64 = nerv_core::params::FEES_SPLIT_DA_PERMILLE;
pub const RELAY_PERMILLE: u64 = nerv_core::params::FEES_SPLIT_RELAY_PERMILLE;

const _: () = assert!(
    PRODUCER_PERMILLE + PROVER_PERMILLE + DA_PERMILLE + RELAY_PERMILLE == 1000,
    "the 40/30/20/10 split (§12.5)"
);

pub const M_MAX: u64 = FEES_DYNAMIC_FLOOR_M_MAX;
pub const BASE_FLOOR_NANO: u64 = FEES_DYNAMIC_FLOOR_BASE_FLOOR_NANO;
/// The trailing reveal window (genesis-config; erratum 161).
pub const WINDOW: usize = 64;
/// The trigger multiple over the window median (genesis-config).
pub const TRIGGER_MULT: u128 = 4;

const _: () = assert!(M_MAX >= 1);
const _: () = assert!(FEES_DYNAMIC_FLOOR_DECAY_NUM <= FEES_DYNAMIC_FLOOR_DECAY_DEN);
const _: () = assert!(FEES_DYNAMIC_FLOOR_DECAY_DEN > 0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeeSplit {
    pub producer_nano: u64,
    pub prover_nano: u64,
    pub da_nano: u64,
    pub relay_nano: u64,
}

impl FeeSplit {
    /// The exact split: the three smaller shares floor-divided at their
    /// permilles, the dust to the producer — no nano lost or minted
    /// (erratum 161).
    pub fn split(fee_nano: u64) -> FeeSplit {
        let f = u128::from(fee_nano);
        let prover = (f * PROVER_PERMILLE as u128 / 1000) as u64;
        let da = (f * DA_PERMILLE as u128 / 1000) as u64;
        let relay = (f * RELAY_PERMILLE as u128 / 1000) as u64;
        FeeSplit {
            producer_nano: fee_nano - prover - da - relay,
            prover_nano: prover,
            da_nano: da,
            relay_nano: relay,
        }
    }

    pub fn total_nano(&self) -> u64 {
        self.producer_nano + self.prover_nano + self.da_nano + self.relay_nano
    }
}

/// Σ_j |centerlift(Δ_B[j])| over the header-carried reveal — the frozen
/// codec's rail mixture as one integer aggregate (erratum 161).
pub fn reveal_statistic(delta_b: &[u8; 512]) -> u128 {
    let mut sum = 0u128;
    for c in delta_b.chunks_exact(8) {
        let v = u64::from_le_bytes(c.try_into().expect("8-byte chunk"));
        sum += (v as i64 as i128).unsigned_abs() as u128;
    }
    sum
}

fn lower_median(window: &[u128]) -> Option<u128> {
    if window.is_empty() {
        return None;
    }
    let mut sorted = window.to_vec();
    sorted.sort_unstable();
    Some(sorted[(sorted.len() - 1) / 2])
}

/// The D.4 admission floor: m ∈ [1, M_max], escalating on anomaly,
/// decaying by λ otherwise — hysteresis is the decay itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionFloor {
    m: u64,
    window: VecDeque<u128>,
}

impl Default for AdmissionFloor {
    fn default() -> Self {
        AdmissionFloor::genesis()
    }
}

impl AdmissionFloor {
    pub fn genesis() -> AdmissionFloor {
        AdmissionFloor { m: 1, window: VecDeque::with_capacity(WINDOW) }
    }

    pub fn multiplier(&self) -> u64 {
        self.m
    }

    pub fn floor_nano(&self) -> u64 {
        BASE_FLOOR_NANO.saturating_mul(self.m)
    }

    pub fn admits(&self, declared_fee_nano: u64) -> bool {
        declared_fee_nano >= self.floor_nano()
    }

    /// One interval's observation: push the statistic, escalate or decay.
    pub fn observe_reveal(&mut self, delta_b: &[u8; 512]) {
        let stat = reveal_statistic(delta_b);
        self.window.push_back(stat);
        while self.window.len() > WINDOW {
            self.window.pop_front();
        }
        let triggered = match lower_median(&self.window.make_contiguous()) {
            Some(median) if median > 0 => stat >= TRIGGER_MULT * median,
            _ => false,
        };
        if triggered {
            self.m = (self.m + 1).min(M_MAX);
        } else {
            self.m = (self.m * FEES_DYNAMIC_FLOOR_DECAY_NUM / FEES_DYNAMIC_FLOOR_DECAY_DEN)
                .max(1);
        }
    }

    pub fn window(&mut self) -> &[u128] {
        self.window.make_contiguous()
    }
}

impl Encode for AdmissionFloor {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&(self.window.len() as u32).to_le_bytes());
        for s in &self.window {
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    fn encoded_len(&self) -> usize {
        8 + 4 + 16 * self.window.len()
    }
}

impl Decode for AdmissionFloor {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let m = r.read_u64()?;
        if m < 1 || m > M_MAX {
            return Err(CodecError::InvariantViolated("admission-floor multiplier out of range"));
        }
        let n = r.read_seq_len()?;
        if n > WINDOW {
            return Err(CodecError::SeqTooLarge { count: n, max: WINDOW });
        }
        let mut window = VecDeque::with_capacity(n);
        for _ in 0..n {
            window.push_back(u128::from_le_bytes(r.take_array::<16>()?));
        }
        Ok(AdmissionFloor { m, window })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn split_ratio(fee: u64) -> (u64, u64, u64, u64) {
        let s = FeeSplit::split(fee);
        (s.producer_nano, s.prover_nano, s.da_nano, s.relay_nano)
    }

    #[test]
    fn param_pins() {
        assert_eq!((PRODUCER_PERMILLE, PROVER_PERMILLE, DA_PERMILLE, RELAY_PERMILLE),
                   (400, 300, 200, 100));
        assert_eq!(M_MAX, 8);
        assert_eq!(BASE_FLOOR_NANO, 1000);
        assert_eq!(WINDOW, 64);
        assert_eq!(TRIGGER_MULT, 4);
    }

    #[test]
    fn split_is_exact_with_dust_to_producer() {
        // Round numbers: exact.
        assert_eq!(split_ratio(1000), (400, 300, 200, 100));
        assert_eq!(split_ratio(1_000_000), (400_000, 300_000, 200_000, 100_000));
        // Dust: 999 → 300/1000 shares floor; producer takes the 3-nano dust.
        assert_eq!(split_ratio(999), (402, 299, 199, 99));
        assert_eq!(split_ratio(1), (1, 0, 0, 0));
        assert_eq!(split_ratio(0), (0, 0, 0, 0));
        assert_eq!(split_ratio(3), (3, 0, 0, 0));
        assert_eq!(split_ratio(4), (2, 1, 0, 0), "the prover's 1.2 floors to 1");
        assert_eq!(split_ratio(5), (3, 1, 1, 0));
        assert_eq!(split_ratio(10), (5, 3, 2, 1), "the relay's 1.0 is exact");
        // Total preservation across a spread.
        for fee in [0u64, 1, 2, 7, 999, 1000, 1001, 123_456_789, u64::MAX / 4] {
            let s = FeeSplit::split(fee);
            assert_eq!(s.total_nano(), fee, "fee={fee}");
            assert!(s.producer_nano + 3 >= s.prover_nano + s.da_nano + s.relay_nano);
        }
        // The producer share is always ≥ the exact real share − 3 (dust ≤ 3).
        let f = 10u64;
        assert_eq!(FeeSplit::split(f).producer_nano, 4);
    }

    fn reveal_with(coord0: u64) -> [u8; 512] {
        let mut b = [0u8; 512];
        b[..8].copy_from_slice(&coord0.to_le_bytes());
        b
    }

    #[test]
    fn statistic_is_centerlift_l1() {
        assert_eq!(reveal_statistic(&[0u8; 512]), 0);
        assert_eq!(reveal_statistic(&reveal_with(100)), 100);
        // A wrapped negative: 2^64 − 5 centerlifts to −5.
        assert_eq!(reveal_statistic(&reveal_with((-5i64) as u64)), 5);
        assert_eq!(reveal_statistic(&reveal_with(i64::MIN as u64)), 1u128 << 63);
        // Mixed.
        let mut b = [0u8; 512];
        b[..8].copy_from_slice(&3u64.to_le_bytes());
        b[8..16].copy_from_slice(&((-4i64) as u64).to_le_bytes());
        b[16..24].copy_from_slice(&5u64.to_le_bytes());
        assert_eq!(reveal_statistic(&b), 12);
    }

    #[test]
    fn floor_escalates_decays_and_caps() {
        let mut f = AdmissionFloor::genesis();
        assert_eq!(f.multiplier(), 1);
        assert_eq!(f.floor_nano(), 1000);
        assert!(f.admits(1000) && !f.admits(999));

        // A calm window: median 100, statistic 100 — no trigger, m stays 1.
        for _ in 0..10 {
            f.observe_reveal(&reveal_with(100));
        }
        assert_eq!(f.multiplier(), 1);

        // An anomaly: statistic 401 ≥ 4·100.
        f.observe_reveal(&reveal_with(401));
        assert_eq!(f.multiplier(), 2);
        assert_eq!(f.floor_nano(), 2000);
        assert!(f.admits(2000) && !f.admits(1999));

        // Sustained anomaly escalates to the cap.
        for _ in 0..10 {
            f.observe_reveal(&reveal_with(1000));
        }
        assert_eq!(f.multiplier(), M_MAX);
        assert_eq!(f.floor_nano(), 8000);

        // The decay horizon: three quiet intervals return m to 1
        // (8→4→2→1) — D.4's hysteresis.
        f.observe_reveal(&reveal_with(0));
        assert_eq!(f.multiplier(), 4);
        f.observe_reveal(&reveal_with(0));
        assert_eq!(f.multiplier(), 2);
        f.observe_reveal(&reveal_with(0));
        assert_eq!(f.multiplier(), 1);
        f.observe_reveal(&reveal_with(0));
        assert_eq!(f.multiplier(), 1, "the floor rests at 1");

        // A zero median never triggers.
        let mut z = AdmissionFloor::genesis();
        for _ in 0..20 {
            z.observe_reveal(&reveal_with(0));
        }
        assert_eq!(z.multiplier(), 1);
        z.observe_reveal(&reveal_with(u64::MAX));
        assert_eq!(z.multiplier(), 1, "median 0: the trigger is inert");
    }

    #[test]
    fn window_is_bounded_and_codec_roundtrips() {
        let mut f = AdmissionFloor::genesis();
        for i in 0..(WINDOW as u128 + 20) {
            f.observe_reveal(&reveal_with(100 + i as u64));
        }
        assert_eq!(f.window().len(), WINDOW);
        let enc = f.encode();
        assert_eq!(enc.len(), f.encoded_len());
        let dec = AdmissionFloor::decode(&enc).unwrap();
        assert_eq!(dec, f);
        assert_eq!(dec.multiplier(), f.multiplier());
        assert!(AdmissionFloor::decode(&enc[..enc.len() - 1]).is_err());
        // Out-of-range multiplier rejected.
        let mut bad = enc.clone();
        bad[0..8].copy_from_slice(&0u64.to_le_bytes());
        assert!(AdmissionFloor::decode(&bad).is_err());
        let mut bad = enc.clone();
        bad[0..8].copy_from_slice(&(M_MAX + 1).to_le_bytes());
        assert!(AdmissionFloor::decode(&bad).is_err());
        // Over-cap window rejected.
        let mut bad = enc.clone();
        bad[8..12].copy_from_slice(&((WINDOW + 1) as u32).to_le_bytes());
        assert!(AdmissionFloor::decode(&bad).is_err());
        // Genesis roundtrips.
        assert_eq!(AdmissionFloor::decode(&AdmissionFloor::genesis().encode()).unwrap(),
                   AdmissionFloor::genesis());
    }
}


