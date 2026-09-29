#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::hash::Hash256;
use nerv_core::params::{D3_EXPIRY_MIN_BLOCKS_T_MIN, D3_EXPIRY_MAX_BLOCKS_T_MAX};
use nerv_core::types::{FeeSats, Height, ShardSet};
use nerv_custody::error::CustodyError;
use nerv_custody::tx::{InputSet, LegShell, Output, TransactionShell};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for c in out.chunks_exact_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        out
    }
}

fn leg(seed: &mut Rng, shard: nerv_core::types::ShardId, n_in: usize, n_out: usize) -> LegShell {
    LegShell {
        shard,
        inputs: InputSet::new((0..n_in).map(|_| Hash256::from_bytes(seed.bytes32())).collect()),
        outputs: (0..n_out)
            .map(|_| Output {
                cm: Hash256::from_bytes(seed.bytes32()),
                sealed_note: vec![0xA5; 128],
                value: 100 + seed.next() % 1000,
                conditional: false,
                revert_cm: None,
            })
            .collect(),
        fee: FeeSats::from_u64(1000 + seed.next() % 1000),
        anchor: Hash256::from_bytes(seed.bytes32()),
        expiry: Height::from_u64(5000),
        weight_version: 1,
    }
}

#[test]
fn canonicalization_and_txid_stability() {
    let mut rng = Rng(0x11A);
    let g = ShardSet::genesis();
    let tx = TransactionShell {
        legs: vec![
            leg(&mut rng, g.ids()[9], 2, 2),
            leg(&mut rng, g.ids()[2], 1, 1),
            leg(&mut rng, g.ids()[30], 3, 1),
        ],
    };
    let id = tx.txid().unwrap();
    let mut reordered = tx.clone();
    reordered.legs.reverse();
    assert_eq!(reordered.txid().unwrap(), id, "leg order does not affect txid");
    // any byte change in any leg changes the txid
    let mut changed = tx.clone();
    changed.legs[0].outputs[0].sealed_note[0] ^= 1;
    assert_ne!(changed.txid().unwrap(), id);
}

#[test]
fn revert_output_pairing_enforced() {
    let mut rng = Rng(0x11B);
    let g = ShardSet::genesis();
    let mut tx = TransactionShell { legs: vec![leg(&mut rng, g.ids()[3], 2, 2)] };
    tx.legs[0].outputs[1].conditional = true;
    assert!(matches!(tx.txid(), Err(CustodyError::MissingRevertOutput)));
    tx.legs[0].outputs[1].revert_cm = Some(Hash256::from_bytes(rng.bytes32()));
    assert!(tx.txid().is_ok());
    assert!(tx.check_revert_pairing().is_ok());
    // unconditional with pairing is rejected
    tx.legs[0].outputs[0].revert_cm = Some(Hash256::from_bytes(rng.bytes32()));
    assert!(matches!(tx.txid(), Err(CustodyError::UnexpectedRevertOutput)));
}

#[test]
fn expiry_bounds_are_the_d3_range() {
    let mut rng = Rng(0x11C);
    let g = ShardSet::genesis();
    let mut tx = TransactionShell { legs: vec![leg(&mut rng, g.ids()[5], 1, 1)] };
    let h = Height::from_u64(1000);
    let lo = 1000 + D3_EXPIRY_MIN_BLOCKS_T_MIN as u64;
    let hi = 1000 + D3_EXPIRY_MAX_BLOCKS_T_MAX as u64;
    for e in [lo, hi, lo + 1, hi - 1] {
        tx.legs[0].expiry = Height::from_u64(e);
        assert!(tx.check_expiry_bounds(h).is_ok(), "expiry {e} in range");
    }
    for e in [lo - 1, hi + 1, 0, u64::MAX] {
        tx.legs[0].expiry = Height::from_u64(e);
        assert!(matches!(
            tx.check_expiry_bounds(h),
            Err(CustodyError::ExpiryOutOfBounds { .. })
        ), "expiry {e} out of range");
    }
}

#[test]
fn max_legs_enforced() {
    let mut rng = Rng(0x11D);
    let g = ShardSet::genesis();
    let legs: Vec<LegShell> = (0..256)
        .map(|i| leg(&mut rng, g.ids()[(i % 64) as usize], 1, 1))
        .collect();
    // 256 legs over 64 shards necessarily duplicates shards; use distinct
    // shards via split instead: canonicalize must reject duplicates first
    let tx = TransactionShell { legs };
    assert!(matches!(tx.canonicalize(), Err(CustodyError::DuplicateShardLeg { .. })));
    // build a genuine max-legs transaction via recursive splits
    let mut set = ShardSet::from_vec(vec![nerv_core::types::ShardId::new(1, 0).unwrap(), nerv_core::types::ShardId::new(1, 1).unwrap()]).unwrap();
    let legs: Vec<LegShell> = set
        .ids()
        .iter()
        .map(|&s| leg(&mut rng, s, 1, 1))
        .collect();
    assert_eq!(legs.len(), 2);
    let tx2 = TransactionShell { legs };
    assert!(tx2.canonicalize().is_ok());
}

#[test]
fn max_legs_boundary() {
    // 1024 ten-bit shards: 256 distinct legs is legal; 257 is not.
    let ids: Vec<nerv_core::types::ShardId> = (0..1024u16)
        .map(|v| nerv_core::types::ShardId { bits: 10, value: v })
        .collect();
    let set = ShardSet::from_vec(ids.clone()).unwrap();
    let mut rng = Rng(0x11E);
    let legs: Vec<LegShell> = (0..256)
        .map(|i| leg(&mut rng, ids[i], 1, 1))
        .collect();
    let tx = TransactionShell { legs };
    assert!(tx.canonicalize().is_ok(), "256 legs over 256 distinct shards is legal");
    let mut more = tx.clone();
    let extra = leg(&mut rng, ids[256], 1, 1);
    more.legs.push(extra);
    assert!(matches!(
        more.canonicalize(),
        Err(CustodyError::TooManyLegs { found: 257, max: 256 })
    ));
}
