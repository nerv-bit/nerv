//! Integration: nerv-core's public surface consumed exactly as an external
//! crate would — params codegen reachability, the hash/field/codec pipeline,
//! topology homing, and the canonical order.

use nerv_core::constants;
use nerv_core::field::Goldilocks;
use nerv_core::fixed_point::{Mac15, Q15};
use nerv_core::hash::{hash_to_goldilocks, Hash256, Xof};
use nerv_core::params;
use nerv_core::types::{kappa, Epoch, Interval, LegIndex, LegKey, ShardSet, TxId};
use nerv_core::{Decode, Encode};

#[test]
fn params_codegen_is_reachable_externally() {
    assert_eq!(params::PROOFS_FIELD_MODULUS, 18_446_744_069_414_584_321);
    assert_eq!(params::PROTOCOL_SHARD_COUNT_GENESIS, 64);
    assert_eq!(params::TIMING_EPOCH_SECS, 86_400);
    assert_ne!(params::PARAMS_TOML_DIGEST, [0u8; 32]);
    assert_eq!(params::ECONOMY_BUCKETS.len(), 7);
    assert_eq!(params::SEAL_PSS_HANDOFF_INTERVALS, 2);
}

#[test]
fn hash_field_codec_pipeline() {
    let msg = b"integration";
    let h = Hash256::concat(&constants::TXID, msg);
    assert_ne!(h, Hash256::concat(&constants::NULLIFIER, msg));

    let v = hash_to_goldilocks(&constants::NULLIFIER, msg);
    assert!(v < params::PROOFS_FIELD_MODULUS);
    let (a, b) = (
        Goldilocks::from_u64_reduce(v),
        Goldilocks::from_u64_reduce(v.wrapping_add(1)),
    );
    let enc = (a * b).encode();
    assert_eq!(enc.len(), 8);
    assert_eq!(Goldilocks::decode(&enc), Ok(a * b));

    assert_eq!(Hash256::decode(&h.encode()), Ok(h));

    let mut x1 = Xof::new(&constants::W_GEN, h.as_bytes());
    let mut x2 = Xof::new(&constants::W_GEN, h.as_bytes());
    assert_eq!(x1.next_u64(), x2.next_u64());
}

#[test]
fn kappa_homes_into_genesis_topology() {
    let set = ShardSet::genesis();
    let addr: &[u8] = b"recipient-canonical-encoding";
    let home = set.home(addr).expect("genesis partition is total");
    assert!(home.is_kappa_home(&kappa(addr)));
    assert!(set.contains(&home));
    assert_eq!(ShardSet::decode(&set.encode()), Ok(set));
}

#[test]
fn timing_types_and_leg_order() {
    assert_eq!(Interval::from_secs(86_400).epoch(), Epoch::from_u64(1));
    let t = TxId::hash_canonical(b"legs");
    let mut keys = vec![
        LegKey::new(t, LegIndex::from_u8(3)),
        LegKey::new(t, LegIndex::from_u8(1)),
    ];
    keys.sort();
    assert!(keys[0].leg < keys[1].leg);
}

#[test]
fn mac15_canonical_delta() {
    let mut m = Mac15::new();
    m.add(Q15::from_bits(1 << 14), 3).expect("protocol-bounded");
    assert_eq!(m.resolve_u64(), 2);
}
