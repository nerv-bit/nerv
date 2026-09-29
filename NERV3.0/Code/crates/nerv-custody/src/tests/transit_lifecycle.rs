#![allow(clippy::unwrap_used, clippy::expect_used)]

use nerv_core::hash::Hash256;
use nerv_core::types::{Height, LegIndex, ShardSet, TxId};
use nerv_custody::error::CustodyError;
use nerv_custody::transit::{transit_key, TransitEntry, TransitEntryState, TransitLog};

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
    fn txid(&mut self) -> TxId {
        TxId::from_hash(Hash256::from_bytes(self.bytes32()))
    }
    fn entry(&mut self, h: u64, expiry: u64) -> TransitEntry {
        let g = ShardSet::genesis();
        TransitEntry::new_pending(
            self.txid(),
            g.ids()[(self.next() % 64) as usize],
            LegIndex::from_u8((self.next() % 256) as u8),
            Height::from_u64(h),
            Height::from_u64(expiry),
        )
    }
}

#[test]
fn cross_shard_lifecycle_with_witnesses() {
    // WP §4.5 + D.3, end to end: an issue leg enters Pending; the receiving
    // shard's log carries the entry; claim consumes it with a witness that
    // proves state; a second claim is impossible.
    let mut rng = Rng(0x7A417);
    let g = ShardSet::genesis();
    let issuing = g.ids()[7];
    let mut log = TransitLog::new();

    let entry = TransitEntry::new_pending(
        rng.txid(),
        g.ids()[40],
        LegIndex::from_u8(1),
        Height::from_u64(100),
        Height::from_u64(1000),
    );
    log.insert_pending(entry).unwrap();
    let root_after_insert = log.root();

    // a receiving-shard validator checks the issue-condition witness
    let p = log.membership_proof(&entry);
    assert!(p.verify_membership(&root_after_insert, &entry));

    // claim at height 150: state consumed
    let claimed = log.claim(&entry.key(), Height::from_u64(150)).unwrap();
    assert_eq!(claimed.state, TransitEntryState::Claimed);
    let root_after_claim = log.root();
    let pc = log.membership_proof(&claimed);
    assert!(pc.verify_membership(&root_after_claim, &claimed));
    // stale Pending proof fails against the new root
    assert!(!p.verify_membership(&root_after_claim, &entry));

    // double claim / revert: impossible
    assert!(matches!(
        log.claim(&entry.key(), Height::from_u64(151)),
        Err(CustodyError::TransitEntryConsumed { .. })
    ));
    assert!(matches!(
        log.revert(&entry.key(), Height::from_u64(10_000)),
        Err(CustodyError::TransitEntryConsumed { .. })
    ));

    // a fresh key's non-membership proves fresh transit (rule 4)
    let fresh = rng.entry(100, 2000);
    assert!(log
        .non_membership_proof(&fresh.key())
        .verify_non_membership(&root_after_claim, &fresh.key()));
    // claimed key is not "absent" — its non-membership proof fails
    assert!(!log
        .non_membership_proof(&entry.key())
        .verify_non_membership(&root_after_claim, &entry.key()));
}

#[test]
fn deterministic_reversion_flow() {
    let mut rng = Rng(0x0F1D3);
    let g = ShardSet::genesis();
    let mut log = TransitLog::new();

    let a = rng.entry(10, 500);
    let b = rng.entry(10, 600);
    log.insert_pending(a).unwrap();
    log.insert_pending(b).unwrap();

    // before the grace boundary: nothing due, early revert is an error
    assert!(log.reversion_due(Height::from_u64(509)).is_empty());
    assert!(matches!(
        log.revert(&a.key(), Height::from_u64(509)),
        Err(CustodyError::ReversionTooEarly { .. })
    ));

    // at 510: a is due (500 + 10); b is not
    let due = log.reversion_due(Height::from_u64(510));
    assert_eq!(due.len(), 1);
    assert_eq!(due[0].key(), a.key());
    let reverted = log.revert(&a.key(), Height::from_u64(510)).unwrap();
    assert_eq!(reverted.state, TransitEntryState::Reverted);

    // b reverts later
    let due2 = log.reversion_due(Height::from_u64(610));
    assert_eq!(due2.len(), 1);
    log.revert(&b.key(), Height::from_u64(700)).unwrap();

    // nothing left due; the log is fully consumed
    assert!(log.reversion_due(Height::from_u64(1_000_000)).is_empty());
    assert!(matches!(
        log.revert(&a.key(), Height::from_u64(2_000_000)),
        Err(CustodyError::TransitEntryConsumed { state: "Reverted" })
    ));
    log.validate_consistency().unwrap();
}

#[test]
fn txid_binding_of_transit_keys() {
    // the same (txid, shard, leg) always maps to the same key; any field
    // change gives a different key
    let mut rng = Rng(0x4E1B);
    let t = rng.txid();
    let g = ShardSet::genesis();
    let leg = LegIndex::from_u8(0);
    let k = transit_key(&t, &g.ids()[7], leg);
    assert_eq!(k, transit_key(&t, &g.ids()[7], leg));
    assert_ne!(k, transit_key(&t, &g.ids()[8], leg));
    assert_ne!(k, transit_key(&rng.txid(), &g.ids()[7], leg));
    assert_ne!(k, transit_key(&t, &g.ids()[7], LegIndex::from_u8(1)));

    // leg replay prevention: same key in two logs behaves identically
    let e = TransitEntry::new_pending(
        t,
        g.ids()[7],
        leg,
        Height::from_u64(1),
        Height::from_u64(10),
    );
    let mut l1 = TransitLog::new();
    let mut l2 = TransitLog::new();
    l1.insert_pending(e).unwrap();
    l2.insert_pending(e).unwrap();
    assert_eq!(l1.root(), l2.root(), "identical entries ⇒ identical roots");
}

