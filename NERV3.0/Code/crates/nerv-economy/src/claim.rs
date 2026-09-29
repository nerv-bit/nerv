//! The claim rail (WP §12.3; erratum 160): the EMISSION tree, claim
//! witnesses, and claim-leg validation.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use nerv_core::constants::{EMISSION_EMPTY, EMISSION_LEAF, EMISSION_NODE};
use nerv_core::hash::Hash256;

use crate::emission::{
    claim_nullifier, eligibility_digest, note_commitment, AccountId, EmissionLedger, LedgerError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ClaimError {
    #[error("inclusion proof failed for account {account:?}")]
    InclusionFailed { account: AccountId },
    #[error("claim nullifier {nullifier:?} is already spent")]
    NullifierSpent { nullifier: [u8; 32] },
    #[error("declared input {declared} exceeds the grantable {spendable}")]
    OverSpend { declared: u64, spendable: u64 },
    #[error("the claim proof failed")]
    BadProof,
}

pub const EMISSION_TREE_DEPTH: usize = 32;
pub const EMISSION_TREE_CAPACITY: u64 = 1 << EMISSION_TREE_DEPTH;

pub fn emission_leaf(account: &AccountId, committed: u64, spendable: u64) -> Hash256 {
    let mut msg = Vec::with_capacity(72);
    msg.extend_from_slice(account.as_bytes());
    msg.extend_from_slice(&committed.to_le_bytes());
    msg.extend_from_slice(&spendable.to_le_bytes());
    Hash256::concat(&EMISSION_LEAF, &msg)
}

pub fn emission_node(l: &Hash256, r: &Hash256) -> Hash256 {
    let mut msg = [0u8; 64];
    msg[..32].copy_from_slice(l.as_bytes());
    msg[32..].copy_from_slice(r.as_bytes());
    Hash256::concat(&EMISSION_NODE, &msg)
}

fn empty_digests() -> &'static [Hash256; EMISSION_TREE_DEPTH + 1] {
    static TABLE: OnceLock<[Hash256; EMISSION_TREE_DEPTH + 1]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = std::array::from_fn(|_| Hash256::from_bytes([0u8; 32]));
        t[0] = Hash256::concat(&EMISSION_EMPTY, &[0u8; 32]);
        for k in 0..EMISSION_TREE_DEPTH {
            t[k + 1] = emission_node(&t[k], &t[k]);
        }
        t
    })
}

/// The depth-32 append-only tree over emission accounts in canonical
/// (map) order — the claim rail's inclusion object (erratum 160).
#[derive(Clone, Debug, Default)]
pub struct EmissionTree {
    levels: Vec<Vec<Hash256>>,
    count: u64,
}

impl EmissionTree {
    pub fn new() -> EmissionTree {
        EmissionTree::default()
    }

    pub fn from_ledger(ledger: &EmissionLedger) -> EmissionTree {
        let mut t = EmissionTree::new();
        // accounts() yields (&AccountId, &AccountHolder, &AccountEntry); the
        // holder isn't part of the leaf preimage (it lives at the wallet
        // boundary), so we deliberately ignore it here.
        for (id, _holder, entry) in ledger.accounts() {
            t.append(emission_leaf(id, entry.committed_nano, entry.spendable_nano));
        }
        t
    }

    pub fn append(&mut self, leaf: Hash256) -> u64 {
        let index = self.count;
        let mut node = leaf;
        let mut k = 0usize;
        loop {
            if k == self.levels.len() {
                self.levels.push(Vec::new());
            }
            let lvl = &mut self.levels[k];
            lvl.push(node);
            if lvl.len() % 2 == 1 {
                break;
            }
            let n = lvl.len();
            let (l, r) = (lvl[n - 2], lvl[n - 1]);
            node = emission_node(&l, &r);
            k += 1;
        }
        self.count += 1;
        index
    }

    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn root(&self) -> Hash256 {
        let empty = empty_digests();
        if self.count == 0 {
            return empty[EMISSION_TREE_DEPTH];
        }
        let mut r = empty[0];
        for k in 0..EMISSION_TREE_DEPTH {
            if (self.count >> k) & 1 == 1 {
                let left = self
                    .levels
                    .get(k)
                    .and_then(|l| l.last().copied())
                    .unwrap_or(empty[k]);
                r = emission_node(&left, &r);
            } else {
                r = emission_node(&r, &empty[k]);
            }
        }
        r
    }

   /// The leaf's position (the wallet's witness coordinate).
    pub fn position(&self, leaf: &Hash256) -> Option<u64> {
        self.levels
            .first()?
            .iter()
            .position(|l| l == leaf)
            .map(|i| i as u64)
    }

    pub fn witness(&self, index: u64) -> Option<Vec<Hash256>> {
        if index >= self.count {
            return None;
        }
        let empty = empty_digests();
        let mut siblings = Vec::with_capacity(EMISSION_TREE_DEPTH);
        let mut j = index;
        for k in 0..EMISSION_TREE_DEPTH {
            let sib = j ^ 1;
            siblings.push(
                self.levels
                    .get(k)
                    .filter(|l| (sib as usize) < l.len())
                    .and_then(|l| l.get(sib as usize))
                    .copied()
                    .unwrap_or(empty[k]),
            );
            j >>= 1;
        }
        Some(siblings)
    }
}

/// Adversarial input is rejected, never panics.
pub fn verify_emission_witness(
    root: &Hash256,
    index: u64,
    leaf: &Hash256,
    siblings: &[Hash256],
) -> bool {
    if index >= EMISSION_TREE_CAPACITY || siblings.len() > EMISSION_TREE_DEPTH {
        return false;
    }
    let empty = empty_digests();
    let mut cur = *leaf;
    let mut j = index;
    for k in 0..EMISSION_TREE_DEPTH {
        let sib = if k < siblings.len() { siblings[k] } else { empty[k] };
        cur = if j & 1 == 0 { emission_node(&cur, &sib) } else { emission_node(&sib, &cur) };
        j >>= 1;
    }
    cur == *root
}

/// The claim rail's witness: everything a claim leg carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimWitness {
    pub account: AccountId,
    pub committed_nano: u64,
    pub spendable_nano: u64,
    pub index: u64,
    pub siblings: Vec<Hash256>,
    /// The claim nullifier (H("nerv.claim.null" ‖ ck ‖ bucket)).
    pub nullifier: [u8; 32],
    /// The eligibility digest the ledger records.
    pub eligibility: [u8; 32],
}

impl ClaimWitness {
    pub fn inclusion_leaf(&self) -> Hash256 {
        emission_leaf(&self.account, self.committed_nano, self.spendable_nano)
    }

    pub fn verify_inclusion(&self, root: &Hash256) -> bool {
        verify_emission_witness(root, self.index, &self.inclusion_leaf(), &self.siblings)
    }
}

/// The claimant's derivation surface (chunk 19's wallet uses this).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimLegParts {
    pub ck: [u8; 32],
    pub bucket: &'static str,
    pub amount_nano: u64,
    pub blinding: [u8; 32],
}

impl ClaimLegParts {
    pub fn commitment(&self) -> [u8; 32] {
        note_commitment(&self.ck, self.bucket, self.amount_nano, &self.blinding)
    }

    pub fn nullifier(&self) -> [u8; 32] {
        claim_nullifier(&self.ck, self.bucket)
    }

    pub fn eligibility(&self) -> [u8; 32] {
        eligibility_digest(&self.ck, self.bucket, self.amount_nano)
    }
}

/// Validate a claim leg against the ledger and its emission tree:
/// inclusion, nullifier freshness, amount bounds. Returns the account
/// to mark. The leg's shard-side settlement (the executor's rules) is
/// the state layer's; this is the ledger-side gate.
pub fn verify_claim_leg(
    ledger: &mut EmissionLedger,
    tree: &EmissionTree,
    root: &Hash256,
    witness: &ClaimWitness,
    declared_amount_nano: u64,
) -> Result<AccountId, ClaimError> {
    if !witness.verify_inclusion(root) {
        return Err(ClaimError::InclusionFailed { account: witness.account });
    }
    if ledger.nullifiers().contains(&witness.nullifier) {
        return Err(ClaimError::NullifierSpent { nullifier: witness.nullifier });
    }
    let grantable = witness.committed_nano.saturating_sub(witness.spendable_nano);
    if declared_amount_nano > grantable {
        return Err(ClaimError::OverSpend {
            declared: declared_amount_nano,
            spendable: grantable,
        });
    }
    ledger
        .grant_claim(
            witness.account.as_bytes(),
            declared_amount_nano,
            &witness.nullifier,
            &witness.eligibility,
        )
        .map_err(|e| match e {
            LedgerError::NullifierSpent { nullifier } => {
                ClaimError::NullifierSpent { nullifier }
            }
            LedgerError::Insufficient { amount, spendable } => {
                ClaimError::OverSpend { declared: amount, spendable }
            }
            other => ClaimError::BadProof,
        })
}
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::emission::{derive_claim_key, AccountEntry, AccountHolder, NoteAccount, SignedAccount};
    use crate::schedule::EmissionSchedule;
    use crate::testutil::SplitMix64;

    fn h(seed: u64) -> Hash256 {
        Hash256::from_bytes(SplitMix64::new(seed).bytes32())
    }

    fn reference_root(leaves: &[Hash256], empty: &[Hash256; EMISSION_TREE_DEPTH + 1]) -> Hash256 {
        fn rec(k: usize, i: u64, leaves: &[Hash256], empty: &[Hash256; EMISSION_TREE_DEPTH + 1]) -> Hash256 {
            if (i << k) >= leaves.len() as u64 {
                return empty[k];
            }
            if k == 0 {
                return leaves[i as usize];
            }
            let l = rec(k - 1, 2 * i, leaves, empty);
            let r = rec(k - 1, 2 * i + 1, leaves, empty);
            emission_node(&l, &r)
        }
        rec(EMISSION_TREE_DEPTH, 0, leaves, empty)
    }

    fn leaves_of(n: u64) -> Vec<Hash256> {
        (0..n).map(|i| emission_leaf(&AccountId::from_bytes(*h(i).as_bytes()), i, i * 2)).collect()
    }

    #[test]
    fn tree_matches_reference_and_witnesses_verify() {
        let empty = *empty_digests();
        for n in [0u64, 1, 2, 3, 5, 17, 64, 100] {
            let leaves = leaves_of(n);
            let mut t = EmissionTree::new();
            for l in &leaves {
                t.append(*l);
            }
            assert_eq!(t.len(), n);
            assert_eq!(t.root(), reference_root(&leaves, &empty), "n={n}");
            for i in 0..n {
                let w = t.witness(i).unwrap();
                assert!(verify_emission_witness(&t.root(), i, &leaves[i as usize], &w), "n={n} i={i}");
                assert!(!verify_emission_witness(&t.root(), i, &leaves[((i + 1) % n) as usize], &w));
                let mut bad = w.clone();
                if !bad.is_empty() {
                    bad[0] = h(999);
                    assert!(!verify_emission_witness(&t.root(), i, &leaves[i as usize], &bad));
                }
                let mut short = w.clone();
                short.pop();
                assert!(!verify_emission_witness(&t.root(), i, &leaves[i as usize], &short));
            }
            assert!(t.witness(n).is_none());
            assert!(t.position(&h(12345)).is_none());
        }
        // The empty tree's root is E_32.
        assert_eq!(EmissionTree::new().root(), empty_digests()[EMISSION_TREE_DEPTH]);
        // position finds appended leaves.
        let mut t = EmissionTree::new();
        let l0 = leaves_of(3)[0];
        t.append(l0);
        assert_eq!(t.position(&l0), Some(0));
        assert!(!verify_emission_witness(&t.root(), EMISSION_TREE_CAPACITY, &l0, &[]));
    }

    fn note_fixture(amount: u64) -> (EmissionLedger, [u8; 32], [u8; 32], [u8; 32], AccountId) {
        let mut ledger = EmissionLedger::new();
        ledger.set_epoch(nerv_core::types::Epoch::from_u64(100));
        let ck = derive_claim_key(&[9u8; 32]);
        let commitment = note_commitment(&ck, "community", amount, &[2u8; 32]);
        let eligibility = eligibility_digest(&ck, "community", amount);
        let nullifier = claim_nullifier(&ck, "community");
        let id = ledger.add_note(NoteAccount { commitment, eligibility }, amount);
        (ledger, commitment, eligibility, nullifier, id)
    }

    fn witness_for(
        ledger: &EmissionLedger,
        tree: &EmissionTree,
        id: &AccountId,
        nullifier: &[u8; 32],
        eligibility: &[u8; 32],
    ) -> ClaimWitness {
        let (_, entry) = ledger.account(id).unwrap();
        let leaf = emission_leaf(id, entry.committed_nano, entry.spendable_nano);
        let index = tree.position(&leaf).unwrap();
        ClaimWitness {
            account: *id,
            committed_nano: entry.committed_nano,
            spendable_nano: entry.spendable_nano,
            index,
            siblings: tree.witness(index).unwrap(),
            nullifier: *nullifier,
            eligibility: *eligibility,
        }
    }

    #[test]
    fn claim_leg_happy_path_and_failures() {
        let amount = 1_000u64;
        let (mut ledger, commitment, eligibility, nullifier, id) = note_fixture(amount);
        let tree = EmissionTree::from_ledger(&ledger);
        let root = tree.root();
        let w = witness_for(&ledger, &tree, &id, &nullifier, &eligibility);
        assert!(w.verify_inclusion(&root));

        // Happy path: claim 400 of 1000.
        let claimed = verify_claim_leg(&mut ledger, &tree, &root, &w, 400).unwrap();
        assert_eq!(claimed, id);
        assert_eq!(ledger.account(&id).unwrap().1.spendable_nano, 400);
        assert!(ledger.nullifiers().contains(&nullifier));

        // Nullifier now spent.
        assert!(matches!(
            verify_claim_leg(&mut ledger, &tree, &root, &w, 100),
            Err(ClaimError::NullifierSpent { .. })
        ));

        // A fresh note: over-spend the grantable.
        let (mut l2, cm2, el2, nf2, id2) = note_fixture(100);
        let t2 = EmissionTree::from_ledger(&l2);
        let r2 = t2.root();
        let w2 = witness_for(&l2, &t2, &id2, &nf2, &el2);
        assert!(matches!(
            verify_claim_leg(&mut l2, &t2, &r2, &w2, 101),
            Err(ClaimError::OverSpend { declared: 101, spendable: 100 })
        ));

        // Inclusion failure: a tampered witness.
        let mut bad = w2.clone();
        bad.committed_nano += 1;
        assert!(matches!(
            verify_claim_leg(&mut l2, &t2, &r2, &bad, 1),
            Err(ClaimError::InclusionFailed { .. })
        ));
        // Wrong root.
        assert!(matches!(
            verify_claim_leg(&mut l2, &t2, &h(77), &w2, 1),
            Err(ClaimError::InclusionFailed { .. })
        ));
        // Stale spendable in the witness (already partially claimed).
        let (mut l3, cm3, el3, nf3, id3) = note_fixture(100);
        let mut t3 = EmissionTree::from_ledger(&l3);
        let r3 = t3.root();
        let w3 = witness_for(&l3, &t3, &id3, &nf3, &el3);
        verify_claim_leg(&mut l3, &t3, &r3, &w3, 60).unwrap();
        // The witness still says spendable=0; a second claim against the
        // STALE witness fails at the nullifier.
        assert!(matches!(
            verify_claim_leg(&mut l3, &t3, &r3, &w3, 40),
            Err(ClaimError::NullifierSpent { .. })
        ));
        let _ = (cm2, cm3, commitment);
    }

    #[test]
    fn claim_leg_parts_derivations() {
        let parts = ClaimLegParts {
            ck: derive_claim_key(&[1u8; 32]),
            bucket: "community",
            amount_nano: 500,
            blinding: [3u8; 32],
        };
        assert_eq!(parts.commitment(), note_commitment(&parts.ck, "community", 500, &[3u8; 32]));
        assert_eq!(parts.nullifier(), claim_nullifier(&parts.ck, "community"));
        assert_eq!(parts.eligibility(), eligibility_digest(&parts.ck, "community", 500));
        // A differing part changes every derivation it touches.
        let other = ClaimLegParts { amount_nano: 501, ..parts };
        assert_ne!(parts.commitment(), other.commitment());
        assert_ne!(parts.eligibility(), other.eligibility());
        assert_eq!(parts.nullifier(), other.nullifier(), "the nullifier is amount-blind");
    }

    #[test]
    fn from_ledger_over_mixed_accounts() {
        let mut ledger = EmissionLedger::new();
        let lk = crate::emission::derive_lek(&[4u8; 32]);
        ledger.add_signed(lk, "founder", *crate::emission::test_vk(1));
        let ck = derive_claim_key(&[5u8; 32]);
        let cm = note_commitment(&ck, "community", 9, &[0u8; 32]);
        let id2 = ledger.add_note(
            NoteAccount { commitment: cm, eligibility: eligibility_digest(&ck, "community", 9) },
            9,
        );
        let tree = EmissionTree::from_ledger(&ledger);
        assert_eq!(tree.len(), 2);
        let leaves: Vec<Hash256> = ledger
            .accounts()
            .map(|(id, _, e)| emission_leaf(id, e.committed_nano, e.spendable_nano))
            .collect();
        assert_eq!(tree.root(), reference_root(&leaves, empty_digests()));
        // Both accounts witness-verify.
        for (id, _, _) in ledger.accounts() {
            let leaf = leaves
                .iter()
                .find(|l| tree.position(l).is_some() && {
                    let p = tree.position(l).unwrap();
                    let w = tree.witness(p).unwrap();
                    verify_emission_witness(&tree.root(), p, l, &w)
                })
                .unwrap();
            let _ = (id, leaf);
        }
        // Direct: each leaf verifies at its position.
        for (i, l) in leaves.iter().enumerate() {
            let w = tree.witness(i as u64).unwrap();
            assert!(verify_emission_witness(&tree.root(), i as u64, l, &w));
        }
        let _ = id2;
    }
}