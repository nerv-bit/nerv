//! ALL domain-separation strings in one auditable file.
//!
//! The WP's hash formulas are domain-separated by message-prefix convention
//! (cm = BLAKE3("nerv.cm" ‖ v ‖ ρ ‖ d ‖ r)). Every domain used anywhere in
//! the protocol lives here and nowhere else: a reviewer can audit the entire
//! domain-separation surface in one read, and adding a domain is a visible
//! diff to exactly one file.
//!
//! Provenance is marked per constant:
//!   [WP §x.y]        — the string appears literally in the whitepaper.
//!   [genesis-config] — introduced by the implementation where the WP leaves
//!                      the call site's domain unspecified; recorded here so
//!                      document and code never diverge silently (the DSR-5
//!                      discipline, applied from line one).
//!
//! `Domain` construction is private to this module on purpose: the only way
//! to mint a domain is to add a line to this file.

/// A domain-separation string for a hash call site.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Domain(&'static str);

impl Domain {
    const fn new(s: &'static str) -> Domain {
        Domain(s)
    }

    /// The domain string (e.g. `"nerv.cm"`).
    pub const fn as_str(&self) -> &'static str {
        self.0
    }

    /// The domain string as bytes — the prefix of the hashed message.
    pub const fn as_bytes(&self) -> &'static [u8] {
        self.0.as_bytes()
    }
}

/// Note commitments: `cm = BLAKE3("nerv.cm" ‖ v ‖ ρ ‖ d ‖ r)` [WP §3.3]
pub const NOTE_COMMITMENT: Domain = Domain::new("nerv.cm");
/// Nullifier derivation: `nf = BLAKE3-ToField("nerv.nf" ‖ nk ‖ ρ)` [WP §3.4]
pub const NULLIFIER: Domain = Domain::new("nerv.nf");
/// Address → shard-home key: `κ(addr) = BLAKE3("nerv.shard" ‖ addr)` [WP §3.2, §8.2]
pub const SHARD_HOMING: Domain = Domain::new("nerv.shard");
/// Account-slot index: `slot(addr) = BLAKE3("nerv.slot" ‖ addr) mod 224` [WP §7.2]
pub const ACCOUNT_SLOT: Domain = Domain::new("nerv.slot");
/// Composite state commitment `C_t` [WP §4.2]
pub const STATE_COMMITMENT: Domain = Domain::new("nerv.state");
/// Derived-state root `D_t` [WP §4.2]
pub const DERIVED_STATE: Domain = Domain::new("nerv.derived");
/// Note-encryption KDF context: `k = BLAKE3-KDF(ss, "nerv.note")` [WP §3.2]
pub const NOTE_KDF: Domain = Domain::new("nerv.note");
/// Transaction identity: the message is the canonical serialization of all
/// legs [WP §3.6]; the domain prefix is [genesis-config], following the WP's
/// own prefix convention, recorded here per the constants-file policy.
pub const TXID: Domain = Domain::new("nerv.txid");

/// Round-constant and IV derivation for the frozen NERV-Poseidon2-G16
/// permutation (nerv-custody::poseidon2 — the native reference, DSR-7).
pub const POSEIDON2: Domain = Domain::new("nerv.poseidon2");
/// NCT leaf compression: BLAKE3 note commitment → tree digest (E-002).
pub const NCT_LEAF: Domain = Domain::new("nerv.nct.leaf");
/// NCT internal-node compression (level ≥ 1).
pub const NCT_NODE: Domain = Domain::new("nerv.nct.node");
/// NCT empty-subtree digest derivation (E_0).
pub const NCT_EMPTY: Domain = Domain::new("nerv.nct.empty");

/// Quorum-certificate hash compression (WP §4.6): the header commits
/// QC_hash = BLAKE3("nerv.qc" ‖ canonical QC); the full certificate is body data.
pub const QUORUM_CERT: Domain = Domain::new("nerv.qc");
/// The vote message signed by committee members: "nerv.qc.vote" ‖ epoch(LE) ‖ subject.
pub const QC_VOTE: Domain = Domain::new("nerv.qc.vote");
/// Beacon-seeded hash sortition (E-007 / DSR-5): H(randomness ‖ pubkey ‖ epoch).
pub const SORTITION: Domain = Domain::new("nerv.sort");
/// Master seed → spending authority (sk_spend) expansion (WP §3.2 hierarchy).
pub const KEY_SPEND: Domain = Domain::new("nerv.key.spend");
/// Master seed → nullifier-derivation key (nk) expansion (WP §3.2; feeds §3.4).
pub const KEY_NULLIFIER: Domain = Domain::new("nerv.key.nullifier");
/// Master seed → detection/diversification seed (WP §3.2; delegating this
/// seed is viewing capability only — it derives delivery keys, never nk/sk).
pub const KEY_DIVERSIFY: Domain = Domain::new("nerv.key.diversify");
/// Detection seed → per-index diversified delivery keypair expansion.
pub const KEY_DELIVERY: Domain = Domain::new("nerv.key.delivery");

/// Nullifier tree internal-node hashing (E-002: BLAKE3 sparse Merkle depth 256).
pub const NULLIFIER_NODE: Domain = Domain::new("nerv.nullifier.node");
/// Nullifier tree leaf hashing: binds (nf, insertion height).
pub const NULLIFIER_LEAF: Domain = Domain::new("nerv.nullifier.leaf");
/// Nullifier tree empty-leaf sentinel derivation.
pub const NULLIFIER_EMPTY: Domain = Domain::new("nerv.nullifier.empty");
/// Public burn commitments (WP §3.8).
pub const BURN: Domain = Domain::new("nerv.burn");

/// Transit-log entry key (D.3): the per-leg position key H(txid ‖ shard ‖ leg_index).
pub const TRANSIT: Domain = Domain::new("nerv.transit");
/// Transit-log entry hashing (D.3): the entry record is the leaf preimage.
pub const TRANSIT_ENTRY: Domain = Domain::new("nerv.transit.entry");
/// Transit-log internal-node hashing (BLAKE3 sparse Merkle over sorted keys).
pub const TRANSIT_NODE: Domain = Domain::new("nerv.transit.node");
/// Transit-log empty-subtree derivation.
pub const TRANSIT_EMPTY: Domain = Domain::new("nerv.transit.empty");

/// `W`-commitment hashing: `Hash(W ‖ version)`, the value `params_root`
/// commits at codec adoption (WP §7.6). [genesis-config name — the WP
/// names the formula, not the domain]
pub const W_COMMIT: Domain = Domain::new("nerv.w.commit");
/// Beacon-XOF weight expansion (WP §7.6: "a fresh randomness beacon produces
/// candidate W′"). [D-02 inventory row `nerv.w.gen`]
pub const W_GEN: Domain = Domain::new("nerv.w.gen");
/// Certification sample-stream derivation: sampled spark / column–rail
/// subsets are XOF-derived from the candidate's own commitment.
/// [genesis-config: EXT-v1]
pub const W_SPARK: Domain = Domain::new("nerv.w.spark");

/// NERV-Seal public-matrix expansion (WP §6.3.1; D-02 row `nerv.seal` —
/// "seal key derivation"): A = XOF("nerv.seal", epoch seed), row-major.
pub const SEAL: Domain = Domain::new("nerv.seal");
/// NERV-Seal wallet-side short-vector derivation (r, e₁, e₂) from a
/// per-leg 32-byte seed (WP §6.3.1, §6.3.7). [genesis-config: EXT-v1]
pub const SEAL_NOISE: Domain = Domain::new("nerv.seal.noise");
/// NERV-Seal DKG public-material derivation and FS-challenge domain
/// (WP §6.3.5; D-02 row `nerv.seal.dkg`).
pub const SEAL_DKG: Domain = Domain::new("nerv.seal.dkg");
/// Verifiable-partial-decryption FS-challenge domain (WP §6.3.3; D-02 row
/// `nerv.seal.vpd`).
pub const SEAL_VPD: Domain = Domain::new("nerv.seal.vpd");
/// Rotation-record hashing (WP §6.3.6; D.1): the committable summary of an
/// epoch boundary. [genesis-config: EXT-v1]
pub const SEAL_ROTATION: Domain = Domain::new("nerv.seal.rotation");
/// The seal epoch key identifier (WP §5.1 public input): the domain for
/// H(ASeed ‖ T) binding a transaction proof to exactly one epoch key.
/// [genesis-config: EXT-v1]
pub const SEAL_STMT: Domain = Domain::new("nerv.seal.stmt");

/// Fiat–Shamir transcript root (WP §5.1, §5.3, §5.7; D-02 row `nerv.fs`):
/// the statement-11 binding challenge H(all nullifiers ‖ txid ‖ shell
/// digest) and every protocol-level FS transcript. Internal structure is
/// carried by framing and purpose tags, not sub-domains.
pub const FS: Domain = Domain::new("nerv.fs");
/// The per-address nullifier-key commitment: pk_n = BLAKE3("nerv.nf.pk" ‖
/// nk_j), published in the address and committed in the note commitment
/// (erratum 72 — the statement-2/3 ownership binding).
pub const NULLIFIER_PK: Domain = Domain::new("nerv.nf.pk");
/// STARK commitment-tree leaf hashing (chunk 12.5, the native engine):
/// leaf = BLAKE3("nerv.stark.leaf" ‖ row), row = u64-LE words in column
/// order. [genesis-config: EXT-v1]
pub const STARK_LEAF: Domain = Domain::new("nerv.stark.leaf");
/// STARK commitment-tree internal-node compression:
/// node = BLAKE3("nerv.stark.node" ‖ left ‖ right), a fixed 64-byte input.
/// [genesis-config: EXT-v1]
pub const STARK_NODE: Domain = Domain::new("nerv.stark.node");
/// Shard-header hashing (D-02's allocated `nerv.hdr` row; erratum 104):
/// the QC subject and the 𝔾_t leaf preimage.
pub const HEADER: Domain = Domain::new("nerv.hdr");
/// The block's sealed-delta batch sum: H(ct_B) (WP §4.3; erratum 100).
pub const CT_BATCH: Domain = Domain::new("nerv.state.ctb");
/// T_τ tree leaf hashing: the interval's txid-set membership tree (WP §4.3
/// rule 1; erratum 101).
pub const TAU_LEAF: Domain = Domain::new("nerv.ttau.leaf");
/// T_τ tree internal-node compression (fixed 64-byte input).
pub const TAU_NODE: Domain = Domain::new("nerv.ttau.node");
/// T_τ empty-subtree digest derivation.
pub const TAU_EMPTY: Domain = Domain::new("nerv.ttau.empty");
/// Block leg-tree leaf hashing (WP §11.2; erratum 100 — lands with
/// block.rs, chunk 13 part 2).
pub const LEG_TREE_LEAF: Domain = Domain::new("nerv.legtree.leaf");
/// Block leg-tree internal-node compression.
pub const LEG_TREE_NODE: Domain = Domain::new("nerv.legtree.node");
/// Block leg-tree empty-subtree derivation.
pub const LEG_TREE_EMPTY: Domain = Domain::new("nerv.legtree.empty");
/// Fraud-evidence digest binding (WP §4.3, §11.4; erratum 115).
pub const FRAUD: Domain = Domain::new("nerv.fraud");
/// The aggregator's bundle-attestation message (WP §5.5 tier 1):
/// domain ‖ txid_root ‖ count — the staked signature over the bundle.
pub const BUNDLE_ATTEST: Domain = Domain::new("nerv.bundle.attest");
/// The interval-commit digest (WP §5.5 tier 2) — the degraded-mode
/// committee QC's subject.
pub const REGISTRY_COMMIT: Domain = Domain::new("nerv.registry.commit");
/// Inclusion-challenge evidence digest (WP §5.5's degraded mode).
pub const REGISTRY_CHALLENGE: Domain = Domain::new("nerv.registry.challenge");
/// Interval and epoch attestations (D-02's allocated `nerv.att` row):
/// A_τ digests, epoch-attestation digests and chain links, and the
/// interval-digest Merkle tree's nodes.
pub const ATT: Domain = Domain::new("nerv.att");
/// 𝔾_t (D-02's allocated `nerv.beacon` row): the shard-header tree's
/// internal nodes and absent-shard sentinel.
pub const BEACON: Domain = Domain::new("nerv.beacon");
/// Role- and instance-separated sortition randomness derivation (EXT-v1).
pub const SORTITION_DERIVE: Domain = Domain::new("nerv.sort.derive");
/// Slash-evidence digest binding (EXT-v1).
pub const SLASH: Domain = Domain::new("nerv.slash");
/// DA cell-leaf hashing (WP §8.7; erratum 140): position-bound leaves of
/// the row/column trees.
pub const DA_CELL: Domain = Domain::new("nerv.da.cell");
/// DA Merkle-tree internal nodes (fixed 64-byte inputs).
pub const DA_NODE: Domain = Domain::new("nerv.da.node");
/// The DA set root (WP §8.7): the header's DA commitment.
pub const DA_ROOT: Domain = Domain::new("nerv.da.root");
/// The sampling position stream (erratum 141).
pub const DA_SAMPLE: Domain = Domain::new("nerv.da.sample");
/// DA fraud-evidence digest binding (erratum 142).
pub const DA_FRAUD: Domain = Domain::new("nerv.da.fraud");
/// Network peer identity: PeerId = H("nerv.net.peer" ‖ vk) (WP §6.2's
/// message-layer identity; erratum 143).
pub const NET_PEER: Domain = Domain::new("nerv.net.peer");
/// The handshake transcript hash (erratum 144).
pub const NET_HELLO: Domain = Domain::new("nerv.net.hello");
/// Session key derivation over ss_a ‖ ss_b ‖ transcript (erratum 144).
pub const NET_SESSION: Domain = Domain::new("nerv.net.session");
/// The gossip dedup digest (erratum 146).
pub const GOSSIP_MSG: Domain = Domain::new("nerv.gossip.msg");
/// Sphinx header/MAC derivation (D-02's registered row; erratum 149):
/// per-hop meta/stream keys, the replay tag, and the payload commitment.
pub const SPHINX: Domain = Domain::new("nerv.sphinx");
/// Relay-registration signatures (§6.2; erratum 150).
pub const RELAY_REG: Domain = Domain::new("nerv.relay.reg");
/// Emission ledger account identities and root (§12.3; erratum 158):
/// H("nerv.emission" ‖ LEK ‖ bucket) per account.
pub const EMISSION: Domain = Domain::new("nerv.emission");
/// Signed emission credentials: H("nerv.emission.cred" ‖ LEK ‖ bucket ‖
/// amount ‖ epoch), beacon-signed.
pub const EMISSION_CRED: Domain = Domain::new("nerv.emission.cred");
/// The emission ledger's root (erratum 159).
pub const EMISSION_ROOT: Domain = Domain::new("nerv.emission.root");
/// Emission-tree leaves: H("nerv.emission.leaf" ‖ account ‖ committed ‖
/// spendable).
pub const EMISSION_LEAF: Domain = Domain::new("nerv.emission.leaf");
/// Emission-tree internal nodes.
pub const EMISSION_NODE: Domain = Domain::new("nerv.emission.node");
/// Emission-tree empty-subtree digests.
pub const EMISSION_EMPTY: Domain = Domain::new("nerv.emission.empty");
/// Ledger key derivation: LEK = H("nerv.ek" ‖ seed) (erratum 158).
pub const EMISSION_LEK: Domain = Domain::new("nerv.ek");
/// Claim-key derivation: ck = H("nerv.ck" ‖ seed) (erratum 158).
pub const CLAIM_KEY: Domain = Domain::new("nerv.ck");
/// Commitment-note commitments: H("nerv.claim.commit" ‖ ck ‖ bucket ‖
/// amount ‖ blinding).
pub const CLAIM_COMMIT: Domain = Domain::new("nerv.claim.commit");
/// Claim nullifiers: H("nerv.claim.null" ‖ ck ‖ bucket).
pub const CLAIM_NULL: Domain = Domain::new("nerv.claim.null");
/// Claim eligibility digests: H("nerv.claim.elig" ‖ ck ‖ bucket ‖
/// amount) — the external ceremony attestation the ledger records.
pub const CLAIM_ELIG: Domain = Domain::new("nerv.claim.elig");
/// The forecaster state root (WP §4.2's forecaster_state_root_t):
/// H("nerv.derived.fs" ‖ canonical state). [EXT-v1]
pub const DERIVED_FS: Domain = Domain::new("nerv.derived.fs");
/// The §10.3 prediction commitment: H("nerv.derived.pred" ‖ Δ̂_B)
/// — committed before the reveal exists (erratum 170). [EXT-v1]
pub const DERIVED_PRED: Domain = Domain::new("nerv.derived.pred");
/// The ballot's weight-commitment matrix seed derivation (§12.8; erratum
/// 178): H("nerv.ballot.matrix" ‖ referendum_id) → ASeed.
pub const BALLOT_MATRIX: Domain = Domain::new("nerv.ballot.matrix");
/// The ballot's voting nullifier: H("nerv.ballot.null" ‖ nk ‖ referendum).
pub const BALLOT_NULL: Domain = Domain::new("nerv.ballot.null");
/// The ballot's sigma-proof FS domain (erratum 178).
pub const BALLOT_PROOF: Domain = Domain::new("nerv.ballot.proof");
/// The referendum ID derivation (§C.2; erratum 183).
pub const REFERENDUM: Domain = Domain::new("nerv.referendum");









/// The complete inventory. Grows per chunk; every addition is a visible diff
/// to this file. (The PSS handoff window length is a parameter, not a
/// domain: `nerv_core::params::SEAL_PSS_HANDOFF_INTERVALS`.)
pub const ALL: &[Domain] = &[
    NOTE_COMMITMENT,
    NULLIFIER,
    SHARD_HOMING,
    ACCOUNT_SLOT,
    STATE_COMMITMENT,
    DERIVED_STATE,
    NOTE_KDF,
    TXID,
    POSEIDON2,
    NCT_LEAF,
    NCT_NODE,
    NCT_EMPTY,
    QUORUM_CERT,
    QC_VOTE,
    SORTITION,
    KEY_SPEND,
    KEY_NULLIFIER,
    KEY_DIVERSIFY,
    KEY_DELIVERY,
    NULLIFIER_NODE,
    NULLIFIER_LEAF,
    NULLIFIER_EMPTY,
    BURN,
    TRANSIT,
    TRANSIT_ENTRY,
    TRANSIT_NODE,
    TRANSIT_EMPTY,
    W_COMMIT,
    W_GEN,
    W_SPARK,
    SEAL,
    SEAL_NOISE,
    SEAL_DKG,
    SEAL_VPD,
    SEAL_ROTATION,
    SEAL_STMT,
    FS,
    NULLIFIER_PK,
    STARK_LEAF,
    STARK_NODE,
    HEADER,
    CT_BATCH,
    TAU_LEAF,
    TAU_NODE,
    TAU_EMPTY,
    LEG_TREE_LEAF,
    LEG_TREE_NODE,
    LEG_TREE_EMPTY,
    FRAUD,
    BUNDLE_ATTEST,
    REGISTRY_COMMIT,
    REGISTRY_CHALLENGE,
    ATT,
    BEACON,
    SORTITION_DERIVE,
    SLASH,
    DA_CELL,
    DA_NODE,
    DA_ROOT,
    DA_SAMPLE,
    DA_FRAUD,
    NET_PEER,
    NET_HELLO,
    NET_SESSION,
    GOSSIP_MSG,
    SPHINX,
    RELAY_REG,
    EMISSION,
   EMISSION_CRED,
   EMISSION_ROOT,
   EMISSION_LEAF,
   EMISSION_NODE,
   EMISSION_EMPTY,
   EMISSION_LEK,
   CLAIM_KEY,
   CLAIM_COMMIT,
   CLAIM_NULL,
   CLAIM_ELIG,
   DERIVED_FS,
   DERIVED_PRED,
   BALLOT_MATRIX,
   BALLOT_NULL,
   BALLOT_PROOF,
   REFERENDUM,


];


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_domains_distinct_and_wellformed() {
        for (i, a) in ALL.iter().enumerate() {
            assert!(
                a.as_str().starts_with("nerv."),
                "domain `{}` violates the nerv.* prefix convention",
                a.as_str()
            );
            assert!(!a.as_str().contains('\u{0}'), "domain contains a NUL byte");
            for b in &ALL[i + 1..] {
                assert_ne!(a, b, "duplicate domain string `{}`", a.as_str());
            }
        }
    }

    #[test]
    fn inventory_matches_declared_constants() {
        // A new `pub const …: Domain` without an `ALL` entry fails here —
        // the inventory must stay complete.
        for d in [
            NOTE_COMMITMENT,
            NULLIFIER,
            SHARD_HOMING,
            ACCOUNT_SLOT,
            STATE_COMMITMENT,
            DERIVED_STATE,
            NOTE_KDF,
            TXID,
            POSEIDON2,
            NCT_LEAF,
            NCT_NODE,
            NCT_EMPTY,
            QUORUM_CERT,
            QC_VOTE,
            SORTITION,
            KEY_SPEND,
            KEY_NULLIFIER,
            KEY_DIVERSIFY,
            KEY_DELIVERY,
            NULLIFIER_NODE,
            NULLIFIER_LEAF,
            NULLIFIER_EMPTY,
            BURN,
            TRANSIT,
            TRANSIT_ENTRY,
            TRANSIT_NODE,
            TRANSIT_EMPTY,
            W_COMMIT,
            W_GEN,
            W_SPARK,
            SEAL,
            SEAL_NOISE,
            SEAL_DKG,
            SEAL_VPD,
            SEAL_ROTATION,
            SEAL_STMT,
            FS,
            NULLIFIER_PK,
            STARK_LEAF,
            STARK_NODE,
            HEADER,
            CT_BATCH,
            TAU_LEAF,
            TAU_NODE,
            TAU_EMPTY,
            LEG_TREE_LEAF,
            LEG_TREE_NODE,
            LEG_TREE_EMPTY,
            FRAUD,
            BUNDLE_ATTEST,
            REGISTRY_COMMIT,
            REGISTRY_CHALLENGE,
            ATT,
            BEACON,
            SORTITION_DERIVE,
            SLASH,
            DA_CELL,
            DA_NODE,
            DA_ROOT,
            DA_SAMPLE,
            DA_FRAUD,
            NET_PEER,
            NET_HELLO,
            NET_SESSION,
            GOSSIP_MSG,
            SPHINX,
            RELAY_REG,
            EMISSION,
            EMISSION_CRED,
            EMISSION_ROOT,
            EMISSION_LEAF,
            EMISSION_NODE,
            EMISSION_EMPTY,
            EMISSION_LEK,
            CLAIM_KEY,
            CLAIM_COMMIT,
            CLAIM_NULL,
            CLAIM_ELIG,
            DERIVED_FS,
            DERIVED_PRED,
            BALLOT_MATRIX,
            BALLOT_NULL,
            BALLOT_PROOF,
            REFERENDUM,
        ] {
            assert!(ALL.contains(&d), "domain `{}` missing from ALL", d.as_str());
        }
    }
}
