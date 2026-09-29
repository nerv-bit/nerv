//! The protocol-level Fiat–Shamir layer (WP §5.1–§5.3, §5.7; D-02 row
//! `nerv.fs`) — the single FS surface the transaction STARK and every
//! downstream proof composes through.
//!
//! Design (frozen; erratum 59):
//! * One domain, `nerv.fs`. Internal structure is carried by length-framed
//!   parts and per-call purpose tags — never by domain proliferation.
//! * The transcript is a 32-byte BLAKE3 chaining state rolled by EVERY
//!   operation (absorb and squeeze alike) under a monotonic operation
//!   counter: `state ← H("nerv.fs" ‖ state ‖ op ‖ part)`. Squeezes re-key
//!   the state, so the transcript is a strict function of the
//!   absorb/squeeze SEQUENCE — the verifier replays the prover's exact
//!   operation order. FS-in-the-QROM (§5.7's named assumption) is
//!   instantiated by exactly this object.
//! * Field-element challenges are rejection-sampled u64 words below the
//!   Goldilocks prime, with nerv-core's total exact-reduction fallback
//!   (the ToField convention, DSR-11: deterministic on every platform).
//! * Statement 11's binding challenge is the transcript's FIRST squeeze,
//!   over (all nullifiers in canonical leg order ‖ txid ‖ shell digest),
//!   one absorb per nullifier.
//! * The shell digest shares its preimage with the txid (the canonical leg
//!   concatenation) under a distinct FS framing — deliberately redundant
//!   binding of the full public shell, per §5.1's transcript shape.
//! * Public inputs (§5.1): the shell digest, one NCT anchor per
//!   input-bearing leg (canonical leg order; freshness-window checks are
//!   the registry's, against finalized headers), the seal epoch key
//!   identifier (H(ASeed ‖ T) under `nerv.seal.stmt`), the encoder
//!   version tag (uniform across legs — checked), and the expiry height
//!   (frozen: the earliest leg expiry; per-leg D.3 bounds are proven
//!   in-circuit against the shell).
//!
//! Backend-neutral by construction: nothing here depends on plonky3 — the
//! fs_transcripts conformance family pins this operation sequence at M1.

use nerv_core::codec::Encode;
use nerv_core::constants::FS;
use nerv_core::field::{Goldilocks, GOLDILOCKS_PRIME};
use nerv_core::hash::{Hash256, Xof};
use nerv_core::types::{ShardId, TxId};
use nerv_custody::tx::TransactionShell;
use crate::error::FsError;

const VERSION_TAG: &[u8] = b"nerv.fs.v1";
/// Purpose tag: statement-11 binding challenge.
pub const BIND_TAG: &[u8] = b"nerv.fs.bind.v1";
/// Framing tag: shell digest.
pub const SHELL_TAG: &[u8] = b"nerv.fs.shell.v1";
/// Framing tag: public-inputs digest.
pub const PUBS_TAG: &[u8] = b"nerv.fs.pubs.v1";
/// Rejection attempts per field challenge before the exact-reduction
/// fallback (total by construction).
pub const FS_REJECT_ATTEMPTS: usize = 64;

// ---------------------------------------------------------------------------
// The transcript
// ---------------------------------------------------------------------------

/// The Fiat–Shamir transcript: a BLAKE3 state chain rolled by every
/// operation. Cloneable (provers fork at bound points); deterministic and
/// platform-stable.
#[derive(Clone, Debug)]
pub struct FsTranscript {
    state: [u8; 32],
    ops: u64,
}

impl Default for FsTranscript {
    fn default() -> Self {
        FsTranscript::new()
    }
}

impl FsTranscript {
    /// A fresh transcript, seeded from the domain and version tag.
    pub fn new() -> FsTranscript {
        FsTranscript { state: *Hash256::concat(&FS, VERSION_TAG).as_bytes(), ops: 0 }
    }

    /// Absorb one part. Length-framed and op-indexed: unambiguous and
    /// position-binding.
    pub fn absorb_bytes(&mut self, part: &[u8]) {
        let op = self.ops.to_le_bytes();
        self.state = *Hash256::framed(&FS, &[&self.state, &op, part]).as_bytes();
        self.ops += 1;
    }

    pub fn absorb_hash(&mut self, h: &Hash256) {
        self.absorb_bytes(h.as_bytes());
    }

    /// Operations so far (auditing and conformance vectors).
    pub fn ops(&self) -> u64 {
        self.ops
    }

    /// Roll the state under `purpose`; the squeeze stream derives from the
    /// rolled state.
    fn roll(&mut self, purpose: &[u8]) -> Xof {
        let op = self.ops.to_le_bytes();
        self.state = *Hash256::framed(&FS, &[&self.state, &op, purpose]).as_bytes();
        self.ops += 1;
        Xof::new(&FS, &self.state)
    }

    pub fn squeeze_bytes(&mut self, purpose: &[u8], out: &mut [u8]) {
        self.roll(purpose).fill(out);
    }

    pub fn squeeze_u64(&mut self, purpose: &[u8]) -> u64 {
        self.roll(purpose).next_u64()
    }

    pub fn squeeze_hash(&mut self, purpose: &[u8]) -> Hash256 {
        Hash256::from_bytes(self.roll(purpose).read_array::<32>())
    }

    /// One uniform Goldilocks challenge.
    pub fn challenge_f(&mut self, purpose: &[u8]) -> Goldilocks {
        let mut xof = self.roll(purpose);
        sample_f(&mut xof)
    }

    /// N challenges from ONE purpose-keyed roll, rejection-sampled in
    /// sequence from the single stream. Per-call-site choice of one roll
    /// vs N rolls is frozen by the AIR layer's tag discipline.
    pub fn challenge_f_n<const N: usize>(&mut self, purpose: &[u8]) -> [Goldilocks; N] {
        let mut xof = self.roll(purpose);
        let mut out = [Goldilocks::ZERO; N];
        for e in out.iter_mut() {
            *e = sample_f(&mut xof);
        }
        out
    }
}

fn sample_f(xof: &mut Xof) -> Goldilocks {
    for _ in 0..FS_REJECT_ATTEMPTS {
        let v = xof.next_u64();
        if v < GOLDILOCKS_PRIME {
            return Goldilocks::from_u64_reduce(v);
        }
    }
    let bytes = xof.read_array::<32>();
    Goldilocks::from_u64_reduce(Hash256::from_bytes(bytes).reduce_mod(GOLDILOCKS_PRIME))
}

// ---------------------------------------------------------------------------
// The shell digest and canonical nullifiers
// ---------------------------------------------------------------------------

fn legs_bytes(canon: &TransactionShell) -> Vec<u8> {
    let mut buf = Vec::new();
    for leg in &canon.legs {
        leg.encode_into(&mut buf);
    }
    buf
}

/// The shell digest (§5.1 public input): the canonical leg concatenation
/// under the FS shell framing — the same preimage as the txid, distinctly
/// bound.
pub fn shell_digest(shell: &TransactionShell) -> Result<Hash256, FsError> {
    let legs = legs_bytes(&shell.canonicalize()?);
    Ok(Hash256::framed(&FS, &[SHELL_TAG, &legs]))
}

/// All nullifiers in canonical order — legs in canonical order, published
/// order within each leg (statement 11's "all nullifiers").
pub fn canonical_nullifiers(shell: &TransactionShell) -> Result<Vec<Hash256>, FsError> {
    let canon = shell.canonicalize()?;
    Ok(canon.legs.iter().flat_map(|leg| leg.inputs.nullifiers.iter().copied()).collect())
}

// ---------------------------------------------------------------------------
// Statement 11: the binding
// ---------------------------------------------------------------------------

/// A fresh transcript with (nullifiers ‖ txid ‖ shell digest) absorbed and
/// the binding challenge as its first squeeze. The returned transcript
/// continues the same chain — the transaction STARK proves over it.
pub fn bind(
    nullifiers: &[Hash256],
    txid: &TxId,
    shell_digest: &Hash256,
) -> (FsTranscript, Hash256) {
    let mut t = FsTranscript::new();
    for nf in nullifiers {
        t.absorb_hash(nf);
    }
    t.absorb_bytes(txid.as_bytes());
    t.absorb_hash(shell_digest);
    let binding = t.squeeze_hash(BIND_TAG);
    (t, binding)
}

/// The complete transaction-proof transcript setup (§5.1–§5.2): the
/// statement-11 binding, then the public-inputs digest. Returns the bound
/// transcript, the binding challenge, and the public-inputs digest.
pub fn bind_transaction(
    nullifiers: &[Hash256],
    txid: &TxId,
    shell_digest: &Hash256,
    public: &TxPublicInputs,
) -> (FsTranscript, Hash256, Hash256) {
    let (mut t, binding) = bind(nullifiers, txid, shell_digest);
    let pubs_digest = public.digest();
    public.absorb_into(&mut t);
    (t, binding, pubs_digest)
}

// ---------------------------------------------------------------------------
// Public inputs (§5.1)
// ---------------------------------------------------------------------------

/// The transaction proof's public inputs: the shell digest, one NCT anchor
/// per input-bearing leg, the seal epoch key identifier, the encoder
/// version tag, and the expiry height.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxPublicInputs {
    pub shell_digest: Hash256,
    /// (shard, NCT root) per input-bearing leg, canonical leg order.
    pub anchors: Vec<(ShardId, Hash256)>,
    /// H(ASeed ‖ T) under `nerv.seal.stmt`
    /// (`nerv_seal::circuit_stmt::epoch_key_identifier`).
    pub epoch_key_id: Hash256,
    /// The frozen W epoch tag — uniform across all legs (checked).
    pub encoder_version: u64,
    /// The transaction's expiry height: the earliest leg expiry.
    pub expiry: u64,
}

impl TxPublicInputs {
    /// Derives the public inputs from a canonicalizable shell plus the
    /// epoch key identifier.
    pub fn derive(
        shell: &TransactionShell,
        epoch_key_id: Hash256,
    ) -> Result<TxPublicInputs, FsError> {
        let canon = shell.canonicalize()?;
        let shell_digest = Hash256::framed(&FS, &[SHELL_TAG, &legs_bytes(&canon)]);
        let mut anchors = Vec::new();
        for leg in &canon.legs {
            if !leg.inputs.nullifiers.is_empty() {
                anchors.push((leg.shard, leg.anchor));
            }
        }
        let encoder_version = canon.legs[0].weight_version;
        for leg in canon.legs.iter().skip(1) {
            if leg.weight_version != encoder_version {
                return Err(FsError::WeightVersionMismatch {
                    leg: leg.weight_version,
                    tx: encoder_version,
                });
            }
        }
        // canonicalize() guarantees at least one leg.
        let expiry = canon.legs.iter().map(|l| l.expiry.as_u64()).min().unwrap_or(u64::MAX);
        Ok(TxPublicInputs { shell_digest, anchors, epoch_key_id, encoder_version, expiry })
    }

    /// The public-inputs digest: canonical framing of all five §5.1 inputs
    /// (anchors carry shard tags and count).
    pub fn digest(&self) -> Hash256 {
        let mut body = Vec::new();
        body.extend_from_slice(self.shell_digest.as_bytes());
        body.extend_from_slice(&(self.anchors.len() as u32).to_le_bytes());
        for (shard, root) in &self.anchors {
            body.extend_from_slice(&shard.encode());
            body.extend_from_slice(root.as_bytes());
        }
        body.extend_from_slice(self.epoch_key_id.as_bytes());
        body.extend_from_slice(&self.encoder_version.to_le_bytes());
        body.extend_from_slice(&self.expiry.to_le_bytes());
        Hash256::framed(&FS, &[PUBS_TAG, &body])
    }

    /// Absorb the public-inputs digest into a bound transcript.
    pub fn absorb_into(&self, t: &mut FsTranscript) {
        t.absorb_hash(&self.digest());
    }

    /// Full consistency against the shell: anchors (count, order, shards,
    /// roots), expiry, encoder version, and the shell digest. The
    /// registry-side and wallet-side pre-proof check.
    pub fn check_against_shell(&self, shell: &TransactionShell) -> Result<(), FsError> {
        let expect = TxPublicInputs::derive(shell, self.epoch_key_id)?;
        if self.anchors.len() != expect.anchors.len() {
            return Err(FsError::AnchorCount {
                expected: expect.anchors.len(),
                found: self.anchors.len(),
            });
        }
        for (i, (got, want)) in self.anchors.iter().zip(expect.anchors.iter()).enumerate() {
            if got != want {
                return Err(FsError::AnchorMismatch { index: i });
            }
        }
        if self.expiry != expect.expiry {
            return Err(FsError::ExpiryMismatch { expected: expect.expiry, found: self.expiry });
        }
        if self.encoder_version != expect.encoder_version {
            return Err(FsError::WeightVersionMismatch {
                leg: expect.encoder_version,
                tx: self.encoder_version,
            });
        }
        if self.shell_digest != expect.shell_digest {
            return Err(FsError::ShellDigestMismatch);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use nerv_core::types::{FeeSats, Height, ShardSet};
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use proptest::prelude::*;

    fn leg(r: &mut SplitMix64, shard: ShardId, n_in: usize, n_out: usize, version: u64) -> LegShell {
        LegShell {
            shard,
            inputs: InputSet::new((0..n_in).map(|_| Hash256::from_bytes(r.bytes32())).collect()),
            outputs: (0..n_out)
                .map(|_| Output {
                    cm: Hash256::from_bytes(r.bytes32()),
                    sealed_note: r.bytes(48),
                    value: 1 + r.next_u64() % 1_000_000_000,
                    conditional: false,
                    revert_cm: None,
                })
                .collect(),
            fee: FeeSats::from_u64(r.next_u64() % 1_000_000),
            anchor: Hash256::from_bytes(r.bytes32()),
            expiry: Height::from_u64(5_000 + r.next_u64() % 500),
            weight_version: version,
        }
    }

    fn shell(r: &mut SplitMix64, legs: Vec<LegShell>) -> TransactionShell {
        TransactionShell { legs }
    }

    #[test]
    fn chain_is_deterministic_and_position_bound() {
        let mut a = FsTranscript::new();
        let mut b = FsTranscript::new();
        a.absorb_bytes(b"x");
        b.absorb_bytes(b"x");
        a.absorb_bytes(b"y");
        b.absorb_bytes(b"y");
        assert_eq!(a.squeeze_hash(b"t"), b.squeeze_hash(b"t"));
        assert_eq!(a.ops(), b.ops());

        // Order matters.
        let mut c = FsTranscript::new();
        c.absorb_bytes(b"y");
        c.absorb_bytes(b"x");
        assert_ne!(a.squeeze_hash(b"t"), c.squeeze_hash(b"t"));

        // Framing matters: absorb("xy") ≠ absorb("x");absorb("y").
        let mut d = FsTranscript::new();
        d.absorb_bytes(b"xy");
        assert_ne!(a.squeeze_hash(b"t"), d.squeeze_hash(b"t"));

        // Empty parts are legal and distinct ops.
        let mut e = FsTranscript::new();
        e.absorb_bytes(b"");
        e.absorb_bytes(b"");
        assert_eq!(e.ops(), 2);
        assert_ne!(e.squeeze_hash(b"t"), FsTranscript::new().squeeze_hash(b"t"));
    }

    #[test]
    fn squeeze_rekeys_the_state() {
        let mut with = FsTranscript::new();
        let mut without = FsTranscript::new();
        with.absorb_bytes(b"seed");
        without.absorb_bytes(b"seed");
        let _ = with.squeeze_u64(b"mid");
        with.absorb_bytes(b"tail");
        without.absorb_bytes(b"tail");
        assert_ne!(with.squeeze_hash(b"final"), without.squeeze_hash(b"final"));
        assert_eq!(with.ops(), 4);
        assert_eq!(without.ops(), 3);
    }

    #[test]
    fn transcript_chain_pin_against_raw_blake3() {
        // Independent reconstruction of the entire chain with the blake3
        // crate (u32-LE framed parts, plain-prefix XOF), one nullifier deep.
        let nf = Hash256::from_bytes([1u8; 32]);
        let txid = TxId::from_hash(Hash256::from_bytes([2u8; 32]));
        let sd = Hash256::from_bytes([3u8; 32]);

        fn frame(h: &mut blake3::Hasher, part: &[u8]) {
            h.update(&(part.len() as u32).to_le_bytes());
            h.update(part);
        }
        fn absorb(state: &mut [u8; 32], op: u64, part: &[u8]) {
            let mut h = blake3::Hasher::new();
            h.update(FS.as_bytes());
            frame(&mut h, state);
            frame(&mut h, &op.to_le_bytes());
            frame(&mut h, part);
            *state = *h.finalize().as_bytes();
        }

        let mut h = blake3::Hasher::new();
        h.update(FS.as_bytes());
        h.update(VERSION_TAG);
        let mut state = *h.finalize().as_bytes();
        absorb(&mut state, 0, nf.as_bytes());
        absorb(&mut state, 1, txid.as_bytes());
        absorb(&mut state, 2, sd.as_bytes());
        let mut h = blake3::Hasher::new();
        h.update(FS.as_bytes());
        frame(&mut h, &state);
        frame(&mut h, &3u64.to_le_bytes());
        frame(&mut h, BIND_TAG);
        state = *h.finalize().as_bytes();
        let mut h = blake3::Hasher::new();
        h.update(FS.as_bytes());
        h.update(&state);
        let mut reader = h.finalize_xof();
        let mut expect = [0u8; 32];
        reader.fill(&mut expect);

        let (t, binding) = bind(&[nf], &txid, &sd);
        assert_eq!(*binding.as_bytes(), expect);
        assert_eq!(t.ops(), 4);
    }

    #[test]
    fn binding_sensitivity_and_determinism() {
        let mut r = SplitMix64::new(0xF5);
        let a = Hash256::from_bytes(r.bytes32());
        let b = Hash256::from_bytes(r.bytes32());
        let txid = TxId::from_hash(Hash256::from_bytes(r.bytes32()));
        let sd1 = Hash256::from_bytes(r.bytes32());
        let sd2 = Hash256::from_bytes(r.bytes32());

        let (t1, h1) = bind(&[a, b], &txid, &sd1);
        let (t2, h2) = bind(&[a, b], &txid, &sd1);
        assert_eq!(h1, h2);
        assert_eq!(t1.squeeze_hash(b"next"), t2.squeeze_hash(b"next"));

        assert_ne!(h1, bind(&[a], &txid, &sd1).1, "nullifier count binds");
        assert_ne!(h1, bind(&[b, a], &txid, &sd1).1, "nullifier order binds");
        assert_ne!(h1, bind(&[], &txid, &sd1).1);
        assert_ne!(h1, bind(&[a, b], &TxId::from_hash(Hash256::from_bytes(r.bytes32())), &sd1).1);
        assert_ne!(h1, bind(&[a, b], &txid, &sd2).1);

        // Empty nullifier lists (issue-only transactions) still bind.
        let (_, he) = bind(&[], &txid, &sd1);
        assert_ne!(he, bind(&[], &txid, &sd2).1);
    }

    #[test]
    fn shell_digest_shares_the_txid_preimage() {
        let mut r = SplitMix64::new(0xF6);
        let g = ShardSet::genesis();
        let sh = shell(
            &mut r,
            vec![
                leg(&mut r, g.ids()[2], 2, 2, 7),
                leg(&mut r, g.ids()[40], 0, 1, 7),
            ],
        );
        let txid = sh.txid().unwrap();
        let sd = shell_digest(&sh).unwrap();
        let legs = legs_bytes(&sh.canonicalize().unwrap());
        assert_eq!(sd, Hash256::framed(&FS, &[SHELL_TAG, &legs]));
        // The preimage is the txid's, under a distinct binding: neither
        // digest equals the other, both are deterministic.
        assert_ne!(*sd.as_bytes(), *txid.as_bytes());
        assert_eq!(sd, shell_digest(&sh).unwrap());
        let mut other = sh.clone();
        other.legs[0].fee = FeeSats::from_u64(other.legs[0].fee.as_u64() + 1);
        assert_ne!(sd, shell_digest(&other).unwrap());
    }

    #[test]
    fn derive_check_and_every_mismatch() {
        let mut r = SplitMix64::new(0xF7);
        let g = ShardSet::genesis();
        let sh = shell(
            &mut r,
            vec![
                leg(&mut r, g.ids()[7], 2, 1, 4),
                leg(&mut r, g.ids()[30], 1, 2, 4), // two input-bearing legs
            ],
        );
        let key_id = Hash256::from_bytes(r.bytes32());
        let pi = TxPublicInputs::derive(&sh, key_id).unwrap();
        assert_eq!(pi.anchors.len(), 2);
        assert_eq!(pi.anchors[0].0, g.ids()[7]);
        assert_eq!(pi.encoder_version, 4);
        assert_eq!(pi.expiry, sh.canonicalize().unwrap().legs.iter().map(|l| l.expiry.as_u64()).min().unwrap());
        pi.check_against_shell(&sh).unwrap();
        assert_eq!(TxPublicInputs::derive(&sh, key_id).unwrap(), pi);

        // Every tampered field is caught.
        let mut bad = pi.clone();
        bad.anchors.pop();
        assert!(matches!(bad.check_against_shell(&sh), Err(FsError::AnchorCount { expected: 2, found: 1 })));
        let mut bad = pi.clone();
        bad.anchors[1] = (g.ids()[31], bad.anchors[1].1);
        assert!(matches!(bad.check_against_shell(&sh), Err(FsError::AnchorMismatch { index: 1 })));
        let mut bad = pi.clone();
        bad.expiry += 1;
        assert!(matches!(bad.check_against_shell(&sh), Err(FsError::ExpiryMismatch { .. })));
        let mut bad = pi.clone();
        bad.encoder_version = 5;
        assert!(matches!(bad.check_against_shell(&sh), Err(FsError::WeightVersionMismatch { .. })));
        let mut bad = pi.clone();
        bad.shell_digest = Hash256::from_bytes(r.bytes32());
        assert!(matches!(bad.check_against_shell(&sh), Err(FsError::ShellDigestMismatch)));

        // Mixed leg versions rejected at derive.
        let mixed = shell(
            &mut r,
            vec![leg(&mut r, g.ids()[7], 1, 1, 4), leg(&mut r, g.ids()[8], 1, 1, 5)],
        );
        assert!(matches!(
            TxPublicInputs::derive(&mixed, key_id),
            Err(FsError::WeightVersionMismatch { leg: 5, tx: 4 })
        ));

        // Issue-only shell: no anchors, still derives and checks.
        let issue = shell(&mut r, vec![leg(&mut r, g.ids()[9], 0, 2, 4)]);
        let pi2 = TxPublicInputs::derive(&issue, key_id).unwrap();
        assert!(pi2.anchors.is_empty());
        pi2.check_against_shell(&issue).unwrap();
    }

    #[test]
    fn public_digest_binds_every_field() {
        let mut r = SplitMix64::new(0xF8);
        let g = ShardSet::genesis();
        let sh = shell(&mut r, vec![leg(&mut r, g.ids()[3], 1, 1, 1)]);
        let key_id = Hash256::from_bytes(r.bytes32());
        let pi = TxPublicInputs::derive(&sh, key_id).unwrap();
        let d0 = pi.digest();
        assert_eq!(d0, pi.digest());

        let mut bumped = pi.clone();
        bumped.shell_digest = Hash256::from_bytes(r.bytes32());
        assert_ne!(d0, bumped.digest());
        let mut bumped = pi.clone();
        bumped.anchors[0].1 = Hash256::from_bytes(r.bytes32());
        assert_ne!(d0, bumped.digest());
        let mut bumped = pi.clone();
        bumped.anchors.push((g.ids()[4], pi.anchors[0].1));
        assert_ne!(d0, bumped.digest());
        let mut bumped = pi.clone();
        bumped.epoch_key_id = Hash256::from_bytes(r.bytes32());
        assert_ne!(d0, bumped.digest());
        let mut bumped = pi.clone();
        bumped.encoder_version += 1;
        assert_ne!(d0, bumped.digest());
        let mut bumped = pi.clone();
        bumped.expiry += 1;
        assert_ne!(d0, bumped.digest());
    }

    #[test]
    fn bind_transaction_composes_in_order() {
        let mut r = SplitMix64::new(0xF9);
        let g = ShardSet::genesis();
        let sh = shell(&mut r, vec![leg(&mut r, g.ids()[5], 2, 1, 3)]);
        let txid = sh.txid().unwrap();
        let sd = shell_digest(&sh).unwrap();
        let nfs = canonical_nullifiers(&sh).unwrap();
        assert_eq!(nfs.len(), 2);
        let pi = TxPublicInputs::derive(&sh, Hash256::from_bytes(r.bytes32())).unwrap();

        let (t, binding, pubs) = bind_transaction(&nfs, &txid, &sd, &pi);
        assert_eq!(pubs, pi.digest());
        // Same binding as the bare bind…
        let (bare, bare_binding) = bind(&nfs, &txid, &sd);
        assert_eq!(binding, bare_binding);
        // …but the chains diverge afterwards (public inputs absorbed).
        assert_ne!(t.squeeze_hash(b"stmt1"), bare.squeeze_hash(b"stmt1"));
        assert_eq!(t.ops(), bare.ops() + 1);
    }

    #[test]
    fn challenges_are_uniform_bounded_and_total() {
        let mut t = FsTranscript::new();
        let mut nonzero = 0;
        for i in 0..1_000 {
            let c = t.challenge_f(b"dist");
            assert!(c.as_u64() < GOLDILOCKS_PRIME);
            if !c.is_zero() {
                nonzero += 1;
            }
            let _ = i;
        }
        assert!(nonzero > 990);
        let arr = t.challenge_f_n::<8>(b"arr");
        for c in arr {
            assert!(c.as_u64() < GOLDILOCKS_PRIME);
        }
    }

    #[test]
    fn challenge_f_n_reads_one_stream_sequentially() {
        let mut t = FsTranscript::new();
        let got = t.challenge_f_n::<4>(b"test.n");
        // Manual replication through the same roll (child module sees the
        // private state).
        let mut t2 = FsTranscript::new();
        let mut xof = t2.roll(b"test.n");
        let mut expect = [Goldilocks::ZERO; 4];
        for e in expect.iter_mut() {
            *e = sample_f(&mut xof);
        }
        assert_eq!(got, expect);
        assert_eq!(t.state, t2.state);
        assert_eq!(t.ops, t2.ops);
    }

    proptest! {
        #[test]
        fn prop_replay_determinism(seed in any::<u64>(), n_ops in 3usize..=12) {
            let mut r = SplitMix64::new(seed);
            let ops: Vec<(bool, u64)> = (0..n_ops).map(|_| (r.next_bool(), r.next_u64())).collect();
            let run = || {
                let mut t = FsTranscript::new();
                let mut outs = Vec::new();
                for (do_absorb, x) in &ops {
                    if *do_absorb {
                        t.absorb_bytes(&x.to_le_bytes());
                    } else {
                        outs.push(t.challenge_f(b"prop"));
                    }
                }
                (outs, t.squeeze_hash(b"prop.final"))
            };
            let (a, ha) = run();
            let (b, hb) = run();
            prop_assert_eq!(a, b);
            prop_assert_eq!(ha, hb);
        }
    }
}
