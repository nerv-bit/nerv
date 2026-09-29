//! Key lifecycle (WP §6.3.5–§6.3.6; Appendix D.1; errata 50–54).
//!
//! Rotation, per epoch: (1) the incoming committee establishes a FRESH
//! key by the DKG — D.1(a)'s "verifiable resharing to fresh, statistically
//! independent randomness" (no share material, seed, or noise reused; the
//! Ajtai commitments and FS consistency proofs are the DKG's); (2) the
//! outgoing committee opens every still-pending aggregate inside the
//! bounded handoff window (D.6: 2 intervals), with ML-KEM backup delivery
//! of its exact short shares to incoming members (§6.3.6) — each backup
//! opens the public W_j, so a stand-in verifies and produces normal VPDs
//! (erratum 50: cross-committee resharing cannot preserve R_q share
//! shortness, so backups — not λ-weighted reshares — carry the handoff);
//! (3) at window close, unopened batches become missed reveals — public
//! records the knowledge layer's skip-and-carry rule consumes (§10, D.1d)
//! — and all superseded share material is erased (D.1c, operational).
//! Sub-minimum pending batches are force-padded to B_MIN with well-formed
//! zero-encryptions and revealed (§6.3.5): the padded aggregate is decoded
//! with the REAL leg count (erratum 52), so honest pads are invisible and
//! pad-plaintext injection beyond the real envelope is an invalid reveal.
//! The reveal ledger publishes the per-key reveal count — erratum 47's
//! monitorable, with the governance cap as its lever.


use std::collections::BTreeMap;
use std::fmt;
use nerv_core::constants::{SEAL_NOISE, SEAL_ROTATION};
use nerv_core::hash::{Hash256, Xof};


use crate::dkg::{
    assemble_public_key, check_share_commitment, transcript_digest, CommitMatrix, MemberPublic,
    MemberSecret, ShareSecret, COMMITTEE_SIZE, SHARE_BOUND, THRESHOLD,
};
use crate::digitize::Plaintext;
use crate::encrypt::{Ciphertext, PublicKey};
use crate::error::{DkgError, EpochError};
use crate::noise::KEY_BOUND;
use crate::ring::{Mat2x8Ntt, Mat8x8Ntt, Vec8, N};
use crate::sampling::{expand_matrix, ASeed, NoiseSeed};


/// Genesis privacy floor (params' chunk_min; equals chunk_max, erratum 40).
pub const CHUNK_MIN: u64 = nerv_core::params::SEAL_CHUNK_MIN as u64;
/// Handoff window length in beacon intervals (Appendix D.6).
pub const HANDOFF_INTERVALS: u64 = nerv_core::params::SEAL_PSS_HANDOFF_INTERVALS;
/// LWE samples published per reveal: the combined value plus t partials.
pub const SAMPLES_PER_REVEAL: u64 = (1 + THRESHOLD as u64) * N as u64;


// ---------------------------------------------------------------------------
// Epoch identity and key handle
// ---------------------------------------------------------------------------


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct EpochIndex(pub u64);


impl fmt::Display for EpochIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch {}", self.0)
    }
}


/// The epoch's committed key summary: the public key (ASeed ‖ T) and the
/// DKG transcript digest that binds every member transcript (WP §6.3.5).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EpochKeyHandle {
    pub epoch: EpochIndex,
    pub public: PublicKey,
    pub transcript_digest: [u8; 32],
}


impl EpochKeyHandle {
    pub const WIRE_SIZE: usize = 8 + PublicKey::WIRE_SIZE + 32;


    pub fn to_bytes(&self) -> [u8; EpochKeyHandle::WIRE_SIZE] {
        let mut out = [0u8; EpochKeyHandle::WIRE_SIZE];
        out[..8].copy_from_slice(&self.epoch.0.to_le_bytes());
        out[8..8 + PublicKey::WIRE_SIZE].copy_from_slice(&self.public.to_bytes());
        out[8 + PublicKey::WIRE_SIZE..].copy_from_slice(&self.transcript_digest);
        out
    }


    pub fn from_bytes(bytes: &[u8]) -> Result<EpochKeyHandle, EpochError> {
        if bytes.len() != EpochKeyHandle::WIRE_SIZE {
            return Err(EpochError::BadLength { len: bytes.len(), expected: EpochKeyHandle::WIRE_SIZE });
        }
        let mut eb = [0u8; 8];
        eb.copy_from_slice(&bytes[..8]);
        let public = PublicKey::from_bytes(&bytes[8..8 + PublicKey::WIRE_SIZE])?;
        let mut td = [0u8; 32];
        td.copy_from_slice(&bytes[8 + PublicKey::WIRE_SIZE..]);
        Ok(EpochKeyHandle { epoch: EpochIndex(u64::from_le_bytes(eb)), public, transcript_digest: td })
    }
}


/// Establishes an epoch's key: generates nothing — consumes the members'
/// pre-generated secrets and proof seeds, builds and verifies every
/// transcript, assembles the public key, and returns the committable
/// handle. `a_seed` is the epoch's beacon-committed ASeed.
pub fn establish_epoch(
    epoch: EpochIndex,
    a_seed: &ASeed,
    members: &[(MemberSecret, [u8; 32])],
) -> Result<EpochKeyHandle, EpochError> {
    if members.len() != COMMITTEE_SIZE {
        return Err(DkgError::BadCommittee { n: members.len() }.into());
    }
    for (i, (secret, _)) in members.iter().enumerate() {
        if secret.member != (i + 1) as u8 {
            return Err(DkgError::BadMember { member: secret.member, n: COMMITTEE_SIZE }.into());
        }
    }
    let ac = CommitMatrix::expand(a_seed)?;
    let a_mat = expand_matrix(a_seed)?;
    let mut publics: Vec<MemberPublic> = Vec::with_capacity(members.len());
    for (secret, proof_seed) in members {
        publics.push(MemberPublic::build(secret, &ac, &a_mat, a_seed, proof_seed)?);
    }
    for mp in publics.iter() {
        mp.verify(a_seed, COMMITTEE_SIZE, THRESHOLD)?;
    }
    let public = assemble_public_key(&publics, a_seed, COMMITTEE_SIZE)?;
    let transcript_digest = transcript_digest(&publics);
    Ok(EpochKeyHandle { epoch, public, transcript_digest })
}


// ---------------------------------------------------------------------------
// The handoff window
// ---------------------------------------------------------------------------


#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BatchId(pub u64);


/// A pending aggregate awaiting reveal. The batch ciphertext lives with
/// the committee's batch store; the window tracks identity and size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PendingBatch {
    pub id: BatchId,
    pub legs: u64,
}


/// An aggregate unopened at handoff-window close (D.1d): public derived
/// state; the knowledge layer's skip-and-carry rule consumes it (§10).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MissedReveal {
    pub epoch: EpochIndex,
    pub batch: BatchId,
    pub legs: u64,
}


#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HandoffWindow {
    epoch: EpochIndex,
    deadline_interval: u64,
    pending: BTreeMap<u64, u64>,
    opened: Vec<BatchId>,
    missed: Vec<MissedReveal>,
    closed: bool,
}


impl HandoffWindow {
    pub fn new(
        epoch: EpochIndex,
        deadline_interval: u64,
        batches: Vec<PendingBatch>,
    ) -> Result<HandoffWindow, EpochError> {
        let mut pending = BTreeMap::new();
        for b in batches {
            if pending.insert(b.id.0, b.legs).is_some() {
                return Err(EpochError::DuplicateBatch { batch: b.id.0 });
            }
        }
        Ok(HandoffWindow {
            epoch,
            deadline_interval,
            pending,
            opened: Vec::new(),
            missed: Vec::new(),
            closed: false,
        })
    }


    pub fn deadline_interval(&self) -> u64 {
        self.deadline_interval
    }


    pub fn expired(&self, current_interval: u64) -> bool {
        current_interval >= self.deadline_interval
    }


    pub fn is_closed(&self) -> bool {
        self.closed
    }


    pub fn remaining(&self) -> usize {
        self.pending.len()
    }


    pub fn opened(&self) -> &[BatchId] {
        &self.opened
    }


    /// Records a completed reveal. Fails after close or for unknown batches.
    pub fn record_reveal(&mut self, id: BatchId) -> Result<(), EpochError> {
        if self.closed {
            return Err(EpochError::WindowClosed);
        }
        if self.pending.remove(&id.0).is_none() {
            return Err(EpochError::UnknownBatch { batch: id.0 });
        }
        self.opened.push(id);
        Ok(())
    }


    /// Finalizes the window: every still-pending batch becomes a missed
    /// reveal. Deterministic and idempotent — later calls return the same
    /// records.
    pub fn close(&mut self) -> Vec<MissedReveal> {
        if !self.closed {
            let epoch = self.epoch;
            self.missed = self
                .pending
                .iter()
                .map(|(&id, &legs)| MissedReveal { epoch, batch: BatchId(id), legs })
                .collect();
            self.pending.clear();
            self.closed = true;
        }
        self.missed.clone()
    }


    pub fn missed(&self) -> &[MissedReveal] {
        &self.missed
    }
}


// ---------------------------------------------------------------------------
// Forced zero-padding (§6.3.5)
// ---------------------------------------------------------------------------


/// Pads a sub-minimum pending batch to CHUNK_MIN ciphertexts with
/// well-formed encryptions of zero — public-key operations, so any
/// committee member can generate them; deterministic in the ceremony
/// seed for conformance. Decode the padded aggregate with the REAL leg
/// count (erratum 52).
pub fn pad_batch(
    a_ntt: &Mat8x8Ntt,
    t_ntt: &Mat2x8Ntt,
    real: &Ciphertext,
    real_legs: u64,
    ceremony_seed: &[u8; 32],
) -> Result<Ciphertext, EpochError> {
    if real_legs == 0 || real_legs >= CHUNK_MIN {
        return Err(EpochError::PadRange { legs: real_legs, min: CHUNK_MIN });
    }
    let zero = Plaintext::zero();
    let mut out = real.clone();
    for i in 0..(CHUNK_MIN - real_legs) {
        let mut xof = Xof::framed(
            &SEAL_NOISE,
            &[b"nerv.seal.pad", ceremony_seed, &i.to_le_bytes()],
        );
        let nb = xof.read_array::<32>();
        out = out.add(&Ciphertext::encrypt_cached(a_ntt, t_ntt, &NoiseSeed::from_bytes(nb), &zero)?);
    }
    Ok(out)
}


// ---------------------------------------------------------------------------
// Backup delivery (§6.3.6; erratum 53)
// ---------------------------------------------------------------------------


/// The private backup payload: member ‖ share ‖ rho, canonically.
pub const BACKUP_PAYLOAD_SIZE: usize = 1 + 2 * Vec8::WIRE_SIZE;


pub fn backup_payload(share: &ShareSecret) -> [u8; BACKUP_PAYLOAD_SIZE] {
    let mut out = [0u8; BACKUP_PAYLOAD_SIZE];
    out[0] = share.member;
    out[1..1 + Vec8::WIRE_SIZE].copy_from_slice(&share.share.to_bytes());
    out[1 + Vec8::WIRE_SIZE..].copy_from_slice(&share.rho.to_bytes());
    out
}


pub fn open_backup_payload(bytes: &[u8]) -> Result<ShareSecret, EpochError> {
    if bytes.len() != BACKUP_PAYLOAD_SIZE {
        return Err(EpochError::BadLength { len: bytes.len(), expected: BACKUP_PAYLOAD_SIZE });
    }
    let member = bytes[0];
    if member == 0 || member as usize > COMMITTEE_SIZE {
        return Err(EpochError::BadPayloadMember { member, n: COMMITTEE_SIZE });
    }
    let share = Vec8::from_bytes(&bytes[1..1 + Vec8::WIRE_SIZE])?;
    let rho = Vec8::from_bytes(&bytes[1 + Vec8::WIRE_SIZE..])?;
    let sn = inf_norm_vec(&share);
    let rn = inf_norm_vec(&rho);
    if sn > SHARE_BOUND || rn > KEY_BOUND {
        return Err(EpochError::PayloadBound { share: sn, rho: rn });
    }
    Ok(ShareSecret { member, share, rho })
}


/// Verifies an opened backup against the OLD transcript's public share
/// commitment W_j — the verifiable-resharing check; mismatch is complaint
/// and fraud evidence against the sender.
pub fn verify_backup(ac: &CommitMatrix, w_j: &Vec8, share: &ShareSecret) -> bool {
    check_share_commitment(ac, w_j, share)
}


fn inf_norm_vec(v: &Vec8) -> u64 {
    v.polys()
        .iter()
        .map(|p| p.centerlift().iter().map(|x| x.unsigned_abs()).max().unwrap_or(0))
        .max()
        .unwrap_or(0)
}


/// The transport envelope. `sealed` is the ML-KEM-768 + ChaCha20-Poly1305
/// box over [`backup_payload`] produced by the delivery layer (nerv-crypto
/// via nerv-net/wallet — the two wiring call sites, erratum 53); this
/// crate owns the envelope, the payload, and the verification path.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SealedBackup {
    pub from: u8,
    pub to: u8,
    pub epoch: EpochIndex,
    pub sealed: Vec<u8>,
}


impl SealedBackup {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(18 + self.sealed.len());
        out.push(self.from);
        out.push(self.to);
        out.extend_from_slice(&self.epoch.0.to_le_bytes());
        out.extend_from_slice(&(self.sealed.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.sealed);
        out
    }


    pub fn from_bytes(bytes: &[u8]) -> Result<SealedBackup, EpochError> {
        if bytes.len() < 18 {
            return Err(EpochError::BadLength { len: bytes.len(), expected: 18 });
        }
        let from = bytes[0];
        let to = bytes[1];
        let mut eb = [0u8; 8];
        eb.copy_from_slice(&bytes[2..10]);
        let mut lb = [0u8; 8];
        lb.copy_from_slice(&bytes[10..18]);
        let len = u64::from_le_bytes(lb) as usize;
        if bytes.len() != 18 + len {
            return Err(EpochError::BadLength { len: bytes.len(), expected: 18 + len });
        }
        Ok(SealedBackup {
            from,
            to,
            epoch: EpochIndex(u64::from_le_bytes(eb)),
            sealed: bytes[18..].to_vec(),
        })
    }
}


// ---------------------------------------------------------------------------
// The reveal ledger (erratum 47/54)
// ---------------------------------------------------------------------------


/// Per-key reveal counts — the public, monitorable quantity behind the
/// leakage surface of erratum 47: each reveal publishes
/// [`SAMPLES_PER_REVEAL`] LWE samples on the key and its shares. A
/// governance cap turns the monitorable into an enforced bound; `record`
/// fails at cap and the ceremony must treat the batch as a missed reveal.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct RevealLedger {
    cap: Option<u64>,
    reveals: BTreeMap<[u8; 32], u64>,
}


impl RevealLedger {
    pub fn new(cap: Option<u64>) -> RevealLedger {
        RevealLedger { cap, reveals: BTreeMap::new() }
    }


    pub fn cap(&self) -> Option<u64> {
        self.cap
    }


    pub fn reveals(&self, key_id: &[u8; 32]) -> u64 {
        self.reveals.get(key_id).copied().unwrap_or(0)
    }


    pub fn record(&mut self, key_id: &[u8; 32]) -> Result<(), EpochError> {
        let next = self.reveals(key_id) + 1;
        if let Some(cap) = self.cap {
            if next > cap {
                return Err(EpochError::OverRevealCap { reveals: next - 1, cap });
            }
        }
        self.reveals.insert(*key_id, next);
        Ok(())
    }


    pub fn samples_exposed(&self, key_id: &[u8; 32]) -> u64 {
        self.reveals(key_id) * SAMPLES_PER_REVEAL
    }


    pub fn total_reveals(&self) -> u64 {
        self.reveals.values().sum()
    }


    pub fn total_samples_exposed(&self) -> u64 {
        self.total_reveals() * SAMPLES_PER_REVEAL
    }
}


// ---------------------------------------------------------------------------
// The rotation record
// ---------------------------------------------------------------------------


/// The committable epoch-boundary summary (the DKG transcripts themselves
/// are committed under params_root, WP §6.3.5; this record binds the
/// handoff outcome and the new key's digest).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RotationRecord {
    pub epoch_out: EpochIndex,
    pub epoch_in: EpochIndex,
    pub out_transcript_digest: [u8; 32],
    pub in_transcript_digest: [u8; 32],
    pub padded_batches: u64,
    pub opened: u64,
    pub missed: Vec<MissedReveal>,
}


impl RotationRecord {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(92 + 16 * self.missed.len());
        out.extend_from_slice(&self.epoch_out.0.to_le_bytes());
        out.extend_from_slice(&self.epoch_in.0.to_le_bytes());
        out.extend_from_slice(&self.out_transcript_digest);
        out.extend_from_slice(&self.in_transcript_digest);
        out.extend_from_slice(&self.padded_batches.to_le_bytes());
        out.extend_from_slice(&self.opened.to_le_bytes());
        out.extend_from_slice(&(self.missed.len() as u32).to_le_bytes());
        for m in &self.missed {
            out.extend_from_slice(&m.batch.0.to_le_bytes());
            out.extend_from_slice(&m.legs.to_le_bytes());
        }
        out
    }


    pub fn from_bytes(bytes: &[u8]) -> Result<RotationRecord, EpochError> {
        if bytes.len() < 92 {
            return Err(EpochError::BadLength { len: bytes.len(), expected: 92 });
        }
        let u64at = |at: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&bytes[at..at + 8]);
            u64::from_le_bytes(b)
        };
        let epoch_out = EpochIndex(u64at(0));
        let epoch_in = EpochIndex(u64at(8));
        let mut od = [0u8; 32];
        od.copy_from_slice(&bytes[16..48]);
        let mut idg = [0u8; 32];
        idg.copy_from_slice(&bytes[48..80]);
        let padded_batches = u64at(80);
        let opened = u64at(88);
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&bytes[96..100]);
        let count = u32::from_le_bytes(b4) as usize;
        if bytes.len() != 100 + 16 * count {
            return Err(EpochError::BadLength { len: bytes.len(), expected: 100 + 16 * count });
        }
        let mut missed = Vec::with_capacity(count);
        for i in 0..count {
            missed.push(MissedReveal {
                epoch: epoch_out,
                batch: BatchId(u64at(100 + 16 * i)),
                legs: u64at(108 + 16 * i),
            });
        }
        Ok(RotationRecord {
            epoch_out,
            epoch_in,
            out_transcript_digest: od,
            in_transcript_digest: idg,
            padded_batches,
            opened,
            missed,
        })
    }


    pub fn digest(&self) -> [u8; 32] {
        *Hash256::concat(&SEAL_ROTATION, &self.to_bytes()).as_bytes()
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;


    #[test]
    fn window_transitions() {
        let e = EpochIndex(7);
        let mut w = HandoffWindow::new(
            e,
            1_000,
            vec![PendingBatch { id: BatchId(1), legs: 128 }, PendingBatch { id: BatchId(2), legs: 100 }],
        )
        .unwrap();
        assert!(!w.is_closed());
        assert_eq!(w.remaining(), 2);
        assert!(!w.expired(999));
        assert!(w.expired(1_000));
        w.record_reveal(BatchId(1)).unwrap();
        assert_eq!(w.opened(), &[BatchId(1)]);
        assert_eq!(w.remaining(), 1);
        assert!(matches!(
            w.record_reveal(BatchId(9)),
            Err(EpochError::UnknownBatch { batch: 9 })
        ));
        let missed = w.close();
        assert_eq!(missed.len(), 1);
        assert_eq!(missed[0], MissedReveal { epoch: e, batch: BatchId(2), legs: 100 });
        assert!(w.is_closed());
        assert_eq!(w.remaining(), 0);
        assert_eq!(w.close(), missed);
        assert_eq!(w.missed(), &missed[..]);
        assert!(matches!(w.record_reveal(BatchId(1)), Err(EpochError::WindowClosed)));
        assert!(matches!(
            HandoffWindow::new(e, 1, vec![
                PendingBatch { id: BatchId(5), legs: 1 },
                PendingBatch { id: BatchId(5), legs: 2 },
            ]),
            Err(EpochError::DuplicateBatch { batch: 5 })
        ));
        let empty = HandoffWindow::new(e, 1, vec![]).unwrap();
        assert_eq!(empty.close(), vec![]);
    }


    #[test]
    fn ledger_counts_and_caps() {
        assert_eq!(SAMPLES_PER_REVEAL, (1 + 7) * 256);
        let mut l = RevealLedger::new(Some(2));
        let k = [7u8; 32];
        l.record(&k).unwrap();
        l.record(&k).unwrap();
        assert_eq!(l.reveals(&k), 2);
        assert!(matches!(
            l.record(&k),
            Err(EpochError::OverRevealCap { reveals: 2, cap: 2 })
        ));
        assert_eq!(l.reveals(&k), 2);
        let k2 = [8u8; 32];
        l.record(&k2).unwrap();
        assert_eq!(l.total_reveals(), 3);
        assert_eq!(l.samples_exposed(&k), 2 * SAMPLES_PER_REVEAL);
        assert_eq!(l.total_samples_exposed(), 3 * SAMPLES_PER_REVEAL);
        let uncapped = RevealLedger::new(None);
        assert_eq!(uncapped.cap(), None);
        assert_eq!(uncapped.reveals(&k), 0);
    }


    #[test]
    fn handle_and_rotation_wire_roundtrip() {
        let r = RotationRecord {
            epoch_out: EpochIndex(3),
            epoch_in: EpochIndex(4),
            out_transcript_digest: [1; 32],
            in_transcript_digest: [2; 32],
            padded_batches: 2,
            opened: 11,
            missed: vec![
                MissedReveal { epoch: EpochIndex(3), batch: BatchId(9), legs: 128 },
                MissedReveal { epoch: EpochIndex(3), batch: BatchId(10), legs: 40 },
            ],
        };
        let bytes = r.to_bytes();
        assert_eq!(RotationRecord::from_bytes(&bytes).unwrap(), r);
        assert_eq!(r.digest(), RotationRecord::from_bytes(&bytes).unwrap().digest());
        assert!(RotationRecord::from_bytes(&bytes[..99]).is_err());
        let r2 = RotationRecord {
            opened: 12,
            ..r.clone()
        };
        assert_ne!(r.digest(), r2.digest());


        let sb = SealedBackup {
            from: 3,
            to: 3,
            epoch: EpochIndex(3),
            sealed: vec![0xAB; 1_108],
        };
        let sbb = sb.to_bytes();
        assert_eq!(SealedBackup::from_bytes(&sbb).unwrap(), sb);
        assert!(SealedBackup::from_bytes(&sbb[..sbb.len() - 1]).is_err());
    }


    #[test]
    fn pad_batch_range_validation() {
        let a_ntt = Mat8x8Ntt::zero();
        let t_ntt = Mat2x8Ntt::zero();
        let ct = Ciphertext::zero();
        let seed = [0u8; 32];
        assert!(matches!(
            pad_batch(&a_ntt, &t_ntt, &ct, 0, &seed),
            Err(EpochError::PadRange { legs: 0, .. })
        ));
        assert!(matches!(
            pad_batch(&a_ntt, &t_ntt, &ct, CHUNK_MIN, &seed),
            Err(EpochError::PadRange { legs: 128, .. })
        ));
    }


    #[test]
    fn payload_roundtrip_and_validation() {
        let share = ShareSecret {
            member: 4,
            share: Vec8::zero(),
            rho: Vec8::zero(),
        };
        let p = backup_payload(&share);
        assert_eq!(p.len(), BACKUP_PAYLOAD_SIZE);
        assert_eq!(open_backup_payload(&p).unwrap(), share);
        assert!(matches!(
            open_backup_payload(&p[..p.len() - 1]),
            Err(EpochError::BadLength { .. })
        ));
        let mut bad = p;
        bad[0] = 0;
        assert!(matches!(
            open_backup_payload(&bad),
            Err(EpochError::BadPayloadMember { member: 0, .. })
        ));
        bad[0] = 11;
        assert!(matches!(
            open_backup_payload(&bad),
            Err(EpochError::BadPayloadMember { member: 11, .. })
        ));
        // Bound violation: a share coefficient of 71 (> 70).
        let mut vals = [0i64; 256];
        vals[0] = 71;
        let mut polys = *share.share.polys();
        polys[0] = crate::ring::Poly::from_centered(&vals);
        let over = ShareSecret { member: 4, share: Vec8::new(polys), rho: Vec8::zero() };
        assert!(matches!(
            open_backup_payload(&backup_payload(&over)),
            Err(EpochError::PayloadBound { share: 71, .. })
        ));
    }
}
