//! Note plaintext, ML-KEM + AEAD sealing, trial-decryption (WP §3.2; the
//! commitment carries pk_n per erratum 72). Sealed wire: KEM ciphertext ‖
//! ChaCha20-Poly1305(plaintext, AAD = cm).

use std::fmt;
use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::NOTE_KDF;
use nerv_core::error::CodecError;
use nerv_core::hash::Hash256;
use nerv_core::params::{CUSTODY_MEMO_MAX_BYTES, CUSTODY_VALUE_MAX_NANO, CUSTODY_VALUE_MIN_NANO};
use nerv_crypto::aead::{open, seal, AeadKey, Nonce};
use nerv_crypto::kdf::blake3_kdf;
use nerv_crypto::mlkem::{CipherText, CT_LEN};
use zeroize::Zeroize;

use crate::address::{Address, DeliveryKeyPair};
use crate::commitment::{note_commitment, BLINDING_LEN, NONCE_LEN, PK_N_LEN};
use crate::error::{CustodyError, NoteError};

pub const MEMO_MAX: usize = CUSTODY_MEMO_MAX_BYTES as usize;

fn check_value(value: u64) -> Result<(), CustodyError> {
    if value < CUSTODY_VALUE_MIN_NANO || value > CUSTODY_VALUE_MAX_NANO {
        return Err(CustodyError::ValueOutOfRange {
            value,
            min: CUSTODY_VALUE_MIN_NANO,
            max: CUSTODY_VALUE_MAX_NANO,
        });
    }
    Ok(())
}

fn check_memo(memo: &Option<Vec<u8>>) -> Result<(), CustodyError> {
    if let Some(m) = memo {
        if m.len() > MEMO_MAX {
            return Err(CustodyError::MemoTooLarge { len: m.len(), max: MEMO_MAX });
        }
    }
    Ok(())
}

#[derive(Clone)]
pub struct NotePlaintext {
    pub value: u64,
    pub rho: [u8; NONCE_LEN],
    pub blinding: [u8; BLINDING_LEN],
    memo: Option<Vec<u8>>,
}

impl fmt::Debug for NotePlaintext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NotePlaintext(<redacted>)")
    }
}

impl Drop for NotePlaintext {
    fn drop(&mut self) {
        self.value = 0;
        self.rho.as_mut_slice().zeroize();
        self.blinding.as_mut_slice().zeroize();
        if let Some(m) = &mut self.memo {
            m.zeroize();
        }
    }
}

impl NotePlaintext {
    pub fn new(
        value: u64,
        rho: [u8; NONCE_LEN],
        blinding: [u8; BLINDING_LEN],
        memo: Option<Vec<u8>>,
    ) -> Result<NotePlaintext, CustodyError> {
        check_value(value)?;
        let memo = match memo {
            Some(m) if m.is_empty() => None,
            other => other,
        };
        check_memo(&memo)?;
        Ok(NotePlaintext { value, rho, blinding, memo })
    }

    pub fn memo(&self) -> Option<&[u8]> {
        self.memo.as_deref()
    }

    pub(crate) fn validate(&self) -> Result<(), CustodyError> {
        check_value(self.value)?;
        check_memo(&self.memo)
    }
}

impl Encode for NotePlaintext {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(&self.rho);
        out.extend_from_slice(&self.blinding);
        match &self.memo {
            None => out.push(0),
            Some(m) => {
                out.push(m.len() as u8);
                out.extend_from_slice(m);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        8 + NONCE_LEN + BLINDING_LEN + 1 + self.memo.as_ref().map_or(0, |m| m.len())
    }
}

impl Decode for NotePlaintext {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let value = r.read_u64()?;
        if value < CUSTODY_VALUE_MIN_NANO || value > CUSTODY_VALUE_MAX_NANO {
            return Err(CodecError::InvariantViolated("note value outside (0, 2^60]"));
        }
        let rho = r.take_array::<NONCE_LEN>()?;
        let blinding = r.take_array::<BLINDING_LEN>()?;
        let mlen = r.read_u8()? as usize;
        if mlen > MEMO_MAX {
            return Err(CodecError::InvariantViolated("memo exceeds 80 bytes"));
        }
        let memo = if mlen == 0 { None } else { Some(r.take(mlen)?.to_vec()) };
        Ok(NotePlaintext { value, rho, blinding, memo })
    }
}

#[derive(Clone)]
pub struct Note {
    pub value: u64,
    pub rho: [u8; NONCE_LEN],
    pub delivery: [u8; 1184],
    pub blinding: [u8; BLINDING_LEN],
    pub pk_n: [u8; PK_N_LEN],
    memo: Option<Vec<u8>>,
}

impl fmt::Debug for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Note(<redacted>)")
    }
}

impl Drop for Note {
    fn drop(&mut self) {
        self.value = 0;
        self.rho.as_mut_slice().zeroize();
        self.blinding.as_mut_slice().zeroize();
        if let Some(m) = &mut self.memo {
            m.zeroize();
        }
    }
}

impl Note {
    pub fn new(
        value: u64,
        rho: [u8; NONCE_LEN],
        delivery: [u8; 1184],
        blinding: [u8; BLINDING_LEN],
        pk_n: [u8; PK_N_LEN],
        memo: Option<Vec<u8>>,
    ) -> Result<Note, CustodyError> {
        let memo = match memo {
            Some(m) if m.is_empty() => None,
            other => other,
        };
        check_value(value)?;
        check_memo(&memo)?;
        Ok(Note { value, rho, delivery, blinding, pk_n, memo })
    }

    pub fn from_parts(plaintext: &NotePlaintext, delivery: [u8; 1184], pk_n: [u8; PK_N_LEN]) -> Note {
        Note {
            value: plaintext.value,
            rho: plaintext.rho,
            delivery,
            blinding: plaintext.blinding,
            pk_n,
            memo: plaintext.memo.clone(),
        }
    }

    pub fn commitment(&self) -> Result<Hash256, CustodyError> {
        check_value(self.value)?;
        check_memo(&self.memo)?;
        Ok(note_commitment(self.value, &self.rho, &self.delivery, &self.blinding, &self.pk_n))
    }

    pub fn memo(&self) -> Option<&[u8]> {
        self.memo.as_deref()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct SealedNote {
    pub kem_ct: CipherText,
    pub sealed: Vec<u8>,
}

impl fmt::Debug for SealedNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SealedNote")
            .field("kem_ct", &self.kem_ct)
            .field("sealed_bytes", &self.sealed.len())
            .finish()
    }
}

impl SealedNote {
    pub fn wire_len(&self) -> usize {
        CT_LEN + 4 + self.sealed.len()
    }
}

impl Encode for SealedNote {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.kem_ct.encode_into(out);
        out.extend_from_slice(&(self.sealed.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.sealed);
    }
    fn encoded_len(&self) -> usize {
        self.wire_len()
    }
}

impl Decode for SealedNote {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let kem_ct = CipherText::decode_from(r)?;
        let n = r.read_seq_len()?;
        r.enter()?;
        let sealed = r.take(n)?.to_vec();
        r.leave();
        Ok(SealedNote { kem_ct, sealed })
    }
}

/// Seal a note to a recipient's address (WP §3.2; the commitment carries
/// the recipient's pk_n per erratum 72). Deterministic in (plaintext,
/// address, kem_randomness).
pub fn seal_note(
    plaintext: &NotePlaintext,
    recipient: &Address,
    kem_randomness: &[u8; 32],
) -> Result<(Hash256, SealedNote), NoteError> {
    plaintext.validate()?;
    let cm = note_commitment(
        plaintext.value,
        &plaintext.rho,
        recipient.delivery().as_bytes(),
        &plaintext.blinding,
        recipient.pk_n(),
    );
    let (ss, kem_ct) = recipient.delivery().encapsulate(kem_randomness)?;
    let key = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, ss.as_bytes(), &[]));
    let mut wire = plaintext.encode();
    let sealed = seal(&key, &Nonce::ZERO, cm.as_bytes(), &wire)?;
    wire.zeroize();
    Ok((cm, SealedNote { kem_ct, sealed }))
}

/// Trial-decrypt with one candidate delivery keypair; the recovered note
/// is verified against the published commitment (which includes pk_n —
/// the scanning wallet passes its own address's pk_n).
pub fn trial_decrypt(
    sealed: &SealedNote,
    keys: &DeliveryKeyPair,
    published_cm: &Hash256,
    pk_n: &[u8; PK_N_LEN],
) -> Result<Note, NoteError> {
    let ss = keys.dk.decapsulate(&sealed.kem_ct)?;
    let key = AeadKey::from_bytes(blake3_kdf(&NOTE_KDF, ss.as_bytes(), &[]));
    let mut wire = open(&key, &Nonce::ZERO, published_cm.as_bytes(), &sealed.sealed)
        .map_err(|_| NoteError::DecryptionFailed)?;
    let plaintext = NotePlaintext::decode(&wire).map_err(|_| NoteError::MalformedPlaintext)?;
    wire.zeroize();
    let note = Note::from_parts(&plaintext, *keys.ek.as_bytes(), *pk_n);
    if note.commitment()? != *published_cm {
        return Err(NoteError::CommitmentMismatch);
    }
    Ok(note)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::address::{DetectionSeed, MasterSeed, WalletKeys};
    use crate::testutil::SplitMix64;
    use nerv_core::types::ShardSet;

    fn wallet(seed: u64) -> WalletKeys {
        WalletKeys::from_master(&MasterSeed::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    fn addr(w: &WalletKeys, i: u64) -> Address {
        Address::generate(w.detection(), w.nullifier_key(), i, &ShardSet::genesis()).unwrap()
    }

    fn pt(rng: &mut SplitMix64, memo: Option<Vec<u8>>) -> NotePlaintext {
        NotePlaintext::new(
            1 + rng.next_u64() % 1_000_000_000_000,
            rng.bytes32(),
            rng.bytes32(),
            memo,
        )
        .unwrap()
    }

    #[test]
    fn plaintext_validation_and_codec() {
        let mut rng = SplitMix64::new(1);
        let p = pt(&mut rng, Some(b"memo".to_vec()));
        assert!(matches!(
            NotePlaintext::new(0, p.rho, p.blinding, None),
            Err(CustodyError::ValueOutOfRange { value: 0, .. })
        ));
        assert!(matches!(
            NotePlaintext::new(5, p.rho, p.blinding, Some(vec![7u8; 81])),
            Err(CustodyError::MemoTooLarge { len: 81, max: 80 })
        ));
        let enc = p.encode();
        assert_eq!(enc.len(), p.encoded_len());
        let d = NotePlaintext::decode(&enc).unwrap();
        assert_eq!(d.value, p.value);
        assert_eq!(d.memo(), Some(&b"memo"[..]));
        assert!(NotePlaintext::decode(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn note_commitment_carries_pk_n() {
        let mut rng = SplitMix64::new(2);
        let p = pt(&mut rng, None);
        let d = *addr(&wallet(10), 4).delivery().as_bytes();
        let pk_a = addr(&wallet(10), 4).pk_n();
        let pk_b = addr(&wallet(10), 5).pk_n();
        let n = Note::from_parts(&p, d, *pk_a);
        let cm = n.commitment().unwrap();
        assert_eq!(
            cm,
            note_commitment(p.value, &p.rho, &d, &p.blinding, pk_a)
        );
        assert_ne!(cm, Note::from_parts(&p, d, *pk_b).commitment().unwrap());
    }

    #[test]
    fn seal_and_trial_roundtrip() {
        let w = wallet(11);
        let recipient = addr(&w, 7);
        let mut rng = SplitMix64::new(4);
        let p = pt(&mut rng, Some(b"invoice".to_vec()));
        let (cm, sealed) = seal_note(&p, &recipient, &rng.bytes32()).unwrap();

        let keys = w.detection().delivery_keypair(7).unwrap();
        let note = trial_decrypt(&sealed, &keys, &cm, recipient.pk_n()).unwrap();
        assert_eq!(note.commitment().unwrap(), cm);
        assert_eq!(note.value, p.value);
        assert_eq!(note.rho, p.rho);
        assert_eq!(note.pk_n, *recipient.pk_n());

        let enc = sealed.encode();
        assert_eq!(enc.len(), sealed.wire_len());
        assert_eq!(SealedNote::decode(&enc).unwrap(), sealed);
        assert!(SealedNote::decode(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn trial_failures() {
        let w = wallet(12);
        let recipient = addr(&w, 0);
        let mut rng = SplitMix64::new(5);
        let p = pt(&mut rng, None);
        let (cm, sealed) = seal_note(&p, &recipient, &rng.bytes32()).unwrap();

        let wrong = w.detection().delivery_keypair(1).unwrap();
        assert!(matches!(
            trial_decrypt(&sealed, &wrong, &cm, recipient.pk_n()),
            Err(NoteError::DecryptionFailed)
        ));
        let other = wallet(13).detection().delivery_keypair(0).unwrap();
        assert!(matches!(
            trial_decrypt(&sealed, &other, &cm, recipient.pk_n()),
            Err(NoteError::DecryptionFailed)
        ));
        let mut tampered = sealed.clone();
        tampered.sealed[0] ^= 1;
        assert!(matches!(
            trial_decrypt(&tampered, &w.detection().delivery_keypair(0).unwrap(), &cm, recipient.pk_n()),
            Err(NoteError::DecryptionFailed)
        ));
        let fake_cm = Hash256::from_bytes(rng.bytes32());
        assert!(matches!(
            trial_decrypt(&sealed, &w.detection().delivery_keypair(0).unwrap(), &fake_cm, recipient.pk_n()),
            Err(NoteError::DecryptionFailed)
        ));
        // Right key, wrong pk_n: the commitment check fails.
        assert!(matches!(
            trial_decrypt(&sealed, &w.detection().delivery_keypair(0).unwrap(), &cm, addr(&w, 1).pk_n()),
            Err(NoteError::CommitmentMismatch)
        ));
    }

    #[test]
    fn seal_determinism_and_wire_budget() {
        let w = wallet(14);
        let recipient = addr(&w, 2);
        let mut rng = SplitMix64::new(6);
        let p = pt(&mut rng, Some(vec![0xA5u8; 80]));
        let rand = rng.bytes32();
        let (cm1, s1) = seal_note(&p, &recipient, &rand).unwrap();
        let (cm2, s2) = seal_note(&p, &recipient, &rand).unwrap();
        assert_eq!(cm1, cm2);
        assert_eq!(s1, s2);
        let (cm3, s3) = seal_note(&p, &recipient, &rng.bytes32()).unwrap();
        assert_eq!(cm3, cm1);
        assert_ne!(s3.sealed, s1.sealed);
        assert!(s1.wire_len() <= nerv_core::params::BUDGETS_REFERENCE_ENCRYPTED_NOTES_BYTES as usize);
    }
}


