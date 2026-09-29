//! The ONLY module touching fips204 / fips203 (DSR-3). Pinned: fips204 0.1.0,
//! fips203 0.4.0 (workspace Cargo.toml).
//!
//! Maps the real upstream API onto the stable internal surface that
//! `mldsa.rs` / `mlkem.rs` consume. The boundary changes ONLY here on
//! upstream drift; every other module stays crate-agnostic.
//!
//! ## Actual upstream surfaces we adapt
//!
//! fips203 v0.4.0 (ML-KEM-768, NIST level 3):
//!   - `KG::keygen_from_seed(d: [u8;32], z: [u8;32]) -> (EncapsKey, DecapsKey)`
//!     (infallible; deterministic)
//!   - `KG::try_keygen_with_rng(rng) -> Result<(EncapsKey, DecapsKey), &'static str>`
//!   - `EncapsKey::try_encaps_with_rng(rng) -> Result<(SharedSecretKey, CipherText), &'static str>`
//!   - `DecapsKey::try_decaps(&CipherText) -> Result<SharedSecretKey, &'static str>`
//!   - `SerDes` for `EncapsKey`/`DecapsKey`/`CipherText` (fixed-size byte arrays).
//!
//! fips204 v0.1.0 (ML-DSA-65, NIST level 3):
//!   - `KG::try_keygen_with_rng_vt(rng) -> Result<(PublicKey, PrivateKey), &'static str>`
//!   - **No seeded keygen** — wallet determinism is achieved in this module
//!     by seeding a `ChaCha8Rng` with the caller-provided seed. ML-DSA-65
//!     keygen is randomized; we make it deterministic by fixing the RNG.
//!   - `PrivateKey::try_sign_with_rng_ct(rng, msg)` and
//!     `PublicKey::try_verify_vt(msg, sig)`. The `_ct` contract is
//!     constant-time w.r.t. the SK; the rejection loop is allowed to be
//!     random. Wallet determinism comes from seeding the ChaCha8Rng from
//!     `SK[0..32]`, which is safe (the seed is an internal prefix the
//!     holder already controls).
//!   - `SerDes` for `PrivateKey`/`PublicKey`/`Signature`.
//!
//! ## Internal wire format for `dsa_keypair_from_seed`
//!
//! ML-DSA-65 SKs are opaque in fips204 v0.1.0 (no `vk()` extractor); to keep
//! callers' `(kp_bytes, ...)` invariant stable without forcing them to track
//! PK separately, this module stores the keypair as
//!
//!   `bytes = sk_bytes[SK_LEN=4032] || pk_bytes[PK_LEN=1952]` (total 5984)
//!
//! This is opaque to callers; `mldsa.rs` already treats the bytes as a
//! blob and only reads the VK via `dsa_vk_of_keypair`. Internally we
//! reconstruct the PK on verify/dump and the SK on sign.
//!
//! ## FIPS wire lengths (validated at this boundary so any serialization drift fails loudly, never silently)
//!
//!   ML-DSA-65: pk 1952, sk 4032, sig 3309; combined keypair 5984.
//!   ML-KEM-768: ek 1184, dk 2400, ct 1088, ss 32.

use fips203::ml_kem_768 as kem768;
use fips203::traits::{Decaps as KemDecaps, Encaps as KemEncaps, SerDes as KemSerDes};
use fips204::ml_dsa_65 as dsa65;
use fips204::traits::{
    KeyGen as DsaKeyGen, SerDes as DsaSerDes, Signer as DsaSigner, Verifier as DsaVerifier,
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;


use crate::error::CryptoError;

// ---- Wire-length constants (single source of truth, derived from upstream) --

pub(crate) const DSA_PK_LEN: usize = dsa65::PK_LEN;
pub(crate) const DSA_SIG_LEN: usize = dsa65::SIG_LEN;
pub(crate) const DSA_SK_LEN: usize = dsa65::SK_LEN;
/// Internal combined keypair wire length (sk || pk); see module docs.
pub(crate) const DSA_KP_LEN: usize = DSA_SK_LEN + DSA_PK_LEN;

pub(crate) const KEM_EK_LEN: usize = kem768::EK_LEN;
pub(crate) const KEM_DK_LEN: usize = kem768::DK_LEN;
pub(crate) const KEM_CT_LEN: usize = kem768::CT_LEN;
pub(crate) const KEM_SS_LEN: usize = fips203::SSK_LEN;

// ---- Vec → fixed-array helper used by mldsa.rs / mlkem.rs -------------------

pub(crate) fn to_fixed<const N: usize>(
    bytes: Vec<u8>,
    what: &'static str,
) -> Result<[u8; N], CryptoError> {
    let found = bytes.len();
    <[u8; N]>::try_from(bytes)
        .map_err(|_| CryptoError::InvalidLength { what, expected: N, found })
}

// ============================================================================
// ML-DSA-65 (FIPS 204)
// ============================================================================

/// Deterministic keygen from a 32-byte seed: derive a ChaCha8Rng and run
/// ML-DSA-65 keygen with it. Returns the internal `sk || pk` wire format
/// (5984 bytes; see module docs).
pub(crate) fn dsa_keypair_from_seed(seed: &[u8; 32]) -> Result<Vec<u8>, &'static str> {
    let mut rng = ChaCha8Rng::from_seed(*seed);
    let (pk, sk) = <dsa65::KG as DsaKeyGen>::try_keygen_with_rng_vt(&mut rng)
        .map_err(|_| "fips204 rejected the keygen seed")?;
    let pk_bytes = <dsa65::PublicKey as DsaSerDes>::into_bytes(pk);
    let sk_bytes = <dsa65::PrivateKey as DsaSerDes>::into_bytes(sk);
    let mut combined = Vec::with_capacity(DSA_KP_LEN);
    combined.extend_from_slice(&sk_bytes);
    combined.extend_from_slice(&pk_bytes);
    Ok(combined)
}

/// Validate the internal `sk || pk` wire format by deserializing both halves.
pub(crate) fn dsa_keypair_from_bytes(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.len() != DSA_KP_LEN {
        return Err("fips204 keypair wire length mismatch");
    }
    let sk_arr: [u8; DSA_SK_LEN] = bytes[..DSA_SK_LEN]
        .try_into()
        .map_err(|_| "fips204 sk slice misalignment")?;
    let pk_arr: [u8; DSA_PK_LEN] = bytes[DSA_SK_LEN..]
        .try_into()
        .map_err(|_| "fips204 pk slice misalignment")?;
    <dsa65::PrivateKey as DsaSerDes>::try_from_bytes(sk_arr)
        .map_err(|_| "fips204 rejected the serialized private key")?;
    <dsa65::PublicKey as DsaSerDes>::try_from_bytes(pk_arr)
        .map_err(|_| "fips204 rejected the serialized public key")?;
    Ok(())
}

/// Sign `msg` with the SK half of the combined keypair bytes. The signing
/// RNG is deterministic (seeded from `SK[0..32]`) so signatures for the same
/// `(SK, msg)` pair are reproducible — the wallet-friendly choice for
/// replay-protection schemes and for stable addresses.
pub(crate) fn dsa_sign(kp_bytes: &[u8], msg: &[u8]) -> Result<Vec<u8>, &'static str> {
    if kp_bytes.len() != DSA_KP_LEN {
        return Err("fips204 keypair wire length mismatch");
    }
    let sk_arr: [u8; DSA_SK_LEN] = kp_bytes[..DSA_SK_LEN]
        .try_into()
        .map_err(|_| "fips204 sk slice misalignment")?;
    let sk = <dsa65::PrivateKey as DsaSerDes>::try_from_bytes(sk_arr)
        .map_err(|_| "corrupt ML-DSA signing key")?;
    let mut rng_seed = [0u8; 32];
    rng_seed.copy_from_slice(&sk_arr[..32]);
    let mut rng = ChaCha8Rng::from_seed(rng_seed);
    let sig = sk
        .try_sign_with_rng_ct(&mut rng, msg)
        .map_err(|_| "fips204 sign failure")?;
    Ok(<dsa65::Signature as DsaSerDes>::into_bytes(sig).to_vec())
}

/// Extract the PK half of the combined keypair bytes (already validated).
pub(crate) fn dsa_vk_of_keypair(kp_bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    if kp_bytes.len() != DSA_KP_LEN {
        return Err("fips204 keypair wire length mismatch");
    }
    let pk_arr: [u8; DSA_PK_LEN] = kp_bytes[DSA_SK_LEN..]
        .try_into()
        .map_err(|_| "fips204 pk slice misalignment")?;
    // Re-validate (cheap; also enforces the upstream's pk_decode check).
    let _ = <dsa65::PublicKey as DsaSerDes>::try_from_bytes(pk_arr)
        .map_err(|_| "fips204 rejected the serialized public key")?;
    Ok(pk_arr.to_vec())
}

/// Verify `sig_bytes` against the externally-provided PK. Returns `false`
/// for any malformed input (length, SerDes, or verify failure) — never
/// panics, never errors out; this is the hot-path verifier.
pub(crate) fn dsa_verify(vk_bytes: &[u8], msg: &[u8], sig_bytes: &[u8]) -> bool {
    let Ok(pk_arr) = <[u8; DSA_PK_LEN]>::try_from(vk_bytes) else {
        return false;
    };
    let Ok(sig_arr) = <[u8; DSA_SIG_LEN]>::try_from(sig_bytes) else {
        return false;
    };
    let Ok(pk) = <dsa65::PublicKey as DsaSerDes>::try_from_bytes(pk_arr) else {
        return false;
    };
    let Ok(sig) = <dsa65::Signature as DsaSerDes>::try_from_bytes(sig_arr) else {
        return false;
    };
    pk.try_verify_vt(msg, &sig).unwrap_or(false)
}

// ============================================================================
// ML-KEM-768 (FIPS 203)
// ============================================================================

/// Deterministic seeded keygen: split the 64-byte seed into `d` (first 32)
/// and `z` (last 32) per FIPS 203 Algorithm 16. `KG::keygen_from_seed` is
/// infallible — it cannot reject a seed — so we map upstream success to
/// `Ok` and only the upstream wire-format invariants (encoded lengths)
/// surface as `Err`.
///
/// Caller-facing convention preserved: returns `(ek_bytes, dk_bytes)` as
/// separate vectors of exact length `EK_LEN` / `DK_LEN`.
pub(crate) fn kem_keypair_from_seed(
    seed: &[u8; 64],
) -> Result<(Vec<u8>, Vec<u8>), &'static str> {
    let mut d = [0u8; 32];
    let mut z = [0u8; 32];
    d.copy_from_slice(&seed[..32]);
    z.copy_from_slice(&seed[32..]);
    let (ek, dk) = <kem768::KG as fips203::traits::KeyGen>::keygen_from_seed(d, z);
    Ok((
        <kem768::EncapsKey as KemSerDes>::into_bytes(ek).to_vec(),
        <kem768::DecapsKey as KemSerDes>::into_bytes(dk).to_vec(),
    ))
}

/// Explicit-randomness encapsulation: the 32-byte `m` is the FIPS 203
/// message (per Algorithm 17); we feed it as the RNG seed so the output is
/// deterministic for a given `(ek, m)` pair, matching `mlkem.rs`'s
/// deterministic test contract.
pub(crate) fn kem_encapsulate(
    ek_bytes: &[u8],
    m: &[u8; 32],
) -> Result<(Vec<u8>, Vec<u8>), &'static str> {
    let ek_arr: [u8; KEM_EK_LEN] = ek_bytes
        .try_into()
        .map_err(|_| "invalid ML-KEM encapsulation key wire length")?;
    let ek = <kem768::EncapsKey as KemSerDes>::try_from_bytes(ek_arr)
        .map_err(|_| "invalid ML-KEM encapsulation key")?;
    let mut rng = ChaCha8Rng::from_seed(*m);
    let (ssk, ct) = ek
        .try_encaps_with_rng(&mut rng)
        .map_err(|_| "fips203 encaps failure")?;
    Ok((
        <fips203::SharedSecretKey as KemSerDes>::into_bytes(ssk).to_vec(),
        <kem768::CipherText as KemSerDes>::into_bytes(ct).to_vec(),
    ))
}

/// Decapsulate. Implicit rejection (wrong-key decapsulation returns a
/// pseudorandom secret, never an error) is provided by upstream — only
/// malformed `dk` or `ct` bytes error here.
pub(crate) fn kem_decapsulate(dk_bytes: &[u8], ct_bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    let dk_arr: [u8; KEM_DK_LEN] = dk_bytes
        .try_into()
        .map_err(|_| "invalid ML-KEM decapsulation key wire length")?;
    let ct_arr: [u8; KEM_CT_LEN] = ct_bytes
        .try_into()
        .map_err(|_| "invalid ML-KEM ciphertext wire length")?;
    let dk = <kem768::DecapsKey as KemSerDes>::try_from_bytes(dk_arr)
        .map_err(|_| "invalid ML-KEM decapsulation key")?;
    let ct = <kem768::CipherText as KemSerDes>::try_from_bytes(ct_arr)
        .map_err(|_| "invalid ML-KEM ciphertext")?;
    let ssk = dk
        .try_decaps(&ct)
        .map_err(|_| "fips203 decaps failure")?;
    Ok(<fips203::SharedSecretKey as KemSerDes>::into_bytes(ssk).to_vec())
}

// ============================================================================
// Tests — wire-contract invariants the rest of the crate relies on
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_length_contract() {
        // ---- ML-DSA-65 ----
        let seed = [7u8; 32];
        let kp = dsa_keypair_from_seed(&seed).unwrap();
        assert_eq!(kp.len(), DSA_KP_LEN);
        let vk = dsa_vk_of_keypair(&kp).unwrap();
        assert_eq!(vk.len(), DSA_PK_LEN);
        let sig = dsa_sign(&kp, b"provider smoke").unwrap();
        assert_eq!(sig.len(), DSA_SIG_LEN);
        assert!(dsa_verify(&vk, b"provider smoke", &sig));
        assert!(!dsa_verify(&vk, b"other", &sig));
        assert!(!dsa_verify(&vk[..vk.len() - 1], b"provider smoke", &sig));
        assert!(!dsa_verify(&vk, b"provider smoke", &sig[..sig.len() - 1]));
        // round-trip the combined keypair
        dsa_keypair_from_bytes(&kp).unwrap();
        assert!(dsa_keypair_from_bytes(&kp[..kp.len() - 1]).is_err());
        assert!(dsa_keypair_from_bytes(&kp[..DSA_SK_LEN]).is_err());
        assert!(dsa_keypair_from_bytes(&[]).is_err());

        // ---- ML-KEM-768 ----
        let kseed = [9u8; 64];
        let (ek, dk) = kem_keypair_from_seed(&kseed).unwrap();
        assert_eq!((ek.len(), dk.len()), (KEM_EK_LEN, KEM_DK_LEN));
        let (ss, ct) = kem_encapsulate(&ek, &[3u8; 32]).unwrap();
        assert_eq!((ss.len(), ct.len()), (KEM_SS_LEN, KEM_CT_LEN));
        assert_eq!(kem_decapsulate(&dk, &ct).unwrap(), ss);
        // malformed inputs are rejected, not panicking
        assert!(kem_encapsulate(&ek[..KEM_EK_LEN - 1], &[0u8; 32]).is_err());
        assert!(kem_decapsulate(&dk[..KEM_DK_LEN - 1], &ct).is_err());
        assert!(kem_decapsulate(&dk, &ct[..KEM_CT_LEN - 1]).is_err());
    }

    #[test]
    fn dsa_deterministic_keygen_under_same_seed() {
        let seed = [42u8; 32];
        let kp1 = dsa_keypair_from_seed(&seed).unwrap();
        let kp2 = dsa_keypair_from_seed(&seed).unwrap();
        assert_eq!(kp1, kp2);
        // Different seed → different keypair (avalanche sanity).
        let kp3 = dsa_keypair_from_seed(&[43u8; 32]).unwrap();
        assert_ne!(kp1, kp3);
    }

    #[test]
    fn dsa_signing_is_deterministic_for_same_sk_and_message() {
        let seed = [11u8; 32];
        let kp = dsa_keypair_from_seed(&seed).unwrap();
        let msg = b"transaction digest";
        let sig1 = dsa_sign(&kp, msg).unwrap();
        let sig2 = dsa_sign(&kp, msg).unwrap();
        assert_eq!(sig1, sig2);
        let vk = dsa_vk_of_keypair(&kp).unwrap();
        assert!(dsa_verify(&vk, msg, &sig1));
        // Different message → different signature.
        let sig3 = dsa_sign(&kp, b"other digest").unwrap();
        assert_ne!(sig1, sig3);
    }

    #[test]
    fn kem_deterministic_keygen_under_same_seed() {
        let seed = [21u8; 64];
        let (ek1, dk1) = kem_keypair_from_seed(&seed).unwrap();
        let (ek2, dk2) = kem_keypair_from_seed(&seed).unwrap();
        assert_eq!(ek1, ek2);
        assert_eq!(dk1, dk2);
        let (ek3, _) = kem_keypair_from_seed(&[22u8; 64]).unwrap();
        assert_ne!(ek1, ek3);
    }

    #[test]
    fn kem_explicit_randomness_makes_encaps_deterministic() {
        let seed = [33u8; 64];
        let (ek, dk) = kem_keypair_from_seed(&seed).unwrap();
        let m = [44u8; 32];
        let (ss1, ct1) = kem_encapsulate(&ek, &m).unwrap();
        let (ss2, ct2) = kem_encapsulate(&ek, &m).unwrap();
        assert_eq!(ss1, ss2);
        assert_eq!(ct1, ct2);
        assert_eq!(kem_decapsulate(&dk, &ct1).unwrap(), ss1);
        // Different randomness → different (ss, ct) pair.
        let (ss3, ct3) = kem_encapsulate(&ek, &[45u8; 32]).unwrap();
        assert_ne!(ss1, ss3);
        assert_ne!(ct1, ct3);
        assert_eq!(kem_decapsulate(&dk, &ct3).unwrap(), ss3);
    }
}
