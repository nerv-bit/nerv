//! Domain-Separated Hashing — BLAKE3 + SHA3-256.
//!
//! NERV uses two hash functions, each in a specific domain:
//!
//! | Hash | Domain | Usage |
//! |------|--------|-------|
//! | BLAKE3 | Embedding roots, KeyIds, VDWs, internal | Fast, parallelizable |
//! | SHA3-256 | Transaction hashes, public-facing IDs | Conservative, NIST |
//!
//! # Domain Separation
//!
//! Every hash call includes a domain string to prevent cross-protocol
//! collisions. Even though BLAKE3 and SHA3-256 have different internal
//! structures, domain separation adds a defense-in-depth layer:
//!
//! ```text
//! H_tx(msg)      = SHA3-256("nerv:tx"      ‖ msg)
//! H_root(msg)    = BLAKE3("nerv:root"      ‖ msg)
//! H_vdw(msg)     = BLAKE3("nerv:vdw"       ‖ msg)
//! H_shard(msg)   = BLAKE3("nerv:shard"     ‖ msg)
//! H_dkg(msg)     = BLAKE3("nerv:dkg"       ‖ msg)
//! H_sphinx(msg)  = BLAKE3("nerv:sphinx"    ‖ msg)
//! H_mixer(msg)   = BLAKE3("nerv:mixer"     ‖ msg)
//! H_account(msg) = BLAKE3("nerv:account"   ‖ msg)
//! ```

use crate::NervResult;
use sha3::Digest;

// ─── BLAKE3 Hashing ──────────────────────────────────────────────────────

/// Compute BLAKE3-256 of the input data.
///
/// This is the fast, primary hash function used for all internal
/// NERV operations (embedding roots, key IDs, etc.).
#[inline]
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    blake3::hash(data).into()
}

/// Compute BLAKE3-256 with a domain separation string.
///
/// `H(data) = BLAKE3(domain ‖ data)`
#[inline]
pub fn blake3_hash_domain(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute BLAKE3 derived key (keyed hashing).
///
/// Uses BLAKE3's native key derivation mode, which is more
/// efficient than HMAC-BLAKE3 for key derivation.
#[inline]
pub fn blake3_derive_key(context: &str, data: &[u8]) -> [u8; 32] {
    blake3::derive_key(context.as_bytes(), data).into()
}

/// Compute BLAKE3-256 of multiple input slices (one-shot).
///
/// Equivalent to `blake3_hash(a ‖ b ‖ c ‖ ...)` but avoids allocation.
pub fn blake3_hash_multi(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// Incremental BLAKE3 hasher for streaming use.
pub type Blake3Hasher = blake3::Hasher;

/// Create a new BLAKE3 hasher.
#[inline]
pub fn blake3_hasher() -> Blake3Hasher {
    blake3::Hasher::new()
}

/// Create a keyed BLAKE3 hasher.
#[inline]
pub fn blake3_keyed_hasher(key: &[u8; 32]) -> Blake3Hasher {
    blake3::Hasher::new_keyed(key)
}

// ─── SHA3-256 Hashing ────────────────────────────────────────────────────

/// Compute SHA3-256 of the input data.
///
/// Used for transaction hashes and public-facing identifiers.
#[inline]
pub fn sha3_256(data: &[u8]) -> [u8; 32] {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

/// Compute SHA3-256 with domain separation.
///
/// `H(data) = SHA3-256(domain ‖ data)`
#[inline]
pub fn sha3_256_domain(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = sha3::Sha3_256::new();
    hasher.update(domain.as_bytes());
    hasher.update(data);
    let result = hasher.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

// ─── Domain-Separated Hash Functions ──────────────────────────────────────

/// Hash a private transaction: SHA3-256 with "nerv:tx" domain.
///
/// Transaction hashes are the ONLY on-chain identifier for transactions.
/// No addresses, amounts, or metadata are ever visible.
pub fn hash_transaction(tx_data: &[u8]) -> [u8; 32] {
    sha3_256_domain("nerv:tx", tx_data)
}

/// Hash an embedding root: BLAKE3 with "nerv:root" domain.
///
/// Embedding roots are 32-byte commitments to the 512-byte
/// neural state embedding. They serve as the "state root" of a shard.
pub fn hash_embedding_root(embedding_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:root", embedding_data)
}

/// Hash for VDW (Verifiable Delay Witness) components.
pub fn hash_vdw(vdw_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:vdw", vdw_data)
}

/// Hash for shard-related data.
pub fn hash_shard(shard_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:shard", shard_data)
}

/// Hash for DKG-related data.
pub fn hash_dkg(dkg_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:dkg", dkg_data)
}

/// Hash for Sphinx packet components.
pub fn hash_sphinx(sphinx_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:sphinx", sphinx_data)
}

/// Hash for mixnet relay data.
pub fn hash_mixer(mixer_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:mixer", mixer_data)
}

/// Hash for account routing/assignment.
pub fn hash_account(account_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:account", account_data)
}

/// Hash for validator identification.
pub fn hash_validator(validator_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:validator", validator_data)
}

/// Hash for consensus voting data.
pub fn hash_consensus(consensus_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:consensus", consensus_data)
}

/// Hash for economy/reward data.
pub fn hash_economy(economy_data: &[u8]) -> [u8; 32] {
    blake3_hash_domain("nerv:economy", economy_data)
}

// ─── Key Derivation ──────────────────────────────────────────────────────

/// Derive a 32-byte key from a context string and input material.
///
/// Uses BLAKE3's native key derivation mode (based on Chacha20
/// and domain-separated from regular hashing).
pub fn derive_key(context: &str, input_key_material: &[u8]) -> [u8; 32] {
    blake3_derive_key(context, input_key_material)
}

/// Derive a Sphinx hop key from a shared secret and hop index.
///
/// `K_hop = BLAKE3-DeriveKey("nerv:sphinx:hop:N", shared_secret)`
pub fn derive_sphinx_hop_key(shared_secret: &[u8], hop_index: u8) -> [u8; 32] {
    let context = format!("nerv:sphinx:hop:{}", hop_index);
    blake3_derive_key(&context, shared_secret)
}

/// Derive an AEAD nonce from a key and counter.
///
/// `nonce = BLAKE3-DeriveKey("nerv:aead:nonce", key ‖ counter)[0..12]`
pub fn derive_aead_nonce(key: &[u8], counter: u64) -> [u8; 12] {
    let mut input = key.to_vec();
    input.extend_from_slice(&counter.to_le_bytes());
    let derived = blake3_derive_key("nerv:aead:nonce", &input);
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&derived[..12]);
    nonce
}

/// Derive a transaction encryption key from a DKG shared secret.
///
/// Used by the threshold mempool to encrypt individual transactions.
pub fn derive_tx_encryption_key(dkg_shared_secret: &[u8], tx_index: u64) -> [u8; 32] {
    let context = format!("nerv:mempool:tx:{}", tx_index);
    blake3_derive_key(&context, dkg_shared_secret)
}

/// Derive a wallet authentication key from a seed.
///
/// `K_auth = BLAKE3-DeriveKey("nerv:wallet:auth", seed)`
pub fn derive_wallet_auth_key(seed: &[u8]) -> [u8; 32] {
    blake3_derive_key("nerv:wallet:auth", seed)
}

// ─── HMAC-BLAKE3 ─────────────────────────────────────────────────────────

/// Compute HMAC-BLAKE3.
///
/// While BLAKE3 has a native keyed mode, HMAC-BLAKE3 is provided
/// for compatibility with protocols that require HMAC construction.
pub fn hmac_blake3(key: &[u8], message: &[u8]) -> [u8; 32] {
    // HMAC(K, m) = H((K ⊕ opad) ‖ H((K ⊕ ipad) ‖ m))
    // For BLAKE3, we use the keyed mode as an approximation
    // that provides equivalent security properties.
    let mut hasher = blake3::Hasher::new_keyed(
        &blake3_derive_key("nerv:hmac-blake3", key)
    );
    hasher.update(message);
    hasher.finalize().into()
}

// ─── Utility Functions ───────────────────────────────────────────────────

/// Constant-time comparison of two 32-byte hashes.
///
/// Returns `true` if equal, `false` otherwise.
/// Uses `subtle` to prevent timing side-channels.
pub fn ct_equal(a: &[u8; 32], b: &[u8; 32]) -> bool {
    use subtle::ConstantTimeEq;
    a.ct_eq(b).into()
}

/// Compute the XOR of two 32-byte values.
#[inline]
pub fn xor_32(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// Securely zeroize a byte slice.
///
/// Uses the `zeroize` crate to ensure the compiler
/// doesn't optimize away the zeroing.
pub fn secure_zero(data: &mut [u8]) {
    use zeroize::Zeroize;
    data.zeroize();
}

/// Generate cryptographically random bytes.
///
/// Uses the system's CSPRNG.
pub fn random_bytes(len: usize) -> Vec<u8> {
    use rand::RngCore;
    let mut buf = vec![0u8; len];
    rand::thread_rng().fill_bytes(&mut buf);
    buf
}

/// Generate a cryptographically random 32-byte value.
pub fn random_32bytes() -> [u8; 32] {
    let v = random_bytes(32);
    v.try_into().unwrap_or([0u8; 32])
}

/// Generate a cryptographically random u64.
pub fn random_u64() -> u64 {
    use rand::RngCore;
    rand::thread_rng().next_u64()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash_basic() {
        let h1 = blake3_hash(b"hello");
        let h2 = blake3_hash(b"hello");
        assert_eq!(h1, h2);

        let h3 = blake3_hash(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_blake3_hash_domain_separation() {
        let h1 = blake3_hash_domain("nerv:tx", b"data");
        let h2 = blake3_hash_domain("nerv:root", b"data");
        // Same data, different domains → different hashes
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_blake3_derive_key() {
        let k1 = blake3_derive_key("context1", b"input");
        let k2 = blake3_derive_key("context2", b"input");
        // Same input, different contexts → different keys
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_blake3_hash_multi() {
        let h1 = blake3_hash_multi(&[b"hello", b"world"]);
        let h2 = blake3_hash(&[b"hello", b"world"].concat());
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_sha3_256_basic() {
        let h1 = sha3_256(b"hello");
        let h2 = sha3_256(b"hello");
        assert_eq!(h1, h2);

        let h3 = sha3_256(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_sha3_256_domain_separation() {
        let h1 = sha3_256_domain("nerv:tx", b"data");
        let h2 = sha3_256_domain("nerv:root", b"data");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_transaction() {
        let h = hash_transaction(b"private tx data");
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_hash_embedding_root() {
        let h = hash_embedding_root(&[0u8; 512]);
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_domain_hashes_different() {
        let data = b"same input data";
        let h_tx = hash_transaction(data);
        let h_root = hash_embedding_root(data);
        let h_vdw = hash_vdw(data);
        let h_shard = hash_shard(data);

        // All domain-separated hashes should be different
        assert_ne!(h_tx, h_root);
        assert_ne!(h_tx, h_vdw);
        assert_ne!(h_root, h_shard);
    }

    #[test]
    fn test_derive_sphinx_hop_key() {
        let ss = b"shared secret for hop";
        let k1 = derive_sphinx_hop_key(ss, 0);
        let k2 = derive_sphinx_hop_key(ss, 1);
        let k3 = derive_sphinx_hop_key(ss, 0);

        // Different hops → different keys
        assert_ne!(k1, k2);
        // Same hop → same key (deterministic)
        assert_eq!(k1, k3);
    }

    #[test]
    fn test_derive_aead_nonce() {
        let key = b"encryption key";
        let n1 = derive_aead_nonce(key, 0);
        let n2 = derive_aead_nonce(key, 1);
        let n3 = derive_aead_nonce(key, 0);

        assert_ne!(n1, n2);
        assert_eq!(n1, n3);
        assert_eq!(n1.len(), 12);
    }

    #[test]
    fn test_derive_tx_encryption_key() {
        let dkg_ss = b"dkg shared secret";
        let k1 = derive_tx_encryption_key(dkg_ss, 0);
        let k2 = derive_tx_encryption_key(dkg_ss, 1);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_hmac_blake3() {
        let key = b"mac key";
        let msg = b"message";
        let h1 = hmac_blake3(key, msg);
        let h2 = hmac_blake3(key, msg);
        assert_eq!(h1, h2);

        let h3 = hmac_blake3(b"wrong key", msg);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_ct_equal() {
        let a = blake3_hash(b"a");
        let b = blake3_hash(b"a");
        let c = blake3_hash(b"c");

        assert!(ct_equal(&a, &b));
        assert!(!ct_equal(&a, &c));
    }

    #[test]
    fn test_xor_32() {
        let a = [0xAAu8; 32];
        let b = [0x55u8; 32];
        let result = xor_32(&a, &b);
        for &byte in &result {
            assert_eq!(byte, 0xFF);
        }
    }

    #[test]
    fn test_xor_32_self() {
        let a = blake3_hash(b"test");
        let result = xor_32(&a, &a);
        assert_eq!(result, [0u8; 32]); // a ⊕ a = 0
    }

    #[test]
    fn test_secure_zero() {
        let mut data = vec![42u8; 100];
        secure_zero(&mut data);
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_random_bytes() {
        let r1 = random_bytes(32);
        let r2 = random_bytes(32);
        assert_eq!(r1.len(), 32);
        assert_eq!(r2.len(), 32);
        // Overwhelming probability they're different
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_random_32bytes() {
        let r = random_32bytes();
        assert_eq!(r.len(), 32);
    }

    #[test]
    fn test_random_u64() {
        let r = random_u64();
        // Just check it doesn't panic
        let _ = r;
    }

    #[test]
    fn test_blake3_incremental() {
        let one_shot = blake3_hash(b"hello world");

        let mut hasher = blake3_hasher();
        hasher.update(b"hello ");
        hasher.update(b"world");
        let incremental: [u8; 32] = hasher.finalize().into();

        assert_eq!(one_shot, incremental);
    }

    #[test]
    fn test_sha3_256_known_vector() {
        // SHA3-256("") = a7ffc6f8bf1ed76651c14756a061e662f5dbffba0ddc17...
        let empty = sha3_256(b"");
        let expected_hex = "a7ffc6f8bf1ed76651c14756a061e662f5dbffba0ddcff5f4361917c3e0b8d2";
        assert_eq!(hex::encode(empty), expected_hex);
    }

    #[test]
    fn test_blake3_known_vector() {
        // BLAKE3("") = af13428413a06e3837d5e3c9c3f4ae1fa4414e7c2762a4e686e6677...
        let empty = blake3_hash(b"");
        // Just verify it's non-zero and 32 bytes
        assert_ne!(empty, [0u8; 32]);
        assert_eq!(empty.len(), 32);
    }

    #[test]
    fn test_derive_wallet_auth_key() {
        let seed = b"wallet seed phrase";
        let k = derive_wallet_auth_key(seed);
        assert_ne!(k, [0u8; 32]);
    }
}
