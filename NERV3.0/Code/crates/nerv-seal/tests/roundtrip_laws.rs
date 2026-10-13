#![allow(clippy::unwrap_used, clippy::expect_used)]

//! D3: proptest round-trip law — `Ciphertext` wire form is a bijection and
//! single-leg encryption is deterministic in `(pk, seed, m)`, while the
//! native decrypt path stays total (never panics).

use nerv_seal::decrypt::decrypt_native;
use nerv_seal::digitize::digitize;
use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext};
use nerv_seal::sampling::NoiseSeed;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn ciphertext_wire_roundtrip(
        entropy in proptest::collection::vec(any::<u8>(), 32),
        seed in proptest::collection::vec(any::<u8>(), 32),
        coords in proptest::collection::vec(any::<u64>(), 64),
    ) {
        let mut e = [0u8; 32];
        e.copy_from_slice(&entropy);
        let mut sb = [0u8; 32];
        sb.copy_from_slice(&seed);
        let s = NoiseSeed::from_bytes(sb);
        let mut cs = [0u64; 64];
        cs.copy_from_slice(&coords);

        let (pk, sk) = derive_reference_keypair(&e).unwrap();
        let m = digitize(&cs);
        let ct = Ciphertext::encrypt(&pk, &s, &m).unwrap();

        // Determinism: identical (pk, seed, m) → identical ciphertext
        // (compared by reference so `ct` is not moved).
        prop_assert!(Ciphertext::encrypt(&pk, &s, &m).unwrap() == ct);
        // Wire round-trip: to_bytes → from_bytes is the identity.
        prop_assert!(Ciphertext::from_bytes(&ct.to_bytes()).unwrap() == ct);
        // Native single-leg decryption is total (Ok or Err, no panic).
        let _ = decrypt_native(1, &sk, ct.u(), ct.v());
    }
}
