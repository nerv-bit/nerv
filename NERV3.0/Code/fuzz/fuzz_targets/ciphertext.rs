#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_seal::decrypt::decrypt_native;
use nerv_seal::digitize::digitize;
use nerv_seal::encrypt::{derive_reference_keypair, Ciphertext};

// nerv-seal: single-leg encrypt determinism + wire round-trip, and the
// native decrypt path is total. Ring 1024 + 1 KB plaintext (target < 50 MB).
fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }
    let mut entropy = [0u8; 32];
    entropy.copy_from_slice(&data[..32]);
    let (pk, sk) = derive_reference_keypair(&entropy).expect("reference keypair");

    let mut coords = [0u64; 64];
    for (i, c) in coords.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        for (j, byte) in b.iter_mut().enumerate() {
            *byte = data[(32 + i * 8 + j) % data.len()];
        }
        *c = u64::from_le_bytes(b);
    }
    let m = digitize(&coords);

    let mut seed = [0u8; 32];
    for (i, byte) in seed.iter_mut().enumerate() {
        *byte = data[(i * 3) % data.len()];
    }
    let ct1 = Ciphertext::encrypt(&pk, &seed, &m).expect("encrypt");
    let ct2 = Ciphertext::encrypt(&pk, &seed, &m).expect("encrypt");
    assert_eq!(ct1, ct2, "encrypt must be deterministic in (pk, seed, m)");

    let wire = ct1.to_bytes();
    assert_eq!(Ciphertext::from_bytes(&wire).expect("wire"), ct1);

    let _ = decrypt_native(1, &sk, &ct1.u, &ct1.v);
});
