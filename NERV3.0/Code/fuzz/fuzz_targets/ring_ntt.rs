#![no_main]
use libfuzzer_sys::fuzz_target;


fuzz_target!(|data: &[u8]| {
    use nerv_seal::ring::{Poly, N};


    // Extract N coefficients from the fuzzer input.
    if data.len() < 4 * N {
        return;
    }
    let mut coeffs = [0u64; N];
    for (i, c) in coeffs.iter_mut().enumerate() {
        let mut b = [0u8; 4];
        b.copy_from_slice(&data[i * 4..i * 4 + 4]);
        // Reduce to [0, Q) to stay in the valid domain.
        *c = u32::from_le_bytes(b) as u64 % nerv_seal::ring::Q;
    }


    let poly = Poly::new(coeffs);


    // NTT → INTT must be the identity.
    let roundtrip = poly.ntt().intt();
    assert_eq!(roundtrip, poly, "NTT round-trip failed");


    // NTT of the zero polynomial is zero.
    let zero = Poly::zero();
    assert_eq!(zero.ntt().intt(), zero);
});
