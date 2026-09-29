#![no_main]
use libfuzzer_sys::fuzz_target;


fuzz_target!(|data: &[u8]| {
    // Extract 64 u64s from the fuzzer input.
    if data.len() < 8 * 64 {
        return;
    }
    let mut coords = [0u64; 64];
    for (i, c) in coords.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        b.copy_from_slice(&data[i * 8..i * 8 + 8]);
        *c = u64::from_le_bytes(b);
    }


    // Digitize → resolve must be the identity on the input.
    let pt = nerv_seal::digitize::digitize(&coords);
    let sums = pt.slot_values();
    let resolved = nerv_seal::digitize::resolve(&sums);


    // The round-trip identity is a bug if it fails.
    assert_eq!(resolved, coords, "digitize→resolve round-trip failed");
});
