#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_core::hash::Hash256;
use nerv_custody::nct::{verify_witness, NoteCommitmentTree};

// nerv-custody: NCT append → witness → verify round-trip. Every appended
// commitment's witness must reconstruct the live root (total, no panic).
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let mut tree = NoteCommitmentTree::new();
    let n = (data[0] as usize % 64) + 1;
    let mut last: Option<(Hash256, u64)> = None;
    for i in 0..n {
        let mut b = [0u8; 32];
        for (j, byte) in b.iter_mut().enumerate() {
            *byte = data[(i * 32 + j) % data.len()];
        }
        let cm = Hash256::from_bytes(b);
        let idx = tree.append(&cm).expect("append within capacity");
        let wit = tree.witness(idx).expect("witness of appended leaf");
        assert!(
            verify_witness(&tree.root(), wit.leaf_index, &cm, &wit.siblings),
            "NCT witness failed at index {idx}"
        );
        last = Some((cm, idx));
    }
    if let Some((cm, idx)) = last {
        let wit = tree.witness(idx).expect("re-witness");
        assert!(verify_witness(&tree.root(), wit.leaf_index, &cm, &wit.siblings));
    }
});
