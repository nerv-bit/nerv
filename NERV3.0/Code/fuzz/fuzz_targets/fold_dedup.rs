#![no_main]
use libfuzzer_sys::fuzz_target;


fuzz_target!(|data: &[u8]| {
    use nerv_core::hash::Hash256;
    use nerv_core::types::{Interval, TxId};
    use nerv_proofs::{BundleTxids, IntervalLedger};


    // Build a set of txids from the fuzzer input (deterministic).
    if data.len() < 32 {
        return;
    }
    let count = (data[0] as usize).min(8).max(1);
    let mut txids = Vec::with_capacity(count);
    for i in 0..count {
        let start = 1 + i * 32;
        if start + 32 > data.len() {
            break;
        }
        let mut b = [0u8; 32];
        b.copy_from_slice(&data[start..start + 32]);
        txids.push(TxId::from_hash(Hash256::from_bytes(b)));
    }
    if txids.is_empty() {
        return;
    }


    // Build two bundles with overlapping txid sets (the fuzzer's choice).
    let mid = txids.len() / 2;
    let b0 = BundleTxids { index: 0, txids: txids[..=mid].to_vec() };
    let b1 = BundleTxids { index: 1, txids: txids[mid..].to_vec() };


    // Dedup must be deterministic: first-wins by arrival index.
    let ledger = IntervalLedger::new();
    let report = ledger
        .dedup(Interval::from_u64(0), &[b0, b1])
        .expect("dedup must succeed on distinct-index bundles");


    // The result is deterministic: the same inputs always produce the
    // same output.
    let ledger2 = IntervalLedger::new();
    let report2 = ledger2
        .dedup(Interval::from_u64(0), &[b0, b1])
        .expect("dedup determinism");
    assert_eq!(report, report2);
});

