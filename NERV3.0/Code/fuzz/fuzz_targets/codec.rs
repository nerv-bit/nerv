#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_core::codec::Decode;


fuzz_target!(|data: &[u8]| {
    // Every core type's decode path must be total: Err, never panic.
    let _ = u8::decode(data);
    let _ = u16::decode(data);
    let _ = u32::decode(data);
    let _ = u64::decode(data);
    let _ = bool::decode(data);
    let _ = String::decode(data);
    let _ = Vec::<u8>::decode(data);
    let _ = Vec::<u64>::decode(data);
    let _ = nerv_core::hash::Hash256::decode(data);
    let _ = nerv_core::types::ShardId::decode(data);
    let _ = nerv_core::types::TxId::decode(data);
    let _ = nerv_core::field::Goldilocks::decode(data);
    if data.len() >= 512 {
        let _ = nerv_state::block::ShardBlock::decode(data);
        let _ = nerv_state::ShardHeader::decode(data);
    }
});
