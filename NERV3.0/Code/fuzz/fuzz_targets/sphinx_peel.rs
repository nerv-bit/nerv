#![no_main]
use libfuzzer_sys::fuzz_target;


fuzz_target!(|data: &[u8]| {
    // The packet must be exactly PACKET_BYTES after the tag.
    if data.len() < 1 || data[0] != nerv_net::sphinx::MIX_PACKET_TAG {
        return;
    }
    let body = &data[1..];
    if body.len() != nerv_net::sphinx::PACKET_BYTES {
        return;
    }
    let mut bytes = [0u8; nerv_net::sphinx::PACKET_BYTES];
    bytes.copy_from_slice(body);
    let packet = nerv_net::sphinx::Packet(bytes);


    // Use a deterministic (fuzzer-seeded) key: the fuzzer's job is to
    // find peels that panic or hang, not to find the right key.
    let mut seed = [0u8; 64];
    for (i, b) in data.iter().take(64).enumerate() {
        seed[i] = *b;
    }
    if let Ok((_, dk)) = nerv_crypto::mlkem::keypair_from_seed(&seed) {
        // The peel must be total: Ok or Err, never panic.
        let _ = nerv_net::sphinx::peel(&packet, &dk);
    }
});
