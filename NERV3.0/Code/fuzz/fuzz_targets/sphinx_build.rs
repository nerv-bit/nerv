#![no_main]
use libfuzzer_sys::fuzz_target;
use nerv_net::sphinx::{build, encap_rand, peel, relays, Action, PathSpec, TerminalAction};

// nerv-net: Sphinx build → peel round-trip. A 5-relay packet must peel its
// first layer into a `Relay` action carrying forward parts (total, no panic).
fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let mut seed = [0u8; 8];
    seed.copy_from_slice(&data[..8]);
    let seed = u64::from_le_bytes(seed);

    let nodes = relays(5, seed);
    let mut hop = [(nodes[0].0, nodes[0].1); 5];
    for (i, slot) in hop.iter_mut().enumerate() {
        *slot = (nodes[i].0, nodes[i].1);
    }
    let dest = relays(5, seed ^ 0xFFFF).remove(0).0;
    let spec = PathSpec::new(hop, dest, TerminalAction::Deliver);

    let len = (data[7] as usize % 512) + 1;
    let mut payload = vec![0u8; len];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = data[(8 + i) % data.len()];
    }
    let randomness = encap_rand(seed ^ 0xAAAA);
    let packet = build(&spec, &payload, &randomness).expect("sphinx::build");
    let peeled = peel(&packet, &nodes[0].2).expect("sphinx::peel");
    assert!(matches!(peeled.action, Action::Relay));
    assert!(peeled.forward.is_some());
});
