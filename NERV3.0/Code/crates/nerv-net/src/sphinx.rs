//! PQ-Sphinx (WP §6.2; erratum 149): 20 KB uniform packets over a 5-relay
//! path. Per-hop ML-KEM capsules carry the routing meta; the payload
//! region is layered XOR streams (the classical Sphinx surplus adapted to
//! per-hop KEM keys) so the packet's size and structure are identical at
//! every hop. ML-KEM implicit rejection makes decapsulation total — the
//! capsule's AEAD is the misroute/tamper detector.

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::SPHINX;
use nerv_core::error::CodecError;
use nerv_core::hash::{Hash256, Xof};
use nerv_core::types::TxId;
use nerv_crypto::aead::{open, seal, AeadKey, Nonce, TAG_LEN};
use nerv_crypto::kdf::blake3_kdf;
use nerv_crypto::mlkem::{CipherText, DecapsulationKey, EncapsulationKey, ENCAPS_RANDOMNESS_LEN};

use crate::host::PeerId;

pub const PATH_RELAYS: usize = nerv_core::params::MIXNET_PATH_RELAYS as usize;
pub const PACKET_BYTES: usize = nerv_core::params::MIXNET_PACKET_BYTES as usize;
pub const CT_LEN: usize = nerv_crypto::mlkem::CT_LEN;
pub const META_LEN: usize = 65;
pub const CAPSULE_LEN: usize = META_LEN + TAG_LEN;
pub const META_REGION: usize = PATH_RELAYS * CAPSULE_LEN;
/// Fixed-size per-packet header (routing + per-hop nonce + MAC).
/// Derived as `PACKET_BYTES - META_REGION - PAYLOAD_REGION`; pinned to the
/// value asserted below (`5440` per erratum 149).
pub const HEADER_LEN: usize = 5440;
pub const PAYLOAD_REGION: usize = PACKET_BYTES - HEADER_LEN - META_REGION;
pub const MAX_DATA: usize = PAYLOAD_REGION - 2;
pub const FRAG_HEADER: usize = 40;
pub const MAX_FRAG_DATA: usize = MAX_DATA - FRAG_HEADER;
pub const ENCAPS_LEN: usize = ENCAPS_RANDOMNESS_LEN;
pub const MIX_PACKET_TAG: u8 = 4;
pub const MIX_FRAGMENT_TAG: u8 = 5;
pub const FRAG_CLASSES: [usize; 5] = [1, 2, 4, 8, 16];

const ACTION_RELAY: u8 = 0;
const ACTION_DELIVER: u8 = 1;
const ACTION_DROP: u8 = 2;

const _: () = assert!(PATH_RELAYS == 5);
const _: () = assert!(PACKET_BYTES == 20_000);
// HEADER_LEN is itself a const, so an `assert!(HEADER_LEN == 5_440)` would
// be tautological; the derived consts below pin the relationships instead.
const _: () = assert!(META_REGION == 405);
const _: () = assert!(PAYLOAD_REGION == 14_155);
const _: () = assert!(MAX_FRAG_DATA == 14_113);
// Const PartialEq on `[usize; 5]` is not stable on rustc 1.85; the
// relationships above pin the derived constants and the test suite
// (see `crates/nerv-net/src/sphinx.rs` tests) verifies the params
// agree with `FRAG_CLASSES`/`MIXNET_FRAGMENTATION_MAX_PACKETS` at runtime.
#[allow(dead_code)]
const FRAG_CLASSES_PIN: [usize; 5] = [1, 2, 4, 8, 16];
const _: () = assert!(nerv_core::params::MIXNET_FRAGMENTATION_MAX_PACKETS == 16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SphinxError {
    #[error("payload of {len} exceeds the {max}-byte packet capacity")]
    PayloadTooLarge { len: usize, max: usize },
    #[error("payload of {len} exceeds the {max}-byte class-16 capacity")]
    FragmentTooLarge { len: usize, max: usize },
    #[error("class {found} is not one of {{1, 2, 4, 8, 16}}")]
    BadClass { found: usize },
    #[error("capsule authentication failed — wrong relay or tampering")]
    BadCapsule,
    #[error("terminal payload failed its commitment — corrupted in transit")]
    CorruptPayload,
    #[error("payload length prefix exceeds the region")]
    MalformedPayload,
    #[error("forward parts carry {cts} cts and {capsules} capsules, expected 4 each")]
    BadStructure { cts: usize, capsules: usize },
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("codec: {0}")]
    Codec(#[from] CodecError),
}

fn derive_keys(ss: &nerv_crypto::mlkem::SharedSecret) -> (AeadKey, AeadKey) {
    (
        AeadKey::from_bytes(blake3_kdf(&SPHINX, ss.as_bytes(), b"meta")),
        AeadKey::from_bytes(blake3_kdf(&SPHINX, ss.as_bytes(), b"stream")),
    )
}

fn stream_xor(k: &AeadKey, buf: &mut [u8]) {
    let mut xof = Xof::new(&SPHINX, k.as_bytes());
    let mut keystream = vec![0u8; buf.len()];
    xof.fill(&mut keystream);
    for (b, s) in buf.iter_mut().zip(keystream) {
        *b ^= s;
    }
}

pub fn replay_tag(ct0: &[u8; CT_LEN], capsule0: &[u8]) -> [u8; 16] {
    let mut msg = Vec::with_capacity(CT_LEN + capsule0.len());
    msg.extend_from_slice(ct0);
    msg.extend_from_slice(capsule0);
    let h = Hash256::concat(&SPHINX, &msg);
    let mut tag = [0u8; 16];
    tag.copy_from_slice(&h.as_bytes()[..16]);
    tag
}

// ---------------------------------------------------------------------------
// The packet
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet(pub [u8; PACKET_BYTES]);

impl Packet {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(1 + PACKET_BYTES);
        v.push(MIX_PACKET_TAG);
        v.extend_from_slice(&self.0);
        v
    }

    pub fn from_frame(payload: &[u8]) -> Result<Packet, CodecError> {
        if payload.len() != 1 + PACKET_BYTES || payload[0] != MIX_PACKET_TAG {
            return Err(CodecError::InvariantViolated("mix packet frame shape"));
        }
        let mut p = [0u8; PACKET_BYTES];
        p.copy_from_slice(&payload[1..]);
        Ok(Packet(p))
    }
}

impl Encode for Packet {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        PACKET_BYTES
    }
}

impl Decode for Packet {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Packet(r.take_array::<PACKET_BYTES>()?))
    }
}

// ---------------------------------------------------------------------------
// Build and peel
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalAction {
    Deliver,
    Drop,
}

#[derive(Clone, Debug)]
pub struct PathSpec {
    pub eks: [EncapsulationKey; PATH_RELAYS],
    pub nexts: [PeerId; PATH_RELAYS],
    pub terminal: TerminalAction,
}

impl PathSpec {
    /// The wallet-facing constructor: the relay list and the final
    /// destination; the next-hop chain is derived.
    pub fn new(
        relays: [(PeerId, EncapsulationKey); PATH_RELAYS],
        dest: PeerId,
        terminal: TerminalAction,
    ) -> PathSpec {
        let mut nexts = [dest; PATH_RELAYS];
        for i in 0..PATH_RELAYS - 1 {
            nexts[i] = relays[i + 1].0;
        }
        PathSpec { eks: relays.map(|r| r.1), nexts, terminal }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    Relay,
    Deliver { data: Vec<u8> },
    Drop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peeled {
    pub tag: [u8; 16],
    pub next: PeerId,
    pub action: Action,
    pub forward: Option<ForwardParts>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardParts {
    pub cts: Vec<[u8; CT_LEN]>,
    pub capsules: Vec<[u8; CAPSULE_LEN]>,
    pub payload: [u8; PAYLOAD_REGION],
}

pub fn build(
    spec: &PathSpec,
    data: &[u8],
    encap_randomness: &[[u8; ENCAPS_LEN]; PATH_RELAYS],
) -> Result<Packet, SphinxError> {
    if data.len() > MAX_DATA {
        return Err(SphinxError::PayloadTooLarge { len: data.len(), max: MAX_DATA });
    }
    let mut m = [0u8; PAYLOAD_REGION];
    m[0..2].copy_from_slice(&(data.len() as u16).to_le_bytes());
    m[2..2 + data.len()].copy_from_slice(data);
    let commit = Hash256::concat(&SPHINX, &m);

    let mut cts = [[0u8; CT_LEN]; PATH_RELAYS];
    let mut payload = m;
    let mut capsules: Vec<Vec<u8>> = Vec::with_capacity(PATH_RELAYS);
    for i in 0..PATH_RELAYS {
        let (ss, ct) = spec.eks[i].encapsulate(&encap_randomness[i])?;
        let (k_meta, k_stream) = derive_keys(&ss);
        cts[i] = *ct.as_bytes();
        let action = if i + 1 < PATH_RELAYS {
            ACTION_RELAY
        } else {
            match spec.terminal {
                TerminalAction::Deliver => ACTION_DELIVER,
                TerminalAction::Drop => ACTION_DROP,
            }
        };
        let mut meta = [0u8; META_LEN];
        meta[0..32].copy_from_slice(spec.nexts[i].as_hash().as_bytes());
        meta[32] = action;
        meta[33..65].copy_from_slice(commit.as_bytes());
        capsules.push(seal(&k_meta, &Nonce::ZERO, &[], &meta)?);
        stream_xor(&k_stream, &mut payload);
    }

    let mut bytes = [0u8; PACKET_BYTES];
    let mut at = 0usize;
    for ct in &cts {
        bytes[at..at + CT_LEN].copy_from_slice(ct);
        at += CT_LEN;
    }
    for cap in &capsules {
        bytes[at..at + CAPSULE_LEN].copy_from_slice(cap);
        at += CAPSULE_LEN;
    }
    bytes[at..].copy_from_slice(&payload);
    Ok(Packet(bytes))
}

pub fn peel(packet: &Packet, dk: &DecapsulationKey) -> Result<Peeled, SphinxError> {
    let ct0: &[u8; CT_LEN] = packet.0[..CT_LEN].try_into().expect("ct slice");
    let capsule0 = &packet.0[HEADER_LEN..HEADER_LEN + CAPSULE_LEN];
    let tag = replay_tag(ct0, capsule0);

    let ss = dk.decapsulate(&CipherText::from_bytes(*ct0))?;
    let (k_meta, k_stream) = derive_keys(&ss);
    let meta = open(&k_meta, &Nonce::ZERO, &[], capsule0).map_err(|_| SphinxError::BadCapsule)?;
    if meta.len() != META_LEN {
        return Err(SphinxError::BadCapsule);
    }
    let next = PeerId::from_hash(Hash256::from_bytes(
        meta[0..32].try_into().expect("peer id"),
    ));

    let mut payload = [0u8; PAYLOAD_REGION];
    payload.copy_from_slice(&packet.0[HEADER_LEN + META_REGION..]);
    stream_xor(&k_stream, &mut payload);

   let mut forward: Option<ForwardParts> = None;
   let action = match meta[32] {
       ACTION_RELAY => {
           let mut cts = Vec::with_capacity(PATH_RELAYS - 1);
           for i in 1..PATH_RELAYS {
               cts.push(packet.0[i * CT_LEN..(i + 1) * CT_LEN].try_into().expect("ct"));
           }
           let mut capsules = Vec::with_capacity(PATH_RELAYS - 1);
           for i in 1..PATH_RELAYS {
               let s = HEADER_LEN + i * CAPSULE_LEN;
               capsules.push(packet.0[s..s + CAPSULE_LEN].try_into().expect("capsule"));
           }
           forward = Some(ForwardParts { cts, capsules, payload });
           Action::Relay
       }

        ACTION_DELIVER => {
            let commit =
                Hash256::from_bytes(meta[33..65].try_into().expect("commit"));
            if Hash256::concat(&SPHINX, &payload) != commit {
                return Err(SphinxError::CorruptPayload);
            }
            let len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
            if 2 + len > PAYLOAD_REGION {
                return Err(SphinxError::MalformedPayload);
            }
            Action::Deliver { data: payload[2..2 + len].to_vec() }
        }
        ACTION_DROP => Action::Drop,
        _ => return Err(SphinxError::BadCapsule),
    };
    
    Ok(Peeled { tag, next, action, forward })
}

pub fn forward(
    parts: &ForwardParts,
    ct_pad: &[u8; CT_LEN],
    capsule_pad: &[u8; CAPSULE_LEN],
) -> Result<Packet, SphinxError> {
    if parts.cts.len() != PATH_RELAYS - 1 || parts.capsules.len() != PATH_RELAYS - 1 {
        return Err(SphinxError::BadStructure {
            cts: parts.cts.len(),
            capsules: parts.capsules.len(),
        });
    }
    let mut bytes = [0u8; PACKET_BYTES];
    let mut at = 0usize;
    for ct in &parts.cts {
        bytes[at..at + CT_LEN].copy_from_slice(ct);
        at += CT_LEN;
    }
    bytes[at..at + CT_LEN].copy_from_slice(ct_pad);
    at += CT_LEN;
    for cap in &parts.capsules {
        bytes[at..at + CAPSULE_LEN].copy_from_slice(cap);
        at += CAPSULE_LEN;
    }
    bytes[at..at + CAPSULE_LEN].copy_from_slice(capsule_pad);
    at += CAPSULE_LEN;
    bytes[at..].copy_from_slice(&parts.payload);
    Ok(Packet(bytes))
}

// ---------------------------------------------------------------------------
// Fragmentation (D.2/D.6; erratum 151)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentFrame {
    pub txid: TxId,
    pub index: u8,
    pub count: u8,
    pub total_len: u32,
    pub data: Vec<u8>,
}

impl FragmentFrame {
    pub fn to_frame(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(1 + self.encoded_len());
        v.push(MIX_FRAGMENT_TAG);
        self.encode_into(&mut v);
        v
    }

    pub fn from_frame(payload: &[u8]) -> Result<FragmentFrame, CodecError> {
        if payload.len() < 2 || payload[0] != MIX_FRAGMENT_TAG {
            return Err(CodecError::InvariantViolated("mix fragment frame shape"));
        }
        let f = FragmentFrame::decode(&payload[1..])?;
        Ok(f)
    }
}

impl Encode for FragmentFrame {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self.txid.as_bytes());
        out.push(self.index);
        out.push(self.count);
        out.extend_from_slice(&self.total_len.to_le_bytes());
        out.extend_from_slice(&(self.data.len() as u16).to_le_bytes());
        out.extend_from_slice(&self.data);
    }
    fn encoded_len(&self) -> usize {
        FRAG_HEADER + self.data.len()
    }
}

impl Decode for FragmentFrame {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let txid = TxId::decode_from(r)?;
        let index = r.read_u8()?;
        let count = r.read_u8()?;
        if !FRAG_CLASSES.contains(&(count as usize)) {
            return Err(CodecError::InvariantViolated("fragment count is not a size class"));
        }
        if index >= count {
            return Err(CodecError::InvariantViolated("fragment index outside the count"));
        }
        let total_len = r.read_u32()?;
        if u64::from(total_len) > u64::from(count) * MAX_FRAG_DATA as u64 {
            return Err(CodecError::InvariantViolated("fragment total_len exceeds the class capacity"));
        }
        let data_len = r.read_u16()? as usize;
        if data_len != MAX_FRAG_DATA {
            return Err(CodecError::InvariantViolated("fragment data must fill the class"));
        }
        let data = r.take(data_len)?.to_vec();
        Ok(FragmentFrame { txid, index, count, total_len, data })
    }
}

pub fn parse_frame(bytes: &[u8]) -> Result<FragmentFrame, SphinxError> {
    Ok(FragmentFrame::decode(bytes)?)
}

pub fn fragment_class(len: usize) -> Result<usize, SphinxError> {
    let needed = (len + MAX_FRAG_DATA - 1) / MAX_FRAG_DATA;
    FRAG_CLASSES
        .iter()
        .copied()
        .find(|&c| c >= needed)
        .ok_or(SphinxError::FragmentTooLarge { len, max: 16 * MAX_FRAG_DATA })
}

pub fn fragment_with_class(
    payload: &[u8],
    txid: &TxId,
    class: usize,
) -> Result<Vec<FragmentFrame>, SphinxError> {
    if !FRAG_CLASSES.contains(&class) {
        return Err(SphinxError::BadClass { found: class });
    }
    if payload.len() > class * MAX_FRAG_DATA {
        return Err(SphinxError::FragmentTooLarge {
            len: payload.len(),
            max: class * MAX_FRAG_DATA,
        });
    }
    
    let mut data = payload.to_vec();
    data.resize(class * MAX_FRAG_DATA, 0);
    let total_len = payload.len() as u32;
    Ok((0..class)
        .map(|i| FragmentFrame {
            txid: *txid,
            index: i as u8,
            count: class as u8,
            total_len,
            data: data[i * MAX_FRAG_DATA..(i + 1) * MAX_FRAG_DATA].to_vec(),
        })
        .collect())
}

pub fn fragment(payload: &[u8], txid: &TxId) -> Result<Vec<FragmentFrame>, SphinxError> {
    fragment_with_class(payload, txid, fragment_class(payload.len())?)
}

pub fn reassemble(frames: &[FragmentFrame]) -> Option<Vec<u8>> {
    let first = frames.first()?;
    let count = first.count as usize;
    if frames.len() != count || count == 0 {
        return None;
    }
    let mut by_index = std::collections::BTreeMap::new();
    for f in frames {
        if f.txid != first.txid
            || f.count != first.count
            || f.total_len != first.total_len
            || f.data.len() != MAX_FRAG_DATA
        {
            return None;
        }
        if by_index.insert(f.index, f).is_some() {
            return None;
        }
    }
    let mut out = Vec::with_capacity(count * MAX_FRAG_DATA);
    for i in 0..count {
        by_index.get(&(i as u8))?;
        out.extend_from_slice(&by_index[&(i as u8)].data);
    }
    out.truncate(first.total_len as usize);
    Some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::{node_keys, SplitMix64};

    fn relays(n: u64) -> Vec<(PeerId, EncapsulationKey, DecapsulationKey)> {
        (1..=n)
            .map(|i| {
                let (sk, ek, dk) = node_keys(i);
                (PeerId::of(sk.verifying_key()), ek, dk)
            })
            .collect()
    }

    fn encap_rand(seed: u64) -> [[u8; ENCAPS_LEN]; PATH_RELAYS] {
        let mut rng = SplitMix64::new(seed);
        let mut out = [[0u8; ENCAPS_LEN]; PATH_RELAYS];
        for r in &mut out {
            for c in r.chunks_mut(8) {
                c.copy_from_slice(&rng.next_u64().to_le_bytes());
            }
        }
        out
    }

    fn pads(seed: u64) -> ([u8; CT_LEN], [u8; CAPSULE_LEN]) {
        let mut rng = SplitMix64::new(seed ^ 0x9AD);
        let mut ct = [0u8; CT_LEN];
        for c in ct.chunks_mut(8) {
            c.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        let mut cap = [0u8; CAPSULE_LEN];
        for c in cap.chunks_mut(8) {
            c.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        (ct, cap)
    }

    fn txid(seed: u64) -> TxId {
        TxId::from_hash(Hash256::from_bytes(SplitMix64::new(seed).bytes32()))
    }

    fn spec_for(
        relays: &[(PeerId, EncapsulationKey, DecapsulationKey)],
        dest: PeerId,
        terminal: TerminalAction,
    ) -> PathSpec {
        let arr: [(PeerId, EncapsulationKey); PATH_RELAYS] =
            core::array::from_fn(|i| (relays[i].0, relays[i].1.clone()));
        PathSpec::new(arr, dest, terminal)
    }

    #[test]
    fn size_pins() {
        assert_eq!(PACKET_BYTES, 20_000);
        assert_eq!(HEADER_LEN, 5_440);
        assert_eq!(META_LEN, 65);
        assert_eq!(CAPSULE_LEN, 81);
        assert_eq!(META_REGION, 405);
        assert_eq!(PAYLOAD_REGION, 14_155);
        assert_eq!(MAX_DATA, 14_153);
        assert_eq!(FRAG_HEADER, 40);
        assert_eq!(MAX_FRAG_DATA, 14_113);
        assert_eq!(16 * MAX_FRAG_DATA, 225_808);
    }

    #[test]
    fn build_peel_the_full_path() {
        let r = relays(5);
        let (sk_d, _, _) = node_keys(99);
        let dest = PeerId::of(sk_d.verifying_key());
        let spec = spec_for(&r, dest, TerminalAction::Deliver);
        let data = b"the submission payload".to_vec();
        let packet = build(&spec, &data, &encap_rand(1)).unwrap();

        let mut current = packet.clone();
        let mut tags = Vec::new();
        for hop in 0..4 {
            let peeled = peel(&current, &r[hop].2).unwrap();
            assert!(matches!(peeled.action, Action::Relay), "hop {hop}");
            assert_eq!(peeled.next, r[hop + 1].0);
            tags.push(peeled.tag);
            let (ct_pad, cap_pad) = pads(50 + hop as u64);
            current = forward(peeled.forward.as_ref().unwrap(), &ct_pad, &cap_pad).unwrap();
            assert_eq!(current.0.len(), PACKET_BYTES);
        }
        let last = peel(&current, &r[4].2).unwrap();
        assert_eq!(last.next, dest);
        match last.action {
            Action::Deliver { data: got } => assert_eq!(got, data),
            other => panic!("terminal action: {other:?}"),
        }
        tags.push(last.tag);
        let flat: Vec<[u8; 16]> = tags;
        for i in 0..flat.len() {
            for j in i + 1..flat.len() {
                assert_ne!(flat[i], flat[j], "replay tags must be per-hop-distinct");
            }
        }
    }

    #[test]
    fn determinism_and_randomness_sensitivity() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Drop);
        let a = build(&spec, b"x", &encap_rand(2)).unwrap();
        let b = build(&spec, b"x", &encap_rand(2)).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, build(&spec, b"x", &encap_rand(3)).unwrap());
        assert_ne!(a, build(&spec, b"y", &encap_rand(2)).unwrap());
    }

    #[test]
    fn wrong_relay_and_mid_path_injection() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Deliver);
        let packet = build(&spec, b"data", &encap_rand(4)).unwrap();

        // An unrelated relay at hop-1 alignment.
        assert!(matches!(peel(&packet, &r[3].2), Err(SphinxError::BadCapsule)));

        // A hop-2-aligned packet given to relay 4 (skipping relay 2).
        let p1 = peel(&packet, &r[0].2).unwrap();
        let (ct_pad, cap_pad) = pads(60);
        let fwd = forward(p1.forward.as_ref().unwrap(), &ct_pad, &cap_pad).unwrap();
        assert!(matches!(peel(&fwd, &r[3].2), Err(SphinxError::BadCapsule)));
        assert!(peel(&fwd, &r[1].2).is_ok());
    }

    #[test]
    fn tampered_regions_rejected() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Deliver);
        let mut packet = build(&spec, b"data", &encap_rand(5)).unwrap();

        let pristine = packet.clone();
        packet.0[10] ^= 1; // ct slot 0
        assert!(matches!(peel(&packet, &r[0].2), Err(SphinxError::BadCapsule)));
        packet = pristine.clone();
        packet.0[HEADER_LEN + 3] ^= 1; // capsule 0
        assert!(matches!(peel(&packet, &r[0].2), Err(SphinxError::BadCapsule)));
        assert!(peel(&pristine, &r[0].2).is_ok());
    }

    #[test]
    fn corrupted_payload_detected_at_terminal() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Deliver);
        let mut packet = build(&spec, b"payload bytes", &encap_rand(6)).unwrap();
        // Flip a bit in the payload region: intermediate relays forward it
        // blindly (their capsules are intact); the terminal detects.
        packet.0[HEADER_LEN + META_REGION + 700] ^= 1;
        let mut current = packet;
        for hop in 0..4 {
            let peeled = peel(&current, &r[hop].2).unwrap();
            let (ct_pad, cap_pad) = pads(70 + hop as u64);
            current = forward(peeled.forward.as_ref().unwrap(), &ct_pad, &cap_pad).unwrap();
        }
        assert!(matches!(peel(&current, &r[4].2), Err(SphinxError::CorruptPayload)));
    }

    #[test]
    fn cover_terminal_drops() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Drop);
        let packet = build(&spec, &[0u8; 64], &encap_rand(7)).unwrap();
        let mut current = packet;
        for hop in 0..4 {
            let peeled = peel(&current, &r[hop].2).unwrap();
            let (ct_pad, cap_pad) = pads(80 + hop as u64);
            current = forward(peeled.forward.as_ref().unwrap(), &ct_pad, &cap_pad).unwrap();
        }
        let last = peel(&current, &r[4].2).unwrap();
        assert_eq!(last.action, Action::Drop);
        assert!(last.forward.is_none());
    }

    #[test]
    fn payload_capacity() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Deliver);
        let full = vec![0xA5u8; MAX_DATA];
        let p = build(&spec, &full, &encap_rand(8)).unwrap();
        let mut current = p;
        for hop in 0..4 {
            let peeled = peel(&current, &r[hop].2).unwrap();
            let (ct_pad, cap_pad) = pads(90 + hop as u64);
            current = forward(peeled.forward.as_ref().unwrap(), &ct_pad, &cap_pad).unwrap();
        }
        match peel(&current, &r[4].2).unwrap().action {
            Action::Deliver { data } => assert_eq!(data.len(), MAX_DATA),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            build(&spec, &vec![0u8; MAX_DATA + 1], &encap_rand(8)),
            Err(SphinxError::PayloadTooLarge { len, max }) if len == MAX_DATA + 1 && max == MAX_DATA
        ));
    }

    #[test]
    fn replay_tag_stability() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Drop);
        let a = build(&spec, b"x", &encap_rand(9)).unwrap();
        let b = build(&spec, b"x", &encap_rand(9)).unwrap();
        let c = build(&spec, b"y", &encap_rand(10)).unwrap();
        let ct0: [u8; CT_LEN] = a.0[..CT_LEN].try_into().unwrap();
        let cap0 = &a.0[HEADER_LEN..HEADER_LEN + CAPSULE_LEN];
        let t1 = replay_tag(&ct0, cap0);
        let t2 = replay_tag(&ct0, cap0);
        assert_eq!(t1, t2);
        let ct0b: [u8; CT_LEN] = c.0[..CT_LEN].try_into().unwrap();
        let cap0b = &c.0[HEADER_LEN..HEADER_LEN + CAPSULE_LEN];
        assert_ne!(t1, replay_tag(&ct0b, cap0b));
        assert_eq!(peel(&a, &r[0].2).unwrap().tag, t1);
        assert_eq!(peel(&b, &r[0].2).unwrap().tag, t1, "same randomness → same tag");
        let _ = ct0b;
    }

    #[test]
    fn packet_codec_roundtrip() {
        let r = relays(5);
        let spec = spec_for(&r, r[0].0, TerminalAction::Drop);
        let p = build(&spec, b"z", &encap_rand(11)).unwrap();
        let enc = p.encode();
        assert_eq!(enc.len(), PACKET_BYTES);
        assert_eq!(Packet::decode(&enc).unwrap(), p);
        assert!(Packet::decode(&enc[..PACKET_BYTES - 1]).is_err());
        let frame = p.to_frame();
        assert_eq!(frame.len(), 1 + PACKET_BYTES);
        assert_eq!(frame[0], MIX_PACKET_TAG);
        assert_eq!(Packet::from_frame(&frame).unwrap(), p);
        assert!(Packet::from_frame(&frame[..frame.len() - 1]).is_err());
        assert!(Packet::from_frame(&[9u8, 0]).is_err());
    }

    #[test]
    fn class_selection() {
        assert_eq!(fragment_class(0).unwrap(), 1);
        assert_eq!(fragment_class(1).unwrap(), 1);
        assert_eq!(fragment_class(MAX_FRAG_DATA).unwrap(), 1);
        assert_eq!(fragment_class(MAX_FRAG_DATA + 1).unwrap(), 2);
        assert_eq!(fragment_class(2 * MAX_FRAG_DATA).unwrap(), 2);
        assert_eq!(fragment_class(2 * MAX_FRAG_DATA + 1).unwrap(), 4);
        assert_eq!(fragment_class(5 * MAX_FRAG_DATA).unwrap(), 8);
        assert_eq!(fragment_class(9 * MAX_FRAG_DATA).unwrap(), 16);
        assert!(matches!(
            fragment_class(16 * MAX_FRAG_DATA + 1),
            Err(SphinxError::FragmentTooLarge { .. })
        ));
        assert!(matches!(
            fragment_with_class(b"x", &txid(1), 3),
            Err(SphinxError::BadClass { found: 3 })
        ));
    }

    #[test]
    fn fragmentation_roundtrip() {
        let mut rng = SplitMix64::new(0xFFA);
        let payload: Vec<u8> = (0..100_000).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let t = txid(2);
        let frames = fragment(&payload, &t).unwrap();
        assert_eq!(frames.len(), fragment_class(100_000).unwrap());
        for f in &frames {
            assert_eq!(f.data.len(), MAX_FRAG_DATA);
            assert_eq!(f.total_len as usize, payload.len());
        }
        let mut shuffled = frames.clone();
        for i in (1..shuffled.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        assert_eq!(reassemble(&shuffled).unwrap(), payload);
    }

    #[test]
    fn reassembly_failures() {
        let t = txid(3);
        let payload = vec![7u8; 3 * MAX_FRAG_DATA - 500];
        let frames = fragment(&payload, &t).unwrap();
        assert_eq!(frames.len(), 4);

        // Missing one frame.
        let partial: Vec<FragmentFrame> = frames[..3].to_vec();
        assert!(reassemble(&partial).is_none());

        // Duplicate index.
        let mut dup = frames[..3].to_vec();
        dup.push(frames[2].clone());
        assert!(reassemble(&dup).is_none());

        // Foreign txid mixed in.
        let mut foreign = frames[..3].to_vec();
        let mut f = frames[3].clone();
        f.txid = txid(4);
        foreign.push(f);
        assert!(reassemble(&foreign).is_none());

        // Count / total_len mismatch.
        let mut mismatch = frames[..3].to_vec();
        let mut f = frames[3].clone();
        f.total_len += 1;
        mismatch.push(f);
        assert!(reassemble(&mismatch).is_none());

        // Short data (violates the fill invariant).
        let mut short = frames[..3].to_vec();
        let mut f = frames[3].clone();
        f.data.truncate(10);
        short.push(f);
        assert!(reassemble(&short).is_none());

        assert!(reassemble(&[]).is_none());
        assert_eq!(reassemble(&frames).unwrap(), payload);
    }

    #[test]
    fn frame_codec_strictness() {
        let t = txid(5);
        let frames = fragment(b"payload", &t).unwrap();
        let f = &frames[0];
        let enc = f.encode();
        assert_eq!(enc.len(), FRAG_HEADER + MAX_FRAG_DATA);
        assert_eq!(parse_frame(&enc).unwrap(), *f);

        let mut bad = enc.clone();
        bad[32] = 5; // index ≥ count
        assert!(parse_frame(&bad).is_err());
        let mut bad = enc.clone();
        bad[33] = 3; // count not a class
        assert!(parse_frame(&bad).is_err());
        let mut bad = enc.clone();
        bad[34..38].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_frame(&bad).is_err());
        let mut bad = enc.clone();
        let dl = enc.len() - 2;
        bad[dl..dl + 2].copy_from_slice(&7u16.to_le_bytes());
        assert!(parse_frame(&bad).is_err());
        assert!(parse_frame(&enc[..enc.len() - 1]).is_err());

        let frame = f.to_frame();
        assert_eq!(frame[0], MIX_FRAGMENT_TAG);
        assert_eq!(FragmentFrame::from_frame(&frame).unwrap(), *f);
    }

    #[test]
    fn cover_class_padding_up() {
        // A wallet may deliberately pad up a class for cover.
        let t = txid(6);
        let frames = fragment_with_class(b"tiny", &t, 8).unwrap();
        assert_eq!(frames.len(), 8);
        assert_eq!(reassemble(&frames).unwrap(), b"tiny");
    }
}
