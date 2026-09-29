//! The staked relay registry (WP §6.2; erratum 152): ML-DSA-attested
//! ML-KEM relay records and the wallet's diversified path selection.
//! Stake escrow and slashing are nerv-economy's (chunk 17); the on-chain
//! commitment is the consensus wiring. Selection is wallet-local policy
//! (E-007) — never canonical.

use std::collections::BTreeMap;
use std::net::SocketAddr;

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::constants::RELAY_REG;
use nerv_core::error::CodecError;
use nerv_core::hash::Xof;
use nerv_crypto::mlkem::EncapsulationKey;
use nerv_crypto::mldsa::{Signature, SigningKey, VerifyingKey};

use crate::host::PeerId;

pub const MAX_ADDRS: usize = 16;
pub const OPERATOR_LEN: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RegistryError {
    #[error("relay registration signature failed")]
    BadSignature,
    #[error("the signing key does not match the record's verifying key")]
    KeyMismatch,
    #[error("codec: {0}")]
    Codec(#[from] CodecError),
    #[error("registry holds {have} relays, need {need}")]
    TooFewRelays { have: usize, need: usize },
    #[error("crypto: {0}")]
    Crypto(#[from] nerv_crypto::CryptoError),
    #[error("internal: {0}")]
    Internal(&'static str),
}

fn encode_addr(a: &SocketAddr, out: &mut Vec<u8>) {
    match a {
        SocketAddr::V4(v) => {
            out.push(4);
            out.extend_from_slice(&v.ip().octets());
        }
        SocketAddr::V6(v) => {
            out.push(6);
            out.extend_from_slice(&v.ip().octets());
        }
    }
    out.extend_from_slice(&a.port().to_le_bytes());
}

fn decode_addr(r: &mut Reader<'_>) -> Result<SocketAddr, CodecError> {
    match r.read_u8()? {
        4 => {
            let oct = r.take_array::<4>()?;
            let port = r.read_u16()?;
            Ok(SocketAddr::from(([oct[0], oct[1], oct[2], oct[3]], port)))
        }
        6 => {
            let oct = r.take_array::<16>()?;
            let port = r.read_u16()?;
            Ok(SocketAddr::from((oct, port)))
        }
        tag => Err(CodecError::InvalidOptionTag { tag }),
    }
}

/// One staked relay: identity, the PQ delivery key, the addresses, the
/// operator tag, and the stake amount (data — the escrow is the economy's).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelayRecord {
    pub vk: VerifyingKey,
    pub kem_ek: EncapsulationKey,
    pub addrs: Vec<SocketAddr>,
    pub operator: [u8; OPERATOR_LEN],
    pub stake_nerv: u64,
}

impl RelayRecord {
    pub fn id(&self) -> PeerId {
        PeerId::of(&self.vk)
    }
}

impl Encode for RelayRecord {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.vk.encode_into(out);
        self.kem_ek.encode_into(out);
        out.extend_from_slice(&(self.addrs.len() as u32).to_le_bytes());
        for a in &self.addrs {
            encode_addr(a, out);
        }
        out.extend_from_slice(&self.operator);
        out.extend_from_slice(&self.stake_nerv.to_le_bytes());
    }
    fn encoded_len(&self) -> usize {
        self.vk.encoded_len()
            + self.kem_ek.encoded_len()
            + 4
            + self.addrs.iter().map(|a| 3 + 16).sum::<usize>()
            + OPERATOR_LEN
            + 8
    }
}

impl Decode for RelayRecord {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let vk = VerifyingKey::decode_from(r)?;
        let kem_ek = EncapsulationKey::decode_from(r)?;
        let n = r.read_seq_len()?;
        if n > MAX_ADDRS {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_ADDRS });
        }
        let mut addrs = Vec::with_capacity(n);
        for _ in 0..n {
            addrs.push(decode_addr(r)?);
        }
        let operator = r.take_array::<OPERATOR_LEN>()?;
        let stake_nerv = r.read_u64()?;
        Ok(RelayRecord { vk, kem_ek, addrs, operator, stake_nerv })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedRelayRecord {
    pub record: RelayRecord,
    pub signature: Signature,
}

fn registration_message(record: &RelayRecord) -> Vec<u8> {
    let mut m = Vec::with_capacity(RELAY_REG.as_bytes().len() + record.encoded_len());
    m.extend_from_slice(RELAY_REG.as_bytes());
    record.encode_into(&mut m);
    m
}

impl SignedRelayRecord {
    pub fn register(sk: &SigningKey, record: RelayRecord) -> Result<SignedRelayRecord, RegistryError> {
        if sk.verifying_key() != &record.vk {
            return Err(RegistryError::KeyMismatch);
        }
        let signature = sk.sign(&registration_message(&record))?;
        Ok(SignedRelayRecord { record, signature })
    }

    pub fn verify(&self) -> bool {
        self.record.vk.verify(&registration_message(&self.record), &self.signature)
    }

    pub fn id(&self) -> PeerId {
        self.record.id()
    }
}

impl Encode for SignedRelayRecord {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.record.encode_into(out);
        self.signature.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.record.encoded_len() + self.signature.encoded_len()
    }
}

impl Decode for SignedRelayRecord {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let record = RelayRecord::decode_from(r)?;
        let signature = Signature::decode_from(r)?;
        Ok(SignedRelayRecord { record, signature })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RelayRegistry {
    entries: BTreeMap<PeerId, SignedRelayRecord>,
}

impl RelayRegistry {
    pub fn new() -> RelayRegistry {
        RelayRegistry::default()
    }

    /// Insert (or replace) a record; the signature is verified first —
    /// the registry only ever holds self-certified entries.
    pub fn insert(&mut self, signed: SignedRelayRecord) -> Result<PeerId, RegistryError> {
        if !signed.verify() {
            return Err(RegistryError::BadSignature);
        }
        let id = signed.id();
        self.entries.insert(id, signed);
        Ok(id)
    }

    pub fn get(&self, id: &PeerId) -> Option<&SignedRelayRecord> {
        self.entries.get(id)
    }

    pub fn remove(&mut self, id: &PeerId) -> Option<SignedRelayRecord> {
        self.entries.remove(id)
    }

    pub fn contains(&self, id: &PeerId) -> bool {
        self.entries.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = PeerId> + '_ {
        self.entries.keys().copied()
    }
}

fn shuffle<T>(v: &mut [T], xof: &mut Xof) {
    // Wallet-local policy: the modulo bias is ≤ 2⁻⁶⁴ at relay-roster
    // sizes and selection is never canonical (erratum 152).
    for i in (1..v.len()).rev() {
        let j = (xof.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// The wallet's path selection: one XOF stream over the wallet seed,
/// Fisher–Yates within operator groups, round-robin across groups —
/// deterministic in (seed, registry content), at most one relay per
/// operator while the roster allows (erratum 152).
pub fn select(
    registry: &RelayRegistry,
    wallet_seed: &[u8; 32],
    count: usize,
) -> Result<Vec<SignedRelayRecord>, RegistryError> {
    if registry.len() < count {
        return Err(RegistryError::TooFewRelays { have: registry.len(), need: count });
    }
    let mut groups: BTreeMap<[u8; OPERATOR_LEN], Vec<SignedRelayRecord>> = BTreeMap::new();
    for r in registry.entries.values() {
        groups.entry(r.record.operator).or_default().push(r.clone());
    }
    let mut xof = Xof::new(&RELAY_REG, wallet_seed);
    for g in groups.values_mut() {
        shuffle(g, &mut xof);
    }
    let group_list: Vec<Vec<SignedRelayRecord>> = groups.into_values().collect();
    let mut cursors = vec![0usize; group_list.len()];
    let mut out = Vec::with_capacity(count);
    loop {
        let mut progressed = false;
        for (gi, g) in group_list.iter().enumerate() {
            if out.len() == count {
                break;
            }
            if cursors[gi] < g.len() {
                out.push(g[cursors[gi]].clone());
                cursors[gi] += 1;
                progressed = true;
            }
        }
        if out.len() == count || !progressed {
            break;
        }
    }
    if out.len() != count {
        return Err(RegistryError::TooFewRelays { have: out.len(), need: count });
    }
    Ok(out)
}

/// The sphinx-ready path: [(PeerId, ek); 5] for PathSpec::new.
pub fn select_path(
    registry: &RelayRegistry,
    wallet_seed: &[u8; 32],
) -> Result<[(PeerId, EncapsulationKey); crate::sphinx::PATH_RELAYS], RegistryError> {
    let s = select(registry, wallet_seed, crate::sphinx::PATH_RELAYS)?;
    Ok(core::array::from_fn(|i| (s[i].record.id(), s[i].record.kem_ek.clone())))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::node_keys;

    fn record(seed: u64, operator: [u8; OPERATOR_LEN], port: u16) -> SignedRelayRecord {
        let (sk, ek, _) = node_keys(seed);
        let rec = RelayRecord {
            vk: *sk.verifying_key(),
            kem_ek: ek,
            addrs: vec![format!("10.0.0.1:{port}").parse().unwrap()],
            operator,
            stake_nerv: 1_000,
        };
        SignedRelayRecord::register(&sk, rec).unwrap()
    }

    fn op(tag: u8) -> [u8; OPERATOR_LEN] {
        let mut o = [0u8; OPERATOR_LEN];
        o[0] = tag;
        o
    }

    #[test]
    fn registration_and_verification() {
        let (sk, _, _) = node_keys(1);
        let (other, ek, _) = node_keys(2);
        let rec = RelayRecord {
            vk: *sk.verifying_key(),
            kem_ek: ek,
            addrs: vec!["127.0.0.1:9000".parse().unwrap()],
            operator: op(1),
            stake_nerv: 5,
        };
        assert!(matches!(
            SignedRelayRecord::register(&other, rec.clone()),
            Err(RegistryError::KeyMismatch)
        ));
        let signed = SignedRelayRecord::register(&sk, rec).unwrap();
        assert!(signed.verify());

        // Every tampered field breaks the signature.
        for tamper in [
            |r: &mut RelayRecord| r.stake_nerv += 1,
            |r: &mut RelayRecord| r.operator[0] ^= 1,
            |r: &mut RelayRecord| r.addrs.push("127.0.0.1:1".parse().unwrap()),
        ] {
            let mut bad = signed.clone();
            tamper(&mut bad.record);
            assert!(!bad.verify());
        }
        let mut bad = signed.clone();
        let mut sb = *bad.signature.as_bytes();
        sb[100] ^= 1;
        bad.signature = nerv_crypto::mldsa::Signature::from_bytes(sb);
        assert!(!bad.verify());
    }

    #[test]
    fn registry_insert_lookup_remove() {
        let mut reg = RelayRegistry::new();
        assert!(reg.is_empty());
        let a = record(3, op(1), 9000);
        let id = a.id();
        reg.insert(a.clone()).unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(&id));
        assert_eq!(reg.get(&id).unwrap(), &a);

        // A forged record is rejected at insert.
        let (_, ek_f, _) = node_keys(4);
        let mut forged = a.clone();
        forged.record.kem_ek = ek_f;
        assert!(matches!(reg.insert(forged), Err(RegistryError::BadSignature)));

        // Replacement by the same id.
        let replaced = record(3, op(1), 9001);
        reg.insert(replaced.clone()).unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.get(&id).unwrap(), &replaced);

        assert!(reg.remove(&id).is_some());
        assert!(reg.is_empty() && reg.get(&id).is_none());
    }

    fn populated() -> RelayRegistry {
        let mut reg = RelayRegistry::new();
        // 3 operators × 3 relays.
        for k in 0..9u64 {
            let seed = 10 + k;
            let operator = op((k % 3) as u8 + 1);
            reg.insert(record(seed, operator, 9000 + k as u16)).unwrap();
        }
        reg
    }

    #[test]
    fn selection_determinism_distinctness_spread() {
        let reg = populated();
        let seed = [7u8; 32];
        let a = select(&reg, &seed, 5).unwrap();
        let b = select(&reg, &seed, 5).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
        let ids: Vec<PeerId> = a.iter().map(|r| r.id()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "no relay twice");
        // Operator spread: the first three picks are distinct operators.
        let ops: Vec<[u8; OPERATOR_LEN]> = a[..3].iter().map(|r| r.record.operator).collect();
        assert_ne!(ops[0], ops[1]);
        assert_ne!(ops[0], ops[2]);
        assert_ne!(ops[1], ops[2]);
        // All picks are from the registry.
        for r in &a {
            assert!(reg.contains(&r.id()));
        }
    }

    #[test]
    fn selection_sensitivity() {
        let reg = populated();
        let a = select(&reg, &[1u8; 32], 5).unwrap();
        let b = select(&reg, &[2u8; 32], 5).unwrap();
        let ids_a: Vec<PeerId> = a.iter().map(|r| r.id()).collect();
        let ids_b: Vec<PeerId> = b.iter().map(|r| r.id()).collect();
        assert_ne!(ids_a, ids_b, "seed sensitivity");

        let mut smaller = populated();
        let victim = smaller.ids().next().unwrap();
        smaller.remove(&victim).unwrap();
        let c = select(&smaller, &[1u8; 32], 5).unwrap();
        assert!(!c.iter().any(|r| r.id() == victim), "roster sensitivity");
    }

    #[test]
    fn selection_too_few_and_single_operator() {
        let mut small = RelayRegistry::new();
        for k in 0..3u64 {
            small.insert(record(30 + k, op(9), 9100 + k as u16)).unwrap();
        }
        assert!(matches!(
            select(&small, &[1u8; 32], 5),
            Err(RegistryError::TooFewRelays { have: 3, need: 5 })
        ));
        // One operator, 5 relays: round-robin degenerates to sequence —
        // still 5 distinct.
        let mut mono = RelayRegistry::new();
        for k in 0..5u64 {
            mono.insert(record(40 + k, op(1), 9200 + k as u16)).unwrap();
        }
        let picked = select(&mono, &[1u8; 32], 5).unwrap();
        assert_eq!(picked.len(), 5);
    }

    #[test]
    fn select_path_shape() {
        let reg = populated();
        let path = select_path(&reg, &[5u8; 32]).unwrap();
        assert_eq!(path.len(), 5);
        for (id, ek) in &path {
            let r = reg.get(id).unwrap();
            assert_eq!(&r.record.kem_ek, ek);
        }
        // Determinism.
        assert_eq!(select_path(&reg, &[5u8; 32]).unwrap(), path);
    }

    #[test]
    fn record_codec_roundtrip() {
        let a = record(50, op(2), 9300);
        let enc = a.record.encode();
        assert_eq!(enc.len(), a.record.encoded_len());
        assert_eq!(RelayRecord::decode(&enc).unwrap(), a.record);
        assert!(RelayRecord::decode(&enc[..enc.len() - 1]).is_err());

        let enc = a.encode();
        assert_eq!(enc.len(), a.encoded_len());
        let dec = SignedRelayRecord::decode(&enc).unwrap();
        assert_eq!(dec, a);
        assert!(dec.verify());
        assert!(SignedRelayRecord::decode(&enc[..enc.len() - 1]).is_err());

        // V6 addresses round-trip.
        let (sk, ek, _) = node_keys(51);
        let rec = RelayRecord {
            vk: *sk.verifying_key(),
            kem_ek: ek,
            addrs: vec!["[2001:db8::1]:443".parse().unwrap()],
            operator: op(3),
            stake_nerv: 0,
        };
        let signed = SignedRelayRecord::register(&sk, rec).unwrap();
        assert_eq!(SignedRelayRecord::decode(&signed.encode()).unwrap(), signed);

        // Unknown address family rejected.
        let mut bad = a.record.encode();
        let at = a.record.vk.encoded_len() + a.record.kem_ek.encoded_len() + 4;
        bad[at] = 9;
        assert!(RelayRecord::decode(&bad).is_err());
    }
}
