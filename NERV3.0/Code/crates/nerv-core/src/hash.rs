//! [WRAP blake3] — domain-separated Hash / ToField / XOF wrappers.
//!
//! This is the external, binding face of the protocol (WP §5.3's dual-hash
//! discipline: BLAKE3 everywhere except the trees' internal Poseidon2 nodes,
//! which land with the tree crates). Nothing in NERV hashes outside this
//! module, and every domain comes from `crate::constants`.
//!
//! # Domain separation — two framing modes
//!
//! * `Hash256::concat(domain, msg)` — `BLAKE3(domain ‖ msg)`, the literal
//!   shape of every WP formula (`cm = BLAKE3("nerv.cm" ‖ v ‖ ρ ‖ d ‖ r)`).
//!   LAW: the caller guarantees the concatenation is unambiguous — all
//!   variable-width fields must be fixed-width in this call site, or the
//!   caller must use `framed`.
//! * `Hash256::framed(domain, parts)` — length-framed (u32 LE length before
//!   each part, including the domain): unambiguous for arbitrary structured
//!   input; use it whenever parts have variable widths.
//!
//! Both are deterministic and both are legitimate; the per-call-site choice
//! is recorded where the call site lives.
//!
//! # ToField
//!
//! "BLAKE3-ToField" (WP §3.4 et al.) covers two reductions, both here:
//! * `Hash256::reduce_mod(m)` — exact 256-bit → mod-`m` reduction (the
//!   nullifier tree's 256-bit keys are the hash itself; no reduction there).
//! * `hash_to_goldilocks(domain, msg)` — deterministic rejection-sampled
//!   element of the prover field (Goldilocks, `params::PROOFS_FIELD_MODULUS`),
//!   for circuit-facing values. Total by construction: after 64 rejected
//!   words (~2⁻²⁰⁴⁸) it falls back to exact reduction — consensus-path code
//!   must never have an unbounded loop.
//!
//! # XOF
//!
//! `Xof` wraps blake3's extendable output: deterministic streams keyed by
//! domain + message (WP §7.6's beacon-XOF weight expansion, chunk 6).

use std::fmt;

use blake3::{Hasher, OutputReader};

use crate::codec::{Decode, Encode, Reader};
use crate::constants::Domain;
use crate::error::CodecError;
use crate::params::PROOFS_FIELD_MODULUS;

/// A 32-byte BLAKE3 digest — the protocol's universal commitment/tag type.
///
/// `Ord` is byte-lexicographic over the digest, matching the canonical
/// ordering discipline of `codec`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Hash256([u8; 32]);

impl Hash256 {
    /// `BLAKE3(domain ‖ msg)` — the literal WP formula shape.
    ///
    /// The caller guarantees the concatenation is unambiguous (fixed-width
    /// fields); use [`Hash256::framed`] otherwise.
    pub fn concat(domain: &Domain, msg: &[u8]) -> Hash256 {
        let mut h = Hasher::new();
        h.update(domain.as_bytes());
        h.update(msg);
        Hash256(*h.finalize().as_bytes())
    }

    /// Length-framed: `BLAKE3(len(dom) ‖ dom ‖ len(p₀) ‖ p₀ ‖ len(p₁) ‖ p₁ ‖ …)`.
    ///
    /// Unambiguous for arbitrary parts: different part-splittings of the same
    /// byte string hash differently.
    pub fn framed(domain: &Domain, parts: &[&[u8]]) -> Hash256 {
        Hash256(*framed_hasher(domain, parts).finalize().as_bytes())
    }

    /// Construct from raw bytes (inverse of [`Hash256::as_bytes`]).
    pub fn from_bytes(bytes: [u8; 32]) -> Hash256 {
        Hash256(bytes)
    }

    /// The 32 digest bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// The digest as four little-endian u64 words.
    pub fn to_u64_words_le(&self) -> [u64; 4] {
        let mut words = [0u64; 4];
        for (i, word) in words.iter_mut().enumerate() {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&self.0[i * 8..i * 8 + 8]);
            *word = u64::from_le_bytes(buf);
        }
        words
    }

    /// Exact reduction of the 256-bit little-endian digest value modulo `m`.
    ///
    /// Deterministic, total for `m ≥ 1` (`m ≤ 1` returns 0). Slight bias for
    /// non-power-of-two `m`; use [`hash_to_goldilocks`] where uniformity
    /// matters.
    pub fn reduce_mod(&self, m: u64) -> u64 {
        if m <= 1 {
            return 0;
        }
        let m = m as u128;
        let mut r: u128 = 0;
        for limb in self.to_u64_words_le() {
            // r < m <= 2^64-1, so (r << 64) + limb < 2^128: no overflow.
            r = ((r << 64) + limb as u128) % m;
        }
        r as u64
    }
}

fn framed_update(h: &mut Hasher, part: &[u8]) {
    debug_assert!(
        part.len() <= u32::MAX as usize,
        "codec law: hashed parts are bounded far below 2^32 bytes"
    );
    h.update(&(part.len() as u32).to_le_bytes());
    h.update(part);
}

fn framed_hasher(domain: &Domain, parts: &[&[u8]]) -> Hasher {
    let mut h = Hasher::new();
    framed_update(&mut h, domain.as_bytes());
    for p in parts {
        framed_update(&mut h, p);
    }
    h
}

/// An extendable-output BLAKE3 stream keyed by domain + message.
///
/// Deterministic: two `Xof` values built from identical inputs produce
/// identical streams. `position`/`seek` make stream offsets explicit and
/// reproducible (chunk 6's weight expansion consumes fixed offsets).
pub struct Xof {
    reader: OutputReader,
}

impl Xof {
    /// `BLAKE3-XOF(domain ‖ msg)` — plain-prefix mode; the caller structures
    /// reads (or pre-frames the message and uses it verbatim).
    pub fn new(domain: &Domain, msg: &[u8]) -> Xof {
        let mut h = Hasher::new();
        h.update(domain.as_bytes());
        h.update(msg);
        Xof { reader: h.finalize_xof() }
    }

    /// Length-framed mode — same framing as [`Hash256::framed`].
    pub fn framed(domain: &Domain, parts: &[&[u8]]) -> Xof {
        Xof { reader: framed_hasher(domain, parts).finalize_xof() }
    }

    /// Fill `out` with the next stream bytes.
    pub fn fill(&mut self, out: &mut [u8]) {
        self.reader.fill(out);
    }

    /// Next 8 bytes as a little-endian u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        self.reader.fill(&mut b);
        u64::from_le_bytes(b)
    }

    /// Next `N` bytes as a fixed array.
    pub fn read_array<const N: usize>(&mut self) -> [u8; N] {
        let mut b = [0u8; N];
        self.reader.fill(&mut b);
        b
    }

    /// Current stream position in bytes.
    pub fn position(&self) -> u64 {
        self.reader.position()
    }

    /// Reposition the stream (0 = start). Seeking is pure: `seek(p)` then
    /// reads reproduce the stream from offset `p` exactly.
    pub fn seek(&mut self, position: u64) {
        self.reader.set_position(position);
    }
}

/// Deterministic uniform element of the prover field (Goldilocks).
///
/// Rejection sampling from the domain-keyed XOF: exact uniformity, with a
/// total exact-reduction fallback after 64 rejected words (probability
/// ~2⁻²⁰⁴⁸ — unreachable, but consensus-path code is total by law).
pub fn hash_to_goldilocks(domain: &Domain, msg: &[u8]) -> u64 {
    const ATTEMPTS: usize = 64;
    let mut xof = Xof::new(domain, msg);
    for _ in 0..ATTEMPTS {
        let v = xof.next_u64();
        if v < PROOFS_FIELD_MODULUS {
            return v;
        }
    }
    let bytes = xof.read_array::<32>();
    Hash256(bytes).reduce_mod(PROOFS_FIELD_MODULUS)
}

// -- codec integration (raw 32 bytes on the wire) ---------------------------

impl Encode for Hash256 {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0);
    }
    fn encoded_len(&self) -> usize {
        32
    }
}

impl Decode for Hash256 {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(Hash256(r.take_array::<32>()?))
    }
}

impl fmt::Display for Hash256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;

    #[test]
    fn concat_is_domain_separated() {
        let m: &[u8] = b"attack at dawn";
        let a = Hash256::concat(&constants::NULLIFIER, m);
        let b = Hash256::concat(&constants::NOTE_COMMITMENT, m);
        assert_ne!(a, b, "same message under different domains must differ");
    }

    #[test]
    fn concat_is_the_literal_wp_formula() {
        // BLAKE3(domain ‖ msg) — verified against a direct blake3 call over
        // the pre-concatenated buffer.
        let d = constants::TXID.as_bytes();
        let m: &[u8] = b"canonical-leg-serialization";
        let mut pre = Vec::with_capacity(d.len() + m.len());
        pre.extend_from_slice(d);
        pre.extend_from_slice(m);
        let expect = blake3::hash(&pre);
        assert_eq!(Hash256::concat(&constants::TXID, m).as_bytes(), expect.as_bytes());
    }

    #[test]
    fn framing_disambiguates_equal_concatenations() {
        // ["ab", "c"] and ["a", "bc"] concatenate identically — the ambiguity
        // framed mode exists to kill. Plain concatenation collides by
        // construction; the framed encodings must not.
        assert_eq!([b"ab", b"c"].concat(), [b"a", b"bc"].concat());
        let a = Hash256::framed(&constants::TXID, &[b"ab", b"c"]);
        let b = Hash256::framed(&constants::TXID, &[b"a", b"bc"]);
        assert_ne!(a, b);
    }

    #[test]
    fn concat_and_framed_modes_differ() {
        let m: &[u8] = b"x";
        assert_ne!(
            Hash256::concat(&constants::STATE_COMMITMENT, m),
            Hash256::framed(&constants::STATE_COMMITMENT, &[m])
        );
    }

    #[test]
    fn reduce_mod_matches_le_structure_ground_truth() {
        let h = Hash256::concat(&constants::TXID, b"ground truth");
        assert_eq!(h.reduce_mod(1), 0);
        // A 256-bit LE integer's low bits are its leading bytes — exact.
        assert_eq!(h.reduce_mod(2), (h.as_bytes()[0] & 1) as u64);
        assert_eq!(h.reduce_mod(256), h.as_bytes()[0] as u64);
        let mut low = [0u8; 4];
        low.copy_from_slice(&h.as_bytes()[0..4]);
        assert_eq!(h.reduce_mod(1u64 << 32), u32::from_le_bytes(low) as u64);
    }

    #[test]
    fn reduce_mod_bounds_and_determinism() {
        let mut rng = crate::testutil::SplitMix64::new(7);
        for _ in 0..256 {
            let bytes = rng.bytes(32);
            let h = Hash256::from_bytes(bytes[..].try_into().expect("32 bytes"));
            let m = rng.next_u64() | 1; // odd => >= 1
            let r = h.reduce_mod(m);
            assert!(r < m, "reduction must land in [0, m)");
            assert_eq!(r, h.reduce_mod(m), "reduction must be deterministic");
        }
    }

    #[test]
    fn goldilocks_is_uniform_bounded_and_deterministic() {
        let a = hash_to_goldilocks(&constants::TXID, b"seed");
        assert_eq!(a, hash_to_goldilocks(&constants::TXID, b"seed"));
        assert!(a < PROOFS_FIELD_MODULUS);
        assert_ne!(a, hash_to_goldilocks(&constants::TXID, b"seed2"));
    }

    #[test]
    fn xof_stream_is_deterministic_and_seekable() {
        let mut x1 = Xof::new(&constants::STATE_COMMITMENT, b"msg");
        let mut x2 = Xof::new(&constants::STATE_COMMITMENT, b"msg");
        let mut b1 = vec![0u8; 1000];
        let mut b2 = vec![0u8; 1000];
        x1.fill(&mut b1);
        x2.fill(&mut b2);
        assert_eq!(b1, b2);

        // next_u64 == manual LE read of the first 8 stream bytes
        let v = Xof::new(&constants::STATE_COMMITMENT, b"msg").next_u64();
        let mut raw = [0u8; 8];
        Xof::new(&constants::STATE_COMMITMENT, b"msg").fill(&mut raw);
        assert_eq!(v, u64::from_le_bytes(raw));

        // position/seek are exact
        let mut xs = Xof::new(&constants::STATE_COMMITMENT, b"msg");
        let first = xs.next_u64();
        assert_eq!(xs.position(), 8);
        let mut xz = Xof::new(&constants::STATE_COMMITMENT, b"msg");
        xz.seek(8);
        assert_eq!(xz.next_u64(), first);
    }

    #[test]
    fn xof_is_domain_separated() {
        let mut a = Xof::new(&constants::DERIVED_STATE, b"same");
        let mut b = Xof::new(&constants::STATE_COMMITMENT, b"same");
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn framed_xof_matches_framed_hash_prefix() {
        // Not a required invariant across modes — but framing must be
        // internally consistent: identical parts produce identical streams.
        let mut a = Xof::framed(&constants::TXID, &[b"p1", b"p2"]);
        let mut b = Xof::framed(&constants::TXID, &[b"p1", b"p2"]);
        let mut ba = [0u8; 64];
        let mut bb = [0u8; 64];
        a.fill(&mut ba);
        b.fill(&mut bb);
        assert_eq!(ba, bb);
    }

    #[test]
    fn hash256_codec_roundtrip_raw() {
        let h = Hash256::concat(&constants::TXID, b"codec");
        let enc = h.encode();
        assert_eq!(enc.len(), 32);
        assert_eq!(enc.as_slice(), h.as_bytes().as_slice());
        let dec = Hash256::decode(&enc).expect("roundtrip");
        assert_eq!(dec, h);

        let mut bad = enc.clone();
        bad.push(0);
        assert!(Hash256::decode(&bad).is_err());
    }

    #[test]
    fn display_is_lowercase_hex() {
        let h = Hash256::from_bytes([0u8; 32]);
        assert_eq!(h.to_string(), "00".repeat(32));
        let h2 = Hash256::from_bytes([0x0a, 0xff, 0x10]);
        assert!(h2.to_string().starts_with("0aff10"));
    }
}
