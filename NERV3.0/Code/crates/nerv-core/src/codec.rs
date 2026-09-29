//! The canonical codec: length-prefixed little-endian, total order, no serde.
//!
//! This is THE protocol wire format and THE canonical-serialization source
//! for hashing (`txid = BLAKE3(canonical serialization of all legs)` — WP
//! §3.6). It is schema-fixed and typed; it is deliberately NOT the
//! self-describing container format of the conformance registry
//! (`nerv-conformance`'s dynamic tree encoder freezes golden vectors —
//! different job, different format, both canonical).
//!
//! # Wire format (v1 — frozen; changes are governance-visible)
//!
//! * integers — fixed-width LE: `u8` 1B, `u16` 2B, `u32` 4B, `u64` 8B, `u128` 16B
//! * `bool`   — single byte `0x00` / `0x01`
//! * `[T; N]` — N items in order, no prefix (N is compile-time); `[u8; N]`
//!   is therefore N raw bytes
//! * `Vec<T>` — u32-LE count, then items
//! * `Option<T>` — tag byte `0x00` (None) / `0x01` (Some), then `T`'s encoding
//! * `String` — u32-LE byte length + UTF-8 bytes (capped at `MAX_STRING_LEN`)
//! * tuples   — fields in declaration order
//!
//! # Codec laws
//!
//! * **L1** — every `Decode` type consumes ≥ 1 byte. `Vec` decoding relies on
//!   this for DoS safety (count ≤ remaining bytes). No type in this crate
//!   violates it; downstream implementors must uphold it.
//! * **L2** — decoding is strict: trailing bytes, truncation, invalid tags,
//!   invalid UTF-8, and over-long counts are errors. A value has exactly one
//!   canonical encoding, and no canonical encoding is a strict prefix of
//!   another (self-delimiting format).
//! * **L3** — the canonical total order is byte-lexicographic over encodings.
//!   NOTE: for multi-byte LE integers this is NOT numeric order (1 > 256 in
//!   canonical order). Determinism is the requirement — this is the order WP
//!   §4.3's "canonically ordered by (txid, leg_index)" means.
//! * **L4** — no floats (P5), no serde, no self-describing tags: two
//!   different types may share wire bytes; the schema disambiguates.
//! * **L5** — strings in consensus types are ≤ `MAX_STRING_LEN`; sequences
//!   are ≤ `MAX_SEQ_LEN`; construction sites must enforce these.

use std::cmp::Ordering;

use crate::error::CodecError;

/// Maximum container nesting accepted by the decoder (stack-overflow defense).
pub const MAX_DEPTH: u32 = 128;
/// Maximum sequence length accepted by the decoder (DoS cap).
pub const MAX_SEQ_LEN: usize = 1 << 30;
/// Maximum string byte-length accepted by the decoder.
pub const MAX_STRING_LEN: usize = 1 << 16;

/// Canonical encoding into a byte sink.
pub trait Encode {
    /// Append the canonical encoding of `self` to `out`.
    fn encode_into(&self, out: &mut Vec<u8>);

    /// The exact encoded byte length of `self` (for allocation; must equal
    /// `self.encode().len()` — tested for every impl in this crate).
    fn encoded_len(&self) -> usize;

    /// Convenience: the canonical encoding as a fresh `Vec`.
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        self.encode_into(&mut out);
        out
    }
}

/// Canonical decoding from a reader.
pub trait Decode: Sized {
    /// Decode one value from `r` (streaming entry point: the reader is left
    /// positioned after the value).
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError>;

    /// Decode exactly one value from `buf`; trailing bytes are an error.
    fn decode(buf: &[u8]) -> Result<Self, CodecError> {
        let mut r = Reader::new(buf);
        let v = Self::decode_from(&mut r)?;
        r.finish_exhausted()?;
        Ok(v)
    }
}

/// Strict, bounds-checked, depth-limited canonical reader.
///
/// After any `Err`, a reader is poisoned: it must not be reused (container
/// depth may be inconsistent). Errors propagate outward by `?`.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    depth: u32,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, pos: 0, depth: 0 }
    }

    /// Undecoded bytes remaining.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// True iff the reader is positioned at the end.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }

    /// Fail unless the reader is exhausted (used by `Decode::decode`).
    pub fn finish_exhausted(&self) -> Result<(), CodecError> {
        match self.remaining() {
            0 => Ok(()),
            n => Err(CodecError::TrailingBytes { excess: n }),
        }
    }

    /// Enter one container nesting level (call before decoding children of a
    /// container; pair with [`Reader::leave`]).
    pub fn enter(&mut self) -> Result<(), CodecError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            Err(CodecError::DepthLimit { max: MAX_DEPTH })
        } else {
            Ok(())
        }
    }

    /// Leave one container nesting level.
    pub fn leave(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Borrow the next `n` bytes, advancing the cursor.
    pub fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        if n > self.remaining() {
            return Err(CodecError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Borrow the next `N` bytes as a fixed array, advancing the cursor.
    pub fn take_array<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        let s = self.take(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(s);
        Ok(out)
    }

    pub fn read_u8(&mut self) -> Result<u8, CodecError> {
        let s = self.take(1)?;
        Ok(s[0])
    }

    pub fn read_u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(self.take_array::<2>()?))
    }

    pub fn read_u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.take_array::<4>()?))
    }

    pub fn read_u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.take_array::<8>()?))
    }

    pub fn read_u128(&mut self) -> Result<u128, CodecError> {
        Ok(u128::from_le_bytes(self.take_array::<16>()?))
    }

    pub fn read_bool(&mut self) -> Result<bool, CodecError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            byte => Err(CodecError::InvalidBool { byte }),
        }
    }

    /// Read and validate a sequence count (law L1 ⇒ count ≤ remaining).
    pub fn read_seq_len(&mut self) -> Result<usize, CodecError> {
        let n = self.read_u32()? as usize;
        if n > MAX_SEQ_LEN {
            return Err(CodecError::SeqTooLarge { count: n, max: MAX_SEQ_LEN });
        }
        if n > self.remaining() {
            return Err(CodecError::SeqLenOverrun { count: n, remaining: self.remaining() });
        }
        Ok(n)
    }

    /// Read and validate a string byte-length.
    pub fn read_string_len(&mut self) -> Result<usize, CodecError> {
        let n = self.read_u32()? as usize;
        if n > MAX_STRING_LEN {
            return Err(CodecError::StringTooLong { len: n, max: MAX_STRING_LEN });
        }
        if n > self.remaining() {
            return Err(CodecError::Truncated);
        }
        Ok(n)
    }
}

// -- primitives --------------------------------------------------------------

macro_rules! impl_int {
    ($t:ty, $w:expr, $read:ident) => {
        impl Encode for $t {
            fn encode_into(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn encoded_len(&self) -> usize {
                $w
            }
        }
        impl Decode for $t {
            fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
                r.$read()
            }
        }
    };
}

impl_int!(u8, 1, read_u8);
impl_int!(u16, 2, read_u16);
impl_int!(u32, 4, read_u32);
impl_int!(u64, 8, read_u64);
impl_int!(u128, 16, read_u128);

impl Encode for bool {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
    fn encoded_len(&self) -> usize {
        1
    }
}

impl Decode for bool {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        r.read_bool()
    }
}

impl Encode for str {
    fn encode_into(&self, out: &mut Vec<u8>) {
        debug_assert!(
            self.len() <= MAX_STRING_LEN,
            "codec law L5: strings are bounded by MAX_STRING_LEN"
        );
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        out.extend_from_slice(self.as_bytes());
    }
    fn encoded_len(&self) -> usize {
        4 + self.len()
    }
}

impl Encode for String {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.as_str().encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.as_str().encoded_len()
    }
}

impl Decode for String {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_string_len()?;
        let b = r.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|_| CodecError::InvalidUtf8)
    }
}

// -- containers ---------------------------------------------------------------

impl<T: Encode> Encode for Option<T> {
    fn encode_into(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(v) => {
                out.push(1);
                v.encode_into(out);
            }
        }
    }
    fn encoded_len(&self) -> usize {
        1 + self.as_ref().map_or(0, |v| v.encoded_len())
    }
}

impl<T: Decode> Decode for Option<T> {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        match r.read_u8()? {
            0 => Ok(None),
            1 => Ok(Some(T::decode_from(r)?)),
            tag => Err(CodecError::InvalidOptionTag { tag }),
        }
    }
}

impl<T: Encode> Encode for Vec<T> {
    fn encode_into(&self, out: &mut Vec<u8>) {
        debug_assert!(
            self.len() <= MAX_SEQ_LEN,
            "codec law L5: sequences are bounded by MAX_SEQ_LEN"
        );
        out.extend_from_slice(&(self.len() as u32).to_le_bytes());
        for item in self {
            item.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        4 + self.iter().map(|x| x.encoded_len()).sum::<usize>()
    }
}

impl<T: Decode> Decode for Vec<T> {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let n = r.read_seq_len()?;
        r.enter()?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(T::decode_from(r)?);
        }
        r.leave();
        Ok(v)
    }
}

impl<T: Encode, const N: usize> Encode for [T; N] {
    fn encode_into(&self, out: &mut Vec<u8>) {
        for x in self {
            x.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        self.iter().map(|x| x.encoded_len()).sum()
    }
}

impl<T: Decode, const N: usize> Decode for [T; N] {
    fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
        let mut v = Vec::with_capacity(N);
        for _ in 0..N {
            v.push(T::decode_from(r)?);
        }
        // Unreachable by construction: exactly N items were pushed.
        v.try_into()
            .map_err(|_| CodecError::InvariantViolated("array decode: exactly N items pushed"))
    }
}

// -- tuples --------------------------------------------------------------------

macro_rules! impl_tuple {
    ($(($n:tt : $t:ident)),+) => {
        impl<$($t: Encode),+> Encode for ($($t,)+) {
            fn encode_into(&self, out: &mut Vec<u8>) {
                $(self.$n.encode_into(out);)+
            }
            fn encoded_len(&self) -> usize {
                0 $(+ self.$n.encoded_len())+
            }
        }
        impl<$($t: Decode),+> Decode for ($($t,)+) {
            fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
                Ok(($(<$t as Decode>::decode_from(r)?,)+))
            }
        }
    };
}

impl_tuple!((0: A), (1: B));
impl_tuple!((0: A), (1: B), (2: C));
impl_tuple!((0: A), (1: B), (2: C), (3: D));

// -- canonical ordering ---------------------------------------------------------

/// An owned canonical encoding, for keys and sorting (the (txid, leg_index)
/// ordering of WP §4.3 is an `Ord` over these).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Encoded(pub Vec<u8>);

impl Encoded {
    /// The canonical encoding of `value`.
    pub fn new<T: Encode + ?Sized>(value: &T) -> Encoded {
        Encoded(value.encode())
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Decode `T` from this encoding (strict).
    pub fn decode<T: Decode>(&self) -> Result<T, CodecError> {
        T::decode(&self.0)
    }
}

/// The canonical total order (law L3): byte-lexicographic over encodings.
pub fn canonical_cmp<T: Encode + ?Sized>(a: &T, b: &T) -> Ordering {
    a.encode().cmp(&b.encode())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    /// Roundtrip + strictness: exact length, decode-equality, every strict
    /// prefix fails, any trailing byte fails (law L2: self-delimiting).
    fn roundtrip<T>(v: T)
    where
        T: Encode + Decode + PartialEq + std::fmt::Debug,
    {
        let enc = v.encode();
        assert_eq!(enc.len(), v.encoded_len(), "encoded_len must be exact");
        let dec = T::decode(&enc).expect("roundtrip decode");
        assert_eq!(&dec, &v);
        for cut in 0..enc.len() {
            assert!(
                T::decode(&enc[..cut]).is_err(),
                "truncation at byte {cut} must be rejected (self-delimiting law)"
            );
        }
        let mut ext = enc.clone();
        ext.push(0);
        assert!(T::decode(&ext).is_err(), "trailing bytes must be rejected");
    }

    #[test]
    fn primitives_roundtrip() {
        for v in [0u8, 1, 0x7f, 0xff] {
            roundtrip(v);
        }
        for v in [0u16, 1, u16::MAX] {
            roundtrip(v);
        }
        for v in [0u32, 1, u32::MAX] {
            roundtrip(v);
        }
        for v in [0u64, 1, u64::MAX] {
            roundtrip(v);
        }
        for v in [0u128, 1, u128::MAX] {
            roundtrip(v);
        }
        roundtrip(true);
        roundtrip(false);
    }

    #[test]
    fn bool_is_strict() {
        assert!(matches!(bool::decode(&[0x02]), Err(CodecError::InvalidBool { byte: 2 })));
        assert_eq!(bool::decode(&[0x00]), Ok(false));
        assert_eq!(bool::decode(&[0x01]), Ok(true));
    }

    #[test]
    fn option_roundtrip_and_strict_tag() {
        roundtrip(Option::<u64>::None);
        roundtrip(Some(0xdeadbeefu64));
        assert!(matches!(
            Option::<u8>::decode(&[0x02]),
            Err(CodecError::InvalidOptionTag { tag: 2 })
        ));
    }

    #[test]
    fn vec_roundtrip_overrun_and_cap() {
        let v: Vec<u64> = (0..17).collect();
        roundtrip(v);
        roundtrip(Vec::<u8>::new());

        // count 5, only 3 item bytes present
        let mut bad = Vec::new();
        bad.extend_from_slice(&5u32.to_le_bytes());
        bad.extend_from_slice(&[1u8, 2, 3]);
        assert!(matches!(
            Vec::<u8>::decode(&bad),
            Err(CodecError::SeqLenOverrun { count: 5, remaining: 3 })
        ));

        // count above the parser cap
        let mut huge = Vec::new();
        huge.extend_from_slice(&((MAX_SEQ_LEN as u32) + 1).to_le_bytes());
        assert!(matches!(
            Vec::<u8>::decode(&huge),
            Err(CodecError::SeqTooLarge { .. })
        ));
    }

    #[test]
    fn string_roundtrip_utf8_and_cap() {
        roundtrip(String::from("nerv.cm"));
        roundtrip(String::new());
        roundtrip(String::from("τ-extreme-Ω"));

        // invalid UTF-8 (0xC3 expects a continuation; 0x28 is not one)
        let mut bad = Vec::new();
        bad.extend_from_slice(&2u32.to_le_bytes());
        bad.extend_from_slice(&[0xC3, 0x28]);
        assert!(matches!(String::decode(&bad), Err(CodecError::InvalidUtf8)));

        // over-cap length fires before any buffer is needed
        let mut long = Vec::new();
        long.extend_from_slice(&((MAX_STRING_LEN as u32) + 1).to_le_bytes());
        assert!(matches!(String::decode(&long), Err(CodecError::StringTooLong { .. })));
    }

    #[test]
    fn fixed_arrays_are_raw_bytes() {
        let a: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
        assert_eq!(a.encode(), vec![0xAA, 0xBB, 0xCC, 0xDD]);
        roundtrip(a);
        roundtrip([7u64, 8, 9]);
        roundtrip([[1u8, 2], [3, 4], [5, 6]]); // nested fixed arrays
    }

    #[test]
    fn tuples_roundtrip() {
        roundtrip((0x1234u32, 7u8, true));
        roundtrip((1u8, 2u16, 3u32, 4u64));
        roundtrip((Vec::<u8>::new(), Some(3u32)));
    }

    #[test]
    fn depth_limit_enforced() {
        let mut r = Reader::new(&[]);
        let mut entered = 0u32;
        while r.enter().is_ok() {
            entered += 1;
        }
        assert_eq!(entered, MAX_DEPTH);
        assert!(matches!(r.enter(), Err(CodecError::DepthLimit { .. })));
    }

    #[test]
    fn canonical_order_is_byte_lexicographic_and_total() {
        let mut rng = SplitMix64::new(99);
        let vals: Vec<u64> = (0..100).map(|_| rng.next_u64()).collect();

        let mut by_encoded: Vec<u64> = vals.clone();
        by_encoded.sort_by(|a, b| Encoded::new(a).cmp(&Encoded::new(b)));
        let mut by_fn: Vec<u64> = vals.clone();
        by_fn.sort_by(|a, b| canonical_cmp(a, b));
        assert_eq!(by_encoded, by_fn, "Encoded ordering == canonical_cmp ordering");

        // Document the law with a ground-truth pair: LE byte-lex order is NOT
        // numeric order (determinism is the requirement — law L3).
        //   1u64    -> [01 00 00 00 00 00 00 00]
        //   256u64  -> [00 01 00 00 00 00 00 00]   => 256 < 1 canonically.
        assert_eq!(canonical_cmp(&1u64, &256u64), Ordering::Greater);
        assert_eq!(canonical_cmp(&1u64, &1u64), Ordering::Equal);
    }

    #[test]
    fn encoded_wrapper_decodes() {
        let e = Encoded::new(&(7u32, false));
        assert_eq!(e.decode::<(u32, bool)>(), Ok((7, false)));
        let mut bad = e.0.clone();
        bad.push(0);
        assert!((Encoded(bad)).decode::<(u32, bool)>().is_err());
    }

    #[test]
    fn streaming_decode_consumes_in_order() {
        let buf = (0x0102u16, 5u8, true).encode();
        let mut r = Reader::new(&buf);
        assert_eq!(u16::decode_from(&mut r), Ok(0x0102));
        assert_eq!(u8::decode_from(&mut r), Ok(5));
        assert_eq!(bool::decode_from(&mut r), Ok(true));
        assert!(r.is_exhausted());
    }

    #[test]
    fn randomized_roundtrips() {
        let mut rng = SplitMix64::new(0x5EED_0001);
        for _ in 0..200 {
            roundtrip(rng.next_u64());
            roundtrip(rng.next_bool());
            roundtrip(rng.next_u32());

            let len = (rng.next_u64() % 33) as usize;
            let s: String = (0..len)
                .map(|_| {
                    let idx = (rng.next_u64() % 26) as u8;
                    (b'a' + idx) as char
                })
                .collect();
            roundtrip(s);

            let n = (rng.next_u64() % 32) as usize;
            let v: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();
            roundtrip(v);

            roundtrip(if rng.next_bool() { Some(rng.next_u64()) } else { None });

            let nested: Vec<Vec<u8>> = (0..(rng.next_u64() % 8))
                .map(|_| rng.bytes((rng.next_u64() % 17) as usize))
                .collect();
            roundtrip(nested);

            let arr: [u64; 4] = [rng.next_u64(), rng.next_u64(), rng.next_u64(), rng.next_u64()];
            roundtrip(arr);

            roundtrip((rng.next_u64(), rng.next_bool(), rng.bytes(5)));
        }
    }

    #[test]
    fn encoded_len_law_every_type_at_least_one_byte() {
        // Law L1 spot checks across representative types.
        assert!(1u8.encoded_len() >= 1);
        assert!(false.encoded_len() >= 1);
        assert!(String::new().encoded_len() >= 1);
        assert!(Vec::<u64>::new().encoded_len() >= 1);
        assert!(None::<u8>.encoded_len() >= 1);
        assert!((0u8,).encoded_len() >= 1);
    }
}
