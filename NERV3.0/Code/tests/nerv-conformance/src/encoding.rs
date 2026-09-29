//! Canonical container encoding for conformance data (the registry's frozen
//! wire format — deliberately independent of the protocol codec, which lands
//! in nerv-core at chunk 2 with its own versioning).
//!
//! Discipline: length-prefixed little-endian; tags are single bytes; maps
//! carry keys in sorted (BTreeMap) order; decoding is total (bounds-checked,
//! depth-limited, exact-consume) and strict (unknown tags, duplicate keys,
//! trailing bytes are errors). The byte encoding defines the total order.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::error::EncError;

const TAG_INT: u8 = b'I';
const TAG_BOOL: u8 = b'b';
const TAG_STR: u8 = b'S';
const TAG_BYTES: u8 = b'X';
const TAG_SEQ: u8 = b'[';
const TAG_MAP: u8 = b'M';

/// Maximum nesting depth accepted by the decoder (stack-overflow defense).
pub const MAX_DEPTH: u32 = 128;

/// A canonical, deterministically encodable value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Canonical {
    Int(u64),
    Bool(bool),
    Str(String),
    Bytes(Vec<u8>),
    Seq(Vec<Canonical>),
    Map(BTreeMap<String, Canonical>),
}

impl Canonical {
    // -- constructors -------------------------------------------------------
    pub fn int(n: u64) -> Canonical {
        Canonical::Int(n)
    }
    pub fn str(s: &str) -> Canonical {
        Canonical::Str(s.to_string())
    }
    pub fn bytes(b: &[u8]) -> Canonical {
        Canonical::Bytes(b.to_vec())
    }
    pub fn seq(items: Vec<Canonical>) -> Canonical {
        Canonical::Seq(items)
    }
    pub fn map() -> Canonical {
        Canonical::Map(BTreeMap::new())
    }
    /// Builder: insert into a map (no-op with a debug assertion otherwise).
    pub fn with(mut self, key: &str, value: Canonical) -> Canonical {
        match &mut self {
            Canonical::Map(m) => {
                m.insert(key.to_string(), value);
            }
            _ => debug_assert!(false, "with() on a non-map canonical"),
        }
        self
    }

    // -- encoding -----------------------------------------------------------
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Canonical::Int(n) => {
                out.push(TAG_INT);
                out.extend_from_slice(&n.to_le_bytes());
            }
            Canonical::Bool(b) => {
                out.push(TAG_BOOL);
                out.push(u8::from(*b));
            }
            Canonical::Str(s) => {
                out.push(TAG_STR);
                write_len(out, s.len());
                out.extend_from_slice(s.as_bytes());
            }
            Canonical::Bytes(b) => {
                out.push(TAG_BYTES);
                write_len(out, b.len());
                out.extend_from_slice(b);
            }
            Canonical::Seq(items) => {
                out.push(TAG_SEQ);
                write_len(out, items.len());
                for item in items {
                    item.write(out);
                }
            }
            Canonical::Map(m) => {
                out.push(TAG_MAP);
                write_len(out, m.len());
                for (k, v) in m {
                    write_len(out, k.len());
                    out.extend_from_slice(k.as_bytes());
                    v.write(out);
                }
            }
        }
    }

    // -- strict typed accessors (registry decoding) -------------------------
    pub fn as_u64(&self) -> Result<u64, EncError> {
        match self {
            Canonical::Int(n) => Ok(*n),
            _ => Err(EncError::TypeMismatch { wanted: "int", got: self.type_name() }),
        }
    }
    pub fn as_bool(&self) -> Result<bool, EncError> {
        match self {
            Canonical::Bool(b) => Ok(*b),
            _ => Err(EncError::TypeMismatch { wanted: "bool", got: self.type_name() }),
        }
    }
    pub fn as_str(&self) -> Result<&str, EncError> {
        match self {
            Canonical::Str(s) => Ok(s),
            _ => Err(EncError::TypeMismatch { wanted: "str", got: self.type_name() }),
        }
    }
    pub fn as_bytes(&self) -> Result<&[u8], EncError> {
        match self {
            Canonical::Bytes(b) => Ok(b),
            _ => Err(EncError::TypeMismatch { wanted: "bytes", got: self.type_name() }),
        }
    }
    pub fn as_seq(&self) -> Result<&[Canonical], EncError> {
        match self {
            Canonical::Seq(s) => Ok(s),
            _ => Err(EncError::TypeMismatch { wanted: "seq", got: self.type_name() }),
        }
    }
    pub fn as_map(&self) -> Result<&BTreeMap<String, Canonical>, EncError> {
        match self {
            Canonical::Map(m) => Ok(m),
            _ => Err(EncError::TypeMismatch { wanted: "map", got: self.type_name() }),
        }
    }
    pub fn get(&self, key: &str) -> Result<&Canonical, EncError> {
        self.as_map()?
            .get(key)
            .ok_or_else(|| EncError::MissingKey { key: key.to_string() })
    }

    fn type_name(&self) -> &'static str {
        match self {
            Canonical::Int(_) => "int",
            Canonical::Bool(_) => "bool",
            Canonical::Str(_) => "str",
            Canonical::Bytes(_) => "bytes",
            Canonical::Seq(_) => "seq",
            Canonical::Map(_) => "map",
        }
    }

    // -- conversions --------------------------------------------------------
    /// Convert a parsed TOML tree to canonical form. Floats and datetimes are
    /// P5 violations and rejected here (defense in depth: the loader rejects
    /// them before hashing; this makes the conversion itself total-safe).
    pub fn from_toml(v: &toml::Value) -> Result<Canonical, EncError> {
        match v {
            toml::Value::String(s) => Ok(Canonical::Str(s.clone())),
            toml::Value::Integer(i) => {
                if *i < 0 {
                    Err(EncError::NegativeInt { value: *i })
                } else {
                    Ok(Canonical::Int(*i as u64))
                }
            }
            toml::Value::Boolean(b) => Ok(Canonical::Bool(*b)),
            toml::Value::Array(a) => {
                let mut items = Vec::with_capacity(a.len());
                for x in a {
                    items.push(Canonical::from_toml(x)?);
                }
                Ok(Canonical::Seq(items))
            }
            toml::Value::Table(t) => {
                let mut m = BTreeMap::new();
                for (k, x) in t.iter() {
                    m.insert(k.clone(), Canonical::from_toml(x)?);
                }
                Ok(Canonical::Map(m))
            }
            toml::Value::Float(_) | toml::Value::Datetime(_) => Err(EncError::ForbiddenTomlType),
        }
    }
}

impl Ord for Canonical {
    fn cmp(&self, other: &Self) -> Ordering {
        self.encode().cmp(&other.encode())
    }
}
impl PartialOrd for Canonical {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn write_len(out: &mut Vec<u8>, n: usize) {
    let n = u32::try_from(n).unwrap_or(u32::MAX); // registry payloads are far below 2^32
    out.extend_from_slice(&n.to_le_bytes());
}

// ---------------------------------------------------------------------------

pub fn decode(bytes: &[u8]) -> Result<Canonical, EncError> {
    let mut r = Reader::new(bytes);
    let v = r.read_value(0)?;
    r.finish()?;
    Ok(v)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], EncError> {
        if n > self.remaining() {
            return Err(EncError::Truncated);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_len(&mut self) -> Result<usize, EncError> {
        let b = self.take(4)?;
        let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
        // Guards absurd allocations before Vec::with_capacity.
        if n > self.remaining() {
            return Err(EncError::Truncated);
        }
        Ok(n)
    }
    fn read_u64(&mut self) -> Result<u64, EncError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }
    fn read_value(&mut self, depth: u32) -> Result<Canonical, EncError> {
        if depth > MAX_DEPTH {
            return Err(EncError::DepthLimit);
        }
        let tag = self.take(1)?[0];
        match tag {
            TAG_INT => Ok(Canonical::Int(self.read_u64()?)),
            TAG_BOOL => {
                let b = self.take(1)?[0];
                match b {
                    0 => Ok(Canonical::Bool(false)),
                    1 => Ok(Canonical::Bool(true)),
                    _ => Err(EncError::InvalidBool { byte: b }),
                }
            }
            TAG_STR => {
                let n = self.read_len()?;
                let b = self.take(n)?;
                let s = String::from_utf8(b.to_vec()).map_err(|_| EncError::InvalidUtf8)?;
                Ok(Canonical::Str(s))
            }
            TAG_BYTES => {
                let n = self.read_len()?;
                Ok(Canonical::Bytes(self.take(n)?.to_vec()))
            }
            TAG_SEQ => {
                let n = self.read_len()?;
                let mut v = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    v.push(self.read_value(depth + 1)?);
                }
                Ok(Canonical::Seq(v))
            }
            TAG_MAP => {
                let n = self.read_len()?;
                let mut m = BTreeMap::new();
                for _ in 0..n {
                    let klen = self.read_len()?;
                    let kb = self.take(klen)?;
                    let k = String::from_utf8(kb.to_vec()).map_err(|_| EncError::InvalidUtf8)?;
                    let v = self.read_value(depth + 1)?;
                    if m.insert(k, v).is_some() {
                        return Err(EncError::DuplicateKey);
                    }
                }
                Ok(Canonical::Map(m))
            }
            t => Err(EncError::UnknownTag { tag: t }),
        }
    }
    fn finish(&self) -> Result<(), EncError> {
        if self.pos == self.buf.len() {
            Ok(())
        } else {
            Err(EncError::Trailing { excess: self.remaining() })
        }
    }
}
