// src/utils/serialization.rs
// ============================================================================
// PROTOBUF SERIALIZATION FOR DETERMINISTIC ENCODING
// ============================================================================
// This module provides Protobuf serialization/deserialization for all NERV
// data structures. Protobuf is chosen for:
// 1. Deterministic encoding (critical for cryptographic signatures)
// 2. Compact binary format (3-10x smaller than JSON)
// 3. Backward/forward compatibility
// 4. Fast parsing and generation
// ============================================================================

use prost::Message;
use serde::{Serialize, Deserialize};
use std::io::{Read, Write};
use thiserror::Error;

// ============================================================================
// ERROR DEFINITIONS
// ============================================================================

/// Serialization errors
#[derive(Error, Debug)]
pub enum SerializeError {
    #[error("Protobuf encoding error: {0}")]
    ProtobufEncode(#[from] prost::EncodeError),
    
    #[error("Protobuf decoding error: {0}")]
    ProtobufDecode(#[from] prost::DecodeError),
    
    #[error("I/O error during serialization: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    
    #[error("Size limit exceeded: {0}")]
    SizeLimit(usize),
}

// ============================================================================
// PROTOBUF ENCODER/DECODER TRAITS
// ============================================================================

/// Trait for types that can be serialized to Protobuf
pub trait ProtobufEncoder: Message + Default {
    /// Serialize to bytes
    fn to_protobuf(&self) -> Result<Vec<u8>, SerializeError> {
        let mut buf = Vec::with_capacity(self.encoded_len());
        self.encode(&mut buf)?;
        Ok(buf)
    }
    
    /// Serialize to hex string (for debugging)
    fn to_protobuf_hex(&self) -> Result<String, SerializeError> {
        let bytes = self.to_protobuf()?;
        Ok(hex::encode(bytes))
    }
    
    /// Serialize to base64 (for web APIs)
    fn to_protobuf_base64(&self) -> Result<String, SerializeError> {
        let bytes = self.to_protobuf()?;
        Ok(base64::encode(bytes))
    }
    /// Encode with canonical (fully deterministic) options if needed
pub fn encode_canonical<M: Message>(&self, msg: &M) -> Result<Vec<u8>, SerializeError> {
    let mut buf = Vec::with_capacity(msg.encoded_len());
    // prost does not have built-in canonical mode, but we can enforce map key sorting
    // via custom encoding if maps are used in critical messages.
    // For now, standard encode is sufficient as NERV messages avoid unordered maps.
    msg.encode(&mut buf)
        .map_err(|e| SerializeError::Encoding(e.to_string()))?;
    Ok(buf)
}


}

/// Trait for types that can be deserialized from Protobuf
pub trait ProtobufDecoder: Message + Default {
    /// Deserialize from bytes
    fn from_protobuf(bytes: &[u8]) -> Result<Self, SerializeError> {
        Ok(Self::decode(bytes)?)
    }
    
    /// Deserialize from hex string
    fn from_protobuf_hex(hex_str: &str) -> Result<Self, SerializeError> {
        let bytes = hex::decode(hex_str).map_err(|e| SerializeError::InvalidFormat(e.to_string()))?;
        Self::from_protobuf(&bytes)
    }
    
    /// Deserialize from base64 string
    fn from_protobuf_base64(base64_str: &str) -> Result<Self, SerializeError> {
        let bytes = base64::decode(base64_str).map_err(|e| SerializeError::InvalidFormat(e.to_string()))?;
        Self::from_protobuf(&bytes)
    }
}

// Auto-implement for all Prost Message types
impl<T: Message + Default> ProtobufEncoder for T {}
impl<T: Message + Default> ProtobufDecoder for T {}

// ============================================================================
// NERV-SPECIFIC SERIALIZATION TYPES
// ============================================================================

/// Fixed-size byte array wrapper for Protobuf serialization
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixedBytes<const N: usize>(pub [u8; N]);

impl<const N: usize> FixedBytes<N> {
    pub fn new(bytes: [u8; N]) -> Self {
        Self(bytes)
    }
    
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
    
    pub fn to_vec(&self) -> Vec<u8> {
        self.0.to_vec()
    }
}

// Implement Protobuf serialization for FixedBytes
impl<const N: usize> From<FixedBytes<N>> for Vec<u8> {
    fn from(value: FixedBytes<N>) -> Self {
        value.0.to_vec()
    }
}

impl<const N: usize> TryFrom<Vec<u8>> for FixedBytes<N> {
    type Error = SerializeError;
    
    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(SerializeError::InvalidFormat(
                format!("Expected {} bytes, got {}", N, value.len())
            ));
        }
        
        let mut bytes = [0u8; N];
        bytes.copy_from_slice(&value);
        Ok(FixedBytes(bytes))
    }
}

/// Embedding vector (512 f32 values)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingVec(pub [f32; 512]);

impl EmbeddingVec {
    pub fn new(values: [f32; 512]) -> Self {
        Self(values)
    }
    
    pub fn zeros() -> Self {
        Self([0.0; 512])
    }
    
    pub fn from_slice(slice: &[f32]) -> Result<Self, SerializeError> {
        if slice.len() != 512 {
            return Err(SerializeError::InvalidFormat(
                format!("Expected 512 floats, got {}", slice.len())
            ));
        }
        
        let mut values = [0.0; 512];
        values.copy_from_slice(slice);
        Ok(Self(values))
    }
    
    /// Element-wise addition (for homomorphic updates)
    pub fn add(&self, other: &EmbeddingVec) -> EmbeddingVec {
        let mut result = [0.0; 512];
        for i in 0..512 {
            result[i] = self.0[i] + other.0[i];
        }
        EmbeddingVec(result)
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &EmbeddingVec) -> EmbeddingVec {
        let mut result = [0.0; 512];
        for i in 0..512 {
            result[i] = self.0[i] - other.0[i];
        }
        EmbeddingVec(result)
    }
    
    /// Check if two embeddings are equal within error bound
    pub fn approx_eq(&self, other: &EmbeddingVec, epsilon: f32) -> bool {
        for i in 0..512 {
            if (self.0[i] - other.0[i]).abs() > epsilon {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// VARINT ENCODING/DECODING (for compact size representation)
// ============================================================================

/// Encode a u64 as varint bytes
pub fn encode_varint(value: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(10);
    let mut v = value;
    
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
    
    buf
}

/// Decode varint bytes to u64
pub fn decode_varint<R: Read>(reader: &mut R) -> Result<u64, SerializeError> {
    let mut result = 0u64;
    let mut shift = 0;
    
    loop {
        let mut byte = [0u8; 1];
        reader.read_exact(&mut byte)?;
        
        let b = byte[0];
        result |= ((b & 0x7F) as u64) << shift;
        
        if b & 0x80 == 0 {
            break;
        }
        
        shift += 7;
        if shift >= 64 {
            return Err(SerializeError::InvalidFormat("Varint too long".to_string()));
        }
    }
    
    Ok(result)
}

// ============================================================================
// SIZE-LIMITED READER/WRITER (for security)
// ============================================================================

/// Reader that limits the maximum number of bytes that can be read
pub struct LimitedReader<R> {
    inner: R,
    limit: usize,
    read: usize,
}

impl<R: Read> LimitedReader<R> {
    pub fn new(inner: R, limit: usize) -> Self {
        Self {
            inner,
            limit,
            read: 0,
        }
    }
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Calculate maximum we can read without exceeding limit
        let max_read = self.limit - self.read;
        if max_read == 0 {
            return Ok(0);
        }
        
        let to_read = buf.len().min(max_read);
        let n = self.inner.read(&mut buf[..to_read])?;
        self.read += n;
        
        if self.read > self.limit {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Read limit exceeded",
            ));
        }
        
        Ok(n)
    }
}

/// Writer that limits the maximum number of bytes that can be written
pub struct LimitedWriter<W> {
    inner: W,
    limit: usize,
    written: usize,
}

impl<W: Write> LimitedWriter<W> {
    pub fn new(inner: W, limit: usize) -> Self {
        Self {
            inner,
            limit,
            written: 0,
        }
    }
    
    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for LimitedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let max_write = self.limit - self.written;
        if max_write == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Write limit exceeded",
            ));
        }
        
        let to_write = buf.len().min(max_write);
        let n = self.inner.write(&buf[..to_write])?;
        self.written += n;
        
        Ok(n)
    }
    
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

// ============================================================================
// SERIALIZATION UTILITIES
// ============================================================================

/// Serialize a value to bytes with size limit
pub fn serialize_with_limit<T: Serialize>(
    value: &T,
    limit: usize,
) -> Result<Vec<u8>, SerializeError> {
    let mut writer = LimitedWriter::new(Vec::new(), limit);
    bincode::serialize_into(&mut writer, value)
        .map_err(|e| SerializeError::InvalidFormat(e.to_string()))?;
    
    Ok(writer.into_inner())
}

/// Deserialize bytes to a value with size limit
pub fn deserialize_with_limit<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    limit: usize,
) -> Result<T, SerializeError> {
    let mut reader = LimitedReader::new(bytes, limit);
    bincode::deserialize_from(&mut reader)
        .map_err(|e| SerializeError::InvalidFormat(e.to_string()))
}

/// Calculate serialized size without actually serializing
pub fn serialized_size<T: Serialize>(value: &T) -> Result<usize, SerializeError> {
    bincode::serialized_size(value)
        .map(|size| size as usize)
        .map_err(|e| SerializeError::InvalidFormat(e.to_string()))
}

// ============================================================================
// PROTOBUF DEFINITIONS (Generated from .proto files)
// ============================================================================

// Note: In practice, these would be generated from .proto files at build time.
// For now, we define them manually for clarity.

/// Protobuf message for Verifiable Delay Witness (VDW)
#[derive(Clone, PartialEq, Message)]
pub struct VdwProto {
    #[prost(bytes, tag = "1")]
    pub tx_hash: Vec<u8>,  // 32 bytes
    
    #[prost(uint64, tag = "2")]
    pub shard_id: u64,
    
    #[prost(uint64, tag = "3")]
    pub lattice_height: u64,
    
    #[prost(bytes, tag = "4")]
    pub delta_proof: Vec<u8>,  // ≤750 bytes
    
    #[prost(bytes, tag = "5")]
    pub final_root: Vec<u8>,  // 32 bytes
    
    #[prost(bytes, tag = "6")]
    pub attestation: Vec<u8>,  // TEE attestation
    
    #[prost(bytes, tag = "7")]
    pub signature: Vec<u8>,  // Dilithium signature
    
    #[prost(uint64, tag = "8")]
    pub timestamp: u64,
    
    #[prost(uint64, tag = "9")]
    pub monotonic_counter: u64,
}

impl VdwProto {
    /// Create a new VDW protobuf message
    pub fn new(
        tx_hash: [u8; 32],
        shard_id: u64,
        lattice_height: u64,
        delta_proof: Vec<u8>,
        final_root: [u8; 32],
        attestation: Vec<u8>,
        signature: Vec<u8>,
        timestamp: u64,
        monotonic_counter: u64,
    ) -> Self {
        Self {
            tx_hash: tx_hash.to_vec(),
            shard_id,
            lattice_height,
            delta_proof,
            final_root: final_root.to_vec(),
            attestation,
            signature,
            timestamp,
            monotonic_counter,
        }
    }
    
    /// Validate VDW size constraints (≤1.8KB)
    pub fn validate_size(&self) -> Result<(), SerializeError> {
        let size = self.encoded_len();
        if size > super::super::params::VDW_MAX_SIZE {
            return Err(SerializeError::SizeLimit(size));
        }
        Ok(())
    }
}

/// Protobuf message for neural vote in consensus
#[derive(Clone, PartialEq, Message)]
pub struct NeuralVoteProto {
    #[prost(bytes, tag = "1")]
    pub predicted_hash: Vec<u8>,  // 32 bytes
    
    #[prost(bytes, tag = "2")]
    pub partial_signature: Vec<u8>,  // BLS partial signature
    
    #[prost(double, tag = "3")]
    pub reputation_score: f64,
    
    #[prost(bytes, tag = "4")]
    pub tee_attestation: Vec<u8>,
    
    #[prost(uint64, tag = "5")]
    pub timestamp: u64,
}

impl NeuralVoteProto {
    /// Calculate the size of the vote message (target: 128 bytes)
    pub fn size(&self) -> usize {
        self.encoded_len()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fixed_bytes_serialization() {
        let original = FixedBytes::new([1, 2, 3, 4, 5]);
        let bytes: Vec<u8> = original.clone().into();
        let restored = FixedBytes::<5>::try_from(bytes).unwrap();
        
        assert_eq!(original, restored);
    }
    
    #[test]
    fn test_embedding_vec_operations() {
        let e1 = EmbeddingVec::zeros();
        let mut values = [0.0; 512];
        values[0] = 1.0;
        values[1] = 2.0;
        let e2 = EmbeddingVec::new(values);
        
        let sum = e1.add(&e2);
        assert!(sum.approx_eq(&e2, 0.0001));
        
        let diff = sum.sub(&e2);
        assert!(diff.approx_eq(&e1, 0.0001));
    }
    
    #[test]
    fn test_varint_encoding() {
        let test_cases = vec![
            (0u64, vec![0x00]),
            (1, vec![0x01]),
            (127, vec![0x7F]),
            (128, vec![0x80, 0x01]),
            (300, vec![0xAC, 0x02]),
            (u64::MAX, vec![0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01]),
        ];
        
        for (value, expected) in test_cases {
            let encoded = encode_varint(value);
            assert_eq!(encoded, expected, "Failed for value {}", value);
            
            let mut reader = &encoded[..];
            let decoded = decode_varint(&mut reader).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
        }
    }
    
    #[test]
    fn test_limited_reader() {
        let data = vec![1, 2, 3, 4, 5];
        let mut reader = LimitedReader::new(&data[..], 3);
        
        let mut buf = [0u8; 2];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [1, 2]);
        
        // Try to read more than limit
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 1); // Only 1 byte left before hitting limit
        assert_eq!(buf[0], 3);
        
        // Should return 0 now (limit reached)
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
    }
    
    #[test]
    fn test_vdw_proto_validation() {
        let mut vdw = VdwProto::new(
            [0u8; 32],
            1,
            100,
            vec![0u8; 500], // 500 byte proof
            [0u8; 32],
            vec![0u8; 100],
            vec![0u8; 100],
            1234567890,
            1,
        );
        
        // This should be valid (well under 1.8KB)
        assert!(vdw.validate_size().is_ok());
        
        // Make it too large
        vdw.delta_proof = vec![0u8; 2000]; // 2KB proof alone
        assert!(vdw.validate_size().is_err());
    }
    
    #[test]
    fn test_protobuf_serialization_roundtrip() {
        let original = NeuralVoteProto {
            predicted_hash: vec![1u8; 32],
            partial_signature: vec![2u8; 48],
            reputation_score: 0.95,
            tee_attestation: vec![3u8; 64],
            timestamp: 1234567890,
        };
        
        // Serialize
        let bytes = original.to_protobuf().unwrap();
        assert!(bytes.len() > 0);
        
        // Deserialize
        let restored = NeuralVoteProto::from_protobuf(&bytes).unwrap();
        
        assert_eq!(original.predicted_hash, restored.predicted_hash);
        assert_eq!(original.partial_signature, restored.partial_signature);
        assert!((original.reputation_score - restored.reputation_score).abs() < 0.0001);
        assert_eq!(original.tee_attestation, restored.tee_attestation);
        assert_eq!(original.timestamp, restored.timestamp);
    }
}
