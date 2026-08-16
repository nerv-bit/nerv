use blake3;
use halo2_gadgets::poseidon::primitives::{
    self as poseidon_primitives,
    Spec, ConstantLength, Hash as PoseidonHash, P128Pow5T3,
};
use pasta_curves::pallas::Scalar as Fp;  // Assuming Fp is pallas::Scalar (matches circuits)

// Poseidon implementation (simplified interface)
pub mod poseidon {
    use super::*;  // Re-export Fp, etc.
    use halo2_gadgets::poseidon::primitives::{
        Spec, ConstantLength, Hash as PoseidonHash, P128Pow5T3,
    };

    /// Production Poseidon hasher matching Halo2 circuit config
    /// Uses P128Pow5T3 (width=3, secure for hash) with full rounds
    /// Fixed input length: 512 field elements (embedding size)
    pub struct NervPoseidon;

    impl NervPoseidon {
        pub fn new() -> Self { Self }

        /// Hash 512-dim embedding → 32-byte digest (single squeezed field)
        /// Matches circuit commitment for recursive provability
        pub fn hash_embedding(&self, embedding: &[Fp; 512]) -> [u8; 32] {
            // Spec: width=3 (t=3), rate=2 (absorb 2 per round), matches common secure hash params
            // Adjust if your LatentLedger circuit uses different t/rate/full_rounds
            let spec = Spec::<Fp, 3, 2>::new(8, 56);  // 8 full, 56 partial - standard secure

            // Fixed-length hasher for exactly 512 inputs
            let mut hasher = PoseidonHash::<_, Spec<Fp, 3, 2>, ConstantLength<512>, 3, 2>::init(
                spec,
                ConstantLength::<512>,
            );

            // Absorb all 512 elements
            hasher.update(embedding);

            // Squeeze one field (32 bytes)
            let digest = hasher.squeeze();

            // Convert scalar to 32-byte little-endian
            let mut result = [0u8; 32];
            result.copy_from_slice(&digest.to_repr()[..32]);
            result
        }
    }
}


// Dual hash strategy
pub enum HashStrategy {
    Blake3,
    Poseidon,
}

pub struct NervHasher {
    strategy: HashStrategy,
    poseidon: Option<poseidon::NervPoseidon>,
}

impl NervHasher {
    pub fn new(strategy: HashStrategy) -> Self {
        let poseidon = match strategy {
            HashStrategy::Poseidon => Some(poseidon::NervPoseidon::new()),
            _ => None,
        };
        
        Self { strategy, poseidon }
    }
    
    pub fn hash_embedding(&self, embedding: &[Fp]) -> [u8; 32] {
    match self.strategy {
        HashStrategy::Blake3 => {
            let serialized = bincode::serialize(embedding).unwrap();
            blake3::hash(&serialized).into()
        }
        HashStrategy::Poseidon => {
            // Fixed 512-dim embedding
            let array: [Fp; 512] = embedding.try_into().expect("Embedding must be 512 elements");
            poseidon::NervPoseidon::new().hash_embedding(&array)
        }
    }
}

pub fn hash_bytes(&self, data: &[u8]) -> [u8; 32] {
    match self.strategy {
        HashStrategy::Blake3 => blake3::hash(data).into(),
        HashStrategy::Poseidon => {
            // Arbitrary-length sponge mode: chunk bytes into fields (rate=2 → ~64 bytes/round)
            let spec = Spec::<Fp, 3, 2>::new(8, 56);

            // Use variable-length sponge (no ConstantLength)
            let mut hasher = poseidon_primitives::Hash::<_, Spec<Fp, 3, 2>, poseidon_primitives::Orb, 3, 2>::init(
                spec,
                poseidon_primitives::Orb::new(0),  // Domain separator 0 for general hash
            );

            // Absorb bytes in chunks (convert to fields)
            for chunk in data.chunks(31) {  // <32 bytes to fit field
                let mut field_bytes = [0u8; 32];
                field_bytes[..chunk.len()].copy_from_slice(chunk);
                let field = Fp::from_repr(field_bytes.into()).unwrap();
                hasher.update(&[field]);
            }

            let digest = hasher.squeeze();
            let mut result = [0u8; 32];
            result.copy_from_slice(&digest.to_repr()[..32]);
            result
        }
    }
}

    
   
