use blake3;

// Poseidon implementation (simplified interface)
pub mod poseidon {
    use crate::embedding::circuit::Fp;
    
    pub struct NervPoseidon;
    
    impl NervPoseidon {
        pub fn new() -> Self { Self }
        
        pub fn hash_embedding(&self, embedding: &[Fp; 512]) -> [u8; 32] {
            // Simplified - real implementation would match circuit
            let mut result = [0u8; 32];
            for (i, &elem) in embedding.iter().enumerate() {
                result[i % 32] ^= (elem.to_repr().as_ref()[0]) as u8;
            }
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
                let array: [Fp; 512] = embedding.try_into().unwrap();
                self.poseidon.as_ref().unwrap().hash_embedding(&array)
            }
        }
    }
    
    pub fn hash_bytes(&self, data: &[u8]) -> [u8; 32] {
        match self.strategy {
            HashStrategy::Blake3 => blake3::hash(data).into(),
            HashStrategy::Poseidon => {
                // Convert bytes to field elements and hash
                // Simplified for now
                let mut result = [0u8; 32];
                for (i, &byte) in data.iter().enumerate() {
                    result[i % 32] ^= byte;
                }
                result
            }
        }
    }
}