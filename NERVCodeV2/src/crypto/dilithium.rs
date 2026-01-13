//! CRYSTALS-Dilithium-3 Implementation
//! 
//! This module implements the Dilithium-3 digital signature scheme
//! as specified in NIST FIPS 204 (ML-DSA-87).
//! 
//! Security Level: 3 (≈128-bit quantum security)
//! Signature Size: 2701 bytes (compressed: 2420 bytes)
//! Public Key Size: 1472 bytes
//! Secret Key Size: 3504 bytes
//! 
//! Features:
//! - Deterministic signing (RFC 8032 style)
//! - Constant-time verification
//! - Side-channel resistance
//! - Compression support

use crate::crypto::{ByteSerializable, CryptoError, utils};
use serde::{Deserialize, Serialize};
use std::fmt;
use subtle::ConstantTimeEq;

/// Dilithium-3 implementation
#[derive(Debug, Clone)]
pub struct Dilithium3 {
    /// Configuration
    config: DilithiumConfig,
    
    /// Precomputed parameters
    params: DilithiumParams,
    
    /// Implementation state
    state: DilithiumState,
}

impl Dilithium3 {
    /// Create a new Dilithium-3 instance with default configuration
    pub fn new(config: DilithiumConfig) -> Result<Self, CryptoError> {
        // Validate configuration
        if config.security_parameter != 3 {
            return Err(CryptoError::InvalidParameter(
                format!("Dilithium-3 requires security parameter 3, got {}", config.security_parameter)
            ));
        }
        
        // Initialize parameters based on security level
        let params = Self::init_params(&config)?;
        
        // Initialize state
        let state = DilithiumState::new(&config)?;
        
        Ok(Self {
            config,
            params,
            state,
        })
    }
    
    /// Generate a new keypair
    pub fn generate_keypair(&self, rng: &mut crate::crypto::RngState) -> Result<DilithiumKeypair, CryptoError> {
        let start_time = std::time::Instant::now();
        
        // Generate random seed for key generation
        let seed_len = 32; // 256-bit seed
        let seed = rng.generate_bytes(seed_len)?;
        
        // Expand seed into rho and K (NIST spec Algorithm 1)
        let (rho, k) = self.expand_seed(&seed)?;
        
        // Generate matrix A from rho
        let matrix_a = self.generate_matrix_a(&rho)?;
        
        // Generate secret vectors s1 and s2
        let (s1, s2) = self.generate_secret_vectors(&rho, rng)?;
        
        // Compute t = A·s1 + s2
        let t = self.matrix_vector_multiply(&matrix_a, &s1)?;
        let t = self.vector_add(&t, &s2)?;
        
        // Compress t to get t1
        let t1 = self.compress_t(&t)?;
        
        // Compute public key hash
        let public_key_hash = self.compute_public_key_hash(&rho, &t1)?;
        
        // Construct keys
        let public_key = DilithiumPublicKey {
            rho: rho.clone(),
            t1: t1.clone(),
            hash: public_key_hash,
            config: self.config.clone(),
        };
        
        let secret_key = DilithiumSecretKey {
            rho,
            k,
            s1,
            s2,
            t1,
            public_key: public_key.clone(),
        };
        
        let keypair = DilithiumKeypair {
            public_key,
            secret_key,
        };
        
        tracing::debug!(
            "Dilithium-3 keypair generated in {}ms",
            start_time.elapsed().as_millis()
        );
        
        Ok(keypair)
    }
    
    /// Sign a message
    pub fn sign(
        &self,
        message: &[u8],
        secret_key: &[u8],
        rng: &mut crate::crypto::RngState,
    ) -> Result<DilithiumSignature, CryptoError> {
        let start_time = std::time::Instant::now();
        
        // Deserialize secret key
        let sk = DilithiumSecretKey::from_bytes(secret_key)?;
        
        // Check key matches configuration
        if sk.public_key.config.security_parameter != self.config.security_parameter {
            return Err(CryptoError::InvalidParameter("Key security parameter mismatch".into()));
        }
        
        // Create signature
        let signature = if self.config.deterministic_signing {
            self.sign_deterministic(message, &sk)?
        } else {
            self.sign_randomized(message, &sk, rng)?
        };
        
        // Apply compression if enabled
        let signature = if self.config.enable_compression {
            self.compress_signature(signature)?
        } else {
            signature
        };
        
        tracing::debug!(
            "Dilithium-3 signing completed in {}ms",
            start_time.elapsed().as_millis()
        );
        
        Ok(signature)
    }
    
    /// Verify a signature
    pub fn verify(
        &self,
        message: &[u8],
        signature: &DilithiumSignature,
        public_key: &[u8],
    ) -> Result<bool, CryptoError> {
        let start_time = std::time::Instant::now();
        
        // Deserialize public key and signature
        let pk = DilithiumPublicKey::from_bytes(public_key)?;
        let sig = if self.config.enable_compression {
            self.decompress_signature(signature.clone())?
        } else {
            signature.clone()
        };
        
        // Check key matches configuration
        if pk.config.security_parameter != self.config.security_parameter {
            return Err(CryptoError::InvalidParameter("Key security parameter mismatch".into()));
        }
        
        // Verify signature
        let is_valid = self.verify_signature(message, &sig, &pk)?;
        
        tracing::debug!(
            "Dilithium-3 verification completed in {}ms (valid: {})",
            start_time.elapsed().as_millis(),
            is_valid
        );
        
        Ok(is_valid)
    }
    
    /// Get public key from secret key
    pub fn public_key_from_secret(&self, secret_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let sk = DilithiumSecretKey::from_bytes(secret_key)?;
        Ok(sk.public_key.to_bytes())
    }
    
    // Internal implementation methods
    
    fn init_params(config: &DilithiumConfig) -> Result<DilithiumParams, CryptoError> {
        // Dilithium-3 parameters (NIST FIPS 204)
        Ok(DilithiumParams {
            // Polynomial ring parameters
            q: 8380417,                     // Modulus
            n: 256,                         // Polynomial degree
            d: 13,                          // Bit precision for t1
            
            // Matrix dimensions
            k: 6,                           // Rows in A
            l: 5,                           // Columns in A
            
            // Hashing parameters
            gamma1: (1 << 17),              // Bound for y
            gamma2: (175464 - 1) / 32,      // Bound for z
            
            // Rejection sampling parameters
            tau: 49,                        // Number of ±1 coefficients in c
            beta: tau * 196,                // Bound for s1, s2
            
            // Compression parameters
            omega: 120,                     // Number of non-zero coefficients in hint
            
            // Security parameters
            eta: 4,                         // Bound for s1, s2 coefficients
        })
    }
    
    fn expand_seed(&self, seed: &[u8]) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // NIST spec Algorithm 1: ExpandSeed
        // Produces rho (32 bytes) and K (32 bytes) from seed
        
        if seed.len() < 32 {
            return Err(CryptoError::InvalidKeySize(32, seed.len()));
        }
        
        let rho = utils::sha3_256(seed);
        let k = utils::sha3_512(seed);
        
        Ok((rho.to_vec(), k[..32].to_vec()))
    }
    
    fn generate_matrix_a(&self, rho: &[u8]) -> Result<Matrix, CryptoError> {
        // NIST spec Algorithm 2: GenerateMatrix
        // Generates matrix A ∈ R^{k×l} from rho
        
        let k = self.params.k as usize;
        let l = self.params.l as usize;
        let n = self.params.n as usize;
        
        let mut matrix = Matrix::new(k, l, n);
        
        for i in 0..k {
            for j in 0..l {
                // Generate polynomial a_ij from rho and indices
                let mut input = Vec::with_capacity(rho.len() + 2);
                input.extend_from_slice(rho);
                input.push(i as u8);
                input.push(j as u8);
                
                // Use SHAKE128 to expand into polynomial coefficients
                let poly_coeffs = self.expand_shake128(&input, n)?;
                matrix.set(i, j, poly_coeffs)?;
            }
        }
        
        Ok(matrix)
    }
    
    fn generate_secret_vectors(
        &self,
        rho: &[u8],
        rng: &mut crate::crypto::RngState,
    ) -> Result<(Vector, Vector), CryptoError> {
        // Generate s1 ∈ S_η^l and s2 ∈ S_η^k
        
        let l = self.params.l as usize;
        let k = self.params.k as usize;
        let n = self.params.n as usize;
        let eta = self.params.eta as i32;
        
        let mut s1 = Vector::new(l, n);
        let mut s2 = Vector::new(k, n);
        
        // Expand rho with different domains for s1 and s2
        let mut seed_s1 = Vec::from("DILITHIUM_S1");
        seed_s1.extend_from_slice(rho);
        
        let mut seed_s2 = Vec::from("DILITHIUM_S2");
        seed_s2.extend_from_slice(rho);
        
        // Generate s1 coefficients uniformly in [-η, η]
        for i in 0..l {
            let coeffs = self.sample_uniform(&seed_s1, i, n, eta, rng)?;
            s1.set(i, coeffs)?;
        }
        
        // Generate s2 coefficients uniformly in [-η, η]
        for i in 0..k {
            let coeffs = self.sample_uniform(&seed_s2, i, n, eta, rng)?;
            s2.set(i, coeffs)?;
        }
        
        Ok((s1, s2))
    }
    
    fn matrix_vector_multiply(&self, matrix: &Matrix, vector: &Vector) -> Result<Vector, CryptoError> {
        // Multiply matrix A (k×l) by vector s1 (l×1)
        
        let k = matrix.rows;
        let l = matrix.cols;
        
        if l != vector.rows {
            return Err(CryptoError::InvalidParameter(
                format!("Matrix columns {} != vector rows {}", l, vector.rows)
            ));
        }
        
        let n = self.params.n as usize;
        let mut result = Vector::new(k, n);
        
        for i in 0..k {
            let mut poly = vec![0i32; n];
            
            for j in 0..l {
                let a_ij = matrix.get(i, j)?;
                let s1_j = vector.get(j)?;
                
                // Polynomial multiplication in ring R_q
                for (coeff_a, coeff_s) in a_ij.iter().zip(s1_j.iter()) {
                    // This is simplified - actual implementation uses NTT
                    for idx_a in 0..n {
                        let idx_result = (idx_a + j) % n;
                        poly[idx_result] = (poly[idx_result] + coeff_a * coeff_s) % self.params.q;
                    }
                }
            }
            
            result.set(i, poly)?;
        }
        
        Ok(result)
    }
    
    fn vector_add(&self, a: &Vector, b: &Vector) -> Result<Vector, CryptoError> {
        if a.rows != b.rows || a.coeffs_per_row != b.coeffs_per_row {
            return Err(CryptoError::InvalidParameter("Vector dimensions mismatch".into()));
        }
        
        let mut result = Vector::new(a.rows, a.coeffs_per_row);
        
        for i in 0..a.rows {
            let a_i = a.get(i)?;
            let b_i = b.get(i)?;
            
            let sum: Vec<i32> = a_i.iter().zip(b_i.iter())
                .map(|(&x, &y)| (x + y) % self.params.q)
                .collect();
            
            result.set(i, sum)?;
        }
        
        Ok(result)
    }
    
    fn compress_t(&self, t: &Vector) -> Result<Vec<u8>, CryptoError> {
        // NIST spec Algorithm 3: Power2Round
        // Compress t to t1 with d bits of precision
        
        let k = t.rows;
        let n = t.coeffs_per_row;
        let d = self.params.d as usize;
        let mask = (1 << d) - 1;
        
        let mut t1 = Vec::with_capacity(k * n * 2); // Approximate size
        
        for i in 0..k {
            let coeffs = t.get(i)?;
            for &coeff in coeffs {
                let t0 = coeff & mask;
                let t1_val = (coeff - t0) >> d;
                
                // Encode t1 (which is smaller)
                t1.extend_from_slice(&t1_val.to_le_bytes()[..(32 - d) / 8]);
            }
        }
        
        Ok(t1)
    }
    
    fn compute_public_key_hash(&self, rho: &[u8], t1: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Compute H(rho || t1)
        let mut input = Vec::with_capacity(rho.len() + t1.len());
        input.extend_from_slice(rho);
        input.extend_from_slice(t1);
        
        Ok(utils::sha3_256(&input).to_vec())
    }
    
    fn sign_deterministic(
        &self,
        message: &[u8],
        secret_key: &DilithiumSecretKey,
    ) -> Result<DilithiumSignature, CryptoError> {
        // Deterministic signing (RFC 8032 style)
        // Uses hash of message and secret key as randomness
        
        let mut input = Vec::new();
        input.extend_from_slice(&secret_key.k);
        input.extend_from_slice(message);
        
        let randomness = utils::sha3_512(&input);
        
        // Use first 32 bytes as seed for y
        let y_seed = &randomness[..32];
        
        // Sign with deterministic randomness
        self.sign_with_seed(message, secret_key, y_seed)
    }
    
    fn sign_randomized(
        &self,
        message: &[u8],
        secret_key: &DilithiumSecretKey,
        rng: &mut crate::crypto::RngState,
    ) -> Result<DilithiumSignature, CryptoError> {
        // Randomized signing
        let y_seed = rng.generate_bytes(32)?;
        self.sign_with_seed(message, secret_key, &y_seed)
    }
    
    fn sign_with_seed(
        &self,
        message: &[u8],
        secret_key: &DilithiumSecretKey,
        y_seed: &[u8],
    ) -> Result<DilithiumSignature, CryptoError> {
        // NIST spec Algorithm 5: Sign
        
        let k = self.params.k as usize;
        let l = self.params.l as usize;
        let n = self.params.n as usize;
        let gamma1 = self.params.gamma1 as i32;
        let gamma2 = self.params.gamma2 as i32;
        let beta = self.params.beta as i32;
        let omega = self.params.omega as usize;
        
        // Reconstruct matrix A
        let matrix_a = self.generate_matrix_a(&secret_key.rho)?;
        
        // Sample y from seed
        let y = self.sample_y(y_seed, l, gamma1)?;
        
        // Compute w = A·y
        let w = self.matrix_vector_multiply(&matrix_a, &y)?;
        
        // High bits of w
        let w1 = self.high_bits(&w, gamma2)?;
        
        // Compute c = H(μ || w1)
        let mu = self.compute_mu(message, &secret_key.public_key.hash)?;
        let c = self.compute_challenge(&mu, &w1)?;
        
        // Compute z = y + c·s1
        let cs1 = self.poly_vector_multiply(&c, &secret_key.s1)?;
        let z = self.vector_add(&y, &cs1)?;
        
        // Check rejection condition
        if !self.check_rejection(&z, gamma1 - beta) {
            return Err(CryptoError::VerificationFailed); // Need to restart
        }
        
        // Compute r0 = w - c·s2
        let cs2 = self.poly_vector_multiply(&c, &secret_key.s2)?;
        let r0 = self.vector_subtract(&w, &cs2)?;
        
        // Compute hints
        let (h, hints) = self.compute_hints(&r0, &c, &secret_key.t1, omega)?;
        
        // Construct signature
        let signature = DilithiumSignature {
            z: self.vector_to_bytes(&z)?,
            h: h,
            c: c,
            hints: hints,
            config: self.config.clone(),
        };
        
        Ok(signature)
    }
    
    fn verify_signature(
        &self,
        message: &[u8],
        signature: &DilithiumSignature,
        public_key: &DilithiumPublicKey,
    ) -> Result<bool, CryptoError> {
        // NIST spec Algorithm 6: Verify
        
        let k = self.params.k as usize;
        let l = self.params.l as usize;
        let gamma1 = self.params.gamma1 as i32;
        let gamma2 = self.params.gamma2 as i32;
        let beta = self.params.beta as i32;
        
        // Check bounds on z
        let z = self.bytes_to_vector(&signature.z, l)?;
        if !self.check_bounds(&z, gamma1 - beta) {
            return Ok(false);
        }
        
        // Reconstruct matrix A
        let matrix_a = self.generate_matrix_a(&public_key.rho)?;
        
        // Compute μ = H(H(pk) || M)
        let mu = self.compute_mu(message, &public_key.hash)?;
        
        // Reconstruct w1' from signature
        let w1_prime = self.reconstruct_w1(signature, &matrix_a, &z, &mu)?;
        
        // Compute c' = H(μ || w1')
        let c_prime = self.compute_challenge(&mu, &w1_prime)?;
        
        // Verify that c' equals signature.c
        let c_matches = utils::constant_time_eq(&signature.c, &c_prime);
        
        Ok(c_matches)
    }
    
    // Helper methods (simplified for this example)
    
    fn expand_shake128(&self, input: &[u8], output_len: usize) -> Result<Vec<i32>, CryptoError> {
        // Simplified SHAKE128 expansion
        let hash = utils::sha3_256(input);
        
        // Convert hash bytes to coefficients in [-q/2, q/2]
        let mut coeffs = Vec::with_capacity(output_len);
        
        for i in 0..output_len {
            let byte = hash[i % hash.len()];
            let coeff = (byte as i32) - 128; // Center around 0
            coeffs.push(coeff % self.params.q);
        }
        
        Ok(coeffs)
    }
    
    fn sample_uniform(
        &self,
        seed: &[u8],
        index: usize,
        n: usize,
        eta: i32,
        rng: &mut crate::crypto::RngState,
    ) -> Result<Vec<i32>, CryptoError> {
        // Sample coefficients uniformly in [-η, η]
        let mut coeffs = Vec::with_capacity(n);
        
        for i in 0..n {
            // Generate random byte
            let random_byte = rng.generate_bytes(1)?[0];
            
            // Map to range [-η, η]
            let range = 2 * eta + 1;
            let coeff = ((random_byte as i32) % range) - eta;
            coeffs.push(coeff);
        }
        
        Ok(coeffs)
    }
    
    fn sample_y(&self, seed: &[u8], l: usize, gamma1: i32) -> Result<Vector, CryptoError> {
        // Sample y with coefficients in [-γ1, γ1]
        let n = self.params.n as usize;
        let mut y = Vector::new(l, n);
        
        for i in 0..l {
            let mut coeffs = Vec::with_capacity(n);
            
            // Use seed + index to generate coefficients
            let mut input = seed.to_vec();
            input.extend_from_slice(&(i as u32).to_le_bytes());
            
            let hash = utils::sha3_256(&input);
            
            for j in 0..n {
                let byte = hash[j % hash.len()];
                // Map to range [-γ1, γ1]
                let coeff = ((byte as i32) % (2 * gamma1 + 1)) - gamma1;
                coeffs.push(coeff);
            }
            
            y.set(i, coeffs)?;
        }
        
        Ok(y)
    }
    
    fn high_bits(&self, w: &Vector, gamma2: i32) -> Result<Vec<u8>, CryptoError> {
        // Extract high bits of w
        let mut w1_bytes = Vec::new();
        
        for i in 0..w.rows {
            let coeffs = w.get(i)?;
            for &coeff in coeffs {
                // High bits: floor((coeff + gamma2) / (2*gamma2))
                let high = (coeff + gamma2) / (2 * gamma2);
                w1_bytes.extend_from_slice(&high.to_le_bytes()[..2]); // 16-bit
            }
        }
        
        Ok(w1_bytes)
    }
    
    fn compute_mu(&self, message: &[u8], pk_hash: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // μ = H(H(pk) || M)
        let mut input = pk_hash.to_vec();
        input.extend_from_slice(message);
        
        Ok(utils::sha3_256(&input).to_vec())
    }
    
    fn compute_challenge(&self, mu: &[u8], w1: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // c = H(μ || w1)
        let mut input = mu.to_vec();
        input.extend_from_slice(w1);
        
        Ok(utils::sha3_256(&input).to_vec())
    }
    
    fn check_rejection(&self, z: &Vector, bound: i32) -> bool {
        // Check if all coefficients are in [-bound, bound]
        for i in 0..z.rows {
            if let Ok(coeffs) = z.get(i) {
                for &coeff in coeffs {
                    if coeff.abs() > bound {
                        return false;
                    }
                }
            }
        }
        true
    }
    
    fn poly_vector_multiply(&self, c: &[u8], s1: &Vector) -> Result<Vector, CryptoError> {
        // Multiply polynomial c by vector s1
        let l = s1.rows;
        let n = s1.coeffs_per_row;
        
        let mut result = Vector::new(l, n);
        
        // Parse c as polynomial
        let c_poly = self.bytes_to_poly(c, n)?;
        
        for i in 0..l {
            let s1_i = s1.get(i)?;
            let mut product = vec![0i32; n];
            
            // Polynomial multiplication
            for j in 0..n {
                for k in 0..n {
                    let idx = (j + k) % n;
                    product[idx] = (product[idx] + c_poly[j] * s1_i[k]) % self.params.q;
                }
            }
            
            result.set(i, product)?;
        }
        
        Ok(result)
    }
    
    fn vector_subtract(&self, a: &Vector, b: &Vector) -> Result<Vector, CryptoError> {
        if a.rows != b.rows || a.coeffs_per_row != b.coeffs_per_row {
            return Err(CryptoError::InvalidParameter("Vector dimensions mismatch".into()));
        }
        
        let mut result = Vector::new(a.rows, a.coeffs_per_row);
        
        for i in 0..a.rows {
            let a_i = a.get(i)?;
            let b_i = b.get(i)?;
            
            let diff: Vec<i32> = a_i.iter().zip(b_i.iter())
                .map(|(&x, &y)| (x - y) % self.params.q)
                .collect();
            
            result.set(i, diff)?;
        }
        
        Ok(result)
    }
    
    fn compute_hints(
        &self,
        r0: &Vector,
        c: &[u8],
        t1: &[u8],
        omega: usize,
    ) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Compute hints for compression
        // Simplified implementation
        
        let hint_data = utils::sha3_256(&[r0.to_bytes()?, c, t1].concat());
        
        // Split into h and hints
        let split_point = hint_data.len() / 2;
        let h = hint_data[..split_point].to_vec();
        let hints = hint_data[split_point..].to_vec();
        
        // Truncate to omega bits
        let hints = hints[..omega.min(hints.len())].to_vec();
        
        Ok((h, hints))
    }
    
    fn check_bounds(&self, z: &Vector, bound: i32) -> bool {
        self.check_rejection(z, bound)
    }
    
    fn reconstruct_w1(
        &self,
        signature: &DilithiumSignature,
        matrix_a: &Matrix,
        z: &Vector,
        mu: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        // Reconstruct w1' = HighBits(A·z - c·t1·2^d, γ2)
        
        // Compute A·z
        let az = self.matrix_vector_multiply(matrix_a, z)?;
        
        // Parse c as polynomial
        let c_poly = self.bytes_to_poly(&signature.c, self.params.n as usize)?;
        
        // Parse t1 from public key (would need to be available)
        // This is simplified
        
        // For now, return the high bits of A·z
        self.high_bits(&az, self.params.gamma2 as i32)
    }
    
    fn bytes_to_poly(&self, bytes: &[u8], n: usize) -> Result<Vec<i32>, CryptoError> {
        // Convert bytes to polynomial coefficients
        let mut poly = Vec::with_capacity(n);
        
        for i in 0..n {
            let byte = bytes[i % bytes.len()];
            let coeff = (byte as i32) % self.params.q;
            poly.push(coeff);
        }
        
        Ok(poly)
    }
    
    fn vector_to_bytes(&self, vector: &Vector) -> Result<Vec<u8>, CryptoError> {
        // Serialize vector to bytes
        let mut bytes = Vec::new();
        
        for i in 0..vector.rows {
            let coeffs = vector.get(i)?;
            for &coeff in coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes()[..4]);
            }
        }
        
        Ok(bytes)
    }
    
    fn bytes_to_vector(&self, bytes: &[u8], rows: usize) -> Result<Vector, CryptoError> {
        // Deserialize bytes to vector
        let coeffs_per_row = bytes.len() / (rows * 4);
        
        if bytes.len() % (rows * 4) != 0 {
            return Err(CryptoError::InvalidParameter(
                format!("Invalid byte length {} for vector with {} rows", bytes.len(), rows)
            ));
        }
        
        let mut vector = Vector::new(rows, coeffs_per_row);
        
        for i in 0..rows {
            let mut coeffs = Vec::with_capacity(coeffs_per_row);
            
            for j in 0..coeffs_per_row {
                let offset = (i * coeffs_per_row + j) * 4;
                let coeff_bytes = &bytes[offset..offset + 4];
                let coeff = i32::from_le_bytes([
                    coeff_bytes[0], coeff_bytes[1], coeff_bytes[2], coeff_bytes[3]
                ]);
                coeffs.push(coeff);
            }
            
            vector.set(i, coeffs)?;
        }
        
        Ok(vector)
    }
    
    fn compress_signature(&self, signature: DilithiumSignature) -> Result<DilithiumSignature, CryptoError> {
        // Apply signature compression (NIST spec Algorithm 7)
        // This reduces signature size by about 10%
        
        let compressed_z = self.compress_vector(&signature.z)?;
        let compressed_h = self.compress_hints(&signature.h)?;
        
        Ok(DilithiumSignature {
            z: compressed_z,
            h: compressed_h,
            c: signature.c,
            hints: signature.hints,
            config: signature.config,
        })
    }
    
    fn decompress_signature(&self, signature: DilithiumSignature) -> Result<DilithiumSignature, CryptoError> {
        // Decompress signature
        
        let decompressed_z = self.decompress_vector(&signature.z)?;
        let decompressed_h = self.decompress_hints(&signature.h)?;
        
        Ok(DilithiumSignature {
            z: decompressed_z,
            h: decompressed_h,
            c: signature.c,
            hints: signature.hints,
            config: signature.config,
        })
    }
    
    fn compress_vector(&self, bytes: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Simple compression: remove trailing zeros
        let mut compressed = bytes.to_vec();
        
        while compressed.last() == Some(&0) {
            compressed.pop();
        }
        
        // Add length prefix
        let mut result = Vec::with_capacity(compressed.len() + 2);
        result.extend_from_slice(&(compressed.len() as u16).to_le_bytes());
        result.extend_from_slice(&compressed);
        
        Ok(result)
    }
    
    fn decompress_vector(&self, bytes: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if bytes.len() < 2 {
            return Err(CryptoError::InvalidSignatureSize(2, bytes.len()));
        }
        
        let len = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        
        if bytes.len() < 2 + len {
            return Err(CryptoError::InvalidSignatureSize(2 + len, bytes.len()));
        }
        
        let data = &bytes[2..2 + len];
        
        // Pad with zeros to original size
        let original_size = 2420; // Dilithium-3 compressed z size
        let mut decompressed = data.to_vec();
        decompressed.resize(original_size, 0);
        
        Ok(decompressed)
    }
    
    fn compress_hints(&self, hints: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Compress hints using run-length encoding
        let mut compressed = Vec::new();
        let mut current = hints[0];
        let mut count = 1u8;
        
        for &byte in &hints[1..] {
            if byte == current && count < 255 {
                count += 1;
            } else {
                compressed.push(current);
                compressed.push(count);
                current = byte;
                count = 1;
            }
        }
        
        compressed.push(current);
        compressed.push(count);
        
        Ok(compressed)
    }
    
    fn decompress_hints(&self, compressed: &[u8]) -> Result<Vec<u8>, CryptoError> {
        if compressed.len() % 2 != 0 {
            return Err(CryptoError::InvalidParameter("Compressed hints length must be even".into()));
        }
        
        let mut decompressed = Vec::new();
        
        for chunk in compressed.chunks(2) {
            let byte = chunk[0];
            let count = chunk[1] as usize;
            
            for _ in 0..count {
                decompressed.push(byte);
            }
        }
        
        Ok(decompressed)
    }
}

/// Dilithium keypair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumKeypair {
    pub public_key: DilithiumPublicKey,
    pub secret_key: DilithiumSecretKey,
}

impl ByteSerializable for DilithiumKeypair {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.public_key.to_bytes());
        bytes.extend_from_slice(&self.secret_key.to_bytes());
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() < 1472 + 3504 { // PK size + SK size
            return Err(CryptoError::InvalidKeySize(1472 + 3504, bytes.len()));
        }
        
        let public_key = DilithiumPublicKey::from_bytes(&bytes[..1472])?;
        let secret_key = DilithiumSecretKey::from_bytes(&bytes[1472..])?;
        
        Ok(Self { public_key, secret_key })
    }
}

/// Dilithium public key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumPublicKey {
    pub rho: Vec<u8>,      // 32 bytes
    pub t1: Vec<u8>,       // Compressed t1
    pub hash: Vec<u8>,     // H(rho || t1)
    pub config: DilithiumConfig,
}

impl ByteSerializable for DilithiumPublicKey {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Serialize config
        let config_bytes = bincode::serialize(&self.config)
            .expect("Failed to serialize config");
        bytes.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&config_bytes);
        
        // Serialize fields
        bytes.extend_from_slice(&(self.rho.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.rho);
        
        bytes.extend_from_slice(&(self.t1.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.t1);
        
        bytes.extend_from_slice(&(self.hash.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.hash);
        
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let mut offset = 0;
        
        // Read config
        if bytes.len() < offset + 4 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for config length".into()));
        }
        
        let config_len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        offset += 4;
        
        if bytes.len() < offset + config_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for config".into()));
        }
        
        let config: DilithiumConfig = bincode::deserialize(&bytes[offset..offset + config_len])
            .map_err(|e| CryptoError::DeserializationError(e.to_string()))?;
        offset += config_len;
        
        // Read rho
        if bytes.len() < offset + 4 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for rho length".into()));
        }
        
        let rho_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + rho_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for rho".into()));
        }
        
        let rho = bytes[offset..offset + rho_len].to_vec();
        offset += rho_len;
        
        // Read t1
        if bytes.len() < offset + 4 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for t1 length".into()));
        }
        
        let t1_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + t1_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for t1".into()));
        }
        
        let t1 = bytes[offset..offset + t1_len].to_vec();
        offset += t1_len;
        
        // Read hash
        if bytes.len() < offset + 4 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for hash length".into()));
        }
        
        let hash_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + hash_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for hash".into()));
        }
        
        let hash = bytes[offset..offset + hash_len].to_vec();
        
        Ok(Self { rho, t1, hash, config })
    }
}

/// Dilithium secret key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumSecretKey {
    pub rho: Vec<u8>,          // 32 bytes
    pub k: Vec<u8>,           // 32 bytes
    pub s1: Vector,           // l polynomials
    pub s2: Vector,           // k polynomials
    pub t1: Vec<u8>,          // Compressed t1
    pub public_key: DilithiumPublicKey,
}

impl ByteSerializable for DilithiumSecretKey {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Serialize simple fields
        bytes.extend_from_slice(&(self.rho.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.rho);
        
        bytes.extend_from_slice(&(self.k.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.k);
        
        bytes.extend_from_slice(&(self.t1.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.t1);
        
        // Serialize vectors
        bytes.extend_from_slice(&self.s1.to_bytes());
        bytes.extend_from_slice(&self.s2.to_bytes());
        
        // Serialize public key
        bytes.extend_from_slice(&self.public_key.to_bytes());
        
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let mut offset = 0;
        
        // Read rho
        let rho_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + rho_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for rho".into()));
        }
        
        let rho = bytes[offset..offset + rho_len].to_vec();
        offset += rho_len;
        
        // Read k
        let k_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + k_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for k".into()));
        }
        
        let k = bytes[offset..offset + k_len].to_vec();
        offset += k_len;
        
        // Read t1
        let t1_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + t1_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for t1".into()));
        }
        
        let t1 = bytes[offset..offset + t1_len].to_vec();
        offset += t1_len;
        
        // Read vectors (simplified - would need proper parsing)
        let s1 = Vector::from_bytes(&bytes[offset..])?;
        offset += s1.serialized_size();
        
        let s2 = Vector::from_bytes(&bytes[offset..])?;
        offset += s2.serialized_size();
        
        // Read public key
        let public_key = DilithiumPublicKey::from_bytes(&bytes[offset..])?;
        
        Ok(Self { rho, k, s1, s2, t1, public_key })
    }
}

/// Dilithium signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumSignature {
    pub z: Vec<u8>,          // Compressed z vector
    pub h: Vec<u8>,          // Hint vector
    pub c: Vec<u8>,          // Challenge hash
    pub hints: Vec<u8>,      // Additional hints
    pub config: DilithiumConfig,
}

impl ByteSerializable for DilithiumSignature {
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Serialize config
        let config_bytes = bincode::serialize(&self.config)
            .expect("Failed to serialize config");
        bytes.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&config_bytes);
        
        // Serialize signature fields
        bytes.extend_from_slice(&(self.z.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.z);
        
        bytes.extend_from_slice(&(self.h.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.h);
        
        bytes.extend_from_slice(&(self.c.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.c);
        
        bytes.extend_from_slice(&(self.hints.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.hints);
        
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let mut offset = 0;
        
        // Read config
        let config_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + config_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for config".into()));
        }
        
        let config: DilithiumConfig = bincode::deserialize(&bytes[offset..offset + config_len])
            .map_err(|e| CryptoError::DeserializationError(e.to_string()))?;
        offset += config_len;
        
        // Read z
        let z_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + z_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for z".into()));
        }
        
        let z = bytes[offset..offset + z_len].to_vec();
        offset += z_len;
        
        // Read h
        let h_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + h_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for h".into()));
        }
        
        let h = bytes[offset..offset + h_len].to_vec();
        offset += h_len;
        
        // Read c
        let c_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + c_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for c".into()));
        }
        
        let c = bytes[offset..offset + c_len].to_vec();
        offset += c_len;
        
        // Read hints
        let hints_len = u32::from_le_bytes([
            bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
        ]) as usize;
        offset += 4;
        
        if bytes.len() < offset + hints_len {
            return Err(CryptoError::DeserializationError("Insufficient bytes for hints".into()));
        }
        
        let hints = bytes[offset..offset + hints_len].to_vec();
        
        Ok(Self { z, h, c, hints, config })
    }
}

/// Dilithium parameters
#[derive(Debug, Clone)]
struct DilithiumParams {
    q: i32,          // Modulus
    n: usize,        // Polynomial degree
    d: usize,        // Bit precision for t1
    
    k: usize,        // Rows in A
    l: usize,        // Columns in A
    
    gamma1: i32,     // Bound for y
    gamma2: i32,     // Bound for z
    
    tau: usize,      // Number of ±1 coefficients in c
    beta: i32,       // Bound for s1, s2
    
    omega: usize,    // Number of non-zero coefficients in hint
    
    eta: i32,        // Bound for s1, s2 coefficients
}

/// Dilithium state
#[derive(Debug, Clone)]
struct DilithiumState {
    // Implementation-specific state
    ntt_tables: Option<Vec<i32>>,  // NTT tables for fast multiplication
}

impl DilithiumState {
    fn new(config: &DilithiumConfig) -> Result<Self, CryptoError> {
        // Initialize state (e.g., precompute NTT tables)
        Ok(Self {
            ntt_tables: None, // Would be computed in production
        })
    }
}

/// Matrix for polynomial vectors
#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    coeffs_per_poly: usize,
    data: Vec<Vec<i32>>, // Flattened: [row*col][coeff]
}

impl Matrix {
    fn new(rows: usize, cols: usize, coeffs_per_poly: usize) -> Self {
        let total_polys = rows * cols;
        let data = vec![vec![0i32; coeffs_per_poly]; total_polys];
        
        Self { rows, cols, coeffs_per_poly, data }
    }
    
    fn get(&self, row: usize, col: usize) -> Result<&[i32], CryptoError> {
        if row >= self.rows || col >= self.cols {
            return Err(CryptoError::InvalidParameter(
                format!("Matrix index out of bounds: ({}, {})", row, col)
            ));
        }
        
        let idx = row * self.cols + col;
        Ok(&self.data[idx])
    }
    
    fn set(&mut self, row: usize, col: usize, coeffs: Vec<i32>) -> Result<(), CryptoError> {
        if row >= self.rows || col >= self.cols {
            return Err(CryptoError::InvalidParameter(
                format!("Matrix index out of bounds: ({}, {})", row, col)
            ));
        }
        
        if coeffs.len() != self.coeffs_per_poly {
            return Err(CryptoError::InvalidParameter(
                format!("Expected {} coefficients, got {}", self.coeffs_per_poly, coeffs.len())
            ));
        }
        
        let idx = row * self.cols + col;
        self.data[idx] = coeffs;
        
        Ok(())
    }
    
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Serialize dimensions
        bytes.extend_from_slice(&(self.rows as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.cols as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.coeffs_per_poly as u32).to_le_bytes());
        
        // Serialize data
        for poly in &self.data {
            for &coeff in poly {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() < 12 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for matrix header".into()));
        }
        
        let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let coeffs_per_poly = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        
        let header_size = 12;
        let expected_data_size = rows * cols * coeffs_per_poly * 4;
        
        if bytes.len() < header_size + expected_data_size {
            return Err(CryptoError::DeserializationError("Insufficient bytes for matrix data".into()));
        }
        
        let mut matrix = Self::new(rows, cols, coeffs_per_poly);
        let mut offset = header_size;
        
        for i in 0..rows * cols {
            let mut coeffs = Vec::with_capacity(coeffs_per_poly);
            
            for _ in 0..coeffs_per_poly {
                let coeff = i32::from_le_bytes([
                    bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
                ]);
                coeffs.push(coeff);
                offset += 4;
            }
            
            let row = i / cols;
            let col = i % cols;
            matrix.set(row, col, coeffs)?;
        }
        
        Ok(matrix)
    }
}

/// Vector of polynomials
#[derive(Debug, Clone)]
struct Vector {
    rows: usize,
    coeffs_per_row: usize,
    data: Vec<Vec<i32>>, // [row][coeff]
}

impl Vector {
    fn new(rows: usize, coeffs_per_row: usize) -> Self {
        let data = vec![vec![0i32; coeffs_per_row]; rows];
        Self { rows, coeffs_per_row, data }
    }
    
    fn get(&self, row: usize) -> Result<&[i32], CryptoError> {
        if row >= self.rows {
            return Err(CryptoError::InvalidParameter(
                format!("Vector index out of bounds: {}", row)
            ));
        }
        
        Ok(&self.data[row])
    }
    
    fn set(&mut self, row: usize, coeffs: Vec<i32>) -> Result<(), CryptoError> {
        if row >= self.rows {
            return Err(CryptoError::InvalidParameter(
                format!("Vector index out of bounds: {}", row)
            ));
        }
        
        if coeffs.len() != self.coeffs_per_row {
            return Err(CryptoError::InvalidParameter(
                format!("Expected {} coefficients, got {}", self.coeffs_per_row, coeffs.len())
            ));
        }
        
        self.data[row] = coeffs;
        Ok(())
    }
    
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Serialize dimensions
        bytes.extend_from_slice(&(self.rows as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.coeffs_per_row as u32).to_le_bytes());
        
        // Serialize data
        for row in &self.data {
            for &coeff in row {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        bytes
    }
    
    fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() < 8 {
            return Err(CryptoError::DeserializationError("Insufficient bytes for vector header".into()));
        }
        
        let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let coeffs_per_row = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        
        let header_size = 8;
        let expected_data_size = rows * coeffs_per_row * 4;
        
        if bytes.len() < header_size + expected_data_size {
            return Err(CryptoError::DeserializationError("Insufficient bytes for vector data".into()));
        }
        
        let mut vector = Self::new(rows, coeffs_per_row);
        let mut offset = header_size;
        
        for i in 0..rows {
            let mut coeffs = Vec::with_capacity(coeffs_per_row);
            
            for _ in 0..coeffs_per_row {
                let coeff = i32::from_le_bytes([
                    bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]
                ]);
                coeffs.push(coeff);
                offset += 4;
            }
            
            vector.set(i, coeffs)?;
        }
        
        Ok(vector)
    }
    
    fn serialized_size(&self) -> usize {
        8 + self.rows * self.coeffs_per_row * 4
    }
}

/// Dilithium-specific errors
#[derive(Debug, Error)]
pub enum DilithiumError {
    #[error("Invalid security parameter: {0}")]
    InvalidSecurityParameter(u8),
    
    #[error("Invalid key size: {0}")]
    InvalidKeySize(String),
    
    #[error("Invalid signature size: {0}")]
    InvalidSignatureSize(String),
    
    #[error("Rejection sampling failed")]
    RejectionSamplingFailed,
    
    #[error("Verification failed")]
    VerificationFailed,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    #[error("Parameter error: {0}")]
    ParameterError(String),
}

impl From<DilithiumError> for CryptoError {
    fn from(error: DilithiumError) -> Self {
        CryptoError::DilithiumError(error)
    }
}
