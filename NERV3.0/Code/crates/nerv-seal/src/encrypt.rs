//! Encryption and public aggregate addition (WP §6.3.1–§6.3.2).
//!
//! ct = (u, v) with u = A·r + e₁ ∈ R^8 and v = T·r + e₂ + scale·m ∈ R^2:
//! A is the epoch's XOF expansion, T = s·A + E₀ the DKG's public output
//! (erratum 20), (r, e₁, e₂) the wallet's per-leg CBD(2) triplet under
//! `nerv.seal.noise`, and m the digitized delta. Aggregation is
//! componentwise polynomial addition mod q — public, linear,
//! order-invariant; producers apply it to build ct_B.
//!
//! `derive_reference_keypair` is the native twin of the DKG's output
//! contract (DSR-7): s and E₀ drawn CBD(2), hence bounded by
//! `noise::KEY_SHORT_BOUND`. The protocol never holds a single-party
//! secret key; threshold partials arrive with chunk 9.

use nerv_core::constants::SEAL_NOISE;
use nerv_core::hash::Xof;

use crate::digitize::Plaintext;
use crate::error::SealError;
use crate::noise::SCALE;
use crate::ring::{Mat2x8, Mat2x8Ntt, Mat8x8, Mat8x8Ntt, Vec2, Vec8};
use crate::sampling::{derive_short_triplet, expand_matrix, sample_short_vec8, ASeed, NoiseSeed};

/// The per-shard-epoch seal public key: the A-expansion seed plus T. The
/// seed is the wire object for A (never the 64-polynomial matrix, erratum
/// 21); T is the DKG's committed output.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PublicKey {
    a_seed: ASeed,
    t: Mat2x8,
}

impl PublicKey {
    /// 32-byte seed ‖ two T rows (erratum 33).
    pub const WIRE_SIZE: usize = 32 + 2 * Vec8::WIRE_SIZE;

    pub fn new(a_seed: ASeed, t: Mat2x8) -> PublicKey {
        PublicKey { a_seed, t }
    }

    pub fn a_seed(&self) -> &ASeed {
        &self.a_seed
    }

    pub fn t(&self) -> &Mat2x8 {
        &self.t
    }

    /// Re-derives A from the committed seed — the public verification of
    /// the matrix ciphertexts were formed against.
    pub fn expand_a(&self) -> Result<Mat8x8, SealError> {
        expand_matrix(&self.a_seed)
    }

    pub fn to_bytes(&self) -> [u8; PublicKey::WIRE_SIZE] {
        let mut out = [0u8; PublicKey::WIRE_SIZE];
        out[..32].copy_from_slice(self.a_seed.as_bytes());
        let rows = [Vec8::new(*self.t.row(0)), Vec8::new(*self.t.row(1))];
        for (i, row) in rows.iter().enumerate() {
            let at = 32 + i * Vec8::WIRE_SIZE;
            out[at..at + Vec8::WIRE_SIZE].copy_from_slice(&row.to_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<PublicKey, SealError> {
        if bytes.len() != PublicKey::WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: PublicKey::WIRE_SIZE });
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&bytes[..32]);
        let row0 = Vec8::from_bytes(&bytes[32..32 + Vec8::WIRE_SIZE])?;
        let row1 = Vec8::from_bytes(&bytes[32 + Vec8::WIRE_SIZE..])?;
        Ok(PublicKey {
            a_seed: ASeed::from_bytes(seed),
            t: Mat2x8::new([*row0.polys(), *row1.polys()]),
        })
    }
}

/// A sealed delta: u = A·r + e₁, v = T·r + e₂ + scale·m (WP §6.3.1).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ciphertext {
    u: Vec8,
    v: Vec2,
}

impl Ciphertext {
    /// 10,240 bytes — u dominates (WP §6.3.8, erratum 33).
    pub const WIRE_SIZE: usize = Vec8::WIRE_SIZE + Vec2::WIRE_SIZE;

    pub const fn zero() -> Ciphertext {
        Ciphertext { u: Vec8::zero(), v: Vec2::zero() }
    }

    pub fn u(&self) -> &Vec8 {
        &self.u
    }

    pub fn v(&self) -> &Vec2 {
        &self.v
    }

    /// §6.3.2 aggregate addition: componentwise polynomial addition mod q.
    pub fn add(&self, rhs: &Ciphertext) -> Ciphertext {
        Ciphertext { u: self.u.add(&rhs.u), v: self.v.add(&rhs.v) }
    }

    pub fn to_bytes(&self) -> [u8; Ciphertext::WIRE_SIZE] {
        let mut out = [0u8; Ciphertext::WIRE_SIZE];
        out[..Vec8::WIRE_SIZE].copy_from_slice(&self.u.to_bytes());
        out[Vec8::WIRE_SIZE..].copy_from_slice(&self.v.to_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Ciphertext, SealError> {
        if bytes.len() != Ciphertext::WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: Ciphertext::WIRE_SIZE });
        }
        let u = Vec8::from_bytes(&bytes[..Vec8::WIRE_SIZE])?;
        let v = Vec2::from_bytes(&bytes[Vec8::WIRE_SIZE..])?;
        Ok(Ciphertext { u, v })
    }

    /// Seals `m` under the public key with wallet entropy `seed`.
    /// Deterministic in (pk, seed, m).
    pub fn encrypt(
        pk: &PublicKey,
        seed: &NoiseSeed,
        m: &Plaintext,
    ) -> Result<Ciphertext, SealError> {
        let a_ntt = expand_matrix(pk.a_seed())?.ntt();
        let t_ntt = pk.t().ntt();
        Ciphertext::seal(&a_ntt, &t_ntt, seed, m)
    }

    /// The cached-matrix path: callers holding the epoch's expanded and
    /// transformed matrices skip the re-expansion.
    pub fn encrypt_cached(
        a_ntt: &Mat8x8Ntt,
        t_ntt: &Mat2x8Ntt,
        seed: &NoiseSeed,
        m: &Plaintext,
    ) -> Result<Ciphertext, SealError> {
        Ciphertext::seal(a_ntt, t_ntt, seed, m)
    }

    fn seal(
        a_ntt: &Mat8x8Ntt,
        t_ntt: &Mat2x8Ntt,
        seed: &NoiseSeed,
        m: &Plaintext,
    ) -> Result<Ciphertext, SealError> {
        let (r, e1, e2) = derive_short_triplet(seed)?;
        let r_ntt = r.ntt();
        let u = a_ntt.mul_vec(&r_ntt).intt().add(&e1);
        let scaled = Vec2::new([
            m.pair().poly(0).mul_scalar(SCALE),
            m.pair().poly(1).mul_scalar(SCALE),
        ]);
        let v = t_ntt.mul_vec(&r_ntt).intt().add(&e2).add(&scaled);
        Ok(Ciphertext { u, v })
    }
}

impl Default for Ciphertext {
    fn default() -> Self {
        Ciphertext::zero()
    }
}

/// Native reference of the DKG's output contract (DSR-7): (pk, s) with s
/// and T's E₀ drawn CBD(2) — bounded by `noise::KEY_SHORT_BOUND`, the
/// exact bound chunk 9's DKG consistency proofs must establish for the
/// live key. Consumed by the differential suites, conformance vectors,
/// and decrypt's native path.
pub fn derive_reference_keypair(entropy: &[u8; 32]) -> Result<(PublicKey, Vec8), SealError> {
    let a_seed = ASeed::from_bytes(*entropy);
    let a = expand_matrix(&a_seed)?;
    // Role-tagged framing inside SEAL_NOISE keeps the reference stream
    // disjoint from wallet triplet streams at the same domain.
    let tag: &[u8] = b"nerv.seal.refkey";
    let mut xof = Xof::framed(&SEAL_NOISE, &[tag, &entropy[..]]);
    let s = sample_short_vec8(&mut xof)?;
    let base = a.mul_vec_transpose(&s);
    let row0 = base.add(&sample_short_vec8(&mut xof)?);
    let row1 = base.add(&sample_short_vec8(&mut xof)?);
    let t = Mat2x8::new([*row0.polys(), *row1.polys()]);
    Ok((PublicKey::new(a_seed, t), s))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::digitize::digitize;
    use crate::noise::{chunk_noise_variance, HALF_SCALE, HONEST_LEG_WORST_NOISE, KEY_SHORT_BOUND, VAR_LEG_REF};
    use crate::ring::Poly;
    use proptest::prelude::*;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn noise_seed(state: &mut u64) -> NoiseSeed {
        let mut b = [0u8; 32];
        for ch in b.chunks_mut(8) {
            ch.copy_from_slice(&splitmix64(state).to_le_bytes());
        }
        NoiseSeed::from_bytes(b)
    }

    fn coords(state: &mut u64) -> [u64; 64] {
        let mut c = [0u64; 64];
        for v in c.iter_mut() {
            *v = splitmix64(state);
        }
        c
    }

    fn scale_of(m: &Plaintext) -> Vec2 {
        Vec2::new([
            m.pair().poly(0).mul_scalar(SCALE),
            m.pair().poly(1).mul_scalar(SCALE),
        ])
    }

    // v^{(i)} − s·u − scale·m^{(i)}: the exact decryption residual.
    fn noise_of(key_s: &Vec8, ct: &Ciphertext, scaled_m: &Vec2) -> [Poly; 2] {
        let su = key_s.dot(ct.u());
        [
            ct.v().poly(0).sub(&su).sub(&scaled_m.poly(0)),
            ct.v().poly(1).sub(&su).sub(&scaled_m.poly(1)),
        ]
    }

    fn check_bounded(res: &[Poly; 2], bound: i64, ctx: &str) {
        for (i, p) in res.iter().enumerate() {
            for (j, v) in p.centerlift().iter().enumerate() {
                assert!(v.abs() <= bound, "{ctx}: i={i} j={j} noise={v} bound={bound}");
            }
        }
    }

    #[test]
    fn encrypt_is_deterministic_and_sensitive() {
        let (pk, _) = derive_reference_keypair(&[0x42; 32]).unwrap();
        let mut s = 0xE0C1u64;
        let m = digitize(&coords(&mut s));
        let seed = noise_seed(&mut s);

        let ct1 = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        let ct2 = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        assert_eq!(ct1.to_bytes(), ct2.to_bytes());

        let seed2 = noise_seed(&mut s);
        let ct3 = Ciphertext::encrypt(&pk, &seed2, &m).unwrap();
        assert_ne!(ct1.u(), ct3.u());
        assert_ne!(ct1.v(), ct3.v());

        let m2 = digitize(&coords(&mut s));
        let ct4 = Ciphertext::encrypt(&pk, &seed, &m2).unwrap();
        assert_ne!(ct1.v(), ct4.v());

        let (pk2, _) = derive_reference_keypair(&[0x43; 32]).unwrap();
        let ct5 = Ciphertext::encrypt(&pk2, &seed, &m).unwrap();
        assert_ne!(ct1.u(), ct5.u());
        assert_ne!(ct1.v(), ct5.v());
    }

    #[test]
    fn single_leg_residual_within_provable_bound() {
        let (pk, s) = derive_reference_keypair(&[0x77; 32]).unwrap();
        let mut st = 0xE0C2u64;
        let m = digitize(&coords(&mut st));
        let seed = noise_seed(&mut st);
        let ct = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        let res = noise_of(&s, &ct, &scale_of(&m));
        check_bounded(&res, HONEST_LEG_WORST_NOISE, "single leg");
    }

    #[test]
    fn add_laws_hold() {
        let (pk, _) = derive_reference_keypair(&[0x21; 32]).unwrap();
        let mut st = 0xE0C3u64;
        let m = digitize(&coords(&mut st));
        let a = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m).unwrap();
        let b = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m).unwrap();
        let c = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m).unwrap();
        let z = Ciphertext::zero();
        assert_eq!(a.add(&z), a);
        assert_eq!(z.add(&a), a);
        assert_eq!(a.add(&b), b.add(&a));
        assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
        assert_eq!(Ciphertext::default(), z);
    }

    #[test]
    fn aggregation_is_homomorphic() {
        // WP §6.3.2: ct₁ + ct₂ decrypts to scale·(m₁ + m₂) plus summed noise.
        let (pk, s) = derive_reference_keypair(&[0x55; 32]).unwrap();
        let mut st = 0xE0C4u64;
        let m1 = digitize(&coords(&mut st));
        let m2 = digitize(&coords(&mut st));
        let ct1 = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m1).unwrap();
        let ct2 = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m2).unwrap();
        let sum = ct1.add(&ct2);
        let scaled_sum = scale_of(&m1).add(&scale_of(&m2));
        let res = noise_of(&s, &sum, &scaled_sum);
        check_bounded(&res, 2 * HONEST_LEG_WORST_NOISE, "aggregate of 2");
    }

    #[test]
    fn chunk_noise_matches_the_sigma_model() {
        // 128 honest legs of zero plaintext: the aggregate's decryption
        // residual IS the chunk noise. Empirically validates the exact
        // integer variance model (σ² = 128·4097) and the margin claim.
        let (pk, s) = derive_reference_keypair(&[0x66; 32]).unwrap();
        let a_ntt = pk.expand_a().unwrap().ntt();
        let t_ntt = pk.t().ntt();
        let zero = Plaintext::zero();
        let mut st = 0xE0C5u64;
        let mut agg = Ciphertext::zero();
        for _ in 0..128 {
            let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &noise_seed(&mut st), &zero)
                .unwrap();
            agg = agg.add(&ct);
        }
        let res = noise_of(&s, &agg, &scale_of(&zero));
        let mut sum_sq: u64 = 0;
        let mut max_sq: u64 = 0;
        for p in res.iter() {
            for v in p.centerlift().iter() {
                let sq = (v * v) as u64;
                sum_sq += sq;
                max_sq = max_sq.max(sq);
            }
        }
        // Σν² over the 512 component-coefficients vs 512·σ²_chunk (the
        // sample-mean of squares is unbiased for the marginal variance).
        let expected = 512 * 128 * crate::noise::VAR_LEG_REF;
        assert!(
            sum_sq > expected * 4 / 5 && sum_sq < expected * 6 / 5,
            "Σν² = {sum_sq}, expected ≈ {expected}"
        );
        // Max |ν| within 5σ, and strictly inside the decode margin.
        assert!(max_sq <= 25 * chunk_noise_variance(128), "max |ν|² = {max_sq}");
        assert!((max_sq as i64) < HALF_SCALE * HALF_SCALE);
    }

    #[test]
    fn wire_roundtrips_and_validates() {
        let (pk, _) = derive_reference_keypair(&[0x88; 32]).unwrap();
        let pkb = pk.to_bytes();
        assert_eq!(pkb.len(), 16_416);
        assert_eq!(PublicKey::from_bytes(&pkb).unwrap(), pk);
        assert!(matches!(
            PublicKey::from_bytes(&pkb[..16_415]),
            Err(SealError::BadLength { expected: 16_416, .. })
        ));
        let mut bad = pkb;
        bad[32..36].copy_from_slice(&u32::MAX.to_le_bytes()); // first T coefficient
        assert!(matches!(
            PublicKey::from_bytes(&bad),
            Err(SealError::UnreducedCoefficient { value }) if value == u32::MAX as u64
        ));

        let mut st = 0xE0C6u64;
        let m = digitize(&coords(&mut st));
        let ct = Ciphertext::encrypt(&pk, &noise_seed(&mut st), &m).unwrap();
        let ctb = ct.to_bytes();
        assert_eq!(ctb.len(), 10_240);
        assert_eq!(Ciphertext::from_bytes(&ctb).unwrap(), ct);
        assert!(matches!(
            Ciphertext::from_bytes(&ctb[..10_239]),
            Err(SealError::BadLength { expected: 10_240, .. })
        ));
        let mut badct = ctb;
        badct[4..8].copy_from_slice(&u32::MAX.to_le_bytes()); // first u coefficient
        assert!(matches!(
            Ciphertext::from_bytes(&badct),
            Err(SealError::UnreducedCoefficient { .. })
        ));
    }

    #[test]
    fn cached_path_matches_expansion() {
        let (pk, _) = derive_reference_keypair(&[0x99; 32]).unwrap();
        let mut st = 0xE0C7u64;
        let m = digitize(&coords(&mut st));
        let seed = noise_seed(&mut st);
        let direct = Ciphertext::encrypt(&pk, &seed, &m).unwrap();
        let a_ntt = pk.expand_a().unwrap().ntt();
        let t_ntt = pk.t().ntt();
        let cached = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &seed, &m).unwrap();
        assert_eq!(direct.to_bytes(), cached.to_bytes());
    }

    #[test]
    fn reference_keypair_honors_the_dkg_contract() {
        let (pk, s) = derive_reference_keypair(&[0xAA; 32]).unwrap();
        for p in s.polys().iter() {
            for v in p.centerlift().iter() {
                assert!(v.unsigned_abs() <= KEY_SHORT_BOUND);
            }
        }
        // T − s·A is bounded too: the E₀ rows the budget assumes.
        let a = pk.expand_a().unwrap();
        let base = a.mul_vec_transpose(&s);
        for i in 0..2 {
            let row = Vec8::new(*pk.t().row(i));
            let e0 = row.sub(&base);
            for p in e0.polys().iter() {
                for v in p.centerlift().iter() {
                    assert!(v.unsigned_abs() <= KEY_SHORT_BOUND);
                }
            }
        }
        let (pk2, s2) = derive_reference_keypair(&[0xAB; 32]).unwrap();
        assert_ne!(pk, pk2);
        assert_ne!(s, s2);
        let (pk3, s3) = derive_reference_keypair(&[0xAA; 32]).unwrap();
        assert_eq!((pk, s), (pk3, s3));
    }

    proptest! {
        #[test]
        fn prop_encrypt_cached_deterministic(seed in any::<u64>()) {
            let (pk, _) = derive_reference_keypair(&[0x11; 32]).unwrap();
            let a_ntt = pk.expand_a().unwrap().ntt();
            let t_ntt = pk.t().ntt();
            let mut st = seed ^ 0xE0C1;
            let m = digitize(&coords(&mut st));
            let ns = noise_seed(&mut st);
            let c1 = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &ns, &m).unwrap();
            let c2 = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &ns, &m).unwrap();
            prop_assert_eq!(c1, c2);
        }

        #[test]
        fn prop_aggregate_residual_bounded(legs in 1u64..=16, seed in any::<u64>()) {
            let (pk, s) = derive_reference_keypair(&[0x22; 32]).unwrap();
            let a_ntt = pk.expand_a().unwrap().ntt();
            let t_ntt = pk.t().ntt();
            let mut st = seed ^ 0xE0C2;
            let mut agg = Ciphertext::zero();
            let mut scaled_sum = Vec2::zero();
            for _ in 0..legs {
                let m = digitize(&coords(&mut st));
                let ct = Ciphertext::encrypt_cached(&a_ntt, &t_ntt, &noise_seed(&mut st), &m)
                    .unwrap();
                agg = agg.add(&ct);
                scaled_sum = scaled_sum.add(&scale_of(&m));
            }
            let res = noise_of(&s, &agg, &scaled_sum);
            for p in res.iter() {
                for v in p.centerlift().iter() {
                    prop_assert!(v.abs() <= legs as i64 * HONEST_LEG_WORST_NOISE);
                }
            }
        }
    }
}
