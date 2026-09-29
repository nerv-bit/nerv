//! Cross-module integration for `ring` (chunk 7, part 1): the full §6.3.1
//! arithmetic pipeline (key structure → encryption → exact decryption),
//! the §6.3.2 homomorphic-aggregation property, chunk-scale accumulation
//! within the closed budget, and the WP §6.3.8 genesis wire sizes.

#![allow(clippy::unwrap_used)]

use nerv_seal::ring::{Mat2x8, Mat8x8, Mat8x8Ntt, Poly, Vec2, Vec8, N, Q};

use nerv_seal::noise::{HONEST_LEG_WORST_NOISE as LEG_NOISE_BOUND, SCALE};

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform_poly(s: &mut u64) -> Poly {
    let mut a = [0u64; N];
    for c in a.iter_mut() {
        *c = splitmix64(s) % Q;
    }
    Poly::new(a)
}

/// CBD(η = 2) support: coefficients in [−2, 2] (the distribution's exact
/// range; the §6.3.7 3σ bound is satisfied with margin).
fn short_poly(s: &mut u64) -> Poly {
    let mut vals = [0i64; N];
    for v in vals.iter_mut() {
        *v = (splitmix64(s) % 5) as i64 - 2;
    }
    Poly::from_centered(&vals)
}

fn short_vec(s: &mut u64) -> Vec8 {
    let mut v = [Poly::zero(); 8];
    for p in v.iter_mut() {
        *p = short_poly(s);
    }
    Vec8::new(v)
}

/// Digit-valued plaintext polynomial: coefficients in [0, 1023].
fn digit_poly(s: &mut u64) -> Poly {
    let mut a = [0u64; N];
    for c in a.iter_mut() {
        *c = splitmix64(s) % 1024;
    }
    Poly::new(a)
}

struct KeyMaterial {
    a_ntt: Mat8x8Ntt,
    s: Vec8,
    t: Mat2x8,
}

fn key_material(seed: &mut u64) -> KeyMaterial {
    let mut rows = [[Poly::zero(); 8]; 8];
    for row in rows.iter_mut() {
        for p in row.iter_mut() {
            *p = uniform_poly(seed);
        }
    }
    let a = Mat8x8::new(rows);
    let secret = short_vec(seed);
    let base = a.mul_vec_transpose(&secret);
    let row0 = base.add(&short_vec(seed));
    let row1 = base.add(&short_vec(seed));
    let t = Mat2x8::new([*row0.polys(), *row1.polys()]);
    KeyMaterial { a_ntt: a.ntt(), s: secret, t }
}


struct Leg {
    u: Vec8,
    v: Vec2,
    m: Vec2,
}

fn encrypt_leg(k: &KeyMaterial, s: &mut u64) -> Leg {
    let r = short_vec(s);
    let r_ntt = r.ntt();
    let u = k.a_ntt.mul_vec(&r_ntt).intt().add(&short_vec(s)); // A·r + e₁
    let m = Vec2::new([digit_poly(s), digit_poly(s)]);
    let scaled_m = Vec2::new([m.poly(0).mul_scalar(SCALE), m.poly(1).mul_scalar(SCALE)]);
    let v = k.t.mul_vec(&r).add(&Vec2::new([short_poly(s), short_poly(s)])).add(&scaled_m);
    Leg { u, v, m }
}

/// m̂^{(i)} = v^{(i)} − s·u (the shared dot product serves both components).
fn decrypt(k: &KeyMaterial, u: &Vec8, v: &Vec2) -> [Poly; 2] {
    let su = k.s.dot(u);
    [v.poly(0).sub(&su), v.poly(1).sub(&su)]
}

fn check_noise(mhat: &Poly, m: &Poly, bound: i64, ctx: &str) {
    let cl = mhat.centerlift();
    let mc = m.coefficients();
    for j in 0..N {
        let diff = cl[j] - (SCALE as i64) * (mc[j] as i64);
        assert!(diff.abs() <= bound, "{ctx}: j={j} noise={diff} bound={bound}");
    }
}

#[test]
fn section_631_pipeline_decrypts_exactly() {
    let mut s = 0x6311_0000u64;
    let k = key_material(&mut s);
    let leg = encrypt_leg(&k, &mut s);
    let mhat = decrypt(&k, &leg.u, &leg.v);

    for i in 0..2 {
        check_noise(&mhat[i], leg.m.poly(i), LEG_NOISE_BOUND, "single leg");
        // Provable decode: |ν| ≤ 2050 < SCALE/2 = 4096, so rounding
        // recovers the digit exactly — deterministically, not statistically.
        let cl = mhat[i].centerlift();
        let mc = leg.m.poly(i).coefficients();
        for j in 0..N {
            let decoded = (cl[j] + (SCALE as i64) / 2).div_euclid(SCALE as i64);
            assert_eq!(decoded, mc[j] as i64, "decode i={i} j={j}");
        }
    }
}

#[test]
fn homomorphic_aggregation_property() {
    // WP §6.3.2: ct_B = ct_1 + ct_2 decrypts to scale·(m_1 + m_2) plus the
    // summed noise — the additive homomorphism the whole channel rests on.
    let mut s = 0x6312_0000u64;
    let k = key_material(&mut s);
    let leg1 = encrypt_leg(&k, &mut s);
    let leg2 = encrypt_leg(&k, &mut s);

    let u_b = leg1.u.add(&leg2.u);
    let v_b = leg1.v.add(&leg2.v);
    let mhat = decrypt(&k, &u_b, &v_b);

    let m_sum = Vec2::new([
        leg1.m.poly(0).add(leg2.m.poly(0)),
        leg1.m.poly(1).add(leg2.m.poly(1)),
    ]);
    for i in 0..2 {
        // Provable bound doubles; decode at this size is statistical
        // (observed noise ≈ 2·σ_leg ≈ 45), pinned by the fixed seed.
        check_noise(&mhat[i], m_sum.poly(i), 2 * LEG_NOISE_BOUND, "aggregate of 2");
        let cl = mhat[i].centerlift();
        let mc = m_sum.poly(i).coefficients();
        for j in 0..N {
            let decoded = (cl[j] + (SCALE as i64) / 2).div_euclid(SCALE as i64);
            assert_eq!(decoded, mc[j] as i64, "aggregate decode i={i} j={j}");
        }
    }
}

#[test]
fn aggregate_chunk_of_128_stays_in_budget() {
    // The genesis chunk size (erratum 19). Per-component noise is bounded
    // by 128·LEG_NOISE_BOUND (provable, worst case); the plaintext side is
    // the adversarial digit-sum budget: SCALE·(128·1023) + noise < Q/2 —
    // the same inequality lib.rs enforces at compile time, here observed
    // on real data.
    let mut s = 0x6313_0000u64;
    let k = key_material(&mut s);

    let mut u_b = Vec8::zero();
    let mut v_b = Vec2::zero();
    let mut m_sum = [Vec2::zero(), Vec2::zero()][0];
    let mut m_sum = Vec2::zero();
    for _ in 0..128 {
        let leg = encrypt_leg(&k, &mut s);
        u_b = u_b.add(&leg.u);
        v_b = v_b.add(&leg.v);
        m_sum = m_sum.add(&leg.m);
    }

    let mhat = decrypt(&k, &u_b, &v_b);
    for i in 0..2 {
        check_noise(&mhat[i], m_sum.poly(i), 128 * LEG_NOISE_BOUND, "chunk of 128");
    }

    // The closed budget, observed: worst-case digit sums plus worst-case
    // noise stay inside the centered range.
    let worst_plaintext = SCALE * (128 255);
    let worst_noise = 128 * LEG_NOISE_BOUND as u64;
    assert!(worst_plaintext + worst_noise < Q / 2);

    // Sum-of-digits stays within the slot headroom q/scale.
    let headroom = Q / SCALE;
    for i in 0..2 {
        for &d in m_sum.poly(i).coefficients().iter() {
            assert!(d < headroom);
        }
    }
}

#[test]
fn wire_sizes_match_wp_genesis_budget() {
    assert_eq!(Poly::WIRE_SIZE, 1024);
    assert_eq!(Vec8::WIRE_SIZE, 8192); // u dominates (§6.3.8)
    assert_eq!(Vec2::WIRE_SIZE, 2048); // v
    assert_eq!(Vec8::WIRE_SIZE + Vec2::WIRE_SIZE, 10_240); // "≈ 8–10 KB"

    let mut s = 0x6314_0000u64;
    let k = key_material(&mut s);
    let leg = encrypt_leg(&k, &mut s);
    assert_eq!(leg.u.to_bytes().len(), 8192);
    assert_eq!(leg.v.to_bytes().len(), 2048);
    assert_eq!(Vec8::from_bytes(&leg.u.to_bytes()).unwrap(), leg.u);
    assert_eq!(Vec2::from_bytes(&leg.v.to_bytes()).unwrap(), leg.v);
}

