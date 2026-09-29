//! FRI — the low-degree test (WP §5.3, §5.7; the ethSTARK-class public
//! specification, native implementation per register 57). The engine's
//! polynomial commitment: a codeword over a coset domain D₀ (size M) is
//! folded with per-round challenges α (drawn after each layer's root is
//! absorbed) until a final layer of size 2^f, which is sent as
//! COEFFICIENTS capped at
//!     cap = 2^(f − log_blowup)   (floor 1).
//! The cap is what the proof proves: with the driver's contract that the
//! input codeword is the evaluation of a polynomial of degree < M/2^b,
//! the honest folded polynomial has at most cap coefficients, and a
//! random codeword's folds cannot agree with any capped polynomial's
//! evaluations at the random query positions. Layers are committed as
//! BLAKE3 row-Merkle trees (leaf = both ExtF components at one domain
//! position); each query authenticates the top-bit-flip value pair at
//! every layer via one shared auth path (`merkle::verify_pair_opening`)
//! and checks the fold
//!     g(x²) = (f(x) + f(−x))/2 + α·(f(x) − f(−x))/(2x)
//! against the next layer's opened value.
//!
//! `max_log_arity` is vestigial (folding is strictly binary); grinding is
//! `commit_pow_bits + query_pow_bits` bits applied between the final
//! absorption and the query draws. All arithmetic is exact (DSR-11).

use crate::air::fs::FsTranscript;
use crate::security::FriShape;
use crate::stark::domain::TwoAdicDomain;
use crate::stark::ext_field::ExtF;
use crate::stark::fft;
use crate::stark::merkle::{verify_pair_opening, CodewordMatrix, MerkleTree};
use crate::stark::transcript_common::{
    check_grind, grind, sample_ext, sample_index, TAG_FRI_FOLD, TAG_FRI_FINAL, TAG_FRI_QUERY,
    TAG_POW,
};
use nerv_core::codec::Encode;
use nerv_core::field::{Goldilocks, GOLDILOCKS_PRIME};
use nerv_core::hash::Hash256;

const INV2: Goldilocks = Goldilocks::from_u64_reduce((GOLDILOCKS_PRIME + 1) / 2);

/// The final polynomial's coefficient cap: 2^(f−b), floor 1 (a constant —
/// the over-folded regime where the claim collapses below degree 1).
pub fn final_coeff_cap(log_final_poly_len: usize, log_blowup: usize) -> usize {
    if log_final_poly_len > log_blowup {
        1usize << (log_final_poly_len - log_blowup)
    } else {
        1
    }
}

/// One committed layer's query opening: the value pair (f(x), f(−x)) at
/// the left-half index and the shared auth path (length log size − 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairOpening {
    pub low: ExtF,
    pub high: ExtF,
    pub path: Vec<Hash256>,
}

/// One query's openings, one per committed layer (in fold order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriQuery {
    pub openings: Vec<PairOpening>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FriProof {
    /// Committed layer roots, sizes M .. 2^(f+1).
    pub roots: Vec<Hash256>,
    /// The final polynomial's coefficients (≤ cap; trailing zeros trimmed).
    pub final_coeffs: Vec<ExtF>,
    pub pow_nonce: u64,
    pub queries: Vec<FriQuery>,
    /// The drawn query positions — redrawn and cross-checked by the
    /// verifier; the composition layer consumes them to place its outer
    /// openings at the same indices.
    pub positions: Vec<u64>,

}

/// Fold one layer: `out[j] = E(x_j) + α·O(x_j)/x_j` for x_j = dom.point(j),
/// j < half. The inverse domain is itself geometric — a power walk, two
/// inversions total, no per-point inversion.
fn fold_layer(values: &[ExtF], dom: &TwoAdicDomain, alpha: ExtF) -> Vec<ExtF> {
    let size = dom.size();
    assert_eq!(values.len(), size, "layer length must equal the domain size");
    assert!(size >= 2, "cannot fold the trivial domain");
    let half = size / 2;
    let mut out = Vec::with_capacity(half);
    let mut xinv = dom.shift().inverse();
    let inv_gen = dom.gen().inverse();
    for j in 0..half {
        let e = (values[j] + values[j + half]).scale(INV2);
        let o = (values[j] - values[j + half]).scale(INV2);
        out.push(e + alpha * o.scale(xinv));
        xinv = xinv * inv_gen;
    }
    out
}

/// The verifier's single-pair fold — must agree with `fold_layer`
/// element-for-element (pinned by test).
fn fold_pair(low: ExtF, high: ExtF, x: Goldilocks, alpha: ExtF) -> ExtF {
    let e = (low + high).scale(INV2);
    let o = (low - high).scale(INV2);
    e + alpha * o.scale(x.inverse())
}

fn commit_layer(values: &[ExtF]) -> MerkleTree {
    let n = values.len();
    let mut c0 = Vec::with_capacity(n);
    let mut c1 = Vec::with_capacity(n);
    for v in values {
        let (a, b) = v.components();
        c0.push(a);
        c1.push(b);
    }
    MerkleTree::commit(&CodewordMatrix::from_columns(&[c0, c1]))
}

/// Both sides absorb the final coefficients identically: one part,
/// tag-prefixed, canonical component encoding.
fn absorb_final(t: &mut FsTranscript, coeffs: &[ExtF]) {
    let mut buf = Vec::with_capacity(TAG_FRI_FINAL.len() + 16 * coeffs.len());
    buf.extend_from_slice(TAG_FRI_FINAL);
    for c in coeffs {
        c.encode_into(&mut buf);
    }
    t.absorb_bytes(&buf);
}

fn build_query(
    q: u64,
    layers: &[Vec<ExtF>],
    trees: &[MerkleTree],
    domains: &[TwoAdicDomain],
) -> FriQuery {
    let mut openings = Vec::with_capacity(trees.len());
    for i in 0..trees.len() {
        let half = domains[i].size() / 2;
        let j = (q as usize) % half;
        openings.push(PairOpening {
            low: layers[i][j],
            high: layers[i][j + half],
            path: trees[i].auth_path_pair(j),
        });
    }
    FriQuery { openings }
}

/// Prove low-degreeness of `values` over `dom`. Driver contract: the
/// codeword is the evaluation over `dom` of a polynomial of degree <
/// `dom.size() / 2^log_blowup`. Total — a contract-violating input yields
/// a proof that fails verification (the debug assert makes driver bugs
/// loud in development).
#[allow(clippy::too_many_lines)]
pub fn prove_fri(
    values: &[ExtF],
    dom: &TwoAdicDomain,
    shape: &FriShape,
    t: &mut FsTranscript,
) -> FriProof {
    let m = dom.log_size();
    let f = shape.log_final_poly_len;
    let b = shape.log_blowup;
    assert!(b >= 1, "rate-1 FRI is vacuous: log_blowup must be at least 1");
    assert!(m > f, "need at least one fold: log_size {m} > log_final {f}");
    assert_eq!(values.len(), dom.size(), "codeword length must equal the domain size");
    let r = m - f;
    let cap = final_coeff_cap(f, b);
    let pow_bits = shape.commit_pow_bits + shape.query_pow_bits;

    let mut layers: Vec<Vec<ExtF>> = Vec::with_capacity(r + 1);
    let mut domains: Vec<TwoAdicDomain> = Vec::with_capacity(r + 1);
    let mut trees: Vec<MerkleTree> = Vec::with_capacity(r);
    let mut roots: Vec<Hash256> = Vec::with_capacity(r);
    layers.push(values.to_vec());
    domains.push(*dom);
    let tree0 = commit_layer(&layers[0]);
    t.absorb_hash(&tree0.root());
    roots.push(tree0.root());
    trees.push(tree0);

    for i in 0..r {
        let alpha = sample_ext(t, TAG_FRI_FOLD);
        layers.push(fold_layer(&layers[i], &domains[i], alpha));
        domains.push(domains[i].halve());
        if i + 1 < r {
            let tree = commit_layer(&layers[i + 1]);
            t.absorb_hash(&tree.root());
            roots.push(tree.root());
            trees.push(tree);
        }
    }

    let final_dom = domains[r];
    let mut coeffs = fft::interpolate(&layers[r], &final_dom);
    while coeffs.last() == Some(&ExtF::ZERO) {
        coeffs.pop();
    }
    debug_assert!(
        coeffs.len() <= cap,
        "codeword exceeds the claimed degree bound — driver contract violated"
    );
    coeffs.truncate(cap);
    absorb_final(t, &coeffs);

    let nonce = grind(t, TAG_POW, pow_bits);

let mut queries = Vec::with_capacity(shape.num_queries);
    let mut positions = Vec::with_capacity(shape.num_queries);
    for _ in 0..shape.num_queries {
        let q = sample_index(t, TAG_FRI_QUERY, m);
        positions.push(q);
        queries.push(build_query(q, &layers, &trees, &domains));
    }


    FriProof { roots, final_coeffs: coeffs, pow_nonce: nonce, queries, positions }

}

/// Verify a FRI proof. Never panics on adversarial proofs: every proof
/// field is length-checked before use, and all indices are derived from
/// the redrawn query positions, not from the proof.
pub fn verify_fri(
    proof: &FriProof,
    dom: &TwoAdicDomain,
    shape: &FriShape,
    t: &mut FsTranscript,
) -> bool {
    let m = dom.log_size();
    let f = shape.log_final_poly_len;
    let b = shape.log_blowup;
    assert!(b >= 1, "rate-1 FRI is vacuous: log_blowup must be at least 1");
    assert!(m > f, "need at least one fold: log_size {m} > log_final {f}");
    let r = m - f;
    let cap = final_coeff_cap(f, b);
    let pow_bits = shape.commit_pow_bits + shape.query_pow_bits;

  if proof.roots.len() != r
        || proof.queries.len() != shape.num_queries
        || proof.positions.len() != shape.num_queries
    {
        return false;
    }
    if proof.final_coeffs.len() > cap {
        return false;
    }


    let mut final_dom = *dom;
    for _ in 0..r {
        final_dom = final_dom.halve();
    }
    let final_evals = fft::evaluate(&proof.final_coeffs, &final_dom);

    t.absorb_hash(&proof.roots[0]);
    let mut alphas = Vec::with_capacity(r);
    for i in 0..r {
        alphas.push(sample_ext(t, TAG_FRI_FOLD));
        if i + 1 < r {
            t.absorb_hash(&proof.roots[i + 1]);
        }
    }
    absorb_final(t, &proof.final_coeffs);
    if !check_grind(t, TAG_POW, pow_bits, proof.pow_nonce) {
        return false;
    }

  for (k, qp) in proof.queries.iter().enumerate() {
        let q = sample_index(t, TAG_FRI_QUERY, m);
        if proof.positions[k] != q {
            return false;
        }
        if qp.openings.len() != r {
            return false;
        }

        let mut dom_i = *dom;
        for i in 0..r {
            let half = dom_i.size() / 2;
            let j = (q as usize) % half;
            let op = &qp.openings[i];
            if op.path.len() + 1 != dom_i.log_size() {
                return false;
            }
            let (c0l, c1l) = op.low.components();
            let (c0h, c1h) = op.high.components();
            if !verify_pair_opening(&proof.roots[i], j, &[c0l, c1l], &[c0h, c1h], &op.path) {
                return false;
            }
            let expected = fold_pair(op.low, op.high, dom_i.point(j), alphas[i]);
            let target = if i + 1 < r {
                let next = &qp.openings[i + 1];
                if j < half / 2 {
                    next.low
                } else {
                    next.high
                }
            } else {
                final_evals[j]
            };
            if expected != target {
                return false;
            }
            dom_i = dom_i.halve();
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;

    fn gf(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }

    fn ef(rng: &mut SplitMix64) -> ExtF {
        ExtF::new(gf(rng), gf(rng))
    }

    fn seeded(seed: u64) -> FsTranscript {
        let mut t = FsTranscript::new();
        t.absorb_bytes(&seed.to_le_bytes());
        t
    }

    fn shape(f: usize, b: usize, q: usize, pow: usize) -> FriShape {
        FriShape {
            log_blowup: b,
            num_queries: q,
            log_final_poly_len: f,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: pow,
        }
    }

    fn low_degree_values(rng: &mut SplitMix64, dom: &TwoAdicDomain, b: usize) -> Vec<ExtF> {
        let n = dom.size() >> b;
        let coeffs: Vec<ExtF> = (0..n).map(|_| ef(rng)).collect();
        fft::evaluate(&coeffs, dom)
    }

    fn horner(coeffs: &[ExtF], x: Goldilocks) -> ExtF {
        let x = ExtF::from_base(x);
        coeffs.iter().rev().fold(ExtF::ZERO, |acc, c| acc * x + *c)
    }

    #[test]
    fn final_coeff_cap_table() {
        assert_eq!(final_coeff_cap(4, 2), 4);
        assert_eq!(final_coeff_cap(4, 3), 2);
        assert_eq!(final_coeff_cap(4, 4), 1);
        assert_eq!(final_coeff_cap(3, 4), 1);
        assert_eq!(final_coeff_cap(0, 2), 1);
        assert_eq!(final_coeff_cap(6, 1), 32);
    }

    #[test]
    fn fold_layer_matches_polynomial_semantics() {
        let mut rng = SplitMix64::new(0xF01);
        for (m, b) in [(6usize, 2usize), (5, 1), (4, 3)] {
            let dom = TwoAdicDomain::standard_coset(m);
            let n = dom.size() >> b;
            let coeffs: Vec<ExtF> = (0..n).map(|_| ef(&mut rng)).collect();
            let vals = fft::evaluate(&coeffs, &dom);
            let alpha = ef(&mut rng);
            let folded = fold_layer(&vals, &dom, alpha);
            let half = dom.size() / 2;

            // Reference 1: Horner at x and −x.
            for j in 0..half {
                let x = dom.point(j);
                let px = horner(&coeffs, x);
                let pnx = horner(&coeffs, -x);
                let want =
                    (px + pnx).scale(INV2) + alpha * (px - pnx).scale(INV2).scale(x.inverse());
                assert_eq!(folded[j], want, "m={m} b={b} j={j}");
            }

            // Reference 2: the folded polynomial E' + α·O' evaluated over the
            // folded domain.
            let mut ep = Vec::new();
            let mut op = Vec::new();
            for (i, c) in coeffs.iter().enumerate() {
                if i % 2 == 0 {
                    ep.push(*c);
                } else {
                    op.push(*c);
                }
            }
            let len = ep.len().max(op.len());
            ep.resize(len, ExtF::ZERO);
            op.resize(len, ExtF::ZERO);
            let g: Vec<ExtF> = ep.iter().zip(&op).map(|(e, o)| *e + alpha * *o).collect();
            assert_eq!(folded, fft::evaluate(&g, &dom.halve()), "m={m} b={b}");
        }
    }

    #[test]
    fn fold_pair_matches_fold_layer() {
        let mut rng = SplitMix64::new(0xF02);
        for m in 2..=6usize {
            let dom = TwoAdicDomain::standard_coset(m);
            let vals: Vec<ExtF> = (0..dom.size()).map(|_| ef(&mut rng)).collect();
            let alpha = ef(&mut rng);
            let folded = fold_layer(&vals, &dom, alpha);
            for j in 0..dom.size() / 2 {
                assert_eq!(
                    folded[j],
                    fold_pair(vals[j], vals[j + dom.size() / 2], dom.point(j), alpha),
                    "m={m} j={j}"
                );
            }
        }
    }

    #[test]
    fn honest_prove_verify_roundtrip() {
        let configs = [
            (6usize, 2usize, 3usize, 20usize, 0usize),
            (8, 4, 3, 24, 0),
            (5, 0, 2, 16, 0),
            (4, 1, 1, 12, 8),
            (2, 1, 1, 4, 0),
            (3, 2, 1, 8, 0),
        ];
        for (m, f, b, q, pow) in configs {
            let dom = TwoAdicDomain::standard_coset(m);
            let mut rng = SplitMix64::new(0x900 + m as u64);
            let values = low_degree_values(&mut rng, &dom, b);
            let sh = shape(f, b, q, pow);

            let mut tp = seeded(0x577);
            let proof = prove_fri(&values, &dom, &sh, &mut tp);
            let mut tv = seeded(0x577);
            assert!(verify_fri(&proof, &dom, &sh, &mut tv), "m={m} f={f} b={b} q={q} pow={pow}");

            // Determinism: an identically-seeded transcript reproduces the proof.
            let mut tp2 = seeded(0x577);
            assert_eq!(prove_fri(&values, &dom, &sh, &mut tp2), proof);

            // Transcript binding: a different prior state rejects.
            let mut other = seeded(0x578);
            assert!(!verify_fri(&proof, &dom, &sh, &mut other));

            // Query count and final-length shape mismatches reject.
            let sh_q = shape(f, b, q + 1, pow);
            let mut tq = seeded(0x577);
            assert!(!verify_fri(&proof, &dom, &sh_q, &mut tq));
            let sh_f = shape(f + 1, b, q, pow);
            let mut tf = seeded(0x577);
            assert!(!verify_fri(&proof, &dom, &sh_f, &mut tf));
        }
    }

    #[test]
    fn zero_codeword_roundtrip() {
        let dom = TwoAdicDomain::standard_coset(5);
        let values = vec![ExtF::ZERO; dom.size()];
        let sh = shape(2, 2, 8, 0);
        let mut tp = seeded(0x2E0);
        let proof = prove_fri(&values, &dom, &sh, &mut tp);
        assert!(proof.final_coeffs.is_empty(), "the zero polynomial has no coefficients");
        let mut tv = seeded(0x2E0);
        assert!(verify_fri(&proof, &dom, &sh, &mut tv));
    }

    #[test]
    fn rejects_adversarial_random_codeword() {
        // A malicious prover honestly folds a NON-low-degree codeword and
        // sends the best coefficient prefix within the cap. The fold chain
        // is internally consistent — the cap and the last fold's check
        // against the capped polynomial's evaluations are what reject it.
        let m = 6usize;
        let f = 4usize;
        let b = 2usize;
        let qn = 64usize;
        let dom = TwoAdicDomain::standard_coset(m);
        let sh = shape(f, b, qn, 0);
        let cap = final_coeff_cap(f, b);
        let r = m - f;

        let mut rng = SplitMix64::new(0xBAAD);
        let values: Vec<ExtF> = (0..dom.size()).map(|_| ef(&mut rng)).collect();

        // Adversarial pipeline (prove_fri's flow without the driver contract).
        let mut t = seeded(0x11);
        let mut layers = vec![values];
        let mut domains = vec![dom];
        let mut trees = Vec::new();
        let mut roots = Vec::new();
        let tree0 = commit_layer(&layers[0]);
        t.absorb_hash(&tree0.root());
        roots.push(tree0.root());
        trees.push(tree0);
        for i in 0..r {
            let alpha = sample_ext(&mut t, TAG_FRI_FOLD);
            layers.push(fold_layer(&layers[i], &domains[i], alpha));
            domains.push(domains[i].halve());
            if i + 1 < r {
                let tree = commit_layer(&layers[i + 1]);
                t.absorb_hash(&tree.root());
                roots.push(tree.root());
                trees.push(tree);
            }
        }
        let mut coeffs = fft::interpolate(&layers[r], &domains[r]);
        coeffs.truncate(cap);
        absorb_final(&mut t, &coeffs);
        let nonce = grind(&mut t, TAG_POW, 0);
     let mut queries = Vec::with_capacity(qn);
        let mut positions = Vec::with_capacity(qn);
        for _ in 0..qn {
            let qpos = sample_index(&mut t, TAG_FRI_QUERY, m);
            positions.push(qpos);
            queries.push(build_query(qpos, &layers, &trees, &domains));
        }
        let proof = FriProof { roots, final_coeffs: coeffs, pow_nonce: nonce, queries, positions };


        let mut tv = seeded(0x11);
        assert!(!verify_fri(&proof, &dom, &sh, &mut tv));
        let mut tv2 = seeded(0x11);
        assert!(!verify_fri(&proof, &dom, &sh, &mut tv2), "rejection is deterministic");
    }

    #[test]
    fn tamper_battery() {
        let dom = TwoAdicDomain::standard_coset(6);
        let mut rng = SplitMix64::new(0x7A1);
        let values = low_degree_values(&mut rng, &dom, 3);
        let sh = shape(2, 3, 8, 0);
        let mut tp = seeded(0x577);
        let proof = prove_fri(&values, &dom, &sh, &mut tp);

        let check = |p: &FriProof| {
            let mut t = seeded(0x577);
            verify_fri(p, &dom, &sh, &mut t)
        };
        assert!(check(&proof));

        for k in 0..proof.roots.len() {
            let mut p = proof.clone();
            let mut b = *p.roots[k].as_bytes();
            b[0] ^= 1;
            p.roots[k] = Hash256::from_bytes(b);
            assert!(!check(&p), "root {k}");
        }

        {
            let mut p = proof.clone();
            p.roots.pop();
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.roots.push(p.roots[0]);
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.roots.swap(0, 1.min(p.roots.len() - 1));
            assert!(!check(&p), "swapped roots change the transcript order");
        }

        {
            let mut p = proof.clone();
            p.final_coeffs[0] = p.final_coeffs[0] + ExtF::ONE;
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.final_coeffs.pop();
            assert!(!check(&p), "dropping a significant coefficient changes the evaluations");
        }
        {
            let mut p = proof.clone();
            p.final_coeffs.push(ExtF::ONE);
            assert!(!check(&p), "appending a coefficient violates the cap or the evaluations");
        }

        {
            let mut p = proof.clone();
            p.queries.pop();
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.queries.push(p.queries[0].clone());
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.queries[0].openings.pop();
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.queries[0].openings[0].low = p.queries[0].openings[0].low + ExtF::ONE;
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            p.queries[0].openings[0].high = p.queries[0].openings[0].high + ExtF::ONE;
            assert!(!check(&p));
        }
        {
            let mut p = proof.clone();
            if !p.queries[0].openings[0].path.is_empty() {
                let mut b = *p.queries[0].openings[0].path[0].as_bytes();
                b[31] ^= 1;
                p.queries[0].openings[0].path[0] = Hash256::from_bytes(b);
                assert!(!check(&p));
            }
        }
        {
            let mut p = proof.clone();
            if !p.queries[0].openings[0].path.is_empty() {
                p.queries[0].openings[0].path.pop();
                assert!(!check(&p));
            }
        }
    }

    #[test]
    fn forged_grinding_nonce_rejected() {
        let dom = TwoAdicDomain::standard_coset(6);
        let mut rng = SplitMix64::new(0x7A2);
        let values = low_degree_values(&mut rng, &dom, 3);
        let sh = shape(2, 3, 8, 16);
        let mut tp = seeded(0x577);
        let mut proof = prove_fri(&values, &dom, &sh, &mut tp);
        proof.pow_nonce += 1;
        let mut tv = seeded(0x577);
        assert!(!verify_fri(&proof, &dom, &sh, &mut tv));
    }

    #[test]
    #[should_panic(expected = "vacuous")]
    fn rejects_rate_one_config() {
        let dom = TwoAdicDomain::standard_coset(4);
        let sh = shape(1, 0, 4, 0);
        let mut t = seeded(1);
        let _ = prove_fri(&vec![ExtF::ZERO; 16], &dom, &sh, &mut t);
    }

    #[test]
    #[should_panic(expected = "at least one fold")]
    fn rejects_no_fold_config() {
        let dom = TwoAdicDomain::standard_coset(4);
        let sh = shape(4, 2, 4, 0);
        let mut t = seeded(1);
        let _ = prove_fri(&vec![ExtF::ZERO; 16], &dom, &sh, &mut t);
    }

    #[test]
    #[should_panic(expected = "domain size")]
    fn rejects_length_mismatch() {
        let dom = TwoAdicDomain::standard_coset(4);
        let sh = shape(1, 2, 4, 0);
        let mut t = seeded(1);
        let _ = prove_fri(&vec![ExtF::ZERO; 8], &dom, &sh, &mut t);
    }
}
