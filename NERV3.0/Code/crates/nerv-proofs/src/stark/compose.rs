//! The DEEP-ALI composition (WP §5.3; ethSTARK-class, native): everything
//! between a padded trace and a verifiable proof, shared verbatim by the
//! prover and verifier drivers.
//!
//! PROTOCOL (transcript order, both sides identical):
//!   absorb statement shape → trace root → publics → α
//!   → quotient root → ζ → opened ζ-values → γ (DEEP challenge)
//!   → FRI over the DEEP combination (roots, folds, final, grinding,
//!     queries — all internal to `fri`) → outer + ζ checks (pure).
//!
//! STRUCTURE. One commitment domain D = standard_coset(log_n + b + log_R),
//! C = N·2^b·R, with R the quotient headroom: R·N ≥ deg(Q)+1, so the
//! quotient is a single ExtF codeword over D — no chunking, no per-chunk
//! domains. The prover commits the trace (rows = W Goldilocks) and the
//! quotient (rows = 2 Goldilocks); the preprocessed trace is NOT committed
//! — the verifier holds the authenticated table and evaluates it at ζ by
//! Lagrange (the VK-authenticity rule: whatever prep the verifier did not
//! authenticate changes F(ζ) and the proof fails).
//!
//! The DEEP combination: S(x) = Σ_i γ^i(T_i(x)−T_i(ζ)) +
//! Σ_i γ^{W+i}(T_i(gx)−T_i(gζ)) + γ^{2W}(Q(x)−Q(ζ)); h = S/(x−ζ) is a
//! polynomial of degree < N·R exactly when every opened value is honest
//! (S(ζ)=0), and FRI proves deg h < C/2^b. Each FRI query additionally
//! reveals the trace rows at x and gx and the quotient row at x (Merkle
//! openings against the committed roots); the verifier checks
//! h(x)·(x−ζ) = S(x) from the revealed values — binding the opened
//! ζ-values and the committed codewords to the FRI instance. The final
//! check evaluates the AIR's α-folded constraints at ζ from the opened
//! values, the verifier's prep, and the boundary-selector polynomials,
//! and requires F(ζ) = Q(ζ)·Z_H(ζ).
//!
//! SOUNDNESS TERMS (the security model's, exactly): α-fold n/|EF|; DEEP-ALI
//! d·N/|EF|; batched-opening bind w/|EF| with w = 2W+1 (the γ-random
//! linear-combination argument); FRI b·q/2 + grinding.
//!
//! CONTRACTS (panics are driver bugs, never adversarial): the AIR reads
//! only existing witness/public/prep indices; `publics` and the verifier's
//! prep table must cover the AIR's reads; every proof field is
//! length-checked before use, so verification is total on adversarial
//! proofs. Rayon is confined to pure, index-ordered passes — proofs are
//! byte-identical across machines and runs (DSR-11).


use rayon::prelude::*;
use crate::air::builder::Air;
use crate::air::fs::FsTranscript;
use crate::security::FriShape;
use crate::stark::collector::{
    quotient_coeff_bound, selector_vals_ext, PointCollector, SelectorVals, TrueDegree,
};
use crate::stark::domain::{two_adic_generator, TwoAdicDomain, TWO_ADICITY};
use crate::stark::ext_field::{batch_inverse, ExtF};
use crate::stark::fft;
use crate::stark::fri::{prove_fri, verify_fri, FriProof};
use crate::stark::merkle::{CodewordMatrix, MerkleTree, RowOpening};
use crate::stark::transcript_common::{
    absorb_instance, sample_ext, TAG_ALPHA, TAG_DEEP, TAG_ZETA,
};
use nerv_core::codec::Encode;
use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;


// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------


/// The composition's configuration: trace height, FRI shape, and the
/// quotient headroom they induce. `log_c` is the commitment domain's log
/// size: log_n + log_blowup + log_chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub log_n: usize,
    pub log_blowup: usize,
    pub log_chunks: usize,
    pub log_final: usize,
    pub log_c: usize,
    pub fri: FriShape,
    pub true_degree: TrueDegree,
}


impl Plan {
    pub fn new(
        log_n: usize,
        fri: &FriShape,
        true_degree: &TrueDegree,
    ) -> Result<Plan, ComposeError> {
        if log_n == 0 {
            return Err(ComposeError::TraceTooSmall);
        }
        if fri.log_blowup == 0 {
            return Err(ComposeError::VacuousFri);
        }
        if fri.num_queries == 0 {
            return Err(ComposeError::NoQueries);
        }
        let n = 1usize << log_n;
        let qb = quotient_coeff_bound(true_degree, log_n);
        let log_chunks = chunk_log(qb, n);
        let log_c = log_n + fri.log_blowup + log_chunks;
        if log_c > TWO_ADICITY {
            return Err(ComposeError::DomainOverflow { log_c });
        }
        if log_c <= fri.log_final_poly_len {
            return Err(ComposeError::NoFold {
                log_c,
                log_final: fri.log_final_poly_len,
            });
        }
        Ok(Plan {
            log_n,
            log_blowup: fri.log_blowup,
            log_chunks,
            log_final: fri.log_final_poly_len,
            log_c,
            fri: *fri,
            true_degree: *true_degree,
        })
    }


    pub const fn n(&self) -> usize {
        1usize << self.log_n
    }


    pub const fn c(&self) -> usize {
        1usize << self.log_c
    }
}


/// Smallest r with n·2^r ≥ qb (the quotient headroom's log).
fn chunk_log(qb: usize, n: usize) -> usize {
    let mut r = 0usize;
    while (n << r) < qb {
        r += 1;
    }
    r
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ComposeError {
    #[error("trace height must be at least 2 (log_n ≥ 1)")]
    TraceTooSmall,
    #[error("rate-1 FRI is vacuous: log_blowup must be at least 1")]
    VacuousFri,
    #[error("FRI needs at least one query")]
    NoQueries,
    #[error("commitment domain 2^{log_c} exceeds the field's two-adicity 2^32")]
    DomainOverflow { log_c: usize },
    #[error("FRI final layer 2^{log_final} leaves no fold from the domain 2^{log_c}")]
    NoFold { log_c: usize, log_final: usize },
    #[error("trace height {found} does not match the plan's 2^{log_n}")]
    TraceHeight { log_n: usize, found: usize },
    #[error("trace rows are ragged")]
    RaggedTrace,
    #[error("trace has zero columns")]
    ZeroWidth,
    #[error("preprocessed height {found} does not match the plan's 2^{log_n}")]
    PrepHeight { log_n: usize, found: usize },
    #[error("preprocessed rows are ragged")]
    RaggedPrep,
    #[error("prover assertion counts disagree: {over_d} over the domain vs {at_zeta} at ζ")]
    AssertionCount { over_d: usize, at_zeta: usize },
    #[error("ζ landed in the trace domain (probability ~2^-115)")]
    ZetaInTraceDomain,
    #[error("ζ landed in the commitment domain (probability ~2^-100)")]
    ZetaInCommitmentDomain,
}


// ---------------------------------------------------------------------------
// Shared evaluation helpers
// ---------------------------------------------------------------------------


fn batch_inverse_g(vals: &mut [Goldilocks]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let mut prefix = Vec::with_capacity(n);
    let mut acc = Goldilocks::ONE;
    for v in vals.iter() {
        debug_assert!(!v.is_zero(), "batch_inverse_g: zero input");
        acc = acc * *v;
        prefix.push(acc);
    }
    let mut inv = prefix[n - 1].inverse();
    for i in (1..n).rev() {
        let orig = vals[i];
        vals[i] = inv * prefix[i - 1];
        inv = inv * orig;
    }
    vals[0] = inv;
}


fn horner_ext(coeffs: &[Goldilocks], x: ExtF) -> ExtF {
    coeffs
        .iter()
        .rev()
        .fold(ExtF::ZERO, |acc, c| acc * x + ExtF::from_base(*c))
}


/// Z_H(x) = x^N − 1 over a domain disjoint from H: one power walk on the
/// coset structure (x_j^N = shift^N·(gen^N)^j).
fn zh_over_domain(log_n: usize, dom: &TwoAdicDomain) -> Vec<Goldilocks> {
    let n = 1u64 << log_n;
    let gen_n = dom.gen().pow(n);
    let mut cur = dom.shift().pow(n);
    let mut out = Vec::with_capacity(dom.size());
    for _ in 0..dom.size() {
        out.push(cur - Goldilocks::ONE);
        cur = cur * gen_n;
    }
    out
}


/// The boundary-selector polynomials' values at each point of a domain
/// disjoint from H: first = Z_H/(N(x−1)), last = Z_H/(N(x−h^{N−1})),
/// transition = x − h^{N−1}. Batch-inverted; cross-pinned against
/// `selector_vals_ext` in the tests.
pub struct SelectorArrays {
    pub first: Vec<Goldilocks>,
    pub last: Vec<Goldilocks>,
    pub transition: Vec<Goldilocks>,
}


pub fn selector_arrays(log_n: usize, dom: &TwoAdicDomain) -> SelectorArrays {
    let n = 1u64 << log_n;
    assert!(
        dom.shift().pow(n) != Goldilocks::ONE,
        "the domain's coset intersects the trace subgroup"
    );
    let h_last = two_adic_generator(log_n).inverse();
    let n_inv = Goldilocks::from_u64_reduce(n).inverse();
    let zh = zh_over_domain(log_n, dom);
    let pts: Vec<Goldilocks> = dom.points().collect();
    let mut d_first: Vec<Goldilocks> = pts.iter().map(|x| *x - Goldilocks::ONE).collect();
    let mut d_last: Vec<Goldilocks> = pts.iter().map(|x| *x - h_last).collect();
    batch_inverse_g(&mut d_first);
    batch_inverse_g(&mut d_last);
    let first = zh.iter().zip(&d_first).map(|(z, d)| *z * *d * n_inv).collect();
    let last = zh.iter().zip(&d_last).map(|(z, d)| *z * *d * n_inv).collect();
    let transition = pts.iter().map(|x| *x - h_last).collect();
    SelectorArrays { first, last, transition }
}


/// Lagrange coefficients of H at an out-of-domain ζ:
/// L_i(ζ) = Z_H(ζ)·h^i/(N·(ζ−h^i)). One batch inversion.
pub fn lagrange_weights(log_n: usize, zeta: ExtF) -> Vec<ExtF> {
    let n = 1usize << log_n;
    let h = two_adic_generator(log_n);
    let zh = zeta.pow(n as u64) - ExtF::ONE;
    let mut hs = Vec::with_capacity(n);
    let mut diffs = Vec::with_capacity(n);
    let mut hp = Goldilocks::ONE;
    for _ in 0..n {
        hs.push(hp);
        diffs.push(zeta - ExtF::from_base(hp));
        hp = hp * h;
    }
    batch_inverse(&mut diffs);
    let n_inv = ExtF::from_base(Goldilocks::from_u64_reduce(n as u64).inverse());
    hs.into_iter()
        .zip(diffs)
        .map(|(hi, d)| zh * d * n_inv * ExtF::from_base(hi))
        .collect()
}


/// Evaluate an N-row table's columns at ζ through the Lagrange weights:
/// col_j(ζ) = Σ_i L_i(ζ)·table[i][j]. Prover and verifier share this path.
pub fn eval_table_at_ext(weights: &[ExtF], table: &[Vec<Goldilocks>]) -> Vec<ExtF> {
    if table.is_empty() {
        return Vec::new();
    }
    assert_eq!(weights.len(), table.len(), "table height must match the weights");
    let width = table[0].len();
    let mut out = vec![ExtF::ZERO; width];
    for (w, row) in weights.iter().zip(table) {
        assert_eq!(row.len(), width, "ragged table");
        for (o, v) in out.iter_mut().zip(row) {
            *o = *o + w.scale(*v);
        }
    }
    out
}


/// ζ ∈ H ⟺ ζ^N = 1 (exact: μ_N ⊆ GF(p)).
pub fn zeta_in_trace_domain(log_n: usize, zeta: ExtF) -> bool {
    zeta.pow(1u64 << log_n) == ExtF::ONE
}


/// ζ ∈ D ⟺ (ζ/shift)^C = 1 (exact: μ_C ⊆ GF(p)).
pub fn zeta_in_domain(dom: &TwoAdicDomain, zeta: ExtF) -> bool {
    let shift_inv = ExtF::from_base(dom.shift()).inverse();
    (zeta * shift_inv).pow(dom.size() as u64) == ExtF::ONE
}


// ---------------------------------------------------------------------------
// Transcript helpers (both sides call the identical sequence)
// ---------------------------------------------------------------------------


fn absorb_publics(t: &mut FsTranscript, publics: &[Goldilocks]) {
    let words: Vec<u64> = publics.iter().map(|p| p.as_u64()).collect();
    absorb_instance(t, &words);
}


fn absorb_deep_values(t: &mut FsTranscript, tz: &[ExtF], tzn: &[ExtF], qz: ExtF) {
    let mut buf = Vec::with_capacity(4 + 16 * (tz.len() + tzn.len() + 1));
    buf.extend_from_slice(&(tz.len() as u32).to_le_bytes());
    for v in tz.iter().chain(tzn) {
        v.encode_into(&mut buf);
    }
    qz.encode_into(&mut buf);
    t.absorb_bytes(&buf);
}


fn shape_instance(plan: &Plan, width: usize, prep_width: usize, pub_count: usize) -> Vec<u64> {
    vec![
        plan.log_n as u64,
        plan.log_c as u64,
        plan.log_blowup as u64,
        plan.log_chunks as u64,
        plan.log_final as u64,
        plan.fri.num_queries as u64,
        plan.fri.max_log_arity as u64,
        plan.fri.commit_pow_bits as u64,
        plan.fri.query_pow_bits as u64,
        width as u64,
        prep_width as u64,
        pub_count as u64,
    ]
}


fn gamma_powers(gamma: ExtF, width: usize) -> Vec<ExtF> {
    let mut out = Vec::with_capacity(2 * width + 1);
    let mut g = ExtF::ONE;
    for _ in 0..2 * width + 1 {
        out.push(g);
        g = g * gamma;
    }
    out
}


// ---------------------------------------------------------------------------
// Proof object
// ---------------------------------------------------------------------------


/// One FRI query's outer openings: the trace rows at x and g·x and the
/// quotient row at x (the low half of the layer-0 pair).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OuterOpenings {
    pub trace_low: RowOpening,
    pub trace_next: RowOpening,
    pub quotient: RowOpening,
}


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComposedProof {
    pub trace_root: Hash256,
    pub quotient_root: Hash256,
    /// T_i(ζ) — challenge-field values (ζ ∈ EF).
    pub trace_zeta: Vec<ExtF>,
    /// T_i(g·ζ).
    pub trace_zeta_next: Vec<ExtF>,
    /// Q(ζ).
    pub quotient_zeta: ExtF,
    pub num_assertions: u64,
    pub fri: FriProof,
    pub outer: Vec<OuterOpenings>,
}


// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------


#[allow(clippy::too_many_lines)]
pub fn compose_prove<A>(
    plan: &Plan,
    air: &A,
    trace_rows: &[Vec<Goldilocks>],
    prep_rows: &[Vec<Goldilocks>],
    publics: &[Goldilocks],
    t: &mut FsTranscript,
) -> Result<ComposedProof, ComposeError>
where
    A: Sync,
    for<'a> A: Air<PointCollector<'a, Goldilocks>> + Air<PointCollector<'a, ExtF>>,
{
    let n = plan.n();
    if trace_rows.len() != n {
        return Err(ComposeError::TraceHeight { log_n: plan.log_n, found: trace_rows.len() });
    }
    let width = trace_rows[0].len();
    if width == 0 {
        return Err(ComposeError::ZeroWidth);
    }
    if trace_rows.iter().any(|r| r.len() != width) {
        return Err(ComposeError::RaggedTrace);
    }
    let prep_width = prep_rows.first().map_or(0, |r| r.len());
    if !prep_rows.is_empty() {
        if prep_rows.len() != n {
            return Err(ComposeError::PrepHeight { log_n: plan.log_n, found: prep_rows.len() });
        }
        if prep_rows.iter().any(|r| r.len() != prep_width) {
            return Err(ComposeError::RaggedPrep);
        }
    }


    let trace_dom = TwoAdicDomain::subgroup(plan.log_n);
    let dom = TwoAdicDomain::standard_coset(plan.log_c);
    let c_size = dom.size();
    let step = dom.shift_step_for(plan.log_n);


    // Trace coefficients, then the LDE over D (rayon over columns).
    let trace_coeffs: Vec<Vec<Goldilocks>> = (0..width)
        .into_par_iter()
        .map(|c| {
            let col: Vec<Goldilocks> = trace_rows.iter().map(|r| r[c]).collect();
            fft::interpolate(&col, &trace_dom)
        })
        .collect();
    let trace_cols: Vec<Vec<Goldilocks>> = trace_coeffs
        .par_iter()
        .map(|co| fft::evaluate(co, &dom))
        .collect();
    let trace_mat = CodewordMatrix::from_columns(&trace_cols);
    drop(trace_cols);


    let prep_mat = if prep_width == 0 {
        None
    } else {
        let cols: Vec<Vec<Goldilocks>> = (0..prep_width)
            .into_par_iter()
            .map(|c| {
                let col: Vec<Goldilocks> = prep_rows.iter().map(|r| r[c]).collect();
                let co = fft::interpolate(&col, &trace_dom);
                fft::evaluate(&co, &dom)
            })
            .collect();
        Some(CodewordMatrix::from_columns(&cols))
    };


    // Commit the trace; open the transcript.
    let trace_tree = MerkleTree::commit(&trace_mat);
    let trace_root = trace_tree.root();
    absorb_instance(t, &shape_instance(plan, width, prep_width, publics.len()));
    t.absorb_hash(&trace_root);
    absorb_publics(t, publics);
    let alpha = sample_ext(t, TAG_ALPHA);


    // α-folded constraints over D (rayon over points; pure per point).
    let sels = selector_arrays(plan.log_n, &dom);
    let empty_prep: [Goldilocks; 0] = [];
    let folds: Vec<ExtF> = (0..c_size)
        .into_par_iter()
        .map(|j| {
            let cur = trace_mat.row(j);
            let nxt = trace_mat.row((j + step) % c_size);
            let prep: &[Goldilocks] = match &prep_mat {
                Some(m) => m.row(j),
                None => &empty_prep,
            };
            let sv = SelectorVals {
                first: sels.first[j],
                last: sels.last[j],
                transition: sels.transition[j],
            };
            PointCollector::new(cur, nxt, prep, publics, sv, alpha).evaluate(air).0
        })
        .collect();
    let num_assertions = {
        let prep: &[Goldilocks] = match &prep_mat {
            Some(m) => m.row(0),
            None => &empty_prep,
        };
        let sv = SelectorVals {
            first: sels.first[0],
            last: sels.last[0],
            transition: sels.transition[0],
        };
        PointCollector::new(trace_mat.row(0), trace_mat.row(step), prep, publics, sv, alpha)
            .evaluate(air)
            .1
    };


    // Quotient over D, committed.
    let zh = zh_over_domain(plan.log_n, &dom);
    let mut zh_inv = zh.clone();
    batch_inverse_g(&mut zh_inv);
    let q_evals: Vec<ExtF> = folds
        .iter()
        .zip(&zh_inv)
        .map(|(f, zi)| *f * ExtF::from_base(*zi))
        .collect();
    let q_c0: Vec<Goldilocks> = q_evals.iter().map(|q| q.components().0).collect();
    let q_c1: Vec<Goldilocks> = q_evals.iter().map(|q| q.components().1).collect();
    let quot_mat = CodewordMatrix::from_columns(&[q_c0, q_c1]);
    let quot_tree = MerkleTree::commit(&quot_mat);
    let quot_root = quot_tree.root();
    t.absorb_hash(&quot_root);


    // ζ and the opened values.
    let zeta = sample_ext(t, TAG_ZETA);
    if zeta_in_trace_domain(plan.log_n, zeta) {
        return Err(ComposeError::ZetaInTraceDomain);
    }
    if zeta_in_domain(&dom, zeta) {
        return Err(ComposeError::ZetaInCommitmentDomain);
    }
    let g = two_adic_generator(plan.log_n);
    let gz = ExtF::from_base(g) * zeta;
    let trace_zeta: Vec<ExtF> = trace_coeffs.iter().map(|co| horner_ext(co, zeta)).collect();
    let trace_zeta_next: Vec<ExtF> = trace_coeffs.iter().map(|co| horner_ext(co, gz)).collect();


    let lw = lagrange_weights(plan.log_n, zeta);
    let prep_z = eval_table_at_ext(&lw, prep_rows);
    let sels_z = selector_vals_ext(plan.log_n, zeta);
    let pubs_z: Vec<ExtF> = publics.iter().map(|p| ExtF::from_base(*p)).collect();
    let (f_zeta, count_z) = PointCollector::new(
        &trace_zeta,
        &trace_zeta_next,
        &prep_z,
        &pubs_z,
        sels_z,
        alpha,
    )
    .evaluate(air);
    if count_z != num_assertions {
        return Err(ComposeError::AssertionCount { over_d: num_assertions, at_zeta: count_z });
    }
    let zh_z = zeta.pow(n as u64) - ExtF::ONE;
    let quotient_zeta = f_zeta * zh_z.inverse();


    absorb_deep_values(t, &trace_zeta, &trace_zeta_next, quotient_zeta);
    let gamma = sample_ext(t, TAG_DEEP);
    let powers = gamma_powers(gamma, width);


    // The DEEP combination over D: h = S/(x−ζ) with
    // S = A + V + γ^{2W}·Q − const, A and V evaluated by FFT.
    let mut a_coeffs: Vec<ExtF> = vec![ExtF::ZERO; n];
    for (i, co) in trace_coeffs.iter().enumerate() {
        let w = powers[i];
        for (a, c) in a_coeffs.iter_mut().zip(co) {
            *a = *a + w.scale(*c);
        }
    }
    let mut v_coeffs: Vec<ExtF> = vec![ExtF::ZERO; n];
    let mut gk = Goldilocks::ONE;
     for (k, vc) in v_coeffs.iter_mut().enumerate() {
        let mut acc = ExtF::ZERO;
        for (i, co) in trace_coeffs.iter().enumerate() {
            acc = acc + powers[width + i].scale(co[k]);
        }
        *vc = acc.scale(gk);
        gk = gk * g;
    }

    let mut const_term = ExtF::ZERO;
    for (w, v) in powers.iter().zip(&trace_zeta) {
        const_term = const_term + *w * *v;
    }
    for (w, v) in powers[width..2 * width].iter().zip(&trace_zeta_next) {
        const_term = const_term + *w * *v;
    }
    const_term = const_term + powers[2 * width] * quotient_zeta;


    let a_evals = fft::evaluate(&a_coeffs, &dom);
    let v_evals = fft::evaluate(&v_coeffs, &dom);
    let mut deep_inv: Vec<ExtF> =
        dom.points().map(|x| ExtF::from_base(x) - zeta).collect();
    batch_inverse(&mut deep_inv);
    let mut h = Vec::with_capacity(c_size);
    for j in 0..c_size {
        let s = a_evals[j] + v_evals[j] + powers[2 * width] * q_evals[j] - const_term;
        h.push(s * deep_inv[j]);
    }


    let fri = prove_fri(&h, &dom, &plan.fri, t);


    // Outer openings at each query's low position.
    let half = c_size / 2;
    let mut outer = Vec::with_capacity(fri.positions.len());
    for &q in &fri.positions {
        let j = (q as usize) % half;
        let jn = j + step;
        let quot_row = quot_mat.row(j);
        outer.push(OuterOpenings {
            trace_low: RowOpening {
                row: trace_mat.row(j).to_vec(),
                path: trace_tree.auth_path(j),
            },
            trace_next: RowOpening {
                row: trace_mat.row(jn).to_vec(),
                path: trace_tree.auth_path(jn),
            },
            quotient: RowOpening {
                row: vec![quot_row[0], quot_row[1]],
                path: quot_tree.auth_path(j),
            },
        });
    }


    Ok(ComposedProof {
        trace_root,
        quotient_root: quot_root,
        trace_zeta,
        trace_zeta_next,
        quotient_zeta,
        num_assertions: num_assertions as u64,
        fri,
        outer,
    })
}


fn vc_index(v_coeffs: &[ExtF], vc: &ExtF) -> usize {
    (vc as *const ExtF as usize - v_coeffs.as_ptr() as usize) / size_of::<ExtF>()
}


// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------


pub fn compose_verify<A>(
    plan: &Plan,
    air: &A,
    width: usize,
    prep_rows: &[Vec<Goldilocks>],
    publics: &[Goldilocks],
    proof: &ComposedProof,
    t: &mut FsTranscript,
) -> Result<bool, ComposeError>
where
    A: Sync,
    for<'a> A: Air<PointCollector<'a, Goldilocks>> + Air<PointCollector<'a, ExtF>>,
{
    let n = plan.n();
    if width == 0 {
        return Err(ComposeError::ZeroWidth);
    }
    let prep_width = prep_rows.first().map_or(0, |r| r.len());
    if !prep_rows.is_empty() {
        if prep_rows.len() != n {
            return Err(ComposeError::PrepHeight { log_n: plan.log_n, found: prep_rows.len() });
        }
        if prep_rows.iter().any(|r| r.len() != prep_width) {
            return Err(ComposeError::RaggedPrep);
        }
    }


    // Proof-shape validation: adversarial input is rejected, never panics.
    if proof.trace_zeta.len() != width || proof.trace_zeta_next.len() != width {
        return Ok(false);
    }
    if proof.outer.len() != plan.fri.num_queries {
        return Ok(false);
    }


    let dom = TwoAdicDomain::standard_coset(plan.log_c);
    let c_size = dom.size();
    let step = dom.shift_step_for(plan.log_n);


    // Transcript replay — the prover's sequence, exactly.
    absorb_instance(t, &shape_instance(plan, width, prep_width, publics.len()));
    t.absorb_hash(&proof.trace_root);
    absorb_publics(t, publics);
    let alpha = sample_ext(t, TAG_ALPHA);
    t.absorb_hash(&proof.quotient_root);
    let zeta = sample_ext(t, TAG_ZETA);
    if zeta_in_trace_domain(plan.log_n, zeta) || zeta_in_domain(&dom, zeta) {
        return Ok(false);
    }
    absorb_deep_values(t, &proof.trace_zeta, &proof.trace_zeta_next, proof.quotient_zeta);
    let gamma = sample_ext(t, TAG_DEEP);


    if !verify_fri(&proof.fri, &dom, &plan.fri, t) {
        return Ok(false);
    }


    // Outer checks: one per FRI query, at the layer-0 low position.
    let powers = gamma_powers(gamma, width);
    let half = c_size / 2;
    for (k, o) in proof.outer.iter().enumerate() {
        if o.trace_low.row.len() != width
            || o.trace_next.row.len() != width
            || o.quotient.row.len() != 2
        {
            return Ok(false);
        }
        let q = proof.fri.positions[k];
        let j = (q as usize) % half;
        let jn = j + step;
        if !o.trace_low.verify(&proof.trace_root, j) {
            return Ok(false);
        }
        if !o.trace_next.verify(&proof.trace_root, jn) {
            return Ok(false);
        }
        if !o.quotient.verify(&proof.quotient_root, j) {
            return Ok(false);
        }
        let x_lo = ExtF::from_base(dom.point(j));
        let h_val = proof.fri.queries[k].openings[0].low;
        let mut s = ExtF::ZERO;
        for i in 0..width {
            s = s + powers[i] * (ExtF::from_base(o.trace_low.row[i]) - proof.trace_zeta[i]);
            s = s + powers[width + i]
                * (ExtF::from_base(o.trace_next.row[i]) - proof.trace_zeta_next[i]);
        }
        let qv = ExtF::new(o.quotient.row[0], o.quotient.row[1]);
        s = s + powers[2 * width] * (qv - proof.quotient_zeta);
        if h_val * (x_lo - zeta) != s {
            return Ok(false);
        }
    }


    // ζ-check: the α-folded constraints at ζ against Q(ζ)·Z_H(ζ), with the
    // VERIFIER's prep (Lagrange over its authenticated table).
    let lw = lagrange_weights(plan.log_n, zeta);
    let prep_z = eval_table_at_ext(&lw, prep_rows);
    let sels_z = selector_vals_ext(plan.log_n, zeta);
    let pubs_z: Vec<ExtF> = publics.iter().map(|p| ExtF::from_base(*p)).collect();
    let (f_zeta, count) = PointCollector::new(
        &proof.trace_zeta,
        &proof.trace_zeta_next,
        &prep_z,
        &pubs_z,
        sels_z,
        alpha,
    )
    .evaluate(air);
    if count as u64 != proof.num_assertions {
        return Ok(false);
    }
    let zh_z = zeta.pow(n as u64) - ExtF::ONE;
    if f_zeta != proof.quotient_zeta * zh_z {
        return Ok(false);
    }


    Ok(true)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::air::builder::{Air as _, AirBuilder};
    use crate::air::chips::conservation::{
        gen_cons_prep, gen_cons_trace, ConservationChip, PREP_ACTIVE, WIDTH as CONS_W,
    };
    use crate::stark::collector::measure_true;
    use crate::testutil::SplitMix64;
    use nerv_core::field::GOLDILOCKS_PRIME;


    fn gf(rng: &mut SplitMix64) -> Goldilocks {
        Goldilocks::from_u64_reduce(rng.next_u64())
    }


    fn seeded(seed: u64) -> FsTranscript {
        let mut t = FsTranscript::new();
        t.absorb_bytes(&seed.to_le_bytes());
        t
    }


    fn shape(b: usize, f: usize, q: usize, pow: usize) -> FriShape {
        FriShape {
            log_blowup: b,
            num_queries: q,
            log_final_poly_len: f,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: pow,
        }
    }


    struct CounterAir;


    impl<B: AirBuilder> Air<B> for CounterAir {
        fn eval(&self, b: &mut B) {
            let cur = b.witness(0, 0);
            let nxt = b.witness(1, 0);
            let one = B::constant(1);
            let step = b.is_transition();
            b.assert_zero(step * (nxt - cur - one), "counter_step");
            let first = b.is_first_row();
            let want = b.public(0);
            b.assert_zero(first * (cur - want), "counter_init");
        }
    }


    fn counter_trace() -> Vec<Vec<Goldilocks>> {
        (0..8).map(|i| vec![Goldilocks::from_u64_reduce(i)]).collect()
    }


    fn counter_plan(b: usize, f: usize, q: usize, pow: usize) -> Plan {
        let td = measure_true(&CounterAir);
        Plan::new(3, &shape(b, f, q, pow), &td).unwrap()
    }


    #[test]
    fn plan_validation() {
        let td = measure_true(&CounterAir);
        assert!(matches!(
            Plan::new(0, &shape(1, 1, 4, 0), &td),
            Err(ComposeError::TraceTooSmall)
        ));
        assert!(matches!(
            Plan::new(3, &shape(0, 1, 4, 0), &td),
            Err(ComposeError::VacuousFri)
        ));
        assert!(matches!(
            Plan::new(3, &shape(1, 1, 0, 0), &td),
            Err(ComposeError::NoQueries)
        ));
        assert!(matches!(
            Plan::new(3, &shape(1, 5, 4, 0), &td),
            Err(ComposeError::NoFold { .. })
        ));
        assert!(matches!(
            Plan::new(30, &shape(2, 1, 4, 0), &td),
            Err(ComposeError::DomainOverflow { .. })
        ));
        // headroom: qb = 9 at N = 8 ⇒ one chunk level.
        let p = counter_plan(1, 1, 16, 0);
        assert_eq!(p.log_chunks, 1);
        assert_eq!(p.log_c, 5);
        assert_eq!(p.n(), 8);
        assert_eq!(p.c(), 32);
    }


    #[test]
    fn dimension_errors() {
        let plan = counter_plan(1, 1, 16, 0);
        let mut t = seeded(1);
        let rows7: Vec<Vec<Goldilocks>> = (0..7).map(|i| vec![Goldilocks::from_u64_reduce(i)]).collect();
        assert!(matches!(
            compose_prove(&plan, &CounterAir, &rows7, &[], &[Goldilocks::ZERO], &mut t),
            Err(ComposeError::TraceHeight { found: 7, .. })
        ));
       let mut ragged = counter_trace();
        ragged[3].push(Goldilocks::ZERO);
        let mut t = seeded(2);
        assert!(matches!(
            compose_prove(&plan, &CounterAir, &ragged, &[], &[Goldilocks::ZERO], &mut t),
            Err(ComposeError::RaggedTrace)
        ));
        let mut t = seeded(3);
        assert!(matches!(
            compose_prove(&plan, &CounterAir, &rows7, &[], &[Goldilocks::ZERO], &mut t),
            Err(ComposeError::TraceHeight { found: 7, .. })
        ));

    }


    #[test]
    fn selector_arrays_match_ext_and_identity() {
        for log_n in 1..=6usize {
            let dom = TwoAdicDomain::standard_coset(log_n + 2);
            let arr = selector_arrays(log_n, &dom);
            let n = 1usize << log_n;
            let n_g = Goldilocks::from_u64_reduce(n as u64);
            let h_last = two_adic_generator(log_n).inverse();
            for j in [0usize, 1, dom.size() / 2, dom.size() - 1] {
                let x = dom.point(j);
                let zh = x.pow(n as u64) - Goldilocks::ONE;
                assert_eq!(arr.first[j] * n_g * (x - Goldilocks::ONE), zh, "log={log_n} j={j}");
                assert_eq!(arr.last[j] * n_g * (x - h_last), zh, "log={log_n} j={j}");
                assert_eq!(arr.transition[j], x - h_last);
            }
            for j in (0..dom.size()).step_by((dom.size() / 8).max(1)) {
                let s = selector_vals_ext(log_n, ExtF::from_base(dom.point(j)));
                assert_eq!(ExtF::from_base(arr.first[j]), s.first, "log={log_n} j={j}");
                assert_eq!(ExtF::from_base(arr.last[j]), s.last, "log={log_n} j={j}");
                assert_eq!(ExtF::from_base(arr.transition[j]), s.transition, "log={log_n} j={j}");
            }
        }
    }


    #[test]
    fn lagrange_partition_and_interpolation() {
        let mut rng = SplitMix64::new(0x1A6);
        for log_n in 1..=6usize {
            let zeta = ExtF::new(gf(&mut rng), gf(&mut rng));
            if zeta_in_trace_domain(log_n, zeta) {
                continue;
            }
            let lw = lagrange_weights(log_n, zeta);
            let sum = lw.iter().fold(ExtF::ZERO, |a, b| a + *b);
            assert_eq!(sum, ExtF::ONE, "log={log_n}: partition of unity");
            let n = 1usize << log_n;
            let width = 3;
            let table: Vec<Vec<Goldilocks>> =
                (0..n).map(|_| (0..width).map(|_| gf(&mut rng)).collect()).collect();
            let got = eval_table_at_ext(&lw, &table);
            let dom = TwoAdicDomain::subgroup(log_n);
            for c in 0..width {
                let col: Vec<Goldilocks> = table.iter().map(|r| r[c]).collect();
                let coeffs = fft::interpolate(&col, &dom);
                assert_eq!(got[c], horner_ext(&coeffs, zeta), "log={log_n} c={c}");
            }
        }
        // empty table: empty result, no panic.
        assert!(eval_table_at_ext(&[ExtF::ONE; 8], &[]).is_empty());
    }


    #[test]
    fn horner_matches_fft() {
        let mut rng = SplitMix64::new(0x40F);
        for log in 1..=6usize {
            let dom = TwoAdicDomain::standard_coset(log);
            let coeffs: Vec<Goldilocks> = (0..dom.size()).map(|_| gf(&mut rng)).collect();
            let evals = fft::evaluate(&coeffs, &dom);
            for j in (0..dom.size()).step_by((dom.size() / 5).max(1)) {
                assert_eq!(horner_ext(&coeffs, ExtF::from_base(dom.point(j))), evals[j]);
            }
        }
    }


    #[test]
    fn batch_inverse_g_matches_per_element() {
        let mut rng = SplitMix64::new(0x817);
        for n in 0usize..17 {
            let mut vals: Vec<Goldilocks> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut v = gf(&mut rng);
                while v.is_zero() {
                    v = gf(&mut rng);
                }
                vals.push(v);
            }
            let expect: Vec<Goldilocks> = vals.iter().map(|v| v.inverse()).collect();
            batch_inverse_g(&mut vals);
            assert_eq!(vals, expect, "n={n}");
        }
    }


    #[test]
    fn zeta_membership_predicates() {
        let dom = TwoAdicDomain::standard_coset(5);
        let z_d = ExtF::from_base(dom.point(3));
        assert!(zeta_in_domain(&dom, z_d));
        assert!(!zeta_in_trace_domain(5, z_d));
        let z_h = ExtF::from_base(two_adic_generator(5));
        assert!(zeta_in_trace_domain(5, z_h));
        assert!(!zeta_in_domain(&dom, z_h));
        let z_ext = ExtF::new(Goldilocks::ONE, Goldilocks::ONE);
        assert!(!zeta_in_trace_domain(5, z_ext));
        assert!(!zeta_in_domain(&dom, z_ext));
        let z_one = ExtF::ONE;
        assert!(zeta_in_trace_domain(5, z_one));
        assert!(!zeta_in_domain(&dom, z_one));
    }


    #[test]
    fn counter_roundtrip_determinism_grinding() {
        for (b, f, q, pow) in [(1usize, 1usize, 16usize, 0usize), (2, 1, 24, 8), (1, 2, 12, 0)] {
            let plan = counter_plan(b, f, q, pow);
            let rows = counter_trace();
            let pubs = [Goldilocks::ZERO];


            let mut tp = seeded(0xC0);
            let proof =
                compose_prove(&plan, &CounterAir, &rows, &[], &pubs, &mut tp).unwrap();
            let mut tv = seeded(0xC0);
            assert!(
                compose_verify(&plan, &CounterAir, 1, &[], &pubs, &proof, &mut tv).unwrap(),
                "b={b} f={f} q={q} pow={pow}"
            );


            let mut tp2 = seeded(0xC0);
            assert_eq!(
                compose_prove(&plan, &CounterAir, &rows, &[], &pubs, &mut tp2).unwrap(),
                proof,
                "determinism"
            );


            let mut tv2 = seeded(0xC1);
            assert!(!compose_verify(&plan, &CounterAir, 1, &[], &pubs, &proof, &mut tv2).unwrap());


            let mut tv3 = seeded(0xC0);
            assert!(!compose_verify(
                &plan,
                &CounterAir,
                1,
                &[],
                &[Goldilocks::ONE],
                &proof,
                &mut tv3
            )
            .unwrap());


            let mut tv4 = seeded(0xC0);
            assert!(!compose_verify(&plan, &CounterAir, 2, &[], &pubs, &proof, &mut tv4).unwrap());
        }
    }


    #[test]
    fn conservation_roundtrip_with_prep() {
        let entries = vec![(100u64, true), (60u64, false), (40u64, false)];
        let mut rows = gen_cons_trace(&entries);
        rows.push(vec![Goldilocks::ZERO; CONS_W]);
        let mut prep = gen_cons_prep(1, 2, 0);
        prep.push(vec![Goldilocks::ZERO; 6]);


        let td = measure_true(&ConservationChip::new());
        let plan = Plan::new(2, &shape(1, 1, 16, 0), &td).unwrap();
        assert_eq!(plan.log_chunks, 2);
        assert_eq!(plan.log_c, 5);


        let chip = ConservationChip::new();
        let mut tp = seeded(0xE0);
        let proof = compose_prove(&plan, &chip, &rows, &prep, &[], &mut tp).unwrap();
        let mut tv = seeded(0xE0);
        assert!(compose_verify(&plan, &chip, CONS_W, &prep, &[], &proof, &mut tv).unwrap());


        // THE authenticity pin: a verifier holding a different prep table
        // rejects — the ζ-check runs on the verifier's own Lagrange values.
        let mut bad_prep = prep.clone();
        bad_prep[0][PREP_ACTIVE] = Goldilocks::ZERO;
        let mut tv2 = seeded(0xE0);
        assert!(!compose_verify(&plan, &chip, CONS_W, &bad_prep, &[], &proof, &mut tv2).unwrap());


        // A prep of the wrong height is a setup error, not a rejection.
        let mut short_prep = prep.clone();
        short_prep.pop();
        let mut tv3 = seeded(0xE0);
        assert!(matches!(
            compose_verify(&plan, &chip, CONS_W, &short_prep, &[], &proof, &mut tv3),
            Err(ComposeError::PrepHeight { .. })
        ));
    }


    #[test]
    fn compose_tamper_battery() {
        let plan = counter_plan(1, 1, 16, 0);
        let rows = counter_trace();
        let pubs = [Goldilocks::ZERO];
        let mut tp = seeded(0x7A);
        let proof = compose_prove(&plan, &CounterAir, &rows, &[], &pubs, &mut tp).unwrap();


        let check = |p: &ComposedProof| {
            let mut t = seeded(0x7A);
            compose_verify(&plan, &CounterAir, 1, &[], &pubs, p, &mut t).unwrap()
        };
        assert!(check(&proof));


        let flip = |h: &mut Hash256| {
            let mut b = *h.as_bytes();
            b[0] ^= 1;
            *h = Hash256::from_bytes(b);
        };


        let mut p = proof.clone();
        flip(&mut p.trace_root);
        assert!(!check(&p), "trace root");


        let mut p = proof.clone();
        flip(&mut p.quotient_root);
        assert!(!check(&p), "quotient root");


        let mut p = proof.clone();
        p.trace_zeta[0] = p.trace_zeta[0] + ExtF::ONE;
        assert!(!check(&p), "trace_zeta");


        let mut p = proof.clone();
        p.trace_zeta_next[0] = p.trace_zeta_next[0] + ExtF::ONE;
        assert!(!check(&p), "trace_zeta_next");


        let mut p = proof.clone();
        p.quotient_zeta = p.quotient_zeta + ExtF::ONE;
        assert!(!check(&p), "quotient_zeta");


        let mut p = proof.clone();
        p.num_assertions += 1;
        assert!(!check(&p), "num_assertions");


        let mut p = proof.clone();
        p.outer[0].trace_low.row[0] = p.outer[0].trace_low.row[0] + Goldilocks::ONE;
        assert!(!check(&p), "outer trace row");


        let mut p = proof.clone();
        p.outer[0].trace_next.row[0] = p.outer[0].trace_next.row[0] + Goldilocks::ONE;
        assert!(!check(&p), "outer next row");


        let mut p = proof.clone();
        p.outer[0].quotient.row[0] = p.outer[0].quotient.row[0] + Goldilocks::ONE;
        assert!(!check(&p), "outer quotient row");


        let mut p = proof.clone();
        p.outer[0].quotient.row.push(Goldilocks::ZERO);
        assert!(!check(&p), "outer quotient width");


        let mut p = proof.clone();
        p.outer.pop();
        assert!(!check(&p), "missing outer");


        let mut p = proof.clone();
        p.outer.push(p.outer[0].clone());
        assert!(!check(&p), "extra outer");


        let mut p = proof.clone();
        if !p.outer[0].trace_low.path.is_empty() {
            p.outer[0].trace_low.path.pop();
            assert!(!check(&p), "short outer path");
        }


        let mut p = proof.clone();
        p.fri.positions[0] ^= 1;
        assert!(!check(&p), "positions");


        let mut p = proof.clone();
        p.fri.queries[0].openings[0].low = p.fri.queries[0].openings[0].low + ExtF::ONE;
        assert!(!check(&p), "fri layer-0 value");


        let mut p = proof.clone();
        p.fri.final_coeffs[0] = p.fri.final_coeffs[0] + ExtF::ONE;
        assert!(!check(&p), "fri final");
    }


    #[test]
    fn wrong_plan_rejects() {
        let rows = counter_trace();
        let pubs = [Goldilocks::ZERO];
        let plan = counter_plan(1, 1, 16, 0);
        let mut tp = seeded(0x99);
        let proof = compose_prove(&plan, &CounterAir, &rows, &[], &pubs, &mut tp).unwrap();


        let other = counter_plan(1, 1, 20, 0);
        let mut tv = seeded(0x99);
        assert!(!compose_verify(&other, &CounterAir, 1, &[], &pubs, &proof, &mut tv).unwrap());
    }


    #[test]
    fn goldilocks_prime_is_not_one() {
        // Sanity for the membership predicates' exactness premise.
        assert_ne!(GOLDILOCKS_PRIME, 1);
    }
}


trait ConcatEight {
    fn concat_eight(&self, _: &[Vec<Goldilocks>]) -> Vec<Vec<Goldilocks>>;
}
impl ConcatEight for [Vec<Goldilocks>] {
    fn concat_eight(&self, _: &[Vec<Goldilocks>]) -> Vec<Vec<Goldilocks>> {
        unreachable!()
    }
}
