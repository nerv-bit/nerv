//! The whole-transaction AIR (WP §5.1–§5.2): custody + delta + seal in ONE
//! constraint system, one Fiat–Shamir transcript, statement 11 bound by the
//! caller's `bind_transaction` prior to the engine draws. Statements 6–10
//! are chained end to end: the encoder proves δ = W·ΔS into DREG; the digit
//! rows prove δ's 8-bit decomposition into PT; the m-binding ties the seal's
//! AX_MB digits to PT at the v-transform output rows; the seal chip proves
//! (u, v) well-formed over those digits.
//!
//! Composition geometry (register 65): modules side-by-side in columns over
//! H = max(custody, delta, seal) rows; custody's registers extend into the
//! taller tail; every seal leg shares rows [0, 242) and ONE prep block (the
//! seal prep is leg-independent — phases, twiddles, and MOP depend only on
//! the epoch key and the transform structure; pinned by test). Cross-module
//! bindings: m ↔ PT; per-leg fee ↔ two 32-bit public words (two-limb form is
//! collision-free — a single 64-bit word admits a fee+p bit alias); the
//! expiry block (public expiry < 2^40, E = q·C + rem, 16 time-bucket one-hots
//! derived with a strict upper bound — uniqueness of the bucket); type
//! one-hots preprocessed from public leg structure.
//!
//! The verifier regenerates its own prep from public data (`gen_tx_prep`),
//! differentially pinned against the prover's. Driver obligations: every leg
//! expiry < 2^40 (native, `ExpiryTooLarge`); the shell's ct bytes must
//! serialize the proven (u, v) (`check_ct_binding` — the tie that makes
//! ct_B aggregation sound). Claim legs remain a later statement set (WP
//! §12.3; register 65). Engine runs at these oracle widths are the
//! production-form milestone (register 61): the relations here are the
//! DSR-7 reference and are fully testable via NativeEval.

use nerv_core::codec::{Decode, Encode, Reader};
use nerv_core::error::CodecError;
use nerv_core::field::Goldilocks;
use nerv_codec::codec_w::CodecW;
use nerv_codec::features::FeatureVector;
use nerv_custody::error::CustodyError;
use nerv_custody::nct::DEPTH;
use nerv_custody::tx::TransactionShell;
use nerv_seal::SealError;
use nerv_seal::digitize::Plaintext;
use nerv_seal::encrypt::PublicKey;
use nerv_seal::ring::{fwd_twiddle, inv_twiddle, Mat2x8, Mat2x8Ntt, Mat8x8Ntt, Poly, Vec8};
use nerv_seal::sampling::expand_matrix;


use crate::air::builder::{Air, AirBuilder};
use crate::air::chips::blake3::{
    gen_prep as bl3_prep, CHUNK_END, CHUNK_START, PARENT, ROOT,
};
use crate::air::chips::conservation::{gen_cons_prep, ACCP_BASE};
use crate::air::chips::encoder::{
    PF_MAC, PF_MAC_FINAL, PF_MAC_START, PF_SEL, PF_WM, PF_WMF, PF_WMT, PF_WMW, PF_WS, PF_WSF,
    PF_WST, PF_WSW,
};
use crate::air::chips::merkle_poseidon2::MerkleChip;
use crate::air::chips::seal_chip::{
    gen_seal_trace, AX_MB, F_INV_T, F_IS_INV, F_IS_T_OUT, F_LEVEL, F_MACI, F_MACJ, F_IS_INPUT,
    F_FWD_T, MOP, N as SEAL_N, PV_SEAL_COPY, SealChip, SealLegInput, TW,
    LEG_W as SEAL_LEG_W, PREP_COLS as SEAL_PREP, ROWS as SEAL_ROWS,
};
use crate::air::custody_air::{
    gen_custody_trace, set_burn_flags, set_input_flags, set_output_flags, set_revert_flags,
    BL3_PREP_BASE, BL3_PREP_W, BP, BURN_REG_W, CONS_PREP_BASE, IN_FLAGS, INPUT_REG_W,
    MERKLE_PREP_W, OUT_REG_W, REV_REG_W, CustodyAir,
};
use crate::air::delta_air;
use crate::air::delta_air::{
    gen_delta_trace, FEEB, LEG_W, PV_DELTA_ROW, PV_DIGIT, PV_FINAL, PV_LOG, PV_LOG0,
    PV_LOG_ACTIVE, PV_TIE, PV_VOL, PV_VOL0, PV_VOL_ACTIVE, R_PT, R_TIME, R_TYPE, ROWS_PER_LEG,
    TIE_FEE, TIE_LOG, TIE_MACF, TIE_STRIDE, TIE_VOL, TIE_ZROW, DeltaAir, LegShape,
};
use crate::air::chips::seal_chip::F_IS_MAC;
use crate::air::fs::FsTranscript;
use crate::error::{WitnessError, WitnessGenError};
use crate::security::FriShape;
use crate::stark::prover::{ProveError, Prover};
use crate::stark::verifier::{Verifier, VerifyError};
use crate::witness_gen::TransactionWitness;
use crate::stark::prover::Proved; 


const MASK32: u64 = 0xFFFF_FFFF;
/// The time-bucket modulus: the epoch length in intervals (erratum 4).
pub const C_EPOCH: u64 = nerv_core::types::INTERVALS_PER_EPOCH;
const C_EPOCH_F: u64 = C_EPOCH;
const _: () = assert!(C_EPOCH == 86_400);
/// The in-circuit expiry bound (register 65); heights stay far below it.
pub const EXPIRY_BITS: u32 = 40;


/// Expiry-block layout (per leg, stride EXP_STRIDE).
pub const EXP_EB: usize = 0; // 64 bits of E
pub const EXP_Q: usize = 64; // quotient
pub const EXP_QB: usize = 65; // 25 bits
pub const EXP_REM: usize = 90; // remainder
pub const EXP_RB: usize = 91; // 17 bits
pub const EXP_BF: usize = 108; // 16 bucket flags
pub const EXP_XB: usize = 124; // 17 bits
pub const EXP_YB: usize = 141; // 17 bits
pub const EXP_STRIDE: usize = 158;
const _: () = assert!(EXP_YB + 17 == EXP_STRIDE);


pub const SEAL_PUBS_PER_LEG: usize = 8 * SEAL_N + 2 * SEAL_N;


#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TxError {
    #[error(transparent)]
    Shell(#[from] CustodyError),
    #[error(transparent)]
    Witness(#[from] WitnessError),
    #[error(transparent)]
    WitnessGen(#[from] WitnessGenError),
    #[error(transparent)]
    Seal(#[from] SealError),
    #[error(transparent)]
    Prove(#[from] ProveError),
    #[error(transparent)]
    Verify(#[from] VerifyError),
    #[error("leg {leg}: expiry {value} exceeds the 2^40 in-circuit bound")]
    ExpiryTooLarge { leg: usize, value: u64 },
    #[error("leg {leg}: shell ct does not serialize the proven (u, v)")]
    CtMismatch { leg: usize },
}


#[derive(Clone, Debug)]
pub struct TxAir {
    pub custody: CustodyAir,
    pub delta: DeltaAir,
    pub seal_col_base: usize,
    pub seal_prep_base: usize,
    pub seal_public_base: usize,
    pub exp_col_base: usize,
    pub tx_prep_base: usize,
    pub fee_pub_base: usize,
    pub expiry_pub_base: usize,
    pub burns_per_leg: Vec<usize>,
}


impl TxAir {
    pub fn rows(&self) -> usize {
        self.custody.rows().max(self.delta.rows()).max(SEAL_ROWS)
    }


    pub fn cols(&self) -> usize {
        let n = self.delta.n_legs();
        self.exp_col_base + EXP_STRIDE * n
    }


    pub fn prep_cols(&self) -> usize {
        self.tx_prep_base + 5 * self.delta.n_legs()
    }


    pub fn pub_count(&self) -> usize {
        let n = self.delta.n_legs();
        let c = &self.custody;
        4 * c.n_inputs
            + 8 * c.n_inputs
            + 8 * c.n_outputs
            + 8 * c.n_cond()
            + 8 * c.n_burns
            + SEAL_PUBS_PER_LEG * n
            + 2 * n
            + n
    }


    /// The cross-module bindings (statements' chaining), shared verbatim by
    /// the composed eval and the module-slice test AIRs.
    fn bindings<B: AirBuilder>(&self, b: &mut B) {
        let one = B::constant(1);
        let n = self.delta.n_legs();
        let d_cb = self.delta.col_base;
        let s_pb = self.seal_prep_base;
        let delta_row = b.preprocessed(self.delta.prep_base + PV_DELTA_ROW);


        // (1) m-binding: seal AX_MB digits == delta PT, at the v-transforms'
        // F10 rows (232 for ring 0, 241 for ring 1).
        let is_inv = b.preprocessed(s_pb + F_IS_INV);
        let is_t_out = b.preprocessed(s_pb + F_IS_T_OUT);
        for l in 0..n {
            let s_cb = self.seal_col_base + SEAL_LEG_W * l;
            let d_leg = d_cb + LEG_W * l;
            for ring in 0..2usize {
                let gate =
                    is_inv.clone() * is_t_out.clone() * b.preprocessed(s_pb + F_INV_T + 8 + ring);
                for c in 0..SEAL_N {
                    let mut m_val = B::constant(0);
                    for t in 0..8 {
                        m_val = m_val
                            + b.witness(0, s_cb + AX_MB + 8 * c + t) * B::constant(1u64 << t);
                    }
                    let pt = b.witness(0, d_leg + R_PT + SEAL_N * ring + c);
                    b.assert_zero(gate.clone() * (m_val - pt), "tx_m_bind");
                }
            }
        }


        // (2) fee: delta FEEB limbs == the two public fee words (exact
        // 32-bit limb equality — no bit aliasing possible).
        for l in 0..n {
            let d_leg = d_cb + LEG_W * l;
            let mut lo = B::constant(0);
            let mut hi = B::constant(0);
            for t in 0..32 {
                lo = lo + b.witness(0, d_leg + FEEB + t) * B::constant(1u64 << t);
                hi = hi + b.witness(0, d_leg + FEEB + 32 + t) * B::constant(1u64 << t);
            }
            b.assert_zero(
                delta_row.clone() * (lo - b.public(self.fee_pub_base + 2 * l)),
                "tx_fee_lo",
            );
            b.assert_zero(
                delta_row.clone() * (hi - b.public(self.fee_pub_base + 2 * l + 1)),
                "tx_fee_hi",
            );
        }


        // (3) expiry blocks: E == public, E < 2^40, E = q·C + rem, bucket
        // one-hot with strict uniqueness, copies into delta's registers.
        for l in 0..n {
            let eb = self.exp_col_base + EXP_STRIDE * l;
            let d_leg = d_cb + LEG_W * l;
            let mut e = B::constant(0);
            for t in 0..64 {
                let bit = b.witness(0, eb + EXP_EB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "tx_ebit");
                if t < EXPIRY_BITS as usize {
                    e = e + bit.clone() * B::constant(1u64 << t);
                } else {
                    b.assert_zero(bit, "tx_ecap");
                }
            }
            b.assert_zero(e.clone() - b.public(self.expiry_pub_base + l), "tx_epub");


            let q = b.witness(0, eb + EXP_Q);
            let mut q_r = B::constant(0);
            for t in 0..25 {
                let bit = b.witness(0, eb + EXP_QB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "tx_qbit");
                q_r = q_r + bit.clone() * B::constant(1u64 << t);
            }
            b.assert_zero(q.clone() - q_r, "tx_q_recomp");
            let rem = b.witness(0, eb + EXP_REM);
            let mut rem_r = B::constant(0);
            for t in 0..17 {
                let bit = b.witness(0, eb + EXP_RB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "tx_rbit");
                rem_r = rem_r + bit.clone() * B::constant(1u64 << t);
            }
            b.assert_zero(rem.clone() - rem_r, "tx_rem_recomp");
            b.assert_zero(q * B::constant(C_EPOCH_F) + rem.clone() - e, "tx_div");


            let mut bsum = B::constant(0);
            let mut bc = B::constant(0);
            let mut bc1 = B::constant(0);
            for t in 0..16 {
                let f = b.witness(0, eb + EXP_BF + t);
                b.assert_zero(f.clone() * (f.clone() - one.clone()), "tx_bfbit");
                bsum = bsum + f.clone();
                bc = bc + f.clone() * B::constant(t as u64);
                bc1 = bc1 + f.clone() * B::constant((t + 1) as u64);
            }
            b.assert_zero(bsum - one.clone(), "tx_bf_one");
            let mut xb = B::constant(0);
            for t in 0..17 {
                let bit = b.witness(0, eb + EXP_XB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "tx_xbit");
                xb = xb + bit.clone() * B::constant(1u64 << t);
            }
            let mut yb = B::constant(0);
            for t in 0..17 {
                let bit = b.witness(0, eb + EXP_YB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "tx_ybit");
                yb = yb + bit.clone() * B::constant(1u64 << t);
            }
            let c_f = B::constant(C_EPOCH_F);
            let rem16 = rem.clone() * B::constant(16);
            // x = 16·rem − b·C ∈ [0, 2^17): forces b·C ≤ 16·rem.
            b.assert_zero(rem16.clone() - bc.clone() * c_f.clone() - xb, "tx_xb");
            // y' = (b+1)·C − 16·rem − 1 ∈ [0, 2^17): forces 16·rem < (b+1)C
            // strictly — with the lower bound, b = ⌊16·rem/C⌋ uniquely.
            b.assert_zero(bc1.clone() * c_f - rem16.clone() - one.clone() - yb, "tx_yb");


            for t in 0..16 {
                b.assert_zero(
                    delta_row.clone()
                        * (b.witness(0, d_leg + R_TIME + t) - b.witness(0, eb + EXP_BF + t)),
                    "tx_time_copy",
                );
            }
            for t in 0..5 {
                b.assert_zero(
                    delta_row.clone()
                        * (b.witness(0, d_leg + R_TYPE + t)
                            - b.preprocessed(self.tx_prep_base + 5 * l + t)),
                    "tx_type_copy",
                );
            }
        }
    }
}


impl<B: AirBuilder> Air<B> for TxAir {
    fn eval(&self, b: &mut B) {
        self.custody.eval(b);
        self.delta.eval(b);
        let n = self.delta.n_legs();
        for l in 0..n {
            SealChip::at(
                self.seal_col_base + SEAL_LEG_W * l,
                self.seal_prep_base,
                self.seal_public_base + SEAL_PUBS_PER_LEG * l,
            )
            .eval(b);
        }
        self.bindings(b);
    }
}


fn leg_kind_index(air: &TxAir, l: usize) -> usize {
    // Erratum 3's order: SingleShard, CrossShardSpend, CrossShardIssue,
    // Claim, Burn. Claim legs are a later statement set (register 65).
    if air.burns_per_leg[l] > 0 {
        4
    } else if air.delta.legs[l].n_in == 0 {
        2
    } else if air.delta.n_legs() > 1 {
        1
    } else {
        0
    }
}


pub fn build_tx_air(shell: &TransactionShell) -> Result<TxAir, TxError> {
    let canon = shell.canonicalize()?;
    let n_in: usize = canon.legs.iter().map(|l| l.inputs.nullifiers.len()).sum();
    let n_out: usize = canon.legs.iter().map(|l| l.outputs.len()).sum();
    let n_burns: usize = canon.legs.iter().map(|l| l.burns.len()).sum();
    let mut cond_outputs = Vec::new();
    {
        let mut idx = 0usize;
        for leg in &canon.legs {
            for out in &leg.outputs {
                if out.conditional {
                    cond_outputs.push(idx);
                }
                idx += 1;
            }
        }
    }
    let legs: Vec<LegShape> = canon
        .legs
        .iter()
        .map(|l| LegShape { n_in: l.inputs.nullifiers.len(), n_out: l.outputs.len() })
        .collect();
    let burns_per_leg: Vec<usize> = canon.legs.iter().map(|l| l.burns.len()).collect();
    let custody = CustodyAir::new(n_in, n_out, cond_outputs, n_burns, DEPTH);
    let delta = DeltaAir {
        legs,
        cons_base: custody.cons_base(),
        cust_reg_base: custody.reg_base(),
        cust_input_reg_w: INPUT_REG_W,
        cust_n_in: n_in,
        cust_out_reg_w: OUT_REG_W,
        col_base: custody.cols(),
        prep_base: custody.prep_cols(),
    };
    let n = delta.n_legs();
    let seal_col_base = custody.cols() + delta.cols();
    let seal_prep_base = custody.prep_cols() + delta.prep_cols();
    let exp_col_base = seal_col_base + SEAL_LEG_W * n;
    let tx_prep_base = seal_prep_base + SEAL_PREP;
    let pub_end = 4 * n_in + 8 * n_in + 8 * n_out + 8 * custody.n_cond() + 8 * n_burns;
    Ok(TxAir {
        custody,
        delta,
        seal_col_base,
        seal_prep_base,
        seal_public_base: pub_end,
        exp_col_base,
        tx_prep_base,
        fee_pub_base: pub_end + SEAL_PUBS_PER_LEG * n,
        expiry_pub_base: pub_end + SEAL_PUBS_PER_LEG * n + 2 * n,
        burns_per_leg,
    })
}


// ---------------------------------------------------------------------------
// Prover-side generation
// ---------------------------------------------------------------------------


#[derive(Clone, Debug)]
pub struct TxTrace {
    pub trace: Vec<Vec<Goldilocks>>,
    pub prep: Vec<Vec<Goldilocks>>,
    pub publics: Vec<Goldilocks>,
    pub air: TxAir,
}


pub fn centered_of(p: &Poly) -> [i64; SEAL_N] {
    let cl = p.centerlift();
    let mut out = [0i64; SEAL_N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = cl[i];
    }
    out
}


fn digits_of(pt: &Plaintext) -> [u16; 512] {
    let d = pt.digits();
    let mut out = [0u16; 512];
    for (i, o) in out.iter_mut().enumerate() {
        *o = d[i];
    }
    out
}


fn mat2x8_of(pk: &PublicKey) -> Mat2x8 {
    Mat2x8::new([
        pk.t().row(0).clone(),
        pk.t().row(1).clone(),
    ])
}


/// The seal witness conversion (wallet API surface).
pub fn seal_input_of(witness: &TransactionWitness, leg: usize) -> SealLegInput {
    let sw = &witness.seal[leg];
    SealLegInput {
        r: (0..8).map(|j| centered_of(&sw.r.polys()[j])).collect(),
        e1: (0..8).map(|j| centered_of(&sw.e1.polys()[j])).collect(),
        e2: (0..2).map(|j| centered_of(&sw.e2.polys()[j])).collect(),
        m: digits_of(&sw.plaintext),
    }
}


fn put_bits(row: &mut [Goldilocks], col: usize, v: u64, n: usize) {
    for t in 0..n {
        row[col + t] = Goldilocks::from_u32(((v >> t) & 1) as u32);
    }
}


fn put_word(row: &mut [Goldilocks], col: usize, v: u64) {
    row[col] = Goldilocks::from_u64_reduce(v);
}


fn fill_expiry_block(trace: &mut [Vec<Goldilocks>], base: usize, e: u64) {
    let rem = e % C_EPOCH;
    let q = e / C_EPOCH;
    let bucket = (16 * rem) / C_EPOCH;
    let x = 16 * rem - bucket * C_EPOCH;
    let yp = (bucket + 1) * C_EPOCH - 16 * rem - 1;
    for row in trace.iter_mut() {
        put_bits(row, base + EXP_EB, e, 64);
        put_word(row, base + EXP_Q, q);
        put_bits(row, base + EXP_QB, q, 25);
        put_word(row, base + EXP_REM, rem);
        put_bits(row, base + EXP_RB, rem, 17);
        for t in 0..16 {
            row[base + EXP_BF + t] = Goldilocks::from_u32(u32::from(t == bucket as usize));
        }
        put_bits(row, base + EXP_XB, x, 17);
        put_bits(row, base + EXP_YB, yp, 17);
    }
}


fn place_type_flags(prep: &mut [Vec<Goldilocks>], air: &TxAir) {
    let h_d = air.delta.rows();
    for l in 0..air.delta.n_legs() {
        let k = leg_kind_index(air, l);
        for r in 0..h_d {
            prep[r][air.tx_prep_base + 5 * l + k] = Goldilocks::ONE;
        }
    }
}


fn place_composed_flags(prep: &mut [Vec<Goldilocks>], air: &TxAir) {
    let c = &air.custody;
    let n_in = c.n_inputs;
    let db = air.delta.prep_base;
    for l in 0..air.delta.n_legs() {
        let row = n_in + c.n_outputs + c.n_burns + l;
        prep[row][db + TIE_STRIDE * l + TIE_FEE] = Goldilocks::ONE;
    }
    prep[n_in - 1][db + air.delta.boundary_col()] = Goldilocks::ONE;
}


#[allow(clippy::too_many_lines)]
pub fn gen_tx_trace(
    witness: &TransactionWitness,
    shell: &TransactionShell,
    w: &CodecW,
    epoch_pk: &PublicKey,
) -> Result<TxTrace, TxError> {
    witness.validate(shell, w, epoch_pk)?;
    let air = build_tx_air(shell)?;
    let canon = shell.canonicalize()?;
    let n_in = air.custody.n_inputs;
    let n = air.delta.n_legs();


    let ct = gen_custody_trace(&witness.custody, shell, DEPTH)?;
    let h_c = ct.trace.len();


    let cons = air.custody.cons_base();
    let boundary = [
        ct.trace[n_in - 1][cons + ACCP_BASE].as_u64(),
        ct.trace[n_in - 1][cons + ACCP_BASE + 1].as_u64(),
        ct.trace[n_in - 1][cons + ACCP_BASE + 2].as_u64(),
    ];
    let in_values: Vec<u64> = witness.custody.inputs.iter().map(|i| i.opening.value).collect();
    let out_values: Vec<u64> =
        witness.custody.outputs.iter().map(|o| o.opening.value).collect();
    let features: Vec<FeatureVector> =
        witness.delta.legs.iter().map(|d| d.features.clone()).collect();
    let dt = gen_delta_trace(
        w,
        &features,
        &witness.custody.fees,
        &air.delta.legs,
        &in_values,
        &out_values,
        boundary,
    )?;
    let h_d = dt.trace.len();


    let a_seed = *epoch_pk.a_seed();
    let t_mat = mat2x8_of(epoch_pk);
    let mut seal_traces = Vec::with_capacity(n);
    for l in 0..n {
        seal_traces.push(gen_seal_trace(&seal_input_of(witness, l), &a_seed, &t_mat)?);
    }


    let h = h_c.max(h_d).max(SEAL_ROWS);
    let cols = air.cols();
    let prep_w = air.prep_cols();
    let mut trace = vec![vec![Goldilocks::ZERO; cols]; h];
    let mut prep = vec![vec![Goldilocks::ZERO; prep_w]; h];


    let c_cols = air.custody.cols();
    let c_prep_w = air.custody.prep_cols();
    for r in 0..h_c {
        trace[r][..c_cols].copy_from_slice(&ct.trace[r]);
        prep[r][..c_prep_w].copy_from_slice(&ct.prep[r]);
    }
    if h > h_c {
        let reg = air.custody.reg_base();
        let reg_end = reg
            + INPUT_REG_W * air.custody.n_inputs
            + OUT_REG_W * air.custody.n_outputs
            + REV_REG_W * air.custody.n_cond()
            + BURN_REG_W * air.custody.n_burns
            + 8;
        for r in h_c..h {
            for c in reg..reg_end {
                trace[r][c] = trace[h_c - 1][c];
            }
        }
    }
    let d_col = air.delta.col_base;
    let d_prep = air.delta.prep_base;
    for r in 0..h_d {
        for (c, v) in dt.trace[r].iter().enumerate() {
            trace[r][d_col + c] = *v;
        }
        for (c, v) in dt.prep[r].iter().enumerate() {
            prep[r][d_prep + c] = *v;
        }
    }
    let s_prep = air.seal_prep_base;
    for r in 0..SEAL_ROWS {
        for (c, v) in seal_traces[0].prep[r].iter().enumerate() {
            prep[r][s_prep + c] = *v;
        }
    }
    for l in 0..n {
        let s_col = air.seal_col_base + SEAL_LEG_W * l;
        for r in 0..SEAL_ROWS {
            for (c, v) in seal_traces[l].trace[r].iter().enumerate() {
                trace[r][s_col + c] = *v;
            }
        }
    }
    place_type_flags(&mut prep, &air);
    place_composed_flags(&mut prep, &air);
    for l in 0..n {
        let e = canon.legs[l].expiry.as_u64();
        if e >= (1u64 << EXPIRY_BITS) {
            return Err(TxError::ExpiryTooLarge { leg: l, value: e });
        }
        fill_expiry_block(&mut trace, air.exp_col_base + EXP_STRIDE * l, e);
    }


    let mut publics = Vec::with_capacity(air.pub_count());
    publics.extend_from_slice(&ct.publics);
    for l in 0..n {
        publics.extend_from_slice(&seal_traces[l].publics);
    }
    for l in 0..n {
        let fee = canon.legs[l].fee.as_u64();
        publics.push(Goldilocks::from_u64_reduce(fee & MASK32));
        publics.push(Goldilocks::from_u64_reduce(fee >> 32));
    }
    for l in 0..n {
        publics.push(Goldilocks::from_u64_reduce(canon.legs[l].expiry.as_u64()));
    }
    Ok(TxTrace { trace, prep, publics, air })
}


// ---------------------------------------------------------------------------
// Verifier-side prep regeneration (public data only)
// ---------------------------------------------------------------------------


fn custody_prep(air: &TxAir) -> Vec<Vec<Goldilocks>> {
    let c = &air.custody;
    let n_in = c.n_inputs;
    let n_out = c.n_outputs;
    let n_cond = c.n_cond();
    let n_burns = c.n_burns;
    let rows = c.rows();
    let mut prep = vec![vec![Goldilocks::ZERO; c.prep_cols()]; rows];


    let cons = gen_cons_prep(n_in, n_out + n_burns, air.delta.n_legs());
    for r in 0..cons.len() {
        prep[r][CONS_PREP_BASE..CONS_PREP_BASE + 6].copy_from_slice(&cons[r]);
    }
    for i in 0..n_in {
        set_input_flags(&mut prep, i, 25 * i);
    }
    for o in 0..n_out {
        set_output_flags(&mut prep, o, n_in, 25 * n_in + 22 * o);
    }
    for k in 0..n_cond {
        set_revert_flags(&mut prep, k, n_in, n_out, 25 * n_in + 22 * n_out + 22 * k);
    }
    for k in 0..n_burns {
        set_burn_flags(
            &mut prep,
            k,
            n_in,
            n_out,
            n_cond,
            25 * n_in + 22 * n_out + 22 * n_cond + k,
        );
    }
    let mprep = MerkleChip::new(c.depth).gen_prep();
    for r in 0..mprep.len() {
        prep[r][..MERKLE_PREP_W].copy_from_slice(&mprep[r]);
    }


    let place = |prep: &mut Vec<Vec<Goldilocks>>, gw: usize, len: u8, flags: u32| {
        let bp = bl3_prep(len, 0, flags);
        for r in 0..66 {
            prep[66 * gw + r][BL3_PREP_BASE..BL3_PREP_BASE + BL3_PREP_W]
                .copy_from_slice(&bp[r]);
        }
    };
    let cm_windows = |prep: &mut Vec<Vec<Goldilocks>>, w0: usize| {
        for bi in 0..16 {
            let mut f = 0;
            if bi == 0 {
                f |= CHUNK_START;
            }
            if bi == 15 {
                f |= CHUNK_END;
            }
            place(prep, w0 + bi, 64, f);
        }
        for bi in 0..5 {
            let mut f = 0;
            if bi == 0 {
                f |= CHUNK_START;
            }
            if bi == 4 {
                f |= CHUNK_END;
            }
            let len = if bi < 4 { 64 } else { 15 };
            place(prep, w0 + 16 + bi, len, f);
        }
        place(prep, w0 + 21, 64, PARENT | ROOT);
    };
    for i in 0..n_in {
        let w0 = 25 * i;
        cm_windows(&mut prep, w0);
        place(&mut prep, w0 + 22, 64, CHUNK_START);
        place(&mut prep, w0 + 23, 7, CHUNK_END | ROOT);
        place(&mut prep, w0 + 24, 43, CHUNK_START | CHUNK_END | ROOT);
    }
    for o in 0..n_out {
        cm_windows(&mut prep, 25 * n_in + 22 * o);
    }
    for k in 0..n_cond {
        cm_windows(&mut prep, 25 * n_in + 22 * n_out + 22 * k);
    }
    for k in 0..n_burns {
        place(&mut prep, 25 * n_in + 22 * n_out + 22 * n_cond + k, 50, CHUNK_START | CHUNK_END | ROOT);
    }
    prep
}


#[allow(clippy::too_many_lines)]
fn delta_prep(air: &TxAir, w: &CodecW) -> Vec<Vec<Goldilocks>> {
    let d = &air.delta;
    let n = d.n_legs();
    let h_d = d.rows();
    let mut prep = vec![vec![Goldilocks::ZERO; d.prep_cols()]; h_d];
    let fr = ROWS_PER_LEG * n;
    for r in 0..h_d {
        prep[r][PV_DELTA_ROW] = Goldilocks::ONE;
    }
    for r in 0..fr {
        prep[r][delta_air::PV_DELTA_COPY] = Goldilocks::ONE;
    }
    prep[fr][PV_FINAL] = Goldilocks::ONE;
    for l in 0..n {
        let base = ROWS_PER_LEG * l;
        let leg = &d.legs[l];
        let n_val = leg.n_in + leg.n_out;
        for k in 0..8 {
            let r = base + k;
            prep[r][PV_VOL] = Goldilocks::ONE;
            if k == 0 {
                prep[r][PV_VOL0] = Goldilocks::ONE;
            }
            if k < leg.n_out {
                prep[r][PV_VOL_ACTIVE] = Goldilocks::ONE;
                prep[r][PV_TIE + TIE_STRIDE * l + TIE_VOL + k] = Goldilocks::ONE;
            }
        }
        for k in 0..16 {
            let r = base + 8 + k;
            prep[r][PV_LOG] = Goldilocks::ONE;
            if k == 0 {
                prep[r][PV_LOG0] = Goldilocks::ONE;
            }
            if k < n_val {
                prep[r][PV_LOG_ACTIVE] = Goldilocks::ONE;
                prep[r][PV_TIE + TIE_STRIDE * l + TIE_LOG + k] = Goldilocks::ONE;
            }
        }
        for j in 0..64 {
            for f_i in 0..12 {
                let r = base + 24 + 12 * j + f_i;
                prep[r][PF_MAC] = Goldilocks::ONE;
                if f_i == 0 {
                    prep[r][PF_MAC_START] = Goldilocks::ONE;
                }
                if f_i == 11 {
                    prep[r][PF_MAC_FINAL] = Goldilocks::ONE;
                    prep[r][PV_TIE + TIE_STRIDE * l + TIE_MACF + j] = Goldilocks::ONE;
                }
                prep[r][PF_SEL + f_i] = Goldilocks::ONE;
                for k in 0..224 {
                    let wv = w.weight(j, k);
                    prep[r][PF_WM + k] =
                        Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WS + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..4 {
                    let wv = w.weight(j, 224 + k);
                    prep[r][PF_WMF + k] =
                        Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WSF + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..5 {
                    let wv = w.weight(j, 228 + k);
                    prep[r][PF_WMT + k] =
                        Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WST + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..16 {
                    let wv = w.weight(j, 240 + k);
                    prep[r][PF_WMW + k] =
                        Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WSW + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
            }
        }
        for j in 0..64 {
            let r = base + 792 + j;
            prep[r][PV_DIGIT] = Goldilocks::ONE;
            prep[r][PV_TIE + TIE_STRIDE * l + delta_air::TIE_DIG + j] = Goldilocks::ONE;
        }
        prep[base + 855][PV_TIE + TIE_STRIDE * l + TIE_ZROW] = Goldilocks::ONE;
    }
    prep
}


fn seal_prep(epoch_pk: &PublicKey) -> Result<Vec<Vec<Goldilocks>>, SealError> {
    let a_seed = epoch_pk.a_seed();
    let a = expand_matrix(a_seed)?;
    let a_ntt: Mat8x8Ntt = a.ntt();
    let t_ntt = mat2x8_of(epoch_pk).ntt();
    let mut prep = vec![vec![Goldilocks::ZERO; SEAL_PREP]; SEAL_ROWS];
    for row in 0..SEAL_ROWS {
        let f = if row < 72 {
            crate::air::chips::seal_chip::F_IS_FWD
        } else if row < 152 {
            F_IS_MAC
        } else {
            F_IS_INV
        };
        prep[row][f] = Goldilocks::ONE;
    }
    for row in 0..SEAL_ROWS - 1 {
        prep[row][PV_SEAL_COPY] = Goldilocks::ONE;
    }
    for j in 0..8 {
        for l in 0..9 {
            let row = 9 * j + l;
            if l < 8 {
                prep[row][F_LEVEL + l] = Goldilocks::ONE;
            }
            if l == 0 {
                prep[row][F_IS_INPUT] = Goldilocks::ONE;
            }
            if l == 8 {
                prep[row][F_IS_T_OUT] = Goldilocks::ONE;
                prep[row][F_FWD_T + j] = Goldilocks::ONE;
            }
        }
    }
    for row in 72..152 {
        prep[row][F_MACJ + (row - 72) % 8] = Goldilocks::ONE;
        prep[row][F_MACI + (row - 72) / 8] = Goldilocks::ONE;
    }
    for t in 0..10 {
        for l in 0..9 {
            let row = 152 + 9 * t + l;
            prep[row][F_INV_T + t] = Goldilocks::ONE;
            if l < 8 {
                prep[row][F_LEVEL + l] = Goldilocks::ONE;
            }
            if l == 0 {
                prep[row][F_IS_INPUT] = Goldilocks::ONE;
            }
            if l == 8 {
                prep[row][F_IS_T_OUT] = Goldilocks::ONE;
            }
        }
    }
    for j in 0..8 {
        for l in 0..8 {
            let m = 1usize << l;
            for blk in 0..(SEAL_N / (2 * m)) {
                for jj in 0..m {
                    let c = 2 * m * blk + jj;
                    prep[9 * j + l][TW + c] =
                        Goldilocks::from_u64_reduce(fwd_twiddle(l, jj));
                }
            }
        }
    }
    for t in 0..10 {
        for l in 0..8 {
            let m = 128usize >> l;
            for blk in 0..(SEAL_N / (2 * m)) {
                for jj in 0..m {
                    let c = 2 * m * blk + jj;
                    prep[152 + 9 * t + l][TW + c] =
                        Goldilocks::from_u64_reduce(inv_twiddle(7 - l, jj));
                }
            }
        }
    }
    for blk in 0..10 {
        let is_v = blk >= 8;
        let idx = blk - 8 * usize::from(is_v);
        for jj in 0..8 {
            let row_i = 72 + 8 * blk + jj;
            let mop: &[u64; SEAL_N] = if is_v {
                t_ntt.row(idx)[jj].values()
            } else {
                a_ntt.row(idx)[jj].values()
            };
            for c in 0..SEAL_N {
                prep[row_i][MOP + c] = Goldilocks::from_u64_reduce(mop[c]);
            }
        }
    }
    Ok(prep)
}


/// The verifier's own preprocessed table: a pure function of the shell's
/// structure, the codec W, and the epoch key — no witness data. Pinned
/// against the prover's by test.
pub fn gen_tx_prep(
    air: &TxAir,
    w: &CodecW,
    epoch_pk: &PublicKey,
) -> Result<Vec<Vec<Goldilocks>>, SealError> {
    let h = air.rows();
    let mut prep = vec![vec![Goldilocks::ZERO; air.prep_cols()]; h];
    let cprep = custody_prep(air);
    for r in 0..cprep.len() {
        prep[r][..cprep[r].len()].copy_from_slice(&cprep[r]);
    }
    let dprep = delta_prep(air, w);
    let db = air.delta.prep_base;
    for r in 0..dprep.len() {
        for (c, v) in dprep[r].iter().enumerate() {
            prep[r][db + c] = *v;
        }
    }
    let sprep = seal_prep(epoch_pk)?;
    let sb = air.seal_prep_base;
    for r in 0..SEAL_ROWS {
        for (c, v) in sprep[r].iter().enumerate() {
            prep[r][sb + c] = *v;
        }
    }
    place_type_flags(&mut prep, air);
    place_composed_flags(&mut prep, air);
    Ok(prep)
}


// ---------------------------------------------------------------------------
// ct serialization binding
// ---------------------------------------------------------------------------


/// The leg's ct bytes as the u32-LE serialization of (u, v) — erratum 21's
/// wire format (u's 8 polynomials, then v's 2).
pub fn serialize_uv(publics: &[Goldilocks], air: &TxAir, leg: usize) -> Vec<u8> {
    let spb = air.seal_public_base + SEAL_PUBS_PER_LEG * leg;
    let mut out = Vec::with_capacity(10 * SEAL_N * 4);
    for i in 0..8 {
        for c in 0..SEAL_N {
            out.extend_from_slice(&(publics[spb + SEAL_N * i + c].as_u64() as u32).to_le_bytes());
        }
    }
    for k in 0..2 {
        for c in 0..SEAL_N {
            out.extend_from_slice(
                &(publics[spb + 8 * SEAL_N + SEAL_N * k + c].as_u64() as u32).to_le_bytes(),
            );
        }
    }
    out
}


pub fn check_ct_binding(
    canon: &TransactionShell,
    air: &TxAir,
    publics: &[Goldilocks],
) -> Result<(), usize> {
    for (l, leg) in canon.legs.iter().enumerate() {
        if leg.ct != serialize_uv(publics, air, l) {
            return Err(l);
        }
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransactionProof {
   pub proved: crate::stark::prover::Proved,
   pub publics: Vec<Goldilocks>,
}


/// The wire form: log_n ‖ width ‖ ComposedProof ‖ publics. The
/// preprocessed table never travels — the verifier regenerates it from
/// (shell structure, W, epoch key); decode yields an empty `prep`.
impl Encode for TransactionProof {
   fn encode_into(&self, out: &mut Vec<u8>) {
       out.extend_from_slice(&(self.proved.log_n as u32).to_le_bytes());
       out.extend_from_slice(&(self.proved.width as u32).to_le_bytes());
       self.proved.proof.encode_into(out);
       out.extend_from_slice(&(self.publics.len() as u32).to_le_bytes());
       for p in &self.publics {
           p.encode_into(out);
       }
   }
   fn encoded_len(&self) -> usize {
       4 + 4 + self.proved.proof.encoded_len() + 4 + 8 * self.publics.len()
   }
}


impl Decode for TransactionProof {
   fn decode_from(r: &mut Reader<'_>) -> Result<Self, CodecError> {
       let log_n = r.read_u32()? as usize;
       let width = r.read_u32()? as usize;
       if log_n == 0 || log_n > 32 || width == 0 {
           return Err(CodecError::InvariantViolated("transaction proof shape out of range"));
       }
       let proof = crate::stark::compose::ComposedProof::decode_from(r)?;
       let n = r.read_seq_len()?;
       if n > 1 << 24 {
           return Err(CodecError::SeqTooLarge { count: n, max: 1 << 24 });
       }
       let mut publics = Vec::with_capacity(n);
       for _ in 0..n {
           publics.push(Goldilocks::decode_from(r)?);
       }
       Ok(TransactionProof {
           proved: crate::stark::prover::Proved { proof, log_n, width, prep: Vec::new() },
           publics,
       })
   }
}



/// Prove a whole transaction. The transcript `t` must already carry the
/// statement-11 binding (`fs::bind_transaction` over the shell's
/// nullifiers, txid, shell digest, and `TxPublicInputs`) — the engine's
/// draws continue that chain, binding the proof to the full statement.
pub fn prove_transaction(
    fri: &FriShape,
    witness: &TransactionWitness,
    shell: &TransactionShell,
    w: &CodecW,
    epoch_pk: &PublicKey,
    t: &mut FsTranscript,
) -> Result<TransactionProof, TxError> {
    let tx = gen_tx_trace(witness, shell, w, epoch_pk)?;
    let canon = shell.canonicalize()?;
    if let Err(leg) = check_ct_binding(&canon, &tx.air, &tx.publics) {
        return Err(TxError::CtMismatch { leg });
    }
    let proved = Prover::new(*fri).prove(&tx.air, &tx.trace, &tx.prep, &tx.publics, t)?;
    Ok(TransactionProof { proved, publics: tx.publics })
}


/// Verify a whole transaction against the shell, the codec W, and the
/// epoch key (the verifier's own prep — a differing table rejects). The
/// transcript `t` must carry the same statement-11 binding as the prover's.
pub fn verify_transaction(
    fri: &FriShape,
    shell: &TransactionShell,
    w: &CodecW,
    epoch_pk: &PublicKey,
    proof: &TransactionProof,
    t: &mut FsTranscript,
) -> Result<bool, TxError> {
    let air = build_tx_air(shell)?;
    if proof.proved.width != air.cols() || proof.publics.len() != air.pub_count() {
        return Ok(false);
    }
    let h = air.rows();
    if proof.proved.log_n != h.max(2).next_power_of_two().ilog2() as usize {
        return Ok(false);
    }
    let canon = shell.canonicalize()?;
    for (l, leg) in canon.legs.iter().enumerate() {
        if leg.expiry.as_u64() >= (1u64 << EXPIRY_BITS) {
            return Err(TxError::ExpiryTooLarge { leg: l, value: leg.expiry.as_u64() });
        }
    }
    if check_ct_binding(&canon, &air, &proof.publics).is_err() {
        return Ok(false);
    }
    let prep = gen_tx_prep(&air, w, epoch_pk)?;
    Ok(Verifier::new(*fri).verify(&air, &proof.proved, &prep, &proof.publics, t)?)
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::custody_air::{CustodyWitness, InputWitness, OutputWitness};
    use crate::testutil::SplitMix64;
    use nerv_codec::features::{build_leg_features, LegKind, LegMovement};
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_core::hash::Hash256;
    use nerv_core::types::{FeeSats, Height, ShardSet, INTERVALS_PER_EPOCH};
    use nerv_custody::commitment::NoteOpening;
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::nullifier::derive_nullifier;
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use nerv_custody::{Address, MasterSeed, WalletKeys};
    use nerv_seal::digitize::digitize;
    use nerv_seal::encrypt::derive_reference_keypair;
    use nerv_seal::sampling::NoiseSeed;


    struct Fixture {
        shell: TransactionShell,
        custody: CustodyWitness,
        seeds: Vec<NoiseSeed>,
    }


    fn fixture(seed: u64) -> Fixture {
        let mut rng = SplitMix64::new(seed);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| Address::generate(det, wk.nullifier_key(), i, &g).unwrap();
        let opening = |v: u64, i: u64| NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let nanos = 1_000_000_000u64;
        let in1 = opening(50 * nanos, 0);
        let in2 = opening(35 * nanos, 1);
        let out_c = opening(25 * nanos, 2);
        let out_ch = opening(9_999 * nanos / 1000, 3);
        let out_b = opening(50 * nanos, 4);
        let mut tree = NoteCommitmentTree::new();
        for _ in 0..6 {
            tree.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
        }
        let i1 = tree.append(&in1.commitment().unwrap()).unwrap();
        let i2 = tree.append(&in2.commitment().unwrap()).unwrap();
        let root = tree.root();
        let sib = |i: u64| -> Vec<[Goldilocks; 4]> {
            tree.witness(i).unwrap().siblings.iter().map(|d| d.to_elements().unwrap()).collect()
        };
        let inputs = vec![
            InputWitness {
                opening: in1,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: i1,
                siblings: sib(i1),
                anchor: root,
            },
            InputWitness {
                opening: in2,
                nullifier_key: wk.nullifier_key_at(1),
                leaf_index: i2,
                siblings: sib(i2),
                anchor: root,
            },
        ];
        let nf1 = derive_nullifier(&inputs[0].nullifier_key, &inputs[0].opening.rho);
        let nf2 = derive_nullifier(&inputs[1].nullifier_key, &inputs[1].opening.rho);
        let mk = |o: &NoteOpening| Output {
            cm: o.commitment().unwrap(),
            sealed_note: vec![0xA5; 48],
            value: o.value,
            conditional: false,
            revert_cm: None,
        };
        let leg7 = LegShell {
            shard: g.ids()[7],
            inputs: InputSet::new(vec![nf1, nf2]),
            outputs: vec![mk(&out_c), mk(&out_ch)],
            fee: FeeSats::from_u64(600_000),
            anchor: root.as_hash256(),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![0x11; 100],
            burns: vec![],
        };
        let leg40 = LegShell {
            shard: g.ids()[40],
            inputs: InputSet::new(vec![]),
            outputs: vec![mk(&out_b)],
            fee: FeeSats::from_u64(400_000),
            anchor: Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![0x22; 100],
            burns: vec![],
        };
        Fixture {
            shell: TransactionShell { legs: vec![leg7, leg40] },
            custody: CustodyWitness {
                inputs,
                outputs: vec![
                    OutputWitness { opening: out_c },
                    OutputWitness { opening: out_ch },
                    OutputWitness { opening: out_b },
                ],
                reverts: vec![],
                burns: vec![],
                fees: vec![600_000, 400_000],
            },
            seeds: vec![
                NoiseSeed::from_bytes(rng.bytes32()),
                NoiseSeed::from_bytes(rng.bytes32()),
            ],
        }
    }


    fn w() -> CodecW {
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), nerv_codec::codec_w::WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }


    fn pk() -> PublicKey {
        derive_reference_keypair(&[0xE0; 32]).unwrap().0
    }


    fn gen(f: &Fixture) -> (CodecW, PublicKey, TransactionWitness) {
        let w = w();
        let pk = pk();
        let wit = TransactionWitness::generate(&f.shell, f.custody.clone(), &w, &pk, &f.seeds)
            .unwrap();
        (w, pk, wit)
    }


    struct NoSeal<'a>(&'a TxAir);
    impl<B: AirBuilder> Air<B> for NoSeal<'_> {
        fn eval(&self, b: &mut B) {
            self.0.custody.eval(b);
            self.0.delta.eval(b);
            self.0.bindings(b);
        }
    }


    struct SealOnly<'a>(&'a TxAir);
    impl<B: AirBuilder> Air<B> for SealOnly<'_> {
        fn eval(&self, b: &mut B) {
            let n = self.0.delta.n_legs();
            for l in 0..n {
                SealChip::at(
                    self.0.seal_col_base + SEAL_LEG_W * l,
                    self.0.seal_prep_base,
                    self.0.seal_public_base + SEAL_PUBS_PER_LEG * l,
                )
                .eval(b);
            }
            self.0.bindings(b);
        }
    }


    #[test]
    fn layout_pins() {
        let f = fixture(0x71);
        let air = build_tx_air(&f.shell).unwrap();
        let n = air.delta.n_legs();
        assert_eq!(air.custody.n_inputs, 2);
        assert_eq!(air.custody.n_outputs, 3);
        assert_eq!(air.rows(), air.custody.rows());
        assert!(air.rows() > air.delta.rows());
        assert!(air.rows() > SEAL_ROWS);
        assert_eq!(
            air.cols(),
            air.custody.cols() + air.delta.cols() + SEAL_LEG_W * n + EXP_STRIDE * n
        );
        assert_eq!(air.prep_cols(), air.tx_prep_base + 5 * n);
        assert_eq!(air.seal_prep_base, air.custody.prep_cols() + air.delta.prep_cols());
        assert_eq!(air.seal_prep_base + SEAL_PREP, air.tx_prep_base);
        assert_eq!(
            air.pub_count(),
            4 * 2 + 8 * 2 + 8 * 3 + 8 * 2 + 8 * 0 + SEAL_PUBS_PER_LEG * n + 3 * n
        );
        assert_eq!(SEAL_PUBS_PER_LEG, 2560);
    }


    #[test]
    fn full_differential() {
        let f = fixture(0x7A1);
        let (w, pk, wit) = gen(&f);
        let tx = gen_tx_trace(&wit, &f.shell, &w, &pk).unwrap();


        let a = NativeEval::check_with_prep(
            tx.trace.clone(),
            tx.prep.clone(),
            tx.publics.clone(),
            &NoSeal(&tx.air),
            8,
        );
        assert!(a.is_ok(), "custody+delta+bindings: {a:?}");


        let slice: Vec<Vec<Goldilocks>> = tx.trace[..SEAL_ROWS].to_vec();
        let pslice: Vec<Vec<Goldilocks>> = tx.prep[..SEAL_ROWS].to_vec();
        let b = NativeEval::check_with_prep(
            slice,
            pslice,
            tx.publics.clone(),
            &SealOnly(&tx.air),
            8,
        );
        assert!(b.is_ok(), "seal+bindings: {b:?}");


        // DREG == w.apply(features); PT == digitize(delta).
        for l in 0..2 {
            let delta = &wit.delta.legs[l].delta;
            let e = tx.air.delta.col_base + LEG_W * l;
            let zrow = ROWS_PER_LEG * l + 855;
            for j in 0..64 {
                assert_eq!(tx.trace[zrow][e + delta_air::R_DREG + 2 * j].as_u64(), delta.0[j] & MASK32);
                assert_eq!(tx.trace[zrow][e + delta_air::R_DREG + 2 * j + 1].as_u64(), delta.0[j] >> 32);
            }
            let pt = digitize(&delta.0);
            for s in 0..512 {
                assert_eq!(tx.trace[zrow][e + R_PT + s].as_u64(), u64::from(pt.digits()[s]));
            }
        }


        // Publics: u,v == witness's; fees; expiries; bucket/kind pins.
        for l in 0..2 {
            let spb = tx.air.seal_public_base + SEAL_PUBS_PER_LEG * l;
            for i in 0..8 {
                for c in 0..SEAL_N {
                    assert_eq!(
                        tx.publics[spb + SEAL_N * i + c].as_u64(),
                        wit.seal[l].u.poly(i).coefficients()[c]
                    );
                }
            }
            for k in 0..2 {
                for c in 0..SEAL_N {
                    assert_eq!(
                        tx.publics[spb + 8 * SEAL_N + SEAL_N * k + c].as_u64(),
                        wit.seal[l].v.poly(k).coefficients()[c]
                    );
                }
            }
            let fee = f.custody.fees[l];
            assert_eq!(
                tx.publics[tx.air.fee_pub_base + 2 * l].as_u64(),
                fee & MASK32
            );
            assert_eq!(tx.publics[tx.air.fee_pub_base + 2 * l + 1].as_u64(), fee >> 32);
            assert_eq!(
                tx.publics[tx.air.expiry_pub_base + l].as_u64(),
                5_000
            );
        }
        // expiry 5000: rem 5000, bucket 0 — R_TIME[0] set, others clear.
        let e0 = tx.air.delta.col_base;
        for t in 0..16 {
            let want = Goldilocks::from_u32(u32::from(t == 0));
            assert_eq!(tx.trace[100][e0 + R_TIME + t], want, "t={t}");
        }
        // kinds: leg 0 CrossShardSpend (1), leg 1 CrossShardIssue (2).
        assert_eq!(tx.trace[100][e0 + R_TYPE + 1], Goldilocks::ONE);
        assert_eq!(tx.trace[100][e0 + R_TYPE + 2], Goldilocks::ONE);
        for t in [0usize, 3, 4] {
            assert_eq!(tx.trace[100][e0 + R_TYPE + t], Goldilocks::ZERO);
        }
        let e1 = e0 + LEG_W;
        for t in [0usize, 1, 3, 4] {
            assert_eq!(tx.trace[100][e1 + R_TYPE + t], Goldilocks::ZERO);
        }
    }


    #[test]
    fn verifier_prep_matches_prover_prep() {
        let f = fixture(0x7B);
        let (w, pk, wit) = gen(&f);
        let tx = gen_tx_trace(&wit, &f.shell, &w, &pk).unwrap();
        let vp = gen_tx_prep(&tx.air, &w, &pk).unwrap();
        assert_eq!(vp, tx.prep, "the verifier's prep regeneration must be exact");
    }


    #[test]
    fn seal_prep_is_leg_independent() {
        let f = fixture(0x7C);
        let (_w, pk, wit) = gen(&f);
        let a_seed = *pk.a_seed();
        let t_mat = mat2x8_of(&pk);
        let s0 = gen_seal_trace(&seal_input_of(&wit, 0), &a_seed, &t_mat).unwrap();
        let s1 = gen_seal_trace(&seal_input_of(&wit, 1), &a_seed, &t_mat).unwrap();
        assert_eq!(s0.prep, s1.prep);
        // and equal to the standalone replication
        let sp = seal_prep(&pk).unwrap();
        assert_eq!(sp, s0.prep);
    }


    #[test]
    fn bucket_derivation_pins() {
        for (e, want) in [
            (0u64, 0usize),
            (5399, 0),
            (5400, 1),
            (5000, 0),
            (86399, 15),
            (86400, 0),
            (86400 + 5400, 1),
            (10 * 86400 + 86399, 15),
        ] {
            let mut trace = vec![vec![Goldilocks::ZERO; EXP_STRIDE]; 1];
            fill_expiry_block(&mut trace, 0, e);
            let bucket = (0..16).position(|t| trace[0][EXP_BF + t] == Goldilocks::ONE).unwrap();
            assert_eq!(bucket, want, "e={e}");
            let rem = e % C_EPOCH;
            let q = e / C_EPOCH;
            assert_eq!(trace[0][EXP_Q].as_u64(), q);
            assert_eq!(trace[0][EXP_REM].as_u64(), rem);
            assert_eq!(trace[0][EXP_XB].as_u64(), 16 * rem - want as u64 * C_EPOCH);
            assert_eq!(trace[0][EXP_YB].as_u64(), (want as u64 + 1) * C_EPOCH - 16 * rem - 1);
        }
    }


    #[test]
    fn tamper_battery() {
        let f = fixture(0x7A3);
        let (w, pk, wit) = gen(&f);
        let tx = gen_tx_trace(&wit, &f.shell, &w, &pk).unwrap();
        let check_a = |trace: Vec<Vec<Goldilocks>>, publics: &[Goldilocks]| {
            NativeEval::check_with_prep(
                trace,
                tx.prep.clone(),
                publics.to_vec(),
                &NoSeal(&tx.air),
                4,
            )
        };
        let check_b = |trace: &[Vec<Goldilocks>]| {
            NativeEval::check_with_prep(
                trace[..SEAL_ROWS].to_vec(),
                tx.prep[..SEAL_ROWS].to_vec(),
                tx.publics.clone(),
                &SealOnly(&tx.air),
                4,
            )
        };
        assert!(check_a(tx.trace.clone(), &tx.publics).is_ok());
        assert!(check_b(&tx.trace).is_ok());


        let e0 = tx.air.delta.col_base;
        let s0 = tx.air.seal_col_base;


        let mut bad = tx.trace.clone();
        bad[100][e0 + delta_air::R_DREG] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "dreg");


        let mut bad = tx.trace.clone();
        bad[100][e0 + R_PT] += Goldilocks::ONE;
        assert!(check_a(bad.clone(), &tx.publics).is_err(), "pt threading");
        assert!(check_b(&bad).is_err(), "pt m-binding");


        let mut bad = tx.trace.clone();
        bad[232][s0 + AX_MB + 3] += Goldilocks::ONE;
        assert!(check_b(&bad).is_err(), "seal m digit");


        let mut bad = tx.trace.clone();
        bad[0][e0 + FEEB + 9] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "fee bits");


        let mut bad = tx.trace.clone();
        bad[5][tx.air.exp_col_base + EXP_EB + 3] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "expiry bits");


        let mut bad = tx.trace.clone();
        bad[5][tx.air.exp_col_base + EXP_BF] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "bucket one-hot");


        let mut bad = tx.trace.clone();
        bad[5][e0 + R_TIME] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "time copy");


        let mut bad = tx.trace.clone();
        bad[5][e0 + R_TYPE] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "type copy");


        let mut bad = tx.trace.clone();
        bad[100][tx.air.custody.reg_base()] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "custody register");


        let inr = LEG_W * 2 + tx.air.delta.col_base - tx.air.delta.col_base;
        let mut bad = tx.trace.clone();
        let inr_col = tx.air.delta.col_base + LEG_W * 2 + 0;
        bad[10][inr_col] += Goldilocks::ONE;
        assert!(check_a(bad, &tx.publics).is_err(), "inr boundary");


        let mut bad = tx.trace.clone();
        bad[3][s0] += Goldilocks::ONE;
        assert!(check_b(&bad).is_err(), "seal state");


        let mut pubs = tx.publics.clone();
        pubs[16] += Goldilocks::ONE;
        assert!(check_a(tx.trace.clone(), &pubs).is_err(), "public nf word");


        let mut pubs = tx.publics.clone();
        pubs[tx.air.fee_pub_base] += Goldilocks::ONE;
        assert!(check_a(tx.trace.clone(), &pubs).is_err(), "public fee word");
    }


    #[test]
    fn ct_binding_check() {
        let f = fixture(0x7D);
        let (w, pk, wit) = gen(&f);
        let tx = gen_tx_trace(&wit, &f.shell, &w, &pk).unwrap();
        let mut canon = f.shell.canonicalize().unwrap();
        for l in 0..2 {
            canon.legs[l].ct = serialize_uv(&tx.publics, &tx.air, l);
        }
        assert!(check_ct_binding(&canon, &tx.air, &tx.publics).is_ok());
        canon.legs[1].ct[10] ^= 1;
        assert_eq!(check_ct_binding(&canon, &tx.air, &tx.publics), Err(1));
        canon.legs[1].ct.clear();
        assert_eq!(check_ct_binding(&canon, &tx.air, &tx.publics), Err(1));
    }


    #[test]
    fn prove_rejects_ct_mismatch_and_oversized_expiry() {
        let f = fixture(0x7E);
        let (w, pk, wit) = gen(&f);
        let mut t = FsTranscript::new();
        let fri = FriShape {
            log_blowup: 4,
            num_queries: 56,
            log_final_poly_len: 1,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        };
        // The fixture's dummy ct cannot serialize (u, v): rejected BEFORE
        // any engine work.
        assert!(matches!(
            prove_transaction(&fri, &wit, &f.shell, &w, &pk, &mut t),
            Err(TxError::CtMismatch { leg: 0 })
        ));


        let mut shell = f.shell.clone();
        shell.legs[0].expiry = Height::from_u64(1u64 << 40);
        let mut t2 = FsTranscript::new();
        assert!(matches!(
            gen_tx_trace(&wit, &shell, &w, &pk),
            Err(TxError::ExpiryTooLarge { leg: 0, .. })
        ));
        let _ = &mut t2;
    }


    #[test]
    fn movement_kinds_match_derivation() {
        // The fixture's movements are exactly what leg_kind_index derives.
        let f = fixture(0x7F);
        let air = build_tx_air(&f.shell).unwrap();
        assert_eq!(leg_kind_index(&air, 0), 1); // CrossShardSpend
        assert_eq!(leg_kind_index(&air, 1), 2); // CrossShardIssue
        let _ = LegKind::CrossShardSpend;
        let _ = INTERVALS_PER_EPOCH;
    }
}
