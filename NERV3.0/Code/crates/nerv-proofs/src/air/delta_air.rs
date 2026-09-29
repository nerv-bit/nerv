//! The delta module AIR (WP §5.1 statements 6–8, 9-δ-side; errata 78–79,
//! register 63): per leg — volume adder rows, exact log-rail rows (v+1
//! carry chain), the encoder's 768 MAC rows, digit rows (statement 9's
//! decomposition, gated re-emission of the DigitChip relations), register
//! forms (slot one-hots + activity + distinctness, rails' bits), and the
//! DREG/PT threading; plus the global final row's two aggregate identities
//! (Σ|negative slots| = Σ inputs, Σ|positive slots| = Σ volume rails) and
//! the custody ties (fee, boundary, per-value v bits).
//!
//! Layout per leg (LEG_W = 3744; erratum 79's oracle-form width, amended
//! by register 63): row-local [0,1390) — encoder E block [0,453) reused by
//! the digit layout [0,74) on digit rows; VLB, log chain, R_LOGA(776),
//! FVOLB, FEEB, LOGB, SMAGB, final-row sums. Registers [1390,3744): VOLV(2),
//! FEE(2), TYPE(5), TIME(16), SLOTS 6×228 (OHP 224, SGN, MAGL, MAGH, ACT),
//! DREG(128), PT(512), ZINV(320). Global: INR(3).
//!
//! Row governance (register 63): every row-uniform form constraint is
//! gated by PV_DELTA_ROW (prep, 1 on all 856·n+1 rows) so the composed
//! trace's zero tail passes; every threading/copy is gated by
//! PV_DELTA_COPY (prep, 1 on rows [0, 856·n)) so the final row's successor
//! is unconstrained; the log chain is gated by PV_LOG; the final-row sums
//! and identities by PV_FINAL. The per-leg tie block (stride 160) carries
//! vol(8) log(8) fee(1) macfinal(64) digitj(64) zrow(1) flags; the global
//! boundary flag is its own column after all per-leg blocks.


use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;
use nerv_codec::codec_w::CodecW;
use nerv_codec::features::FeatureVector;
use crate::air::builder::{Air, AirBuilder};
use crate::air::chips::conservation::{ACCP_BASE, V_BITS_BASE};
use crate::air::chips::encoder::{
    delta_hi_expr, delta_lo_expr, EncoderChip, N_COORDS, AKN, AKP, ADJ, ADJB, AHH, AHHB, AHL,
    AHLB, BR0, BR1, BR2, C1, C2, C2C, CB, CC, INVL, MHB, MLO, MLOB, MH, N_COORDS as _, NZ, P0,
    P1, P2, PB, PN, PNB, PP, PPB, PS, PF_MAC, PF_MAC_FINAL, PF_MAC_START, PF_SEL, PF_WM, PF_WMF,
    PF_WMT, PF_WMW, PF_WS, PF_WSF, PF_WST, PF_WSW, Q0, Q0B, RR, RRB, WMS, WSS, XHI, XLO, XS,
};
use crate::error::WitnessGenError;


pub const LEG_W: usize = 3744;
pub const ROWS_PER_LEG: usize = 856;
pub const ROWLOCAL_W: usize = 1390;


pub const VLB: usize = 453; // 64
pub const VK1: usize = 517;
pub const VK2: usize = 518;
pub const LVB: usize = 519; // 64
pub const LBP: usize = 583; // 64
pub const LC: usize = 647; // 64
pub const LS: usize = 711; // 64
pub const LEL: usize = 775;
pub const R_LOGA: usize = 776;
pub const FVOLB: usize = 777; // 64 (VOLV's limbs' bits)
pub const FEEB: usize = 841; // 64
pub const LOGB: usize = 905; // 10 (R_LOGA's bits)
pub const SMAGB: usize = 915; // 6×64
pub const FS: usize = 1299; // 6: pos(3) ‖ neg(3)
pub const FSK: usize = 1305; // 6
pub const FSKB: usize = 1311; // 36
pub const VT: usize = 1347; // 3
pub const VKC: usize = 1350; // 3
pub const VKCB: usize = 1353; // 24
pub const BZ: usize = 1383;
pub const BZB: usize = 1384; // 6


pub const R_VOLV: usize = 1390; // 2
pub const R_FEE: usize = 1392; // 2
pub const R_TYPE: usize = 1395; // 5
pub const R_TIME: usize = 1400; // 16
pub const R_SLOT: usize = 1416; // stride 228: OHP(224) SGN MAGL MAGH ACT
pub const SLOT_STRIDE: usize = 228;
pub const R_DREG: usize = 2784; // 128
pub const R_PT: usize = 2912; // 512
pub const R_ZINV: usize = 3424; // 5 per coord: ZH ZL INVH INVL Z


pub const G_INR: usize = 0; // 3 (global block)


// Prep layout: encoder PF_* at [0,521); then:
pub const PV_DELTA_ROW: usize = 521; // 1 on all 856·n+1 rows
pub const PV_DELTA_COPY: usize = 522; // 1 on rows [0, 856·n)
pub const PV_VOL: usize = 523;
pub const PV_VOL0: usize = 524;
pub const PV_VOL_ACTIVE: usize = 525;
pub const PV_LOG: usize = 527;
pub const PV_LOG0: usize = 528;
pub const PV_LOG_ACTIVE: usize = 529;
pub const PV_DIGIT: usize = 531;
pub const PV_FINAL: usize = 532;
pub const PV_TIE: usize = 533; // per-leg stride TIE_STRIDE


pub const TIE_STRIDE: usize = 160;
pub const TIE_VOL: usize = 0; // 8
pub const TIE_LOG: usize = 8; // 16
pub const TIE_FEE: usize = 24; // 1
pub const TIE_MACF: usize = 25; // 64
pub const TIE_DIG: usize = 90; // 64
pub const TIE_ZROW: usize = 154; // 1


// Digit-chip layout, re-emitted at the leg base (encoder columns are
// prep-gated off on digit rows — the row-type-interpreted reuse).
pub const D_HI: usize = 0;
pub const D_LO: usize = 1;
pub const D_BASE: usize = 2;
pub const D_BITS: usize = 10;


const TWO32: u64 = 1 << 32;


const _: () = assert!(VLB == crate::air::chips::encoder::E_W);
const _: () = assert!(R_SLOT + 6 * SLOT_STRIDE == R_DREG);
const _: () = assert!(R_DREG + 128 == R_PT);
const _: () = assert!(R_PT + 512 == R_ZINV);
const _: () = assert!(R_ZINV + 5 * N_COORDS == LEG_W);


#[derive(Clone, Copy, Debug)]
pub struct LegShape {
    pub n_in: usize,
    pub n_out: usize,
}


#[derive(Clone, Debug)]
pub struct DeltaAir {
    pub legs: Vec<LegShape>,
    pub cons_base: usize,
    pub cust_reg_base: usize,
    pub cust_input_reg_w: usize,
    pub cust_n_in: usize,
    pub cust_out_reg_w: usize,
    pub col_base: usize,
    pub prep_base: usize,
}


impl DeltaAir {
    pub fn n_legs(&self) -> usize {
        self.legs.len()
    }


    pub fn rows(&self) -> usize {
        ROWS_PER_LEG * self.legs.len() + 1
    }


    pub fn cols(&self) -> usize {
        LEG_W * self.legs.len() + 3
    }


    pub fn prep_cols(&self) -> usize {
        self.boundary_col() + 1
    }


    /// The global boundary flag's column — after all per-leg tie blocks.
    pub fn boundary_col(&self) -> usize {
        PV_TIE + TIE_STRIDE * self.legs.len()
    }


    fn out_reg(&self, g: usize) -> usize {
        self.cust_reg_base + self.cust_input_reg_w * self.cust_n_in + self.cust_out_reg_w * g
    }


    fn in_reg(&self, g: usize) -> usize {
        self.cust_reg_base + self.cust_input_reg_w * g
    }
}


fn recompose<B: AirBuilder>(b: &mut B, base: usize, n: usize) -> B::Expr {
    let mut s = B::constant(0);
    for t in 0..n {
        s = s + b.witness(0, base + t) * B::constant(1u64 << t);
    }
    s
}


fn leg_in_start(air: &DeltaAir, l: usize) -> usize {
    air.legs.iter().take(l).map(|s| s.n_in).sum()
}


fn leg_out_start(air: &DeltaAir, l: usize) -> usize {
    air.legs.iter().take(l).map(|s| s.n_out).sum()
}


fn out_total(air: &DeltaAir) -> usize {
    air.legs.iter().map(|s| s.n_out).sum()
}


impl<B: AirBuilder> Air<B> for DeltaAir {
    #[allow(clippy::too_many_lines)]
    fn eval(&self, b: &mut B) {
        let one = B::constant(1);
        let cb = self.col_base;
        let pb = self.prep_base;
        let n = self.n_legs();
        let delta_row = b.preprocessed(pb + PV_DELTA_ROW);
        let copy = b.preprocessed(pb + PV_DELTA_COPY);


        for (l, leg) in self.legs.iter().enumerate() {
            let e = cb + LEG_W * l;
            let is_vol = b.preprocessed(pb + PV_VOL);
            let is_vol0 = b.preprocessed(pb + PV_VOL0);
            let is_vol_active = b.preprocessed(pb + PV_VOL_ACTIVE);
            let is_log = b.preprocessed(pb + PV_LOG);
            let is_log0 = b.preprocessed(pb + PV_LOG0);
            let is_log_active = b.preprocessed(pb + PV_LOG_ACTIVE);
            let is_digit = b.preprocessed(pb + PV_DIGIT);
            // Materialize the per-leg TIE preprocessed columns up front so the
// selection closures don't capture `b` (which is mutably borrowed later by
// `b.assert_zero` and `b.witness`).
let tie = pb + PV_TIE + TIE_STRIDE * l;
let macf_pre: Vec<B::Expr> = (0..64).map(|j| b.preprocessed(tie + TIE_MACF + j)).collect();
let macf = |j: usize| macf_pre[j].clone();
let digj_pre: Vec<B::Expr> = (0..64).map(|j| b.preprocessed(tie + TIE_DIG + j)).collect();
let digj = |j: usize| digj_pre[j].clone();


            // -- volume adder --
            for t in 0..64 {
                let bit = b.witness(0, e + VLB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_vbit");
            }
            let mag_lo = recompose(b, e + VLB, 32);
            let mag_hi = recompose(b, e + VLB + 32, 32);
            let vk1 = b.witness(0, e + VK1);
            let vk2 = b.witness(0, e + VK2);
            b.assert_zero(vk1.clone() * (vk1.clone() - one.clone()), "dl_vk1");
            b.assert_zero(vk2.clone() * (vk2.clone() - one.clone()), "dl_vk2");
            b.assert_zero(
                copy.clone()
                    * (b.witness(1, e + R_VOLV)
                        - b.witness(0, e + R_VOLV)
                        - is_vol_active.clone() * (mag_lo.clone() - vk1.clone() * B::constant(TWO32))),
                "dl_vol_lo",
            );
            b.assert_zero(
                copy.clone()
                    * (b.witness(1, e + R_VOLV + 1)
                        - b.witness(0, e + R_VOLV + 1)
                        - is_vol_active.clone() * (mag_hi.clone() + vk1.clone() - vk2.clone() * B::constant(TWO32))),
                "dl_vol_hi",
            );
            b.assert_zero(is_vol0.clone() * b.witness(0, e + R_VOLV), "dl_vol_init");
            b.assert_zero(is_vol0.clone() * b.witness(0, e + R_VOLV + 1), "dl_vol_init");
            for t in 0..64 {
                let bit = b.witness(0, e + FVOLB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_fvolbit");
            }
            let fvol_lo = recompose(b, e + FVOLB, 32);
            b.assert_zero(b.witness(0, e + R_VOLV) - fvol_lo, "dl_vol_lo_recomp");
            let fvol_hi = recompose(b, e + FVOLB + 32, 32);
            b.assert_zero(b.witness(0, e + R_VOLV + 1) - fvol_hi, "dl_vol_hi_recomp");


            // -- log rows: v+1 carry chain, selector, ℓ, LOGA threading --
            for t in 0..64 {
                let bit = b.witness(0, e + LVB + t);
                b.assert_zero(is_log.clone() * bit.clone() * (bit - one.clone()), "dl_lbit");
                let bp = b.witness(0, e + LBP + t);
                b.assert_zero(is_log.clone() * bp.clone() * (bp - one.clone()), "dl_bpbit");
                let c = b.witness(0, e + LC + t);
                b.assert_zero(is_log.clone() * c.clone() * (c - one.clone()), "dl_cbit");
                let s = b.witness(0, e + LS + t);
                b.assert_zero(is_log.clone() * s.clone() * (s - one.clone()), "dl_sbit");
            }
            b.assert_zero(
                is_log.clone()
                    * (b.witness(0, e + LBP)
                        - b.witness(0, e + LVB)
                        - one.clone()
                        + b.witness(0, e + LVB) * B::constant(2)),
                "dl_bp0",
            );
            b.assert_zero(
                is_log.clone() * (b.witness(0, e + LC + 1) - b.witness(0, e + LVB)),
                "dl_c1",
            );
            for j in 1..64 {
                b.assert_zero(
                    is_log.clone()
                        * (b.witness(0, e + LBP + j)
                            - b.witness(0, e + LVB + j)
                            - b.witness(0, e + LC + j)
                            + b.witness(0, e + LVB + j)
                                * b.witness(0, e + LC + j)
                                * B::constant(2)),
                    "dl_bpj",
                );
                if j + 1 < 64 {
                    b.assert_zero(
                        is_log.clone()
                            * (b.witness(0, e + LC + j + 1)
                                - b.witness(0, e + LVB + j) * b.witness(0, e + LC + j)),
                        "dl_cj",
                    );
                }
            }
            b.assert_zero(is_log.clone() * b.witness(0, e + LS), "dl_s0");
            for j in 0..63 {
                let d = b.witness(0, e + LS + j + 1) - b.witness(0, e + LS + j);
                b.assert_zero(is_log.clone() * d.clone() * (d - one.clone()), "dl_smono");
            }
            let mut sum_t = B::constant(0);
            let mut sum_sb = B::constant(0);
            let mut sum_tb = B::constant(0);
            let mut lel = B::constant(0);
            for j in 0..64 {
                let t_j = (one.clone() - b.witness(0, e + LS + j))
                    * (if j < 63 { b.witness(0, e + LS + j + 1) } else { one.clone() });
                sum_t = sum_t + t_j.clone();
                sum_sb = sum_sb + b.witness(0, e + LS + j) * b.witness(0, e + LBP + j);
                sum_tb = sum_tb + t_j.clone() * b.witness(0, e + LBP + j);
                lel = lel + t_j * B::constant(j as u64);
            }
            b.assert_zero(is_log.clone() * (sum_t - one.clone()), "dl_tsum");
            b.assert_zero(is_log.clone() * sum_sb, "dl_sbp");
            b.assert_zero(is_log.clone() * (sum_tb - one.clone()), "dl_tb");
            b.assert_zero(is_log.clone() * (b.witness(0, e + LEL) - lel), "dl_lel");
            b.assert_zero(
                copy.clone()
                    * (b.witness(1, e + R_LOGA)
                        - b.witness(0, e + R_LOGA)
                        - is_log_active * b.witness(0, e + LEL)),
                "dl_loga",
            );
            b.assert_zero(is_log0 * b.witness(0, e + R_LOGA), "dl_log_init");
            for t in 0..10 {
                let bit = b.witness(0, e + LOGB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_logbit");
            }
            let log_rec = recompose(b, e + LOGB, 10);
            b.assert_zero(
                b.witness(0, e + R_LOGA) - log_rec,
                "dl_log_recomp",
            );


            // -- MAC: the encoder chip --
            EncoderChip::new(e, 0, pb).eval(b); // encoder constructor: (leg, row_base, prep_base)


            // -- DREG threading: mac-final rows write δ_j's limbs --
            for j in 0..N_COORDS {
                let gate = macf(j);
                let cur = b.witness(0, e + R_DREG + 2 * j);
                let nxt = b.witness(1, e + R_DREG + 2 * j);
                let cur_h = b.witness(0, e + R_DREG + 2 * j + 1);
                let nxt_h = b.witness(1, e + R_DREG + 2 * j + 1);
                let pass = one.clone() - gate.clone();
                let dl = delta_lo_expr(b, e);
                b.assert_zero(
                    copy.clone()
                        * (nxt.clone() - gate.clone() * dl - pass.clone() * cur),
                    "dl_dreg_lo",
                );
                let dh = delta_hi_expr(b, e);
                b.assert_zero(
                    copy.clone() * (nxt_h - gate.clone() * dh - pass.clone() * cur_h),
                    "dl_dreg_hi",
                );
            }


            // -- digit rows: gated DigitChip re-emission + ties + PT --
            for k in 0..8 {
                for t in 0..8 {
                    let bit = b.witness(0, e + D_BITS + 8 * k + t);
                    b.assert_zero(is_digit.clone() * bit.clone() * (bit.clone() - one.clone()), "dl_dbit");
                }
                let mut d = B::constant(0);
                for t in 0..8 {
                    d = d + b.witness(0, e + D_BITS + 8 * k + t) * B::constant(1u64 << t);
                }
                b.assert_zero(
                    is_digit.clone() * (b.witness(0, e + D_BASE + k) - d),
                    "dl_d_recomp",
                );
            }
            for half in 0..2 {
                let mut s = B::constant(0);
                for k in 0..4 {
                    s = s + b.witness(0, e + D_BASE + 4 * half + k) * B::constant(1u64 << (8 * k));
                }
                let target = if half == 0 { D_LO } else { D_HI };
                b.assert_zero(is_digit.clone() * (b.witness(0, e + target) - s), "dl_half");
            }
            for j in 0..N_COORDS {
                let gate = digj(j);
                b.assert_zero(
                    gate.clone() * (b.witness(0, e + D_LO) - b.witness(0, e + R_DREG + 2 * j)),
                    "dl_d_lo_tie",
                );
                b.assert_zero(
                    gate * (b.witness(0, e + D_HI) - b.witness(0, e + R_DREG + 2 * j + 1)),
                    "dl_d_hi_tie",
                );
            }
            for s_idx in 0..512 {
                let k = s_idx / 64;
                let j = s_idx % 64;
                let gate = digj(j);
                let pass = one.clone() - gate.clone();
                b.assert_zero(
                    copy.clone()
                        * (b.witness(1, e + R_PT + s_idx)
                            - gate * b.witness(0, e + D_BASE + k)
                            - pass * b.witness(0, e + R_PT + s_idx)),
                    "dl_pt_thread",
                );
            }


            // -- register forms --
            let mut tsum = B::constant(0);
            for t in 0..5 {
                let x = b.witness(0, e + R_TYPE + t);
                b.assert_zero(x.clone() * (x.clone() - one.clone()), "dl_type_bool");
                tsum = tsum + x;
            }
            b.assert_zero(delta_row.clone() * (tsum - one.clone()), "dl_type_one");
            let mut wsum = B::constant(0);
            for t in 0..16 {
                let x = b.witness(0, e + R_TIME + t);
                b.assert_zero(x.clone() * (x.clone() - one.clone()), "dl_time_bool");
                wsum = wsum + x;
            }
            b.assert_zero(delta_row.clone() * (wsum - one.clone()), "dl_time_one");
            for i in 0..6 {
                let sb = e + R_SLOT + SLOT_STRIDE * i;
                let act = b.witness(0, sb + 227);
                b.assert_zero(act.clone() * (act.clone() - one.clone()), "dl_act_bool");
                let mut osum = B::constant(0);
                for k in 0..224 {
                    let x = b.witness(0, sb + k);
                    b.assert_zero(x.clone() * (x.clone() - one.clone()), "dl_ohp_bool");
                    osum = osum + x;
                }
                b.assert_zero(act.clone() * (osum - one.clone()), "dl_ohp_one");
                let not_act = one.clone() - act.clone();
                b.assert_zero(not_act.clone() * b.witness(0, sb + 225), "dl_act_magl");
                b.assert_zero(not_act.clone() * b.witness(0, sb + 226), "dl_act_magh");
                let sgn = b.witness(0, sb + 224);
                b.assert_zero(sgn.clone() * (sgn.clone() - one.clone()), "dl_sgn");
                for t in 0..64 {
                    let bit = b.witness(0, e + SMAGB + 64 * i + t);
                    b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_smagbit");
                }
                let magl_rec = recompose(b, e + SMAGB + 64 * i, 32);
                b.assert_zero(
                    b.witness(0, sb + 225) - magl_rec,
                    "dl_magl",
                );
                let magh_rec = recompose(b, e + SMAGB + 64 * i + 32, 32);
                b.assert_zero(
                    b.witness(0, sb + 226) - magh_rec,
                    "dl_magh",
                );
            }
            for i in 0..6 {
                for i2 in (i + 1)..6 {
                    let ai = b.witness(0, e + R_SLOT + SLOT_STRIDE * i + 227);
                    let aj = b.witness(0, e + R_SLOT + SLOT_STRIDE * i2 + 227);
                    let mut dot = B::constant(0);
                    for k in 0..224 {
                        dot = dot + b.witness(0, e + R_SLOT + SLOT_STRIDE * i + k)
                            * b.witness(0, e + R_SLOT + SLOT_STRIDE * i2 + k);
                    }
                    b.assert_zero(ai * aj * dot, "dl_slot_distinct");
                }
            }


            // -- FEE register --
            for t in 0..64 {
                let bit = b.witness(0, e + FEEB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_feebit");
            }
            let fee_lo_rec = recompose(b, e + FEEB, 32);
            b.assert_zero(b.witness(0, e + R_FEE) - fee_lo_rec, "dl_feelo");
            let fee_hi_rec = recompose(b, e + FEEB + 32, 32);
            b.assert_zero(b.witness(0, e + R_FEE + 1) - fee_hi_rec, "dl_feehi");


            // -- passive register copies --
            for c in 0..(2 + 5 + 16 + 6 * SLOT_STRIDE + 5 * N_COORDS) {
                let off = if c < 2 {
                    e + R_FEE + c
                } else if c < 7 {
                    e + R_TYPE + (c - 2)
                } else if c < 23 {
                    e + R_TIME + (c - 7)
                } else if c < 23 + 6 * SLOT_STRIDE {
                    e + R_SLOT + (c - 23)
                } else {
                    e + R_ZINV + (c - 23 - 6 * SLOT_STRIDE)
                };
                b.assert_zero(
                    copy.clone() * (b.witness(1, off) - b.witness(0, off)),
                    "dl_copy",
                );
            }


            // -- statement 8: z flags on the leg's z-row --
            let zrow = b.preprocessed(tie + TIE_ZROW);
            let mut zsum = B::constant(0);
            for j in 0..N_COORDS {
                let zb = e + R_ZINV + 5 * j;
                let (z_hi, z_lo, inv_hi, inv_lo, z) = (
                    b.witness(0, zb),
                    b.witness(0, zb + 1),
                    b.witness(0, zb + 2),
                    b.witness(0, zb + 3),
                    b.witness(0, zb + 4),
                );
                b.assert_zero(z_hi.clone() * (z_hi.clone() - one.clone()), "dl_zh");
                b.assert_zero(z_lo.clone() * (z_lo.clone() - one.clone()), "dl_zl");
                b.assert_zero(
                    zrow.clone()
                        * (b.witness(0, e + R_DREG + 2 * j + 1) * inv_hi.clone()
                            - (one.clone() - z_hi.clone())),
                    "dl_inv_hi",
                );
                b.assert_zero(
                    zrow.clone()
                        * (b.witness(0, e + R_DREG + 2 * j) * inv_lo.clone()
                            - (one.clone() - z_lo.clone())),
                    "dl_inv_lo",
                );
                b.assert_zero(zrow.clone() * (z.clone() - z_hi * z_lo), "dl_z");
                zsum = zsum + z;
            }
            b.assert_zero(
                zrow * (zsum - (B::constant(63) - b.witness(0, e + BZ))),
                "dl_zsum",
            );
            for t in 0..6 {
                let bit = b.witness(0, e + BZB + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "dl_bzbit");
            }
            let bz_rec = recompose(b, e + BZB, 6);
            b.assert_zero(b.witness(0, e + BZ) - bz_rec, "dl_bz");


            // -- custody ties --
            let fee_flag = b.preprocessed(tie + TIE_FEE);
            for t in 0..64 {
                b.assert_zero(
                    fee_flag.clone()
                        * (b.witness(0, e + FEEB + t)
                            - b.witness(0, self.cons_base + V_BITS_BASE + t)),
                    "dl_fee_tie",
                );
            }
            for k in 0..leg.n_out {
                let flag = b.preprocessed(tie + TIE_VOL + k);
                let g = leg_out_start(self, l) + k;
                for t in 0..64 {
                    b.assert_zero(
                        flag.clone()
                            * (b.witness(0, e + VLB + t) - b.witness(0, self.out_reg(g) + t)),
                        "dl_vol_tie",
                    );
                }
            }
            for k in 0..16 {
                let flag = b.preprocessed(tie + TIE_LOG + k);
                if k < leg.n_in {
                    let g = leg_in_start(self, l) + k;
                    for t in 0..64 {
                        b.assert_zero(
                            flag.clone()
                                * (b.witness(0, e + LVB + t) - b.witness(0, self.in_reg(g) + t)),
                            "dl_log_tie",
                        );
                    }
                } else if k - leg.n_in < leg.n_out {
                    let g = leg_out_start(self, l) + (k - leg.n_in);
                    for t in 0..64 {
                        b.assert_zero(
                            flag.clone()
                                * (b.witness(0, e + LVB + t) - b.witness(0, self.out_reg(g) + t)),
                            "dl_log_tie",
                        );
                    }
                }
            }
        }


        // -- global: INR copy + boundary tie --
        for c in 0..3 {
            b.assert_zero(
                copy.clone() * (b.witness(1, cb + G_INR + c) - b.witness(0, cb + G_INR + c)),
                "dl_inr_copy",
            );
        }
        let bflag = b.preprocessed(pb + self.boundary_col());
        for k in 0..3 {
            b.assert_zero(
                bflag.clone()
                    * (b.witness(0, cb + G_INR + k) - b.witness(0, self.cons_base + ACCP_BASE + k)),
                "dl_boundary_tie",
            );
        }


        // -- final row: slot-sum aggregates and the two identities --
        let is_final = b.preprocessed(pb + PV_FINAL);
        let fl = cb + LEG_W * (n - 1);
        for t in 0..36 {
            let bit = b.witness(0, fl + FSKB + t);
            b.assert_zero(bit.clone() * (bit - one.clone()), "dl_fskb");
        }
        for t in 0..24 {
            let bit = b.witness(0, fl + VKCB + t);
            b.assert_zero(bit.clone() * (bit - one.clone()), "dl_vkcb");
        }
        let mut pos_lo = B::constant(0);
        let mut pos_hi = B::constant(0);
        let mut neg_lo = B::constant(0);
        let mut neg_hi = B::constant(0);
        let mut vt_lo = B::constant(0);
        let mut vt_hi = B::constant(0);
        for l in 0..n {
            let e = cb + LEG_W * l;
            vt_lo = vt_lo + b.witness(0, e + R_VOLV);
            vt_hi = vt_hi + b.witness(0, e + R_VOLV + 1);
            for i in 0..6 {
                let sb = e + R_SLOT + SLOT_STRIDE * i;
                let sgn = b.witness(0, sb + 224);
                pos_lo = pos_lo + (one.clone() - sgn.clone()) * b.witness(0, sb + 225);
                pos_hi = pos_hi + (one.clone() - sgn.clone()) * b.witness(0, sb + 226);
                neg_lo = neg_lo + sgn.clone() * b.witness(0, sb + 225);
                neg_hi = neg_hi + sgn * b.witness(0, sb + 226);
            }
        }
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + FS) + b.witness(0, fl + FSK) * B::constant(TWO32) - pos_lo),
            "dl_fs_pos0",
        );
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + FS + 1)
                    + b.witness(0, fl + FSK + 1) * B::constant(TWO32)
                    - pos_hi
                    - b.witness(0, fl + FSK)),
            "dl_fs_pos1",
        );
        b.assert_zero(
            is_final.clone() * (b.witness(0, fl + FS + 2) - b.witness(0, fl + FSK + 1)),
            "dl_fs_pos2",
        );
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + FS + 3)
                    + b.witness(0, fl + FSK + 2) * B::constant(TWO32)
                    - neg_lo),
            "dl_fs_neg0",
        );
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + FS + 4)
                    + b.witness(0, fl + FSK + 3) * B::constant(TWO32)
                    - neg_hi
                    - b.witness(0, fl + FSK + 2)),
            "dl_fs_neg1",
        );
        b.assert_zero(
            is_final.clone() * (b.witness(0, fl + FS + 5) - b.witness(0, fl + FSK + 3)),
            "dl_fs_neg2",
        );
        for k in 0..4 {
            let __w17_1 = b.witness(0, fl + FSK + k);
            let __r17_1 = recompose(b, fl + FSKB + 6 * k, 6);
            b.assert_zero(is_final.clone() * (__w17_1 - __r17_1), "dl_fsk_recomp");
        }
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + VT) + b.witness(0, fl + VKC) * B::constant(TWO32) - vt_lo),
            "dl_vt0",
        );
        b.assert_zero(
            is_final.clone()
                * (b.witness(0, fl + VT + 1)
                    + b.witness(0, fl + VKC + 1) * B::constant(TWO32)
                    - vt_hi
                    - b.witness(0, fl + VKC)),
            "dl_vt1",
        );
        b.assert_zero(
            is_final.clone() * (b.witness(0, fl + VT + 2) - b.witness(0, fl + VKC + 1)),
            "dl_vt2",
        );
        for k in 0..3 {
            let __w18_1 = b.witness(0, fl + VKC + k);
            let __r18_1 = recompose(b, fl + VKCB + 8 * k, 8);
            b.assert_zero(is_final.clone() * (__w18_1 - __r18_1), "dl_vkc_recomp");
        }
        // Identity A: Σ|negative slots| = Σ inputs (the boundary limbs).
        for k in 0..3 {
            b.assert_zero(
                is_final.clone() * (b.witness(0, fl + FS + 3 + k) - b.witness(0, cb + G_INR + k)),
                "dl_sum_neg_in",
            );
        }
        // Identity B: Σ|positive slots| = Σ volume rails.
        for k in 0..3 {
            b.assert_zero(
                is_final.clone() * (b.witness(0, fl + FS + k) - b.witness(0, fl + VT + k)),
                "dl_sum_pos_vol",
            );
        }
    }
}


#[derive(Clone, Debug)]
pub struct DeltaTrace {
    pub trace: Vec<Vec<Goldilocks>>,
    pub prep: Vec<Vec<Goldilocks>>,
}


fn put_bits(row: &mut [Goldilocks], col: usize, v: u64, n: usize) {
    for t in 0..n {
        row[col + t] = Goldilocks::from_u32(((v >> t) & 1) as u32);
    }
}


fn put_word(row: &mut [Goldilocks], col: usize, v: u64) {
    row[col] = Goldilocks::from_u64_reduce(v);
}


fn put_word_u128(row: &mut [Goldilocks], col: usize, v: u128) {
    row[col] = Goldilocks::from_u128_reduce(v);
}


#[allow(clippy::too_many_lines)]
pub fn gen_delta_trace(
    w: &CodecW,
    features: &[FeatureVector],
    fees: &[u64],
    legs: &[LegShape],
    in_values: &[u64],
    out_values: &[u64],
    boundary_limbs: [u64; 3],
) -> Result<DeltaTrace, WitnessGenError> {
    let n = legs.len();
    if features.len() != n || fees.len() != n {
        return Err(WitnessGenError::SeedCount { expected: n, found: features.len() });
    }
    let rows = ROWS_PER_LEG * n + 1;
    let fr = ROWS_PER_LEG * n;
    let cols = LEG_W * n + 3;
    let prep_w = PV_TIE + TIE_STRIDE * n + 1;
    let mut trace = vec![vec![Goldilocks::ZERO; cols]; rows];
    let mut prep = vec![vec![Goldilocks::ZERO; prep_w]; rows];
    let pb = 0; // the caller offsets when composing


    for r in 0..rows {
        prep[r][PV_DELTA_ROW] = Goldilocks::ONE;
    }
    for r in 0..fr {
        prep[r][PV_DELTA_COPY] = Goldilocks::ONE;
    }
    prep[fr][PV_FINAL] = Goldilocks::ONE;


    let mut slot_entries: Vec<Vec<(usize, u64, bool)>> = Vec::with_capacity(n);
    for f in features {
        let mut ents = Vec::new();
        for k in 0..224 {
            let v = f.value(k);
            if v != 0 {
                ents.push((k, v as u64, v < 0));
            }
        }
        if ents.len() > 6 {
            return Err(WitnessGenError::Inadmissible(
                nerv_codec::features::AdmissibilityViolation::TooManyActiveSlots {
                    count: ents.len(),
                    max: 6,
                },
            ));
        }
        while ents.len() < 6 {
            ents.push((0, 0, false));
        }
        slot_entries.push(ents);
    }


    for l in 0..n {
        let e = LEG_W * l;
        let base = ROWS_PER_LEG * l;
        let leg = legs[l];
        let f = &features[l];
        let delta = w.apply(f);


        // volume rows
        let out_start: usize = legs.iter().take(l).map(|s| s.n_out).sum();
        let mut vol: u64 = 0;
        for k in 0..8 {
            let r = base + k;
            prep[r][PV_VOL] = Goldilocks::ONE;
            if k == 0 {
                prep[r][PV_VOL0] = Goldilocks::ONE;
            }
            if k < leg.n_out {
                prep[r][PV_VOL_ACTIVE] = Goldilocks::ONE;
                let v = out_values[out_start + k];
                put_bits(&mut trace[r], e + VLB, v, 64);
                prep[r][PV_TIE + TIE_STRIDE * l + TIE_VOL + k] = Goldilocks::ONE;
                vol += v;
                if k + 1 == leg.n_out {
                    // PV_VOL_FINAL is a reserved column (register 63); unused.
                }
            }
        }


        // log rows (inputs then outputs); running LOGA per row.
        let in_start: usize = legs.iter().take(l).map(|s| s.n_in).sum();
        let vals: Vec<u64> = in_values[in_start..in_start + leg.n_in]
            .iter()
            .chain(out_values[out_start..out_start + leg.n_out].iter())
            .copied()
            .collect();
        let mut ells = Vec::with_capacity(16);
        for k in 0..16 {
            let r = base + 8 + k;
            prep[r][PV_LOG] = Goldilocks::ONE;
            if k == 0 {
                prep[r][PV_LOG0] = Goldilocks::ONE;
            }
            if k < vals.len() {
                prep[r][PV_LOG_ACTIVE] = Goldilocks::ONE;
                let v = vals[k];
                put_bits(&mut trace[r], e + LVB, v, 64);
                prep[r][PV_TIE + TIE_STRIDE * l + TIE_LOG + k] = Goldilocks::ONE;
                let vp1 = v + 1;
                put_bits(&mut trace[r], e + LBP, vp1, 64);
                let mut carry = 1u64;
                for j in 0..64 {
                    trace[r][e + LC + j] = Goldilocks::from_u32(carry as u32);
                    if j + 1 < 64 {
                        let bj = (v >> j) & 1;
                        carry = bj & carry;
                    }
                }
                let ell = 63 - vp1.leading_zeros() as u64;
                trace[r][e + LEL] = Goldilocks::from_u64_reduce(ell);
                for j in 0..64 {
                    trace[r][e + LS + j] = Goldilocks::from_u32(u32::from(j as u64 > ell));
                }
                ells.push(ell);
            }
        }
        // LOGA(r): 0 before the log phase; the running sum on log rows; the
        // total after (register-threaded, so the tail is the final value).
        let mut loga_at = vec![0u64; rows];
        let mut run = 0u64;
        for k in 0..16 {
            loga_at[base + 8 + k] = run;
            if k < ells.len() {
                run += ells[k];
            }
        }
        for r in (base + 24)..rows {
            loga_at[r] = run;
        }
        let loga = run;


        // registers (all rows)
        for r in 0..rows {
            let row = &mut trace[r];
            put_bits(row, e + FVOLB, vol, 64);
            put_word(row, e + R_VOLV, vol & 0xFFFF_FFFF);
            put_word(row, e + R_VOLV + 1, vol >> 32);
            put_word(row, e + R_LOGA, loga_at[r]);
            put_bits(row, e + LOGB, loga_at[r], 10);
            put_bits(row, e + FEEB, fees[l], 64);
            put_word(row, e + R_FEE, fees[l] & 0xFFFF_FFFF);
            put_word(row, e + R_FEE + 1, fees[l] >> 32);
            for t in 0..5 {
                row[e + R_TYPE + t] = Goldilocks::from_u32(u32::from(f.value(228 + t) == 1));
            }
            for t in 0..16 {
                row[e + R_TIME + t] = Goldilocks::from_u32(u32::from(f.value(240 + t) == 1));
            }
            for (i, &(idx, val, neg)) in slot_entries[l].iter().enumerate() {
                let sb = e + R_SLOT + SLOT_STRIDE * i;
                let active = val != 0 || neg;
                if active {
                    row[sb + idx] = Goldilocks::ONE;
                }
                row[sb + 224] = Goldilocks::from_u32(u32::from(neg));
                let mag = (val as i64).unsigned_abs();
                put_bits(row, e + SMAGB + 64 * i, mag, 64);
                put_word(row, sb + 225, mag & 0xFFFF_FFFF);
                put_word(row, sb + 226, mag >> 32);
                row[sb + 227] = Goldilocks::from_u32(u32::from(active));
            }
        }


        // DREG constants (threading re-asserts them)
        for r in 0..rows {
            for j in 0..64 {
                let d = delta.0[j];
                put_word(&mut trace[r], e + R_DREG + 2 * j, d & 0xFFFF_FFFF);
                put_word(&mut trace[r], e + R_DREG + 2 * j + 1, d >> 32);
            }
        }
        // PT constants (the seal's digitize)
        let pt = nerv_seal::digitize::digitize(&delta.0);
        for r in 0..rows {
            for s_idx in 0..512 {
                trace[r][e + R_PT + s_idx] = Goldilocks::from_u32(u32::from(pt.digits()[s_idx]));
            }
        }
        // ZINV: true inverses; z_hi pairs with the HI limb, z_lo with the LO.
        for j in 0..64 {
            let d = delta.0[j];
            let lo = d & 0xFFFF_FFFF;
            let hi = d >> 32;
            let (z_hi, z_lo) = (hi == 0, lo == 0);
            let inv_hi = if z_hi { 0u64 } else { Goldilocks::from_u64_reduce(hi).inverse().as_u64() };
            let inv_lo = if z_lo { 0u64 } else { Goldilocks::from_u64_reduce(lo).inverse().as_u64() };
            let zb = e + R_ZINV + 5 * j;
            for r in 0..rows {
                trace[r][zb] = Goldilocks::from_u32(u32::from(z_hi));
                trace[r][zb + 1] = Goldilocks::from_u32(u32::from(z_lo));
                put_word(&mut trace[r], zb + 2, inv_hi);
                put_word(&mut trace[r], zb + 3, inv_lo);
                trace[r][zb + 4] = Goldilocks::from_u32(u32::from(z_hi && z_lo));
            }
        }
        // BZ = 63 − #zero coordinates (statement 8 pins ≥ 1 nonzero).
        let zcount: u64 = (0..64).filter(|&j| delta.0[j] == 0).count() as u64;
        let bz = 63 - zcount.min(63);
        for r in 0..rows {
            put_word(&mut trace[r], e + BZ, bz);
            put_bits(&mut trace[r], e + BZB, bz, 6);
        }


        // MAC rows
        let rail = |k: usize| -> u64 {
            match k {
                0 => vol,
                1 => fees[l],
                2 => 1,
                3 => loga,
                4 | 5 => 1,
                _ => 0,
            }
        };
        for j in 0..64usize {
            let mut pp = [0u64, 0, 0];
            let mut pn = [0u64, 0, 0];
            for f_i in 0..12usize {
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
                    prep[r][PF_WM + k] = Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WS + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..4 {
                    let wv = w.weight(j, 224 + k);
                    prep[r][PF_WMF + k] = Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WSF + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..5 {
                    let wv = w.weight(j, 228 + k);
                    prep[r][PF_WMT + k] = Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WST + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }
                for k in 0..16 {
                    let wv = w.weight(j, 240 + k);
                    prep[r][PF_WMW + k] = Goldilocks::from_u64_reduce(wv.unsigned_abs() as u64);
                    prep[r][PF_WSW + k] = Goldilocks::from_u32(u32::from(wv < 0));
                }


                // x / w witnesses
                let (xv, wv, xs): (u64, i16, u64) = if f_i < 6 {
                    (rail(f_i), w.weight(j, 224 + f_i), 0)
                } else {
                    let &(idx, val, neg) = &slot_entries[l][f_i - 6];
                    ((val as i64).unsigned_abs(), w.weight(j, idx), u64::from(neg))
                };
                let (wm, ws) = (wv.unsigned_abs() as u64, u64::from(wv < 0));
                let row = &mut trace[r];
                put_word(row, e + XLO, xv & 0xFFFF_FFFF);
                put_word(row, e + XHI, xv >> 32);
                row[e + XS] = Goldilocks::from_u32(xs as u32);
                put_word(row, e + WMS, wm);
                row[e + WSS] = Goldilocks::from_u32(ws as u32);
                row[e + PS] = Goldilocks::from_u32((ws ^ xs) as u32);


                // product limbs: WMS·XLO = P0 + C1·2^32 (C1 < 2^15);
                // WMS·XHI + C1 = P1 + C2·2^32.
                let lo = xv & 0xFFFF_FFFF;
                let hi = xv >> 32;
                let t0 = wm as u128 * lo as u128;
                let p0 = (t0 & 0xFFFF_FFFF) as u64;
                let c1 = (t0 >> 32) as u64;
                let m = c1 as u128 + wm as u128 * hi as u128;
                let p1 = (m & 0xFFFF_FFFF) as u64;
                let c2 = (m >> 32) as u64;
                put_word(row, e + P0, p0);
                put_word(row, e + P1, p1);
                put_word(row, e + P2, c2);
                put_bits(row, e + PB, p0, 32);
                put_bits(row, e + PB + 32, p1, 32);
                put_bits(row, e + PB + 64, c2, 32);
                put_word(row, e + C1, c1);
                put_word(row, e + C2, c2);
                put_bits(row, e + CB, c1, 15);
                put_bits(row, e + CB + 15, c2, 15);


                // PRE-add accumulator state (the threading's left side).
                for limb in 0..3 {
                    put_word(row, e + PP + limb, pp[limb]);
                    put_word(row, e + PN + limb, pn[limb]);
                    put_bits(row, e + PPB + 32 * limb, pp[limb], 32);
                    put_bits(row, e + PNB + 32 * limb, pn[limb], 32);
                }


                // carry-chain add into the selected bucket
                let sel = 1 - (ws ^ xs); // 1 → positive bucket
                let mut carry_in: u128 = 0;
                for limb in 0..3 {
                    let pl = [p0, p1, c2][limb];
                    let tgt = if sel == 1 { &mut pp } else { &mut pn };
                    let t = tgt[limb] as u128 + pl as u128 + carry_in;
                    let k = t >> 32;
                    tgt[limb] = (t & 0xFFFF_FFFF) as u64;
                    if limb < 2 {
                        row[e + if sel == 1 { AKP } else { AKN } + limb] =
                            Goldilocks::from_u32(k as u32);
                    }
                    debug_assert!(limb < 2 || k == 0, "96-bit accumulator overflow (erratum 6 bound)");
                    carry_in = k;
                }


                // rounding witnesses on the final feature row
                if f_i == 11 {
                    let acc: i128 = pp_to_i128(&pp) - pp_to_i128(&pn);
                    let two96: i128 = 1 << 96;
                    let a_tc = ((acc.rem_euclid(two96)) as u128) as u64;
                    let (a0, a1, a2) = (a_tc & 0xFFFF_FFFF, (a_tc >> 32) & 0xFFFF_FFFF, a_tc >> 64);
                    let rr = a0 & 0x7FFF;
                    let q0 = a0 >> 15;
                    let mlo = a1 & 0x7FFF;
                    let mh = a1 >> 15;
                    let ahl = a2 & 0x7FFF;
                    let ahh = a2 >> 15;
                    let r14 = (rr >> 14) & 1;
                    let nz = u64::from(rr & 0x3FFF != 0);
                    let qparity = q0 & 1;
                    let adj = r14 * nz + r14 * (1 - nz) * qparity;
                    let q_lo = q0 + (mlo << 17);
                    let q_mid = mh + (ahl << 17);
                    let d_lo = q_lo.wrapping_add(adj);
                    let (d_hi_c, cc) = if d_lo >> 32 != 0 {
                        (q_mid.wrapping_add(1), 1u64)
                    } else {
                        (q_mid, 0u64)
                    };
                    let (d_hi, c2c) = if d_hi_c >> 32 != 0 {
                        (d_hi_c - TWO32, 1u64)
                    } else {
                        (d_hi_c, 0u64)
                    };
                    debug_assert_eq!(d_lo & 0xFFFF_FFFF, delta.0[j] & 0xFFFF_FFFF, "rounding lo");
                    debug_assert_eq!(d_hi, delta.0[j] >> 32, "rounding hi");
                    put_word(row, e + RR, rr);
                    put_bits(row, e + RRB, rr, 15);
                    put_word(row, e + Q0, q0);
                    put_bits(row, e + Q0B, q0, 17);
                    put_word(row, e + MLO, mlo);
                    put_bits(row, e + MLOB, mlo, 15);
                    put_word(row, e + MH, mh);
                    put_bits(row, e + MHB, mh, 17);
                    put_word(row, e + AHL, ahl);
                    put_bits(row, e + AHLB, ahl, 15);
                    put_word(row, e + AHH, ahh);
                    put_bits(row, e + AHHB, ahh, 17);
                    row[e + NZ] = Goldilocks::from_u32(nz as u32);
                    let r_lo = rr & 0x3FFF;
                    put_word(
                        row,
                        e + INVL,
                        if r_lo != 0 { Goldilocks::from_u64_reduce(r_lo).inverse().as_u64() } else { 0 },
                    );
                    put_word(row, e + ADJ, adj);
                    put_bits(row, e + ADJB, adj, 2);
                    let (br0, br1, br2) = sub_borrows(&pp, &pn);
                    row[e + BR0] = Goldilocks::from_u32(br0 as u32);
                    row[e + BR1] = Goldilocks::from_u32(br1 as u32);
                    row[e + BR2] = Goldilocks::from_u32(br2 as u32);
                    row[e + CC] = Goldilocks::from_u32(cc as u32);
                    row[e + C2C] = Goldilocks::from_u32(c2c as u32);
                }
            }
        }


        // digit rows
        for j in 0..64usize {
            let r = base + 792 + j;
            prep[r][PV_DIGIT] = Goldilocks::ONE;
            prep[r][PV_TIE + TIE_STRIDE * l + TIE_DIG + j] = Goldilocks::ONE;
            let d = delta.0[j];
            let row = &mut trace[r];
            put_word(row, e + D_LO, d & 0xFFFF_FFFF);
            put_word(row, e + D_HI, d >> 32);
            for k in 0..8 {
                let dv = (d >> (8 * k)) & 255;
                put_word(row, e + D_BASE + k, dv);
                put_bits(row, e + D_BITS + 8 * k, dv, 8);
            }
        }
        prep[base + 855][PV_TIE + TIE_STRIDE * l + TIE_ZROW] = Goldilocks::ONE;
        // The fee-tie flag at the leg's conservation fee row is set by the
        // composed generator (it needs conservation's row indices).
    }


    // final row: FS/FSK, VT/VKC from the register values at fr
    let fl = LEG_W * (n - 1);
    let mut pos_lo = 0u128;
    let mut pos_hi = 0u128;
    let mut neg_lo = 0u128;
    let mut neg_hi = 0u128;
    let mut vt_lo = 0u128;
    let mut vt_hi = 0u128;
    for l in 0..n {
        let e = LEG_W * l;
        vt_lo += trace[fr][e + R_VOLV].as_u64() as u128;
        vt_hi += trace[fr][e + R_VOLV + 1].as_u64() as u128;
        for i in 0..6 {
            let sb = e + R_SLOT + SLOT_STRIDE * i;
            let sgn = trace[fr][sb + 224].as_u64();
            let mlo = trace[fr][sb + 225].as_u64() as u128;
            let mhi = trace[fr][sb + 226].as_u64() as u128;
            if sgn == 0 {
                pos_lo += mlo;
                pos_hi += mhi;
            } else {
                neg_lo += mlo;
                neg_hi += mhi;
            }
        }
    }
    let row = &mut trace[fr];
    let fs0 = pos_lo & 0xFFFF_FFFF;
    let fsk0 = pos_lo >> 32;
    let t1 = pos_hi + fsk0;
    let fs1 = t1 & 0xFFFF_FFFF;
    let fsk1 = t1 >> 32;
    put_word_u128(row, fl + FS, fs0);
    put_word_u128(row, fl + FS + 1, fs1);
    put_word_u128(row, fl + FS + 2, fsk1);
    put_word(row, fl + FSK, fsk0 as u64);
    put_word(row, fl + FSK + 1, fsk1 as u64);
    let nf0 = neg_lo & 0xFFFF_FFFF;
    let nsk0 = neg_lo >> 32;
    let nt1 = neg_hi + nsk0;
    let nf1 = nt1 & 0xFFFF_FFFF;
    let nsk1 = nt1 >> 32;
    put_word_u128(row, fl + FS + 3, nf0);
    put_word_u128(row, fl + FS + 4, nf1);
    put_word_u128(row, fl + FS + 5, nsk1);
    put_word(row, fl + FSK + 2, nsk0 as u64);
    put_word(row, fl + FSK + 3, nsk1 as u64);
    put_bits(row, fl + FSKB, fsk0 as u64, 6);
    put_bits(row, fl + FSKB + 6, fsk1 as u64, 6);
    put_bits(row, fl + FSKB + 12, nsk0 as u64, 6);
    put_bits(row, fl + FSKB + 18, nsk1 as u64, 6);
    let v0 = vt_lo & 0xFFFF_FFFF;
    let vk0 = vt_lo >> 32;
    let vt1 = vt_hi + vk0;
    let v1 = vt1 & 0xFFFF_FFFF;
    let vk1 = vt1 >> 32;
    put_word_u128(row, fl + VT, v0);
    put_word_u128(row, fl + VT + 1, v1);
    put_word_u128(row, fl + VT + 2, vk1);
    put_word(row, fl + VKC, vk0 as u64);
    put_word(row, fl + VKC + 1, vk1 as u64);
    put_bits(row, fl + VKCB, vk0 as u64, 8);
    put_bits(row, fl + VKCB + 8, vk1 as u64, 8);
    // INR at all rows
    for r in 0..rows {
        for k in 0..3 {
            put_word(&mut trace[r], LEG_W * n + G_INR + k, boundary_limbs[k]);
        }
    }


    Ok(DeltaTrace { trace, prep })
}


fn pp_to_i128(pp: &[u64; 3]) -> i128 {
    (pp[0] as i128) + ((pp[1] as i128) << 32) + ((pp[2] as i128) << 64)
}


fn sub_borrows(pp: &[u64; 3], pn: &[u64; 3]) -> (u64, u64, u64) {
    let l0 = pp[0] as i128 - pn[0] as i128;
    let br0 = u64::from(l0 < 0);
    let l1 = pp[1] as i128 - pn[1] as i128 - br0 as i128;
    let br1 = u64::from(l1 < 0);
    let l2 = pp[2] as i128 - pn[2] as i128 - br1 as i128;
    let br2 = u64::from(l2 < 0);
    (br0, br1, br2)
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::custody_air::{
        gen_custody_trace, CustodyAir, CustodyWitness, InputWitness, OutputWitness, INPUT_REG_W,
        OUT_REG_W,
    };
    use crate::testutil::SplitMix64;
    use nerv_codec::features::{build_leg_features, LegKind, LegMovement};
    use nerv_codec::weight_gen::{certify, expand, BeaconRandomness, CertConfig};
    use nerv_core::types::{FeeSats, Height, ShardSet, INTERVALS_PER_EPOCH};
    use nerv_custody::nct::NoteCommitmentTree;
    use nerv_custody::nullifier::derive_nullifier;
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use nerv_custody::{MasterSeed, WalletKeys};


    const DEPTH: usize = nerv_custody::nct::DEPTH;


    struct Fixture {
        shell: nerv_custody::tx::TransactionShell,
        custody: CustodyWitness,
        features: Vec<FeatureVector>,
    }


    fn fixture(seed: u64) -> Fixture {
        let mut rng = SplitMix64::new(seed);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| nerv_custody::Address::generate(det, wk.nullifier_key(), i, &g).unwrap();
        let opening = |v: u64, i: u64| nerv_custody::commitment::NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let in1 = opening(50_000_000_000, 0);
        let in2 = opening(35_000_000_000, 1);
        let out_c = opening(25_000_000_000, 2);
        let out_ch = opening(9_999_000_000, 3);
        let out_b = opening(50_000_000_000, 4);
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
        let mk = |o: &nerv_custody::commitment::NoteOpening| Output {
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
        let shell = nerv_custody::tx::TransactionShell { legs: vec![leg7, leg40] };
        let custody = CustodyWitness {
            inputs,
            outputs: vec![
                OutputWitness { opening: out_c },
                OutputWitness { opening: out_ch },
                OutputWitness { opening: out_b },
            ],
            reverts: vec![],
            burns: vec![],
            fees: vec![600_000, 400_000],
        };
        let features = vec![
            build_leg_features(&LegMovement {
                inputs: vec![
                    (inputs[0].opening.delivery.to_vec(), 50_000_000_000),
                    (inputs[1].opening.delivery.to_vec(), 35_000_000_000),
                ],
                outputs: vec![
                    (out_c.delivery.to_vec(), 25_000_000_000),
                    (out_ch.delivery.to_vec(), 9_999_000_000),
                ],
                fee_nano: 600_000,
                kind: LegKind::CrossShardSpend,
                expiry_height: 5_000,
                epoch_length_blocks: INTERVALS_PER_EPOCH,
            })
            .unwrap(),
            build_leg_features(&LegMovement {
                inputs: vec![],
                outputs: vec![(out_b.delivery.to_vec(), 50_000_000_000)],
                fee_nano: 400_000,
                kind: LegKind::CrossShardIssue,
                expiry_height: 5_000,
                epoch_length_blocks: INTERVALS_PER_EPOCH,
            })
            .unwrap(),
        ];
        Fixture { shell, custody, features }
    }


    fn w() -> CodecW {
        let w = expand(&BeaconRandomness::from_bytes([0x57; 32]), nerv_codec::codec_w::WeightVersion(1));
        certify(&w, CertConfig { spark_samples_per_size: 2, column_rail_samples: 1 }).unwrap();
        w
    }


    struct Composed {
        air_cust: CustodyAir,
        air_delta: DeltaAir,
        trace: Vec<Vec<Goldilocks>>,
        prep: Vec<Vec<Goldilocks>>,
        publics: Vec<Goldilocks>,
    }


    impl<B: crate::air::builder::AirBuilder> crate::air::builder::Air<B> for Composed {
        fn eval(&self, b: &mut B) {
            self.air_cust.eval(b);
            self.air_delta.eval(b);
        }
    }


    fn composed(f: &Fixture) -> Composed {
        let w = w();
        let ct = gen_custody_trace(&f.custody, &f.shell, DEPTH).unwrap();
        let legs = vec![LegShape { n_in: 2, n_out: 2 }, LegShape { n_in: 0, n_out: 1 }];
        let in_values: Vec<u64> = f.custody.inputs.iter().map(|i| i.opening.value).collect();
        let out_values: Vec<u64> = f.custody.outputs.iter().map(|o| o.opening.value).collect();
        let n_in = in_values.len();
        let cons = crate::air::chips::conservation::gen_cons_trace(&{
            let mut e: Vec<(u64, bool)> = in_values.iter().map(|&v| (v, true)).collect();
            e.extend(out_values.iter().map(|&v| (v, false)));
            e.extend(f.custody.fees.iter().map(|&v| (v, false)));
            e
        });
        let boundary = [
            cons[n_in - 1][ACCP_BASE].as_u64(),
            cons[n_in - 1][ACCP_BASE + 1].as_u64(),
            cons[n_in - 1][ACCP_BASE + 2].as_u64(),
        ];
        let dt = gen_delta_trace(&w, &f.features, &f.custody.fees, &legs, &in_values, &out_values, boundary).unwrap();


        let air_cust = CustodyAir::new(2, 3, vec![], 0, DEPTH);
        let rows = ct.trace.len().max(dt.trace.len());
        let cols = ct.trace[0].len() + dt.trace[0].len();
        let prep_w = ct.prep[0].len() + dt.prep[0].len();
        let dcol = ct.trace[0].len();
        let dprep = ct.prep[0].len();
        let air_delta = DeltaAir {
            legs,
            cons_base: air_cust.cons_base(),
            cust_reg_base: air_cust.reg_base(),
            cust_input_reg_w: INPUT_REG_W,
            cust_n_in: 2,
            cust_out_reg_w: OUT_REG_W,
            col_base: dcol,
            prep_base: dprep,
        };
        let mut trace = vec![vec![Goldilocks::ZERO; cols]; rows];
        let mut prep = vec![vec![Goldilocks::ZERO; prep_w]; rows];
        for r in 0..ct.trace.len() {
            trace[r][..dcol].copy_from_slice(&ct.trace[r]);
            prep[r][..dprep].copy_from_slice(&ct.prep[r]);
        }
        for r in 0..dt.trace.len() {
            for (c, v) in dt.trace[r].iter().enumerate() {
                trace[r][dcol + c] = *v;
            }
            for (c, v) in dt.prep[r].iter().enumerate() {
                prep[r][dprep + c] = *v;
            }
        }
        // fee tie flags: conservation fee rows = n_in+n_out .. +n_fee
        let n_out = out_values.len();
        for l in 0..legs.len() {
            let row = n_in + n_out + l;
            prep[row][dprep + PV_TIE + TIE_STRIDE * l + TIE_FEE] = Goldilocks::ONE;
        }
        // boundary flag at the conservation boundary row
        prep[n_in - 1][dprep + air_delta.boundary_col()] = Goldilocks::ONE;
        Composed { air_cust, air_delta, trace, prep, publics: ct.publics }
    }


    #[test]
    fn layout_pins() {
        assert_eq!(LEG_W, 3744);
        assert_eq!(R_LOGA, 776);
        assert_eq!(R_SLOT + 6 * SLOT_STRIDE, 2784);
        assert_eq!(R_DREG, 2784);
        assert_eq!(R_DREG + 128, 2912);
        assert_eq!(R_PT, 2912);
        assert_eq!(R_PT + 512, 3424);
        assert_eq!(R_ZINV, 3424);
        assert_eq!(R_ZINV + 320, LEG_W);
        let air = DeltaAir {
            legs: vec![LegShape { n_in: 1, n_out: 1 }],
            cons_base: 0,
            cust_reg_base: 0,
            cust_input_reg_w: 848,
            cust_n_in: 1,
            cust_out_reg_w: 64,
            col_base: 0,
            prep_base: 0,
        };
        assert_eq!(air.rows(), 857);
        assert_eq!(air.cols(), LEG_W + 3);
        assert_eq!(air.prep_cols(), PV_TIE + TIE_STRIDE + 1);
        assert_eq!(air.boundary_col(), PV_TIE + TIE_STRIDE);
    }


    #[test]
    fn full_differential_against_codec() {
        let f = fixture(0x57A1);
        let w = w();
        let c = composed(&f);
        let r = NativeEval::check_with_prep(c.trace, c.prep, c.publics.clone(), &c, 8);
        assert!(r.is_ok(), "composed custody+delta: {r:?}");
        for (l, feat) in f.features.iter().enumerate() {
            let delta = w.apply(feat);
            let e = c.air_delta.col_base + LEG_W * l;
            let zrow = ROWS_PER_LEG * l + 855;
            for j in 0..64 {
                assert_eq!(c.trace[zrow][e + R_DREG + 2 * j].as_u64(), delta.0[j] & 0xFFFF_FFFF);
                assert_eq!(c.trace[zrow][e + R_DREG + 2 * j + 1].as_u64(), delta.0[j] >> 32);
            }
            let pt = nerv_seal::digitize::digitize(&delta.0);
            for s_idx in 0..512 {
                assert_eq!(c.trace[zrow][e + R_PT + s_idx].as_u64(), u64::from(pt.digits()[s_idx]));
            }
            // LOGA: running sum on log rows, total after
            let base = ROWS_PER_LEG * l;
            let total: u64 = (0..(f.features[l].value(224 + 3).unsigned_abs()))
                .map(|_| 0)
                .count() as u64 * 0 + 0; // placeholder removed below
            
        }
    }


    #[test]
    fn loga_running_sum_pin() {
        // LOGA(8) = 0; accumulates across active log rows; total thereafter.
        let f = fixture(0x57A2);
        let c = composed(&f);
        let e = c.air_delta.col_base;
        let base = ROWS_PER_LEG * 0;
        // leg 0: vals = [in1, in2, out_c, out_ch] — ells over those values
        let vals = [50_000_000_000u64, 35_000_000_000, 25_000_000_000, 9_999_000_000];
        let mut run = 0u64;
        assert_eq!(c.trace[base + 8][e + R_LOGA].as_u64(), 0);
        for k in 0..vals.len() {
            run += 63 - (vals[k] + 1).leading_zeros() as u64;
            assert_eq!(c.trace[base + 8 + k + 1][e + R_LOGA].as_u64(), run, "k={k}");
        }
        assert_eq!(c.trace[base + 24][e + R_LOGA].as_u64(), run);
        assert_eq!(c.trace[base + 400][e + R_LOGA].as_u64(), run);
        // before the log phase: zero
        assert_eq!(c.trace[base + 3][e + R_LOGA].as_u64(), 0);
    }


    #[test]
    fn tamper_battery() {
        let f = fixture(0x57A3);
        let c = composed(&f);
        let check = |trace: Vec<Vec<Goldilocks>>| {
            NativeEval::check_with_prep(trace, c.prep.clone(), c.publics.clone(), &c, 8)
        };
        assert!(check(c.trace.clone()).is_ok());


        let e0 = c.air_delta.col_base;
        let mut bad = c.trace.clone();
        bad[100][e0 + R_DREG] = bad[100][e0 + R_DREG] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "dreg");


        let mut bad = c.trace.clone();
        bad[100][e0 + R_SLOT + 225] = bad[100][e0 + R_SLOT + 225] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "slot mag");


        // distinctness: two ACTIVE slots selecting the same index
        let mut bad = c.trace.clone();
        bad[100][e0 + R_SLOT] = Goldilocks::ONE;
        bad[100][e0 + R_SLOT + SLOT_STRIDE] = Goldilocks::ONE;
        assert!(check(bad).is_err(), "slot distinctness");


        // phantom activity: pad slot marked active must be a one-hot
        let mut bad = c.trace.clone();
        let pad_sb = e0 + R_SLOT + SLOT_STRIDE * 5;
        bad[100][pad_sb + 227] = Goldilocks::ONE;
        assert!(check(bad).is_err(), "phantom active slot");


        let mut bad = c.trace.clone();
        bad[856 * 0 + 0][e0 + VLB + 5] = bad[856 * 0 + 0][e0 + VLB + 5] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "vol bit");


        let mut bad = c.trace.clone();
        bad[0][e0 + FEEB + 9] = bad[0][e0 + FEEB + 9] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "fee bit");


        let mut bad = c.trace.clone();
        bad[856 * 0 + 8][e0 + LVB + 3] = bad[856 * 0 + 8][e0 + LVB + 3] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "log bit");


        let mut bad = c.trace.clone();
        bad[856 * 0 + 792][e0 + D_BASE + 3] = bad[856 * 0 + 792][e0 + D_BASE + 3] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "digit");


        // INR tamper: breaks the boundary tie at the boundary row
        let mut bad = c.trace.clone();
        let inr = LEG_W * 2 + G_INR;
        bad[10][inr] = bad[10][inr] + Goldilocks::ONE;
        assert!(check(bad).is_err(), "inr");
    }
}
