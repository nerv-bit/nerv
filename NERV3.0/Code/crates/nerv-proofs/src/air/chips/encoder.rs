//! The sparse W·ΔS encoder chip (WP §5.1 statement 7; §5.3's encoder chip;
//! erratum 79): one row per (output coordinate j ∈ [0,64), feature f ∈
//! [0,12)) — 768 rows per leg, f-uniform constraints driven by the
//! preprocessed 12-way `F_SEL` selector. Features 0–5 are the fixed-family
//! rails (volume, fee, count, log, type, time); 6–11 are the six slot
//! entries (one-hot registers). δ_j = round½even(Σ w·x / 2^15) mod 2^64
//! with the accumulator as split positive/negative 3-limb sums — the
//! two's-complement rounding chain is exact, differentially pinned to
//! nerv-codec's `CodecW::apply` (which uses nerv-core's
//! `round_half_even_pow2`). Statement 8: per-coordinate nonzero flags with
//! an inverse trick and a bounded count.
//!
//! All columns are row-local MAC state at `E` offsets within the leg
//! block; registers (slots, rails, δ, plaintext) live in the delta_air leg
//! block and are read at fixed offsets. Products: two-limb magnitude
//! × 15-bit weight with 15-bit carry limbs; accumulation limb-wise with
//! boolean carries; the Ppos/Pneg limbs are boolean-decomposed (u32 force,
//! carry uniqueness). Max degree 3.

use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder};

pub const N_COORDS: usize = 64;
pub const N_FEATURES: usize = 12;
pub const MAC_ROWS: usize = N_COORDS * N_FEATURES; // 768

pub const P0: usize = 0;
pub const P1: usize = 1;
pub const P2: usize = 2;
pub const PB: usize = 3; // 96
pub const C1: usize = 99;
pub const C2: usize = 100;
pub const CB: usize = 101; // 30
pub const PS: usize = 131;
pub const PP: usize = 132; // 3
pub const PN: usize = 135; // 3
pub const PPB: usize = 138; // 96
pub const PNB: usize = 234; // 96
pub const AKP: usize = 330; // 3
pub const AKN: usize = 333; // 3
pub const RR: usize = 336;
pub const RRB: usize = 337; // 15
pub const Q0: usize = 352;
pub const Q0B: usize = 353; // 17
pub const MLO: usize = 370;
pub const MLOB: usize = 371; // 15
pub const MH: usize = 386;
pub const MHB: usize = 387; // 17
pub const AHL: usize = 404;
pub const AHLB: usize = 405; // 15
pub const AHH: usize = 420;
pub const AHHB: usize = 421; // 17
pub const NZ: usize = 438;
pub const INVL: usize = 439;
pub const ADJ: usize = 440;
pub const ADJB: usize = 441; // 2
pub const BR0: usize = 443;
pub const BR1: usize = 444;
pub const BR2: usize = 445;
pub const CC: usize = 446;
pub const C2C: usize = 447;
pub const XLO: usize = 448;
pub const XHI: usize = 449;
pub const XS: usize = 450;
pub const WMS: usize = 451;
pub const WSS: usize = 452;
pub const E_W: usize = 453;

// Prep (within the delta prep block):
pub const PF_MAC: usize = 0;
pub const PF_MAC_START: usize = 1;
pub const PF_MAC_FINAL: usize = 2;
pub const PF_SEL: usize = 11; // 12 one-hots
pub const PF_WM: usize = 23; // 224
pub const PF_WS: usize = 247; // 224
pub const PF_WMF: usize = 471; // 4
pub const PF_WSF: usize = 475; // 4
pub const PF_WMT: usize = 479; // 5
pub const PF_WST: usize = 484; // 5
pub const PF_WMW: usize = 489; // 16
pub const PF_WSW: usize = 505; // 16

const TWO32: u64 = 1 << 32;

fn recompose<B: AirBuilder>(b: &mut B, base: usize, n: usize) -> B::Expr {
    let mut s = B::constant(0);
    for t in 0..n {
        s = s + b.witness(0, base + t) * B::constant(1u64 << t);
    }
    s
}

/// The encoder chip for one leg. `reg` is the leg's register block base
/// (delta_air layout); `prep` the delta prep base; the MAC rows are
/// [row_base, row_base + 768) — flagged by prep, so the chip's constraints
/// are safe to evaluate at every trace row.
#[derive(Clone, Copy, Debug)]
pub struct EncoderChip {
    pub leg: usize,
    pub row_base: usize,
    pub prep_base: usize,
}

impl EncoderChip {
    pub const fn new(leg: usize, row_base: usize, prep_base: usize) -> EncoderChip {
        EncoderChip { leg, row_base, prep_base }
    }
}

impl<B: AirBuilder> Air<B> for EncoderChip {
    fn eval(&self, b: &mut B) {
        let pb = self.prep_base;
        let e = self.leg;
        let one = B::constant(1);
        let is_mac = b.preprocessed(pb + PF_MAC);
        let is_start = b.preprocessed(pb + PF_MAC_START);
        let is_final = b.preprocessed(pb + PF_MAC_FINAL);
        let mid = is_mac.clone() - is_final.clone(); // 1 on non-final MAC rows

        // -- x / w selection (prep-driven; F_SEL is preprocessed one-hot) --
        // Register offsets (delta_air layout, within the leg block):
        const R_VOLV: usize = 1390;
        const R_FEE: usize = 1392;
        const R_LOGA: usize = 776;
        const R_TYPE: usize = 1395;
        const R_TIME: usize = 1400;
        const R_SLOT: usize = 1416; // stride 228: OHP(224) SGN MAGL MAGH ACT


        // Eagerly materialize the PF_SEL preprocessed one-hot into a `Vec` so the
// selection closure can hand out clones without re-borrowing `b` (which is
// mutated later by `b.assert_zero(...)`).
let pf_sel: Vec<B::Expr> = (0..14).map(|i| b.preprocessed(pb + PF_SEL + i)).collect();
let f = |i: usize| pf_sel[i].clone();
        let mut xlo = B::constant(0);
        let mut xhi = B::constant(0);
        let mut xs = B::constant(0);
        xlo = xlo + f(0) * b.witness(0, e + R_VOLV);
        xhi = xhi + f(0) * b.witness(0, e + R_VOLV + 1);
        xlo = xlo + f(1) * b.witness(0, e + R_FEE);
        xhi = xhi + f(1) * b.witness(0, e + R_FEE + 1);
        xlo = xlo + f(2) + f(4) + f(5);
        xlo = xlo + f(3) * b.witness(0, e + R_LOGA);
        for i in 0..6 {
            let sb = e + R_SLOT + 228 * i;
            xlo = xlo + f(6 + i) * b.witness(0, sb + 224);
            xhi = xhi + f(6 + i) * b.witness(0, sb + 225);
            xs = xs + f(6 + i) * b.witness(0, sb + 226);
        }
        b.assert_zero(b.witness(0, e + XLO) - xlo, "enc_xlo_sel");
        b.assert_zero(b.witness(0, e + XHI) - xhi, "enc_xhi_sel");
        b.assert_zero(b.witness(0, e + XS) - xs, "enc_xs_sel");

        // w selection: fixed rails direct; type/time/slot via one-hots.
        let mut wms = B::constant(0);
        let mut wss = B::constant(0);
        for k in 0..4 {
            wms = wms + f(k) * b.preprocessed(pb + PF_WMF + k);
            wss = wss + f(k) * b.preprocessed(pb + PF_WSF + k);
        }
        let mut wt = B::constant(0);
        let mut wts = B::constant(0);
        for t in 0..5 {
            wt = wt + b.witness(0, e + R_TYPE + t) * b.preprocessed(pb + PF_WMT + t);
            wts = wts + b.witness(0, e + R_TYPE + t) * b.preprocessed(pb + PF_WST + t);
        }
        wms = wms + f(4) * wt;
        wss = wss + f(4) * wts;
        let mut ww = B::constant(0);
        let mut wws = B::constant(0);
        for t in 0..16 {
            ww = ww + b.witness(0, e + R_TIME + t) * b.preprocessed(pb + PF_WMW + t);
            wws = wws + b.witness(0, e + R_TIME + t) * b.preprocessed(pb + PF_WSW + t);
        }
        wms = wms + f(5) * ww;
        wss = wss + f(5) * wws;
        for i in 0..6 {
            let sb = e + R_SLOT + 228 * i;
            let mut si = B::constant(0);
            let mut sis = B::constant(0);
            for k in 0..224 {
                si = si + b.witness(0, sb + k) * b.preprocessed(pb + PF_WM + k);
                sis = sis + b.witness(0, sb + k) * b.preprocessed(pb + PF_WS + k);
            }
            wms = wms + f(6 + i) * si;
            wss = wss + f(6 + i) * sis;
        }
        b.assert_zero(b.witness(0, e + WMS) - wms, "enc_wms_sel");
        b.assert_zero(b.witness(0, e + WSS) - wss, "enc_wss_sel");

        // Product sign; product limbs.
        let ps = b.witness(0, e + PS);
        let wssv = b.witness(0, e + WSS);
        let xsv = b.witness(0, e + XS);
        b.assert_zero(ps.clone() - wssv.clone() - xsv.clone() + wssv * xsv * B::constant(2), "enc_ps");
        let wmsv = b.witness(0, e + WMS);
        let xlov = b.witness(0, e + XLO);
        let xhiv = b.witness(0, e + XHI);
        let c1 = b.witness(0, e + C1);
        let c2 = b.witness(0, e + C2);
        b.assert_zero(
            b.witness(0, e + P0) + c1.clone() * B::constant(TWO32) - wmsv.clone() * xlov.clone(),
            "enc_p0",
        );
        b.assert_zero(
            b.witness(0, e + P1) + c2.clone() * B::constant(TWO32)
                - wmsv.clone() * xhiv.clone()
                - c1.clone(),
            "enc_p1",
        );
        b.assert_zero(b.witness(0, e + P2) - c2.clone(), "enc_p2");
        // Ranges: product limbs and carries.
        for t in 0..96 {
            let bit = b.witness(0, e + PB + t);
            b.assert_zero(bit.clone() * (bit - one.clone()), "enc_pbit");
        }
        for t in 0..15 {
            let bit = b.witness(0, e + CB + t);
            b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "enc_cbit");
        }
        for t in 0..15 {
            let bit = b.witness(0, e + CB + 15 + t);
            b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "enc_cbit");
        }
        let p0_rec = recompose(b, e + PB, 32);
        b.assert_zero(
            b.witness(0, e + P0) - p0_rec,
            "enc_p0_recomp",
        );
        let p1_rec = recompose(b, e + PB + 32, 32);
        b.assert_zero(
            b.witness(0, e + P1) - p1_rec,
            "enc_p1_recomp",
        );
        let p2_rec = recompose(b, e + PB + 64, 32);
        b.assert_zero(
            b.witness(0, e + P2) - p2_rec,
            "enc_p2_recomp",
        );
        let c1_rec = recompose(b, e + CB, 15);
        b.assert_zero(c1.clone() - c1_rec, "enc_c1_recomp");
        let c2_rec = recompose(b, e + CB + 15, 15);
        b.assert_zero(c2.clone() - c2_rec, "enc_c2_recomp");
        // Accumulator limbs are u32 (bits) — carry uniqueness.
        for t in 0..192 {
            let bit = b.witness(0, e + PPB + t);
            b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "enc_ppbit");
        }
        for limb in 0..3 {
            let pp_rec = recompose(b, e + PPB + 32 * limb, 32);
            b.assert_zero(
                b.witness(0, e + PP + limb) - pp_rec,
                "enc_pp_recomp",
            );
            let pn_rec = recompose(b, e + PNB + 32 * limb, 32);
            b.assert_zero(
                b.witness(0, e + PN + limb) - pn_rec,
                "enc_pn_recomp",
            );
        }

        // -- Accumulation threading (three gated families per limb) --
        let p0 = b.witness(0, e + P0);
        let p1 = b.witness(0, e + P1);
        let p2 = b.witness(0, e + P2);
        for limb in 0..3 {
            let pl = match limb {
                0 => p0.clone(),
                1 => p1.clone(),
                _ => p2.clone(),
            };
            let pp = b.witness(0, e + PP + limb);
            let pn = b.witness(0, e + PN + limb);
            let npp = b.witness(1, e + PP + limb);
            let npn = b.witness(1, e + PN + limb);
            let akp = b.witness(0, e + AKP + limb);
            let akn = b.witness(0, e + AKN + limb);
            b.assert_zero(akp.clone() * (akp.clone() - one.clone()), "enc_akp_bool");
            b.assert_zero(akn.clone() * (akn.clone() - one.clone()), "enc_akn_bool");
            let cin_p = if limb == 0 { B::constant(0) } else { b.witness(0, e + AKP + limb - 1) };
            let cin_n = if limb == 0 { B::constant(0) } else { b.witness(0, e + AKN + limb - 1) };
            b.assert_zero(
                mid.clone()
                    * (npp.clone() - pp.clone()
                        - (one.clone() - ps.clone()) * (pl.clone() + cin_p.clone())
                        + akp.clone() * B::constant(TWO32)),
                "enc_pp_add",
            );
            b.assert_zero(
                mid.clone()
                    * (npn.clone() - pn.clone() - ps.clone() * (pl.clone() + cin_n.clone())
                        + akn.clone() * B::constant(TWO32)),
                "enc_pn_add",
            );

            b.assert_zero(is_final.clone() * npp.clone(), "enc_pp_reset");
            b.assert_zero(is_final.clone() * npn.clone(), "enc_pn_reset");
            b.assert_zero((one.clone() - is_mac.clone()) * (npp.clone() - pp.clone()), "enc_pp_pass");
            b.assert_zero((one.clone() - is_mac.clone()) * (npn.clone() - pn.clone()), "enc_pn_pass");
            b.assert_zero(is_start.clone() * pp.clone(), "enc_pp_init");
            b.assert_zero(is_start.clone() * pn.clone(), "enc_pn_init");
        }

        // -- Final-row rounding (gated is_final): inline post-adds, exact
        //    two's-complement subtract, floor/tie-even shift, δ limbs --
        let mut post_pp = [B::constant(0), B::constant(0), B::constant(0)];
        let mut post_pn = [B::constant(0), B::constant(0), B::constant(0)];
        for limb in 0..3 {
            let pl = match limb {
                0 => p0.clone(),
                1 => p1.clone(),
                _ => p2.clone(),
            };
            post_pp[limb] = b.witness(0, e + PP + limb)
                + (one.clone() - ps.clone()) * pl.clone()
                - b.witness(0, e + AKP + limb) * B::constant(TWO32);
            post_pn[limb] = b.witness(0, e + PN + limb)
                + ps.clone() * pl.clone()
                - b.witness(0, e + AKN + limb) * B::constant(TWO32);
        }
        let br0 = b.witness(0, e + BR0);
        let br1 = b.witness(0, e + BR1);
        let br2 = b.witness(0, e + BR2);
        for x in [&br0, &br1, &br2] {
            b.assert_zero(x.clone() * (x.clone() - one.clone()), "enc_br_bool");
        }
        let a0 = post_pp[0].clone() - post_pn[0].clone() + br0.clone() * B::constant(TWO32);
        let a1 = post_pp[1].clone() - post_pn[1].clone() - br0.clone()
            + br1.clone() * B::constant(TWO32);
        let a2 = post_pp[2].clone() - post_pn[2].clone() - br1.clone()
            + br2.clone() * B::constant(TWO32);
        // Decompositions (range-check A's limbs).
        b.assert_zero(a1.clone() - (b.witness(0, e + MLO) + b.witness(0, e + MH) * B::constant(1 << 15)), "enc_a1_dec");
        b.assert_zero(a2.clone() - (b.witness(0, e + AHL) + b.witness(0, e + AHH) * B::constant(1 << 15)), "enc_a2_dec");

        for (base, n) in [(RRB, 15usize), (MLOB, 15), (MHB, 17), (AHLB, 15), (AHHB, 17), (Q0B, 17)] {
            for t in 0..n {
                let bit = b.witness(0, e + base + t);
                b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "enc_rbit");
            }
        }
        let q0_rec = recompose(b, e + Q0B, 17);
        b.assert_zero(b.witness(0, e + Q0) - q0_rec, "enc_q0_recomp");
        let mlo_rec = recompose(b, e + MLOB, 15);
        b.assert_zero(b.witness(0, e + MLO) - mlo_rec, "enc_mlo_recomp");
        let mh_rec = recompose(b, e + MHB, 17);
        b.assert_zero(b.witness(0, e + MH) - mh_rec, "enc_mh_recomp");
        let ahl_rec = recompose(b, e + AHLB, 15);
        b.assert_zero(b.witness(0, e + AHL) - ahl_rec, "enc_ahl_recomp");
        let ahh_rec = recompose(b, e + AHHB, 17);
        b.assert_zero(b.witness(0, e + AHH) - ahh_rec, "enc_ahh_recomp");
        let rr_rec = recompose(b, e + RRB, 15);
        b.assert_zero(b.witness(0, e + RR) - rr_rec, "enc_rr_recomp");
        // A_0 = 2^15·Q0 + r; r's bits are RRB.
        b.assert_zero(
            b.witness(0, e + Q0) * B::constant(1 << 15) + b.witness(0, e + RR) - a0,
            "enc_a0_shift",
        );

        // q limbs (range-exact by construction): q_lo = Q0 + MLO·2^17;
        // q_mid = MH + AHL·2^17; q_hi = AHH.
        let q_lo = b.witness(0, e + Q0) + b.witness(0, e + MLO) * B::constant(1 << 17);
        let q_mid = b.witness(0, e + MH) + b.witness(0, e + AHL) * B::constant(1 << 17);

        // Rounding adjust: gt = r > 2^14; tie = r == 2^14 ∧ q odd.
        let r14 = b.witness(0, e + RRB + 14);
        let mut r_lo13 = B::constant(0);
        for t in 0..14 {
            r_lo13 = r_lo13 + b.witness(0, e + RRB + t);
        }
        let nz = b.witness(0, e + NZ);
        b.assert_zero(nz.clone() * (nz.clone() - one.clone()), "enc_nz_bool");
        b.assert_zero(r_lo13 * b.witness(0, e + INVL) - nz.clone(), "enc_nz");
        let qparity = b.witness(0, e + Q0B);
        let gt = r14.clone() * nz.clone();
        let tie = r14 * (one.clone() - nz) * qparity;
        b.assert_zero(
            b.witness(0, e + ADJ) - gt - tie,
            "enc_adj",
        );
        for t in 0..2 {
            let bit = b.witness(0, e + ADJB + t);
            b.assert_zero(bit.clone() * (bit.clone() - one.clone()), "enc_adjbit");
        }
        let adj_rec = recompose(b, e + ADJB, 2);
        b.assert_zero(b.witness(0, e + ADJ) - adj_rec, "enc_adj_recomp");

        // δ limbs: δ_lo = q_lo + ADJ − CC·2^32; δ_hi = q_mid + CC − C2C·2^32.
        // (DREG threading is delta_air's: it re-derives these limbs via
        // delta_lo_expr/delta_hi_expr at the final rows.)
        let cc = b.witness(0, e + CC);
        let c2c = b.witness(0, e + C2C);
        b.assert_zero(cc.clone() * (cc.clone() - one.clone()), "enc_cc_bool");
        b.assert_zero(c2c.clone() * (c2c.clone() - one.clone()), "enc_c2c_bool");
        let _d_lo = q_lo + b.witness(0, e + ADJ) - cc.clone() * B::constant(TWO32);
        let _d_hi = q_mid + c2c.clone() * B::constant(TWO32);
    }
}


/// The final-row δ limb expressions, re-derived for delta_air's DREG
/// threading (identical arithmetic; kept in one place by sharing the
/// column constants above).
pub fn delta_lo_expr<B: AirBuilder>(b: &mut B, e: usize) -> B::Expr {
    let q_lo = b.witness(0, e + Q0) + b.witness(0, e + MLO) * B::constant(1 << 17);
    q_lo + b.witness(0, e + ADJ) - b.witness(0, e + CC) * B::constant(TWO32)
}

pub fn delta_hi_expr<B: AirBuilder>(b: &mut B, e: usize) -> B::Expr {
    let q_mid = b.witness(0, e + MH) + b.witness(0, e + AHL) * B::constant(1 << 17);
    q_mid + b.witness(0, e + CC) - b.witness(0, e + C2C) * B::constant(TWO32)
}
