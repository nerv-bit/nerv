//! The seal chip (WP §5.1 statement 10; §5.3's seal chip; errata 80/83):
//! the in-circuit NTT over R_q — the proof crate's [NOVEL ★] surface —
//! proving (u, v) = (A·r + e₁, T·r + e₂ + scale·m) with bounded smallness,
//! differentially pinned end-to-end against
//! nerv_seal::circuit_stmt::apply_statement_10 (DSR-7).
//!
//! Layout (per leg; erratum 83): 242 rows — 8 forward transforms of r_j
//! (9 rows each: input+level-0, levels 1..7, output), 80 MAC rows (10
//! outputs × 8 product rows), 10 inverse transforms (9 rows each). One
//! shared 256-column state block S; registers REG_R/UNTT/VNTT/ACC; a
//! ~41K-column aux block, row-type-interpreted. All butterflies at fixed
//! indices (c, c±2^ℓ); twiddles (TW), MAC operands (MOP), and all phase
//! flags are preprocessed — one constraint family set serves every
//! transform of its kind. 2,560 public inputs per leg (u then v).
//!
//! Soundness design (erratum 83): canonical checks = paired 32-bit
//! decompositions of x and x+δ (δ = 2^32−q); the inverse's (s−t) via an
//! explicit canonical dwit; div-check remainders non-canonical-permitted
//! (outputs canonical-pinned); ACC as an exact integer chain; bitrevs
//! bound by direct fixed-index constraints.

use nerv_core::field::Goldilocks;
use nerv_seal::noise::SCALE;
use nerv_seal::ring::{bitrev8, fwd_twiddle, inv_twiddle, N_INV, Q};
use crate::air::builder::AirExpr;
use crate::air::builder::{Air, AirBuilder};

pub const N: usize = nerv_seal::ring::N;
pub const DELTA: u64 = (1u64 << 32) - Q;

pub const ROWS: usize = 242;
pub const S: usize = 0;
pub const REG_R: usize = 256;
pub const REG_UNTT: usize = 2304;
pub const REG_VNTT: usize = 4352;
pub const REG_ACC: usize = 4864;
pub const AX_Q: usize = 5120;
pub const AX_RB: usize = 13312;
pub const AX_OC: usize = 21504;
pub const AX_DW: usize = 37888;
pub const AX_K: usize = 41984;
pub const AX_RAUX: usize = 43008;
pub const AX_MB: usize = 43776;
pub const AX_W: usize = 45824;
pub const LEG_W: usize = 46336;

pub const TW: usize = 0;
pub const MOP: usize = 256;
pub const F_IS_FWD: usize = 512;
pub const F_IS_MAC: usize = 513;
pub const F_IS_INV: usize = 514;
pub const F_IS_INPUT: usize = 515;
pub const F_IS_T_OUT: usize = 516;
pub const F_LEVEL: usize = 517;
pub const F_FWD_T: usize = 525;
pub const F_MACJ: usize = 533;
pub const F_MACI: usize = 541;
pub const F_INV_T: usize = 551;
/// 1 on rows [0, ROWS−1): the seal region's copy gate. Register threads
/// hold within the region and never constrain the row after it — the
/// composed trace continues in other modules' columns there (register 65).
pub const PV_SEAL_COPY: usize = 561;
pub const PREP_COLS: usize = 562;


const _: () = assert!(LEG_W == 46336);
const _: () = assert!(PREP_COLS == 562);


#[derive(Clone, Copy, Debug)]
pub struct SealChip {
    pub col_base: usize,
    pub prep_base: usize,
    pub public_base: usize,
}

impl SealChip {
    pub const fn new() -> SealChip {
        SealChip { col_base: 0, prep_base: 0, public_base: 0 }
    }

    pub const fn at(col_base: usize, prep_base: usize, public_base: usize) -> SealChip {
        SealChip { col_base, prep_base, public_base }
    }
}

impl Default for SealChip {
    fn default() -> Self {
        SealChip::new()
    }
}

fn rc<B: AirBuilder>(b: &B, base: usize, n: usize) -> B::Expr {
    let mut s = B::Expr::zero();
    for t in 0..n {
        s = s + b.witness(0, base + t) * B::constant(1u64 << t);
    }
    s
}

fn divck<B: AirBuilder>(b: &mut B, gate: B::Expr, prod: B::Expr, qbase: usize, rbase: usize) {
    let one = B::constant(1);
    for t in 0..32 {
        let q = b.witness(0, qbase + t);
        b.assert_zero(gate.clone() * q.clone() * (q - one.clone()), "sc_qbit");
        let r = b.witness(0, rbase + t);
        b.assert_zero(gate.clone() * r.clone() * (r - one.clone()), "sc_rbit");
    }
    let qv = rc(b, qbase, 32);
    let rv = rc(b, rbase, 32);
    b.assert_zero(gate * (prod - qv * B::constant(Q) - rv), "sc_div");
}

fn canon<B: AirBuilder>(b: &mut B, gate: B::Expr, x: B::Expr, base: usize) {
    let one = B::constant(1);
    for t in 0..64 {
        let bit = b.witness(0, base + t);
        b.assert_zero(gate.clone() * bit.clone() * (bit - one.clone()), "sc_cbit");
    }
    let lo = rc(b, base, 32);
    let hi = rc(b, base + 32, 32);
    b.assert_zero(gate.clone() * (x.clone() - lo), "sc_clo");
    b.assert_zero(gate * (x + B::constant(DELTA) - hi), "sc_chi");
}

impl<B: AirBuilder> Air<B> for SealChip {
    fn eval(&self, b: &mut B) {
        let (cb, pb) = (self.col_base, self.prep_base);
        let one = B::constant(1);
        let two = B::constant(2);
        let qc = B::constant(Q);
        let seal_copy = b.preprocessed(pb + PV_SEAL_COPY);


        let is_fwd = b.preprocessed(pb + F_IS_FWD);
        let is_mac = b.preprocessed(pb + F_IS_MAC);
        let is_inv = b.preprocessed(pb + F_IS_INV);
        let is_input = b.preprocessed(pb + F_IS_INPUT);
        let is_t_out = b.preprocessed(pb + F_IS_T_OUT);

        // ---- F1: r input binding (smallness: 2 mag bits => |r| <= 3) ----
        {
            let gate = is_fwd.clone() * is_input.clone();
            for i in 0..N {
                let base = cb + AX_RAUX + 3 * i;
                let s = b.witness(0, base);
                let m1 = b.witness(0, base + 1);
                let m0 = b.witness(0, base + 2);
                // Bit constraints first; each asserts `gate * m_i * (m_i - 1) = 0`.
                // The `(m - one)` term moves `m`, so clone the witness before
                // any other use later in the loop body.
                b.assert_zero(gate.clone() * s.clone() * (s.clone() - one.clone()), "sc_r_sgn");
                b.assert_zero(gate.clone() * m1.clone() * (m1.clone() - one.clone()), "sc_r_m1");
                b.assert_zero(gate.clone() * m0.clone() * (m0.clone() - one.clone()), "sc_r_m0");
                let mag = m1 * two.clone() + m0;
                let r_canon = mag.clone() + s.clone() * (qc.clone() - two.clone() * mag);
                let sc = b.witness(0, cb + S + bitrev8(i));
                b.assert_zero(gate.clone() * (sc - r_canon), "sc_r_bind");
            }
        }

        // ---- F2: forward CT levels ----
        for lvl in 0..8 {
            let gate = is_fwd.clone() * b.preprocessed(pb + F_LEVEL + lvl);
            let m = 1usize << lvl;
            let mut bidx = 0usize;
            for blk in 0..(N / (2 * m)) {
                for j in 0..m {
                    let c = 2 * m * blk + j;
                    let c2 = c + m;
                    let a_lo = b.witness(0, cb + S + c);
                    let a_up = b.witness(0, cb + S + c2);
                    let tw = b.preprocessed(pb + TW + c);
                    let qbase = cb + AX_Q + 32 * bidx;
                    let rbase = cb + AX_RB + 32 * bidx;
                    let gate_arg = gate.clone();
                    let tw_arg = tw.clone();
                    let prod = a_up.clone() * tw_arg;
                    divck(b, gate_arg, prod, qbase, rbase);
                    let rv = rc(b, rbase, 32);
                    let k1 = b.witness(0, cb + AX_K + 2 * bidx);
                    let k2 = b.witness(0, cb + AX_K + 2 * bidx + 1);
                    b.assert_zero(gate.clone() * k1.clone() * (k1.clone() - one.clone()), "sc_kbool");
                    b.assert_zero(gate.clone() * k2.clone() * (k2.clone() - one.clone()), "sc_kbool");
                    let out_lo = b.witness(1, cb + S + c);
                    let out_up = b.witness(1, cb + S + c2);
                    b.assert_zero(
                        gate.clone()
                            * (out_lo.clone() - a_lo.clone() - rv.clone() + k1.clone() * qc.clone()),
                        "sc_fwd_lo",
                    );
                    b.assert_zero(
                        gate.clone()
                            * (out_up.clone() - a_lo.clone() + rv.clone() - k2.clone() * qc.clone()),
                        "sc_fwd_up",
                    );
                    canon(b, gate.clone(), out_lo, cb + AX_OC + 64 * 2 * bidx);
                    canon(b, gate.clone(), out_up, cb + AX_OC + 64 * (2 * bidx + 1));
                    bidx += 1;
                }
            }
        }

        // ---- F3: REG_R writes (blocked copies) ----
          for j in 0..8 {
            let wgate = b.preprocessed(pb + F_FWD_T + j);
            let cgate = (one.clone() - wgate.clone()) * seal_copy.clone();

            for c in 0..N {
                let w = b.witness(1, cb + REG_R + 256 * j + c);
                let s = b.witness(0, cb + S + c);
                b.assert_zero(wgate.clone() * (w - s), "sc_regr_write");
                let cur = b.witness(0, cb + REG_R + 256 * j + c);
                let nxt = b.witness(1, cb + REG_R + 256 * j + c);
                b.assert_zero(cgate.clone() * (nxt - cur), "sc_regr_copy");
            }
        }

        // ---- F4: MAC div-checks (j-selected, shared Q/R aux) ----
        for j in 0..8 {
            let gate = is_mac.clone() * b.preprocessed(pb + F_MACJ + j);
            for c in 0..N {
                let mop = b.preprocessed(pb + MOP + c);
                let rj = b.witness(0, cb + REG_R + 256 * j + c);
                divck(b, gate.clone(), mop * rj, cb + AX_Q + 32 * c, cb + AX_RB + 32 * c);
            }
        }

        // ---- F5: ACC init / update / copy ----
        {
            let g0 = is_mac.clone() * b.preprocessed(pb + F_MACJ);
            let gu = is_mac.clone() * (one.clone() - b.preprocessed(pb + F_MACJ));
            let gc = (one.clone() - is_mac.clone()) * seal_copy.clone();
            for c in 0..N {
                let rv = rc(b, cb + AX_RB + 32 * c, 32);
                let nxt = b.witness(1, cb + REG_ACC + c);
                b.assert_zero(g0.clone() * (nxt.clone() - rv.clone()), "sc_acc_init");
                let cur = b.witness(0, cb + REG_ACC + c);
                b.assert_zero(gu.clone() * (nxt.clone() - cur.clone() - rv.clone()), "sc_acc_upd");
                b.assert_zero(gc.clone() * (nxt.clone() - cur), "sc_acc_copy");
                let _ = gc.clone();
            }
        }

        // ---- F6: MAC final (k + canonical) and register writes ----
        {
            let gate = is_mac.clone() * b.preprocessed(pb + F_MACJ + 7);
            for c in 0..N {
                let kbase = cb + AX_K + 4 * c;
                for t in 0..4 {
                    let k = b.witness(0, kbase + t);
                    b.assert_zero(gate.clone() * k.clone() * (k - one.clone()), "sc_kbool");
                }
                let kv = rc(b, kbase, 4);
                let rv = rc(b, cb + AX_RB + 32 * c, 32);
                let acc = b.witness(0, cb + REG_ACC + c);
                let out_expr = acc + rv - kv * qc.clone();
                canon(b, gate.clone(), out_expr, cb + AX_OC + 64 * c);
            }
            for i in 0..10 {
                let wgate = gate.clone() * b.preprocessed(pb + F_MACI + i);
                let target = if i < 8 {
                    cb + REG_UNTT + 256 * i
                } else {
                    cb + REG_VNTT + 256 * (i - 8)
                };
                for c in 0..N {
                    let out_lo = rc(b, cb + AX_OC + 64 * c, 32);
                    let w = b.witness(1, target + c);
                    b.assert_zero(wgate.clone() * (w - out_lo), "sc_macw");
                }
                let block = b.preprocessed(pb + F_MACJ + 7) * b.preprocessed(pb + F_MACI + i);
                let cgate = (one.clone() - block) * seal_copy.clone();

                for c in 0..N {
                    let cur = b.witness(0, target + c);
                    let nxt = b.witness(1, target + c);
                    b.assert_zero(cgate.clone() * (nxt - cur), "sc_rcopy");
                }
            }
        }

        // ---- F8: inverse inputs ----
        for t in 0..10 {
            let gate = is_inv.clone() * is_input.clone() * b.preprocessed(pb + F_INV_T + t);
            let src = if t < 8 {
                cb + REG_UNTT + 256 * t
            } else {
                cb + REG_VNTT + 256 * (t - 8)
            };
            for c in 0..N {
                let s = b.witness(0, cb + S + c);
                let r = b.witness(0, src + c);
                b.assert_zero(gate.clone() * (s - r), "sc_invin");
            }
        }

        // ---- F9: inverse CT levels (via canonical dwit) ----
        for r_lvl in 0..8 {
            let gate = is_inv.clone() * b.preprocessed(pb + F_LEVEL + r_lvl);
            let m = 128usize >> r_lvl;
            let mut bidx = 0usize;
            for blk in 0..(N / (2 * m)) {
                for j in 0..m {
                    let c = 2 * m * blk + j;
                    let c2 = c + m;
                    let s_v = b.witness(0, cb + S + c);
                    let t_v = b.witness(0, cb + S + c2);
                    let tw = b.preprocessed(pb + TW + c);
                    let dw = rc(b, cb + AX_DW + 32 * bidx, 32);
                    let ksgn = b.witness(0, cb + AX_K + 2 * bidx);
                    let gate_arg = gate.clone();
                    b.assert_zero(gate_arg.clone() * ksgn.clone() * (ksgn.clone() - one.clone()), "sc_dwk");
                    b.assert_zero(
                        gate.clone()
                            * (s_v.clone() - t_v.clone() - dw.clone() + ksgn.clone() * qc.clone()),
                        "sc_dwdiff",
                    );
                    let qbase = cb + AX_Q + 32 * bidx;
                    let rbase = cb + AX_RB + 32 * bidx;
                    divck(b, gate.clone(), dw.clone() * tw.clone(), qbase, rbase);
                    let rv = rc(b, rbase, 32);
                    let k1 = b.witness(0, cb + AX_K + 2 * bidx + 1);
                    b.assert_zero(gate.clone() * k1.clone() * (k1.clone() - one.clone()), "sc_kbool");
                    let out_lo = b.witness(1, cb + S + c);
                    let out_up = b.witness(1, cb + S + c2);
                    b.assert_zero(
                        gate.clone()
                            * (out_lo.clone() - s_v.clone() - t_v.clone() + k1 * qc.clone()),
                        "sc_inv_lo",
                    );
                    b.assert_zero(gate.clone() * (out_up.clone() - rv.clone()), "sc_inv_up");
                    canon(b, gate.clone(), out_lo, cb + AX_OC + 64 * 2 * bidx);
                    canon(b, gate.clone(), out_up, cb + AX_OC + 64 * (2 * bidx + 1));
                    bidx += 1;
                }
            }
        }

        // ---- F10: inverse outputs (N_INV + bitrev + e + scale·m + public) ----
        for t in 0..10 {
            let gate = is_inv.clone() * is_t_out.clone() * b.preprocessed(pb + F_INV_T + t);
            let puboff = self.public_base
                + if t < 8 { 256 * t } else { 2048 + 256 * (t - 8) };
            for c in 0..N {
                let raw = b.witness(0, cb + S + bitrev8(c));
                let ninv = B::constant(N_INV);
                divck(b, gate.clone(), raw * ninv, cb + AX_Q + 32 * c, cb + AX_RB + 32 * c);
                let out = rc(b, cb + AX_RB + 32 * c, 32);
                canon(b, gate.clone(), out.clone(), cb + AX_OC + 64 * c);

                let ebase = cb + AX_RAUX + 3 * c;
                let s = b.witness(0, ebase);
                let m1 = b.witness(0, ebase + 1);
                let m0 = b.witness(0, ebase + 2);
                b.assert_zero(gate.clone() * s.clone() * (s.clone() - one.clone()), "sc_e_sgn");
                b.assert_zero(gate.clone() * m1.clone() * (m1.clone() - one.clone()), "sc_e_m1");
                b.assert_zero(gate.clone() * m0.clone() * (m0.clone() - one.clone()), "sc_e_m0");
                let mag = m1.clone() * two.clone() + m0.clone();
                let e_canon = mag.clone() + s.clone() * (qc.clone() - two.clone() * mag.clone());

                for t2 in 0..8 {
                    let mb = b.witness(0, cb + AX_MB + 8 * c + t2);
                    b.assert_zero(gate.clone() * mb.clone() * (mb.clone() - one.clone()), "sc_mbit");
                }
                let mv = rc(b, cb + AX_MB + 8 * c, 8);
                for t2 in 0..2 {
                    let wb = b.witness(0, cb + AX_W + 2 * c + t2);
                    b.assert_zero(gate.clone() * wb.clone() * (wb - one.clone()), "sc_wbit");
                }
                let wv = rc(b, cb + AX_W + 2 * c, 2) - one.clone();
                let pb = b.public(puboff + c);
                b.assert_zero(
                    gate.clone()
                        * (out
                            + e_canon
                            + B::constant(SCALE) * mv
                            - pb
                            - wv * qc.clone()),
                    "sc_bind",
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generation (the native twin pipeline)
// ---------------------------------------------------------------------------

use nerv_seal::ring::{Mat2x8, Mat8x8Ntt, Poly};
use nerv_seal::sampling::{expand_matrix, ASeed};
use nerv_seal::SealError;

#[derive(Clone, Debug)]
pub struct SealLegInput {
    pub r: Vec<[i64; N]>,
    pub e1: Vec<[i64; N]>,
    pub e2: Vec<[i64; N]>,
    pub m: [u16; 512],
}

#[derive(Clone, Debug)]
pub struct SealTrace {
    pub trace: Vec<Vec<Goldilocks>>,
    pub prep: Vec<Vec<Goldilocks>>,
    pub publics: Vec<Goldilocks>,
}

fn add_mod(a: u64, b: u64) -> u64 {
    ((a as u128 + b as u128) % Q as u128) as u64
}

fn sub_mod(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + Q - b
    }
}

fn mul_q(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}

fn fwd_states(input: &[u64; N]) -> Vec<[u64; N]> {
    let mut st = vec![[0u64; N]; 9];
    for i in 0..N {
        st[0][i] = input[bitrev8(i)];
    }
    for lvl in 0..8 {
        let m = 1usize << lvl;
        let cur = st[lvl];
        let mut nxt = cur;
        for blk in 0..(N / (2 * m)) {
            for j in 0..m {
                let c = 2 * m * blk + j;
                let u = cur[c];
                let v = mul_q(cur[c + m], fwd_twiddle(lvl, j));
                nxt[c] = add_mod(u, v);
                nxt[c + m] = sub_mod(u, v);
            }
        }
        st[lvl + 1] = nxt;
    }
    st
}

fn inv_states(input: &[u64; N]) -> Vec<[u64; N]> {
    let mut st = vec![[0u64; N]; 9];
    st[0] = *input;
    for r_lvl in 0..8 {
        let m = 128usize >> r_lvl;
        let cur = st[r_lvl];
        let mut nxt = cur;
        for blk in 0..(N / (2 * m)) {
            for j in 0..m {
                let c = 2 * m * blk + j;
                let s = cur[c];
                let t = cur[c + m];
                nxt[c] = add_mod(s, t);
                nxt[c + m] = mul_q(sub_mod(s, t), inv_twiddle(7 - r_lvl, j));
            }
        }
        st[r_lvl + 1] = nxt;
    }
    st
}

fn put_val(row: &mut [Goldilocks], col: usize, v: u64) {
    row[col] = Goldilocks::from_u64_reduce(v);
}

fn put_bits(row: &mut [Goldilocks], col: usize, v: u64, n: usize) {
    for t in 0..n {
        row[col + t] = Goldilocks::from_u32(((v >> t) & 1) as u32);
    }
}

fn put_canon_bits(row: &mut [Goldilocks], col: usize, x: u64) {
    put_bits(row, col, x, 32);
    put_bits(row, col + 32, x + DELTA, 32);
}

pub fn gen_seal_trace(
    input: &SealLegInput,
    a_seed: &ASeed,
    t: &Mat2x8,
) -> Result<SealTrace, SealError> {
    let a = expand_matrix(a_seed)?;
    let a_ntt: Mat8x8Ntt = a.ntt();
    let t_ntt = t.ntt();
    let canon = |v: i64| if v >= 0 { v as u64 } else { Q - v.unsigned_abs() };

    let mut r_states: Vec<Vec<[u64; N]>> = Vec::with_capacity(8);
    let mut r_ntt = vec![[0u64; N]; 8];
    for j in 0..8 {
        let rc_: [u64; N] = std::array::from_fn(|i| canon(input.r[j][i]));
        let st = fwd_states(&rc_);
        r_ntt[j] = st[8];
        r_states.push(st);
    }

    let mut u_ntt = vec![[0u64; N]; 8];
    for i in 0..8 {
        for c in 0..N {
            let mut s = 0u64;
            for j in 0..8 {
                s = add_mod(s, mul_q(a_ntt.row(i)[j].values()[c], r_ntt[j][c]));
            }
            u_ntt[i][c] = s;
        }
    }
    let mut v_ntt = vec![[0u64; N]; 2];
    for k in 0..2 {
        for c in 0..N {
            let mut s = 0u64;
            for j in 0..8 {
                s = add_mod(s, mul_q(t_ntt.row(k)[j].values()[c], r_ntt[j][c]));
            }
            v_ntt[k][c] = s;
        }
    }

    let u_inv: Vec<Vec<[u64; N]>> = (0..8).map(|i| inv_states(&u_ntt[i])).collect();
    let v_inv: Vec<Vec<[u64; N]>> = (0..2).map(|k| inv_states(&v_ntt[k])).collect();

    let e1c: Vec<[u64; N]> =
        (0..8).map(|i| std::array::from_fn(|c| canon(input.e1[i][c]))).collect();
    let e2c: Vec<[u64; N]> =
        (0..2).map(|k| std::array::from_fn(|c| canon(input.e2[k][c]))).collect();

    let u_pub: Vec<[u64; N]> = (0..8)
        .map(|i| std::array::from_fn(|c| add_mod(mul_q(u_inv[i][8][bitrev8(c)], N_INV), e1c[i][c])))
        .collect();
    let v_pub: Vec<[u64; N]> = (0..2)
        .map(|k| {
            std::array::from_fn(|c| {
                let scaled = (u64::from(input.m[256 * k + c]) * SCALE) % Q;
                add_mod(add_mod(mul_q(v_inv[k][8][bitrev8(c)], N_INV), e2c[k][c]), scaled)
            })
        })
        .collect();

    let mut trace = vec![vec![Goldilocks::ZERO; LEG_W]; ROWS];
    let mut prep = vec![vec![Goldilocks::ZERO; PREP_COLS]; ROWS];
    for row in 0..ROWS {
        let f = if row < 72 {
            F_IS_FWD
        } else if row < 152 {
            F_IS_MAC
        } else {
            F_IS_INV
        };
        prep[row][f] = Goldilocks::ONE;
    }
    for row in 0..ROWS - 1 {
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
    for t_ in 0..10 {
        for l in 0..9 {
            let row = 152 + 9 * t_ + l;
            prep[row][F_INV_T + t_] = Goldilocks::ONE;
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

    // Forward transforms.
    for j in 0..8 {
        let st = &r_states[j];
        for l in 0..9 {
            let row = &mut trace[9 * j + l];
            for c in 0..N {
                put_val(row, S + c, st[l][c]);
            }
        }
        {
            let row = &mut trace[9 * j];
            for i in 0..N {
                let v = input.r[j][i];
                row[AX_RAUX + 3 * i] = Goldilocks::from_u32(u32::from(v < 0));
                put_bits(row, AX_RAUX + 3 * i + 1, v.unsigned_abs(), 2);
            }
        }
        for l in 0..8 {
            let m = 1usize << l;
            let mut bidx = 0usize;
            for blk in 0..(N / (2 * m)) {
                for jj in 0..m {
                    let c = 2 * m * blk + jj;
                    let c2 = c + m;
                    let tw = fwd_twiddle(l, jj);
                    let prod = u128::from(st[l][c2]) * u128::from(tw);
                    let qv = (prod / u128::from(Q)) as u64;
                    let rv = (prod % u128::from(Q)) as u64;
                    let row = &mut trace[9 * j + l];
                    put_bits(row, AX_Q + 32 * bidx, qv, 32);
                    put_bits(row, AX_RB + 32 * bidx, rv, 32);
                    row[AX_K + 2 * bidx] =
                        Goldilocks::from_u32(u32::from(st[l][c] + rv >= Q));
                    row[AX_K + 2 * bidx + 1] =
                        Goldilocks::from_u32(u32::from(st[l][c] < rv));
                    put_canon_bits(row, AX_OC + 64 * 2 * bidx, st[l + 1][c]);
                    put_canon_bits(row, AX_OC + 64 * (2 * bidx + 1), st[l + 1][c2]);
                    prep[9 * j + l][TW + c] = Goldilocks::from_u64_reduce(tw);
                    bidx += 1;
                }
            }
        }
        for row in (9 * j + 9)..ROWS {
            for c in 0..N {
                put_val(&mut trace[row], REG_R + 256 * j + c, r_ntt[j][c]);
            }
        }
    }

     // MACs.
    let mut acc_final = [0u64; N];
    for blk in 0..10 {
        let is_v = blk >= 8;
        let idx = blk - 8 * usize::from(is_v);
        let mut acc = [0u64; N];

        for jj in 0..8 {
            let row_i = 72 + 8 * blk + jj;
            let mop: &[u64; N] = if is_v {
                t_ntt.row(idx)[jj].values()
            } else {
                a_ntt.row(idx)[jj].values()
            };
            for c in 0..N {
                let prod = u128::from(mop[c]) * u128::from(r_ntt[jj][c]);
                let qv = (prod / u128::from(Q)) as u64;
                let rv = (prod % u128::from(Q)) as u64;
                put_bits(&mut trace[row_i], AX_Q + 32 * c, qv, 32);
                put_bits(&mut trace[row_i], AX_RB + 32 * c, rv, 32);
                prep[row_i][MOP + c] = Goldilocks::from_u64_reduce(mop[c]);
                acc[c] += rv;
            }
            for c in 0..N {
                put_val(&mut trace[row_i + 1], REG_ACC + c, acc[c]);
            }
        }
        let row_i = 72 + 8 * blk + 7;
        let out_arr: [u64; N] = std::array::from_fn(|c| acc[c] % Q);
        for c in 0..N {
            put_bits(&mut trace[row_i], AX_K + 4 * c, acc[c] / Q, 4);
            put_canon_bits(&mut trace[row_i], AX_OC + 64 * c, out_arr[c]);
        }
        let target = if is_v { REG_VNTT + 256 * idx } else { REG_UNTT + 256 * idx };
        let src_val: &[u64; N] =
            if is_v { &v_ntt[idx] } else { &u_ntt[idx] };
         for row in (row_i + 1)..ROWS {
            for c in 0..N {
                put_val(&mut trace[row], target + c, src_val[c]);
            }
        }
        acc_final.copy_from_slice(&acc);
    }
    // ACC holds block 9's final value through the inverse section — the
    // copy gate's demand on rows [152, ROWS).
    for row in 152..ROWS {
        for c in 0..N {
            put_val(&mut trace[row], REG_ACC + c, acc_final[c]);
        }
    }


    // Inverse transforms.
    for t_ in 0..10 {
        let (st, is_u) = if t_ < 8 { (&u_inv[t_], true) } else { (&v_inv[t_ - 8], false) };
        for l in 0..9 {
            let row = &mut trace[152 + 9 * t_ + l];
            for c in 0..N {
                put_val(row, S + c, st[l][c]);
            }
        }
        for l in 0..8 {
            let m = 128usize >> l;
            let mut bidx = 0usize;
            for blk in 0..(N / (2 * m)) {
                for jj in 0..m {
                    let c = 2 * m * blk + jj;
                    let c2 = c + m;
                    let tw = inv_twiddle(7 - l, jj);
                    let (s_v, t_v) = (st[l][c], st[l][c2]);
                    let dw = sub_mod(s_v, t_v);
                    let prod = u128::from(dw) * u128::from(tw);
                    let qv = (prod / u128::from(Q)) as u64;
                    let rv = (prod % u128::from(Q)) as u64;
                    let row = &mut trace[152 + 9 * t_ + l];
                    put_bits(row, AX_DW + 32 * bidx, dw, 32);
                    row[AX_K + 2 * bidx] = Goldilocks::from_u32(u32::from(s_v < t_v));
                    row[AX_K + 2 * bidx + 1] =
                        Goldilocks::from_u32(u32::from(s_v + t_v >= Q));
                    put_bits(row, AX_Q + 32 * bidx, qv, 32);
                    put_bits(row, AX_RB + 32 * bidx, rv, 32);
                    put_canon_bits(row, AX_OC + 64 * 2 * bidx, st[l + 1][c]);
                    put_canon_bits(row, AX_OC + 64 * (2 * bidx + 1), st[l + 1][c2]);
                    prep[152 + 9 * t_ + l][TW + c] = Goldilocks::from_u64_reduce(tw);
                    bidx += 1;
                }
            }
        }
        let row_i = 152 + 9 * t_ + 8;
        let (e_arr, pub_arr) = if is_u {
            (&input.e1[t_], &u_pub[t_])
        } else {
            (&input.e2[t_ - 8], &v_pub[t_ - 8])
        };
        for c in 0..N {
            let raw = st[8][bitrev8(c)];
            let prod = u128::from(raw) * u128::from(N_INV);
            let qv = (prod / u128::from(Q)) as u64;
            let out = (prod % u128::from(Q)) as u64;
            let row = &mut trace[row_i];
            put_bits(row, AX_Q + 32 * c, qv, 32);
            put_bits(row, AX_RB + 32 * c, out, 32);
            put_canon_bits(row, AX_OC + 64 * c, out);
            let ev = e_arr[c];
            row[AX_RAUX + 3 * c] = Goldilocks::from_u32(u32::from(ev < 0));
            put_bits(row, AX_RAUX + 3 * c + 1, ev.unsigned_abs(), 2);
            let extra: u64 = if is_u {
                0
            } else {
                u64::from(input.m[256 * (t_ - 8) + c]) * SCALE
            };
            if !is_u {
                put_bits(row, AX_MB + 8 * c, u64::from(input.m[256 * (t_ - 8) + c]), 8);
            }
            let sum2 =
                out as i128 + canon(ev) as i128 + extra as i128 - pub_arr[c] as i128;
            let w = sum2.div_euclid(Q as i128);
            put_bits(row, AX_W + 2 * c, (w + 1) as u64, 2);
        }
    }

    let mut publics = Vec::with_capacity(2560);
    for i in 0..8 {
        for c in 0..N {
            publics.push(Goldilocks::from_u64_reduce(u_pub[i][c]));
        }
    }
    for k in 0..2 {
        for c in 0..N {
            publics.push(Goldilocks::from_u64_reduce(v_pub[k][c]));
        }
    }
    Ok(SealTrace { trace, prep, publics })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::sym::measure;
    use crate::testutil::SplitMix64;
    use nerv_seal::circuit_stmt::{apply_statement_10, check_statement_10};
    use nerv_seal::digitize::{digitize, Plaintext};
    use nerv_seal::encrypt::derive_reference_keypair;
    use nerv_seal::ring::{Vec2, Vec8};

    fn fixture(seed: u64) -> (SealLegInput, ASeed, Mat2x8) {
        let mut rng = SplitMix64::new(seed);
        let (pk, _) = derive_reference_keypair(&[seed as u8; 32]).unwrap();
        let small = |rng: &mut SplitMix64| -> [i64; N] {
            std::array::from_fn(|_| (rng.next_u64() % 5) as i64 - 2)
        };
        let r: Vec<[i64; N]> = (0..8).map(|_| small(&mut rng)).collect();
        let e1: Vec<[i64; N]> = (0..8).map(|_| small(&mut rng)).collect();
        let e2: Vec<[i64; N]> = (0..2).map(|_| small(&mut rng)).collect();
        let mut delta = [0u64; 64];
        for v in delta.iter_mut() {
            *v = rng.next_u64();
        }
        let m: [u16; 512] = digitize(&delta).digits();
        (
            SealLegInput { r, e1, e2, m },
            *pk.a_seed(),
            Mat2x8::new([
                *Vec8::new(*pk.t().row(0).clone()).polys(),
                *Vec8::new(*pk.t().row(1).clone()).polys(),
            ]),
        )
    }

    #[test]
    fn layout_pins() {
        assert_eq!(ROWS, 242);
        assert_eq!(LEG_W, 46336);
        assert_eq!(PREP_COLS, 562);
        assert_eq!(AX_W + 512, LEG_W);
        assert_eq!(F_INV_T + 10, PV_SEAL_COPY);
        assert_eq!(PV_SEAL_COPY + 1, PREP_COLS);

    }

    #[test]
    fn measured_families_and_degree() {
        let (n, d) = measure(&SealChip::new());
        assert!(d <= 2, "max witness degree {d}");
        assert!(n > 900_000 && n < 1_100_000, "family count {n}");
        assert!(n < (1 << 21), "D.2 seal-chip ceiling breached: {n}");
    }

    #[test]
    fn differential_against_circuit_stmt() {
        let (input, a_seed, t) = fixture(0x5EA1);

        let r_v = Vec8::new(std::array::from_fn(|j| Poly::from_centered(&input.r[j])));
        let e1_v = Vec8::new(std::array::from_fn(|j| Poly::from_centered(&input.e1[j])));
        let e2_v =
            Vec2::new(std::array::from_fn(|k| Poly::from_centered(&input.e2[k])));
        let plaintext = Plaintext::from_digits(&input.m).unwrap();
        let (u, v) = apply_statement_10(&a_seed, &t, &r_v, &e1_v, &e2_v, &plaintext).unwrap();
        check_statement_10(&a_seed, &t, &u, &v, &plaintext, &r_v, &e1_v, &e2_v).unwrap();

        let st = gen_seal_trace(&input, &a_seed, &t).unwrap();
        assert_eq!(st.trace.len(), ROWS);
        assert_eq!(st.trace[0].len(), LEG_W);
        assert_eq!(st.publics.len(), 2560);

        for i in 0..8 {
            for c in 0..N {
                assert_eq!(st.publics[256 * i + c].as_u64(), u.poly(i).coefficients()[c]);
            }
        }
        for k in 0..2 {
            for c in 0..N {
                assert_eq!(
                    st.publics[2048 + 256 * k + c].as_u64(),
                    v.poly(k).coefficients()[c]
                );
            }
        }

        for j in 0..8 {
            let expect = Poly::from_centered(&input.r[j]).ntt();
            for c in 0..N {
                assert_eq!(
                    st.trace[9 * j + 8][S + c].as_u64(),
                    expect.values()[c],
                    "fwd state pin j={j} c={c}"
                );
            }
        }

        let chip = SealChip::new();
        assert!(NativeEval::check_with_prep(st.trace, st.prep, st.publics, &chip, 4).is_ok());
    }

    #[test]
    fn tamper_battery() {
        let (input, a_seed, t) = fixture(0x5EA2);
        let chip = SealChip::new();

        let mut st = gen_seal_trace(&input, &a_seed, &t).unwrap();
        st.trace[0][S + 7] = st.trace[0][S + 7] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(st.trace.clone(), st.prep.clone(), st.publics.clone(), &chip, 8)
            .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "sc_r_bind" || e.name.starts_with("sc_fwd")));
        st.trace[0][S + 7] = st.trace[0][S + 7] - Goldilocks::ONE;

        st.trace[3][S + 100] = st.trace[3][S + 100] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(st.trace.clone(), st.prep.clone(), st.publics.clone(), &chip, 8).is_err());
        st.trace[3][S + 100] = st.trace[3][S + 100] - Goldilocks::ONE;

        st.trace[72][AX_RB + 5] = st.trace[72][AX_RB + 5] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(st.trace.clone(), st.prep.clone(), st.publics.clone(), &chip, 8)
            .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "sc_div" || e.name == "sc_rbit"));
        st.trace[72][AX_RB + 5] = st.trace[72][AX_RB + 5] - Goldilocks::ONE;

        st.trace[80][REG_R + 3] = st.trace[80][REG_R + 3] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(st.trace.clone(), st.prep.clone(), st.publics.clone(), &chip, 8).is_err());
        st.trace[80][REG_R + 3] = st.trace[80][REG_R + 3] - Goldilocks::ONE;

        st.trace[152][S + 9] = st.trace[152][S + 9] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(st.trace.clone(), st.prep.clone(), st.publics.clone(), &chip, 8)
            .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "sc_invin" || e.name.starts_with("sc_inv")));
        st.trace[152][S + 9] = st.trace[152][S + 9] - Goldilocks::ONE;

        let mut pubs = st.publics.clone();
        pubs[100] = pubs[100] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(st.trace, st.prep, pubs, &chip, 8).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "sc_bind"));
    }
}
