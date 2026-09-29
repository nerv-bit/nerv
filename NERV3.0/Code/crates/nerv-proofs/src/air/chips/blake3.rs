//! The BLAKE3 compression chip (WP §5.3, D.2's 16-bit lookup mandate):
//! one full compression per 66-row window — init row, 56 G-function rows
//! (7 rounds × 8 sequential Gs), 8 final-XOR rows, 1 output row —
//! differentially tested against the blake3 crate end-to-end (DSR-7).
//!
//! Layout (713 witness columns, self-contained; chips at different
//! row-windows share all columns):
//!   STATE[16] CV[8] MSG[16] MX[1] GIN[4] GV[8] K[4]
//!   GINB[128] MIDB[128] ROTB[64] XORB[128] FXOP[128] FXO[64] OUT[16]
//! Prep (280): phase flags, message-permutation one-hots, G input/output
//! position one-hots, final-XOR selection one-hots, per-window params.
//! Registers (STATE/CV/MSG/OUT) copy within the window via transitions.
//!
//! u32 discipline: every add operand is u32-forced by a boolean bit
//! decomposition (GINB/MIDB/ROTB, msg bits at row 0) or is a
//! recomposition of boolean bits; wrapping adds are exact field
//! equations with range-checked carries ({0,1,2} for 3-term, {0,1} for
//! 2-term). XOR: per-bit degree-2 identities; rotations: linear
//! recompositions at shifted bit positions. Max witness degree 3.


use nerv_core::field::Goldilocks;
use crate::air::builder::{Air, AirBuilder};
pub const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
pub const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
pub const CHUNK_START: u32 = 1;
pub const CHUNK_END: u32 = 2;
pub const PARENT: u32 = 4;
pub const ROOT: u32 = 8;


pub const ROWS: usize = 66;
pub const STATE: usize = 0;
pub const CV: usize = 16;
pub const MSG: usize = 24;
pub const MX: usize = 40;
pub const GIN: usize = 41;
pub const GV: usize = 45;
pub const K: usize = 53;
pub const GINB: usize = 57;
pub const MIDB: usize = 185;
pub const ROTB: usize = 313;
pub const XORB: usize = 377;
pub const FXOP: usize = 505;
pub const FXO: usize = 633;
pub const OUT: usize = 697;
pub const WITNESS_COLS: usize = 713;


pub const P_INIT: usize = 0;
pub const P_G: usize = 1;
pub const P_FIN: usize = 2;
pub const P_SELMSG: usize = 4;
pub const P_SELIN: usize = 20;
pub const P_WSTATE: usize = 84;
pub const P_FXA: usize = 164;
pub const P_FXB: usize = 196;
pub const P_FXW: usize = 244;
pub const P_PARAM: usize = 276;
pub const PREP_COLS: usize = 280;


const DIAG: [usize; 16] = [0, 5, 10, 15, 1, 6, 11, 12, 2, 7, 8, 13, 3, 4, 9, 14];
const TWO32: u64 = 1 << 32;


const _: () = assert!(WITNESS_COLS == 713);
const _: () = assert!(PREP_COLS == 280);


// ---------------------------------------------------------------------------
// Native references
// ---------------------------------------------------------------------------


fn g_native(a: u32, b: u32, c: u32, d: u32, mx: u32) -> [u32; 4] {
    let a = a.wrapping_add(b).wrapping_add(mx);
    let d = (d ^ a).rotate_right(16);
    let c = c.wrapping_add(d);
    let b = (b ^ c).rotate_right(12);
    let a = a.wrapping_add(b).wrapping_add(mx);
    let d = (d ^ a).rotate_right(8);
    let c = c.wrapping_add(d);
    let b = (b ^ c).rotate_right(7);
    [a, b, c, d]
}


/// Round r (1..=7) on the current state, message permuted r−1 times.
/// Sequential in-place: the diagonal step reads the column step's outputs.
fn round_native(state: &mut [u32; 16], block: &[u32; 16], r: usize) {
    let mut m = *block;
    for _ in 0..(r - 1) {
        let old = m;
        for i in 0..16 {
            m[i] = old[MSG_PERMUTATION[i]];
        }
    }
    let s0 = *state;
    for grp in 0..4 {
        let i = grp * 4;
        let [a, b, c, d] = g_native(s0[i], s0[i + 1], s0[i + 2], s0[i + 3], m[grp]);
        state[i] = a;
        state[i + 1] = b;
        state[i + 2] = c;
        state[i + 3] = d;
    }
    let s1 = *state;
    for grp in 0..4 {
        let i = grp * 4;
        let (ia, ib, ic, id) = (DIAG[i], DIAG[i + 1], DIAG[i + 2], DIAG[i + 3]);
        let [a, b, c, d] = g_native(s1[ia], s1[ib], s1[ic], s1[id], m[grp + 4]);
        state[ia] = a;
        state[ib] = b;
        state[ic] = c;
        state[id] = d;
    }
}


/// The compression function — the native twin.
pub fn compress_native(cv: &[u32; 8], block: &[u32; 16], len: u8, counter: u64, flags: u32) -> [u32; 16] {
    let mut state: [u32; 16] = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        u32::from(len), counter as u32, (counter >> 32) as u32, flags,
    ];
    for r in 1..=7 {
        round_native(&mut state, block, r);
    }
    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= cv[i];
    }
    state
}


fn words_from_bytes(b: &[u8]) -> [u32; 16] {
    let mut w = [0u32; 16];
    for i in 0..16 {
        w[i] = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
    }
    w
}


fn cv_of(out: &[u32; 16]) -> [u32; 8] {
    [out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]]
}


/// BLAKE3 of an arbitrary byte string via the native pipeline — the
/// differential bridge to the blake3 crate and to Hash256::concat.
pub fn hash_native(msg: &[u8]) -> [u8; 32] {
    let finish = |out: &[u32; 16]| {
        let mut d = [0u8; 32];
        for i in 0..8 {
            d[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        d
    };
    let compress_chunk = |chunk: &[u8], last_block_extra: u32| -> [u32; 8] {
        let nblocks = chunk.len().div_ceil(64).max(1);
        let mut cv = IV;
        let mut out = [0u32; 16];
        for bi in 0..nblocks {
            let start = bi * 64;
            let end = (start + 64).min(chunk.len());
            let mut block = [0u8; 64];
            block[..end - start].copy_from_slice(&chunk[start..end]);
            let mut flags = 0u32;
            if bi == 0 {
                flags |= CHUNK_START;
            }
            if bi == nblocks - 1 {
                flags |= CHUNK_END | last_block_extra;
            }
            out = compress_native(&cv, &words_from_bytes(&block), (end - start) as u8, 0, flags);
            cv = cv_of(&out);
        }
        cv
    };
    if msg.len() <= 1024 {
        let out_cv = compress_chunk(msg, ROOT);
        // For a single chunk the last compression's full output is the
        // root output; re-run to capture it.
        let nblocks = msg.len().div_ceil(64).max(1);
        let start = (nblocks - 1) * 64;
        let end = (start + 64).min(msg.len());
        let mut block = [0u8; 64];
        block[..end - start].copy_from_slice(&msg[start..end]);
        let mut cv = IV;
        let mut out = [0u32; 16];
        for bi in 0..nblocks {
            let s = bi * 64;
            let e = (s + 64).min(msg.len());
            let mut blk = [0u8; 64];
            blk[..e - s].copy_from_slice(&msg[s..e]);
            let mut flags = 0u32;
            if bi == 0 {
                flags |= CHUNK_START;
            }
            if bi == nblocks - 1 {
                flags |= CHUNK_END | ROOT;
            }
            out = compress_native(&cv, &words_from_bytes(&blk), (e - s) as u8, 0, flags);
            cv = cv_of(&out);
        }
        let _ = out_cv;
        finish(&out)
    } else {
        let chunks: Vec<&[u8]> = msg.chunks(1024).collect();
        let cvs: Vec<[u32; 8]> = chunks.iter().map(|&c| compress_chunk(c, 0)).collect();
        fn sub(cvs: &[[u32; 8]]) -> [u32; 8] {
            if cvs.len() == 1 {
                return cvs[0];
            }
            let half = 1usize << (cvs.len() - 1).ilog2();
            let (l, r) = (sub(&cvs[..half]), sub(&cvs[half..]));
            let mut block = [0u32; 16];
            block[..8].copy_from_slice(&l);
            block[8..].copy_from_slice(&r);
            cv_of(&compress_native(&IV, &block, 64, 0, PARENT))
        }
        let half = 1usize << (cvs.len() - 1).ilog2();
        let (l, r) = (sub(&cvs[..half]), sub(&cvs[half..]));
        let mut block = [0u32; 16];
        block[..8].copy_from_slice(&l);
        block[8..].copy_from_slice(&r);
        finish(&compress_native(&IV, &block, 64, 0, PARENT | ROOT))
    }
}


// ---------------------------------------------------------------------------
// The chip
// ---------------------------------------------------------------------------


#[derive(Clone, Copy, Debug)]
pub struct Blake3Compression {
    pub col_base: usize,
    pub prep_base: usize,
}


impl Blake3Compression {
    pub const fn new() -> Blake3Compression {
        Blake3Compression { col_base: 0, prep_base: 0 }
    }


    pub const fn at(col_base: usize, prep_base: usize) -> Blake3Compression {
        Blake3Compression { col_base, prep_base }
    }


    pub const fn rows() -> usize {
        ROWS
    }
}


impl Default for Blake3Compression {
    fn default() -> Self {
        Blake3Compression::new()
    }
}


fn recompose<B: AirBuilder>(builder: &mut B, base: usize, n: usize) -> B::Expr {
    let mut sum = B::constant(0);
    for j in 0..n {
        sum = sum + builder.witness(0, base + j) * B::constant(1u64 << j);
    }
    sum
}


fn assert_bits<B: AirBuilder>(builder: &mut B, base: usize, n: usize, gate: &B::Expr, name: &'static str) {
    for j in 0..n {
        let b = builder.witness(0, base + j);
        let e = b.clone() * (b - B::constant(1));
        builder.assert_zero(gate.clone() * e, name);
    }
}


fn rot_recompose<B: AirBuilder>(builder: &mut B, base: usize, n: u32) -> B::Expr {
    let mut sum = B::constant(0);
    for j in 0..32 {
        let bit = builder.witness(0, base + ((j + n as usize) % 32));
        sum = sum + bit * B::constant(1u64 << j);
    }
    sum
}


fn xor_bits<B: AirBuilder>(
    builder: &mut B,
    x_base: usize,
    y_base: usize,
    z_base: usize,
    gate: &B::Expr,
    name: &'static str,
) {
    for j in 0..32 {
        let x = builder.witness(0, x_base + j);
        let y = builder.witness(0, y_base + j);
        let z = builder.witness(0, z_base + j);
        let e = z - x.clone() - y.clone() + x * y * B::constant(2);
        builder.assert_zero(gate.clone() * e, name);
    }
}


impl<B: AirBuilder> Air<B> for Blake3Compression {
    fn eval(&self, builder: &mut B) {
        let cb = self.col_base;
        let pb = self.prep_base;
        let is_init = builder.preprocessed(pb + P_INIT);
        let is_g = builder.preprocessed(pb + P_G);
        let is_fin = builder.preprocessed(pb + P_FIN);
        let copy = is_init.clone() + is_g.clone() + is_fin.clone();


        // Register copies (within-window transitions).
        for k in 0..16 {
            let a = builder.witness(0, cb + MSG + k);
            let b = builder.witness(1, cb + MSG + k);
            builder.assert_zero(copy.clone() * (b - a), "bl3_msg_copy");
        }
        for k in 0..8 {
            let a = builder.witness(0, cb + CV + k);
            let b = builder.witness(1, cb + CV + k);
            builder.assert_zero(copy.clone() * (b - a), "bl3_cv_copy");
        }


        // Init row: state = cv ‖ IV ‖ (len, ctr, flags).
        for i in 0..8 {
            let s = builder.witness(0, cb + STATE + i);
            let c = builder.witness(0, cb + CV + i);
            builder.assert_zero(is_init.clone() * (s - c), "bl3_init_cv");
        }
        for i in 0..4 {
            let s = builder.witness(0, cb + STATE + 8 + i);
            let iv = B::constant(u64::from(IV[i]));
            builder.assert_zero(is_init.clone() * (s - iv), "bl3_init_iv");
        }
        for i in 0..4 {
            let s = builder.witness(0, cb + STATE + 12 + i);
            let p = builder.preprocessed(pb + P_PARAM + i);
            builder.assert_zero(is_init.clone() * (s - p), "bl3_init_param");
        }


        // Msg u32 checks at row 0 (bits in [GINB, GINB+512)).
        for i in 0..16 {
            assert_bits(builder, cb + GINB + 32 * i, 32, &is_init, "bl3_msgbit");
            let m = builder.witness(0, cb + MSG + i);
            let r = recompose(builder, cb + GINB + 32 * i, 32);
            builder.assert_zero(is_init.clone() * (m - r), "bl3_msg_recompose");
        }


        // G rows.
        let mx = builder.witness(0, cb + MX);
        let mut mx_sel = B::constant(0);
        for k in 0..16 {
            mx_sel = mx_sel
                + builder.preprocessed(pb + P_SELMSG + k) * builder.witness(0, cb + MSG + k);
        }
        builder.assert_zero(is_g.clone() * (mx.clone() - mx_sel), "bl3_mx_sel");


        for i in 0..4 {
            let gi = builder.witness(0, cb + GIN + i);
            let mut sel = B::constant(0);
            for j in 0..16 {
                sel = sel
                    + builder.preprocessed(pb + P_SELIN + 16 * i + j)
                        * builder.witness(0, cb + STATE + j);
            }
            builder.assert_zero(is_g.clone() * (gi.clone() - sel), "bl3_gin_sel");
        }
        for i in 0..4 {
            assert_bits(builder, cb + GINB + 32 * i, 32, &is_g, "bl3_ginbit");
            let r = recompose(builder, cb + GINB + 32 * i, 32);
            builder.assert_zero(is_g.clone() * (r - builder.witness(0, cb + GIN + i)), "bl3_gin_recompose");
        }


        let (a, b, c, d) = (
            builder.witness(0, cb + GIN),
            builder.witness(0, cb + GIN + 1),
            builder.witness(0, cb + GIN + 2),
            builder.witness(0, cb + GIN + 3),
        );
        let (a1, b1, c1, d1) = (
            builder.witness(0, cb + GV),
            builder.witness(0, cb + GV + 1),
            builder.witness(0, cb + GV + 2),
            builder.witness(0, cb + GV + 3),
        );
        let (a2, b2, c2, d2) = (
            builder.witness(0, cb + GV + 4),
            builder.witness(0, cb + GV + 5),
            builder.witness(0, cb + GV + 6),
            builder.witness(0, cb + GV + 7),
        );
        let (k1, k2, k3, k4) = (
            builder.witness(0, cb + K),
            builder.witness(0, cb + K + 1),
            builder.witness(0, cb + K + 2),
            builder.witness(0, cb + K + 3),
        );


        // Wrapping adds with range-checked carries.
        builder.assert_zero(
            is_g.clone() * (a.clone() + b.clone() + mx.clone() - a1.clone() - B::constant(TWO32) * k1.clone()),
            "bl3_add1",
        );
        builder.assert_zero(is_g.clone() * (k1.clone() * (k1.clone() - B::constant(1)) * (k1 - B::constant(2))), "bl3_k1range");
        builder.assert_zero(
            is_g.clone() * (c.clone() + d1.clone() - c1.clone() - B::constant(TWO32) * k2.clone()),
            "bl3_add2",
        );
        builder.assert_zero(is_g.clone() * (k2.clone() * (k2.clone() - B::constant(1))), "bl3_k2range");
        builder.assert_zero(
            is_g.clone() * (a1.clone() + b1.clone() + mx.clone() - a2.clone() - B::constant(TWO32) * k3.clone()),
            "bl3_add3",
        );
        builder.assert_zero(is_g.clone() * (k3.clone() * (k3.clone() - B::constant(1)) * (k3 - B::constant(2))), "bl3_k3range");
        builder.assert_zero(
            is_g.clone() * (c1.clone() + d2.clone() - c2.clone() - B::constant(TWO32) * k4.clone()),
            "bl3_add4",
        );
        builder.assert_zero(is_g.clone() * (k4.clone() * (k4.clone() - B::constant(1))), "bl3_k4range");


        // Intermediate bit decompositions (u32 forcing).
        for t in 0..4 {
            assert_bits(builder, cb + MIDB + 32 * t, 32, &is_g, "bl3_midbit");
            let r = recompose(builder, cb + MIDB + 32 * t, 32);
            let v = builder.witness(0, cb + GV + 2 * t); // a1, c1, a2, c2 at GV[0,2,4,6]
            builder.assert_zero(is_g.clone() * (r - v), "bl3_mid_recompose");
        }
        for t in 0..2 {
            assert_bits(builder, cb + ROTB + 32 * t, 32, &is_g, "bl3_rotbit");
            let r = recompose(builder, cb + ROTB + 32 * t, 32);
            let v = builder.witness(0, cb + GV + 3 - 2 * t); // d1 at GV[3], b1 at GV[1]
            builder.assert_zero(is_g.clone() * (r - v), "bl3_rot_recompose");
        }


        // XOR identities.
        xor_bits(builder, cb + GINB + 96, cb + MIDB, cb + XORB, &is_g, "bl3_xor1");
        xor_bits(builder, cb + GINB + 32, cb + MIDB + 32, cb + XORB + 32, &is_g, "bl3_xor2");
        xor_bits(builder, cb + ROTB, cb + MIDB + 64, cb + XORB + 64, &is_g, "bl3_xor3");
        xor_bits(builder, cb + ROTB + 32, cb + MIDB + 96, cb + XORB + 96, &is_g, "bl3_xor4");


        // Rotations as recompositions.
        let rot16 = rot_recompose(builder, cb + XORB, 16);
        builder.assert_zero(is_g.clone() * (d1 - rot16), "bl3_rot16");
        let rot12 = rot_recompose(builder, cb + XORB + 32, 12);
        builder.assert_zero(is_g.clone() * (b1 - rot12), "bl3_rot12");
        let rot8 = rot_recompose(builder, cb + XORB + 64, 8);
        builder.assert_zero(is_g.clone() * (d2 - rot8), "bl3_rot8");
        let rot7 = rot_recompose(builder, cb + XORB + 96, 7);
        builder.assert_zero(is_g.clone() * (b2 - rot7), "bl3_rot7");


        // State blend: next_state = G outputs at written positions.
        let out_recomps: Vec<B::Expr> = (0..2)
            .map(|w| recompose(builder, cb + FXO + 32 * w, 32))
            .collect();
        for j in 0..16 {
            let mut blend = B::constant(0);
            for t in 0..4 {
                blend = blend
                    + builder.preprocessed(pb + P_WSTATE + 5 * j + t)
                        * builder.witness(0, cb + GV + 4 + t);
            }
            blend = blend
                + builder.preprocessed(pb + P_WSTATE + 5 * j + 4)
                    * builder.witness(0, cb + STATE + j);
            let pass = builder.witness(0, cb + STATE + j);
            let val = is_g.clone() * blend + (is_init.clone() + is_fin.clone()) * pass;
            let nxt = builder.witness(1, cb + STATE + j);
            builder.assert_zero(copy.clone() * (nxt - val), "bl3_state_step");
        }


        // Final-XOR rows: two output words per row.
        for w in 0..2 {
            let mut a_sel = B::constant(0);
            for j in 0..16 {
                a_sel = a_sel
                    + builder.preprocessed(pb + P_FXA + 16 * w + j)
                        * builder.witness(0, cb + STATE + j);
            }
            let mut b_sel = B::constant(0);
            for j in 0..16 {
                b_sel = b_sel
                    + builder.preprocessed(pb + P_FXB + 24 * w + j)
                        * builder.witness(0, cb + STATE + j);
            }
            for j in 0..8 {
                b_sel = b_sel
                    + builder.preprocessed(pb + P_FXB + 24 * w + 16 + j)
                        * builder.witness(0, cb + CV + j);
            }
            let ab = cb + FXOP + 64 * w;
            assert_bits(builder, ab, 32, &is_fin, "bl3_fxabit");
            assert_bits(builder, ab + 32, 32, &is_fin, "bl3_fxubit");
            let a_rec = recompose(builder, ab, 32);
            builder.assert_zero(
                is_fin.clone() * (a_rec - a_sel),
                "bl3_fxa_recompose",
            );
            let b_rec = recompose(builder, ab + 32, 32);
            builder.assert_zero(
                is_fin.clone() * (b_rec - b_sel),
                "bl3_fxb_recompose",
            );
            assert_bits(builder, cb + FXO + 32 * w, 32, &is_fin, "bl3_fxobit");
            for j in 0..32 {
                let x = builder.witness(0, ab + j);
                let y = builder.witness(0, ab + 32 + j);
                let z = builder.witness(0, cb + FXO + 32 * w + j);
                let e = z - x.clone() - y.clone() + x * y * B::constant(2);
                builder.assert_zero(is_fin.clone() * e, "bl3_fxxor");
            }
        }


        // Out-register threading: write this row's two words, pass the rest.
        for j in 0..16 {
            let mut wsel = B::constant(0);
            for w in 0..2 {
                wsel = wsel + builder.preprocessed(pb + P_FXW + 16 * w + j) * out_recomps[w].clone();
            }
            let any = builder.preprocessed(pb + P_FXW + j)
                + builder.preprocessed(pb + P_FXW + 16 + j);
            let cur = builder.witness(0, cb + OUT + j);
            let nxt = builder.witness(1, cb + OUT + j);
            let val = wsel + (B::constant(1) - any) * cur;
            builder.assert_zero(is_fin.clone() * (nxt - val), "bl3_out_step");
        }
    }
}


// ---------------------------------------------------------------------------
// Witness / prep generation
// ---------------------------------------------------------------------------


struct GInter {
    a1: u32, b1: u32, c1: u32, d1: u32,
    a2: u32, b2: u32, c2: u32, d2: u32,
    x1: u32, x2: u32, x3: u32, x4: u32,
    k: [u32; 4],
}


fn g_step(state: &mut [u32; 16], pos: [usize; 4], mx: u32) -> GInter {
    let (a, b, c, d) = (state[pos[0]], state[pos[1]], state[pos[2]], state[pos[3]]);
    let s1 = a as u64 + b as u64 + mx as u64;
    let a1 = (s1 & 0xFFFF_FFFF) as u32;
    let k1 = (s1 >> 32) as u32;
    let x1 = d ^ a1;
    let d1 = x1.rotate_right(16);
    let s2 = c as u64 + d1 as u64;
    let c1 = (s2 & 0xFFFF_FFFF) as u32;
    let k2 = (s2 >> 32) as u32;
    let x2 = b ^ c1;
    let b1 = x2.rotate_right(12);
    let s3 = a1 as u64 + b1 as u64 + mx as u64;
    let a2 = (s3 & 0xFFFF_FFFF) as u32;
    let k3 = (s3 >> 32) as u32;
    let x3 = d1 ^ a2;
    let d2 = x3.rotate_right(8);
    let s4 = c1 as u64 + d2 as u64;
    let c2 = (s4 & 0xFFFF_FFFF) as u32;
    let k4 = (s4 >> 32) as u32;
    let x4 = b1 ^ c2;
    let b2 = x4.rotate_right(7);
    state[pos[0]] = a2;
    state[pos[1]] = b2;
    state[pos[2]] = c2;
    state[pos[3]] = d2;
    GInter { a1, b1, c1, d1, a2, b2, c2, d2, x1, x2, x3, x4, k: [k1, k2, k3, k4] }
}


fn g_positions(grp: usize) -> [usize; 4] {
    if grp < 4 {
        [4 * grp, 4 * grp + 1, 4 * grp + 2, 4 * grp + 3]
    } else {
        let j = grp - 4;
        [DIAG[4 * j], DIAG[4 * j + 1], DIAG[4 * j + 2], DIAG[4 * j + 3]]
    }
}


fn permuted_index(mut i: usize, power: usize) -> usize {
    for _ in 0..power {
        i = MSG_PERMUTATION[i];
    }
    i
}


fn put_word(row: &mut [Goldilocks], col: usize, v: u32) {
    row[col] = Goldilocks::from_u64_reduce(u64::from(v));
}


fn put_bits(row: &mut [Goldilocks], col: usize, v: u32) {
    for j in 0..32 {
        row[col + j] = Goldilocks::from_u32((v >> j) & 1);
    }
}


/// The prep table for one compression window.
pub fn gen_prep(len: u8, counter: u64, flags: u32) -> Vec<Vec<Goldilocks>> {
    let mut prep = vec![vec![Goldilocks::ZERO; PREP_COLS]; ROWS];
    prep[0][P_INIT] = Goldilocks::ONE;
    put_word(&mut prep[0], P_PARAM, u32::from(len));
    put_word(&mut prep[0], P_PARAM + 1, counter as u32);
    put_word(&mut prep[0], P_PARAM + 2, (counter >> 32) as u32);
    put_word(&mut prep[0], P_PARAM + 3, flags);
    for r in 1..=56 {
        let g = r - 1;
        let round = 1 + g / 8;
        let grp = g % 8;
        prep[r][P_G] = Goldilocks::ONE;
        prep[r][P_SELMSG + permuted_index(grp, round - 1)] = Goldilocks::ONE;
        let pos = g_positions(grp);
        for i in 0..4 {
            prep[r][P_SELIN + 16 * i + pos[i]] = Goldilocks::ONE;
        }
        for j in 0..16 {
            match pos.iter().position(|&p| p == j) {
                Some(i) => prep[r][P_WSTATE + 5 * j + i] = Goldilocks::ONE,
                None => prep[r][P_WSTATE + 5 * j + 4] = Goldilocks::ONE,
            }
        }
    }
    for r in 57..=64 {
        let p = r - 57;
        prep[r][P_FIN] = Goldilocks::ONE;
        for w in 0..2 {
            let i = 2 * p + w;
            prep[r][P_FXA + 16 * w + i] = Goldilocks::ONE;
            if i < 8 {
                prep[r][P_FXB + 24 * w + (i + 8)] = Goldilocks::ONE;
            } else {
                prep[r][P_FXB + 24 * w + 16 + (i - 8)] = Goldilocks::ONE;
            }
            prep[r][P_FXW + 16 * w + i] = Goldilocks::ONE;
        }
    }
    prep
}


/// The witness trace for one compression.
pub fn gen_trace(
    cv: &[u32; 8],
    block: &[u32; 16],
    len: u8,
    counter: u64,
    flags: u32,
) -> Vec<Vec<Goldilocks>> {
    let mut state: [u32; 16] = [
        cv[0], cv[1], cv[2], cv[3], cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        u32::from(len), counter as u32, (counter >> 32) as u32, flags,
    ];
    let mut trace = vec![vec![Goldilocks::ZERO; WITNESS_COLS]; ROWS];
    for r in 0..ROWS {
        let row = &mut trace[r];
        for j in 0..16 {
            put_word(row, STATE + j, state[j]);
            put_word(row, MSG + j, block[j]);
        }
        for j in 0..8 {
            put_word(row, CV + j, cv[j]);
        }
    }
    // Row 0: msg u32 bits.
    for i in 0..16 {
        put_bits(&mut trace[0], GINB + 32 * i, block[i]);
    }
    // G rows.
    for r in 1..=56 {
        let g = r - 1;
        let round = 1 + g / 8;
        let grp = g % 8;
        let pos = g_positions(grp);
        let mx = block[permuted_index(grp, round - 1)];
        let gi = g_step(&mut state, pos, mx);
        // Compute `state_before` for all 4 positions BEFORE taking a
        // mutable borrow of `trace[r]` -- the borrow checker can't see
        // that `state_before` only reads `trace[r - 1]`, so we have
        // to land the values into locals first.
        let gins: [u32; 4] = [
            state_before(&trace, r, pos[0]),
            state_before(&trace, r, pos[1]),
            state_before(&trace, r, pos[2]),
            state_before(&trace, r, pos[3]),
        ];
        let row = &mut trace[r];
        put_word(row, MX, mx);
        for i in 0..4 {
            put_word(row, GIN + i, gins[i]);
        }
        put_word(row, GV, gi.a1);
        put_word(row, GV + 1, gi.b1);
        put_word(row, GV + 2, gi.c1);
        put_word(row, GV + 3, gi.d1);
        put_word(row, GV + 4, gi.a2);
        put_word(row, GV + 5, gi.b2);
        put_word(row, GV + 6, gi.c2);
        put_word(row, GV + 7, gi.d2);
        for t in 0..4 {
            put_word(row, K + t, gi.k[t]);
        }
        for i in 0..4 {
            let gin_word = word_at(row, GIN + i);
            put_bits(row, GINB + 32 * i, gin_word);
        }
        put_bits(row, MIDB, gi.a1);
        put_bits(row, MIDB + 32, gi.c1);
        put_bits(row, MIDB + 64, gi.a2);
        put_bits(row, MIDB + 96, gi.c2);
        put_bits(row, ROTB, gi.d1);
        put_bits(row, ROTB + 32, gi.b1);
        put_bits(row, XORB, gi.x1);
        put_bits(row, XORB + 32, gi.x2);
        put_bits(row, XORB + 64, gi.x3);
        put_bits(row, XORB + 96, gi.x4);
    }
    // Final rows: out[i] = state[i] ^ state[i+8]; out[8+i] = state[8+i] ^ cv[i].
    let fin = state;
    let mut out_words = [0u32; 16];
    for i in 0..8 {
        out_words[i] = fin[i] ^ fin[i + 8];
        out_words[i + 8] = fin[i + 8] ^ cv[i];
    }
    for r in 57..=64 {
        let p = r - 57;
        let row = &mut trace[r];
        for w in 0..2 {
            let i = 2 * p + w;
            let (av, bv) = if i < 8 { (fin[i], fin[i + 8]) } else { (fin[i], cv[i - 8]) };
            put_bits(row, FXOP + 64 * w, av);
            put_bits(row, FXOP + 64 * w + 32, bv);
            put_bits(row, FXO + 32 * w, out_words[i]);
        }
    }
    // Out-register threading: row r holds outputs written by rows < r.
    for r in 58..=65 {
        let p = r - 57; // outputs 0..2p written so far
        for j in 0..(2 * p).min(16) {
            put_word(&mut trace[r], OUT + j, out_words[j]);
        }
    }
    trace
}


fn state_before(trace: &[Vec<Goldilocks>], r: usize, pos: usize) -> u32 {
    trace[r - 1][STATE + pos].as_u64() as u32
}


fn word_at(row: &[Goldilocks], col: usize) -> u32 {
    row[col].as_u64() as u32
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::{Air, AirBuilder, NativeEval};
    use crate::testutil::SplitMix64;


    fn check(cv: &[u32; 8], block: &[u32; 16], len: u8, ctr: u64, flags: u32) -> Vec<Vec<Goldilocks>> {
        let trace = gen_trace(cv, block, len, ctr, flags);
        let prep = gen_prep(len, ctr, flags);
        NativeEval::check_with_prep(trace.clone(), prep, vec![], &Blake3Compression::new(), 8).unwrap();
        trace
    }


    #[test]
    fn layout_pins() {
        assert_eq!(ROWS, 66);
        assert_eq!(WITNESS_COLS, 713);
        assert_eq!(PREP_COLS, 280);
        assert_eq!(ROWS * WITNESS_COLS, 47_058);
    }


    #[test]
    fn native_pipeline_matches_blake3_crate() {
        let mut rng = SplitMix64::new(0xB3A1);
        for len in [0usize, 1, 7, 8, 63, 64, 65, 72, 127, 128, 1023, 1024, 1025, 1088, 1263, 2048, 2049, 3072, 4096, 5000] {
            let msg: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let expect = blake3::hash(&msg);
            let got = hash_native(&msg);
            assert_eq!(&got, expect.as_bytes(), "length {len}");
        }
    }


    #[test]
    fn native_pipeline_matches_hash256_concat() {
        use nerv_core::constants::{NOTE_COMMITMENT, NULLIFIER, TXID};
        use nerv_core::hash::Hash256;
        let mut rng = SplitMix64::new(0xB3A2);
        for (dom, len) in [(NOTE_COMMITMENT, 1256usize), (NULLIFIER, 72), (TXID, 300), (TXID, 0)] {
            let msg: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
            let mut pre = Vec::new();
            pre.extend_from_slice(dom.as_bytes());
            pre.extend_from_slice(&msg);
            let got = hash_native(&pre);
            assert_eq!(got, *Hash256::concat(&dom, &msg).as_bytes(), "len {len}");
        }
    }


    #[test]
    fn chip_differential_and_output_register() {
        let mut rng = SplitMix64::new(0xB3A3);
        for t in 0..6 {
            let cv: [u32; 8] = core::array::from_fn(|_| (rng.next_u64() >> 32) as u32);
            let block: [u32; 16] = core::array::from_fn(|_| (rng.next_u64() >> 32) as u32);
            let len = (rng.next_u64() % 65) as u8;
            let ctr = rng.next_u64() % 1000;
            let flags = (rng.next_u64() % 16) as u32;
            let trace = check(&cv, &block, len, ctr, flags);
            let want = compress_native(&cv, &block, len, ctr, flags);
            for i in 0..16 {
                assert_eq!(
                    trace[ROWS - 1][OUT + i].as_u64(),
                    u64::from(want[i]),
                    "t={t} word {i}"
                );
            }
        }
        // Deterministic empty-block edge.
        let trace = check(&IV, &[0u32; 16], 0, 0, CHUNK_START | CHUNK_END | ROOT);
        let want = compress_native(&IV, &[0u32; 16], 0, 0, CHUNK_START | CHUNK_END | ROOT);
        for i in 0..16 {
            assert_eq!(trace[ROWS - 1][OUT + i].as_u64(), u64::from(want[i]));
        }
    }


    #[test]
    fn tamper_classes_rejected() {
        let mut rng = SplitMix64::new(0xB3A4);
        let cv: [u32; 8] = core::array::from_fn(|_| (rng.next_u64() >> 32) as u32);
        let block: [u32; 16] = core::array::from_fn(|_| (rng.next_u64() >> 32) as u32);
        let (len, flags) = (64u8, CHUNK_START);
        let trace = gen_trace(&cv, &block, len, 0, flags);
        let prep = gen_prep(len, 0, flags);
        let chip = Blake3Compression::new();
        assert!(NativeEval::check_with_prep(trace.clone(), prep.clone(), vec![], &chip, 8).is_ok());


        let mut bad = trace.clone();
        bad[20][STATE + 5] = bad[20][STATE + 5] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err(), "state cell");


        let mut bad = trace.clone();
        bad[20][XORB + 7] = bad[20][XORB + 7] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err(), "xor bit");


        let mut bad = trace.clone();
        bad[20][K] = Goldilocks::from_u32(3);
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err(), "carry");


        let mut bad = trace.clone();
        bad[0][GINB + 3] = bad[0][GINB + 3] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err(), "msg bit");


        let mut bad = trace.clone();
        bad[ROWS - 1][OUT + 3] = bad[ROWS - 1][OUT + 3] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, prep.clone(), vec![], &chip, 8).is_err(), "out slot");


        let mut bad = trace.clone();
        bad[0][STATE + 12] = Goldilocks::from_u32(63);
        assert!(NativeEval::check_with_prep(bad, prep, vec![], &chip, 8).is_err(), "init param");


        // Zero padding beyond the window passes (composition soundness).
        let mut padded = trace.clone();
        let mut padded_prep = gen_prep(len, 0, flags);
        for _ in 0..10 {
            padded.push(vec![Goldilocks::ZERO; WITNESS_COLS]);
            padded_prep.push(vec![Goldilocks::ZERO; PREP_COLS]);
        }
        assert!(NativeEval::check_with_prep(padded, padded_prep, vec![], &chip, 8).is_ok());
    }


    #[test]
    fn two_windows_chained_hash() {
        // 72-byte hash (the nullifier shape) as two chained windows.
        let mut rng = SplitMix64::new(0xB3A5);
        let msg: Vec<u8> = (0..72).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let mut b0 = [0u8; 64];
        b0.copy_from_slice(&msg[..64]);
        let mut b1 = [0u8; 64];
        b1[..8].copy_from_slice(&msg[64..]);


        struct TwoWindowAir {
            chip: Blake3Compression,
            chain_prep: usize,
        }
        impl<B: AirBuilder> Air<B> for TwoWindowAir {
            fn eval(&self, builder: &mut B) {
                self.chip.eval(builder);
                let chain = builder.preprocessed(self.chain_prep);
                for j in 0..8 {
                    let o = builder.witness(0, OUT + j);
                    let c = builder.witness(1, CV + j);
                    builder.assert_zero(chain.clone() * (c - o), "chain_cv");
                }
                let first = builder.is_first_row();
                for j in 0..8 {
                    let c = builder.witness(0, CV + j);
                    let iv = B::constant(u64::from(IV[j]));
                    builder.assert_zero(first.clone() * (c - iv), "chain_iv");
                }
            }
        }


        let t0 = gen_trace(&IV, &words_from_bytes(&b0), 64, 0, CHUNK_START);
        let out0 = [
            t0[ROWS - 1][OUT].as_u64() as u32, t0[ROWS - 1][OUT + 1].as_u64() as u32,
            t0[ROWS - 1][OUT + 2].as_u64() as u32, t0[ROWS - 1][OUT + 3].as_u64() as u32,
            t0[ROWS - 1][OUT + 4].as_u64() as u32, t0[ROWS - 1][OUT + 5].as_u64() as u32,
            t0[ROWS - 1][OUT + 6].as_u64() as u32, t0[ROWS - 1][OUT + 7].as_u64() as u32,
        ];
        let t1 = gen_trace(&out0, &words_from_bytes(&b1), 8, 0, CHUNK_END | ROOT);


        let mut trace: Vec<Vec<Goldilocks>> = Vec::new();
        trace.extend(t0);
        trace.extend(t1);
        let chain_col = PREP_COLS; // the caller's own prep column
        let mut prep: Vec<Vec<Goldilocks>> = Vec::new();
        prep.extend(gen_prep(64, 0, CHUNK_START));
        prep.extend(gen_prep(8, 0, CHUNK_END | ROOT));
        for row in prep.iter_mut() {
            row.push(Goldilocks::ZERO);
        }
        prep[ROWS - 1][chain_col] = Goldilocks::ONE;


        let air = TwoWindowAir { chip: Blake3Compression::new(), chain_prep: chain_col };
        assert!(NativeEval::check_with_prep(trace, prep, vec![], &air, 8).is_ok());


        let want = blake3::hash(&msg);
        for i in 0..8 {
            let got = trace[2 * ROWS - 1][OUT + i].as_u64();
            let mut wb = [0u8; 4];
            wb.copy_from_slice(&want.as_bytes()[4 * i..4 * i + 4]);
            assert_eq!(got, u64::from(u32::from_le_bytes(wb)), "digest word {i}");
        }
    }
}
