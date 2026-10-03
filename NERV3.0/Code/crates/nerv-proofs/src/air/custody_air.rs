//! The custody module AIR (WP §5.1 statements 1–5, as amended by erratum
//! 72): n_inputs Merkle memberships (statement 1), per-input cm / nf /
//! pk_n BLAKE3 streams and per-output cm streams (statements 2–3, output
//! binding), conservation (statement 4), ranges (statement 5).
//!
//! Layout: columns [0, 53·n_in) Merkle instances (side-by-side, shared
//! 2,145-row pattern) ‖ [.., +171) conservation ‖ [.., +713) the shared
//! BLAKE3 chip (all hash windows share these columns; phase-multiplexed)
//! ‖ registers. Rows = 66·(25·n_in + 22·n_out) (windows: per input 22 cm
//! + 2 nf + 1 pk_n; per output 22 cm). Preprocessed: Merkle [0..25) ‖
//! conservation [25..31) ‖ BLAKE3 [31..311) ‖ boundary flags [311..).
//! Publics: 4 anchor elements per input ‖ 8 nf words per input ‖ 8 cm
//! words per output.
//!
//! Shared-secret registers (row-invariant copies — the cross-statement
//! consistency load): per input v(64b), ρ(256b), nk(256b), pk_n (8 words
//! + 256b, unconditionally decomposed), cm words (8); per output v(64b);
//! one aux CV (8 words, per-stream cv0 with load holes). The wiring:
//! v/ρ/nk/pk_n bits into the hash windows' message decompositions (the
//! byte misalignment from the 7/11-byte domain prefixes is why wiring is
//! bit-level); nf = H("nerv.nf"‖nk‖ρ) with the same nk whose
//! H("nerv.nf.pk"‖nk) sits in the cm — the erratum-72 binding; input cm
//! → Merkle leaf (row-0 binding); output cm → public.

use nerv_core::field::Goldilocks;
use nerv_core::hash::Hash256;

use nerv_custody::commitment::{nullifier_pk, NoteOpening};
use nerv_custody::nct::NctDigest;
use nerv_custody::nullifier::derive_nullifier;
use nerv_custody::tx::TransactionShell;

use crate::air::builder::{Air, AirBuilder};
use crate::air::chips::blake3::{
    compress_native, gen_prep as bl3_prep, gen_trace as bl3_trace, Blake3Compression, IV as BL3_IV,
    CHUNK_END, CHUNK_START, PARENT, ROOT,
};
use crate::air::chips::blake3::{CV as BL3_CV, GINB as BL3_GINB, MSG as BL3_MSG, OUT as BL3_OUT};
use crate::air::chips::conservation::{
    gen_cons_prep, gen_cons_trace, ConservationChip, V_BITS_BASE,
};
use crate::air::chips::merkle_poseidon2::{
    gen_merkle_trace, MerkleChip, PERM_ROWS as MERKLE_PERM, STATE_COLS as MERKLE_STATE,
    WITNESS_COLS as MERKLE_W,
};

use crate::error::WitnessError;
use nerv_core::types::LegIndex;
use nerv_custody::burn::BurnCommitment;


pub const BL3_W: usize = crate::air::chips::blake3::WITNESS_COLS; // 713
pub const BL3_PREP_W: usize = crate::air::chips::blake3::PREP_COLS; // 280
pub const CONS_W: usize = 171;
pub const MERKLE_PREP_W: usize = 25;
pub const CONS_PREP_BASE: usize = 25;
pub const BL3_PREP_BASE: usize = 31;
pub const BP: usize = BL3_PREP_BASE + BL3_PREP_W; // 311

pub const REG_V: usize = 0;
pub const REG_RHO: usize = 64;
pub const REG_NK: usize = 320;
pub const REG_PKN_W: usize = 576;
pub const REG_PKN_B: usize = 584;
pub const REG_CM: usize = 840;
pub const INPUT_REG_W: usize = 848;
pub const OUT_REG_W: usize = 64;

// boundary flag offsets (from BP)
const F_CHAIN: usize = 0;
const F_IV: usize = 1;
const F_PARENT: usize = 2;
const F_PCV1: usize = 3;
const F_CVLOAD: usize = 4;
pub const IN_FLAGS: usize = 5; // stride 12: TGT_NF, TGT_CMR, TGT_PKN, W_V, W_RHO_CM, W_RHO_NF, W_RHO_NF2, W_NK_NF, W_NK_PK, W_PKN_A, W_PKN_B, CV_IN
const OUT_FLAGS: usize = 0; // stride 3 (at BP+5+12n_in): TGT_CM, CV_OUT, W_V

pub const REV_REG_W: usize = 64;
pub const BURN_REG_W: usize = 64;

/// Windows per input stream: cm(22) + nf(2) + pkn(1).
pub const CM_WINS: usize = 22;
pub const IN_WINS: usize = 25;
pub const OUT_WINS: usize = 22;

#[derive(Clone, Debug)]
pub struct CustodyAir {
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub depth: usize,
    pub cond_outputs: Vec<usize>,
    pub n_burns: usize,
}

impl CustodyAir {
    pub fn new(
        n_inputs: usize,
        n_outputs: usize,
        cond_outputs: Vec<usize>,
        n_burns: usize,
        depth: usize,
    ) -> CustodyAir {
        CustodyAir { n_inputs, n_outputs, cond_outputs, n_burns, depth }
    }

    pub fn n_cond(&self) -> usize {
        self.cond_outputs.len()
    }

    pub fn n_windows(&self) -> usize {
        IN_WINS * self.n_inputs + OUT_WINS * self.n_outputs + 22 * self.n_cond() + self.n_burns
    }

    pub fn cols(&self) -> usize {
        MERKLE_W * self.n_inputs + CONS_W + BL3_W + INPUT_REG_W * self.n_inputs
            + OUT_REG_W * self.n_outputs
            + REV_REG_W * self.n_cond() + BURN_REG_W * self.n_burns
            + 8
    }

    pub fn prep_cols(&self) -> usize {
        BP + 5 + 12 * self.n_inputs + 3 * self.n_outputs + self.n_cond() + 3 * self.n_burns
    }

    pub fn rev_reg(&self, k: usize) -> usize {
        self.aux_base() + 8 + REV_REG_W * k
    }


    pub fn burn_reg(&self, k: usize) -> usize {
        self.rev_reg(self.n_cond()) + BURN_REG_W * k
    }

    pub fn rev_flag(&self, k: usize) -> usize {
        BP + 5 + 12 * self.n_inputs + 3 * self.n_outputs + k
    }

    pub fn burn_flag(&self, k: usize) -> usize {
        self.rev_flag(self.n_cond()) + 3 * k
    }

    pub fn rows(&self) -> usize {
        let w = 66 * self.n_windows();
        let m = (self.depth + 1) * MERKLE_PERM;
        w.max(m)
    }

    pub fn cons_base(&self) -> usize {
        MERKLE_W * self.n_inputs
    }

    pub fn bl3_base(&self) -> usize {
        self.cons_base() + CONS_W
    }

    pub fn reg_base(&self) -> usize {
        self.bl3_base() + BL3_W
    }

    fn in_reg(&self, i: usize) -> usize {
        self.reg_base() + INPUT_REG_W * i
    }

    fn out_reg(&self, o: usize) -> usize {
        self.reg_base() + INPUT_REG_W * self.n_inputs + OUT_REG_W * o
    }

    fn aux_base(&self) -> usize {
        self.reg_base() + INPUT_REG_W * self.n_inputs + OUT_REG_W * self.n_outputs
    }
}

impl<B: AirBuilder> Air<B> for CustodyAir {
    fn eval(&self, b: &mut B) {
        for i in 0..self.n_inputs {
            MerkleChip::at(self.depth, MERKLE_W * i, 4 * i, 0).eval(b);
        }
        ConservationChip::at(self.cons_base(), CONS_PREP_BASE).eval(b);
        Blake3Compression::at(self.bl3_base(), BL3_PREP_BASE).eval(b);

        let one = B::constant(1);
        let last = b.is_last_row();
        let not_last = one.clone() - last;
        let bl3 = self.bl3_base();
        let n_in = self.n_inputs;
        let n_out = self.n_outputs;

        // register copies
        for i in 0..n_in {
            let rb = self.in_reg(i);
            for c in 0..INPUT_REG_W {
                let cur = b.witness(0, rb + c);
                let nxt = b.witness(1, rb + c);
                b.assert_zero(not_last.clone() * (nxt - cur), "reg_copy");
            }
        }
        for o in 0..n_out {
            let rb = self.out_reg(o);
            for c in 0..OUT_REG_W {
                let cur = b.witness(0, rb + c);
                let nxt = b.witness(1, rb + c);
                b.assert_zero(not_last.clone() * (nxt - cur), "reg_copy");
            }
        }
        let cvload = b.preprocessed(BP + F_CVLOAD);
        for j in 0..8 {
            let cur = b.witness(0, self.aux_base() + j);
            let nxt = b.witness(1, self.aux_base() + j);
            b.assert_zero(not_last.clone() * (one.clone() - cvload.clone()) * (nxt - cur), "aux_copy");
        }

        // pk_n register decompose (unconditional): bits boolean, words recompose.
        for i in 0..n_in {
            let rb = self.in_reg(i);
            for t in 0..256 {
                let bit = b.witness(0, rb + REG_PKN_B + t);
                let e = bit.clone() * (bit - one.clone());
                b.assert_zero(e, "pkn_bit");
            }
            for w in 0..8 {
                let mut sum = B::constant(0);
                for t in 0..32 {
                    sum = sum + b.witness(0, rb + REG_PKN_B + 32 * w + t) * B::constant(1u64 << t);
                }
                let word = b.witness(0, rb + REG_PKN_W + w);
                b.assert_zero(word - sum, "pkn_recompose");
            }
        }

        // window chaining / IV starts / parent wiring
        let chain = b.preprocessed(BP + F_CHAIN);
        for j in 0..8 {
            let o = b.witness(0, bl3 + BL3_OUT + j);
            let c = b.witness(1, bl3 + BL3_CV + j);
            b.assert_zero(chain.clone() * (o - c), "hb_chain");
        }
        let iv = b.preprocessed(BP + F_IV);
        for j in 0..8 {
            let c = b.witness(0, bl3 + BL3_CV + j);
            let ivj = B::constant(u64::from(BL3_IV[j]));
            b.assert_zero(iv.clone() * (c - ivj), "hb_iv");
        }
        let parent = b.preprocessed(BP + F_PARENT);
        for j in 0..8 {
            let m = b.witness(0, bl3 + BL3_MSG + j);
            let a = b.witness(0, self.aux_base() + j);
            b.assert_zero(parent.clone() * (m - a), "hb_parentmsg");
        }
        let pcv1 = b.preprocessed(BP + F_PCV1);
        for j in 0..8 {
            let o = b.witness(0, bl3 + BL3_OUT + j);
            let m = b.witness(1, bl3 + BL3_MSG + 8 + j);
            b.assert_zero(pcv1.clone() * (o - m), "hb_pcv1");
        }
       let load = b.preprocessed(BP + F_CVLOAD);
        for j in 0..8 {
            let o = b.witness(0, bl3 + BL3_OUT + j);
            let a = b.witness(1, self.aux_base() + j);
            b.assert_zero(load.clone() * (o - a), "hb_cvload");
        }

        // targets
        for i in 0..n_in {
            let f = b.preprocessed(BP + IN_FLAGS + 12 * i);
            for w in 0..8 {
                let o = b.witness(0, bl3 + BL3_OUT + w);
                let p = b.public(4 * n_in + 8 * i + w);
                b.assert_zero(f.clone() * (o - p), "hb_tgt_nf");
            }
            let f = b.preprocessed(BP + IN_FLAGS + 12 * i + 1);
            for w in 0..8 {
                let o = b.witness(0, bl3 + BL3_OUT + w);
                let r = b.witness(0, self.in_reg(i) + REG_CM + w);
                b.assert_zero(f.clone() * (o - r), "hb_tgt_cmr");
            }
            let f = b.preprocessed(BP + IN_FLAGS + 12 * i + 2);
            for w in 0..8 {
                let o = b.witness(0, bl3 + BL3_OUT + w);
                let r = b.witness(0, self.in_reg(i) + REG_PKN_W + w);
                b.assert_zero(f.clone() * (o - r), "hb_tgt_pkn");
            }
        }
        for o in 0..n_out {
            let f = b.preprocessed(BP + IN_FLAGS + 12 * n_in + 3 * o);
            for w in 0..8 {
                let out = b.witness(0, bl3 + BL3_OUT + w);
                let p = b.public(4 * n_in + 8 * n_in + 8 * o + w);
                b.assert_zero(f.clone() * (out - p), "hb_tgt_cm");
            }
        }

        // leaf binding: cm words == Merkle leaf input at row 0.
        let first = b.is_first_row();
        for i in 0..n_in {
            for w in 0..8 {
                let r = b.witness(0, self.in_reg(i) + REG_CM + w);
                let s = b.witness(0, MERKLE_W * i + MERKLE_STATE + w);
                b.assert_zero(first.clone() * (r - s), "hb_leaf");
            }
        }

        // message wiring (bit-level; the byte offsets are fixed by the formulas)
        let wire = |b: &mut B, gate: &B::Expr, msg_byte: usize, reg: usize, n: usize| {
            for k in 0..n {
                let byte = msg_byte + k;
                let wcol = bl3 + BL3_GINB + 32 * (byte / 4) + 8 * (byte % 4);
                for t in 0..8 {
                    let m = b.witness(0, wcol + t);
                    let r = b.witness(0, reg + 8 * k + t);
                    b.assert_zero(gate.clone() * (m - r), "hb_wire");
                }
            }
        };

           for i in 0..n_in {
            let rb = self.in_reg(i);
            let fl = BP + IN_FLAGS + 12 * i;
            let w_v = b.preprocessed(fl + 3);
            let w_rho_cm = b.preprocessed(fl + 4);
            let w_rho_nf = b.preprocessed(fl + 5);
            let w_rho_nf2 = b.preprocessed(fl + 6);
            let w_nk_nf = b.preprocessed(fl + 7);
            let w_nk_pk = b.preprocessed(fl + 8);
            let w_pkn_a = b.preprocessed(fl + 9);
            let w_pkn_b = b.preprocessed(fl + 10);
            wire(b, &w_v, 7, rb + REG_V, 8);
            wire(b, &w_rho_cm, 15, rb + REG_RHO, 32);
            wire(b, &w_rho_nf, 39, rb + REG_RHO, 25);
            wire(b, &w_rho_nf2, 0, rb + REG_RHO + 200, 7);
            wire(b, &w_nk_nf, 7, rb + REG_NK, 32);
            wire(b, &w_nk_pk, 11, rb + REG_NK, 32);
            wire(b, &w_pkn_a, 47, rb + REG_PKN_B, 17);
            wire(b, &w_pkn_b, 0, rb + REG_PKN_B + 136, 15);
        }

        for o in 0..n_out {
            let f = b.preprocessed(BP + IN_FLAGS + 12 * n_in + 3 * o + 2);
            wire(b, &f, 7, self.out_reg(o) + REG_V, 8);
        }

        // conservation value ties
        for i in 0..n_in {
            let f = b.preprocessed(BP + IN_FLAGS + 12 * i + 11);
            for j in 0..64 {
                let c = b.witness(0, self.cons_base() + V_BITS_BASE + j);
                let r = b.witness(0, self.in_reg(i) + REG_V + j);
                b.assert_zero(f.clone() * (c - r), "hb_cons_v");
            }
        }
        for o in 0..n_out {
            let f = b.preprocessed(BP + IN_FLAGS + 12 * n_in + 3 * o + 1);
            for j in 0..64 {
                let c = b.witness(0, self.cons_base() + V_BITS_BASE + j);
                let r = b.witness(0, self.out_reg(o) + REG_V + j);
                b.assert_zero(f.clone() * (c - r), "hb_cons_v");
            }
        }

         // D.3 reverts (erratum 75) and burns (erratum 76).
        let n_cond = self.n_cond();
        for k in 0..n_cond {
            let rb = self.rev_reg(k);
            for c in 0..64 {
                let cur = b.witness(0, rb + c);
                let nxt = b.witness(1, rb + c);
                b.assert_zero(not_last.clone() * (nxt - cur), "reg_copy");
            }
            let ob = self.out_reg(self.cond_outputs[k]);
            for j in 0..64 {
                let r = b.witness(0, rb + j);
                let o = b.witness(0, ob + j);
                b.assert_zero(r - o, "hb_revert_eq");
            }
            let f = b.preprocessed(self.rev_flag(k));
            for w in 0..8 {
                let out = b.witness(0, bl3 + BL3_OUT + w);
                let p = b.public(4 * n_in + 8 * n_in + 8 * n_out + 8 * k + w);
                b.assert_zero(f.clone() * (out - p), "hb_tgt_rev");
            }
        }

        for k in 0..self.n_burns {
            let rb = self.burn_reg(k);
            for c in 0..64 {
                let cur = b.witness(0, rb + c);
                let nxt = b.witness(1, rb + c);
                b.assert_zero(not_last.clone() * (nxt - cur), "reg_copy");
            }
            let f = b.preprocessed(self.burn_flag(k));
            for w in 0..8 {
                let out = b.witness(0, bl3 + BL3_OUT + w);
                let p = b.public(4 * n_in + 8 * n_in + 8 * n_out + 8 * n_cond + 8 * k + w);
                b.assert_zero(f.clone() * (out - p), "hb_tgt_burn");
            }
            let fmsg = b.preprocessed(self.burn_flag(k) + 1);
            wire(b, &fmsg, 42, rb, 8);
            let fcons = b.preprocessed(self.burn_flag(k) + 2);
            for j in 0..64 {
                let c = b.witness(0, self.cons_base() + V_BITS_BASE + j);
                let r = b.witness(0, rb + j);
                b.assert_zero(fcons.clone() * (c - r), "hb_cons_burn");
            }
        }

    }
}

// ---------------------------------------------------------------------------
// Witness contract
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct InputWitness {
    pub opening: NoteOpening,
    /// nk_j — the preimage of opening.pk_n (erratum 72).
    pub nullifier_key: [u8; 32],
    pub leaf_index: u64,
    pub siblings: Vec<[Goldilocks; 4]>,
    pub anchor: NctDigest,
}

#[derive(Clone, Debug)]
pub struct OutputWitness {
    pub opening: NoteOpening,
}

#[derive(Clone, Debug)]
pub struct CustodyWitness {
    pub inputs: Vec<InputWitness>,
    pub outputs: Vec<OutputWitness>,
    pub fees: Vec<u64>,
    pub reverts: Vec<RevertWitness>,
    pub burns: Vec<BurnWitness>,
}

#[derive(Clone, Debug)]
pub struct CustodyTrace {
    pub trace: Vec<Vec<Goldilocks>>,
    pub prep: Vec<Vec<Goldilocks>>,
    pub publics: Vec<Goldilocks>,
}

#[derive(Clone, Debug)]
pub struct RevertWitness {
    /// The pre-committed revert note (sender's; homed to the issuing shard).
    pub opening: NoteOpening,
}

#[derive(Clone, Copy, Debug)]
pub struct BurnWitness {
    /// The leg (canonical index) the burn is declared on.
    pub leg: u8,
    pub value: u64,
    /// H("nerv.burn" ‖ txid ‖ leg ‖ value) — recomputed and checked at bind.
    pub commitment: Hash256,
}


// ---------------------------------------------------------------------------
// nerv-codec Encode/Decode impls
// ---------------------------------------------------------------------------
//
// The witness contract is the private input to the custody AIR (§5.1
// statements 1–5 + reversion/burn binding). These impls let it round-trip
// through `nerv_core::codec` for testing, indexing, and persistence. The
// wire format is:
//   * u32 LE count, then items in declaration order
//   * For `InputWitness`: opening (NoteEncoding) ‖ nk (32 bytes) ‖
//     leaf_index (u64 LE) ‖ u32 LE sibling count ‖ siblings (each
//     [Goldilocks; 4] is 4×u64 LE) ‖ anchor (NctDigest's existing
//     Encode/Decode).
//   * For `BurnWitness`: u8 (leg) ‖ u64 LE (value) ‖ Hash256 (32 bytes).

impl nerv_core::codec::Encode for InputWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.opening.encode_into(out);
        out.extend_from_slice(&self.nullifier_key);
        out.extend_from_slice(&self.leaf_index.to_le_bytes());
        let n = self.siblings.len() as u32;
        out.extend_from_slice(&n.to_le_bytes());
        for sib in &self.siblings {
            for g in sib {
                out.extend_from_slice(&g.as_u64().to_le_bytes());
            }
        }
        self.anchor.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        // NoteOpening::encoded_len + 32 + 8 + 4 + 32*siblings + anchor_len.
        // We don't know the opening length statically; callers that need
        // pre-allocation should compute it from the note's own encoded_len.
        // For smoke / persistence, callers typically use `encode()` and let
        // the Vec grow; we expose a best-effort constant that matches the
        // common case (32-byte opening: value 8B + rho 12B + delivery 48B +
        // blinding 8B + pk_n 32B = 108B; see `NoteOpening` constants).
        108 + 32 + 8 + 4 + (32 * self.siblings.len()) + 32
    }
}

impl nerv_core::codec::Decode for InputWitness {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let opening = nerv_custody::NoteOpening::decode_from(r)?;
        let nk_bytes = r.take_array::<32>()?;
        let mut nullifier_key = [0u8; 32];
        nullifier_key.copy_from_slice(&nk_bytes);
        let leaf_index = r.read_u64()?;
        let n = r.read_u32()? as usize;
        let mut siblings = Vec::with_capacity(n);
        for _ in 0..n {
            let mut sib = [nerv_core::field::Goldilocks::ZERO; 4];
            for slot in sib.iter_mut() {
                let bytes = r.take_array::<8>()?;
                *slot = nerv_core::field::Goldilocks::from_u64_reduce(u64::from_le_bytes(bytes));
            }
            siblings.push(sib);
        }
        let anchor = nerv_custody::nct::NctDigest::decode_from(r)?;
        Ok(InputWitness { opening, nullifier_key, leaf_index, siblings, anchor })
    }
}

impl nerv_core::codec::Encode for OutputWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.opening.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.opening.encoded_len()
    }
}

impl nerv_core::codec::Decode for OutputWitness {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        Ok(OutputWitness { opening: nerv_custody::NoteOpening::decode_from(r)? })
    }
}

impl nerv_core::codec::Encode for RevertWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        self.opening.encode_into(out);
    }
    fn encoded_len(&self) -> usize {
        self.opening.encoded_len()
    }
}

impl nerv_core::codec::Decode for RevertWitness {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        Ok(RevertWitness { opening: nerv_custody::NoteOpening::decode_from(r)? })
    }
}

impl nerv_core::codec::Encode for BurnWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.push(self.leg);
        out.extend_from_slice(&self.value.to_le_bytes());
        out.extend_from_slice(self.commitment.as_bytes());
    }
    fn encoded_len(&self) -> usize {
        1 + 8 + 32
    }
}

impl nerv_core::codec::Decode for BurnWitness {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let leg = r.read_u8()?;
        let value = r.read_u64()?;
        let bytes = r.take_array::<32>()?;
        let commitment = Hash256::from_bytes(bytes);
        Ok(BurnWitness { leg, value, commitment })
    }
}

impl nerv_core::codec::Encode for CustodyWitness {
    fn encode_into(&self, out: &mut Vec<u8>) {
        encode_vec(out, &self.inputs);
        encode_vec(out, &self.outputs);
        encode_vec_u64(out, &self.fees);
        encode_vec(out, &self.reverts);
        encode_vec(out, &self.burns);
    }
    fn encoded_len(&self) -> usize {
        vec_len(&self.inputs) + vec_len(&self.outputs) + (4 + 8 * self.fees.len())
            + vec_len(&self.reverts) + vec_len(&self.burns)
    }
}

impl nerv_core::codec::Decode for CustodyWitness {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let inputs = decode_vec::<InputWitness>(r)?;
        let outputs = decode_vec::<OutputWitness>(r)?;
        let fees = decode_vec_u64(r)?;
        let reverts = decode_vec::<RevertWitness>(r)?;
        let burns = decode_vec::<BurnWitness>(r)?;
        Ok(CustodyWitness { inputs, outputs, fees, reverts, burns })
    }
}

fn encode_vec<T: nerv_core::codec::Encode>(out: &mut Vec<u8>, v: &[T]) {
    let n = v.len() as u32;
    out.extend_from_slice(&n.to_le_bytes());
    for item in v {
        item.encode_into(out);
    }
}

fn encode_vec_u64(out: &mut Vec<u8>, v: &[u64]) {
    let n = v.len() as u32;
    out.extend_from_slice(&n.to_le_bytes());
    for item in v {
        out.extend_from_slice(&item.to_le_bytes());
    }
}

fn vec_len<T: nerv_core::codec::Encode>(v: &[T]) -> usize {
    4 + v.iter().map(|i| i.encoded_len()).sum::<usize>()
}

fn decode_vec<T: nerv_core::codec::Decode>(r: &mut nerv_core::codec::Reader<'_>) -> Result<Vec<T>, nerv_core::error::CodecError> {
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(T::decode_from(r)?);
    }
    Ok(out)
}

fn decode_vec_u64(r: &mut nerv_core::codec::Reader<'_>) -> Result<Vec<u64>, nerv_core::error::CodecError> {
    let n = r.read_u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.read_u64()?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

fn put_word(row: &mut [Goldilocks], col: usize, v: u32) {
    row[col] = Goldilocks::from_u64_reduce(u64::from(v));
}

fn words_of(h: &Hash256) -> [u32; 8] {
    let mut w = [0u32; 8];
    for i in 0..8 {
        w[i] = u32::from_le_bytes([
            h.as_bytes()[4 * i], h.as_bytes()[4 * i + 1], h.as_bytes()[4 * i + 2],
            h.as_bytes()[4 * i + 3],
        ]);
    }
    w
}


fn cm_message(o: &NoteOpening) -> Vec<u8> {
    let mut m = Vec::with_capacity(1295);
    m.extend_from_slice(b"nerv.cm");
    m.extend_from_slice(&o.value.to_le_bytes());
    m.extend_from_slice(&o.rho);
    m.extend_from_slice(&o.delivery);
    m.extend_from_slice(&o.blinding);
    m.extend_from_slice(&o.pk_n);
    m
}

fn burn_message(txid: &nerv_core::types::TxId, leg: u8, value: u64) -> Vec<u8> {
    let mut m = Vec::with_capacity(50);
    m.extend_from_slice(b"nerv.burn");
    m.extend_from_slice(txid.as_bytes());
    m.push(leg);
    m.extend_from_slice(&value.to_le_bytes());
    m
}



struct Placer<'a> {
    trace: &'a mut [Vec<Goldilocks>],
    prep: &'a mut [Vec<Goldilocks>],
    bl3_base: usize,
}

impl Placer<'_> {
    fn window(&mut self, gw: usize, cv: &[u32; 8], block: &[u8; 64], len: u8, flags: u32) -> [u32; 16] {
        // BLAKE3's native block is 16 little-endian u32 words (64 bytes);
        // reinterpret the byte block so the trace and compress_native both
        // see the words in the layout blake3's algorithm expects.
        let mut words = [0u32; 16];
        for (i, w) in words.iter_mut().enumerate() {
            let off = 4 * i;
            *w = u32::from_le_bytes([
                block[off],
                block[off + 1],
                block[off + 2],
                block[off + 3],
            ]);
        }
        let rows = bl3_trace(cv, &words, len, 0, flags);
        let prep = bl3_prep(len, 0, flags);
        for r in 0..66 {
            let src = &rows[r];
            self.trace[66 * gw + r][self.bl3_base..self.bl3_base + BL3_W]
                .copy_from_slice(src);
            self.prep[66 * gw + r][BL3_PREP_BASE..BL3_PREP_BASE + BL3_PREP_W]
                .copy_from_slice(&prep[r]);
        }
        compress_native(cv, &words, len, 0, flags)
    }

    /// A 2-chunk (1295-byte) stream: 22 windows; returns (digest, cv0).
    fn cm_stream(&mut self, w0: usize, msg: &[u8]) -> ([u32; 8], [u32; 8]) {
        debug_assert_eq!(msg.len(), 1295);
        let mut chunk = |c: usize, gw0: usize| -> ([u32; 8], usize) {
            let cstart = 1024 * c;
            let cend = (cstart + 1024).min(msg.len());
            let nblocks = (cend - cstart).div_ceil(64);
            let mut cv = BL3_IV;
            let mut gw = gw0;
            for bi in 0..nblocks {
                let bstart = cstart + 64 * bi;
                let bend = (bstart + 64).min(cend);
                let mut block = [0u8; 64];
                block[..bend - bstart].copy_from_slice(&msg[bstart..bend]);
                let mut flags = 0;
                if bi == 0 {
                    flags |= CHUNK_START;
                }
                if bi == nblocks - 1 {
                    flags |= CHUNK_END;
                }
                let out = self.window(gw, &cv, &block, (bend - bstart) as u8, flags);
                cv = [out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]];
                gw += 1;
            }
            (cv, gw)
        };
        let (cv0, _) = chunk(0, w0);
        let (cv1, _) = chunk(1, w0 + 16);
        let mut pblock = [0u8; 64];
        for j in 0..8 {
            pblock[4 * j..4 * j + 4].copy_from_slice(&cv0[j].to_le_bytes());
            pblock[32 + 4 * j..36 + 4 * j].copy_from_slice(&cv1[j].to_le_bytes());
        }
        let out = self.window(w0 + 21, &BL3_IV, &pblock, 64, PARENT | ROOT);
        ([out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]], cv0)
    }

    /// A 1-chunk stream (≤ 1024 bytes): blocks from w0; returns the digest.
    fn simple_stream(&mut self, w0: usize, msg: &[u8]) -> [u32; 8] {
        let nblocks = msg.len().div_ceil(64).max(1);
        let mut cv = BL3_IV;
        for bi in 0..nblocks {
            let bstart = 64 * bi;
            let bend = (bstart + 64).min(msg.len());
            let mut block = [0u8; 64];
            block[..bend - bstart].copy_from_slice(&msg[bstart..bend]);
            let mut flags = 0;
            if bi == 0 {
                flags |= CHUNK_START;
            }
            if bi == nblocks - 1 {
                flags |= CHUNK_END | ROOT;
            }
            let out = self.window(w0 + bi, &cv, &block, (bend - bstart) as u8, flags);
            cv = [out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]];
        }
        cv
    }
}

pub fn set_flag(prep: &mut [Vec<Goldilocks>], row: usize, col: usize) {
    prep[row][col] = Goldilocks::ONE;
}
pub fn set_input_flags(prep: &mut [Vec<Goldilocks>], i: usize, w0: usize) {
    let nf = BP + IN_FLAGS + 12 * i;
    // IV starts: cm chunk 0 (w0), cm chunk 1 (w0+16), the parent (w0+21),
    // the nf stream (w0+22), and the pk_n stream (w0+24).
    for lw in [0usize, 16, 21, 22, 24] {
        set_flag(prep, 66 * (w0 + lw), BP + F_IV);
    }
    // cv chains WITHIN each chunk/stream only: cm chunk 0 (w0..w0+14),
    // cm chunk 1 (w0+16..w0+19), the nf stream (w0+22). Chunk, parent,
    // and stream boundaries restart from IV — never chained.
    for lw in [
        0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22,
    ] {
        set_flag(prep, 66 * (w0 + lw) + 65, BP + F_CHAIN);
    }
    set_flag(prep, 66 * (w0 + 21), BP + F_PARENT);
    set_flag(prep, 66 * (w0 + 20) + 65, BP + F_PCV1);
    set_flag(prep, 66 * (w0 + 15) + 65, BP + F_CVLOAD);
    set_flag(prep, 66 * (w0 + 21) + 65, nf + 1); // TGT_CMR
    set_flag(prep, 66 * (w0 + 23) + 65, nf + 0); // TGT_NF
    set_flag(prep, 66 * (w0 + 24) + 65, nf + 2); // TGT_PKN
    set_flag(prep, 66 * (w0 + 0), nf + 3); // W_V
    set_flag(prep, 66 * (w0 + 0), nf + 4); // W_RHO_CM
    set_flag(prep, 66 * (w0 + 22), nf + 5); // W_RHO_NF
    set_flag(prep, 66 * (w0 + 23), nf + 6); // W_RHO_NF2
    set_flag(prep, 66 * (w0 + 22), nf + 7); // W_NK_NF
    set_flag(prep, 66 * (w0 + 24), nf + 8); // W_NK_PK
    set_flag(prep, 66 * (w0 + 19), nf + 9); // W_PKN_A
    set_flag(prep, 66 * (w0 + 20), nf + 10); // W_PKN_B
}


pub fn set_output_flags(prep: &mut [Vec<Goldilocks>], o: usize, n_in: usize, w0: usize) {
    let of = BP + IN_FLAGS + 12 * n_in + 3 * o;
    // IV starts: cm chunk 0 (w0), cm chunk 1 (w0+16), the parent (w0+21).
    for lw in [0usize, 16, 21] {
        set_flag(prep, 66 * (w0 + lw), BP + F_IV);
    }
    // cv chains within the two chunks only; the parent restarts from IV.
    for lw in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19] {
        set_flag(prep, 66 * (w0 + lw) + 65, BP + F_CHAIN);
    }
    set_flag(prep, 66 * (w0 + 21), BP + F_PARENT);
    set_flag(prep, 66 * (w0 + 20) + 65, BP + F_PCV1);
    set_flag(prep, 66 * (w0 + 15) + 65, BP + F_CVLOAD);
    set_flag(prep, 66 * (w0 + 21) + 65, of); // TGT_CM
    set_flag(prep, 66 * (w0 + 0), of + 2); // W_V
}

pub fn set_revert_flags(prep: &mut [Vec<Goldilocks>], k: usize, n_in: usize, n_out: usize, w0: usize) {
    let f = BP + 5 + 12 * n_in + 3 * n_out + k;
    // IV starts: cm chunk 0 (w0), cm chunk 1 (w0+16), the parent (w0+21).
    for lw in [0usize, 16, 21] {
        set_flag(prep, 66 * (w0 + lw), BP + F_IV);
    }
    // cv chains within the two chunks only; the parent restarts from IV.
    for lw in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19] {
        set_flag(prep, 66 * (w0 + lw) + 65, BP + F_CHAIN);
    }
    set_flag(prep, 66 * (w0 + 21), BP + F_PARENT);
    set_flag(prep, 66 * (w0 + 20) + 65, BP + F_PCV1);
    set_flag(prep, 66 * (w0 + 15) + 65, BP + F_CVLOAD);
    set_flag(prep, 66 * (w0 + 21) + 65, f);
}


pub fn set_burn_flags(
    prep: &mut [Vec<Goldilocks>],
    k: usize,
    n_in: usize,
    n_out: usize,
    n_cond: usize,
    w0: usize,
) {
    let f = BP + 5 + 12 * n_in + 3 * n_out + n_cond + 3 * k;
    set_flag(prep, 66 * w0, BP + F_IV);
    set_flag(prep, 66 * w0 + 65, f);
    set_flag(prep, 66 * w0, f + 1);
    set_flag(prep, n_in + n_out + k, f + 2);
}



fn fill_bits(row: &mut [Goldilocks], col: usize, bytes: &[u8]) {
    for (k, &b) in bytes.iter().enumerate() {
        for t in 0..8 {
            row[col + 8 * k + t] = Goldilocks::from_u32(((b >> t) & 1) as u32);
        }
    }
}

pub fn gen_custody_trace(
    w: &CustodyWitness,
    shell: &TransactionShell,
    depth: usize,
) -> Result<CustodyTrace, WitnessError> {
  bind_witness_to_shell(w, shell)?;
    let canon = shell.canonicalize()?;
    let txid = shell.txid()?;
    let n_in = w.inputs.len();
    let n_out = w.outputs.len();
    let n_fee = w.fees.len();
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
    let air = CustodyAir::new(n_in, n_out, cond_outputs, w.burns.len(), depth);
    let (rows, cols, prep_w) = (air.rows(), air.cols(), air.prep_cols());
    let mut trace = vec![vec![Goldilocks::ZERO; cols]; rows];
    let mut prep = vec![vec![Goldilocks::ZERO; prep_w]; rows];

    // conservation (burns are output-class rows: value-capped, not fee-capped)
    let mut entries: Vec<(u64, bool)> =
        w.inputs.iter().map(|iw| (iw.opening.value, true)).collect();
    entries.extend(w.outputs.iter().map(|ow| (ow.opening.value, false)));
    entries.extend(w.burns.iter().map(|bw| (bw.value, false)));
    entries.extend(w.fees.iter().map(|&f| (f, false)));
    let cons_rows = gen_cons_trace(&entries);
    for (r, row) in cons_rows.iter().enumerate() {
        trace[r][air.cons_base()..air.cons_base() + CONS_W].copy_from_slice(row);
    }
    let cons_prep = gen_cons_prep(n_in, n_out + w.burns.len(), n_fee);

    for r in 0..cons_prep.len() {
        prep[r][CONS_PREP_BASE..CONS_PREP_BASE + 6].copy_from_slice(&cons_prep[r]);
    }
    for i in 0..n_in {
        set_flag(&mut prep, i, BP + IN_FLAGS + 12 * i + 11);
    }
    for o in 0..n_out {
        set_flag(&mut prep, n_in + o, BP + IN_FLAGS + 12 * n_in + 3 * o + 1);
    }




    // merkle
    let mprep = MerkleChip::new(depth).gen_prep();
    for r in 0..mprep.len() {
        prep[r][0..MERKLE_PREP_W].copy_from_slice(&mprep[r]);
    }
    for (i, iw) in w.inputs.iter().enumerate() {
        let cm = iw.opening.commitment()?;
        let mt = gen_merkle_trace(depth, &cm, iw.leaf_index, &iw.siblings);
        for (r, row) in mt.iter().enumerate() {
            trace[r][MERKLE_W * i..MERKLE_W * i + MERKLE_W].copy_from_slice(row);
        }
    }

 // hash streams
    let bl3_base = air.bl3_base();
    let n_cond = air.n_cond();
    let mut aux_loads: Vec<(usize, [u32; 8])> = Vec::new();
    {
        let mut p = Placer { trace: &mut trace, prep: &mut prep, bl3_base };
        for (i, iw) in w.inputs.iter().enumerate() {
            let w0 = IN_WINS * i;
            let (_, cv0) = p.cm_stream(w0, &cm_message(&iw.opening));
            aux_loads.push((66 * (w0 + 15) + 65, cv0));
            let mut nf_msg = Vec::with_capacity(71);
            nf_msg.extend_from_slice(b"nerv.nf");
            nf_msg.extend_from_slice(&iw.nullifier_key);
            nf_msg.extend_from_slice(&iw.opening.rho);
            p.simple_stream(w0 + 22, &nf_msg);
            let mut pk_msg = Vec::with_capacity(43);
            pk_msg.extend_from_slice(b"nerv.nf.pk");
            pk_msg.extend_from_slice(&iw.nullifier_key);
            p.simple_stream(w0 + 24, &pk_msg);
        }
        for (o, ow) in w.outputs.iter().enumerate() {
            let w0 = IN_WINS * n_in + OUT_WINS * o;
            let (_, cv0) = p.cm_stream(w0, &cm_message(&ow.opening));
            aux_loads.push((66 * (w0 + 15) + 65, cv0));
        }
        for (k, rw) in w.reverts.iter().enumerate() {
            let w0 = IN_WINS * n_in + OUT_WINS * n_out + 22 * k;
            let (_, cv0) = p.cm_stream(w0, &cm_message(&rw.opening));
            aux_loads.push((66 * (w0 + 15) + 65, cv0));
        }
        for (k, bw) in w.burns.iter().enumerate() {
            let w0 = IN_WINS * n_in + OUT_WINS * n_out + 22 * n_cond + k;
            p.simple_stream(w0, &burn_message(&txid, bw.leg, bw.value));
        }
    }
    for i in 0..n_in {
        set_input_flags(&mut prep, i, IN_WINS * i);
    }
    for o in 0..n_out {
        set_output_flags(&mut prep, o, n_in, IN_WINS * n_in + OUT_WINS * o);
    }
    for k in 0..n_cond {
        set_revert_flags(
            &mut prep, k, n_in, n_out, IN_WINS * n_in + OUT_WINS * n_out + 22 * k,
        );
    }
    for k in 0..w.burns.len() {
        set_burn_flags(
            &mut prep, k, n_in, n_out, n_cond,
            IN_WINS * n_in + OUT_WINS * n_out + 22 * n_cond + k,
        );
    }

    

    // registers
    let reg = air.reg_base();
    let aux = air.aux_base();
    aux_loads.sort_unstable_by_key(|(r, _)| *r);
    for r in 0..rows {
        for (i, iw) in w.inputs.iter().enumerate() {
            let rb = reg + INPUT_REG_W * i;
            fill_bits(&mut trace[r], rb + REG_V, &iw.opening.value.to_le_bytes());
            fill_bits(&mut trace[r], rb + REG_RHO, &iw.opening.rho);
            fill_bits(&mut trace[r], rb + REG_NK, &iw.nullifier_key);
            fill_bits(&mut trace[r], rb + REG_PKN_B, &iw.opening.pk_n);
            let pkn_words = words_of(&Hash256::from_bytes(iw.opening.pk_n));
            for (x, &word) in pkn_words.iter().enumerate() {
                put_word(&mut trace[r], rb + REG_PKN_W + x, word);
            }
            let cm_words = words_of(&iw.opening.commitment()?);
            for (x, &word) in cm_words.iter().enumerate() {
                put_word(&mut trace[r], rb + REG_CM + x, word);
            }

        }
         for (o, ow) in w.outputs.iter().enumerate() {
            let rb = reg + INPUT_REG_W * n_in + OUT_REG_W * o;
            fill_bits(&mut trace[r], rb, &ow.opening.value.to_le_bytes());
        }
        for (k, rw) in w.reverts.iter().enumerate() {
            fill_bits(&mut trace[r], air.rev_reg(k), &rw.opening.value.to_le_bytes());
        }
        for (k, bw) in w.burns.iter().enumerate() {
            fill_bits(&mut trace[r], air.burn_reg(k), &bw.value.to_le_bytes());
        }
        let mut cv: [u32; 8] = [0; 8];
        for &(lr, v) in aux_loads.iter() {
            if r > lr {
                cv = v;
            }
        }

        for (x, &word) in cv.iter().enumerate() {
            put_word(&mut trace[r], aux + x, word);
        }
    }

    // publics
    let mut publics: Vec<Goldilocks> = Vec::with_capacity(
        4 * n_in + 8 * n_in + 8 * n_out + 8 * air.n_cond() + 8 * w.burns.len(),
    );

    for iw in &w.inputs {
        publics.extend_from_slice(&iw.anchor.to_elements()?);
    }
    for leg in &canon.legs {
        for nf in &leg.inputs.nullifiers {
            for &word in words_of(nf).iter() {
                publics.push(Goldilocks::from_u64_reduce(u64::from(word)));
            }
        }
    }
 for leg in &canon.legs {
        for out in &leg.outputs {
            for &word in words_of(&out.cm).iter() {
                publics.push(Goldilocks::from_u64_reduce(u64::from(word)));
            }
        }
    }
    for leg in &canon.legs {
        for out in &leg.outputs {
            if let Some(rcm) = &out.revert_cm {
                for &word in words_of(rcm).iter() {
                    publics.push(Goldilocks::from_u64_reduce(u64::from(word)));
                }
            }
        }
    }
    for bw in &w.burns {
        for &word in words_of(&bw.commitment).iter() {
            publics.push(Goldilocks::from_u64_reduce(u64::from(word)));
        }
    }
    Ok(CustodyTrace { trace, prep, publics })
}


/// The wallet-side pre-proof binding: every shell nullifier equals the
/// witness's derived nullifier; every input's H(nk) equals its opening's
/// pk_n (the erratum-72 binding); anchors, output commitments and values,
/// and fees match. Consumption order: canonical leg order throughout.
pub fn bind_witness_to_shell(
    w: &CustodyWitness,
    shell: &TransactionShell,
) -> Result<(), WitnessError> {
    if w.inputs.is_empty() {
        return Err(WitnessError::NoInputs);
    }
    for (i, iw) in w.inputs.iter().enumerate() {
        if nullifier_pk(&iw.nullifier_key) != iw.opening.pk_n {
            return Err(WitnessError::NullifierKeyMismatch { index: i });
        }
    }
    let canon = shell.canonicalize()?;
    let mut next_input = 0usize;
    let mut next_output = 0usize;
    for (leg_idx, leg) in canon.legs.iter().enumerate() {
        for nf in &leg.inputs.nullifiers {
            let Some(iw) = w.inputs.get(next_input) else {
                return Err(WitnessError::InputCount {
                    expected: w.inputs.len(),
                    found: next_input + 1,
                });
            };
            if derive_nullifier(&iw.nullifier_key, &iw.opening.rho) != *nf {
                return Err(WitnessError::NullifierMismatch { index: next_input });
            }
            if NctDigest::try_from_hash256(&leg.anchor)? != iw.anchor {
                return Err(WitnessError::AnchorMismatch { index: next_input });
            }
            next_input += 1;
        }
        for out in &leg.outputs {
            let Some(ow) = w.outputs.get(next_output) else {
                return Err(WitnessError::InputCount {
                    expected: w.outputs.len(),
                    found: next_output + 1,
                });
            };
            if ow.opening.commitment()? != out.cm || ow.opening.value != out.value {
                return Err(WitnessError::OutputMismatch { index: next_output });
            }
            next_output += 1;
        }
        if w.fees.get(leg_idx).copied() != Some(leg.fee.as_u64()) {
            return Err(WitnessError::FeeMismatch { leg: leg_idx });
        }
    }
    if next_input != w.inputs.len() {
        return Err(WitnessError::InputCount { expected: w.inputs.len(), found: next_input });
    }
    if next_output != w.outputs.len() {
        return Err(WitnessError::InputCount {
            expected: w.outputs.len(),
            found: next_output,
        });
    }
    if w.fees.len() != canon.legs.len() {
        return Err(WitnessError::FeeMismatch { leg: w.fees.len() });
    }

    let cond_count: usize = canon
        .legs
        .iter()
        .map(|l| l.outputs.iter().filter(|o| o.conditional).count())
        .sum();
    if w.reverts.len() != cond_count {
        return Err(WitnessError::RevertCount { expected: cond_count, found: w.reverts.len() });
    }
    let txid = shell.txid()?;
    let mut next_rev = 0usize;
    for leg in &canon.legs {
        for out in &leg.outputs {
            if out.conditional {
                let rw = &w.reverts[next_rev];
                if rw.opening.commitment()? != *out.revert_cm.as_ref().unwrap() {
                    return Err(WitnessError::RevertCmMismatch { index: next_rev });
                }
                if rw.opening.value != out.value {
                    return Err(WitnessError::RevertValueMismatch { index: next_rev });
                }
                next_rev += 1;
            }
        }
    }
    for (k, bw) in w.burns.iter().enumerate() {
        let expect = BurnCommitment::new(&txid, LegIndex::from_u8(bw.leg), bw.value)?;
        if *expect.as_hash() != bw.commitment {
            return Err(WitnessError::BurnMismatch { index: k });
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::air::builder::NativeEval;
    use crate::air::fs::{
        bind_transaction, canonical_nullifiers, shell_digest, TxPublicInputs,
    };
    use crate::testutil::SplitMix64;
    use nerv_core::types::{FeeSats, Height, ShardSet};
    use nerv_custody::nct::{NoteCommitmentTree, DEPTH};
    use nerv_custody::tx::{InputSet, LegShell, Output};
    use nerv_custody::{MasterSeed, WalletKeys};

    const NCT_DEPTH: usize = DEPTH;

    struct Fixture {
        witness: CustodyWitness,
        shell: TransactionShell,
        key_id: Hash256,
    }

    /// Appendix A (corrected arithmetic, erratum 69): inputs 50 + 35 NERV;
    /// outputs Carol 25, change 9.999, Bob 50; fees 0.0006/0.0004.
    fn fixture(seed: u64, two_by_two: bool) -> Fixture {
        let mut rng = SplitMix64::new(seed);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let nk_tree = *wk.nullifier_key();
        let g = ShardSet::genesis();
        let addr = |i: u64| Address_gen(&det, &nk_tree, i, &g);
        let nanos = 1_000_000_000u64;
        let opening = |v: u64, i: u64| NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let in1 = opening(50 * nanos, 0);
        let in2 = opening(35 * nanos, 1);
        let out_carol = opening(25 * nanos, 2);
        let out_change = opening(9_999 * nanos / 1000, 3);
        let out_bob = opening(50 * nanos, 4);
        let out_1x1 = opening(50 * nanos - 600_000, 4);

        let mut tree = NoteCommitmentTree::new();
        for _ in 0..10 {
            tree.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
        }
        let idx1 = tree.append(&in1.commitment().unwrap()).unwrap();
        let idx2 = tree.append(&in2.commitment().unwrap()).unwrap();
        for _ in 0..28 {
            tree.append(&Hash256::from_bytes(rng.bytes32())).unwrap();
        }
        let root = tree.root();
        let sib = |idx: u64| -> Vec<[Goldilocks; 4]> {
            tree.witness(idx).unwrap().siblings.iter().map(|d| d.to_elements().unwrap()).collect()
        };
        let inputs = vec![
            InputWitness {
                opening: in1,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: idx1,
                siblings: sib(idx1),
                anchor: root,
            },
            InputWitness {
                opening: in2,
                nullifier_key: wk.nullifier_key_at(1),
                leaf_index: idx2,
                siblings: sib(idx2),
                anchor: root,
            },
        ];
        let nf1 = derive_nullifier(&inputs[0].nullifier_key, &inputs[0].opening.rho);
        let nf2 = derive_nullifier(&inputs[1].nullifier_key, &inputs[1].opening.rho);
        let mk_out = |o: &NoteOpening| Output {
            cm: o.commitment().unwrap(),
            sealed_note: vec![0xA5; 48],
            value: o.value,
            conditional: false,
            revert_cm: None,
        };
      let leg7 = LegShell {
            shard: g.ids()[7],
            inputs: InputSet::new(vec![nf1, nf2]),
            outputs: vec![mk_out(&out_carol), mk_out(&out_change)],
            fee: FeeSats::from_u64(600_000),
            anchor: root.as_hash256(),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };
        let leg40 = LegShell {
            shard: g.ids()[40],
            inputs: InputSet::new(vec![]),
            outputs: vec![mk_out(&out_bob)],
            fee: FeeSats::from_u64(400_000),
            anchor: Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };


        let (witness, shell) = if two_by_two {
            (
                CustodyWitness {
                    inputs,
                    outputs: vec![
                        OutputWitness { opening: out_carol },
                        OutputWitness { opening: out_change },
                        OutputWitness { opening: out_bob },
                    ],
                    reverts: vec![],
                    burns: vec![],
                    fees: vec![600_000, 400_000],
                },
                TransactionShell { legs: vec![leg7, leg40] },
            )
        } else {
            // 1-in/1-out variant for the tamper tests: 50 NERV in,
            // out 49.9994 NERV, fee 0.0006 NERV — conservation balances.
            (
                CustodyWitness {
                    inputs: vec![inputs[0].clone()],
                    outputs: vec![OutputWitness { opening: out_1x1 }],
                    reverts: vec![],
                    burns: vec![],
                    fees: vec![600_000],
                },
                TransactionShell {
                    legs: vec![LegShell {
                        shard: g.ids()[7],
                        inputs: InputSet::new(vec![nf1]),
                        outputs: vec![mk_out(&out_1x1)],
                        fee: FeeSats::from_u64(600_000),
                        anchor: root.as_hash256(),
                        expiry: Height::from_u64(5_000),
                        weight_version: 1,
                        ct: vec![],
                        burns: vec![],
                    }],
                },
            )
        };

        let key_id = Hash256::from_bytes(rng.bytes32());
        Fixture { witness, shell, key_id }
    }

    fn Address_gen(det: &nerv_custody::DetectionSeed, nk: &[u8; 32], i: u64, set: &ShardSet) -> nerv_custody::Address {
        nerv_custody::Address::generate(det, nk, i, set).unwrap()
    }

    #[test]
    fn layout_pins() {
        let air = CustodyAir::new(2, 3, vec![], 0, NCT_DEPTH);
        assert_eq!(air.n_windows(), 25 * 2 + 22 * 3);
        assert_eq!(air.rows(), 66 * air.n_windows());
        assert_eq!(air.cols(), 53 * 2 + 171 + 713 + 848 * 2 + 64 * 3 + 8);
        assert_eq!(air.prep_cols(), 311 + 5 + 24 + 9);
    }

    #[test]
    fn appendix_a_full_differential() {
        let f = fixture(0xA1, true);
        let air = CustodyAir::new(2, 3, vec![], 0, NCT_DEPTH);
        let ct = gen_custody_trace(&f.witness, &f.shell, NCT_DEPTH).unwrap();
        assert_eq!(ct.trace.len(), air.rows());
        assert_eq!(ct.trace[0].len(), air.cols());
        NativeEval::check_with_prep(
            ct.trace.clone(), ct.prep.clone(), ct.publics.clone(), &air, 8,
        )
        .unwrap();
        // The publics ARE the shell's nullifiers and output commitments.
        let canon = f.shell.canonicalize().unwrap();
        let nfs: Vec<Hash256> =
            canon.legs.iter().flat_map(|l| l.inputs.nullifiers.iter().copied()).collect();
        for (i, nf) in nfs.iter().enumerate() {
            for w in 0..8 {
                let expect = u32::from_le_bytes([
                    nf.as_bytes()[4 * w], nf.as_bytes()[4 * w + 1], nf.as_bytes()[4 * w + 2],
                    nf.as_bytes()[4 * w + 3],
                ]);
                assert_eq!(ct.publics[8 + 8 * i + w].as_u64(), u64::from(expect));
            }
        }
        // Statement-11 binding over the same transaction.
        let txid = f.shell.txid().unwrap();
        let sd = shell_digest(&f.shell).unwrap();
        let nfs2 = canonical_nullifiers(&f.shell).unwrap();
        let pubs = TxPublicInputs::derive(&f.shell, f.key_id).unwrap();
        let (t, binding, pd) = bind_transaction(&nfs2, &txid, &sd, &pubs);
        assert_eq!(pd, pubs.digest());
        assert_ne!(*binding.as_bytes(), [0u8; 32]);
        assert_eq!(t.ops(), nfs2.len() as u64 + 4);
    }

    #[test]
    fn tamper_classes() {
        let f = fixture(0xA3, false);
        let air = CustodyAir::new(1, 1, NCT_DEPTH);
        let ct = gen_custody_trace(&f.witness, &f.shell, NCT_DEPTH).unwrap();
        assert!(NativeEval::check_with_prep(
            ct.trace.clone(), ct.prep.clone(), ct.publics.clone(), &air, 8
        )
        .is_ok());

        // Substituted nullifier key (the erratum-72 attack): rewire the
        // witness to a different nk with a matching pk_n in the opening —
        // the circuit must reject it (the cm pins the original pk_n).
        let mut w = f.witness.clone();
        let mut bad_nk = w.inputs[0].nullifier_key;
        bad_nk[0] ^= 1;
        w.inputs[0].nullifier_key = bad_nk;
        assert!(matches!(
            gen_custody_trace(&w, &f.shell, NCT_DEPTH),
            Err(WitnessError::NullifierKeyMismatch { index: 0 })
        ));
        // And at the trace level: flipping an nk register bit breaks the
        // nf stream (target or wiring).
        let mut bad = ct.trace.clone();
        bad[100][air.reg_base() + REG_NK] =
        bad[100][air.reg_base() + REG_NK] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, ct.prep.clone(), ct.publics.clone(), &air, 8).is_err());


        // Wrong public nullifier word.
        let mut bad = ct.publics.clone();
        bad[8] = bad[8] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(
            ct.trace.clone(), ct.prep.clone(), bad, &air, 8,
        )
        .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "hb_tgt_nf"));

        // Wrong public output-cm word.
        let mut bad = ct.publics.clone();
        let cm_off = 4 + 8; // 4·n_in + 8·n_in with n_in = 1
        bad[cm_off] = bad[cm_off] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(
            ct.trace.clone(), ct.prep.clone(), bad, &air, 8,
        )
        .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "hb_tgt_cm"));

        // pk_n register bit: breaks the pkn target / decompose.
        let mut bad = ct.trace.clone();
        bad[100][air.reg_base() + REG_PKN_B] =
            bad[100][air.reg_base() + REG_PKN_B] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, ct.prep.clone(), ct.publics.clone(), &air, 8).is_err());

        // v register bit: conservation tie or cm wiring breaks.
        let mut bad = ct.trace.clone();
        bad[100][air.reg_base() + REG_V] = bad[100][air.reg_base() + REG_V] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, ct.prep.clone(), ct.publics.clone(), &air, 8).is_err());

          // Tampered merkle sibling: the recomputed root misses the public
        // anchor (merkle_root fails); bind never validated siblings, so
        // generate directly with the tampered witness.
        let mut w = f.witness.clone();
        w.inputs[0].siblings[5][0] = w.inputs[0].siblings[5][0] + Goldilocks::ONE;
        let ct2 = gen_custody_trace(&w, &f.shell, NCT_DEPTH).unwrap();
        assert!(NativeEval::check_with_prep(
            ct2.trace.clone(), ct2.prep.clone(), ct2.publics.clone(), &air, 8
        )
        .is_err());


        // Output value tamper: conservation fails.
        let mut w = f.witness.clone();
        w.outputs[0].opening.value += 1;
        let mut shell = f.shell.clone();
        shell.legs[0].outputs[0].value += 1;
        // cm/value bind mismatch is caught pre-proof:
        assert!(matches!(
            gen_custody_trace(&w, &shell, NCT_DEPTH),
            Err(WitnessError::OutputMismatch { .. })
        ));
        
    }

    #[test]
    fn bind_rejections() {
        let f = fixture(0xA9, false);
        let mut w = f.witness.clone();
        w.inputs[0].nullifier_key[9] ^= 1;
        assert!(matches!(
            bind_witness_to_shell(&w, &f.shell),
            Err(WitnessError::NullifierKeyMismatch { index: 0 })
        ));
         let mut w = f.witness.clone();
        let mut pk = *w.inputs[0].opening.pk_n.as_bytes();
        pk[3] ^= 1;
        w.inputs[0].opening.pk_n = Hash256::from_bytes(pk);
        assert!(matches!(
            bind_witness_to_shell(&w, &f.shell),
            Err(WitnessError::NullifierKeyMismatch { index: 0 })
        ));
    }

    #[test]
    fn d3_revert_and_burn_end_to_end() {
        let mut rng = SplitMix64::new(0xD3);
        let wk = WalletKeys::from_master(&MasterSeed::from_bytes(rng.bytes32()));
        let det = wk.viewing();
        let g = ShardSet::genesis();
        let addr = |i: u64| nerv_custody::Address::generate(&det, wk.nullifier_key(), i, &g).unwrap();
        let opening = |v: u64, i: u64| NoteOpening {
            value: v,
            rho: rng.bytes32(),
            delivery: *addr(i).delivery().as_bytes(),
            blinding: rng.bytes32(),
            pk_n: *addr(i).pk_n(),
        };
        let input = opening(1000, 0);
        let out = opening(500, 1); // conditional, shard 40
        let revert = opening(500, 1);

        let mut tree = NoteCommitmentTree::new();
        let idx = tree.append(&input.commitment().unwrap()).unwrap();
        let root = tree.root();
        let sibs: Vec<[Goldilocks; 4]> = tree
            .witness(idx)
            .unwrap()
            .siblings
            .iter()
            .map(|d| d.to_elements().unwrap())
            .collect();
        let nf = derive_nullifier(&wk.nullifier_key_at(0), &input.rho);

        let leg7 = LegShell {
            shard: g.ids()[7],
            inputs: InputSet::new(vec![nf]),
            outputs: vec![],
            fee: FeeSats::from_u64(100),
            anchor: root.as_hash256(),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };
        let leg40 = LegShell {
            shard: g.ids()[40],
            inputs: InputSet::new(vec![]),
            outputs: vec![Output {
                cm: out.commitment().unwrap(),
                sealed_note: vec![0xA5; 48],
                value: 500,
                conditional: true,
                revert_cm: Some(revert.commitment().unwrap()),
            }],
            fee: FeeSats::from_u64(0),
            anchor: Hash256::from_bytes(rng.bytes32()),
            expiry: Height::from_u64(5_000),
            weight_version: 1,
            ct: vec![],
            burns: vec![],
        };
        let shell = TransactionShell { legs: vec![leg7, leg40] };
        let txid = shell.txid().unwrap();
        let burn_cm = *BurnCommitment::new(&txid, LegIndex::from_u8(0), 400).unwrap().as_hash();

        let witness = CustodyWitness {
            inputs: vec![InputWitness {
                opening: input,
                nullifier_key: wk.nullifier_key_at(0),
                leaf_index: idx,
                siblings: sibs,
                anchor: root,
            }],
            outputs: vec![OutputWitness { opening: out }],
            reverts: vec![RevertWitness { opening: revert }],
            burns: vec![BurnWitness { leg: 0, value: 400, commitment: burn_cm }],
            fees: vec![100, 0],
        };

        let air = CustodyAir::new(1, 1, vec![0], 1, NCT_DEPTH);
        let ct = gen_custody_trace(&witness, &shell, NCT_DEPTH).unwrap();
        assert_eq!(ct.trace.len(), air.rows());
        assert_eq!(air.n_windows(), 25 + 22 + 22 + 1);
        NativeEval::check_with_prep(
            ct.trace.clone(), ct.prep.clone(), ct.publics.clone(), &air, 8,
        )
        .unwrap();
        // Publics: anchor(4) ‖ nf(8) ‖ out cm(8) ‖ revert cm(8) ‖ burn cm(8).
        assert_eq!(ct.publics.len(), 4 + 8 + 8 + 8 + 8);

        // Revert-equation tamper: a revert register bit.
        let mut bad = ct.trace.clone();
        bad[500][air.rev_reg(0)] = bad[500][air.rev_reg(0)] + Goldilocks::ONE;
        let errs = NativeEval::check_with_prep(bad, ct.prep.clone(), ct.publics.clone(), &air, 8)
            .unwrap_err();
        assert!(errs.iter().any(|e| e.name == "hb_revert_eq" || e.name == "reg_copy"));

        // Burn value tamper: conservation + message wiring break.
        let mut bad = ct.trace.clone();
        bad[500][air.burn_reg(0)] = bad[500][air.burn_reg(0)] + Goldilocks::ONE;
        assert!(NativeEval::check_with_prep(bad, ct.prep.clone(), ct.publics.clone(), &air, 8)
            .is_err());

        // Wrong burn public.
        let mut pubs = ct.publics.clone();
        let burn_off = 4 + 8 + 8 + 8;
        pubs[burn_off] = pubs[burn_off] + Goldilocks::ONE;
        let errs =
            NativeEval::check_with_prep(ct.trace.clone(), ct.prep.clone(), pubs, &air, 8).unwrap_err();
        assert!(errs.iter().any(|e| e.name == "hb_tgt_burn"));

        // Bind-side: value mismatch caught pre-proof.
        let mut w2 = witness.clone();
        w2.reverts[0].opening.value = 501;
        assert!(matches!(
            bind_witness_to_shell(&w2, &shell),
            Err(WitnessError::RevertValueMismatch { index: 0 })
        ));
        let mut w3 = witness;
        w3.burns[0].value = 401;
        assert!(matches!(
            bind_witness_to_shell(&w3, &shell),
            Err(WitnessError::BurnMismatch { index: 0 })
        ));
    }
}

