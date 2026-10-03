//! R_q arithmetic and the negacyclic NTT (WP §6.3.1).
//!
//! R_q = Z_q[x]/(x^256 + 1), q = 3·2^30 + 1 — a 32-bit Proth prime with
//! q ≡ 1 mod 512, so a primitive 512-th root of unity ψ exists and the
//! complete negacyclic NTT of length 256 is available. Primality is proven
//! at compile time via Proth's theorem (`PROTH_WITNESS`); a deterministic
//! Miller–Rabin re-verifies at runtime.
//!
//! NTT construction (verified by hand at n = 2 and n = 4, and
//! differentially against the schoolbook twin at n = 256): the forward
//! transform computes the natural-order evaluations a(r_k) with
//! r_k = ψ^{2k+1} — bit-reverse the coefficients, then log₂n Cooley–Tukey
//! levels where level m pairs (i+j, i+j+m) with twiddles
//! ζ_{m,j} = ψ^{n(2j+1)/(2m)} (precomputed as const tables). Evaluation is
//! a ring homomorphism into pointwise arithmetic, so products mod x^n+1
//! are pointwise products in the NTT domain; the inverse runs the levels
//! reversed with ψ^{-1} twiddles, scales by n^{-1}, and bit-reverses.
//!
//! Module shapes (reconciled §6.3.1, erratum 20): A ∈ R^{8×8}; s, r, e₁ ∈
//! R^8; T = s·A + E₀ ∈ R^{2×8} (the WP's "T = A·s" up to the transpose
//! convention that makes v − s·u exact with u = A·r + e₁; both T rows
//! share the s·A base with independent noise rows, preserving s ∈ R^8
//! (rank 8, dimension 2,048), T ∈ R^{2×8}, and §6.3.3's single partial
//! p_j = λ_j·s_j·u serving both components); u = A·r + e₁ ∈ R^8; v = T·r +
//! e₂ + scale·m ∈ R^2; decryption computes v^{(i)} − s·u componentwise.
//!
//! Domain discipline: `Poly`/`Vec8`/`Vec2`/`Mat8x8`/`Mat2x8` are
//! coefficient-domain; the `*Ntt` types are evaluation-domain;
//! multiplication happens only pointwise in the NTT domain. Fast paths and
//! schoolbook reference twins (`*_reference`, DSR-7) are differentially
//! tested bit-for-bit — `ring::ntt` is a listed fuzz target and these
//! twins are its oracles. No randomness or hashing lives here; sampling
//! (part 2) owns XOF expansion, key lifecycle owns zeroization.

use std::fmt;
use crate::error::SealError;

pub const N: usize = 256;
pub const LOG2_N: u32 = 8;

/// R_q = Z_q[x]/(x^256 + 1), q = 0xFFF00001 = 4095·2^20 + 1 — a 32-bit
/// Proth prime with q ≡ 1 mod 512, so a primitive 512-th root of unity ψ
/// exists and the complete negacyclic NTT of length 256 is available.
/// Primality is proven at compile time via Proth's theorem
/// (`PROTH_WITNESS`); a deterministic Miller–Rabin re-verifies at runtime.


pub const Q: u64 = 4_293_918_721; // 0xFFF00001 = 4095·2^20 + 1


pub const HALF_Q: u64 = (Q - 1) / 2;

const fn const_mul(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % Q as u128) as u64
}

const fn const_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = const_mul(acc, base);
        }
        base = const_mul(base, base);
        exp >>= 1;
    }
    acc
}

const fn find_generator() -> u64 {
    // Q − 1 = 2^20·3^2·5·7·13 (erratum 58): g generates Z_q^× iff
    // g^((Q−1)/p) ≠ 1 for every prime p dividing Q − 1.
    let mut g = 2u64;
    loop {
        if const_pow(g, (Q - 1) / 2) != 1
            && const_pow(g, (Q - 1) / 3) != 1
            && const_pow(g, (Q - 1) / 5) != 1
            && const_pow(g, (Q - 1) / 7) != 1
            && const_pow(g, (Q - 1) / 13) != 1
        {
            return g;
        }
        g += 1;
    }
}


/// Smallest generator of Z_q^×. Pub: the circuit twin (seal_chip) and the
/// conformance vectors pin against it.
pub const GENERATOR: u64 = find_generator();

/// Primitive 512-th root of unity: ψ = g^((Q−1)/512). The negacyclic NTT of
/// length 256 needs exactly a 2n-th root.
pub const PSI: u64 = const_pow(GENERATOR, (Q - 1) / 512);
pub const PSI_INV: u64 = const_pow(PSI, 511);
pub const N_INV: u64 = const_pow(N as u64, Q - 2);

/// Proth's theorem: p = k·2^n + 1 (k odd, k < 2^n) is prime iff some a
/// satisfies a^((p−1)/2) ≡ −1 (mod p). Q = 3·2^30 + 1 qualifies; the const
/// eval of this search IS the compile-time primality proof (a composite Q
/// would make this loop never terminate and fail the build).
pub const PROTH_WITNESS: u64 = proth_witness();

const fn proth_witness() -> u64 {
    let mut a = 2u64;
    loop {
        if const_pow(a, (Q - 1) / 2) == Q - 1 {
            return a;
        }
        a += 1;
    }
}

const _: () = assert!((Q >> 31) == 1 && Q % 2 == 1 && (Q - 1) % 512 == 0);
const _: () = assert!(N == 1usize << LOG2_N);
const _: () = assert!(const_pow(PSI, 512) == 1);
const _: () = assert!(const_pow(PSI, 256) == Q - 1); // order exactly 512
const _: () = assert!(const_mul(PSI, PSI_INV) == 1);
const _: () = assert!(const_mul(N_INV, N as u64) == 1);

// Level-ℓ twiddles: ζ_{m,j} = ψ^{N(2j+1)/(2m)} for m = 2^ℓ, j < m. Rows are
// zero-padded to N/2; only j < 2^ℓ is meaningful.
const TWIDDLE_LEN: usize = N / 2;

const fn build_twiddles(root: u64) -> [[u64; TWIDDLE_LEN]; LOG2_N as usize] {
    let mut t = [[0u64; TWIDDLE_LEN]; LOG2_N as usize];
    let mut lvl = 0usize;
    while lvl < LOG2_N as usize {
        let m = 1usize << lvl;
        let z1 = const_pow(root, (N / (2 * m)) as u64);
        let z1_sq = const_mul(z1, z1);
        let mut w = z1;
        let mut j = 0usize;
        while j < m {
            t[lvl][j] = w;
            w = const_mul(w, z1_sq);
            j += 1;
        }
        lvl += 1;
    }
    t
}

const FWD_TWIDDLES: [[u64; TWIDDLE_LEN]; LOG2_N as usize] = build_twiddles(PSI);
const INV_TWIDDLES: [[u64; TWIDDLE_LEN]; LOG2_N as usize] = build_twiddles(PSI_INV);

/// Forward NTT twiddle at Cooley–Tukey `level` (0-based, 0..LOG2_N) for
/// butterfly index `idx` (0..2^level). Public surface so the seal chip
/// (which constrains twiddle usage bit-by-bit at fixed indices) can name
/// the same constants the native uses — DSR-7's twin-pair requirement.
#[inline]
pub const fn fwd_twiddle(level: usize, idx: usize) -> u64 {
    FWD_TWIDDLES[level][idx]
}

/// Inverse NTT twiddle at Cooley–Tukey `level` (0-based, 0..LOG2_N) for
/// butterfly index `idx` (0..2^level). Public surface for the seal chip's
/// inverse-transform constraints.
#[inline]
pub const fn inv_twiddle(level: usize, idx: usize) -> u64 {
    INV_TWIDDLES[level][idx]
}

/// Bit-reverse the lowest 8 bits of `x`. The seal chip's fixed-index
/// bit-reversal constraints reference this directly (erratum 83): each
/// `bitrev8(k)` is a public constant the chip's `bit_rev` family binds
/// against. Accepts `usize` so callers don't have to `.try_into()`-cast
/// the row/column indices that are already `usize` in chip-builder code.
#[inline]
pub const fn bitrev8(x: usize) -> usize {
    let mut x = (x as u8) as usize;
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    x
}

#[inline]
fn add(a: u64, b: u64) -> u64 {
    (a + b) % Q // a, b < Q ⇒ sum < 2Q < 2^33: no overflow
}

#[inline]
fn sub(a: u64, b: u64) -> u64 {
    (a + Q - b) % Q
}

fn bitrev(a: &mut [u64]) {
    let n = a.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
}

fn fmt_limbs(name: &str, a: &[u64], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{name}[{:05x} {:05x} {:05x} … {} coeffs]", a[0], a[1], a[2], a.len())
}

// ---------------------------------------------------------------------------
// Poly — coefficient domain, canonical coefficients in [0, Q)
// ---------------------------------------------------------------------------

/// Strip trailing-zero coefficients so a polynomial's highest-degree slot
/// is non-zero; returns the empty `Vec` if and only if the input is the
/// zero polynomial. Operates over any coefficient ring — does NOT reduce
/// non-canonical coefficients mod `Q`. `try_invert` calls this on data
/// that has already been canonicalized (every slot ∈ [0, Q)), so no
/// reduction is needed here.
fn vec_trim(v: &[u64]) -> Vec<u64> {
    let mut n = v.len();
    while n > 0 && v[n - 1] == 0 {
        n -= 1;
    }
    v[..n].to_vec()
}

/// Modular exponentiation over `u128` to dodge the 32-bit × 32-bit overflow
/// edge: (Q − 1)² ≈ 1.84 × 10¹⁹ which is just under 2⁶⁴ but *barely*, and
/// u128 keeps the multiplication honest on every u64 without panicking.
///
/// Right-to-left square-and-multiply, O(log exp) modular multiplications.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    debug_assert!(modulus > 1);
    let m = modulus as u128;
    let mut result: u64 = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % m) as u64;
        }
        exp >>= 1;
        if exp > 0 {
            base = ((base as u128 * base as u128) % m) as u64;
        }
    }
    result
}

/// Polynomial multiplication in Z_q[x] (schoolbook convolution, no ring
/// reduction). Returns the empty Vec iff either input is empty.
///
/// `try_invert` is the only caller; it consumes the result of `poly_mul`
/// inside the extended Euclidean step where ring reduction has not yet
/// happened. Coefficients that exceed `Q` are reduced mod `Q` so the
/// caller's polynomial-degree invariant (leading ≠ 0 ⇒ deg ≤ len−1)
/// stays monotone.
fn poly_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let m = Q as u128;
    let n = a.len() + b.len() - 1;
    let mut out = vec![0u64; n];
    for i in 0..a.len() {
        let ai = a[i];
        if ai == 0 {
            continue;
        }
        for j in 0..b.len() {
            let bj = b[j];
            if bj == 0 {
                continue;
            }
            let prod = (ai as u128 * bj as u128) % m;
            let sum = out[i + j] as u128 + prod;
            out[i + j] = (sum % m) as u64;
        }
    }
    out
}

/// Coefficient-wise subtraction `a − b` in Z_q. The output length is
/// `max(a.len(), b.len())` so the caller's polynomial-degree invariant
/// is preserved; the resultant leading coefficient is reduced mod `Q`
/// before storing (the natural `a_i + Q − b_i` lands in `[0, 2Q)`).
fn poly_sub(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let m = Q;
    let mut out = vec![0u64; n];
    for i in 0..n {
        let av = if i < a.len() { a[i] } else { 0 };
        let bv = if i < b.len() { b[i] } else { 0 };
        // (av + Q − bv) ∈ [0, 2Q); one conditional subtract normalizes.
        let v = av + m - bv;
        out[i] = if v >= m { v - m } else { v };
    }
    out
}

/// Polynomial long division in Z_q[x]. Returns `(quotient, remainder)`
/// with `a == b·q + r (mod Q)` and `deg(r) < deg(b)` (or `r` empty iff
/// `a < b`). Returns `None` when `b` is the zero polynomial — the
/// leading-coefficient inversion has no inverse there and the caller
/// (extended Euclidean) cannot proceed.
///
/// Invariant: leading coefficients of `a` and `b` are assumed ≥ 1 and `< Q`
/// already; the GCD step in `try_invert` feeds us trimmed polynomials so
/// the “leading is zero ⟹ degree < len” ambiguity never arises.
fn poly_divmod(a_in: &[u64], b_in: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    let b_trim_len = b_in.iter().rposition(|&x| x != 0).map(|i| i + 1).unwrap_or(0);
    if b_trim_len == 0 {
        return None;
    }
    let b = &b_in[..b_trim_len];

    let lead_b = *b.last().expect("b_trim_len > 0 ⇒ b.last() is Some");
    let lead_b_inv = mod_pow(lead_b, Q - 2, Q);

    let mut r = a_in.to_vec();
    let q_len = if a_in.len() >= b.len() { a_in.len() - b.len() + 1 } else { 0 };
    let mut q = vec![0u64; q_len];

    loop {
        let r_deg = match r.iter().rposition(|&x| x != 0) {
            Some(i) => i,
            None => break,
        };
        let b_deg = b.len() - 1;
        if r_deg < b_deg {
            break;
        }
        let lead_r = r[r_deg];
        let coef = ((lead_r as u128 * lead_b_inv as u128) % Q as u128) as u64;
        let pos = r_deg - b_deg;
        q[pos] = coef;
        // Subtract `coef · x^pos · b` from r in Z_q.
        for (j, &bc) in b.iter().enumerate() {
            let idx = pos + j;
            let prod = ((coef as u128 * bc as u128) % Q as u128) as u64;
            let new = r[idx] + Q - prod;
            r[idx] = if new >= Q { new - Q } else { new };
        }
        // Strip the now-zero leading slot.
        while r.last().copied() == Some(0) {
            r.pop();
        }
    }

    Some((q, r))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Poly([u64; N]);


impl Poly {
    pub const WIRE_SIZE: usize = N * 4; // Q < 2^32 ⇒ u32 per coefficient

    pub const fn new(coefficients: [u64; N]) -> Poly {
        Poly(coefficients)
    }

    pub const fn zero() -> Poly {
        Poly([0; N])
    }

    pub fn coefficient(&self, i: usize) -> u64 {
        self.0[i]
    }

    pub fn coefficients(&self) -> &[u64; N] {
        &self.0
    }

    /// Canonical embedding of centered (signed) values — the constructor
    /// for short secrets and noise once sampled. Total for all i64.
    pub fn from_centered(vals: &[i64; N]) -> Poly {
        let mut a = [0u64; N];
        for (o, &v) in a.iter_mut().zip(vals.iter()) {
            *o = if v >= 0 {
                (v as u64) % Q
            } else {
                (Q - (v.unsigned_abs() % Q)) % Q
            };
        }
        Poly(a)
    }

    pub fn add(&self, rhs: &Poly) -> Poly {
        let mut out = [0u64; N];
        for (o, (&x, &y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = add(x, y);
        }
        Poly(out)
    }

    pub fn sub(&self, rhs: &Poly) -> Poly {
        let mut out = [0u64; N];
        for (o, (&x, &y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = sub(x, y);
        }
        Poly(out)
    }

    pub fn neg(&self) -> Poly {
        let mut out = [0u64; N];
        for (o, &x) in out.iter_mut().zip(self.0.iter()) {
            *o = sub(0, x);
        }
        Poly(out)
    }

    pub fn mul_scalar(&self, k: u64) -> Poly {
        let mut out = [0u64; N];
        for (o, &x) in out.iter_mut().zip(self.0.iter()) {
            *o = const_mul(x, k % Q);
        }
        Poly(out)
    }

    /// Negacyclic product mod x^256 + 1 (fast: NTT pointwise).
    pub fn mul(&self, rhs: &Poly) -> Poly {
        self.ntt().mul_pointwise(&rhs.ntt()).intt()
    }

    /// Schoolbook negacyclic product — the DSR-7 native reference twin.
    /// O(n²); used by the differential suite and never on the hot path.
    pub fn mul_reference(&self, rhs: &Poly) -> Poly {
        let mut c = [0u64; N];
        for i in 0..N {
            let ai = self.0[i];
            if ai == 0 {
                continue;
            }
            for j in 0..N {
                let prod = const_mul(ai, rhs.0[j]);
                let k = i + j;
                if k < N {
                    c[k] = add(c[k], prod);
                } else {
                    c[k - N] = sub(c[k - N], prod); // x^N ≡ −1
                }
            }
        }
        Poly(c)
    }

    pub fn ntt(&self) -> PolyNtt {
        let mut a = self.0;
        bitrev(&mut a);
        let mut m = 1usize;
        while m < N {
            let tw = &FWD_TWIDDLES[m.trailing_zeros() as usize];
            let mut i = 0usize;
            while i < N {
                for j in 0..m {
                    let u = a[i + j];
                    let v = const_mul(a[i + j + m], tw[j]);
                    a[i + j] = add(u, v);
                    a[i + j + m] = sub(u, v);
                }
                i += m << 1;
            }
            m <<= 1;
        }
        PolyNtt(a)
    }

    /// Balanced representatives in [−(Q−1)/2, (Q−1)/2].
    pub fn centerlift(&self) -> [i64; N] {
        let mut out = [0i64; N];
        for (o, &c) in out.iter_mut().zip(self.0.iter()) {
            *o = if c <= HALF_Q { c as i64 } else { c as i64 - Q as i64 };
        }
        out
    }

    pub fn to_bytes(&self) -> [u8; Poly::WIRE_SIZE] {
        let mut out = [0u8; Poly::WIRE_SIZE];
        for (i, &c) in self.0.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&(c as u32).to_le_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Poly, SealError> {
        if bytes.len() != Poly::WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: Poly::WIRE_SIZE });
        }
        let mut a = [0u64; N];
        for (i, v) in a.iter_mut().enumerate() {
            let mut b = [0u8; 4];
            b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            let c = u32::from_le_bytes(b) as u64;
            if c >= Q {
                return Err(SealError::UnreducedCoefficient { value: c });
            }
            *v = c;
        }
        Ok(Poly(a))
    }

     /// The constant polynomial 1 (the multiplicative identity).
    pub const fn one() -> Poly {
        let mut a = [0u64; N];
        a[0] = 1;
        Poly(a)
    }

    /// The monomial x^k in R_q (k reduced mod 2N; x^N = −1).
    pub fn monomial(k: usize) -> Poly {
        let k = k % (2 * N);
        let mut a = [0u64; N];
        if k == 0 {
            a[0] = 1;
        } else if k < N {
            a[k] = 1;
        } else {
            a[k - N] = Q - 1;
        }
        Poly(a)
    }

    /// self · x^k (k reduced mod 2N): a rotation, with sign from x^N = −1.
    pub fn mul_monomial(&self, k: usize) -> Poly {
        let k = k % (2 * N);
        if k == 0 {
            return Poly(self.0);
        }
        let (shift, flip) = if k < N { (k, false) } else { (k - N, true) };
        if shift == 0 {
            // x^N · p = −p
            return self.neg();
        }
        let mut out = [0u64; N];
        for (j, o) in out.iter_mut().enumerate() {
            let v = if j >= shift { self.0[j - shift] } else { self.0[j - shift + N] };
            let wrap = j < shift;
            *o = if wrap ^ flip { sub(0, v) } else { v };
        }
        Poly(out)
    }

    /// Multiplicative inverse in R_q, via the extended Euclidean algorithm
    /// on (x^N + 1, self) over Z_q[x]. `None` iff gcd is non-constant (a
    /// zero-divisor in R_q) — non-invertible iff `self` shares a root of
    /// x^N + 1 in Z_q (the primitive 512-th roots of unity).
    ///
    /// Algorithm:
    ///   1. constant-poly fast path: Fermat inversion (Q is prime).
    ///   2. trim trailing zeros so degree = b.len() - 1.
    ///   3. extended Euclidean in Z_q[x] on (x^N+1, self).  Outputs `t`
    ///      with `self · t + (x^N+1) · t' = gcd ∈ Z_q`.
    ///   4. gcd must be a non-zero constant (else self is a zero-divisor).
    ///   5. scale `t` by gcd^(-1) mod Q so `self · t ≡ 1 (mod q)`.
    ///   6. reduce mod x^N+1: x^N ≡ -1 ⇒ coefficient at `i+N` folds (negated)
    ///      into position `i mod N` repeatedly.  Resulting coefficients are
    ///      canonical in [0, Q) (no further centerlift needed — the test
    ///      oracle's expected forms are already canonical).
    pub fn try_invert(&self) -> Option<Poly> {
        let b = vec_trim(&self.0);
        if b.is_empty() {
            return None;
        }
        // (1) constant-poly fast path: gcd is trivially 1; Fermat inversion.
        if b.len() == 1 {
            let inv = mod_pow(b[0], Q - 2, Q);
            let mut out = [0u64; N];
            out[0] = inv;
            return Some(Poly(out));
        }
        // (2-3) extended Euclidean in Z_q[x].
        let mut modulus = vec![0u64; N + 1];
        modulus[0] = 1;
        modulus[N] = 1;
        let mut old_r = modulus;
        let mut r = b;
        let mut old_t: Vec<u64> = Vec::new();
        let mut t: Vec<u64> = vec![1];
        while !r.is_empty() {
            let (q, rem) = poly_divmod(&old_r, &r)?;
            old_r = r;
            r = rem;
            let qt = poly_mul(&q, &t);
            let new_t = poly_sub(&old_t, &qt);
            old_t = t;
            t = new_t;
        }
        // (4) gcd must be a non-zero constant.
        if old_r.len() != 1 || old_r[0] == 0 {
            return None;
        }
        // (5) scale t by gcd^(-1).
        let gcd_inv = mod_pow(old_r[0], Q - 2, Q);
        let m = Q as u128;
        let scaled: Vec<u64> = t
            .iter()
            .map(|&c| (((c as u128 * gcd_inv as u128) % m) as u64))
            .collect();
        // (6) reduce mod x^N + 1: coefficient at i+N folds (negated) into i mod N.
        let mut out = [0u64; N];
        for (i, &c) in scaled.iter().enumerate() {
            let pos = i % N;
            // (-c) canonical in [0, Q): 0 stays 0; non-zero becomes Q - c.
            let neg_c = if c == 0 { 0 } else { Q - c };
            let new = out[pos] + neg_c;
            out[pos] = if new >= Q { new - Q } else { new };
        }
        Some(Poly(out))
    }
}

impl Default for Poly {
    fn default() -> Self {
        Poly::zero()
    }
}

impl fmt::Debug for Poly {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_limbs("Poly", &self.0, f)
    }
}

// ---------------------------------------------------------------------------
// PolyNtt — evaluation domain (values a(ψ^{2k+1}))
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PolyNtt([u64; N]);


impl PolyNtt {
    pub const fn zero() -> PolyNtt {
        PolyNtt([0; N])
    }

    pub fn values(&self) -> &[u64; N] {
        &self.0
    }

    pub fn add(&self, rhs: &PolyNtt) -> PolyNtt {
        let mut out = [0u64; N];
        for (o, (&x, &y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = add(x, y);
        }
        PolyNtt(out)
    }

    pub fn mul_pointwise(&self, rhs: &PolyNtt) -> PolyNtt {
        let mut out = [0u64; N];
        for (o, (&x, &y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = const_mul(x, y);
        }
        PolyNtt(out)
    }

    pub fn intt(&self) -> Poly {
        let mut a = self.0;
        let mut m = N >> 1;
        while m >= 1 {
            let tw = &INV_TWIDDLES[m.trailing_zeros() as usize];
            let mut i = 0usize;
            while i < N {
                for j in 0..m {
                    let s = a[i + j];
                    let t = a[i + j + m];
                    a[i + j] = add(s, t);
                    a[i + j + m] = const_mul(sub(s, t), tw[j]);
                }
                i += m << 1;
            }
            m >>= 1;
        }
        for x in a.iter_mut() {
            *x = const_mul(*x, N_INV);
        }
        bitrev(&mut a);
        Poly(a)
    }
}

impl Default for PolyNtt {
    fn default() -> Self {
        PolyNtt::zero()
    }
}

impl fmt::Debug for PolyNtt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_limbs("PolyNtt", &self.0, f)
    }
}

// ---------------------------------------------------------------------------
// Fixed-shape containers
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Vec8 — s, r, e₁, u; Mat2x8 rows
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Vec8([Poly; 8]);

impl Vec8 {
    pub const WIRE_SIZE: usize = 8 * Poly::WIRE_SIZE; // 8192: u's genesis wire size (§6.3.8)

    pub const fn new(polys: [Poly; 8]) -> Vec8 {
        Vec8(polys)
    }

    pub const fn zero() -> Vec8 {
        Vec8([Poly::zero(); 8])
    }

    pub fn poly(&self, i: usize) -> &Poly {
        &self.0[i]
    }

    pub fn polys(&self) -> &[Poly; 8] {
        &self.0
    }

    pub fn add(&self, rhs: &Vec8) -> Vec8 {
        let mut out = [Poly::zero(); 8];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = x.add(y);
        }
        Vec8(out)
    }

    pub fn sub(&self, rhs: &Vec8) -> Vec8 {
        let mut out = [Poly::zero(); 8];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = x.sub(y);
        }
        Vec8(out)
    }

    /// Aggregate addition (WP §6.3.2): componentwise polynomial addition —
    /// linear, order-invariant, exactly what producers apply to build ct_B.
    pub fn ntt(&self) -> Vec8Ntt {
        let mut out = [PolyNtt::zero(); 8];
        for (o, p) in out.iter_mut().zip(self.0.iter()) {
            *o = p.ntt();
        }
        Vec8Ntt(out)
    }

    /// s·u = Σ_k s_k·u_k (fast path) — the decryption dot product.
    pub fn dot(&self, rhs: &Vec8) -> Poly {
        self.ntt().dot(&rhs.ntt()).intt()
    }

    /// DSR-7 reference twin of `dot`.
    pub fn dot_reference(&self, rhs: &Vec8) -> Poly {
        let mut acc = Poly::zero();
        for k in 0..8 {
            acc = acc.add(&self.0[k].mul_reference(&rhs.0[k]));
        }
        acc
    }

    pub fn to_bytes(&self) -> [u8; Vec8::WIRE_SIZE] {
        let mut out = [0u8; Vec8::WIRE_SIZE];
        for (i, p) in self.0.iter().enumerate() {
            out[i * Poly::WIRE_SIZE..(i + 1) * Poly::WIRE_SIZE].copy_from_slice(&p.to_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Vec8, SealError> {
        if bytes.len() != Vec8::WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: Vec8::WIRE_SIZE });
        }
        let mut out = [Poly::zero(); 8];
        for (i, p) in out.iter_mut().enumerate() {
            *p = Poly::from_bytes(&bytes[i * Poly::WIRE_SIZE..(i + 1) * Poly::WIRE_SIZE])?;
        }
        Ok(Vec8(out))
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Vec8Ntt([PolyNtt; 8]);

impl Vec8Ntt {
    pub const fn zero() -> Vec8Ntt {
        Vec8Ntt([PolyNtt::zero(); 8])
    }

    pub fn add(&self, rhs: &Vec8Ntt) -> Vec8Ntt {
        let mut out = [PolyNtt::zero(); 8];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = x.add(y);
        }
        Vec8Ntt(out)
    }

    pub fn dot(&self, rhs: &Vec8Ntt) -> PolyNtt {
        let mut acc = PolyNtt::zero();
        for k in 0..8 {
            acc = acc.add(&self.0[k].mul_pointwise(&rhs.0[k]));
        }
        acc
    }

    pub fn intt(&self) -> Vec8 {
        let mut out = [Poly::zero(); 8];
        for (o, p) in out.iter_mut().zip(self.0.iter()) {
            *o = p.intt();
        }
        Vec8(out)
    }
}

// ---------------------------------------------------------------------------
// Vec2 — v, e₂, m
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Vec2([Poly; 2]);

impl Vec2 {
    pub const WIRE_SIZE: usize = 2 * Poly::WIRE_SIZE; // 2048

    pub const fn new(polys: [Poly; 2]) -> Vec2 {
        Vec2(polys)
    }

    pub const fn zero() -> Vec2 {
        Vec2([Poly::zero(); 2])
    }

    pub fn poly(&self, i: usize) -> &Poly {
        &self.0[i]
    }

    pub fn polys(&self) -> &[Poly; 2] {
        &self.0
    }

    pub fn add(&self, rhs: &Vec2) -> Vec2 {
        let mut out = [Poly::zero(); 2];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = x.add(y);
        }
        Vec2(out)
    }

    pub fn sub(&self, rhs: &Vec2) -> Vec2 {
        let mut out = [Poly::zero(); 2];
        for (o, (x, y)) in out.iter_mut().zip(self.0.iter().zip(rhs.0.iter())) {
            *o = x.sub(y);
        }
        Vec2(out)
    }

    pub fn ntt(&self) -> Vec2Ntt {
        Vec2Ntt([self.0[0].ntt(), self.0[1].ntt()])
    }

    pub fn to_bytes(&self) -> [u8; Vec2::WIRE_SIZE] {
        let mut out = [0u8; Vec2::WIRE_SIZE];
        for (i, p) in self.0.iter().enumerate() {
            out[i * Poly::WIRE_SIZE..(i + 1) * Poly::WIRE_SIZE].copy_from_slice(&p.to_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Vec2, SealError> {
        if bytes.len() != Vec2::WIRE_SIZE {
            return Err(SealError::BadLength { len: bytes.len(), expected: Vec2::WIRE_SIZE });
        }
        Ok(Vec2([
            Poly::from_bytes(&bytes[0..Poly::WIRE_SIZE])?,
            Poly::from_bytes(&bytes[Poly::WIRE_SIZE..2 * Poly::WIRE_SIZE])?,
        ]))
    }
}

// -- nerv-codec Encode/Decode impls for the wire-domain ring types -----------
//
// We piggy-back on the existing `to_bytes`/`from_bytes` canonical layout
// (u32 LE per coefficient — already the public API). Adding trait impls
// here lets the higher-level `TransactionWitness` round-trip through
// `nerv_core::codec` without duplicating the byte-level definition.

impl nerv_core::codec::Encode for Poly {
    fn encode_into(&self, out: &mut Vec<u8>) {
        let bytes = self.to_bytes();
        out.extend_from_slice(&bytes);
    }
    fn encoded_len(&self) -> usize {
        Poly::WIRE_SIZE
    }
}

impl nerv_core::codec::Decode for Poly {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let mut bytes = [0u8; Poly::WIRE_SIZE];
        bytes.copy_from_slice(r.take(Poly::WIRE_SIZE)?);
        Poly::from_bytes(&bytes).map_err(|_| nerv_core::error::CodecError::InvariantViolated("poly length"))
    }
}

impl nerv_core::codec::Encode for Vec8 {
    fn encode_into(&self, out: &mut Vec<u8>) {
        for p in &self.0 {
            p.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        Vec8::WIRE_SIZE
    }
}

impl nerv_core::codec::Decode for Vec8 {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let mut polys = [Poly::default(); 8];
        for slot in polys.iter_mut() {
            *slot = Poly::decode_from(r)?;
        }
        Ok(Vec8(polys))
    }
}

impl nerv_core::codec::Encode for Vec2 {
    fn encode_into(&self, out: &mut Vec<u8>) {
        for p in &self.0 {
            p.encode_into(out);
        }
    }
    fn encoded_len(&self) -> usize {
        Vec2::WIRE_SIZE
    }
}

impl nerv_core::codec::Decode for Vec2 {
    fn decode_from(r: &mut nerv_core::codec::Reader<'_>) -> Result<Self, nerv_core::error::CodecError> {
        let mut polys = [Poly::default(); 2];
        for slot in polys.iter_mut() {
            *slot = Poly::decode_from(r)?;
        }
        Ok(Vec2(polys))
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Vec2Ntt([PolyNtt; 2]);

impl Vec2Ntt {
    pub const fn zero() -> Vec2Ntt {
        Vec2Ntt([PolyNtt::zero(); 2])
    }

    pub fn add(&self, rhs: &Vec2Ntt) -> Vec2Ntt {
        Vec2Ntt([self.0[0].add(&rhs.0[0]), self.0[1].add(&rhs.0[1])])
    }

    pub fn intt(&self) -> Vec2 {
        Vec2([self.0[0].intt(), self.0[1].intt()])
    }
}

// ---------------------------------------------------------------------------
// Mat8x8 — A (uniform, XOF-expanded in sampling)
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Mat8x8([[Poly; 8]; 8]);

impl Mat8x8 {
    pub const fn new(rows: [[Poly; 8]; 8]) -> Mat8x8 {
        Mat8x8(rows)
    }

    pub const fn zero() -> Mat8x8 {
        Mat8x8([[Poly::zero(); 8]; 8])
    }

    pub fn rows(&self) -> &[[Poly; 8]; 8] {
        &self.0
    }

    pub fn row(&self, i: usize) -> &[Poly; 8] {
        &self.0[i]
    }

    pub fn ntt(&self) -> Mat8x8Ntt {
        let mut out = [[PolyNtt::zero(); 8]; 8];
        for (o, row) in out.iter_mut().zip(self.0.iter()) {
            for (p, q) in o.iter_mut().zip(row.iter()) {
                *p = q.ntt();
            }
        }
        Mat8x8Ntt(out)
    }

    /// u = A·r (row i = Σ_j A_{ij}·r_j).
    pub fn mul_vec(&self, v: &Vec8) -> Vec8 {
        self.ntt().mul_vec(&v.ntt()).intt()
    }

    /// DSR-7 reference twin.
    pub fn mul_vec_reference(&self, v: &Vec8) -> Vec8 {
        let mut out = [Poly::zero(); 8];
        for (i, row) in self.0.iter().enumerate() {
            let mut acc = Poly::zero();
            for (a, b) in row.iter().zip(v.polys().iter()) {
                acc = acc.add(&a.mul_reference(b));
            }
            out[i] = acc;
        }
        Vec8(out)
    }

    /// s·A as a row ((A^T·s)_k = Σ_j A_{jk}·s_j) — the T-base (erratum 20).
    pub fn mul_vec_transpose(&self, v: &Vec8) -> Vec8 {
        self.ntt().mul_vec_transpose(&v.ntt()).intt()
    }

    /// DSR-7 reference twin.
    pub fn mul_vec_transpose_reference(&self, v: &Vec8) -> Vec8 {
        let mut out = [Poly::zero(); 8];
        for (j, row) in self.0.iter().enumerate() {
            for (k, a) in row.iter().enumerate() {
                out[k] = out[k].add(&a.mul_reference(v.poly(j)));
            }
        }
        Vec8(out)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Mat8x8Ntt([[PolyNtt; 8]; 8]);

impl Mat8x8Ntt {
    pub const fn zero() -> Mat8x8Ntt {
        Mat8x8Ntt([[PolyNtt::zero(); 8]; 8])
    }

    pub fn rows(&self) -> &[[PolyNtt; 8]; 8] {
        &self.0
    }

    pub fn row(&self, i: usize) -> &[PolyNtt; 8] {
        &self.0[i]
    }

    pub fn mul_vec(&self, v: &Vec8Ntt) -> Vec8Ntt {
        let mut out = [PolyNtt::zero(); 8];
        for (i, row) in self.0.iter().enumerate() {
            let mut acc = PolyNtt::zero();
            for (a, b) in row.iter().zip(v.polys().iter()) {
                acc = acc.add(&a.mul_pointwise(b));
            }
            out[i] = acc;
        }
        Vec8Ntt(out)
    }

    pub fn mul_vec_transpose(&self, v: &Vec8Ntt) -> Vec8Ntt {
        let mut out = [PolyNtt::zero(); 8];
        for (j, row) in self.0.iter().enumerate() {
            for (k, a) in row.iter().enumerate() {
                out[k] = out[k].add(&a.mul_pointwise(v.poly(j)));
            }
        }
        Vec8Ntt(out)
    }

    pub fn intt(&self) -> Mat8x8 {
        let mut out = [[Poly::zero(); 8]; 8];
        for (o, row) in out.iter_mut().zip(self.0.iter()) {
            for (p, q) in o.iter_mut().zip(row.iter()) {
                *p = q.intt();
            }
        }
        Mat8x8(out)
    }
}

// ---------------------------------------------------------------------------
// Mat2x8 — T = s·A + E₀
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Mat2x8([[Poly; 8]; 2]);

impl Mat2x8 {
    pub const fn new(rows: [[Poly; 8]; 2]) -> Mat2x8 {
        Mat2x8(rows)
    }

    pub const fn zero() -> Mat2x8 {
        Mat2x8([[Poly::zero(); 8]; 2])
    }

    pub fn rows(&self) -> &[[Poly; 8]; 2] {
        &self.0
    }

    pub fn row(&self, i: usize) -> &[Poly; 8] {
        &self.0[i]
    }

    pub fn ntt(&self) -> Mat2x8Ntt {
        let mut out = [[PolyNtt::zero(); 8]; 2];
        for (o, row) in out.iter_mut().zip(self.0.iter()) {
            for (p, q) in o.iter_mut().zip(row.iter()) {
                *p = q.ntt();
            }
        }
        Mat2x8Ntt(out)
    }

    /// v = T·r (row i = Σ_k T_{ik}·r_k).
    pub fn mul_vec(&self, v: &Vec8) -> Vec2 {
        self.ntt().mul_vec(&v.ntt()).intt()
    }

    /// DSR-7 reference twin.
    pub fn mul_vec_reference(&self, v: &Vec8) -> Vec2 {
        let mut out = [Poly::zero(); 2];
        for (i, row) in self.0.iter().enumerate() {
            let mut acc = Poly::zero();
            for (a, b) in row.iter().zip(v.polys().iter()) {
                acc = acc.add(&a.mul_reference(b));
            }
            out[i] = acc;
        }
        Vec2(out)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Mat2x8Ntt([[PolyNtt; 8]; 2]);

impl Mat2x8Ntt {
    pub const fn zero() -> Mat2x8Ntt {
        Mat2x8Ntt([[PolyNtt::zero(); 8]; 2])
    }

    pub fn rows(&self) -> &[[PolyNtt; 8]; 2] {
        &self.0
    }

    pub fn row(&self, i: usize) -> &[PolyNtt; 8] {
        &self.0[i]
    }

    pub fn mul_vec(&self, v: &Vec8Ntt) -> Vec2Ntt {
        let mut out = [PolyNtt::zero(); 2];
        for (i, row) in self.0.iter().enumerate() {
            let mut acc = PolyNtt::zero();
            for (a, b) in row.iter().zip(v.polys().iter()) {
                acc = acc.add(&a.mul_pointwise(b));
            }
            out[i] = acc;
        }
        Vec2Ntt(out)
    }

    pub fn intt(&self) -> Mat2x8 {
        let mut out = [[Poly::zero(); 8]; 2];
        for (o, row) in out.iter_mut().zip(self.0.iter()) {
            for (p, q) in o.iter_mut().zip(row.iter()) {
                *p = q.intt();
            }
        }
        Mat2x8(out)
    }
}

// Vec8Ntt accessors used above — implemented here to keep the section order
// readable.
impl Vec8Ntt {
    pub fn polys(&self) -> &[PolyNtt; 8] {
        &self.0
    }

    pub fn poly(&self, i: usize) -> &PolyNtt {
        &self.0[i]
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---- module-level Z_q[x] helpers (extended-Euclidean building
    // blocks for `Poly::try_invert`). These are PURE-FUNCTION tests on the
    // module-level helpers — they do NOT touch the NTT fast paths.

    /// Compute a · b mod Q using schoolbook (independent reference), so the
    /// production `poly_mul` has an oracle to test against.
    fn reference_poly_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; a.len() + b.len() - 1];
        let m = Q as u128;
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                let prod = (ai as u128 * bj as u128) % m;
                let sum = out[i + j] as u128 + prod;
                out[i + j] = (sum % m) as u64;
            }
        }
        out
    }

    fn reference_poly_sub(a: &[u64], b: &[u64]) -> Vec<u64> {
        let m = Q;
        let n = a.len().max(b.len());
        let mut out = vec![0u64; n];
        for i in 0..n {
            let av = if i < a.len() { a[i] } else { 0 };
            let bv = if i < b.len() { b[i] } else { 0 };
            out[i] = if av >= bv { (av - bv) % m } else { (av + m - bv) % m };
        }
        out
    }

    #[test]
    fn vec_trim_strips_only_trailing_zeros() {
        // Leading zeros survive (the polynomial 0 + x is still `x`).
        assert_eq!(vec_trim(&[0, 1, 2, 0]), vec![0, 1, 2]);
        // The zero polynomial trims to empty (its canonical form).
        assert_eq!(vec_trim(&[0u64; 8]), Vec::<u64>::new());
        // Already-trimmed input is a no-op.
        assert_eq!(vec_trim(&[1, 2, 3]), vec![1, 2, 3]);
        // Only trailing zeros are removed.
        assert_eq!(vec_trim(&[1, 0, 2, 0, 0]), vec![1, 0, 2]);
    }

    #[test]
    fn poly_mul_matches_schoolbook_oracle_randomized() {
        // Use a tiny non-splitmix64 deterministic seed so this test stays
        // fast and reproducible (the poly degree is small enough that
        // exhaustive-checking a few random cases catches real bugs).
        let mut s: u64 = 0xC0FFEE_BEEF;
        for _ in 0..16 {
            let a_len = (s % 8 + 1) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b_len = (s % 8 + 1) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut a = vec![0u64; a_len];
            let mut b = vec![0u64; b_len];
            for x in a.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            for x in b.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            assert_eq!(poly_mul(&a, &b), reference_poly_mul(&a, &b));
        }
    }

    #[test]
    fn poly_mul_empty_input_returns_empty() {
        assert_eq!(poly_mul(&[], &[1, 2]), Vec::<u64>::new());
        assert_eq!(poly_mul(&[1, 2], &[]), Vec::<u64>::new());
    }

    #[test]
    fn poly_sub_matches_canonical_subtraction_oracle() {
        let mut s: u64 = 0xBAD_F00D_DEAD_BEEF;
        for _ in 0..16 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a_len = (s % 8 + 1) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b_len = (s % 8 + 1) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut a = vec![0u64; a_len];
            let mut b = vec![0u64; b_len];
            for x in a.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            for x in b.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            assert_eq!(poly_sub(&a, &b), reference_poly_sub(&a, &b));
        }
    }

    #[test]
    fn poly_divmod_division_identity() {
        // For arbitrary non-zero b, a == b·(a/b) + (a mod b).
        let mut s: u64 = 0x0123_4567_89AB_CDEF;
        for _ in 0..16 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a_len = (s % 12 + 2) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let b_len = (s % 6 + 1) as usize;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut a = vec![0u64; a_len];
            let mut b = vec![0u64; b_len];
            for x in a.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            for x in b.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *x = s % Q;
            }
            // Trim b's leading zeros so `poly_divmod` doesn't think the
            // divisor is zero on a zero-leading test vector.
            let b_trim_end = b.iter().rposition(|&x| x != 0).map(|i| i + 1).unwrap_or(0);
            if b_trim_end == 0 {
                // skip the zero-divisor case (other tests cover it)
                continue;
            }
            let b = &b[..b_trim_end];
            if let Some((q, r)) = poly_divmod(&a, b) {
                // b · q + r should reconstruct a mod Q coefficient-wise.
                let m = Q;
                let mut prod = vec![0u64; b.len() + q.len() - 1];
                for (i, &bi) in b.iter().enumerate() {
                    for (j, &qj) in q.iter().enumerate() {
                        let v = (bi as u128 * qj as u128) % m as u128;
                        let sum = prod[i + j] as u128 + v;
                        prod[i + j] = (sum % m as u128) as u64;
                    }
                }
                let max_len = prod.len().max(r.len()).max(a.len());
                for i in 0..max_len {
                    let lh = if i < prod.len() { prod[i] } else { 0 };
                    let rh = if i < r.len() { r[i] } else { 0 };
                    let av = if i < a.len() { a[i] } else { 0 };
                    let sum = (lh as u128 + rh as u128) % m as u128;
                    assert_eq!(sum as u64, av, "i={i}, a={a:?}, b={b:?}, q={q:?}, r={r:?}");
                }
            }
        }
    }

    #[test]
    fn poly_divmod_zero_divisor_returns_none() {
        assert_eq!(poly_divmod(&[1, 2, 3], &[0, 0, 0]), None);
    }

    #[test]
    fn poly_divmod_smaller_dividend_gives_zero_quotient() {
        // When deg(a) < deg(b), q == [] and r == a (canonicalized).
        let (q, r) = poly_divmod(&[1, 2], &[3, 0, 1]).unwrap();
        assert_eq!(q, Vec::<u64>::new());
        assert_eq!(r, vec![1, 2]);
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    // Independent mod-pow (u128, separate code path from const_pow).
    fn t_pow(mut b: u64, mut e: u64, m: u64) -> u64 {
        let mut acc = 1u64;
        while e > 0 {
            if e & 1 == 1 {
                acc = ((acc as u128 * b as u128) % m as u128) as u64;
            }
            b = ((b as u128 * b as u128) % m as u128) as u64;
            e >>= 1;
        }
        acc
    }

    // Deterministic Miller–Rabin: the first 12 primes are a proven witness
    // set for all n < 3.317·10^24.
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if n % p == 0 {
                return n == p;
            }
        }
        let mut d = n - 1;
        let mut r = 0u32;
        while d % 2 == 0 {
            d /= 2;
            r += 1;
        }
        'outer: for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let mut x = t_pow(a, d, n);
            if x == 1 || x == n - 1 {
                continue;
            }
            for _ in 0..r - 1 {
                x = ((x as u128 * x as u128) % n as u128) as u64;
                if x == n - 1 {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    fn rand_poly(s: &mut u64) -> Poly {
        let mut a = [0u64; N];
        for c in a.iter_mut() {
            *c = splitmix64(s) % Q;
        }
        Poly::new(a)
    }

    fn rand_vec(s: &mut u64) -> Vec8 {
        let mut v = [Poly::zero(); 8];
        for p in v.iter_mut() {
            *p = rand_poly(s);
        }
        Vec8::new(v)
    }

    fn rand_mat(s: &mut u64) -> Mat8x8 {
        let mut m = [[Poly::zero(); 8]; 8];
        for row in m.iter_mut() {
            for p in row.iter_mut() {
                *p = rand_poly(s);
            }
        }
        Mat8x8::new(m)
    }


    fn vec_trim(a: &[u64; N]) -> Vec<u64> {
    let mut v = a.to_vec();
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

fn poly_trim(mut v: Vec<u64>) -> Vec<u64> {
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

/// Schoolbook product over Z_q (low-degree internal use: eeuclid only).
fn poly_mul(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            out[i + j] = add(out[i + j], const_mul(x, y));
        }
    }
    poly_trim(out)
}

fn poly_sub(a: &[u64], b: &[u64]) -> Vec<u64> {
    let len = a.len().max(b.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        out.push(sub(x, y));
    }
    poly_trim(out)
}

/// Division with remainder over Z_q[x]; `b` nonzero (trimmed). Returns
/// `None` only if b's leading coefficient is 0 (degenerate trimmed input).
fn poly_divmod(a: &[u64], b: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    if b.is_empty() {
        return None;
    }
    let db = b.len() - 1;
    let inv_lead = const_pow(b[db], Q - 2);
    let mut r = poly_trim(a.to_vec());
    if r.len() < b.len() {
        return Some((Vec::new(), r));
    }
    let mut q = vec![0u64; r.len() - db];
    while r.len() > db {
        let dr = r.len() - 1;
        let factor = const_mul(r[dr], inv_lead);
        q[dr - db] = factor;
        for i in 0..=db {
            r[dr - db + i] = sub(r[dr - db + i], const_mul(factor, b[i]));
        }
        r = poly_trim(r);
    }
    Some((q, r))
}


    #[test]
    fn q_is_prime_and_well_shaped() {
        assert!(is_prime(Q));
        assert!(Q > (1 << 31) && Q < (1 << 32));
        assert_eq!((Q - 1) % 512, 0);
        // Proth witness semantics: a^((Q−1)/2) ≡ −1 (mod Q).
        assert_eq!(t_pow(PROTH_WITNESS, (Q - 1) / 2, Q), Q - 1);
    }

    #[test]
    fn generator_is_the_smallest_generator() {
        assert_eq!(t_pow(GENERATOR, Q - 1, Q), 1);
        for p in [2u64, 3, 5, 7, 13] {
            assert_ne!(t_pow(GENERATOR, (Q - 1) / p, Q), 1, "factor {p}");
        }
        let mut h = 2u64;
        while h < GENERATOR {
            let is_gen = [2u64, 3, 5, 7, 13].iter().all(|&p| t_pow(h, (Q - 1) / p, Q) != 1);
            assert!(!is_gen, "{h} is a smaller generator");
            h += 1;
        }
    }



    #[test]
    fn psi_has_order_exactly_512() {
        assert_eq!(t_pow(PSI, 512, Q), 1);
        assert_eq!(t_pow(PSI, 256, Q), Q - 1);
        assert_ne!(t_pow(PSI, 128, Q), 1);
        assert_eq!(t_pow(PSI, 511, Q) * PSI % Q, 1);
    }

    #[test]
    fn twiddle_tables_match_independent_powers() {
        for lvl in 0..8usize {
            let m = 1usize << lvl;
            for j in 0..m {
                let e = ((N * (2 * j + 1)) / (2 * m)) as u64;
                assert_eq!(FWD_TWIDDLES[lvl][j], t_pow(PSI, e, Q), "fwd lvl {lvl} j {j}");
                assert_eq!(INV_TWIDDLES[lvl][j], t_pow(PSI_INV, e, Q), "inv lvl {lvl} j {j}");
                // Inverse twiddles really invert.
                assert_eq!(
                    (FWD_TWIDDLES[lvl][j] as u128 * INV_TWIDDLES[lvl][j] as u128 % Q as u128)
                        as u64,
                    1
                );
            }
        }
    }

    #[test]
    fn bitrev_is_the_bit_reversal_permutation() {
        let mut idx = [0u64; N];
        for (i, v) in idx.iter_mut().enumerate() {
            *v = i as u64;
        }
        let orig = idx;
        bitrev(&mut idx);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[255], 255);
        assert_eq!(idx[1], 128);
        assert_eq!(idx[128], 1);
        assert_eq!(idx[3], 192);
        assert_eq!(idx[192], 3);
        assert_eq!(idx[5], 160);
        bitrev(&mut idx);
        assert_eq!(idx, orig); // involution
    }

    #[test]
    fn ntt_intt_roundtrip() {
        let mut s = 0x51EA_0001;
        for _ in 0..200 {
            let p = rand_poly(&mut s);
            assert_eq!(p.ntt().intt(), p);
        }
        assert_eq!(Poly::zero().ntt().intt(), Poly::zero());
    }

     #[test]
    fn ntt_is_linear() {
        let mut s = 0x51EA_0002;
        for _ in 0..20 {
            let a = rand_poly(&mut s);
            let b = rand_poly(&mut s);
            assert_eq!(a.add(&b).ntt(), a.ntt().add(&b.ntt()));
        }
    }

    #[test]
    fn multiplication_is_negacyclic() {
        let mut c = [0u64; N];
        c[1] = 1;
        let x = Poly::new(c);
        let mut c2 = [0u64; N];
        c2[2] = 1;
        let x2 = Poly::new(c2);
        assert_eq!(x.mul(&x), x2); // x·x = x²
        let mut c3 = [0u64; N];
        c3[255] = 1;
        let x255 = Poly::new(c3);
        let mut c4 = [0u64; N];
        c4[128] = 1;
        let x128 = Poly::new(c4);
        // x^255 · x = x^256 = −1
        let prod = x255.mul(&x);
        assert_eq!(prod.coefficient(0), Q - 1);
        assert!(prod.coefficients()[1..].iter().all(|&v| v == 0));
        // x^128 · x^128 = −1
        let prod2 = x128.mul(&x128);
        assert_eq!(prod2.coefficient(0), Q - 1);
        assert!(prod2.coefficients()[1..].iter().all(|&v| v == 0));
        // Identity element
        let mut c5 = [0u64; N];
        c5[0] = 1;
        let one = Poly::new(c5);
        let mut s = 0x51EA_0003;
        let p = rand_poly(&mut s);
        assert_eq!(p.mul(&one), p);
    }

    #[test]
    fn fast_paths_match_reference_twins() {
        let mut s = 0x51EA_0004;
        for _ in 0..60 {
            let a = rand_poly(&mut s);
            let b = rand_poly(&mut s);
            assert_eq!(a.mul(&b), a.mul_reference(&b));
        }
        for _ in 0..12 {
            let m = rand_mat(&mut s);
            let v = rand_vec(&mut s);
            let w = rand_vec(&mut s);
            assert_eq!(m.mul_vec(&v), m.mul_vec_reference(&v));
            assert_eq!(m.mul_vec_transpose(&w), m.mul_vec_transpose_reference(&w));
            assert_eq!(v.dot(&w), v.dot_reference(&w));
            // Mat2x8 built from two rows of m.
            let t = Mat2x8::new([*m.row(3), *m.row(5)]);
            assert_eq!(t.mul_vec(&w), t.mul_vec_reference(&w));
        }
    }

    #[test]
    fn mul_scalar_matches_constant_poly_product() {
        let mut s = 0x51EA_0005;
        for k in [1u64, 2, 1023, 8192, HALF_Q, Q - 1] {
            let p = rand_poly(&mut s);
            let mut c = [0u64; N];
            c[0] = k % Q;
            let kp = Poly::new(c);
            assert_eq!(p.mul_scalar(k), p.mul_reference(&kp));
        }
    }

    #[test]
    fn centerlift_roundtrips_canonical_representatives() {
        let mut s = 0x51EA_0006;
        for _ in 0..100 {
            let p = rand_poly(&mut s);
            assert_eq!(Poly::from_centered(&p.centerlift()), p);
        }
        // Boundary values.
        let mut c = [0u64; N];
        c[0] = HALF_Q;
        c[1] = HALF_Q + 1;
        c[2] = Q - 1;
        let p = Poly::new(c);
        let cl = p.centerlift();
        assert_eq!(cl[0], HALF_Q as i64);
        assert_eq!(cl[1], HALF_Q as i64 + 1 - Q as i64);
        assert_eq!(cl[2], -1);
        assert_eq!(Poly::from_centered(&cl), p);
        // from_centered is total at the extremes.
        let vals = [i64::MIN, -Q as i64, -1, 0, 1, Q as i64, i64::MAX, 0];
        let p2 = Poly::from_centered(&vals);
        let _ = p2.coefficient(0);
    }

    #[test]
    fn serialization_roundtrip_and_validation() {
        let mut s = 0x51EA_0007;
        for _ in 0..50 {
            let p = rand_poly(&mut s);
            let bytes = p.to_bytes();
            assert_eq!(bytes.len(), 1024);
            assert_eq!(Poly::from_bytes(&bytes).unwrap(), p);
        }
        let v = rand_vec(&mut s);
        let vb = v.to_bytes();
        assert_eq!(vb.len(), 8192);
        assert_eq!(Vec8::from_bytes(&vb).unwrap(), v);
        let two = Vec2::new([*v.poly(0), *v.poly(1)]);
        let tb = two.to_bytes();
        assert_eq!(tb.len(), 2048);
        assert_eq!(Vec2::from_bytes(&tb).unwrap(), two);

        // Rejections: wrong length, unreduced coefficient.
        let p = rand_poly(&mut s);
        let mut bad = p.to_bytes();
        bad.push(0);
        assert!(matches!(
            Poly::from_bytes(&bad),
            Err(SealError::BadLength { expected: 1024, .. })
        ));
        let mut bad2 = p.to_bytes();
        bad2[0..4].copy_from_slice(&u32::MAX.to_le_bytes()); // 2^32 − 1 ≥ Q
        assert!(matches!(
            Poly::from_bytes(&bad2),
            Err(SealError::UnreducedCoefficient { value })
                if value == u32::MAX as u64
        ));
        assert!(matches!(
            Vec8::from_bytes(&vb[..8191]),
            Err(SealError::BadLength { expected: 8192, .. })
        ));
    }

    proptest! {
        #[test]
        fn prop_ntt_roundtrip(seed in any::<u64>()) {
            let mut s = seed ^ 0x51EA;
            let p = rand_poly(&mut s);
            prop_assert_eq!(p.ntt().intt(), p);
        }

        #[test]
        fn prop_mul_matches_reference(seed in any::<u64>()) {
            let mut s = seed ^ 0x51EB;
            let a = rand_poly(&mut s);
            let b = rand_poly(&mut s);
            prop_assert_eq!(a.mul(&b), a.mul_reference(&b));
        }

        #[test]
        fn prop_dot_and_mat_vec_match_reference(seed in any::<u64>()) {
            let mut s = seed ^ 0x51EC;
            let m = rand_mat(&mut s);
            let v = rand_vec(&mut s);
            let w = rand_vec(&mut s);
            prop_assert_eq!(m.mul_vec(&v), m.mul_vec_reference(&v));
            prop_assert_eq!(m.mul_vec_transpose(&w), m.mul_vec_transpose_reference(&w));
            prop_assert_eq!(v.dot(&w), v.dot_reference(&w));
            let t = Mat2x8::new([*m.row(1), *m.row(6)]);
            prop_assert_eq!(t.mul_vec(&v), t.mul_vec_reference(&v));
        }

        #[test]
        fn prop_centerlift_and_wire_roundtrip(seed in any::<u64>()) {
            let mut s = seed ^ 0x51ED;
            let p = rand_poly(&mut s);
            prop_assert_eq!(Poly::from_centered(&p.centerlift()), p);
            prop_assert_eq!(Poly::from_bytes(&p.to_bytes()).unwrap(), p);
        }
    }
    fn const_poly(v: u64) -> Poly {
        let mut a = [0u64; N];
        a[0] = v;
        Poly::new(a)
    }

    #[test]
    fn try_invert_known_values() {
        let x_inv = Poly::zero().sub(&Poly::monomial(255));
        assert_eq!(Poly::monomial(1).try_invert(), Some(x_inv));
        assert_eq!(Poly::monomial(5).try_invert(), Some(Poly::zero().sub(&Poly::monomial(251))));
        assert_eq!(Poly::one().try_invert(), Some(Poly::one()));
        assert_eq!(Poly::zero().try_invert(), None);
        // PSI (a primitive 512-th root of unity) inverts to PSI_INV.
        let psi = const_poly(PSI);
        assert_eq!(psi.try_invert(), Some(const_poly(PSI_INV)));
        assert_eq!(psi.mul(&const_poly(PSI_INV)), Poly::one());
        // x − PSI shares the root PSI with x^N + 1 → not invertible.
        let mut coeffs = [0u64; N];
        coeffs[1] = 1;
        coeffs[0] = sub(0, PSI);
        let bad = Poly::new(coeffs);
        assert!(bad.try_invert().is_none());
        // Constant polynomials invert via Fermat.
        assert_eq!(const_poly(7).try_invert(), Some(const_poly(const_pow(7, Q - 2))));
    }

    proptest! {
        #[test]
        fn prop_invert_roundtrip(seed in any::<u64>()) {
            let mut s = seed ^ 0x3C1A;
            let p = rand_poly(&mut s);
            if let Some(inv) = p.try_invert() {
                prop_assert_eq!(p.mul(&inv), Poly::one());
                prop_assert_eq!(inv.mul(&p), Poly::one());
            }
        }
    }

}

