//! Deterministic Miller–Rabin for u64 — the spec validator's authority on
//! the seal modulus q. The parameter is *proven* prime at every load; no
//! constant is trusted, including the one in this repository.

/// Deterministic for all u64: the first 12 primes suffice below 3.3e24.
const WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn powmod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut acc: u64 = 1 % m;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mulmod(acc, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    acc
}

/// Exact primality test for n < 2^64.
pub fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in &WITNESSES {
        if n == p {
            return true;
        }
        if n % p == 0 {
            return false;
        }
    }
    // write n-1 = d * 2^r with d odd
    let mut d = n - 1;
    let mut r: u32 = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    'witness: for &a in &WITNESSES {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r.saturating_sub(1) {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

/// The exponent of 2 in the factorization of n-1 (n > 1), i.e. the largest
/// k with 2^k | n-1. For an NTT over Z_q[x]/(x^d+1), 2^(k) must be >= 2d.
pub fn two_adicity(n: u64) -> u32 {
    let mut k: u32 = 0;
    let mut m = n - 1;
    while m % 2 == 0 {
        m /= 2;
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_primes_and_composites() {
        for p in [2u64, 3, 5, 65537, 2147483647, 4294967291] {
            assert!(is_prime_u64(p), "{p} should be prime");
        }
        for c in [0u64, 1, 4, 9, 561, 1105, 4294967295, 4293918720, 4294966785] {
            assert!(!is_prime_u64(c), "{c} should be composite");
        }
    }

    #[test]
    fn seal_modulus_properties() {
        let q = 4293918721u64;
        assert!(is_prime_u64(q), "the seal modulus must be prime (validated, not trusted)");
        assert_eq!(two_adicity(q), 20, "512 | q-1 required for degree-256 NTT");
    }
}
