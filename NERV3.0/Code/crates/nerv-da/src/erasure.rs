//! GF(2⁸) systematic Reed–Solomon (erratum 139): Cauchy parity, any k of
//! n reconstructs. All arithmetic is table-based and exact.

use crate::error::ErasureError;

const POLY: u16 = 0x11d; // x^8 + x^4 + x^3 + x^2 + 1


const fn build_tables() -> ([u8; 256], [u8; 510]) {
    let mut log = [0u8; 256];
    let mut exp = [0u8; 510];
    let mut x: u16 = 1;
    let mut i = 0usize;
    while i < 510 {
        if i < 255 {
            log[x as usize] = i as u8;
        }
        exp[i] = x as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= POLY;
        }
        i += 1;
    }
    (log, exp)
}


const TABLES: ([u8; 256], [u8; 510]) = build_tables();


#[inline]
pub fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let (log, exp) = TABLES;
    exp[(log[a as usize] as usize) + (log[b as usize] as usize)]
}


#[inline]
pub fn gf_inv(a: u8) -> u8 {
    debug_assert!(a != 0);
    let (log, exp) = TABLES;
    exp[255 - (log[a as usize] as usize)]
}


/// The generator's row `i` over k data coordinates: identity rows 0..k,
/// Cauchy rows k..n (erratum 139).
pub fn generator_row(i: usize, k: usize, n: usize) -> Vec<u8> {
    debug_assert!(i < n && k < n && n <= 256);
    if i < k {
        let mut r = vec![0u8; k];
        r[i] = 1;
        r
    } else {
        let a = (i - k) as u8;
        (0..k).map(|j| gf_inv(a ^ ((n - k + j) as u8))).collect()
    }
}


/// Parity shards for `data` (all equal length).
pub fn encode_parity(data: &[Vec<u8>], parity_count: usize) -> Result<Vec<Vec<u8>>, ErasureError> {
    let k = data.len();
    if k == 0 {
        return Err(ErasureError::NoData);
    }
    if parity_count == 0 {
        return Err(ErasureError::NoParity);
    }
    let n = k + parity_count;
    if n > 256 {
        return Err(ErasureError::TooManyShards { n });
    }
    let len = data[0].len();
    for d in data.iter().skip(1) {
        if d.len() != len {
            return Err(ErasureError::LengthMismatch { a: len, b: d.len() });
        }
    }
    let mut parity = vec![vec![0u8; len]; parity_count];
    let (log, exp) = TABLES;
    for a in 0..parity_count {
        for j in 0..k {
            let c = gf_inv(a as u8 ^ ((n - k + j) as u8));
            let lc = log[c as usize] as usize;
            let dst = &mut parity[a];
            for (d, &s) in dst.iter_mut().zip(data[j].iter()) {
                if s != 0 {
                    *d ^= exp[(log[s as usize] as usize) + lc];
                }
            }
        }
    }
    Ok(parity)
}


fn invert(m: &[Vec<u8>]) -> Option<Vec<Vec<u8>>> {
    let k = m.len();
    let mut a: Vec<Vec<u8>> = (0..k)
        .map(|i| {
            let mut r = m[i].clone();
            r.extend((0..k).map(|j| u8::from(i == j)));
            r
        })
        .collect();
    for col in 0..k {
        let piv = (col..k).find(|&r| a[r][col] != 0)?;
        a.swap(col, piv);
        let ip = gf_inv(a[col][col]);
        for x in a[col].iter_mut().skip(col) {
            if *x != 0 {
                *x = gf_mul(*x, ip);
            }
        }
        for r in 0..k {
            if r != col && a[r][col] != 0 {
                let f = a[r][col];
                for x in col..2 * k {
                    if a[col][x] != 0 {
                        a[r][x] ^= gf_mul(f, a[col][x]);
                    }
                }
            }
        }
    }
    Some((0..k).map(|i| a[i][k..].to_vec()).collect())
}


fn decode_data(
    shards: &[(usize, &[u8])],
    k: usize,
    n: usize,
) -> Result<Vec<Vec<u8>>, ErasureError> {
    if shards.len() != k {
        return Err(ErasureError::ShardCount { expected: k, found: shards.len() });
    }
    let len = shards[0].1.len();
    let mut idx = shards.iter().map(|(i, _)| *i).collect::<Vec<_>>();
    idx.sort_unstable();
    idx.dedup();
    if idx.len() != k {
        return Err(ErasureError::DuplicateIndex);
    }
    for &(i, s) in shards {
        if i >= n {
            return Err(ErasureError::IndexRange { index: i, n });
        }
        if s.len() != len {
            return Err(ErasureError::LengthMismatch { a: len, b: s.len() });
        }
    }
    let m: Vec<Vec<u8>> = shards.iter().map(|(i, _)| generator_row(*i, k, n)).collect();
    let minv = invert(&m).ok_or(ErasureError::Singular)?;
    let mut out = vec![vec![0u8; len]; k];
    for (j, o) in out.iter_mut().enumerate() {
        for t in 0..k {
            let c = minv[j][t];
            if c != 0 {
                for (d, &s) in o.iter_mut().zip(shards[t].1.iter()) {
                    if s != 0 {
                        *d ^= gf_mul(c, s);
                    }
                }
            }
        }
    }
    Ok(out)
}


/// The full n-shard codeword from any k of them (data ‖ parity).
pub fn full_codeword(
    shards: &[(usize, &[u8])],
    k: usize,
    n: usize,
) -> Result<Vec<Vec<u8>>, ErasureError> {
    let data = decode_data(shards, k, n)?;
    let parity = encode_parity(&data, n - k)?;
    Ok(data.into_iter().chain(parity).collect())
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::testutil::SplitMix64;
    use proptest::prelude::*;


    /// Independent GF multiply: Russian-peasant carry-less.
    fn peasant_mul(mut a: u8, mut b: u8) -> u8 {
        let mut acc = 0u8;
        while b != 0 {
            if b & 1 != 0 {
                acc ^= a;
            }
            let hi = a & 0x80;
            a <<= 1;
            if hi != 0 {
                a ^= POLY as u8;
            }
            b >>= 1;
        }
        acc
    }


    #[test]
    fn gf_arithmetic_differential() {
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(1, 7), 7);
        for a in 1u8..=255 {
            assert_eq!(gf_mul(a, gf_inv(a)), 1, "inv {a}");
            assert_eq!(gf_mul(a, 1), a);
        }
        let mut rng = SplitMix64::new(0xE1);
        for _ in 0..20_000 {
            let a = (rng.next_u64() & 0xFF) as u8;
            let b = (rng.next_u64() & 0xFF) as u8;
            assert_eq!(gf_mul(a, b), peasant_mul(a, b), "{a}×{b}");
            assert_eq!(gf_mul(a, b), gf_mul(b, a));
            if a != 0 {
                assert_eq!(gf_mul(gf_mul(a, b), gf_inv(a)), b);
            }
        }
        // The polynomial is irreducible: x ↦ x·g generates the group.
        let mut seen = [false; 256];
        let mut x = 1u8;
        for _ in 0..255 {
            assert!(!seen[x as usize]);
            seen[x as usize] = true;
            x = gf_mul(x, 2);
        }
        assert_eq!(x, 1);
    }


    #[test]
    fn generator_is_mds_by_brute_force() {
        for (n, k) in [(6usize, 3usize), (8, 4), (5, 2), (4, 1), (10, 5)] {
            let rows: Vec<Vec<u8>> = (0..n).map(|i| generator_row(i, k, n)).collect();
            // Identity rows.
            for i in 0..k {
                for j in 0..k {
                    assert_eq!(rows[i][j], u8::from(i == j));
                }
            }
            // Every k-subset of rows inverts.
            let mut subset = vec![0usize; k];
            let mut ok = 0usize;
            let mut total = 0usize;
            let mut all: Vec<usize> = (0..n).collect();
            combine(&mut all, k, &mut subset, 0, &mut |sel| {
                let m: Vec<Vec<u8>> = sel.iter().map(|&i| rows[i].clone()).collect();
                assert!(invert(&m).is_some(), "singular subset {sel:?} (n={n},k={k})");
                ok += 1;
                total += 1;
            });
            assert!(ok > 0 && total == binom(n, k));
        }
    }


    fn binom(n: usize, k: usize) -> usize {
        if k == 0 {
            return 1;
        }
        (n - k + 1..=n).product::<usize>() / (1..=k).product::<usize>()
    }


    fn combine(pool: &[usize], k: usize, cur: &mut Vec<usize>, start: usize, f: &mut dyn FnMut(&[usize])) {
        if cur.len() == k {
            f(cur);
            return;
        }
        for i in start..pool.len() {
            cur.push(pool[i]);
            combine(pool, k, cur, i + 1, f);
            cur.pop();
        }
    }


    fn shards(seed: u64, k: usize, len: usize) -> Vec<Vec<u8>> {
        let mut rng = SplitMix64::new(seed);
        (0..k).map(|_| (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect()).collect()
    }


    #[test]
    fn roundtrip_every_erasure_pattern_small() {
        let k = 3;
        let len = 16;
        let data = shards(0xE2, k, len);
        let parity = encode_parity(&data, k).unwrap();
        let full: Vec<Vec<u8>> = data.iter().cloned().chain(parity).collect();
        let n = full.len();
        // Every k-subset of the 6 shards reconstructs the data.
        let mut sel = Vec::new();
        let all: Vec<usize> = (0..n).collect();
        combine(&all, k, &mut sel, 0, &mut |sel| {
            let chosen: Vec<(usize, &[u8])> =
                sel.iter().map(|&i| (i, full[i].as_slice())).collect();
            assert_eq!(decode_data(&chosen, k, n).unwrap(), data, "subset {sel:?}");
            let cw = full_codeword(&chosen, k, n).unwrap();
            assert_eq!(cw, full, "codeword {sel:?}");
        });
    }


    #[test]
    fn error_paths() {
        assert!(matches!(encode_parity(&[], 1), Err(ErasureError::NoData)));
        let d = shards(1, 2, 8);
        assert!(matches!(encode_parity(&d, 0), Err(ErasureError::NoParity)));
        let mut ragged = shards(2, 2, 8);
        ragged[1].push(0);
        assert!(matches!(
            encode_parity(&ragged, 1),
            Err(ErasureError::LengthMismatch { .. })
        ));
        let big: Vec<Vec<u8>> = (0..255).map(|_| vec![0u8; 4]).collect();
        assert!(matches!(
            encode_parity(&big, 2),
            Err(ErasureError::TooManyShards { n: 257 })
        ));
        let s: Vec<(usize, &[u8])> = vec![(0, &[1u8; 4][..])];
        assert!(matches!(
            decode_data(&s, 2, 4),
            Err(ErasureError::ShardCount { expected: 2, found: 1 })
        ));
        let dup: Vec<(usize, &[u8])> = vec![(0, &[1u8; 4][..]), (0, &[2u8; 4][..])];
        assert!(matches!(decode_data(&dup, 2, 4), Err(ErasureError::DuplicateIndex)));
        let oob: Vec<(usize, &[u8])> = vec![(0, &[1u8; 4][..]), (9, &[2u8; 4][..])];
        assert!(matches!(decode_data(&oob, 2, 4), Err(ErasureError::IndexRange { index: 9, .. })));
    }


    proptest! {
        #[test]
        fn prop_roundtrip_random_erasure(seed in any::<u64>(), k in 2usize..=6, len in 1usize..64) {
            let data = shards(seed, k, len);
            let parity = encode_parity(&data, k).unwrap();
            let full: Vec<Vec<u8>> = data.iter().cloned().chain(parity).collect();
            let n = full.len();
            let mut rng = SplitMix64::new(seed ^ 0xE3);
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..idx.len()).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                idx.swap(i, j);
            }
            idx.truncate(k);
            let chosen: Vec<(usize, &[u8])> =
                idx.iter().map(|&i| (i, full[i].as_slice())).collect();
            prop_assert_eq!(decode_data(&chosen, k, n).unwrap(), data);
            prop_assert_eq!(full_codeword(&chosen, k, n).unwrap(), full);
        }
    }
}
