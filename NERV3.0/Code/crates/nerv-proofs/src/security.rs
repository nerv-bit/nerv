//! D.2's FRI-security gate (erratum 86; native model — the Option-B
//! excision of plonky3, register 57/58): builds the conjectured-security
//! estimate from the AIR's measured shape (`sym::measure`), searches the
//! minimal FRI configuration meeting the floor, and reports the PQ posture
//! per the WP §4.2 accounting.
//!
//! MODEL (conjectured regime — the industry-deployment posture). Every term
//! is the standard bound for its round; the estimate is their min, capped
//! by the commitment's collision resistance:
//!   * α-fold round          error  n / |EF|      n = constraint families
//!   * DEEP-ALI round        error  d·N / |EF|    d = max constraint degree,
//!                                                 N = 2^degree_bits
//!   * batched-opening bind  error  w / |EF|      w = combined openings
//!   * FRI query phase       error  2^(−b·q/2)    Johnson-bound list-decoding
//!                          + grinding bits      conjecture for Reed–Solomon
//!                                                at rate 2^−b (b = log blowup)
//! b = 0 contributes grinding only: rate-1 codewords carry no distance —
//! grinding is that round's only lever.
//!
//! RETIRED with the excision (register 58): the upstream p3-security model,
//! its six regression vectors, and the proven-regime companion figure.
//! Selecting a proven-bound reference implementation (2024/1553-class) is
//! an M1 audit deliverable if the posture requires the companion number.
//! The D.2 POLICY is unchanged: conjectured ≥ 100 bits at classical
//! CR = 128; PQ posture = min(algebraic, 85) — BHT on 256-bit BLAKE3
//! digests (WP §4.2's own assessment).

/// D.2's floor, erratum 86's reading: conjectured ≥ 100 at classical CR.
pub const D2_ALGEBRAIC_FLOOR_BITS: usize = 100;
/// BHT on 256-bit digests (memory-caveated; WP §4.2's own assessment).
pub const PQ_COLLISION_BITS: usize = 85;
/// blake3-256 Merkle commitments, classical birthday bound.
pub const CLASSICAL_COLLISION_BITS: usize = 128;

/// The FRI configuration record (mirrors the frozen `[proofs.fri]` block of
/// specs/params.toml). The native engine folds binary (arity 2);
/// `max_log_arity` is carried for wire-record completeness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FriShape {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub log_final_poly_len: usize,
    pub max_log_arity: usize,
    pub commit_pow_bits: usize,
    pub query_pow_bits: usize,
}

impl FriShape {
    /// Binary folds: log arity 1.
    pub const DEFAULT_ARITY: usize = 1;
}

/// The AIR-side shape, from `sym::measure` — never hand-copied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AirShape {
    pub num_constraints: usize,
    pub max_constraint_degree: usize,
    /// Local/next access: 2 for every NERV AIR.
    pub max_combo: usize,
}

impl AirShape {
    pub const NERV_MAX_DEGREE: usize = 3;
    pub const NERV_MAX_COMBO: usize = 2;

    pub const fn measured(num_constraints: usize) -> AirShape {
        AirShape {
            num_constraints,
            max_constraint_degree: AirShape::NERV_MAX_DEGREE,
            max_combo: AirShape::NERV_MAX_COMBO,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    pub air: AirShape,
    /// Challenge-field bit length (the extension field the proofs run over).
    pub modulus_bits: usize,
    pub collision_resistance: usize,
    /// Committed codewords random-linear-combined into the FRI instance
    /// (main width + preprocessed width + quotient chunks).
    pub num_batched_functions: usize,
}

impl Profile {
    /// The wallet-tier default: 128-bit challenge field (the quadratic
    /// extension of Goldilocks), blake3-256 commitments, explicit batching.
    pub const fn wallet(air: AirShape, num_batched_functions: usize) -> Profile {
        Profile {
            air,
            modulus_bits: 128,
            collision_resistance: CLASSICAL_COLLISION_BITS,
            num_batched_functions,
        }
    }
}

fn ceil_log2(x: usize) -> u32 {
    if x <= 1 {
        0
    } else {
        x.next_power_of_two().ilog2()
    }
}

/// 2^degree_bits as u64 (saturated past 2^64 − 1; traces are ≤ 2^32).
fn trace_len(degree_bits: usize) -> u64 {
    if degree_bits >= 64 {
        u64::MAX
    } else {
        1u64 << degree_bits
    }
}

/// The FRI query-phase term in bits: (b·q)/2 + grinding.
fn fri_query_bits(fri: &FriShape) -> u64 {
    let bq = (fri.log_blowup as u64).saturating_mul(fri.num_queries as u64);
    bq / 2 + (fri.commit_pow_bits + fri.query_pow_bits) as u64
}

/// Conjectured security in bits: the min over every round's standard bound.
pub fn conjectured(profile: &Profile, fri: &FriShape, degree_bits: usize) -> usize {
    let m = profile.modulus_bits as u64;
    let alpha = m.saturating_sub(u64::from(ceil_log2(profile.air.num_constraints)));
    let ali = m.saturating_sub(u64::from(ceil_log2(
        ((profile.air.max_constraint_degree as u64)
            .saturating_mul(trace_len(degree_bits))
            .max(1)
            .min(usize::MAX as u64)) as usize,
    )));
    let batch = m.saturating_sub(u64::from(ceil_log2(profile.num_batched_functions)));
    let bits = (profile.collision_resistance as u64)
        .min(alpha)
        .min(ali)
        .min(batch)
        .min(fri_query_bits(fri));
    bits.min(usize::MAX as u64) as usize
}

/// Search space: blowup 1..=8 (the chunk heuristic keeps blowup ≥ degree−1
/// so the quotient lands in a single chunk), queries 1..=512, grinding
/// {0, 16} (grinding cannot repair the field terms — a miss is reported,
/// not patched). Size proxy minimized: queries × (degree_bits + blowup)
/// + query_pow (each query opens a Merkle path of ~degree_bits + blowup
/// hashes).
pub fn search_min_fri(
    profile: &Profile,
    degree_bits: usize,
    floor: usize,
) -> Option<FriShape> {
    let mut best: Option<(u64, FriShape)> = None;
    for log_blowup in 1..=8usize {
        if profile.air.max_constraint_degree > (1 << log_blowup) + 1 {
            continue;
        }
        for query_pow in [0usize, 16] {
            for num_queries in 1..=512usize {
                let fri = FriShape {
                    log_blowup,
                    num_queries,
                    log_final_poly_len: 0,
                    max_log_arity: FriShape::DEFAULT_ARITY,
                    commit_pow_bits: 0,
                    query_pow_bits: query_pow,
                };
                if conjectured(profile, &fri, degree_bits) >= floor {
                    let proxy = num_queries as u64 * (degree_bits as u64 + log_blowup as u64)
                        + query_pow as u64;
                    if best.as_ref().is_none_or(|(p, _)| proxy < *p) {
                        best = Some((proxy, fri));
                    }
                    break;
                }
            }
        }
    }
    best.map(|(_, f)| f)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateReport {
    pub fri: FriShape,
    pub conjectured_classical: usize,
    /// min(algebraic, PQ collision) — erratum 86's posture.
    pub pq_posture: usize,
}

impl GateReport {
    pub fn meets_floor(&self) -> bool {
        self.conjectured_classical >= D2_ALGEBRAIC_FLOOR_BITS
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SecurityGateError {
    #[error("no FRI configuration meets {floor} bits at degree_bits={degree_bits} over a {modulus_bits}-bit field — widen the challenge field (grinding cannot repair the field terms)")]
    FloorUnreachable { floor: usize, degree_bits: usize, modulus_bits: usize },
    #[error("configuration conjectured {got} < floor {floor}")]
    BelowFloor { got: usize, floor: usize },
}

/// Validates a GIVEN configuration (the CI form: the frozen params.toml
/// FRI block is checked, not searched).
pub fn validate(
    profile: &Profile,
    fri: &FriShape,
    degree_bits: usize,
) -> Result<GateReport, SecurityGateError> {
    let c = conjectured(profile, fri, degree_bits);
    let report = GateReport {
        fri: *fri,
        conjectured_classical: c,
        pq_posture: c.min(PQ_COLLISION_BITS),
    };
    if report.meets_floor() {
        Ok(report)
    } else {
        Err(SecurityGateError::BelowFloor { got: c, floor: D2_ALGEBRAIC_FLOOR_BITS })
    }
}

/// D.2's gate: searches the minimal compliant configuration for a trace
/// size. The frozen result lands in params.toml at M0; CI re-validates it.
pub fn gate(
    profile: &Profile,
    degree_bits: usize,
) -> Result<GateReport, SecurityGateError> {
    let fri = search_min_fri(profile, degree_bits, D2_ALGEBRAIC_FLOOR_BITS).ok_or(
        SecurityGateError::FloorUnreachable {
            floor: D2_ALGEBRAIC_FLOOR_BITS,
            degree_bits,
            modulus_bits: profile.modulus_bits,
        },
    )?;
    validate(profile, &fri, degree_bits)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

 /// The seal chip's measured scale (erratum 83): ~2^20 families;
    /// batching ≈ 2W+1 combined openings at the seal chip's width.
    fn seal_scale_profile() -> Profile {
        Profile::wallet(
            AirShape {
                num_constraints: 1 << 20,
                max_constraint_degree: 5,
                max_combo: AirShape::NERV_MAX_COMBO,
            },
            1 << 16,
        )
    }


    fn shape(log_blowup: usize, num_queries: usize, query_pow: usize) -> FriShape {
        FriShape {
            log_blowup,
            num_queries,
            log_final_poly_len: 0,
            max_log_arity: FriShape::DEFAULT_ARITY,
            commit_pow_bits: 0,
            query_pow_bits: query_pow,
        }
    }

    #[test]
    fn terms_are_the_documented_min() {
        let p = seal_scale_profile();
      // FRI generous (b=8, q=512 → 2048 bits) → the field terms bind:
        // min(cr=128, α=108, ali=105, batch=112).
        assert_eq!(conjectured(&p, &shape(8, 512, 0), 20), 105);
        // Taller traces weaken the DEEP-ALI term: 128 − ceil_log2(5·2^24) = 102.
        assert_eq!(conjectured(&p, &shape(8, 512, 0), 24), 102);

    }

    #[test]
    fn zero_blowup_collapses_to_grinding() {
        // Rate-1 codewords carry no distance: only grinding remains.
        let p = seal_scale_profile();
        assert_eq!(conjectured(&p, &shape(0, 100, 16), 20), 16);
        assert_eq!(conjectured(&p, &shape(0, 512, 0), 20), 0);
    }

    #[test]
    fn search_meets_floor_at_wallet_tier() {
        let p = seal_scale_profile();
        for degree_bits in [16usize, 20, 24] {
            let fri = search_min_fri(&p, degree_bits, D2_ALGEBRAIC_FLOOR_BITS)
                .unwrap_or_else(|| panic!("no config at degree_bits={degree_bits}"));
            let c = conjectured(&p, &fri, degree_bits);
            assert!(c >= D2_ALGEBRAIC_FLOOR_BITS, "degree {degree_bits}: {c}");
            // Minimality at the found grinding level: one fewer query misses.
            let mut fewer = fri;
            fewer.num_queries -= 1;
            assert!(
                conjectured(&p, &fewer, degree_bits) < D2_ALGEBRAIC_FLOOR_BITS,
                "degree {degree_bits}: not minimal"
            );
            // Chunk heuristic respected.
            assert!(p.air.max_constraint_degree <= (1 << fri.log_blowup) + 1);
        }
    }

    #[test]
    fn gate_report_and_pq_posture() {
        let p = seal_scale_profile();
        let r = gate(&p, 20).unwrap();
        assert!(r.meets_floor());
        assert_eq!(r.pq_posture, r.conjectured_classical.min(PQ_COLLISION_BITS));
        assert!(r.pq_posture <= PQ_COLLISION_BITS);
    }

    #[test]
    fn too_small_field_is_unreachable_not_ground() {
        let p = Profile {
            air: AirShape::measured(1 << 12),
            modulus_bits: 64,
            collision_resistance: CLASSICAL_COLLISION_BITS,
            num_batched_functions: 8,
        };
        assert!(search_min_fri(&p, 20, D2_ALGEBRAIC_FLOOR_BITS).is_none());
        assert!(matches!(
            gate(&p, 20),
            Err(SecurityGateError::FloorUnreachable { modulus_bits: 64, .. })
        ));
    }

    #[test]
    fn validate_rejects_weak_config() {
        let p = seal_scale_profile();
        assert!(matches!(
            validate(&p, &shape(4, 2, 0), 20),
            Err(SecurityGateError::BelowFloor { .. })
        ));
    }

    #[test]
    fn monotonicity_spot_checks() {
        let p = seal_scale_profile();
        let base = shape(4, 40, 0);
        let mut more = base;
        more.num_queries = 80;
        assert!(conjectured(&p, &more, 20) >= conjectured(&p, &base, 20));
        let mut ground = base;
        ground.query_pow_bits = 16;
        assert!(conjectured(&p, &ground, 20) >= conjectured(&p, &base, 20));
        // Taller traces never grade above shorter ones.
        assert!(conjectured(&p, &base, 24) <= conjectured(&p, &base, 20));
        // More combined openings never help.
        let mut wide = p;
        wide.num_batched_functions = 1 << 20;
        assert!(conjectured(&wide, &base, 20) <= conjectured(&p, &base, 20));
    }
}

