NERV v3.0 — Implementation Errata Register
Status: living document; append-only. IDs are never reused; supersessions are marked in the entry and indexed at the foot of the file.
Policy
Append-only. Each chunk's delivery appends entries. An entry's figuresmay be amended by a later entry (marked); its text is never rewritten.
DSR-5 enforcement. A whitepaper normative statement contradicted by thecode without a covering register entry is a documentation bug that blocksmerge. The register is the audit trail that keeps document and code fromdiverging silently.
Consumption. The M0 spec review (§§3–6 reviewers) and the M1 dual-auditpacket (both firms) receive this file. Entries cited by Appendix E carry the→ App E.x tag; all others are instantiations within WP latitude.
Prior decision IDs. Chunks 1–5 recorded design decisions under E-0xxidentifiers in their delivery records (e.g. E-002 Poseidon2 tree hashing;E-007 beacon-seeded sortition per DSR-5). Those are decisions within WPlatitude, not errata, and are not duplicated here; they are cited where theyinteract. This register's plain-integer IDs (1–56) and Appendix E's sectionnumbers (E.1–E.12) are distinct namespaces.
Legend
[WP-ERRATUM] — the WP text is wrong as written; Appendix E amends it.
[FROZEN] — instantiation of a blank the WP left unspecified; the frozen rule.
[GENESIS-CONFIG] — a chosen constant; governance-adjustable.
[SIZE] — byte/cost budget reconciliation.
[DOMAIN] — D-02 domain-inventory addition.
[HYGIENE] — internal correction (test, doc, module geography); no protocol surface.

Chunk 6 — nerv-codec (entries 1–17)
1. [FROZEN] Slot reduction
slot(addr) = the full 256-bit little-endian digest value ofBLAKE3("nerv.slot" ‖ addr), reduced mod 224 via Hash256::reduce_mod(nerv-core's ToField convention). Originally instantiated on the low 64 bits;amended at the part-2 reconciliation with the delivered nerv-core hash surface.WP: §7.2 (formula; reduction unspecified).
2. [FROZEN] Log-magnitude rail
Σ ⌊log₂(1+v)⌋ over the leg's input and output values (the WP says "pervalue"; the aggregation is frozen here). WP: §7.2.
3. [FROZEN] Type rails
Five frozen kinds — SingleShard, CrossShardSpend, CrossShardIssue, Claim, Burn— one-hot over rails 228–232; 233–235 reserved-zero until governance activatesthem. WP: §7.2 ("transaction-type flags" unspecified).
4. [FROZEN] Time bucket
⌊16·(expiry mod E)/E⌋ from the leg's public expiry height — the only per-legpublic time coordinate. WP: §7.2 ("coarse time-of-epoch buckets" semanticsunspecified).
5. [WP-ERRATUM] Encoder sparsity bound = 12
The admissible support is ≤ 6 account slots + 6 rails = 12 active features;the encoder chip provisions 12 column selections. WP §7.2's "at most 11 of W's256 columns" is the typical-leg figure. → App E.10. WP: §7.2, §5.4.
6. [FROZEN] δ computation
Exact i128 accumulation over active features; a single round-half-even scalingpoint at 2⁻¹⁵ (nerv-core fixed_point's canonical rule); wrap into ℤ/2⁶⁴.Total on all inputs: a dense 256-term accumulator is bounded by 2⁸⁶ < 2¹²⁷.WP: §7.2.
7. [FROZEN] Feature range bounds
Every coordinate within ±(2⁶²−1); log rail ≤ 1024; count rail exactly 1;volume/fee ≤ 2⁶²−1. WP: §7.2 (bounds unspecified).
8. [FROZEN] Canonical encodings (codec)
Features 2,048 B; W 32,772 B (u64 version ‖ 16,384 × i16); delta 512 B — allfixed-width LE, no length prefixes. WP: —.
9. [FROZEN] W expansion
Xof::framed("nerv.w.gen", [beacon ‖ version]) with u32-LE length framing;16,384 weights read row-major as LE i16. WP: §7.6 (expansion unspecified).
10. [FROZEN] W commitment
Hash256::concat("nerv.w.commit", canonical_bytes) — plain-prefix mode, asingle fixed-width 32,772-byte message. Realizes §7.6's Hash(W ‖ version).WP: §7.6.
11. [FROZEN] Certification sample streams
Xof::framed("nerv.w.spark", [commitment ‖ family u32 ‖ k u32]); family 0 =spark k-subsets of the slot block, family 1 = column–rail 6-subsets; membersdrawn as successive next_u64() % n with distinct-rejection. WP: —.
12. [FROZEN] Certification gate order
Row ℓ2² band → column ℓ2² band → row-ℓ1 fleet band → column-ℓ1 fleet band →row distinctness → exhaustive 𝔽_p column-pair independence (all 32,640 pairs)→ full 𝔽_p rank 64 → sampled spark (k = 3..=13) → sampled column–rail(6 sampled slots ∪ each rail, rank 7). First failure returns. WP: App C.3(order unspecified).
13. [FROZEN] Fleet statistics
Exact-integer lower median and MAD; band |v − median| ≤ 9·MAD. Population-σbands were rejected on first principles: coordinated outliers self-dilute σ(a single outlier self-limits at √N·σ); median/MAD are immune (P8). WP: —.
14. [GENESIS-CONFIG] Certification constants
ROW_L2SQ ∈ [2³⁰, 256·2³⁰]; COL_L2SQ ∈ [2²⁴, 64·2³⁰]; FLEET_BAND_MADS = 9;SPARK_REQUIREMENT = 13. WP: App C.3, App B.
15. [FROZEN] Certificate record
137-byte canonical LE layout (field order frozen in weight_gen.rs);verify() re-runs the battery with the recorded sampling configuration andcompares bytes. WP: App C.3 (record unspecified).
16. [WP-ERRATUM] Spark certification posture
No cheap sound certificate for spark ≥ 13 exists: exact spark is NP-hard, andthe coherence bound spark ≥ 1 + 1/μ is vacuous at 64×256 (random matrices haveμ ≈ 0.5, certifying only spark ≥ 3). The implemented certificate is: soundexhaustive checks (spark ≥ 3 exactly, rank 64, no proportional/duplicate/deadcolumns, norm health, all pairs independent over 𝔽_p — subsuming exactℚ-proportionality of integer columns) + verifiable provenance (a fixed64×k, k ≤ 13 submatrix of the XOF expansion is 𝔽_p-dependent with probability≤ k·p⁻⁵²; union over all ≤ 12-subsets of the 224 slot columns ≈ 2⁻³²³⁰) +deterministic sampled rank checks. → App E.9. WP: App C.3, §7.4.
17. [DOMAIN] Codec domains
nerv.w.commit, nerv.w.spark (EXT-v1) join the WP-listed nerv.w.gen.WP: D-02.

Chunk 7 — nerv-seal I: ring, sampling, digitize (entries 18–29)
18. [FROZEN] Seal modulus
q = 3·2³⁰ + 1 = 3,221,225,473 — a 32-bit Proth prime, q ≡ 1 mod 512.Primality is proven at compile time via Proth's theorem (a const-evaluatedwitness search; a composite q fails the build) and re-verified at runtime bydeterministic Miller–Rabin. Documented replacement procedure: another Prothprime ≡ 1 mod 512 with q > 2.15·10⁹ (e.g. 11·2²⁸ + 1); update params, ring::Q,and the lib.rs pins. WP: §6.3.1.
19. [WP-ERRATUM] Scale/chunk closure v1 — SUPERSEDED (→ 30 → 40)
The WP's chunk ≤ 512 with 10-bit digits cannot close at 32-bit q (adversarialdigit sums force scale ≤ 2¹¹·⁶; aggregate noise forces ≥ 2²¹). First closure:scale 2¹³, chunk 128 — overflow side only; the statistical claim was wrong(→ 30). Retained for the record. → App E.1 (chain head). WP: §6.3.8.
20. [WP-ERRATUM] T-structure reconciliation
"T = A·s + e₀ ∈ R^{2×8}" with A ∈ R^{8×8}, s ∈ R⁸ is dimensionallyinconsistent (A·s ∈ R⁸). Realized: T = s·A + E₀ ∈ R^{2×8}(T_ik = Σ_j s_j·A_jk), both rows sharing the s·A base with independent shortnoise rows — preserving every WP dimension, s ∈ R⁸ (rank 8, dimension 2,048),and §6.3.3's single partial p_j serving both components. → App E.2. WP:§6.3.1.
21. [SIZE] Ring wire format
Coefficients u32-LE: Poly 1,024 B; u (Vec8) 8,192 B; v (Vec2) 2,048 B; ct10,240 B — the "≈ 8–10 KB, u dominates" genesis format. A is never serialized:XOF-expanded from a 32-byte seed; the public-key wire object is (seed, T).WP: §6.3.8.
22. [DOMAIN] Seal sampling domains
nerv.seal (A expansion from the epoch seed — D-02's "seal key derivation");nerv.seal.noise (EXT-v1; wallet triplet in frozen order r ‖ e₁ ‖ e₂).WP: D-02.
23. [FROZEN] CBD(2) layout
128 bytes per polynomial; coefficient j consumes nibble j (low nibble of bytej/2 when j even); value b₀+b₁−b₂−b₃, nibble bits LSB-first. η = 2 ⇒ support[−2, 2], σ = 1. WP: §6.3.8 (layout unspecified).
24. [FROZEN] 3σ rejection honesty
At η = 2 the ±3 rejection is a verified no-op (support ⊂ bound); the machineryexists as statement-10's bound for parameter evolution. Attempt caps (1024)keep every sampler total; uniform per-matrix cap-exhaust < 2⁻²⁰⁰⁰. WP:§6.3.7.
25. [FROZEN] Uniform expansion
Row-major (row, column, coefficient); u32-LE draws; accept < q — exactuniformity, ≈ 1.33 draws per coefficient. WP: —.
26. [FROZEN] Digit slot layout (figures as amended by 33/40)
slot(j, k) = 64k + j (digit-plane); ring 0 = planes 0–3, ring 1 = planes 4–7;all 512 slots digit-bearing; Plaintext wire 1,024 B (u16-LE, slot order).WP: §6.3.1 (layout unspecified).
27. [WP-ERRATUM] Guard-digit semantics — SUPERSEDED (→ 40)
10-bit-era rule (digit 6 = bits 60–63, ≤ 15) deleted at erratum 40: 8 positions× 8 bits = 64 exactly; position 7 is a full digit; its carry-out is themod-2⁶⁴ wrap. WP: §6.3.1.
28. [FROZEN] resolve semantics (exponent as amended by 40)
Total u128 evaluation of Σ_k D_k·2^{8k}, low 64 bits kept — the exact integeridentity behind public carry resolution; differentially tested against anindependent explicit carry-propagation reference. WP: §6.3.2.
29. [FROZEN] The nerv-codec seam
digitize/resolve operate on &[u64; 64]; nerv-seal does not depend onnerv-codec (dependency policy); interop via &delta.0 / Delta(coords); the64 = EMBEDDING_DIM equality is pinned by the conformance crate. WP: —.

Chunk 8 — nerv-seal II: encrypt, decrypt, noise (entries 30–39)
30. [WP-ERRATUM] Digitization v2 — SUPERSEDED (→ 40)
Erratum 19's statistical claim was wrong: the honest per-component noise variance is 2·(rank 8)·(n 256) + 1 = 4,097 (σ ≈ 64 per leg; chunk σ ≈ 724) —erratum 19 had undercounted the module rank. At 10-bit digits the margin collapses to 5.66σ ≈ tens of silent wrong-digit reveals per network-day.Fix (encoding-only, at the time): 9-bit digits, 8 positions, scale 2¹⁴ →11.31σ. Superseded: the dealerless-key variance re-closed the joint (→ 40).→ App E.1 (chain). WP: §6.3.8.
31. [FROZEN] Budget ownership
noise.rs owns SCALE, the exact integer variance model, the provableworst-case bounds, the compile-time closure, and the margin floors. Thekey-side bound contract (KEY_SHORT_BOUND, later KEY_BOUND per 41) is the DKG'sproof obligation — the budget closes against exactly what the proofs establish.WP: §6.3.7.
32. [HYGIENE] Error-surface change — MOOT at 40
UnusedSlotNonZero removed when all 512 slots became digit-bearing; theinvalid-reveal surface moved to decode-time digit-sum validation. WP: —.
33. [SIZE] Public-key wire
ASeed (32 B) ‖ T (2 × 8 polys) = 16,416 B genesis. WP §6.3.8's "≈ 8 KB pershard per epoch" corresponds to a compressed (R2-class) or single-row format,not genesis. → App E.3. WP: §6.3.8.
34. [FROZEN] Decode rule (constants as amended by 40)
d = (cl + scale/2) div_euclid scale — round-half-up of the centeredrepresentative (ties at +scale/2 round up; at −scale/2 round to 0);m̂ = v − su componentwise, one su serving both components (erratum 20);validation is a slot-ascending scan reporting the first offending slot.WP: §6.3.1 ("(m̂ − center)/scale" mapped to this rule).
35. [FROZEN] Canonical envelope (figures as amended by 40)
Every slot ≤ legs·DIGIT_MAX, uniformly (no guard plane); legs ∈ [1, CHUNK_MAX].The privacy floor B_min is a ceremony policy (batches padded to exactlyCHUNK_MAX legs), not a decode rule — decode is a pure function of(legs, su, v). WP: §6.3.2, §6.3.5.
36. [FROZEN] Reveal record
legs (u32 LE) ‖ 512 digit sums (u32 LE) = 2,052 B; Δ_B = resolve(sums) is thederived cross-check; from_bytes re-validates the envelope — a parsed revealis always canonical. WP: §6.3.2 (record unspecified).
37. [FROZEN] Smudging allocation (figures as amended by 40)
SMUDGING_BUDGET = 2¹⁰ per coefficient on |su_combined − s·u_B|∞; the VPD proofsenforce ε per member (erratum 48); post-smudging honest margin 9.49σ(compile-floored ≥ 89σ²); honest false-rejection ≈ 10⁻²¹ per slot. WP:§6.3.3.
38. [FROZEN] Block assembly
Δ_B = Σ_chunks resolve(chunk) mod 2⁶⁴ — never Σ digit sums across chunks(cross-chunk sums exceed the decode headroom). Genesis chunks are exactly 128legs including committee zero-padding. WP: §6.3.2.
39. [WP-ERRATUM] In-envelope collusion residual
A ≥ t committee collusion can shift decodes by whole digit units while stayinginside the canonical envelope — corrupting the advisory aggregate withoutdetection at decode; everything coarser (negative digits, above-envelope sums,wrong chunk size) is a deterministically detected RevealError. §6.3.4's"no step … without a publicly verifiable fraud condition" requires thisqualification. Priced by slashing, stake, and rotation (§2.5); systematiccorruption remains statistically visible in committed forecast residuals overtime (advisory, §10). → App E.5. WP: §6.3.4, §2.5.

Chunk 9 — nerv-seal III: dkg, vpd, epoch, circuit_stmt (entries 40–56)
40. [WP-ERRATUM] The DKG closure (supersedes 19, 27, 30, 32, 35; amends figures in 26, 28, 34, 37)
A dealerless committee key is sum-structured: s and E₀ are n-fold sums of member secrets. With ternary per-member coefficients, key variance is n/2 = 5 and honest |key|∞ ≤ n = 10 — a bound the DKG's FS proofs establish per member. Honest chunk noise σ = 1,619. The 9-bit/2¹⁴ budget collapses to 4.4σ at thatvariance (silent decode failures per network-day: tens of thousands).Closure: 8-bit digits, 8 positions (exactly 64 bits — guard plane deleted,position 7 a full digit), scale 2¹⁵, ternary DKG noise → margin(16,384 − 1,024)/1,619 = 9.49σ (floor ≥ 89σ²); adversarial closurecompile-time: SCALE·(128·255) + 128·122,883 + 1,024 = 1.085·10⁹ < (q−1)/2 =1.611·10⁹. Envelope: legs·255 uniformly, all 512 slots. → App E.1. WP:§6.3.1, §6.3.8, App B.
41. [WP-ERRATUM] DKG noise posture
DKG-side coefficients are ternary CBD(η = 1) (dkg_noise_eta); the WP §6.3.8'sη = 2 applies to the encryptor side (unchanged). KEY_BOUND = n = 10 is theproven key bound (FS statements bind member coefficients to ±1; triangle overn members); the budget closes against exactly this. → App E.2. WP: §6.3.5,§6.3.8, App B.
42. [FROZEN] Evaluation points
γ_j = x^j (ring monomials; j ∈ 1..n distinct mod 2N; pairwise differencesinvertible since x^{j−k} − 1 is coprime to x^N + 1 for 0 < |j−k| < 256).Integer evaluation points would inflate shares to ~7·10⁷ coefficientmagnitudes and destroy the Lyubashevsky response bounds; monomial points keep|s_j|∞ ≤ n·t = 70. Lagrange λ_j ∈ R_q (arbitrary ring elements — publicmultipliers); interpolation identities are self-checked at derivation.→ App E.6. WP: §6.3.5.
43. [FROZEN] Commitment structure
A_c = (A_L, A_R) ∈ (R^{8×8})², XOF-derived from the epoch seed undernerv.seal.dkg; commit(v, ρ) = A_L·v + A_R·ρ (Ajtai form); binding MSIS /hiding MLWE — estimator-gated at M1 (dims provisional; a two-constantchange). T = Σ_i PK_i is literally "the sum of share commitments": PK_i =a_{i,0}·A + E_{0,i} is simultaneously the Ajtai commitment with matrix[A^T | I] and the MLWE public-key contribution (erratum 20's T-shape).→ App E.6. WP: §6.3.5.
44. [WP-ERRATUM] VSS pattern (fragment delivery)
Openings (f_i(γ_j), ρ′{i,j}) travel privately (ML-KEM; epoch.rs);recipients verify each against the public F{i,j} — mismatch is complaint andfraud evidence. W_j = Σ_i F_{i,j} is the transcript-bound share commitmentwhose opening (s_j, ρ̄j = Σ_i ρ′{i,j}) is known exactly to member j —resolving "transcript-derived commitments have no opener". The C_{i,ℓ}coefficient commitments are retained per WP literalism though functionallysubsumed by F + PK (cost: t·8,192 B ≈ 57 KB/member/epoch — consolidationcorrects the chunk-9 prose figure). → App E.6. WP: §6.3.5.
45. [WP-ERRATUM] FS-with-aborts engine
Statement = sparse block-linear system with per-block ∞-bounds; challengeweight 32 (±1, positions mod 256, exact); γ = 2²² with per-group box rejection(accepted z uniform on the public inner box, per-coordinate lemma; w modulobias ≤ 2⁻⁴⁶, stated); attempt cap 1024 (expected ≈ 1.5 attempts DKG, ≈ 4–6VPD). Soundness by forking → MSIS; ZK by rejection + ROM challengeprogramming — the full proofs are the M1 dual-audit gate. Proof wire =8 + (r + k)·1024 B: DKG ≈ 369 KB/member/epoch (r = 152, k = 208); VPD26,632 B (erratum 48 corrects the initial ≈ 22 KB estimate). → App E.8(sizes → E.3). WP: §6.3.3, §6.3.5.
46. [HYGIENE] Module geography
dkg.rs → dkg/ (mod.rs + sigma.rs) — same module path; the shared engine isone auditable file. Ring additions: try_invert (extended Euclid),monomial, mul_monomial, one — the Lagrange toolbox. WP: —.
47. [WP-ERRATUM] (major) The reveal/partial leakage surface
The ceremony's public outputs are themselves LWE samples: per reveal, thecombined value (256 equations on s; noise = the smudging, σ(Σε) ≈ 195 overt = 7) and t partials (equations on each share s_j, secret σ ≈ 5.9; σ(ε_j) ≈74). Per-sample difficulty: the aggregate instance is dominated by thepublic key's own MLWE instance (same secret, dimension 2,048, σ = 2.24 — thereveal instance is the same secret at 87× the noise); the per-partialinstance is distinct and sits between. The unpriced axis is sampleabundance: the PK exposes 4,096 samples, fixed; reveal streams expose up to≈ 1.1·10⁷ (aggregate-only) / 8.9·10⁷ (with partials) per epoch, anddual/FFT-class attacks monetize m ≫ n. Information-theoretically ~2 chunksdetermine the key; hardness is purely computational. Adversarial influenceover u_B is standard chosen-instance LWE. Mitigations: rotation + PSS(erratum 50) bound every instance to one epoch; the ε knob is documented inreserve (128 → 256 per member at ≈ 0.5σ margin; floor 89σ² → 81σ²); a namedM1 estimator line-item; §9.4's kill-switch is the designed fallback; blastradius = per-epoch individual-delta privacy (advisory tier), custody untouchedby construction. Subset rotation would multiply, not limit, the surface.→ App E.4. WP: §2.5, §6.4, §9.4.
48. [FROZEN] VPD construction
p_j = λ_j·(s_j·u_B) + ε_j (λ folded into the partial — WP-literal; combinationis a plain sum). Proof (dkg::sigma, domain nerv.seal.vpd): 9 rows (8 × theW_j commitment + 1 × the partial equation) over the 17-block witness(s_j ‖ ρ̄_j ‖ ε_j), bounds (70 ‖ 10 ‖ 128). ε uniform ±128, exact u16rejection over 257 classes (reject rate 2⁻¹⁶, cap 64). FS context =ASeed ‖ member ‖ n ‖ t ‖ subset ‖ u_B — a partial is valid for exactly one(key, subset, batch ciphertext) triple. Proof 26,632 B; wire 27,661 B; ~7partials ≈ 194 KB transient per 128-leg chunk (≈ 1.5 KB/leg). Smudgingenforcement is per-partial via the witness bound — t·128 = 896 ≤ 1,024structurally; ε is never published, so combination needs no budget check.→ App E.3. WP: §6.3.3.
49. [FROZEN] Ceremony integration + register hygiene
decode_chunk's su seam is realized: |Σp_j − s·u_B|∞ ≤ 896 by proven boundsalone (the ceremony test observes it against the reconstructed key; noprotocol party is ever omniscient). epoch.rs owns subset policy, the per-keyusage ledger, rotation/PSS, ML-KEM delivery, and forced padding. noise.rs doccorrected: per-leg σ ≈ 143 (not 202); all pinned margin figures werevariance-derived and stand. WP: —.
50. [WP-ERRATUM] (major) PSS semantics resolved
Cross-committee resharing of R_q secrets cannot preserve share shortness: newshares are λ-weighted combinations, and Lagrange weights over R_q arenon-short — the Lyubashevsky bounds fail, so there are no VPDs and noverifiable handoff. Same-committee zero-refresh preserves shortness but isredundant at 24-hour cadence with fresh keys. Implemented: a fresh DKG perepoch (D.1(a) satisfied verbatim — fresh, statistically independent randomnesswith the DKG's Ajtai commitments and FS consistency proofs); the handoffwindow served by the outgoing committee's own short shares; ML-KEM backupdelivery of the exact short shares (§6.3.6) — each backup opens the publicW_j, so delivery is verifiable and stand-ins produce normal VPDs; erasure atwindow close (D.1(c), operational); unopened batches become missed reveals(D.1(d)). Forward secrecy holds as D.1 states: cross-epoch share sets areshares of independent secrets — non-combinable by construction; stolensuperseded shares open nothing in the new epoch. → App E.7. WP: App D.1,§6.3.6.
51. [FROZEN] Handoff window
2 beacon intervals (D.6). HandoffWindow is a deterministic state machine(pending → opened | missed); MissedReveal { epoch, batch, legs } is thepublic derived-state record §10's skip-and-carry rule consumes (wired atchunk 17); reveal liveness is never a custody dependency (§6.3.6).→ App E.7. WP: §6.3.6, D.1, D.6.
52. [WP-ERRATUM] Force-padding decode rule
Padded batches decode with legs = the real leg count. Honest pads arezero-encryptions — invisible to the reveal (sums = the real digits exactly);pad-plaintext injection beyond the real envelope is a detectable invalidreveal; within-envelope injection joins the erratum-39 collusion residual(epoch-boundary only, advisory tier). Padding is public-key work — any membercan generate it; deterministic in the ceremony seed. → App E.5. WP:§6.3.5–§6.3.6.
53. [FROZEN] Backup delivery seam
SealedBackup.sealed is the ML-KEM-768 + ChaCha20-Poly1305 box overbackup_payload, produced by the delivery layer (nerv-crypto vianerv-net/wallet — chunks 15/19's wiring call sites); nerv-seal owns theenvelope, the payload, and the W_j-verification path. The in-test seal is areal reference cipher (XOF keystream + BLAKE3 tag), per the no-mocks policy'stest-side second-implementation clause. WP: §6.3.6.
54. [FROZEN] Reveal ledger
SAMPLES_PER_REVEAL = (1 + t)·256 = 2,048 LWE samples per reveal (the combinedvalue plus t partials — erratum 47's surface, made monitorable); per-keycounts keyed by the transcript digest; a governance cap makes it enforceable(record fails at cap → the ceremony treats the batch as a missed reveal).→ App E.4. WP: §6.3.7, §9.4.
55. [FROZEN] circuit_stmt freeze
Statement 9 = the 8-bit digit encoding (per coordinate: eight 8-bit rangechecks + the reconstruction identity Σ d_k·2^{8k} ≡ δ_j mod 2⁶⁴); statement 10= 10 rows (u = A·r + e₁ ×8; v = T·r + e₂ + scale·m ×2) over the 20-blockwitness (r ‖ e₁ ‖ e₂ ‖ m), bounds ±3/±3/±3/255, scale 2¹⁵, A from ASeed.epoch_key_identifier = H("nerv.seal.stmt" ‖ ASeed ‖ T) — §5.1's publicinput realized. The seal chip (nerv-proofs) must agree bit-for-bit (DSR-7;conformance vectors at M1). → App E.11. WP: §5.1.
56. [DOMAIN] Epoch domains
nerv.seal.rotation (rotation records), nerv.seal.stmt (epoch keyidentifier) — EXT-v1. WP: D-02.
57. [WP-ERRATUM] Option B — plonky3 excised; native nerv-stark engine. The prover stack is built in-house per the public ethSTARK-class FRI specification over the WP-mandated BLAKE3 Fiat–Shamir (FsTranscript) and Goldilocks field; prover/verifier reclassify [WRAP plonky3] → [BUILD]. Erratum 85's blake3-chip reuse for the fold phase is restored (the Poseidon2-duplex correction is recorded on the not-taken path). Erratum 89 retired (no git pin; no deny.toml entry). The dual M1 audit now covers the FRI engine.
58. [WP-ERRATUM] Security model, native. p3-security's internal formulas and its six regression vectors retired; conjectured estimate = min(CR, n/|EF|, d·N/|EF|, w/|EF|, b·q/2 + grinding) — each term the standard bound for its round (α-fold, DEEP-ALI, batched-opening bind, Johnson-conjecture FRI). The proven-regime companion figure retired; selecting a proven-bound reference (2024/1553-class) is an M1 audit deliverable. D.2 policy unchanged: conjectured ≥ 100 at CR 128; PQ = min(algebraic, 85).
59. [HYGIENE] constants.rs v3. Duplicate declaration blocks removed (carryover); SEAL_PSS_HANDOFF_INTERVALS removed from the Domain inventory — it is a parameter (params::SEAL_PSS_HANDOFF_INTERVALS), not a domain.
60. [DOMAIN] nerv.stark.leaf / nerv.stark.node (EXT-v1); [FROZEN] commitment structure — one BLAKE3 row-Merkle tree per commitment, leaf = all committed columns at one domain position; power-of-two heights asserted, never padded; batched multi-matrix trees deferred as a proof-size economy (recorded, not soundness-relevant).
61. [WP-ERRATUM] Option B — plonky3 excised; native nerv-stark engine. The prover stack is built in-house per the ethSTARK-class FRI specification over the WP-mandated BLAKE3 Fiat–Shamir (FsTranscript) and the Goldilocks field; prover/verifier reclassify [WRAP plonky3] → [BUILD]. Erratum 85's blake3-chip reuse for the fold phase is restored (the Poseidon2-duplex correction is recorded on the not-taken path). Erratum 89 retired. The M1 dual audit now covers the FRI engine.
62. [WP-ERRATUM] Security model, native. p3-security's formulas retired; conjectured = min(CR, n/|EF|, d·N/|EF|, w/|EF|, b·q/2 + grinding), each term the standard bound for its round — term-for-term the soundness of the delivered DEEP-ALI composition. Proven-regime companion retired; selecting a proven-bound reference is an M1 deliverable. D.2 policy unchanged.
63. [HYGIENE] constants.rs v3. Duplicate blocks removed; SEAL_PSS_HANDOFF_INTERVALS is a parameter, not a domain.
64. [DOMAIN/FROZEN] STARK commitments. nerv.stark.leaf / nerv.stark.node (EXT-v1); one BLAKE3 row-Merkle per commitment, leaf = all committed columns at one domain position; power-of-two heights asserted, never padded; multi-matrix batching deferred (proof-size economy).
65. [FROZEN] The composition (chunk 12.5). Single commitment domain C = N·2^b·R (R the quotient headroom) — quotient is one codeword, no chunking. The preprocessed trace is NOT committed: the verifier holds the authenticated table and Lagrange-evaluates it at ζ — a differing table rejects (pinned). FRI's final layer is sent as coefficients capped at 2^(f−b) (floor 1) — the cap is the low-degree claim's teeth. Outer openings cost O(q·(2W+2)) — the recorded driver cost; the production lookup-gated AIR form (errata 63/70/73/79/83 chain) is what meets the 192 KiB ceiling. The (x−gζ)-divisor DEEP binding (one trace row per query instead of two, soundness-neutral) is recorded as a proof-size roadmap item. Driver zero-padding contract: pad-compatible AIRs only — custody_air's register copies are is_last_row-gated and custody heights (66·k) are never powers of two, so custody-tier proving requires the generator/AIR obligation: generate at pow2 heights or gate copies by a prep flag (conservation's step pattern). Recorded as the chunk-10/11 follow-up; chip-tier end-to-end is delivered and pinned.
66. [GENESIS-CONFIG] [proofs.fri] freeze. b=4, q=56, f=1, pow=0; margin analysis above; any change is a visible spec-hash re-freeze; the conformance validator gains the cross-check (log_c feasibility at the envelope's corners).
67. [FROZEN] delta_air layout v2 + row governance. SLOT_STRIDE 228 (ACTIVE flag at +227: inactive ⟹ mag limbs zero; one-hots and distinctness activity-gated — padded slots are zero one-hots); LEG_W 3744; R_LOGA 776 (was 3740 — out of block); TIE_STRIDE 160 with split TIE_MACF/TIE_DIG flags (the overloaded dj flag forced DREG=0 at digit rows); PV_DELTA_ROW/PV_DELTA_COPY prep gates (521/522) so the composed zero tail passes and the final row's successor is unconstrained; log chain gated by PV_LOG; LOGA/LOGB per-row running values; the final row's aggregate identities are TWO limb-equalities — Σ|neg slots| = boundary IN, Σ|pos slots| = Σ volume rails — (the prior combined identity asserted fees = 0); ZINV pairing fixed (z_hi ↔ HI limb) with true inverses.
68. [HYGIENE] encoder v2. Accumulation carry-ins (limb 1 adds AKP/AKN[0], limb 2 adds [1]); enc_a0_dec deleted (asserted a0 = rr + rr·2^15); enc_a1/a2_dec corrected to the two-term forms A₁ = MLO + 2^15·MH, A₂ = AHL + 2^15·AHH; register offsets follow 63 (R_LOGA 776, stride 228); the delta generator writes pre-add PP/PN (the threading's left side), the correct C1, the full carry chain, true INVL/C2C, and debug-asserts the rounding against the codec's δ (the free differential).
69. [FROZEN] tx_air — the whole-transaction composition. Geometry: modules side-by-side in columns over H = max(custody, delta, seal); custody's registers extend into the taller tail; every seal leg shares rows [0,242) and ONE prep block (the seal prep is leg-independent: phases, twiddles, MOP depend only on the epoch key and transform structure — pinned). Bindings: m ↔ PT at the v-transform F10 rows (closes δ → digits → PT → m → v); per-leg fee ↔ TWO 32-bit public words (a single 64-bit word admits a fee+p bit alias — two-limb equality is collision-free); the expiry block (public per-leg expiry, < 2^40 capped in-circuit AND native-checked at both drivers, E = q·86400 + rem, 16 time-bucket one-hots derived with strict upper-bound uniqueness — replacing delta's free R_TIME/R_TYPE witnesses; type one-hots preprocessed from public structure). Verifier regenerates its own prep from (shell structure, W, epoch key) — gen_tx_prep, differentially pinned against the prover's. Driver obligations: check_ct_binding (shell ct == u32-LE serialize(u,v) — the tie making ct_B aggregation sound); claim legs (kind 3) deferred with the claim-leg statement set (WP §12.3).
70. [WP-ERRATUM] seal_chip region semantics (found in the composition audit). The delivered chip's F5 copy gate (1−is_mac)·is_last_row fired through the inverse section while the generator wrote REG_ACC only on rows 73–152 — the chip's own differential fails at rows 152+. Fixed: PV_SEAL_COPY (rows 0..ROWS−1) gates all three register-copy families (F3/F5/F6), and the generator holds ACC at block 9's final value through the inverse section. Consequence: the chip is composable (its region is self-contained — a 242-row slice verifies standalone, which tx_air's tests exploit).
71. [DOMAIN] Chunk-13 domains (EXT-v1; D-02 additions): nerv.hdr (D-02's
allocated shard-header row, landed), nerv.state.ctb (H(ct_B), §4.3 — frozen
with part 2's ct_B preimage), nerv.ttau.{leaf,node,empty} (the T_τ
inclusion tree), nerv.legtree.{leaf,node,empty} (the block leg tree,
§11.2 — lands with part 2's block.rs). WP: §4.3, §5.5, §11.2.
72. [FROZEN] The T_τ tree. A fixed-depth-32 frontier-fold BLAKE3 Merkle
tree over the interval's deduplicated txid set in canonical byte order
(the leaf order fold::dedup's IntervalSet already fixes). Root fold: r =
E_0; at level k a set bit of the leaf count absorbs levels[k].last() as
the left child (r = node(frontier_k, r)), a clear bit pads (r = node(r,
E_k)); root of the empty set = E_32. Witnesses are variable-length: the
full 32-sibling list with the maximal E_k-valued suffix trimmed —
lossless, since verification pads missing levels with E_k and a real
sibling colliding with an empty digest is a TAU_NODE preimage break.
Capacity 2^32 leaves per interval. WP: §4.3 rule 1, §5.5 tier 2 ("a few
hundred bytes": depth ≈ 20 at 10^6 txids/interval; the trim holds
genesis-scale witnesses well under 1 KB).
73. [FROZEN] The genesis registry reference is (interval 0, the empty
T_τ root), finalized by definition: no txid can exhibit membership in the
empty tree, so blocks building on it settle no legs. WP: §4.3 (the
genesis registry state is unspecified).
74. [FROZEN] The anchor window (§4.3 rule 2) is the last 64 finalized
headers' NCT roots, seeded at genesis with the genesis state's empty NCT
root at position 0. The seed is lossless (no membership witness exists
against the empty NCT) and unblocks height-1 settlement; it is evicted
when the 64th header lands. Rule 2 binds input-bearing legs only — issue
legs carry no anchor binding (the whole-transaction proof's anchor set is
per input shard, §5.1); the delivered custody shells' mandatory anchor
field on issue legs is unconstrained data.
75. [FROZEN] The shard header. header_hash = BLAKE3("nerv.hdr" ‖
canonical encoding) — the QC subject and the 𝔾_t leaf. prev = the
predecessor's C_t (height 1 points at C₀; there is no genesis header —
C₀ is a published value per §13.3). prev_reveal is Option<512 bytes>:
Some carries the previous block's revealed Δ_B verbatim (opaque through
the authority stack, DSR-4); None records a missed reveal ceremony
(D.1(d) — the knowledge layer's skip-and-carry consumes it; Δ_B ≡ 0 is
unambiguous as Some([0; 512])). The producer payout is the full custody
Address, header-committed via header_hash but not C_t-committed (§4.2's
tuple is exactly six fields). D_t is 32 opaque bytes (DSR-4). WP: §4.3.
76. [FROZEN] The settled-leg record carries the FULL transaction shell
plus the leg's index into the canonical shell. Rule 1's witness proves
only txid membership (T_τ's leaves are txids, §5.5/fold-dedup), and the
txid is the hash over ALL legs — so the leg↔txid binding (without which
a producer pairs a registry txid with an unproven leg shell and mints
unproven notes) is verifiable only from the full serialization. One leg
per shard per shell (duplicate-shard-leg rejection) makes the record
per-transaction in the settling shard. DA accounting: a cross-shard
transaction's shell is carried by every settling shard's block; §8.8's
~0.7 KB shell row is the single-shard dominant case. WP: §4.3, §3.6.
77. [FROZEN] The block leg tree (§11.2): fixed-depth-14 frontier-fold
BLAKE3 tree over the block's legs in canonical (txid, leg) order; leaf =
H("nerv.legtree.leaf" ‖ txid ‖ leg) — the 33-byte witness element; capacity
2^14 ≥ the 10,000-leg cap; witnesses trimmed of the maximal empty suffix
(erratum 101's law at depth 14). The root is not header-committed (§4.3's
field list carries none): the witness verifies against the tree rebuilt
from the DA-published block, whose effects are bound by the header's
transit/NCT/nullifier roots (§11.6 archival regeneration). WP: §11.2.
78. [FROZEN] D.3 escrow semantics — the executor's instantiation:
(a) the escrow record is the SPEND leg's transit entry in the issuing
    shard, born Pending iff the shell has ≥1 conditional output, born
    Claimed otherwise (pure rule-4 replay record);
(b) conditional outputs are valid only on issue legs (legs without
    inputs); a conditional output on an input-bearing leg is invalid at
    settlement — otherwise a single-shard transaction mints both its
    output and the revert commitment (D.3's "conditional cross-shard
    output leg" read strictly);
(c) a shell with conditional outputs has exactly one input-bearing leg
    and all legs' expiry fields equal (single-minted reversion; one
    coherent reversion/deadline clock — the WP leaves cross-leg expiry
    coherence unspecified);
(d) the reversion record carries (txid, spend-leg) plus per-issue-leg
    evidence; the shell is re-fetched from the settling chain (the block
    at the entry's birth height) and re-hashed against the entry's txid.
    Reversion mints the revert commitments of the UNSETTLED issue legs'
    conditional outputs (per-leg granularity — D.3's all-or-nothing
    reading strands partially-completed value); settled legs carry
    membership witnesses (any finalized root), unsettled legs carry
    non-membership witnesses against the receiving shard's finalized
    transit root at the escrow's expiry height;
(e) the claim record (Pending → Claimed) carries membership evidence
    that every issue leg settled. Before the grace boundary it is
    optional hygiene; past it, every due-and-triggered entry must be
    resolved by a claim or a reversion record ("a block omitting a due
    reversion is invalid" — the trigger is the due entry whose issue
    legs' height-expiry transit roots are all beacon-finalized, a pure
    function of the beacon view);
(f) airtightness: reversion evidence for an unsettled leg requires that
    leg's height-expiry header finalized — absence at that root plus the
    deadline rule (issue legs settle only at receiving-shard heights ≤
    expiry) precludes any later valid settlement, so
    reversion-then-settlement double-minting is unreachable. This
    weakens D.3's liveness corollary in one corner: a receiving shard
    dead below its expiry height leaves the escrow Pending (the pre-D.3
    abandonment outcome) — the alternative (reverting on a sub-expiry
    tip) reopens double-minting under shard resurrection, which the WP's
    deadline rule alone does not preclude. T_max bounds the wait;
(g) the issue-condition witness (rule 5) is a membership proof of the
    sibling spend leg's transit entry against the sibling shard's
    beacon-finalized transit root; the carried entry's post-settlement
    state is irrelevant (any state proves settlement). The issue leg's
    deadline (rule 5 addition) is receiving-shard height ≤ the leg's
    expiry. WP: §4.5, App D.3.
79. [FROZEN] Executor validity: any failed rule on any leg, reversion
record, or claim record invalidates the whole block (the producer's
validity filtering precedes canonical ordering; an invalid leg never
settles). Block legs are strictly ascending in (txid, leg). WP: §4.3.
80. [FROZEN] The executor's seams: BeaconView supplies the beacon's
finalized T_τ roots and per-shard finalized transit roots; ChainSource
supplies the shard's own settled-leg shells (the block at a transit
entry's birth height, per its height field). D.3's "pure function of
finalized beacon data" is instantiated as beacon-finalized roots plus
the chain/DA data they commit. A ChainSource miss on an omission-scan
lookup skips that entry's trigger (conservative — no false invalidity;
full nodes hold the chain). WP: §4.3, App D.3.
81. [FROZEN] H(ct_B) preimage: BLAKE3("nerv.state.ctb" ‖ ct_wire), ct_wire
the summed Ciphertext's canonical serialization (u ‖ v, 10,240 B genesis).
The executor parses each settled leg's ct (exactly the Ciphertext wire
length — the custody shell's 12,288-B cap admits no valid block) and
sums via Ciphertext::add. WP: §4.3.
82. [HYGIENE] nerv-state dependencies: core, crypto (the block's
quorum certificate — the design doc's list), custody, seal (ct_B
summation — an addition to the doc's list; no cycle: nerv-seal depends
only on nerv-core). The registry/economy deps arrive with chunks 14/17
behind the BeaconView seam. The executor checks only the header's
qc_hash binding; the QC's committee validity is the consensus layer's
(chunk 14).
83. [FROZEN] apply_block is by-value: an invalid block leaves the caller's
state untouched (the moved state is dropped on Err); the store (part 4)
snapshots for fork choice. The anchor-ring update on apply is the finality
seam — chunk 14's consensus gates which blocks reach the executor; apply
== finalize within this crate. Nullifier insertion height = block height.
NCT append order: leg outputs (canonical leg order, within-leg shell
order), then reversion mints (record order, then within-record unsettled
issue legs in canonical leg order, then within-leg conditional-output
order). Records apply: reversions (vec order) then claims (vec order).
The executor checks the header's qc_hash binding only; the QC's
signatures and committee membership are chunk 14's (nerv-consensus::qc).
WP: §4.3, §4.6.
84. [FROZEN] Rule instantiations (executor-side): rule 1 = the leg's T_τ
witness against the header's registry root, which must equal the beacon's
finalized root for its interval (no registry freshness window — a
finalized T_τ is permanently sound and the WP specifies none); rule 2
binds input-bearing legs only (erratum 103), validity = Goldilocks-range
digest + ring membership; rule 5's deadline: issue legs settle only at
receiving-shard heights ≤ expiry; D.3's expiry bounds
[h_include + T_min, h_include + T_max] are enforced at the spend leg's
settlement iff the shell has conditional outputs (issue legs carry the
deadline, not the bounds — their expiry equals the spend's by erratum
107(c)); records validate against the pre-block log with a per-block
consumption set, and a record targeting an entry born in the same block
is rejected (unreachable per rule 5's settlement ordering; conservative).
WP: §4.3, App D.3.
85. [FROZEN] The omission scan (rule 7 / D.3(e)): over the pre-block
Pending entries due at the new height; an entry is triggerable iff its
spend-leg shell is ChainSource-recoverable (txid re-hash included) and
every issue leg's expiry-height transit root is BeaconView-available;
every triggerable due entry must be consumed by a claim or reversion
record in the block, else the block is invalid; non-triggerable due
entries are skipped (conservative — no false invalidity; erratum 109's
discipline). WP: §4.3 rule 7, App D.3.
86. [DOMAIN] nerv.fraud (EXT-v1; D-02): fraud-evidence digest binding —
H("nerv.fraud" ‖ class ‖ canonical encoding); class 0 = the minimal proof
(block ‖ facts ‖ condition), class 1 = the reexecution proof (block).
87. [FROZEN] Fraud evidence, two classes (WP §4.3, §5.5's challenge
window, §11.4). (a) MINIMAL — the block plus declared beacon facts;
verification confirms every claimed fact against the verifier's view, then
evaluates the condition inside the closure of the claims (a read outside
the claims fails, never guesses; a condition whose needed root is
unclaimed is Malformed, not exhibited). Conditions: unresolvable, the
four shell laws, rule 1 (the facts' interval must equal the header's
registry interval — else Malformed), fee total, H(ct_B) (unparseable ct
included), in-block nullifier conflict (first-conflict leg), the
settlement deadline, sibling count, sibling evidence. The in-block
transit arm is unreachable post-resolution (canonical leg order makes
(txid, leg) keys distinct); the executor keeps it as a free invariant
guard — it is not a fraud condition. An exact condition verifies;
anything else rejects (ConditionNotExhibited) — the false-claim slash is
chunks 14/17's machinery. (b) REEXECUTION — the block alone; verified by
a predecessor-state holder (the full node, §4.3's native path) running
Update; any ExecutorError is the proven condition — wrong-root fraud
included. §5.5's registry verify-one challenge is the registry's own
chunk-14 surface, not this crate's.
88. [FROZEN] The node-store seam (DSR-10): one NodeStore trait, two real
backends — MemStore (per-column-family BTreeMaps over the shared 11-byte
key layout shard(3B) ‖ height(8B LE); the testkit's) always compiled,
and RocksStore behind the `rocksdb` cargo feature (the C++ backend is
opt-in so toolchain-less and riscv64 determinism-matrix builds stay pure
Rust — DSR-10's "exactly one seam" is the trait, not the build).
Column families now: meta (schema_version, migration-gated; v1), headers,
blocks, snapshots; emission/derived/da/mempool land with their owning
crates. Per-shard tip markers (blocks, snapshots) occupy the reserved
height key u64::MAX; put_snapshot rejects that height. Puts follow a
nondecreasing-height-per-shard discipline; markers are last-write-wins.
ShardState snapshots serialize the full state (component trees + anchor
ring); decode enforces the component codecs' structural checks — a
self-consistent-but-wrong snapshot is caught by the next block's
prev-chaining. Rebuild-from-DA: ShardArchive + BeaconView → apply
1..=tip, the archive doubling as ChainSource; load_shard prefers the
latest snapshot and applies forward, falling back to a full rebuild on
any fast-path failure (a persistent failure surfaces from the rebuild);
a store read failure propagates.
89. [DOMAIN] Chunk-14 registry domains (EXT-v1; D-02):
nerv.bundle.attest (the aggregator's signed message: domain ‖ txid_root
‖ count u32 LE), nerv.registry.commit (the interval-commit digest — the
degraded-mode QC's subject), nerv.registry.challenge (the
inclusion-challenge evidence digest).
90. [FROZEN] The verification gate (tier 1/tier 2's "verified", WP
§5.5) is tx_air::verify_transaction under the statement-11 binding:
fs::bind_transaction over the canonical shell's nullifiers, txid, shell
digest, and TxPublicInputs::derive with the epoch key identifier computed
from the context's epoch key. VerifyContext = (FriShape, CodecW, epoch
PublicKey). The aggregator's pool is structural + dedup only — the WP's
trust boundary is bundle verification at the registry, which
re-establishes every contained proof independently; the pool's
discretion is not load-bearing (admit() runs the full gate as the
production path; insert() is the post-verification pool). The
TransactionProof wire form lands in tx_air: {log_n, width,
ComposedProof, publics} — the preprocessed table is verifier-regenerable
(gen_tx_prep) and never travels; decode yields an empty prep (verify
paths regenerate their own).
91. [FROZEN] The bundle: the transaction list (canonical shells +
proofs, submission order) + the attestation (ML-DSA over
nerv.bundle.attest ‖ txid_root ‖ count). txid_root = the erratum-101
T_τ tree over the bundle's deduplicated txid set; intra-bundle
duplicates are malformed. The registry enforces only the 4,096 maximum
— "the registry must include any valid bundle" (§5.5): bundle_min is
the aggregator's batching target, not a registry admission floor.
verify_bundle runs per-transaction proof verification — erratum 85's
folding-AIR deferral makes the R1 fallback (per-proof verification;
throughput cost only) the delivered bundle gate; the folded fast path
replaces the loop when the tier-1/2 AIRs land.
92. [FROZEN] The interval commit: arrival-indexed bundles →
IntervalLedger::dedup (first-wins by arrival index; cross-interval
drops) → T_τ = the erratum-101 tree over the interval set →
IntervalCommit {interval, tau_root, bundle summaries in arrival order},
digest under nerv.registry.commit. The degraded QC's subject is that
digest — the attestation binds the whole bundle set (slashing evidence).
93. [FROZEN] Degraded mode (WP §5.5): intake closes at the interval
boundary (1 s); G_τ is awaited until boundary + grace (5 s, params);
past that without delivery the registry committee (21, quorum 15)
attests the commit digest, and the interval finalizes only when the
30 s challenge window closes with no sustained challenge. A sustained
inclusion challenge voids the interval (never finalized; honest txids
re-enter later intervals — their proofs still verify). nerv-registry
depends on nerv-state: the T_τ builder must be bit-identical to the
executor's rule-1 verifier — single-sourced, not twinned (an addition
to the design doc's dependency list; no cycle: state depends on
neither registry nor proofs). Slash escrow is chunk-17 economy
machinery.
94. [FROZEN] The inclusion challenge: the T_τ witness for the txid,
the canonical shell (its re-derived txid must match the witness — the
challenged inclusion is bound), and the ORIGINAL submitted proof. The
beacon re-runs the gate: failure sustains (the attestation's "every
proof verified" is false); success rejects. The challenger never
produces a proof — they present the attested transaction's own.
95. [DOMAIN] D-02's allocated rows land: nerv.att (A_τ and epoch-attestation
digests; the ATT-domain Merkle nodes of the epoch's interval-digest tree),
nerv.beacon (𝔾-tree leaves' absent-shard sentinel and internal nodes). New
EXT-v1 rows: nerv.sort.derive (role- and instance-separated sortition
randomness derivation), nerv.slash (slash-evidence digest binding).
96. [FROZEN] Committee instantiation (§8.3; DSR-5/E-007/E-001/E-003):
(a) R_e = framed(nerv.sort.derive, [epoch-attestation digest of e−1,
"epoch-randomness", e]); R_0 = framed(nerv.sort.derive, [0^32,
"genesis-randomness", 0]) — genesis-config, overridable by the ceremony
at M5 bootstrap;
(b) role randomness r = framed(nerv.sort.derive, [R_e, label, extra]),
labels "shard"‖shard(3B), "decryption"‖shard(3B), "registry",
"attestation"‖interval(8B LE); ranking = nerv-crypto's H(r ‖ pk ‖
selection-epoch), tie-broken by pubkey;
(c) shard and registry committees are fresh per epoch (21/21, params);
(d) the beacon committee (31) is staggered in cohorts sized [11, 10, 10]
by selection-epoch mod 3: at epoch e the cohort selected at e is fresh
(R_e), the other two retain their selections from e−1/e−2; epochs 0–2 are
the bootstrap window (a single 31-member R_e selection; staggering begins
at epoch 3); a validator may hold multiple cohort seats (rare; evidence
identifies offenders by pubkey);
(e) E-001's attestation signer set: 21 of the epoch's beacon committee
per interval, ranked by H(r_att ‖ pk ‖ epoch);
(f) E-003's decryption committee: 10 of the 21 shard-committee members;
(g) the roster is pubkey-deduplicated; selection is roster-order-independent.
97. [FROZEN] A_τ (§11.2): the signed body = tag(1B)=0 ‖ interval ‖ 𝔾_τ ‖
T_τ root ‖ DA root ‖ prev-attestation digest; digest = BLAKE3("nerv.att" ‖
body); the QC's subject is the digest; the chain link is the digest (two
QCs over one body are one attestation). The DA root is committed data
computed by nerv-da (chunk 15) — the field is data whose producer lands
with the DA crate. 𝔾_τ: one leaf per active shard in canonical ShardSet
order — the shard's latest header hash in the interval, empty_leaf(
nerv.beacon) where a shard produced none; a complete binary tree over
the pow2-padded leaves (single leaf → root = leaf; empty → empty_leaf),
nodes H("nerv.beacon" ‖ l ‖ r); witnesses carry the full sibling list
(depth = log2 of the padded count the verifier derives from its known
shard count). Depth 10 at 1,024 shards, matching §11.2's witness table.
The epoch attestation: tag=1 ‖ epoch ‖ Merkle root (same construction,
ATT domain) over the epoch's interval-attestation digests in interval
order ‖ prev epoch-attestation digest (zero at epoch 0); signed by the
final interval's signer subset; its QC epoch is the epoch itself.
98. [FROZEN] Finality (§4.8 T4; §5.5): the beacon's interval finality is
a monotone +1 watermark. Per shard, the finalized line is the contiguous
observed chain up to the finalized height; a header at or below the
finalized height differing from the line is a FinalityViolation (two
valid QCs at one height — the slash surface). Pre-finality fork choice:
the byte-smallest header hash among children of the current tip is
canonical; a smaller child arriving after an extension reorganizes the
suffix (displaced hashes become alts, flushed as violations at
finalization); never past a finalized height. Deterministic per arrival
sequence; BFT production makes forks one-deep and transient. The guard
observes in order (height ≤ tip+1); higher arrivals are the pipeline's
buffering problem (shard_chain, part 3).
99. [FROZEN] Topology (§8.5): the metric is the per-epoch finalized
settled-leg count per shard (days map 1:1 onto 24h epochs). Split: the
trailing 7-epoch average (all 7 epochs required; floor division by the
window's seconds) strictly exceeding S_hi proposes; the same average
strictly exceeding 2× the engineered ceiling is emergency DETECTION only
(adoption is governance's, §C.2). Merge: both siblings' every
trailing-30-epoch per-second rate strictly below S_lo — the streak
reading of "sit below S_lo for 30 consecutive days". Adoption: proposals
at epoch e activate at e+7 (the notice); the engine applies splits/merges
through ShardSet (never re-homing notes); proposals dedupe per target
while pending; the 1,024 cap zeroes the split budget at capacity. The
sunset audit reports the trailing-7-epoch average against
SUNSET_NEGLIGIBLE_LEGS_PER_SEC = 4 (1% of S_lo; genesis-config).
100. [FROZEN] Slash evidence (§2.5, §11.4): InvalidBlock (either fraud
class — the minimal class is beacon-verifiable; the reexecution class
verifies against the predecessor state the verifier supplies),
InvalidInclusion (a sustained registry challenge; the attesting-committee
offender resolution is the economy's — the evidence carries the challenge
and the attested root), InvalidBundle (the aggregator's bundle and the
failing transaction index), DoubleSign (two valid QCs, one epoch, one
(shard, height), two header hashes; the signers in both bitmaps are the
offenders — 15+15 of 21 intersect in ≥ 9). Digests bind under nerv.slash
over a class tag and the underlying digests. The escrow edge into
nerv-economy lands with that crate (chunk 17); this crate's dependency
set is core/crypto/custody/state/registry (+dev: proofs, codec, seal for
fixture construction) — the design doc's economy edge is deferred to its
consumer, cycle-free either way.
101. [FROZEN] HeaderQc (erratum 100's DoubleSign substrate): the QC in
committee context — the subject must be the header's nerv.hdr hash;
heights start at 1; DoubleSign detection requires equal QC epochs (one
committee) and returns the member indices of the signer intersection.
102. [FROZEN] The QC binding (§4.6): the header-commits-QC_hash /
QC-signs-header circularity resolves by the zeroed-placeholder
construction — the QC's subject is signing_hash = BLAKE3("nerv.hdr" ‖
canonical encoding with qc_hash = 0^32); the header's qc_hash commits the
exact assembled QC; header_hash (chaining, 𝔾, A_τ, fraud digests) covers
the full encoding. apply_block checks the qc_hash binding only (erratum
112); the consensus layer's HeaderQc checks subject == signing_hash.
103. [FROZEN] Amendment to 127 (finality linkage): the shard guard's
parent linkage is the predecessor's C_t (the child's prev field), not its
header hash; observe carries (height, header_hash, prev, link) with link =
the header's own committed C_t — computable from the header's fields, so
the guard never needs the applied state.
104. [FROZEN] The delivered registry finalization path (delivery note on
122): the beacon's bundle intake is STRUCTURAL (validate_structure);
proof verification is the signing members' pre-attestation duty
(verify_bundle before signing the DegradedAttestation; the node binary
wires it, chunk 20); the committee QC + the 30 s window are the soundness
carriers (§5.5: committee stake + deterministic verifiability). Intervals
PIPELINE: up to ~30 pending, closed at the boundary, finalized in
interval order at their windows' closes; the in-flight union (all pending
sets) joins the ledger in dedup — a txid in a pending interval cannot
enter a later one; empty intervals carry no window and finalize at the
pending head. Void: the ledger commits the EMPTY set for the interval
(contiguity; the voided txids never settled — they re-enter later
intervals); tau_root returns None; no A_τ is issued; headers observed in
a voided interval finalize only via a later interval's tip
(chain-integrity catch-up).
105. [FROZEN] BeaconView over the Beacon (closes chunk 13's seam):
tau_root from finalized intervals only (watermark-gated); transit_root
from the CANONICAL header at (shard, height) iff height ≤ the shard's
finalized line, advanced by each finalized interval's per-shard
canonical-tip height. Uniform for active and legacy shards: the A_τ's DA
root (committed data; nerv-da, chunk 15) plus the header's QC are the
commitment; 𝔾 (active-set tips) is the light-client view, not the sole
finalization vehicle. The G tree covers the ACTIVE set in canonical
order (126 unchanged); a producing shard's leaf is its guard's canonical
tip at close.
106. [FROZEN] Production (shard_chain::produce): candidates sorted by
(txid, leg); the header sealed by effect simulation over cloned
component trees in erratum 112's order; the QC signed over signing_hash;
the sealed block then VERIFIED by apply_block against the parent state —
the simulation is checked, never trusted; adopted on success. ShardChain
retains its state history (states[0] = genesis), so reorgs of any depth
re-apply against the fork-point state and truncate; non-canonical forks
are recorded as alts without deep state validation (the snapshot path,
chunks 13/20).
107. [FROZEN] The epoch boundary: finalizing the epoch's last interval
(i+1 ≡ 0 mod 86,400) produces the EpochAttestation over the epoch's
interval-digest tree, signed by that interval's signer subset; R_{e+1} =
epoch_randomness(ea.digest(), e+1); the beacon exposes randomness(e).
Beacon::resume_at is the sync/checkpoint constructor (the anchor carries
the interval-chain and epoch-chain digests); mid-epoch resume yields an
epoch compression covering only post-resume digests (the full surface is
chunk 20's sync).
107. [FROZEN] Topology adoption: adopt(new_set) swaps the active set at
the epoch boundary and resets the per-interval G layout (splits'
children and merges' parents start with empty leaves — the tree is
per-interval); chains persist for all shards, legacy included; notes are
never re-homed. The emission-ledger hosting (the design doc's beacon
line) lands with nerv-economy (chunk 17).
108. [HYGIENE] nerv-registry gains build_interval_commit_excluding (the
in-flight filter; 133): each bundle's txid list is pre-filtered against
the exclusion set before dedup.
109. [WP-ERRATUM] The QC subject is the BODY hash, not the full header
hash. The full header hash includes qc_hash = H(QC), and the QC's subject
is the header hash — circular. Resolution (the standard Tendermint/
HotStuff construction): body_hash = H(header with qc_hash zeroed) — the
unsigned header; the committee signs body_hash; the header's qc_hash
commits to the resulting QC. The pair (header, QC) is bound: the QC's
signatures are over the body, the header commits to the QC. The FULL
header hash (with qc_hash) is the identity for 𝔾 leaves, prev-chaining,
and finality. Correction to Part 2's HeaderQc.
110. [FROZEN] The propose path (nerv-state): the producer computes the
header's effect fields (nct/nullifier/transit roots, fee total, ct_B
hash) by applying the body's effects to cloned state components — the
same public APIs the executor uses, in the same order — including the
reversion-mint computation (escrow recovery via the chain source).
The computed header is then verified by running apply_block on a clone
with a dummy QC (self-consistent qc_hash). The round-trip
(compute → assemble → apply_block succeeds) is the drift check. The
BlockBody is the legs + reversions + claims without the header/QC.
111. [FROZEN] The beacon state machine: per-shard ShardFinality guards
fed by QC-validated headers (observation is the caller's pipeline);
the 𝔾 tree rebuilt per interval from the guards' tips (the shard's
latest observed hash, the sentinel for absent shards); the
IntervalAttestation over (𝔾_τ, T_τ, DA, prev); interval finality as
the monotone watermark; the epoch boundary producing the
EpochAttestation and the next epoch's randomness; the interval signer
subset from the epoch's beacon committee (E-001). The committee keys
for each epoch's attestations are supplied by the caller (the
committee module's selection); the beacon stores the attestation chain.
112. [FROZEN] The shard-chain production pipeline: propose (compute the
header via nerv-state's propose, collect the QC over the body hash,
assemble the final block) and the committee's validate-and-sign path
(run apply_block against the predecessor, sign the body hash on
success). The pipeline enforces canonical ordering (resolve_legs) and
the D.3 rule-7 obligations (the omission scan inside propose's
validation). The producer's payout is the header-committed address.
113. [WP-ERRATUM] erasure.rs reclassified [WRAP reed-solomon-simd] →
[BUILD]. The 2D RS seam needs byte-shard encode/reconstruct control
(iterated row/column decoding over an extended square) whose exact
wrapped-API surface cannot be verified in this session; mis-wrapping is
worse than implementing the standard. Delivered: GF(2⁸) (0x11d, generator
2, log/exp tables), systematic Cauchy parity — C[a][j] = (x_a ⊕ y_j)⁻¹
with x_a = a, y_j = (n−k)+j, all distinct ⟹ every square submatrix
nonsingular ⟹ any k of n reconstructs (Gaussian elimination); every
square submatrix of the generator brute-force verified in tests at
(n,k) = (6,3), (8,4). GF arithmetic differentially tested against an
independent peasant (carry-less) multiplier. The workspace's
reed-solomon entries remain declared and inert; a later verified wrap
swap is one file. nerv-da's dependency list (core, crypto) corrected to
core only — crypto was anticipatory and is unused.
114. [FROZEN] The DA square (§8.7): block data → segments of ≤ MAX_K²·
CHUNK_LEN (k a power of two, 1..=64; CHUNK_LEN = 512 B; ≤ 2 MiB/blob) →
k×k data chunks (zero-padded), row-extend (k→2k per row), column-extend
(k→2k per column) → the 2k×2k extended square. Cell leaf =
BLAKE3("nerv.da.cell" ‖ blob(4 LE) ‖ row(2) ‖ col(2) ‖ chunk) —
position-bound; row/column roots are complete-binary Merkle trees over
the leaves ("nerv.da.node", fixed 64-byte nodes; counts are powers of
two by construction). The blob tree commits row_roots ‖ col_roots
(4k leaves); the set root = BLAKE3("nerv.da.root" ‖ shard ‖ height ‖
count ‖ [width ‖ data_len ‖ blob_tree_root]×) — flat fixed-width
framing; the SetCommitment itself is small (≈ 38 B/blob) and travels in
full. Cells authenticate by two paths: chunk → row tree (against
row_root) → blob tree (row_root at index row) → set.blob_tree_roots[blob]
— no set-level tree needed.
115. [FROZEN] Sampling and B7: positions drawn deterministically from
XOF("nerv.da.sample", [seed ‖ shard ‖ height ‖ blob ‖ width]) —
reproducible transcripts. samples_needed(w‰, p‰) = the smallest m with
((1000−w)/1000)^m ≤ (1000−p)/1000, computed exactly in u128 fixed point
(64 fractional bits, ceil-on-miss / floor-on-threshold — sound: the
returned m provably satisfies the bound, at most one above the exact
optimum). Unrecoverability bound: if iterated decoding stalls, every
stuck row was never decodable, so ≥ k+1 of its 2k cells are unknown,
and rows outside the stuck set are fully known (else their columns
would decode) — hence ≥ (k+1)² cells are withheld and the available
fraction is ≤ 3/4. So m samples detect any unrecoverable square with
probability ≥ 1−(3/4)^m, and detect a w-fraction withholding with
probability ≥ 1−(1−w/1000)^m. Genesis parameters (200‰, 999‰) give
m = 31: 0.8³¹ = 7.9·10⁻⁴ ≤ 10⁻³ (B7) and 0.75³¹ ≈ 10⁻⁴ (structural).
The 30 s window is a network property (chunk 16), not this crate's.
116. [FROZEN] The deterministic DA fraud: BadEncodingEvidence — k
authenticated cells of one committed line (row or column), the line's
committed root, and the line's blob-tree path. Verification: every
cell authenticates, the line-root authenticates, the k cells decode to a
full codeword, and the codeword's Merkle root ≠ the committed root —
the committed line was not a codeword (invalid erasure coding ⟹ invalid
block; slashable). If the committed line IS a codeword the decode
reproduces it exactly (k points determine the line), so the check never
fires honestly. Withholding itself has no cryptographically
self-contained proof (reconstructibility is demonstrated by possession,
at which point the data exists): its detection is the B7 sample
transcript, and full availability is proven by reconstruction against
the committed roots. Digests bind under "nerv.da.fraud"; the slash-class
wiring into nerv-consensus is the node's (chunk 20) — consensus does
not depend on nerv-da.
117. [WP-ERRATUM] Option B — libp2p excised; native transport. host.rs
reclassifies [WRAP libp2p] → [BUILD]. Rationale: (a) D-04's posture
strips libp2p of every property we would use (no noise/tls/quic; our own
ML-KEM + ChaCha20-Poly1305 + ML-DSA at the message layer; the ed25519
PeerId demoted to a non-authoritative handle with one expiring deny.toml
skip entry) — what remains is TCP + multiplexing + gossip plumbing whose
policy must be ours anyway, because DSR-8's headers-before-partials
ordering is not a gossipsub configuration; (b) the wrapped 0.5x API
surface (swarm builders, behaviour composition, gossipsub signing
modes) cannot be verified in-session, and mis-wrapping is worse than
implementing the standard (the erratum-139 discipline); (c) the native
path deletes the ed25519 skip entry entirely. Delivered: tokio TCP,
length-prefixed frames, an in-house 1-RTT PQ handshake (erratum 118),
and a connection registry — all generic over AsyncRead+AsyncWrite, so
the chunk-20 testkit's loopback transports ride the same code
(tokio::io::duplex in tests now). Peer identity = H("nerv.net.peer" ‖
vk), authoritative at the message layer. The workspace's libp2p
declaration goes inert (the plonky3 precedent: no git pin, no deny.toml
entry). Peer discovery at genesis is the static PeerBook — the
sortition-selected mesh is small and known per epoch; DHT-style learned
discovery is a roadmap item. Module geography: wire.rs added to the
design doc's file list (the protocol core host.rs wraps).
118. [FROZEN] The wire protocol: 1-RTT mutual PQ handshake. Initiator:
(version 1, ML-DSA vk, fresh ML-KEM-768 ek, ct→responder static ek).
Responder: (vk, ct→initiator fresh ek, ML-DSA sig over the role-tagged
transcript). Initiator: (sig). Transcript T = BLAKE3("nerv.net.hello" ‖
ver ‖ a_vk ‖ a_ek ‖ a_ct ‖ b_vk ‖ b_ct); signatures over
"nerv.net.hello.a"/"…b" ‖ T. Session keys = BLAKE3-KDF(
"nerv.net.session", ss_a ‖ ss_b ‖ T, "c2s"/"s2c"); the initiator sends
on c2s. Both KEM directions contribute: an active interposer must
break ML-KEM or forge ML-DSA. No forward secrecy (static responder key;
rotation = re-dial) — stated honestly; PFS via ephemeral exchange is a
roadmap note. Frames: u32-LE length ‖ ChaCha20-Poly1305 with the nonce
4-zero ‖ u64-LE counter; strict in-order counters = replay protection;
4 MiB payload cap (DA blob headroom); handshake messages are fixed-size
with exact-capped reads. The initiator pins the responder's vk (the
PeerBook entry) — WrongPeer on mismatch; the responder authorizes via
the host filter after the handshake.
119. [FROZEN] Host semantics: bind(signing key, static KEM dk, peer
filter); inbound authenticates by handshake signature, then the filter
authorizes. Dial(PeerInfo) is a single-shot attempt over the addresses
in order (retry/backoff supervision is the node binary's, chunk 20).
Duplicate connections from one PeerId keep the first. Events:
Connected/Frame/Disconnected over a bounded channel (back-pressure per
connection). Send routes by PeerId through the registry — a send racing
a disconnect is lost silently; the Disconnected event is the failure
surface. Shutdown aborts all connections (drop-cancellation) and closes
the event channel. PeerBook records (vk, ML-KEM ek, addrs) with
merge-dedup semantics; addresses are static at genesis.
120. [DOMAIN] nerv.gossip.msg (EXT-v1; D-02): the gossip dedup digest —
H("nerv.gossip.msg" ‖ wire payload).
121. [FROZEN] The gossip engine (DSR-8): flood gossip over the host mesh.
Message tags: 0=Header, 1=Partial, 2=Reveal; 3=Submission (allocated in
submission.rs); chunk 16's relay topics continue at 4+. Dedup by wire
digest, bounded FIFO seen-set (65,536), inserted ONLY on acceptance —
a partial rejected by the ordering gate is not marked seen, so its
re-send after the header arrives is accepted (the DSR-8 adversarial
sequence: partial-early → rejected; header; partial → accepted). The
ordering gate: a Partial for (shard, height) is rejected unless a Header
at that (shard, height) has been accepted. The (shard, height)
reference is the protocol-level ORDERING binding; the cryptographic
partial↔block binding is the VPD proof's u_B context — the ceremony
layer's job, never gossip's. Fork headers at one slot both propagate
(distinct digests; the equivocation evidence flows to finality); the
learned map is first-wins. Zero-height headers are rejected (headers
start at 1). Header retention is a per-shard trailing window (256
heights, pruned on learn): stale partials for pruned heights are
rejected — that block's ceremony is dead and replays must not re-open
the gate; pruning tightens, never loosens. Reveals are carried ungated
(DSR-8 names partials only). The engine is pure logic; the node's event
loop drives it (the recipe is GossipEngine's doc; the end-to-end socket
test is that recipe executed — chunk 20's node and the testkit's
adversarial harness run the same loop).
122. [FROZEN] Submission (§5.5): the submitted transaction is the
registry PoolEntry (canonical shell + proof), tag 3. Client fan-out:
deliver to the first N CONNECTED aggregators in list order (params'
mixnet.submission_fanout = 3 — the censorship-insurance N; list
pre-shuffling for load balance is client policy, not protocol); zero
delivery is an error. Aggregator ingress: decode → the full verification
gate (mempool::admit, erratum 119's production path), reporting
Admitted/Duplicate/Rejected — the duplicate check precedes verification,
so re-submissions of a pooled txid are idempotent. The relay-egress
seam: deliver() is the single-target primitive chunk 16's final mixnet
relay calls with the unwrapped, already-framed payload (the onion's
carried payload IS the encoded SubmissionMessage). Fee bucketing is
wallet-side (chunk 19), not protocol. nerv-net's dependency set (design-
doc correction): core, crypto, seal, state, registry (+dev: custody,
codec, proofs) — custody only transitively in the main graph; no cycle.
123. [FROZEN] PQ-Sphinx (§6.2), the packet: 20,000 bytes = header
(5 × ML-KEM-768 ct = 5,440) ‖ meta region (5 × 81-byte AEAD capsules =
405) ‖ payload region (14,155). Per hop i: (ss_i, ct_i) =
Encapsulate(ek_i, fresh); k_meta/k_stream = BLAKE3-KDF("nerv.sphinx",
ss_i, "meta"/"stream"); capsule_i = ChaCha20-Poly1305(k_meta_i,
next PeerId ‖ action ‖ H(M)). Header and meta region SHIFT one slot per
hop, the vacated slots refilled with fresh randomness — uniform size and
structure at every hop (the nested-AEAD alternative shrinks the body by
(51) per hop and leaks position; rejected). Payload region = M ⊕
⊕_i XOF(k_stream_i): each relay XORs its own stream off (the classical
Sphinx surplus adapted to per-hop KEM keys); relay i cannot strip deeper
than its layer. M = len(u16) ‖ data ‖ pad; the terminal (action Deliver)
verifies H(M) against its capsule-carried commitment — intermediate
bit-flips are detected at the terminal (the submission gate is the
backstop). Replay tag = BLAKE3("nerv.sphinx" ‖ ct_slot0 ‖
capsule0)[0..16], checked pre-decapsulation against the relay's bounded
cache (part 2). ML-KEM's implicit rejection makes decapsulation total —
the capsule's AEAD is the misroute/tamper detector (wrong relay, swapped
ct/capsule, corrupted capsule all fail the open). Frame tags: 4 = mix
packet (relay→relay), 5 = mix-delivered fragment (final relay→
aggregator). The SubmissionMessage rides as class-1 fragment data —
submission.rs's "the payload IS the encoded SubmissionMessage" refines
to "the class-1 fragment frame's data is" (doc note; the aggregator-side
join is the node's, chunk 20). Appendix A's "~9 packets" prose is
superseded by D.2's class set: a 165 KB transaction is class 16
(capacity 16 × 14,113 = 225,808 B; the honest 320 KB wire cost the WP
already carries).
124. [DOMAIN] nerv.sphinx (D-02's registered row, landed) and
nerv.relay.reg (EXT-v1: relay-registration signatures).
125. [FROZEN] Fragmentation (D.2/D.6): class ∈ {1,2,4,8,16} = the packet
count, the minimal class ≥ ⌈len/14,113⌉; larger classes may be chosen
deliberately (cover). The payload is padded to class × 14,113 and split
evenly — every frame's data is exactly 14,113 bytes (a fixed-size
uniformity property, enforced on decode). Frame = txid ‖ index ‖ count ‖
total_len(u32) ‖ data_len ‖ data (40 + data). Reassembly requires the
complete 0..count−1 index set over consistent (txid, count, total_len),
concatenates, truncates the padding to total_len.
126. [FROZEN] The relay registry (§6.2): the record = ML-DSA vk ‖
ML-KEM-768 ek ‖ addresses (≤16) ‖ operator tag (16 B) ‖ stake amount
(data only; the escrow is nerv-economy's, chunk 17). Registration = an
ML-DSA signature over "nerv.relay.reg" ‖ canonical record —
self-certifying (the vk in the record is the verifier). Selection
(E-007's wallet-random-diversified): one XOF stream over the wallet
seed, Fisher–Yates within operator groups (sorted by tag), round-robin
across groups, take 5 — deterministic in (seed, registry content);
the modulo bias of the shuffle draws is ≤ 2⁻⁶⁴ at relay-roster sizes
and selection is wallet-local policy, never canonical. Geography
diversification is wallet policy over the address fields. The on-chain
commitment path (stake, slashing) is the economy/consensus wiring.
host::PeerId gains from_hash (the meta's 32-byte next-hop
reconstruction).
127. [FROZEN] The relay jitter (§6.2; E-004) is the exact inverse-CDF
truncated exponential in integer arithmetic: u ∈ [0,2⁶⁴) uniform →
y = 1 − u·(1−e^{−r}) in Q64 (r = cap/mean, a params pin: cap % mean = 0,
r ≤ 12; genesis r = 5) → delay = round(mean·(−ln y)) ms, clamped at the
cap. ln is a Q64 routine (normalize to m ∈ [1/2,1), atanh power series,
|t| ≤ 1/3); e^{−r} and ln 2 are const integer series. Both
differentially tested against a test-only f64 oracle (DSR-11's dev-only
pattern — #[allow] on the test fns, nothing shipped). The statistical
battery (100k draws) pins mean ≈ 96.6, median ≈ 68.7, P(≤100 ms) ≈
0.636, max ≤ 500, P(0 ms) ≈ 0.5%. Deterministic in the draw; the
entropy seam is one u64 per packet (OS draw in production; entropy
failure fails SAFE — u64::MAX, the maximum delay, never zero).
128. [FROZEN] The replay cache: 16-byte tags, BTreeSet + FIFO eviction
at 65,536 (gossip's dedup precedent), checked BEFORE decapsulation — a
replay costs one hash, not one KEM. A misrouted packet's tag is still
inserted: the same tag can only be the same packet, which would fail
identically.
129. [FROZEN] The relay runtime. handle_packet is pure: one u64 entropy
drives the jitter draw AND the forward pads (XOF-expanded — pads are
never verified, only fresh-looking filler). Terminal relays jitter
before delivery (every relay holds every packet); drops do not (nothing
leaves, nothing to shape). Misroutes (BadCapsule / CorruptPayload) drop
with a stat. run_relay transports (non-mix frames ignored — the relay
binary is not a validator); spawn_relay wires OS entropy. Cover:
cover_packet builds a real 5-hop packet with a Drop terminal and a
uniformly random payload (indistinguishable on the wire — the payload
region is stream-XORed per hop either way); emit_cover re-rolls seeded
paths whose first hop is the sender itself and reports actual sends.
The cover model (cover_model.rs): a bucket-count level tracker — Q32 EMA
(rate 1/16), predict = the rounded level, dummies_for(observed) =
min(cap, max(predict, floor) − observed) — an order-1 linear learner,
integer-exact, deterministic in its observation sequence; floor 1 /
cap 16 genesis-config; the bucket cadence (1 s) is the caller's.
Knowledge-layer class: local, non-canonical, never a consensus input
(§10.5).
130. [FROZEN] The schedule's clock and curves (§12.2; units per params):
one day = one epoch (86,400 s, asserted). Amounts are whole NERV; nano
conversion is exact ×10⁹ at the ledger. The five curves:
(a) linear-vesting — released(d) = 0 (d ≤ cliff), total (d ≥ term),
else floor(total·(d−cliff)/(term−cliff)); day emissions telescope exactly.
(b) subsidy-decline — discrete constant-then-linear decline: per-day
weight 2(T−Y) for i < Y and 2(T−i) for Y ≤ i < T (continuous at the
boundary — the weights at Y−1 and Y are equal); released(d) = floor(
total·W(d)/D) with D = (T−Y)(T+Y+1); released(T) = total exactly. At
genesis parameters year-one (day 360) ≈ 545.32M NERV — the WP's "~545M"
(the discrete triangle's half-day inflates D by (T−Y)/2 against the
continuous normalization; the discrete value is the frozen one).
(c) quarterly-geometric — quarter k (0-based) disburses at day (k+1)·Q
for numerator num^k·den^(Q−1−k) of D = (den^Q−num^Q)/(den−num); at
genesis (9/10, 20 quarters) D = 10²⁰−9²⁰ and the first quarter ≈ 102.46M
NERV; released(term) = total exactly.
(d/e) claim-window and milestone-window — not time-driven: the curve
exposes the window (open for 1 ≤ d ≤ window_days; genesis day 0 is
closed — nothing exists to claim) and burn_unclaimed; emission is
event-driven (claims; signed disbursements) recorded in the ledger and
bounded by window and total. §12.6's cumulative table is ILLUSTRATIVE
(P9: full-claim upper bounds vs realistic rates) and is not a
conformance target; the conformance targets are the exact per-bucket
values (the differential batteries and the T- and term-exactness pins).
131. [FROZEN] Schedule validation (the M1 precondition). from_params
parses per bucket; validate_allocation checks the allocation invariants
— Σ share_permille = 1000, Σ total_nerv = 10,000,000,000, unique names.
Per-bucket consistency: kinds/accounts from the frozen sets; linear:
cliff < term and (linear_days if present) = term − cliff; subsidy:
year_one < term; quarterly: quarters ∈ 1..=64, quarter_days ≥ 1,
term = quarters·quarter_days, 1 ≤ num < den, and den^quarters fits
u128 (checked); claim/milestone: window ∈ 1..=term; claim windows must
declare burn_unclaimed = true (§12.2: unclaimed burns at close);
milestone windows must not declare it. genesis() parses the compile-time
params::ECONOMY_BUCKETS static — total on the committed file (the
conformance crate re-runs both validations); BucketSchedule is
constructible only through from_params, so every instance is validated.
132. [FROZEN] The emission ledger (§12.3) and its key hierarchy.
Ledger keys: LEK = H("nerv.ek" ‖ LEK_seed) (an ML-DSA-65 keypair —
the identity IS the ability to sign); ledger accounts are `nerv.emission`
domain digests of H(LEK ‖ bucket name). Commitment-note secrets: the
identity is the SECRET CLAIM KEY ck = H("nerv.ck" ‖ seed) (256-bit,
uniform); the commitment is H("nerv.claim.commit" ‖ ck ‖ bucket ‖ amount
‖ blinding); the claim nullifier is H("nerv.claim.null" ‖ ck ‖ bucket)
— the nullifier key is the claim key itself, so every commitment-note
account is one secret with derived views (per-claim blinding is fresh;
the claim proof is knowledge of ck binding commitment and nullifier —
mirroring custody's erratum-72 pattern one level up). The [economy]
ledger's `nerv.claim` D-02 row is realized as this commit/nullifier
pair. Eligibility proofs (§12.2: "sybil-resistant attestation,
pre-launch") are EXTERNAL inputs — the ledger records the eligibility
digest (H("nerv.claim.elig" ‖ ck ‖ bucket ‖ amount)); construction is
the ceremony's, out of scope (P9 stated).
133. [FROZEN] Ledger state and its commitments. Every account — signed
or commitment-note — is a BTreeMap-keyed entry {committed_amount_nano,
spendable_nano} with `committed ≥ spendable`: spendable is granted by
crediting a signed credential or proving a note; committed tracks the
schedule's expectation (the M1 audit checks committed). `emission_root`
= BLAKE3("nerv.emission.root" ‖ epoch ‖ [[account_id ‖ committed ‖
spendable] in canonical account order]) — flat fixed-width framing
(order is the map order). Signed credits require a credential:
H("nerv.emission.cred" ‖ LEK ‖ bucket ‖ amount_nano ‖ epoch) signed by
the BEACON key (the deterministic schedule's execution is a beacon
duty — §12.3 "the beacon chain deterministically executes"); the
ledger's `credit` is the pure state transition and validates
epoch/schedule/rotation/eligibility (its window, its signature, its
exact amount). Window close burns = reducing `committed` by the
unclaimed remainder (spendable untouched) and recording the event in
the burn log. The escrow edge (fees → producer payouts, slashing) is
part 3's `staking.rs`/`fees.rs` — the payout accounts there credit
spendable against the schedule's subsidy stream through this same
`credit` with a producer-payout credential.
134. [FROZEN] The claim rail (§12.3, App A). A claim leg is an
ordinary `LegShell` on the claimant's home shard whose anchor is the
emission_root at its epoch (the EMISSION tree below), and whose
`nerv.claim.commit` commitment replaces the note commitment for inputs
— custody's nct.rs cannot carry it (it hashes a NoteOpening's fields;
the ledger's commitment is a different preimage), so the rail is
nerv-economy's own inclusion object: the EMISSION tree — a depth-32
append-only BLAKE3 Merkle tree over leaf = H("nerv.emission.leaf" ‖
account ‖ committed ‖ spendable) in account order. (An anchor
substitution — carrying the emission tree's root through the custody
NCT — was rejected: it rebinds custody's tree to foreign leaves and
breaks the executor's root checks; the account-commitment IS already
the right leaf type.) Claim-leg validation (the executor-side
rule set): (i) anchor is a finalized emission_root (within the
shard-state anchor window — verified by the state layer); (ii) the
claim-leg carries a claim witness {account, committed, spendable,
inclusion, nullifier, proof}; (iii) the nullifier is fresh (the
ledger's claim-nullifier set — its own BTreeSet, committed alongside
emission_root); (iv) the witness's spendable ≥ the leg's declared
input; (v) commitment-accounts must open their note (the claim proof
binds the ck, the commitment, and the nullifier — the E2E check is
H(nullifier-pair) equality and, for commitment accounts, a
discrete-log-free proof of knowledge is OUT OF SCOPE at the ledger
layer: the claim proof is the signed attestation the ceremony issued
(erratum 158) — the ledger's soundness for notes is the eligibility
digest's collision-resistance, stated honestly); (vi) settlement
marks the nullifier spent and transfers spendable out. Reverted claim
legs (the D.3 reversion of an unsatisfied claim) burn the escrowed
amount (committed reduced, spendable never granted) — the §4.5
abandonment case. The `ClaimWitness` carries the account coordinates
and the inclusion proof; `emission root continuity` (the beacon's
finalized-root chain) is chunk 14's attestation machinery — the rail
consumes roots, never chains them.
135. [FROZEN] Fees (§12.5, D.4). The split is exact with dust to the
producer: prover/DA/relay floor-divided at their permilles, producer =
fee − those three (Σ shares = 1000, const-asserted; no nano is lost or
minted). The D.4 admission floor: statistic = Σ_j |centerlift(Δ_B[j])|
over the header-carried 512-byte reveal (the L1 norm of the revealed
aggregate — the frozen codec's rail mixture, deterministic from public
data); trailing window W = 64 intervals (genesis-config); lower median;
trigger = statistic ≥ 4·median ∧ median > 0 (genesis-config); on
trigger m ← min(M_max, m+1), else m ← max(1, m·λ) with λ = 1/2 integer
(D.6) — hysteresis is the decay itself (M_max = 8 needs 3 quiet
intervals to return to 1; the attacker's "decay horizon"). floor =
base_floor·m (params: 1000 nano). EXECUTOR WIRING DEFERRED (the erratum-
69 precedent): "a block including a leg whose declared fee is below the
floor is invalid" is a real rule whose landing is bundled with the node
integration (chunk 20), because the delivered chunk-13/14 test fixtures
predate the floor (700-nano fees vs the 1000-nano genesis base) and
the correction must land as one coherent fixture+rule pass, not a
scattered rewrite. The rule, the floor state machine, and admits() are
complete and tested here.
136. [FROZEN] Subsidy (§12.3): per epoch (= day, one-to-one) the
validator-subsidy bucket's day_emission splits evenly across the active
shard set in canonical ShardId order; the remainder distributes one
nano each to the first shards in that order — exact, lossless,
deterministic. Shard payout accounts are credited through the emission
ledger's beacon-credential path (the bucket-total bound of erratum 139
governs: Σ shard credits for the epoch = the day emission).
137. [FROZEN] Staking (design doc's staking.rs; WP §2.5): a transparent
stake ledger keyed by ML-DSA verifying key. The slash table (genesis-
config, governance-adjustable): DoubleSign 100%, InvalidBlock 100%,
InvalidInclusion 25%, InvalidBundle 10% (permilles 1000/1000/250/100).
Slash consumes stake first, then pending withdrawals (unbonding does not
evade an offense that predates the boundary). Withdrawals are
epoch-boundary-effective: request at e, payable at finalize(e+1).
Evidence is verified by the node via nerv-consensus (chunk 14) — the
economy CANNOT depend on consensus (state→economy would cycle), so
staking consumes (class tag, evidence digest, offender) and the node
supplies the verified triple; the digest binds under the existing
nerv.slash domain via the caller.
138. [FROZEN] The supply ledger (§12.3 M1, §12.6): per-epoch emission
and burn accumulation with the identity supply = emitted − burned
(≥ 0 enforced — a negative is an accounting bug, an error at record
time). Burn categories: TransparentExit (custody burn commitments),
UnclaimedWindow (emission window-close burns), AbandonedIssue (the
state layer's D.3 abandonment). Burn records are reference-deduplicated
(idempotent recording; the reference is the category's native digest).
The publication: (epoch, emitted, per-category burned, supply). The M1
audit: recorded emissions equal the schedule's time-driven replay
(released_by_day) plus the ledger's event-driven grants — supplied as a
cross-check function the conformance job drives.
139. [FROZEN] Emission credit restructure (correcting Part 2): buckets
are MULTI-HOLDER (many contributors, 64 shard payouts) — the per-
credential amount == day_emission check is wrong. The ledger accumulates
per (bucket, epoch) and bounds Σ credits by the day emission (the
beacon attests the intra-bucket split; the ledger enforces the bucket
total — over-emission is impossible even with a malicious split).
Per-account per-epoch double-credit rejected. audit_day: per time-
driven bucket, Σ_{d≤day} credited == released_nano(day) — the schedule-
replay half of M1; per-epoch single-credit for the audit's replay
freshness.
140. [FROZEN] The forecaster architecture class — the instantiation of
§10.2's "linear autoregressive model" over "a window of the last 1,024
(Δ_B, time-bucket, fee-sum) triples": per output dimension j ∈ [0,64),
Δ̂[j] = wrap( c[j] + rne( (Σ_{k=1..1024} a[j][k]·centerlift(Δ_{t+1−k}[j])
+ b[j][k]·fee_{t+1−k} + d[j][k]·bucket_{t+1−k}) / 2¹⁵ ) ) — per-dimension
Q15 AR weights on the dimension's own lagged CENTERED deltas, plus
per-dimension per-lag Q15 weights on the lagged fee sums (raw u64, no
centering — the bias absorbs levels) and the lagged time-bucket indices
(the raw 16-bucket time-of-epoch index, matching features.rs's formula;
one-hot rejected as parameter bloat), plus the i64 bias c[j] and the u64
per-dimension scale s[j] — §10.3's "published per-dimension scales", the
Huber scoring parameter, never a prediction multiplier. K = 1,024 (the
design doc's "Linear AR(1024)"; the full window). Missing lags (a short
window) contribute zero. Accumulation is i128 (adversarial worst case
< 2⁹¹ ≪ 2¹²⁷ — total, no panics); the ONE named rounding point is the
round-half-even at 2⁻¹⁵; the result wraps mod 2⁶⁴. 196,736 learnable
parameters in the canonical order ar ‖ fee ‖ bucket ‖ bias ‖ scale
(§10.2's "canonical gradient ordering (sorted by parameter index)").
The reference parameterization (§7.6's W-epoch revert target): a[·][1..4]
= {1/2, 1/4, 1/8, 1/16} (a decaying filter; sum 15/16 — no unit root),
all else zero, zero bias, s[j] = 2²⁰ (genesis-config). The fee and bucket
channels exist because the WP's input triple lists them; the reference
parameterization leaves both at zero (the honest reading of a genesis
model that has learned nothing yet).
141. [FROZEN] The derived state and D_t. The forecaster STATE = (weights,
Adam moments (m: i128, v: u128 per learnable — storage here, the
transition is adam.rs's), step: u64); forecaster_state_root = BLAKE3(
"nerv.derived.fs" ‖ canonical(weights) ‖ m ‖ v ‖ step). The WINDOW is
EXCLUDED from the root — a pure function of the public reveal stream
that D_t's replay reconstructs anyway; the β^t bias-correction powers
are replay-local intermediates, never committed. Honest cost (P9): the
root's preimage is ≈ 6.7 MB per state; computing D_t hashes it once per
block per shard (advisory-layer only — light clients never touch D_t
(DSR-4); full nodes hash ~430 MB/s at 64 shards and 1 s blocks, inside
BLAKE3's single-core budget). D_t = BLAKE3("nerv.derived" ‖ e_t ‖ fs_root)
— the WP §4.2 formula governs; "the embedding, its full history, and the
forecaster's state are committed under D_t" (§7.5) is realized as: e_t
and fs_root in the digest, the history re-derivable from the header
chain's reveal stream (chunk 13's prev_reveal carries every Δ_B) — the
replay module's job. The embedding: e advances by wrapping addition per
reveal, contiguous per-block heights enforced (a gap must be an explicit
miss); the skip-and-carry rule (D.1(d)/§6.3.6): a missed reveal
advances the height, leaves e AND the forecaster window unchanged
(skipped — no observation exists), and is recorded {height, legs} in the
derived state — the carry IS the unchanged state. The chunk-9 ceremony's
MissedReveal maps to this record at the node's wiring boundary. The
W-epoch reset: weights ← reference, moments ← zero, window ← CLEARED
(the old window's deltas were computed under the previous W; warm-start
is post-boundary aggregates only — §7.6). nerv-knowledge depends only
on core + codec (DSR-1/DSR-2); this file is the deletion boundary's
content.
142. [FROZEN] The deterministic integer Adam (WP §10.2). Representation:
gradients in Q40 i128, clipped at entry to |g| ≤ 2^20 true (P8's
published gradient bound); m in Q40 i128, v in Q80 u128. β1 = 1−2^-4,
β2 = 1−2^-8 (powers of two — every constant exact); ε = 2^-40 (Q40 raw
1 — the fixed epsilon). m ← m + ((g−m) ≫ 4) (floor); v by the
two-branch unsigned floor (g² ≥ v: add the shifted excess; else
subtract — v stays within [min, max]). Bias corrections 1−β^t in Q64 by
binary exponentiation with floor-multiplies (β's Q64 form exact; bc > 0
for t ≥ 1, proven in tests); m̂ = m·2^64/bc1; √v̂ = isqrt(v)·2^32/
isqrt(bc2) — the named rounding points are the two isqrt floors and the
floor divisions (√(a/b) as isqrt(a)/isqrt(b)). ratio = m̂·2^40/(√v̂+ε),
clipped to ±2 (the update-norm clip). Application: δ = −round½even(
α·ratio/2^40), α_w = 16 (Q15 units), α_b = 2^20 (bias units), saturating
conversion into the parameter's range (P8). Totality on corrupt states
via saturating i128 multiplies — exact on the reachable set
(|m| ≤ 2^60, v ≤ 2^120). Newton isqrt (bit-length guess, monotone
descent, adjust-down), differentially tested against binary search.
143. [FROZEN] Huber scoring and the scale (WP §10.3). r_j = centerlift
(Δ_B[j] − Δ̂[j]) (the wrapping subtraction's signed reading). The loss is
the normalized per-dimension Huber L_j = s_j·ρ(r_j/s_j) (ρ quadratic to
|u| = 1, linear beyond), so the blame b_j = ρ'(u_j) ∈ [−1, 1] by
construction (bounded blame). b in Q40. The recorded score is Σ L_j in
Q32 (the loss actually minimized — the challenger basis). S_MAX = 2^40
( with S_MIN = 1) keeps every intermediate inside u128. The scale is NOT
Adam-trained — the normalized-Huber scale gradient is ρ(u)−u·ρ′(u) ≤ 0
always (inflating s lowers that loss; stated honestly) — it updates by
the robust EMA s ← clamp(s + ((|r|−s) ≫ 4), S_MIN, S_MAX) (rate 2^-4,
floor — the classical robust-scale tracker). forecaster.rs AMENDED:
LEARNABLE = 3·PER_CHANNEL + DIMS = 196,672 (the Adam order is ar ‖ fee
‖ bucket ‖ bias; the scale rides the weights' canonical bytes, outside
the moments). MOMENTS_CANONICAL_LEN = 6,293,512.
144. [FROZEN] The block loop and replay (WP §10.3). Per reveal:
commit (the pure prediction from the pre-block state; H(Δ̂) under
"nerv.derived.pred") → reveal → score (residual, blame, loss) →
gradients over the PRE-push window (−b·x/2^15 with round-half-even at
the scaling; −b·fee; −b·bucket; bias −b) → the Adam step → the scale
EMA → push the observation → e advances. The two-phase API is the
node's temporal discipline: commit() before decryption, reveal() after;
reveal re-derives and checks the commitment (a state move between them
is an error). The header-field landing of H(Δ̂) is the node's wiring
(chunk 20 — nerv-knowledge cannot depend on nerv-state, DSR-1); the
binding today is D_{t−1} + the deterministic replay. Misses (D.1(d)):
height advances, e and the forecaster carry UNCHANGED, no observation,
no update; the miss is publicly visible as the successor header's
prev_reveal = None (chunk 13's Option field) — the replay derives its
event stream from the header chain: Some(Δ) → Reveal, None → Miss;
fee-sum from header_t.fee_total; bucket from header_t.height by the
features.rs formula (E = 86,400). D_t per block = derived_root after
the event. Replay: fold the loop, compare each D_t against the
committed chain; mismatch = the advisory fault (WP §10.1 — public,
governance-slashable; custody untouched by construction).
145. [HYGIENE] The design doc's loop.rs lands as block_loop.rs (`loop`
is a Rust keyword; the module path carries the §10.3 name).
146. [FROZEN] The challenger market (§10.4). Registration = (account id,
frozen Weights within the published architecture class); each challenger
maintains its own shadow copy of the public observation window, driven by
the same BlockEvent stream as the incumbent. Commits are DETERMINISTIC
RECOMPUTATIONS recorded pre-reveal: commit(h) requires h == next_height
(the upcoming block), is idempotent, and hashes the prediction from the
pre-reveal window; the timing discipline is structural — no event can
intervene between commit(h) and reveal(h), because any intervening event
advances next_height past h. At reveal, the prediction is recomputed and
compared: a mismatched commit is dropped — no score, no coverage credit
(the honest-commit recomputation is public, so a garbage commit is
self-evident and simply worthless). Misses carry the state (no
observation, no score). Scoring: each forecaster's Huber score under its
OWN published scales — the incumbent's recorded loss_q32 (§10.3) and the
challenger's frozen scales are the same function. The fairness rule: BOTH
window totals sum over the challenger's covered block set (a smaller set
must not look better). The margin: challenger_total·1000 <
incumbent_total·(1000−50), strictly (5% genesis-config, governance-set).
Coverage: ≥ ⌈2016·900/1000⌉ = 1815 verified-scored blocks of the
2,016-block window. The gate outcome carries the challenger's weights
(for installation as incumbent at the next epoch boundary — §10.4; the
incumbent reverts to reference at every W-epoch anyway, §7.6) and the
SkillRecord — the useful-work pool's payout input (nerv-economy's chunk;
the wiring is the node's, chunk 20). forecaster::predict is extracted as
the single shared implementation (no incumbent/challenger drift).
147. [FROZEN] Anomaly flags (§10.5): per-dimension tail |r_j| ≥ 4·s_j
(genesis-config multiple), cascade at ≥ 32 of 64 dimensions (half), a
256-entry advisory ring, max |r|/s in Q40 carried per block. Consumed by
ops and metrics only — never consensus, never slashing (§2.5). The D.4
level-2 automatic response (the fee floor) is economy/fees' delivered
AdmissionFloor over the reveal statistic; the forecaster-relative flags
here are level-1 and stay advisory (§10.5's boundary, restated).
148. [FROZEN] The delete-test harness (Axiom 3 / DSR-1 as executable
checks; the conformance registry's delete_test — "harness lands with
chunk 17"). It lives in crates/nerv-knowledge/tests/ so deleting the
crate deletes the checks with it; the invariant they establish is what
the CI job re-verifies externally. (a) ALWAYS-RUNS — the manifest-graph
firewall, implemented by direct Cargo.toml parsing (the toml crate; no
nested cargo, no lock contention): nerv-knowledge's workspace-internal
build deps ⊆ {nerv-core, nerv-codec} (DSR-2); no authority member's
build closure — path-dependency edges through [dependencies],
[build-dependencies], and [target.*.dependencies] — reaches
nerv-knowledge (DSR-1; absent members like witness are skipped,
forward-compatible); and the strict scan: no member manifest mentions
nerv-knowledge AT ALL, dev-dependencies included — the delete job's
precondition. Diagnostic crates consume frozen vector DATA, never the
crate; generators live inside nerv-knowledge (the design's forcing
function). (b) #[ignore]d FULL JOB — copy the workspace minus
nerv-knowledge/target/.git, patch the members list, cargo check
--workspace --offline against a fresh CARGO_TARGET_DIR with a
30-minute kill-switch (nested-cargo deadlock avoidance: separate
workspace, separate target dir, complete lockfile travels so resolution
stays local). Green = Axiom 3's build half; the full-chain validation of
a seeded fixture is the testkit's (chunk 20, the job's second half).
149. [FROZEN] The inclusion witness (§11.2): the standard form is (txid,
leg, leaf position, leg-tree siblings, block locator {shard, height,
interval}, header hash) ≈ 530 B at the 10,000-leg cap; the cold form adds
the 𝔾 path (shard index + siblings) ≈ +320 B at 1,024 shards (depth 10).
The leg-tree root is a verification INPUT (from the block data or a
trusted full node) — the root is not header-committed (erratum 106);
the honest trust split: the 𝔾 path proves the header is finalized (a
cryptographic fact from the anchor), the leg-tree path proves inclusion
(given the root), and the root's binding to the header requires the
DA data (the full-node path, §11.4's "spot-check random shard blocks").
The portfolio: k legs of one block share the locator and header digest;
the cross-shard receipt: one witness per leg, bound by the shared txid
(§11.6). nerv-witness's dependency set (design-doc correction): core,
crypto, consensus, state — state's leg-tree types are required (no
duplication; the doc's list omitted state).
150. [FROZEN] The light anchor (§11.2, §11.4): the verifier's state is
an epoch-attestation chain (each epoch attestation's prev links to its
predecessor's digest — the walk-back-to-genesis path, ~50 KB/day of
QCs ≈ 18 MB/yr) plus the current epoch's interval attestations (the
forward chain, each linking to its predecessor's digest). The anchor
exposes the latest 𝔾 root; extension verifies each new attestation's QC
and chain link; the epoch boundary rolls the interval list into the
next epoch attestation. The B6 verification cost is the QC verification
(the initial sync's ~2 s of ML-DSA work); steady-state extension is one
QC per interval. The anchor does NOT track shard headers or DA data —
witnesses carry their own shard context.
151. [FROZEN] Archival regeneration (§11.6): from the DA-published
ShardBlock, resolve the legs, rebuild the BlockLegTree, and emit the
witness for any (txid, leg). The 𝔾 path is the CALLER's — it requires
the 𝔾 context (the anchor's root and the shard's index), not the block.
Loss is a non-event: any archival node regenerates any finalized
witness, forever (§11.6). The regenerated witness is bit-identical to
the originally generated one (same tree, same position, same siblings).
152. [FROZEN] The inclusion witness (§11.2): the standard form carries
(txid, leg, leaf position, leg-tree siblings, block locator, header
hash) ≈ 530 B at the 10,000-leg cap; the cold form adds the 𝔾 path
(shard index + siblings) ≈ +320 B at 1,024 shards. The leg-tree root
is a verification INPUT (from the block data or a trusted full node) —
it is not header-committed (erratum 106); the honest trust split: the 𝔾
path proves the header is finalized (a cryptographic fact from the
anchor), the leg-tree path proves inclusion (given the root), and the
root's binding to the header requires the DA data (the full-node
path, §11.4's spot-check). The portfolio: k legs of one block share
the locator and header digest. The cross-shard receipt: one witness
per leg, bound by the shared txid (§11.6). nerv-witness's dependency
set (design-doc correction): core, crypto, consensus, state — state's
leg-tree types are required; the doc's list omitted state.
153. [FROZEN] The light anchor (§11.2, §11.4): the verifier's state is
an epoch-attestation chain (each epoch attestation's prev links to
its predecessor's digest — the walk-back path, ~50 KB/day ≈ 18 MB/yr)
plus the current epoch's interval attestations (the forward chain).
The anchor exposes the latest 𝔾 root; extension verifies each new
attestation's QC and chain link; the epoch boundary rolls the interval
list into the next epoch attestation. The B6 cost is the QC
verification (~2 s initial sync); steady-state is one QC per interval.
154. [FROZEN] Archival regeneration (§11.6): from the DA-published
ShardBlock, resolve the legs, rebuild the BlockLegTree, and emit the
witness for any (txid, leg). The 𝔾 path is the CALLER's (it requires
the anchor context, not the block). The regenerated witness is
bit-identical to the original (same tree, same position, same
siblings). Loss is a non-event.
155. [FROZEN] The ballot's weight commitment and proof (§12.8). The
note-holder ballot carries (voting nullifier, choice, weight commitment,
sigma proof). The weight commitment is an Ajtai commitment over R_q^8
(reusing nerv-seal's CommitMatrix seeded from
H("nerv.ballot.matrix" ‖ referendum_id) — per-referendum matrices,
unlinkable across elections). The weight (u64 nano-NERV) is encoded as
4 limbs of 15 bits, each as a constant polynomial in positions 0–3 of
the value Vec8 (positions 4–7 zero). The blinding is 8 polynomials with
coefficients bounded by 2^10. The sigma proof (the same FS-with-aborts
engine as DKG/VPD, domain "nerv.ballot.proof") proves knowledge of the
opening with limb bounds 2^15 and blinding bounds 2^10 — both < 2^17,
the sigma feasibility limit. The tally opens each commitment (the
voter's separate WeightOpening submission at tally time), sums per
choice, and publishes only the aggregate: the aggregate-only property
is enforced by the RESULT TYPE (no individual weights in the published
tally) and the two-phase commit/reveal (weights hidden during voting,
revealed only to the tallier at count). The full ZK ballot (proving note
ownership AND weight correctness in one STARK) is the [NOVEL ★] M1
audit deliverable; the delivered sigma proof covers the weight
commitment's correctness, and the ownership is attested by the nullifier
derivation from the note's nullifier key.
156. [FROZEN] Chambers (§12.8, App C). The note-holder chamber: any
address holding ≥ 1 nano-NERV may cast one ballot per referendum
(the nullifier enforces one per (nk, referendum)). The validator
chamber: transparent ML-DSA-signed votes weighted by stake from the
staking ledger. The BOOTSTRAP RULE (§12.8): at mainnet the note-holder
chamber is empty (no notes exist); parameter governance rests with the
validator chamber under constitutional constraints until circulation
accumulates. Both-chamber thresholds (§C.2): majority = both chambers;
W-epoch = both + machine checks; constitutional = both, supermajority,
two consecutive epochs. Thresholds evaluated as ≥ (not >) of cast
weight (simple majority of participation, not of total supply — stated
honestly; low participation is the bootstrap risk the WP names).
157. [FROZEN] The tally (§12.8). Two phases: voting (ballots collected,
nullifiers checked, proofs verified — no weights opened) and reveal
(weight openings collected, verified against commitments, summed per
choice). The result carries only (yes, no, abstain) aggregate weights
and the ballot count — never individual weights. The validator tally
is transparent: sum of signed stakes per choice. Quorum: a referendum
is DECIDED only if the total cast weight ≥ the participation floor
(genesis-config: 1% of the chamber's estimated weight — a governance
parameter, not a constitutional one). nerv-governance's dependency
set (design-doc correction): core, crypto, seal, economy — the doc's
"custody, proofs" are consumed by the STARK-tier ballot (M1-gated),
not the delivered sigma tier.
158. [FROZEN] The referendum lifecycle (§12.8, §C.2; erratum 181).
Subject types: Parameter (a parameter-id/value pair), WEpoch (a codec
commitment + machine-check evidence), Constitutional (an amendment-id +
text hash). Tiers: Parameter (both-chamber majority), WEpoch (both
chambers + machine checks), Constitutional (both chambers, supermajority,
two consecutive 24-hour epochs — the confirmation mechanism: the first
referendum's passage at epoch E creates a confirmation referendum at
E+1 on the same subject; adoption requires both to pass). Lifecycle:
Draft → Active(epoch) → Closed(epoch, passed). The passage rule is
tier-specific: Parameter/WEpoch evaluate on the single referendum's
tally; Constitutional evaluates the PAIR. The W-epoch gate: the
referendum carries WEpochEvidence (w_commitment, spark_certified,
norms_certified, independence_certified) — all four must hold for the
gate to open; the actual verification is nerv-codec's certify function
(the evidence is its result, supplied at creation; the governance layer
checks completeness, never re-runs the certification — dependency
direction: governance does not import codec). The participation floor
(erratum 180's 1%) applies to every tier.
159. [FROZEN] Emergency powers (§C.2, §9.4, §8.5): ENUMERATED —
exactly three, no extensibility (the type system enforces the list).
Each is time-boxed: KillSwitch 1 epoch, HashWidening 1 epoch for the
decision (the effect, once executed, is permanent — the box is on the
ACTIVATION, not the widened hashes), AcceleratedSplit 1 epoch. Each
carries its evidence: KillSwitch the lattice-break report hash;
HashWidening the current and proposed digest widths (256 → 384);
AcceleratedSplit the target shard and the sustained-overload evidence
(2× ceiling). All expire automatically; the kill-switch is renewable
(a continuing lattice emergency extends it one epoch at a time);
widening and split are not (one decision, one execution). Passage:
both chambers, simple majority (the expedited path — §C.2's "time-boxed"
means the passage is faster, not the threshold lower), participation
floor applies. nerv-governance depends on: core, crypto, seal, economy
(no consensus/state — the emergency ledger is governance's own state,
its effects on the protocol are consumed by the node's wiring, chunk 20).
160. [DOMAIN] nerv.referendum (the referendum ID derivation —
H("nerv.referendum" ‖ subject ‖ tier ‖ creation_epoch ‖ nonce)).
161. [FROZEN] The diversified address set (§8.6): the wallet generates
addresses deterministically from its detection seed (sequential indices),
keeping the first address in each distinct shard until it has 8 (the
default; a genesis-config constant). The natural homing (κ-derived)
distributes addresses; the wallet's randomness (the seed) chooses the
particular set. The generation is deterministic: the same seed always
produces the same address set. The shard-coverage map is an index from
ShardId → address indices, built at generation and extended as new
shards are needed. The wallet publishes the address set (the ek values
only — the dk and nk never leave).
162. [FROZEN] The scan process: for each output in each settled leg of a
block, the wallet parses the sealed_note bytes as a SealedNote and
trial-decrypts against every delivery key whose address is homed to that
shard. A successful AEAD authentication (via trial_decrypt) identifies
the recipient and checks the commitment (the erratum-72 binding). The
scanned note records (NoteOpening, nullifier key, address index,
precomputed nullifier, received height, leaf position in the NCT). The
wallet's note set is keyed by the commitment; spending marks the
nullifier spent and records the spending txid. The wallet's LOCAL
nullifier set (prevent double-spending in its own constructions) is
separate from the chain's nullifier set.
163. [FROZEN] The diversified address set (§8.6): the wallet generates
addresses deterministically from its detection seed (sequential
indices), keeping the first address in each distinct shard until the
coverage target (8, genesis-config) is met or the generation cap (128,
the coupon-collector ceiling at 64 shards) is hit. The natural homing
(κ-derived from the delivery key) distributes addresses; the wallet's
seed chooses the particular set. The shard-coverage map is
ShardId → address indices, built at generation. The nk and dk never
leave the wallet; only the ek is published. ensure_shard generates
additional addresses until a target shard is covered.
164. [FROZEN] The scan process: for each output in each settled leg,
the wallet parses the sealed_note bytes as a SealedNote and
trial-decrypts against every delivery key whose address is homed to
that shard. A successful AEAD authentication identifies the recipient;
the commitment check (erratum 72's binding) is inside trial_decrypt.
The ScannedNote records (NoteOpening, nullifier key, address index,
precomputed nullifier, memo). The WalletNoteSet is keyed by commitment;
mark_spent records the nullifier and spending txid; the wallet's LOCAL
nullifier set is separate from the chain's. The NCT position (leaf
index, anchor, siblings, shard, height) is recorded alongside for the
prover's membership witness.
165. [FROZEN] The diversified address set (§8.6): the wallet generates
addresses deterministically from its detection seed (sequential
indices), keeping the first address in each distinct shard until the
coverage target (8, genesis-config) is met or the generation cap (128,
the coupon-collector ceiling at 64 shards) is hit. The natural homing
(κ-derived from the delivery key) distributes addresses; the wallet's
seed chooses the particular set. The shard-coverage map is
ShardId → address indices, built at generation. The nk and dk never
leave the wallet; only the ek is published. ensure_shard generates
additional addresses until a target shard is covered.
166. [FROZEN] The scan process: for each output in each settled leg,
the wallet parses the sealed_note bytes as a SealedNote and
trial-decrypts against every delivery key whose address is homed to
that shard. A successful AEAD authentication identifies the recipient;
the commitment check (erratum 72's binding) is inside trial_decrypt.
The ScannedNote records (NoteOpening, nullifier key, address index,
precomputed nullifier, memo). The WalletNoteSet is keyed by commitment;
mark_spent records the nullifier and spending txid; the wallet's LOCAL
nullifier set is separate from the chain's. The NCT position (leaf
index, anchor, siblings, shard, height) is recorded alongside for the
prover's membership witness.
167. [FROZEN] The send pipeline (§2.3, §5.5, D.2). The ProvedTx's
submission payload (SubmissionMessage::Transaction encoding) is
fragmented per D.2 into {1,2,4,8,16} FragmentFrames; each
FragmentFrame's canonical encoding (FRAG_HEADER + 14,113 B = 14,153 B
exactly) is the Sphinx payload of one mixnet packet. For each of the 3
aggregators (params' fan-out), the wallet selects a 5-relay path from
the relay registry, builds the Sphinx packets, and sends them through
the host to the first relay. Different fragments may take different
paths (additional privacy; the default sends all fragments of one
submission on the same path — path-per-fragment is a wallet policy
toggle). Fee bucketing: optional, rounds the declared fee to the
nearest coarse bucket (powers of 2 × base) to remove the selection
side-channel.
168. [FROZEN] Issue-leg completion (§4.5, App A). The wallet's
complete module: after the spend leg's transit entry is beacon-
finalized, the wallet (or a relayer) constructs the issue-leg
settlement witness (the transit membership proof against the
beacon-finalized transit root) and submits the issue leg to the
receiving shard's producer. The wallet's API: (a) check_finalized
(queries the beacon view for the spend leg's transit entry at the
root height), (b) build_witness (constructs the TransitEvidence from
the shard's transit log and the finalized root), (c) the issue-leg
submission (through the same mixnet path as send). The relayer
market competes on this: the receiving shard accepts the issue leg
from any submitter; the wallet's own completion is the self-service
path (§4.5's "permissionlessly completable").
169. [FROZEN] Rehome (§8.5): a self-spend transaction where the wallet
spends its notes on shard A and creates outputs on shard B — the
standard re-homing instrument. The construct module handles the
assembly (it's a cross-shard payment where sender = recipient); the
rehome module wraps it with the wallet's own addresses and returns
the new witnesses for the receiving shard.
170. [FROZEN] The CLI's command surface (design doc: "Wallet ops;
witness regen; epoch-cert verify; supply audit; relayer mode; vector
generation"). Commands: keys (generate / addresses), tx (build /
prove), witness (regen), cert (verify), supply (audit), vectors
(emission-schedule / seal-decode). The wallet's seed travels as a
hex-encoded 32-byte string on the command line or in a file (the BIP-39
phrase is never stored; the master seed is the root). The tx build
command outputs the ConstructedTx's canonical encoding; tx prove reads
it, runs the full STARK pipeline, and outputs the ProvedTx. Commands
requiring a live node (live balance scanning, mixnet submission) carry
their complete argument surface and return a "requires a running node"
error until chunk 20 wires the host + gossip + submission into the CLI's
event loop. The relayer mode (the relay runtime over a bound host) is
chunk 20's nerv-relay binary — the CLI's relay command starts the same
engine through nerv-net's spawn_relay.
171. [FROZEN] Vector generation: the `vectors` command emits the
conformance vector families the conformance crate pins. The
emission-schedule family outputs the full schedule replay (every
bucket's released_by_day at every day boundary from 0 to term + 1),
BLAKE3-hashed for compact pinning. The seal-decode family outputs
digitize/resolve round-trip pairs for deterministic delta vectors.
Generation is seeded and deterministic: the same seed produces the same
vectors bit-for-bit. The output is a binary file (the canonical encoding
of the vector set); the conformance crate re-derives and compares.
172. [FROZEN] The CLI command surface (design doc; erratum 191):
keys (generate / addresses), tx (build / prove — deferred to chunk 20's
node wiring; the offline pipeline requires the wallet's scanned-note
state which travels through the node's RPC, not CLI arguments),
witness (regen from an archival block file), cert (verify an epoch
attestation chain), supply (audit the emission schedule), vectors
(emission-schedule / seal-decode). The wallet's seed travels as a
hex-encoded 32-byte string. The tx commands carry their full argument
surface and return a "requires the node's wallet RPC" error — the
honest boundary, not a placeholder. The relayer mode is the
nerv-relay binary (chunk 20); the CLI does not duplicate it.
173. [FROZEN] Vector generation (erratum 192): the emission-schedule
family outputs every bucket's released_by_day at boundary days (0,
1, each cliff, each cliff+1, each term, max_term, max_term+1) as
(u32 day, u64 released) pairs, BLAKE3-hashed for compact pinning.
The seal-decode family outputs deterministic digitize→resolve
round-trip pairs ((64×u64 input, 64×u64 resolved) per vector). Both
are seeded and reproducible bit-for-bit.
174. [FROZEN] The deterministic clock: a logical nanosecond counter
advanced ONLY by the scheduler's tick(). No wall-clock reads anywhere
in the testkit. Timeouts, jitter windows, and epoch boundaries are all
expressed as logical-time comparisons. The clock implements the
BeaconView's height for anchor freshness in the harness context.
175. [FROZEN] The adversarial scheduler (T5–T7; WP §13.4). The
harness's scheduler owns a loopback transport (in-memory bounded queues
between named endpoints) and a step-driven execution model. Each step:
(1) the scheduler selects a message from the ready set according to
the active schedule (the strategy), (2) delivers it to the target
node's inbox, (3) the node processes it and may emit messages back
into the transport. The schedule strategies: InOrder (FIFO — the
honest baseline), Reorder (swap adjacent messages — T6's adversarial
delays), Drop (censor selected sources — T7's liveness), Delay (hold
messages for N logical ticks then release — T5's expiry boundary), and
RoundRobin (deterministic interleaving). The TLA+ corpus instantiates
these: T5 = Delay(issue_leg, past_expiry) + verify_reversion;
T6 = Reorder + verify_completion_within_bound; T7 = Drop(producer) +
verify_issue_leg_completable_by_anyone. The corpus is data: a fixed
list of (strategy, seed, assertion) triples, executed by one runner.
176. [FROZEN] The node's wiring (DSR-9): the event loop is a
tokio::select over HostEvents, dispatching HostFrames to the gossip
engine, the executor's block assembly, or the relay's mix handler
depending on role. The node maintains its own BeaconView (the
finalized tau roots and transit roots it has observed through the
gossip header stream). serde is used ONLY in the binary for config
parsing — never on the consensus path (DSR-2's review check; the
design doc permits it on CLI surfaces). The node's data directory
holds the RocksDB store (DSR-10's store seam). The executor is
driven by the block-assembly path: when the node has (a) a
QC-validated header from gossip, (b) the block data from DA, and
(c) the current ShardState, it calls apply_block. DA sampling runs
as a background task with its own interval timer.
177. [FROZEN] The relay binary: bind, register (sign a RelayRecord
and submit to the on-chain registry via gossip), then run the relay
engine's event loop — HostEvent::Frame → RelayEngine::handle_packet
→ the jittered forward. The relay does NOT participate in consensus,
gossip, or DA; it is a pure mixnet node. The cover model runs as a
periodic background task (every 1 s of wall time, the cover model's
bucket cadence).
178. [FROZEN] The aggregator binary: bind, listen for
SubmissionMessages, run the verification gate (mempool::admit), and
when the mempool reaches a threshold or a timer fires, build a
Bundle (sign the txid root), and submit it to the registry committee
via the host mesh. The aggregator does NOT run the executor, DA, or
gossip engine; it is a pure prover-market node.
179. [FROZEN] The fuzz targets (design doc: fuzz/; erratum 198): five
cargo-fuzz targets, each a standalone no_main binary in its own
workspace (fuzz/Cargo.toml declares [workspace] so the parent
workspace ignores it). codec: decode arbitrary bytes as every core
type (no panic, only Err). sphinx_peel: arbitrary Packet bytes through
peel (a real DecapsulationKey; wrong keys are the interesting case).
seal_digitize: arbitrary u64[64] through digitize→resolve (the
round-trip identity). ring_ntt: arbitrary coefficient arrays through
ntt→intt (exact round-trip). fold_dedup: arbitrary txid orderings
through the IntervalLedger (first-wins determinism). Every target is
run by `cargo fuzz run <name>`; CI integrates them as a separate job.
180. [FROZEN] The conformance integration tests (design doc:
nerv-conformance; erratum 199): the cross-crate integration test
suite for the chunks 13–19 surface — (a) the seeded-chain test: a
state → propose → apply → verify round-trip through the real executor
and proposer; (b) the registry flow: mempool → bundle → interval →
commit; (c) the emission schedule → ledger → audit identity; (d) the
gossip ordering rule over the testkit's scheduler; (e) the D.4 fee
floor's interaction with the reveal statistic; (f) the challenger
market's gate over a real forecaster pipeline. These tests are the
ones the CI conformance job runs on every PR; the #[ignore] full
tests run on --release.
281. [FROZEN] The wallet state machine: an Elm-architecture (MVVM)
pattern — a pure `update(state, action) → (state', events)` function
that every platform's UI calls. The state carries no platform types,
no async, no I/O. Platform shells (terminal, desktop, web, mobile)
provide storage, networking, clipboard, and notifications through
adapters the state machine never sees. The draft builder (the send
flow's in-progress transaction) carries validation state so the UI
can render inline errors without side effects. The action vocabulary
is exhaustive: every user intent in the wallet is one enum variant,
and the update function is the single place where business rules are
enforced (the same rules on every platform, bit-for-bit).
182. [FROZEN] The design system: a single theme module defining
colors, spacing, and typography used by both the TUI and the GUI.
GitHub's dark palette (tested across millions of developer-hours):
background #0D1117, surface #161B22, border #30363D, text #E6EDF3,
muted #8B949E, accent #58A6FF, success #3FB950, warning #D29922,
error #F85149. Monospace throughout (a crypto wallet displays
hex, amounts, and addresses — monospace is the correct choice).
183. [FROZEN] The egui architecture: one `NervApp` struct implementing
`eframe::App`, with the `nerv-wallet-core` state machine as its only
state. Every user interaction becomes a `WalletAction` dispatched
through `update()`. The desktop entry (main.rs) calls
`eframe::run_native`; the web entry (lib.rs) calls
`eframe::WebRunner`. The same `App` struct serves both — the platform
shim is eframe itself. The GUI crate depends on nerv-wallet-core
(the pure state machine) and nerv-wallet (the domain crate) only for
the native target; the WASM build uses the state machine with the
platform shell delegating the send pipeline to a connected node (the
production-correct architecture for a web wallet: STARK proving
belongs on the node, not in the browser).
184. [FROZEN] The UI layout: a left navigation rail (icons + labels),
a top bar (brand + sync indicator), a bottom status bar (peers,
height, notification count), and a central content area. The theme
is GitHub-dark (erratum 201) rendered through egui's `Visuals`.
Typography: system monospace for all data (amounts, addresses, hashes);
system sans-serif for headings and body. All interactive elements
have keyboard focus; the entire wallet is operable without a mouse.
185. [FROZEN] Secure seed storage. The seed is NEVER stored in
plaintext on any platform. The WalletStorage trait (in wallet-core)
defines the interface: store_seed(seed, password), load_seed(password),
delete_seed(). Every implementation encrypts the seed with a
password-derived key before storage: key = BLAKE3-KDF("nerv.storage",
password, salt), ciphertext = ChaCha20-Poly1305(key, nonce, seed).
The native implementation prefers the OS keychain (macOS Keychain,
Linux secret-service, Windows Credential Manager via the keyring crate)
and falls back to an encrypted file. The WASM implementation uses
encrypted localStorage (the password encryption is the only protection
in the browser — stated honestly). The mobile shell delegates to the
WASM implementation (the PWA in a WebView). The encrypted format:
[salt 32B][nonce 12B][ciphertext 48B] = 92 bytes, base64-encoded
where the transport layer requires text (keyring, localStorage).
186. [FROZEN] Mobile approach: thin native shells wrapping the PWA
(the MetaMask pattern), not native eframe. iOS: a SwiftUI app with
a WKWebView loading the WASM bundle. Android: a Jetpack Compose app
with a WebView loading the same bundle. Benefits: the Rust codebase
is 100% shared (no mobile-specific Rust compilation targets), the
UI is identical across web and mobile, and the app-store distribution
path is standard. The native shells handle: the status bar, safe-area
insets, the back button (Android), biometric unlock (Face ID /
fingerprint), and deep links. Everything inside the WebView is the
same PWA that runs in a desktop browser.
187. [FROZEN] The integration test suite: the wallet-core state
machine is tested against exhaustive invariants (every action on
every reachable state produces a valid successor — no panics, no
invalid states), the draft validation logic is property-tested
(garbage inputs never produce a "ready" draft), and the storage
layer is round-trip tested (encrypt → decrypt under the correct
password, and failure under every wrong password). The Makefile
integrates everything: workspace tests, the firewall test, the WASM
build, mobile builds, and Docker deployment in one `make` invocation.
188. [FROZEN] The block distribution model for testnet: the producer
broadcasts the full ShardBlock encoding via a new gossip message type
(tag 6, BlockData), gated by the same header-before-data ordering rule
as partials (DSR-8). Full nodes receive the BlockData, decode the
ShardBlock, and call apply_block against their tracked ShardState. The
node buffers out-of-order blocks in a pending map keyed by height;
when the missing predecessor arrives, the pipeline drains in order.
The ChainSource for escrow recovery is a MemoryChain (a BTreeMap of
height → resolved legs from applied blocks). For the first testnet
iteration, block distribution is via gossip broadcast (O(n²) traffic
across n nodes — acceptable for testnet); the DA erasure-coding layer
(Gap 4) provides the production path and the light-client sampling.
QC verification is deferred to Gap 2 (epoch parameter tracking); the
block's internal consistency is enforced by apply_block (the header
must chain to the current C_t, the roots must match, the legs must
be settlement-valid).






































































