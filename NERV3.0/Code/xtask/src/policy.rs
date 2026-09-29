//! The workspace dependency policy — the design document's per-crate
//! "Depends on:" lines, as code. The firewall enforces this matrix for every
//! member that exists; membership grows chunk by chunk, so the geography is
//! enforced from the day each crate lands.
//!
//! Addition to the matrix is a deliberate, reviewed act: a member not listed
//! here fails the build with instructions to add it.

/// (crate, allowed internal dependencies, rationale/design reference)
pub const POLICY: &[(&str, &[&str], &str)] = &[
    ("nerv-core", &[], "foundation; external crates only"),
    ("nerv-crypto", &["nerv-core"], "DSR-3: the only door to raw primitives"),
    ("nerv-custody", &["nerv-core", "nerv-crypto"], "WP §3; nothing else"),
    ("nerv-codec", &["nerv-core"], "WP §7.2; canonical, frozen"),
    ("nerv-seal", &["nerv-core", "nerv-crypto"], "WP §6.3; two novel surfaces isolated"),
    (
        "nerv-proofs",
        &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-seal", "nerv-codec"],
        "WP §5; depends on custody, seal, codec",
    ),
    (
        "nerv-state",
        &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-registry", "nerv-economy"],
        "WP §4; executor + C_t",
    ),
    (
        "nerv-registry",
        &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-proofs"],
        "WP §5.5",
    ),
    (
        "nerv-consensus",
        &["nerv-core", "nerv-crypto", "nerv-state", "nerv-registry", "nerv-economy"],
        "WP §4.6–4.7, §8.3",
    ),
    ("nerv-da", &["nerv-core", "nerv-crypto"], "WP §8.7"),
    ("nerv-net", &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-registry"], "WP §6.2"),
    (
        "nerv-knowledge",
        &["nerv-core", "nerv-codec"],
        "WP §10.1: NOTHING depends on it; it depends ONLY on core + codec (DSR-2)",
    ),
    ("nerv-economy", &["nerv-core", "nerv-crypto", "nerv-custody"], "WP §12"),
    (
        "nerv-governance",
        &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-seal", "nerv-proofs", "nerv-economy"],
        "WP §12.8, App C",
    ),
    ("nerv-witness", &["nerv-core", "nerv-crypto", "nerv-consensus"], "WP §11"),
    (
        "nerv-wallet",
        &[
            "nerv-core", "nerv-crypto", "nerv-custody", "nerv-seal", "nerv-proofs",
            "nerv-codec", "nerv-net", "nerv-witness", "nerv-economy",
        ],
        "WP App A",
    ),
    (
        "nerv-node",
        &[
            "nerv-core", "nerv-crypto", "nerv-custody", "nerv-codec", "nerv-seal",
            "nerv-proofs", "nerv-state", "nerv-registry", "nerv-consensus", "nerv-da",
            "nerv-net", "nerv-economy", "nerv-governance", "nerv-witness", "nerv-knowledge",
        ],
        "full node; knowledge allowed only behind the advisory (non-default) feature — DSR-1",
    ),
    ("nerv-relay", &["nerv-core", "nerv-crypto", "nerv-net"], "mixnet relay binary"),
    (
        "nerv-aggregator",
        &["nerv-core", "nerv-crypto", "nerv-custody", "nerv-proofs", "nerv-registry", "nerv-net"],
        "prover-market node",
    ),
    (
        "nerv-cli",
        &[
            "nerv-core", "nerv-crypto", "nerv-custody", "nerv-codec", "nerv-seal",
            "nerv-proofs", "nerv-state", "nerv-registry", "nerv-consensus", "nerv-da",
            "nerv-net", "nerv-economy", "nerv-governance", "nerv-witness", "nerv-wallet",
            "nerv-knowledge",
        ],
        "operator surface, incl. advisory (relayer/metrics) modes",
    ),
    (
        "nerv-testkit",
        &[
            "nerv-core", "nerv-crypto", "nerv-custody", "nerv-codec", "nerv-seal",
            "nerv-proofs", "nerv-state", "nerv-registry", "nerv-consensus", "nerv-da",
            "nerv-net", "nerv-economy", "nerv-governance", "nerv-witness", "nerv-wallet",
            "nerv-knowledge",
        ],
        "in-process multinode harness; adversarial coverage of the knowledge layer's public surface",
    ),
    (
        "nerv-conformance",
        &[
            "nerv-core", "nerv-crypto", "nerv-custody", "nerv-codec", "nerv-seal",
            "nerv-proofs", "nerv-economy", "nerv-knowledge",
        ],
        "vector generators for their owning families; test-side, never authority",
    ),
    ("xtask", &["nerv-conformance"], "policy enforcement binary"),
];

/// Crates that may depend on `nerv-knowledge`. Advisory/test consumers only.
/// Every authority crate in STRICT_AUTHORITY is structurally excluded.
pub const KNOWLEDGE_DEPENDENTS_ALLOWLIST: &[&str] =
    &["nerv-node", "nerv-cli", "nerv-testkit", "nerv-conformance"];

/// The DSR-1 authority set: these crates must never reach `nerv-knowledge`,
/// transitively or directly. (core and crypto are below everything; included.)
pub const STRICT_AUTHORITY: &[&str] = &[
    "nerv-core", "nerv-crypto", "nerv-custody", "nerv-codec", "nerv-seal", "nerv-proofs",
    "nerv-state", "nerv-registry", "nerv-consensus", "nerv-da", "nerv-net", "nerv-economy",
    "nerv-witness",
];

pub fn allowed_deps(name: &str) -> Option<&'static [&'static str]> {
    POLICY.iter().find(|(n, _, _)| *n == name).map(|(_, d, _)| *d)
}

