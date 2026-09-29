//! nerv-core build script: params codegen — the "one source, zero drift" machine.
//!
//! WHAT THIS DOES
//!   Parses `specs/params.toml` (the single source of truth per the design
//!   document's `specs/` contract) and generates the `nerv_core::params`
//!   module: typed constants for every scalar leaf; static tables for the two
//!   structured families (`economy.buckets`, `errata`); a `PARAMS_MAP` leaf
//!   inventory for the conformance zero-drift check; the raw-file BLAKE3
//!   digest; and a minimal set of compile-time asserts.
//!
//! WHAT THIS DELIBERATELY DOES NOT DO
//!   * No semantic validation beyond typing: floats, datetimes, and negative
//!     integers are hard errors (P5 — WP §7.2), and the emitted asserts are a
//!     backstop. The semantic authority (quorum algebra, NTT primality of the
//!     seal modulus, noise headroom, emission totals, spark bound) is
//!     `nerv-conformance::spec`, run in CI — one validator, not two.
//!   * `toml` (which pulls serde internally) is a BUILD-time dependency only;
//!     the runtime consensus path of nerv-core is blake3 + thiserror
//!     (DSR-2: no serde on the consensus path — build-dependencies never
//!     enter the runtime graph the firewall checks).
//!
//! DETERMINISM
//!   Sections, keys, and map entries are emitted in sorted order; the output
//!   is byte-stable for a given params.toml.
//!
//! STRUCTURED FAMILIES
//!   A new array-of-tables family in params.toml is a deliberate act: this
//!   file rejects unknown families with instructions, because a structured
//!   family needs (a) a pinned schema here and (b) a typed validator in
//!   nerv-conformance::spec. Nothing grows silently.

use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| fail("CARGO_MANIFEST_DIR is not set")),
    );
    let spec_path = manifest_dir.join("..").join("..").join("specs").join("params.toml");
    println!("cargo:rerun-if-changed={}", spec_path.display());

    let text = fs::read_to_string(&spec_path)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", spec_path.display())));
    let digest: [u8; 32] = *blake3::hash(text.as_bytes()).as_bytes();
    let tree: toml::Value = toml::from_str(&text)
        .unwrap_or_else(|e| fail(&format!("specs/params.toml: parse error: {e}")));

    let generated = match generate(&tree, &digest) {
        Ok(g) => g,
        Err(problems) => fail(&format!(
            "specs/params.toml rejected with {} problem(s):\n  - {}",
            problems.len(),
            problems.join("\n  - ")
        )),
    };

    let out_dir =
        PathBuf::from(env::var("OUT_DIR").unwrap_or_else(|_| fail("OUT_DIR is not set")));
    let dest = out_dir.join("params.rs");
    fs::write(&dest, generated)
        .unwrap_or_else(|e| fail(&format!("cannot write {}: {e}", dest.display())));
}

fn fail(msg: &str) -> ! {
    eprintln!("nerv-core/build.rs: {msg}");
    process::exit(1);
}

// ---------------------------------------------------------------------------
// Schemas for the structured (array-of-tables) families.
// ---------------------------------------------------------------------------

enum FieldType {
    Str,
    Int,
    OptInt,
    OptBool,
}

struct FieldDef {
    name: &'static str,
    ty: FieldType,
}

fn field_rust_type(t: &FieldType) -> &'static str {
    match t {
        FieldType::Str => "&'static str",
        FieldType::Int => "u64",
        FieldType::OptInt => "Option<u64>",
        FieldType::OptBool => "Option<bool>",
    }
}

fn field_type_name(t: &FieldType) -> &'static str {
    match t {
        FieldType::Str => "a string",
        FieldType::Int => "a non-negative integer",
        FieldType::OptInt => "an optional non-negative integer",
        FieldType::OptBool => "an optional boolean",
    }
}

fn toml_type_name(v: &toml::Value) -> &'static str {
    match v {
        toml::Value::String(_) => "a string",
        toml::Value::Integer(_) => "an integer",
        toml::Value::Float(_) => "a float",
        toml::Value::Boolean(_) => "a boolean",
        toml::Value::Datetime(_) => "a datetime",
        toml::Value::Array(_) => "an array",
        toml::Value::Table(_) => "a table",
    }
}

/// `economy.buckets` — WP §12.2 allocation buckets.
const BUCKET_SCHEMA: &[FieldDef] = &[
    FieldDef { name: "name", ty: FieldType::Str },
    FieldDef { name: "kind", ty: FieldType::Str },
    FieldDef { name: "total_nerv", ty: FieldType::Int },
    FieldDef { name: "share_permille", ty: FieldType::Int },
    FieldDef { name: "account", ty: FieldType::Str },
    FieldDef { name: "term_days", ty: FieldType::Int },
    FieldDef { name: "cliff_days", ty: FieldType::OptInt },
    FieldDef { name: "linear_days", ty: FieldType::OptInt },
    FieldDef { name: "year_one_days", ty: FieldType::OptInt },
    FieldDef { name: "quarters", ty: FieldType::OptInt },
    FieldDef { name: "quarter_days", ty: FieldType::OptInt },
    FieldDef { name: "ratio_num", ty: FieldType::OptInt },
    FieldDef { name: "ratio_den", ty: FieldType::OptInt },
    FieldDef { name: "window_days", ty: FieldType::OptInt },
    FieldDef { name: "burn_unclaimed", ty: FieldType::OptBool },
];

/// `errata` — the machine-hashed erratum register (DSR-5 discipline).
const ERRATA_SCHEMA: &[FieldDef] = &[
    FieldDef { name: "id", ty: FieldType::Str },
    FieldDef { name: "refs", ty: FieldType::Str },
    FieldDef { name: "issue", ty: FieldType::Str },
    FieldDef { name: "resolution", ty: FieldType::Str },
];

// ---------------------------------------------------------------------------
// Generation.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Val {
    U(u64),
    B(bool),
    S(String),
}

impl Val {
    fn map_expr(&self) -> String {
        match self {
            Val::U(v) => format!("ParamValue::U64({v}u64)"),
            Val::B(b) => format!("ParamValue::Bool({b})"),
            Val::S(s) => format!("ParamValue::Str({s:?})"),
        }
    }
}

struct Gen {
    out: String,
    map: Vec<(String, Val)>,
    names: BTreeSet<String>,
    problems: Vec<String>,
}

fn generate(tree: &toml::Value, digest: &[u8; 32]) -> Result<String, Vec<String>> {
    let mut g = Gen {
        out: String::new(),
        map: Vec::new(),
        names: BTreeSet::new(),
        problems: Vec::new(),
    };
    g.header(digest);

    let toml::Value::Table(root) = tree else {
        return Err(vec!["root of params.toml must be a table".into()]);
    };
    for (key, val) in sorted_entries(root) {
        g.check_ident(key, key);
        match val {
            toml::Value::Table(sub) => g.walk_table(key, sub),
            toml::Value::Array(arr) => g.walk_array(key, arr),
            leaf => g.scalar(key, leaf),
        }
    }
    g.finish();

    if g.problems.is_empty() {
        Ok(g.out)
    } else {
        Err(g.problems)
    }
}

fn sorted_entries(t: &toml::map::Map<String, toml::Value>) -> Vec<(&str, &toml::Value)> {
    let mut v: Vec<(&str, &toml::Value)> = t.iter().map(|(k, x)| (k.as_str(), x)).collect();
    v.sort_unstable_by(|a, b| a.0.cmp(b.0));
    v
}

impl Gen {
    fn header(&mut self, digest: &[u8; 32]) {
        self.out.push_str(
            "// GENERATED by nerv-core/build.rs from specs/params.toml — DO NOT EDIT.\n\
             //\n\
             // Single source of truth: specs/params.toml (regenerated on every build;\n\
             // output is byte-stable for a given input). Zero drift: nerv-conformance\n\
             // (CI) compares PARAMS_MAP below against a fresh parse of the live file\n\
             // and runs the full semantic validator; the compile-time asserts at the\n\
             // bottom of this file are a backstop, not the validator. P5: floats,\n\
             // datetimes, and negative integers are rejected at generation.\n\n",
        );
        self.out
            .push_str(&format!("pub const PARAMS_TOML_DIGEST: [u8; 32] = {digest:?};\n\n"));
    }

    fn check_ident(&mut self, key: &str, path: &str) {
        let valid = !key.is_empty()
            && key.chars().next().is_some_and(|c| c.is_ascii_lowercase())
            && key.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !valid {
            self.problems
                .push(format!("invalid key `{key}` at `{path}` — keys must be lower_snake_case"));
        }
    }

    fn walk_table(&mut self, path: &str, t: &toml::map::Map<String, toml::Value>) {
        for (key, val) in sorted_entries(t) {
            let child = format!("{path}.{key}");
            self.check_ident(key, &child);
            match val {
                toml::Value::Table(sub) => self.walk_table(&child, sub),
                toml::Value::Array(arr) => self.walk_array(&child, arr),
                leaf => self.scalar(&child, leaf),
            }
        }
    }

    fn scalar(&mut self, path: &str, v: &toml::Value) {
        let val = match v {
            toml::Value::Integer(i) => {
                if *i < 0 {
                    self.problems
                        .push(format!("P5/type: negative integer {i} at `{path}`"));
                    return;
                }
                Val::U(*i as u64)
            }
            toml::Value::Boolean(b) => Val::B(*b),
            toml::Value::String(s) => {
                // Spec values that exceed `i64::MAX` (e.g., the
                // Goldilocks field modulus `2^64 − 2^32 + 1` =
                // 18_446_744_069_414_584_321) are quoted as strings
                // because the `toml` crate's default deserializer uses
                // `i64` and would otherwise fail at parse time. We
                // try to parse the string as `u64` here so the
                // rest of the pipeline treats them identically.
                match s.parse::<u64>() {
                    Ok(n) => Val::U(n),
                    Err(_) => Val::S(s.clone()),
                }
            }
            toml::Value::Float(_) | toml::Value::Datetime(_) => {
                self.problems.push(format!(
                    "P5: float/datetime at `{path}` — the spec is integer-exact (WP §7.2)"
                ));
                return;
            }
            toml::Value::Array(_) | toml::Value::Table(_) => {
                self.problems
                    .push(format!("internal: scalar() called on a container at `{path}`"));
                return;
            }
        };
        self.push_leaf(path, val);
    }

    fn push_leaf(&mut self, path: &str, val: Val) {
        self.map.push((path.to_string(), val.clone()));
        if let Some(name) = self.const_name_for(path) {
            let decl = match &val {
                Val::U(v) => format!("pub const {name}: u64 = {v}u64;\n"),
                Val::B(b) => format!("pub const {name}: bool = {b};\n"),
                Val::S(s) => format!("pub const {name}: &'static str = {s:?};\n"),
            };
            self.out.push_str(&decl);
        }
    }

    /// Reserve a const name for `path` (dotted, no array indices). `None`
    /// means: not const-emittable, or a duplicate name (recorded as a problem).
    fn const_name_for(&mut self, path: &str) -> Option<String> {
        if path.contains('[') {
            return None;
        }
        let name = path.replace('.', "_").to_uppercase();
        if !self.names.insert(name.clone()) {
            self.problems
                .push(format!("duplicate generated const name `{name}` from `{path}`"));
            return None;
        }
        Some(name)
    }

    fn walk_array(&mut self, path: &str, arr: &[toml::Value]) {
        if arr.is_empty() {
            self.problems
                .push(format!("empty array at `{path}` — nothing in the spec is empty"));
            return;
        }
        match &arr[0] {
            toml::Value::Integer(_) => {
                let mut vals = Vec::with_capacity(arr.len());
                for (i, x) in arr.iter().enumerate() {
                    match x {
                        toml::Value::Integer(v) if *v >= 0 => {
                            vals.push(*v as u64);
                            self.map.push((format!("{path}[{i}]"), Val::U(*v as u64)));
                        }
                        toml::Value::Integer(v) => {
                            self.problems
                                .push(format!("negative integer {v} at `{path}[{i}]`"));
                            return;
                        }
                        _ => {
                            self.problems
                                .push(format!("mixed array element types at `{path}[{i}]`"));
                            return;
                        }
                    }
                }
                if let Some(name) = self.const_name_for(path) {
                    let items = vals.iter().map(|v| format!("{v}u64")).collect::<Vec<_>>().join(", ");
                    self.out
                        .push_str(&format!("pub const {name}: [u64; {}] = [{items}];\n", vals.len()));
                }
            }
            toml::Value::String(_) => {
                let mut vals = Vec::with_capacity(arr.len());
                for (i, x) in arr.iter().enumerate() {
                    match x {
                        toml::Value::String(s) => {
                            vals.push(s.clone());
                            self.map.push((format!("{path}[{i}]"), Val::S(s.clone())));
                        }
                        _ => {
                            self.problems
                                .push(format!("mixed array element types at `{path}[{i}]`"));
                            return;
                        }
                    }
                }
                if let Some(name) = self.const_name_for(path) {
                    let items = vals.iter().map(|s| format!("{s:?}")).collect::<Vec<_>>().join(", ");
                    self.out.push_str(&format!(
                        "pub const {name}: [&'static str; {}] = [{items}];\n",
                        vals.len()
                    ));
                }
            }
            toml::Value::Table(_) => match path {
                "economy.buckets" => self.emit_aot(
                    path,
                    arr,
                    "EconomyBucket",
                    "ECONOMY_BUCKETS",
                    "ECONOMY_BUCKET_COUNT",
                    BUCKET_SCHEMA,
                ),
                "errata" => {
                    self.emit_aot(path, arr, "Erratum", "ERRATA", "ERRATA_COUNT", ERRATA_SCHEMA)
                }
                _ => self.problems.push(format!(
                    "unknown array-of-tables family at `{path}` — extending the spec with a \
                     structured family is deliberate: pin its schema in nerv-core/build.rs and add \
                     the typed validator in nerv-conformance::spec"
                )),
            },
            first => self.problems.push(format!(
                "unsupported array element type ({}) at `{path}`",
                toml_type_name(first)
            )),
        }
    }

    fn emit_aot(
        &mut self,
        path: &str,
        arr: &[toml::Value],
        struct_name: &str,
        static_name: &str,
        count_name: &str,
        schema: &[FieldDef],
    ) {
        self.out.push_str(&format!(
            "\n// `{path}` — structured family (schema pinned in nerv-core/build.rs).\n"
        ));
        self.out.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n");
        self.out.push_str(&format!("pub struct {struct_name} {{\n"));
        for f in schema {
            self.out
                .push_str(&format!("    pub {}: {},\n", f.name, field_rust_type(&f.ty)));
        }
        self.out.push_str("}\n\n");
        self.out
            .push_str(&format!("pub static {static_name}: [{struct_name}; {}] = [\n", arr.len()));

        for (i, elem) in arr.iter().enumerate() {
            let toml::Value::Table(t) = elem else {
                self.problems.push(format!("`{path}[{i}]` is not a table"));
                continue;
            };
            for k in t.keys() {
                if !schema.iter().any(|f| f.name == *k) {
                    self.problems.push(format!(
                        "unknown field `{k}` in `{path}[{i}]` (deny_unknown_fields discipline)"
                    ));
                }
            }
            self.out.push_str(&format!("    {struct_name} {{\n"));
            for f in schema {
                let field_path = format!("{path}[{i}].{}", f.name);
                let expr = match (&f.ty, t.get(f.name)) {
                    (FieldType::Str, Some(toml::Value::String(s))) => {
                        self.map.push((field_path, Val::S(s.clone())));
                        format!("{s:?}")
                    }
                    (FieldType::Int, Some(toml::Value::Integer(v))) if *v >= 0 => {
                        self.map.push((field_path, Val::U(*v as u64)));
                        format!("{v}u64")
                    }
                    (FieldType::OptInt, Some(toml::Value::Integer(v))) if *v >= 0 => {
                        self.map.push((field_path, Val::U(*v as u64)));
                        format!("Some({v}u64)")
                    }
                    (FieldType::OptBool, Some(toml::Value::Boolean(b))) => {
                        self.map.push((field_path, Val::B(*b)));
                        format!("Some({b})")
                    }
                    (FieldType::OptInt, None) | (FieldType::OptBool, None) => "None".to_string(),
                    (ty, found) => {
                        let got = found.map(toml_type_name).unwrap_or("missing");
                        self.problems.push(format!(
                            "field `{field_path}`: expected {}, got {got}",
                            field_type_name(ty)
                        ));
                        // Never written: problems abort the build before fs::write.
                        "/* INVALID */".to_string()
                    }
                };
                self.out.push_str(&format!("        {}: {expr},\n", f.name));
            }
            self.out.push_str("    },\n");
        }
        self.out.push_str("];\n");
        self.out.push_str(&format!("pub const {count_name}: usize = {};\n", arr.len()));
    }

    fn finish(&mut self) {
        self.map.sort_by(|a, b| a.0.cmp(&b.0));
        for w in self.map.windows(2) {
            if w[0].0 == w[1].0 {
                self.problems.push(format!("duplicate leaf path `{}`", w[0].0));
            }
        }
        let n = self.map.len();
        self.out.push_str(
            "\n// Leaf inventory for the conformance zero-drift check (paths sorted).\n\
             #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
             pub enum ParamValue {\n\
             \x20   U64(u64),\n\
             \x20   Bool(bool),\n\
             \x20   Str(&'static str),\n\
             }\n\n",
        );
        self.out.push_str(&format!("pub const PARAMS_LEAF_COUNT: usize = {n};\n"));
        self.out
            .push_str("pub static PARAMS_MAP: &[(&'static str, ParamValue)] = &[\n");
        for (p, v) in &self.map {
            self.out.push_str(&format!("    ({p:?}, {}),\n", v.map_expr()));
        }
        self.out.push_str("];\n");
        self.out.push_str(ASSERTS);
    }
}

const ASSERTS: &str = r#"
// ---------------------------------------------------------------------------
// Compile-time backstop (the full semantic validator is nerv-conformance::spec,
// run in CI; these asserts make the most load-bearing invariants impossible to
// build against, even without CI).
// ---------------------------------------------------------------------------
const _: () = assert!(PROTOCOL_SUPPLY_NERV > 0, "supply must be positive");
const _: () = assert!(PROTOCOL_NANO_PER_NERV == 1_000_000_000, "nano-NERV scaling");
const _: () = assert!(
    (1u64 << PROTOCOL_SHARD_ID_BITS_GENESIS) == PROTOCOL_SHARD_COUNT_GENESIS,
    "shard bits/count (App B)"
);
const _: () = assert!(PROTOCOL_SHARD_COUNT_MAX >= PROTOCOL_SHARD_COUNT_GENESIS, "max >= genesis shards");
const _: () = assert!(TIMING_EPOCH_SECS == 86_400, "epoch is 24h (App B)");
const _: () = assert!(
    CONSENSUS_SHARD_QUORUM > 0 && CONSENSUS_SHARD_QUORUM <= CONSENSUS_SHARD_COMMITTEE_SIZE,
    "shard quorum in (0, size]"
);
const _: () = assert!(CONSENSUS_BEACON_QUORUM <= CONSENSUS_BEACON_COMMITTEE_SIZE, "beacon quorum");
const _: () = assert!(CONSENSUS_REGISTRY_QUORUM <= CONSENSUS_REGISTRY_COMMITTEE_SIZE, "registry quorum");
const _: () = assert!(CONSENSUS_ATTESTATION_QUORUM <= CONSENSUS_ATTESTATION_SIGNERS, "attestation quorum (E-001)");
const _: () = assert!(CUSTODY_VALUE_MAX_NANO == (1u64 << 60), "value cap is 2^60 (§3.2)");
const _: () = assert!(PROOFS_FIELD_MODULUS == 18_446_744_069_414_584_321u64, "Goldilocks modulus (§5.3)");
const _: () = assert!(PROOFS_FRI_SECURITY_BITS >= 100, "FRI security floor (D.2)");
const _: () = assert!(OVERLAY_W_ROWS == 64 && OVERLAY_W_COLS == 256, "W is 64x256 (§7.2)");
const _: () = assert!(
    OVERLAY_SPARK_BOUND >= 2 * OVERLAY_MAX_ACTIVE_SLOTS + 1,
    "spark bound (§7.4)"
);
const _: () = assert!(SEAL_Q < (1u64 << 32), "seal modulus is 32-bit (§6.3.1)");
const _: () = assert!(SEAL_THRESHOLD_T <= SEAL_COMMITTEE_N, "threshold <= committee (E-003)");
const _: () = assert!(SEAL_CHUNK_MIN <= SEAL_CHUNK_MAX, "chunk bounds (§6.3.8)");
const _: () = assert!(FEES_DYNAMIC_FLOOR_M_MAX >= 1, "floor multiplier >= 1 (D.4)");
const _: () = assert!(
    FEES_SPLIT_PRODUCER_PERMILLE
        + FEES_SPLIT_PROVER_PERMILLE
        + FEES_SPLIT_DA_PERMILLE
        + FEES_SPLIT_RELAY_PERMILLE
        == 1000,
    "40/30/20/10 fee split (§12.5)"
);
const _: () = assert!(BUDGETS_CI_WHOLE_AIR_TARGET <= BUDGETS_CI_WHOLE_AIR_CEILING, "AIR budget target <= ceiling (D.2)");
const _: () = assert!(
    BUDGETS_CI_WALLET_PROOF_TARGET_BYTES <= BUDGETS_CI_WALLET_PROOF_HARD_BYTES,
    "wallet proof-size target <= ceiling (D.2)"
);
const _: () = assert!(PROOFS_FRI_POSEIDON2_T == 16, "Poseidon2 state width (§5.3)");
const _: () = assert!(PROOFS_FRI_POSEIDON2_FULL_ROUNDS % 2 == 0, "Poseidon2 full rounds split evenly");
const _: () = assert!(PROOFS_FRI_POSEIDON2_SBOX_DEGREE == 7, "Poseidon2 S-box degree");
"#;

