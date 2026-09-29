//! Axiom 3's CI harness (DSR-1; erratum 174): the firewall as executable
//! checks, living inside the crate it guards — deleting nerv-knowledge
//! deletes the checks with it, and the invariant they establish is what
//! the CI delete job (the Makefile's firewall-check) re-verifies
//! externally by building the copy.
//!
//! 1. `the_firewall_is_a_build_rule` (fast, always runs).
//! 2. `the_workspace_builds_without_nerv_knowledge` (#[ignore], the full
//!    job: minutes of compilation).
//!
//! Implemented by direct manifest parsing — no nested cargo in the fast
//! path, no lock contention.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const AUTHORITY_SET: &[&str] = &[
    "nerv-custody",
    "nerv-codec",
    "nerv-seal",
    "nerv-proofs",
    "nerv-state",
    "nerv-registry",
    "nerv-consensus",
    "nerv-da",
    "nerv-net",
    "nerv-economy",
    "nerv-witness",
];

fn workspace_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nerv-knowledge → the workspace root")
        .to_path_buf();
    let text = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(text.contains("[workspace]"), "the ancestor is the workspace root");
    root
}

fn parse(path: &Path) -> toml::Value {
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn member_paths(root: &Path) -> Vec<String> {
    parse(&root.join("Cargo.toml"))["workspace"]["members"]
        .as_array()
        .expect("workspace.members")
        .iter()
        .map(|v| v.as_str().expect("a member path").to_string())
        .collect()
}

/// name → internal build-dependency names: entries of [dependencies],
/// [build-dependencies], and [target.*.dependencies] whose names are
/// workspace members (the build graph).
fn build_graph(root: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let members: BTreeSet<String> = member_paths(root)
        .into_iter()
        .map(|p| p.rsplit('/').next().unwrap().to_string())
        .collect();
    let mut graph = BTreeMap::new();
    for rel in &members {
        let manifest = parse(&root.join(rel).join("Cargo.toml"));
        let deps: BTreeSet<String> = dep_names(&manifest, false)
            .into_iter()
            .filter(|d| members.contains(d))
            .collect();
        graph.insert(rel.clone(), deps);
    }
    graph
}

/// Every dependency-table entry (package names), recursing through
/// target-specific tables; `include_dev` adds dev-dependencies (the
/// strict scan only — dev edges are outside the build graph).
fn dep_names(manifest: &toml::Value, include_dev: bool) -> BTreeSet<String> {
    fn collect(v: &toml::Value, out: &mut BTreeSet<String>) {
        if let toml::Value::Table(t) = v {
            for name in t.keys() {
                out.insert(name.clone());
            }
        }
    }
    fn walk(v: &toml::Value, include_dev: bool, out: &mut BTreeSet<String>) {
        let toml::Value::Table(t) = v else { return };
        for (k, sub) in t {
            if k == "dev-dependencies" {
                if include_dev {
                    collect(sub, out);
                }
            } else if k.ends_with("dependencies") {
                collect(sub, out);
            } else {
                walk(sub, include_dev, out);
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(manifest, include_dev, &mut out);
    out
}

fn build_closure(graph: &BTreeMap<String, BTreeSet<String>>, from: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from.to_string()];
    while let Some(n) = stack.pop() {
        if !seen.insert(n.clone()) {
            continue;
        }
        if let Some(deps) = graph.get(&n) {
            for d in deps {
                if graph.contains_key(d) {
                    stack.push(d.clone());
                }
            }
        }
    }
    seen
}

#[test]
fn the_firewall_is_a_build_rule() {
    let root = workspace_root();
    let graph = build_graph(&root);
    assert!(
        graph.contains_key("nerv-knowledge"),
        "the member list changed — update this test's assumptions"
    );

    // (a) DSR-2's direction: knowledge's own internal deps ⊆ {core, codec}.
    let own: Vec<&String> = graph["nerv-knowledge"]
        .iter()
        .filter(|d| graph.contains_key(*d))
        .collect();
    assert!(
        own.iter().all(|d| *d == "nerv-core" || *d == "nerv-codec"),
        "nerv-knowledge's internal deps must be ⊆ {{nerv-core, nerv-codec}}; found {own:?}"
    );

    // (b) DSR-1: no authority member's build closure reaches knowledge.
    for name in AUTHORITY_SET {
        if !graph.contains_key(name) {
            continue; // e.g. nerv-witness, chunk 18
        }
        let closure = build_closure(&graph, name);
        assert!(
            !closure.contains("nerv-knowledge"),
            "DSR-1 violated: {name}'s build closure reaches nerv-knowledge: {closure:?}"
        );
    }

    // (c) The strict scan — the delete job's precondition: no member
    // manifest mentions nerv-knowledge at all (dev included). Diagnostic
    // crates consume frozen vector DATA; generators live in nerv-knowledge.
    for rel in member_paths(&root) {
        if rel == "crates/nerv-knowledge" {
            continue;
        }
        let manifest = parse(&root.join(&rel).join("Cargo.toml"));
        let all = dep_names(&manifest, true);
        assert!(
            !all.contains("nerv-knowledge"),
            "{rel} references nerv-knowledge — Axiom 3's delete job requires the workspace \
             knowledge-free (generators live in nerv-knowledge; vectors travel as data)"
        );
    }
}

#[test]
#[ignore = "the full Axiom-3 job: minutes of compilation; `cargo test -p nerv-knowledge -- --ignored`"]
fn the_workspace_builds_without_nerv_knowledge() {
    let root = workspace_root();
    let tmp = std::env::temp_dir().join(format!(
        "nerv-delete-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let knowledge = root.join("crates/nerv-knowledge");
    assert!(knowledge.exists());

    let skip = |p: &Path| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        name == "target" || name == ".git" || p.starts_with(&knowledge)
    };
    copy_tree(&root, &tmp, &skip).expect("copy the workspace");

    // Patch the members list: drop the knowledge line.
    let manifest_path = tmp.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    let patched: String = text
        .lines()
        .filter(|l| !l.contains("\"crates/nerv-knowledge\""))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&manifest_path, patched + "\n").unwrap();
    assert!(!tmp.join("crates/nerv-knowledge").exists());

    // Fresh target dir; the lockfile travels (its stale knowledge entry is
    // pruned locally — offline, no registry access).
    let status = run_with_deadline(
        std::process::Command::new("cargo")
            .args(["check", "--workspace", "--offline"])
            .env("CARGO_TARGET_DIR", tmp.join("target"))
            .current_dir(&tmp),
        1800,
    );
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        status.success(),
        "Axiom 3 failed: the workspace does not build without nerv-knowledge \
         (run with --nocapture for the compiler output)"
    );
}

fn copy_tree(src: &Path, dst: &Path, skip: &dyn Fn(&Path) -> bool) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let p = entry.path();
        if skip(&p) {
            continue;
        }
        if p.is_dir() {
            copy_tree(&p, &dst.join(entry.file_name()), skip)?;
        } else {
            std::fs::copy(&p, &dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn run_with_deadline(mut cmd: std::process::Command, secs: u64) -> std::process::ExitStatus {
    let mut child = cmd.spawn().expect("spawn cargo");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
    loop {
        match child.try_wait().expect("poll cargo") {
            Some(status) => return status,
            None if std::time::Instant::now() > deadline => {
                let _ = child.kill();
                panic!("the nested cargo exceeded its {secs}s deadline");
            }
            None => std::thread::sleep(std::time::Duration::from_millis(500)),
        }
    }
}

