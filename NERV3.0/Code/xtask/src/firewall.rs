//! DSR-1 firewall: the dependency-direction policy as a build rule.
//!
//! Checks, against the live `cargo metadata` graph:
//!   1. every workspace member is present in the policy matrix;
//!   2. every member's internal (non-dev) dependencies ⊆ its allowed set;
//!   3. `nerv-knowledge`'s internal dependencies are exactly {core, codec};
//!   4. only allowlisted advisory/test consumers depend on `nerv-knowledge`;
//!   5. no STRICT_AUTHORITY crate reaches `nerv-knowledge` transitively.
//!
//! The workspace-minus-knowledge *build* (delete test, Axiom 3) lands with
//! nerv-knowledge in chunk 17, where there is something to delete.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Result};
use cargo_metadata::{DependencyKind, Metadata, MetadataCommand, Package, PackageId};

use crate::policy;

pub fn run() -> Result<()> {
    let meta = MetadataCommand::new()
        .exec()
        .map_err(|e| anyhow::anyhow!("cargo metadata failed: {e}"))?;

    let members = member_packages(&meta);
    if members.is_empty() {
        bail!("no workspace members found");
    }

    // name -> internal non-dev dependency names
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for pkg in &members {
        graph.insert(pkg.name.clone(), internal_deps(&meta, &pkg.id));
    }

    let mut violations: Vec<String> = Vec::new();
    let mut lines: Vec<String> = Vec::new();

    for (name, deps) in &graph {
        let Some(allowed) = policy::allowed_deps(name) else {
            violations.push(format!(
                "member `{name}` is not in the dependency policy — add it (deliberately) to xtask/src/policy.rs"
            ));
            continue;
        };
        for dep in deps {
            if dep == "nerv-knowledge" {
                if !policy::KNOWLEDGE_DEPENDENTS_ALLOWLIST.contains(&name.as_str()) {
                    violations.push(format!(
                        "DSR-1: `{name}` depends on nerv-knowledge (allowed: advisory/test consumers only: {:?})",
                        policy::KNOWLEDGE_DEPENDENTS_ALLOWLIST
                    ));
                }
                continue;
            }
            if !allowed.contains(&dep.as_str()) {
                violations.push(format!(
                    "dependency direction: `{name}` -> `{dep}` is not in `{name}`'s allowed set {allowed:?}"
                ));
            }
        }
        if name == "nerv-knowledge" {
            // Exactly {core, codec} — DSR-2's one-command answer.
            let expect: BTreeSet<&str> = ["nerv-core", "nerv-codec"].into_iter().collect();
            let actual: BTreeSet<&str> = deps.iter().map(|s| s.as_str()).collect();
            if actual != expect {
                violations.push(format!(
                    "DSR-2: nerv-knowledge must depend on exactly {{nerv-core, nerv-codec}}; found {actual:?}"
                ));
            }
        }
        lines.push(format!("  {:<22} -> [{}]", name, deps.iter().cloned().collect::<Vec<_>>().join(", ")));
    }

    // Transitive reachability: strict authority crates must never reach knowledge.
    for name in graph.keys() {
        if policy::STRICT_AUTHORITY.contains(&name.as_str()) && reaches(&graph, name, "nerv-knowledge") {
            violations.push(format!(
                "DSR-1 (transitive): authority crate `{name}` reaches nerv-knowledge through its dependency closure"
            ));
        }
    }

    println!("firewall: {} member(s) checked", graph.len());
    for l in &lines {
        println!("{l}");
    }

    if violations.is_empty() {
        println!("firewall: PASS");
        Ok(())
    } else {
        eprintln!("firewall: {} violation(s):", violations.len());
        for v in &violations {
            eprintln!("  - {v}");
        }
        bail!("DSR-1 firewall failed");
    }
}

fn member_packages(meta: &Metadata) -> Vec<&Package> {
    meta.packages
        .iter()
        .filter(|p| meta.workspace_members.contains(&p.id))
        .collect()
}

/// Internal (workspace-member) dependencies of `id`, excluding dev-deps.
/// Build-deps are included: build scripts execute at build time and count.
fn internal_deps(meta: &Metadata, id: &PackageId) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Some(resolve) = &meta.resolve else {
        return out;
    };
    let Some(node) = resolve.nodes.iter().find(|n| &n.id == id) else {
        return out;
    };
    for dep in &node.deps {
        let is_dev = dep.dep_kinds.iter().all(|k| k.kind == DependencyKind::Development);
        if is_dev {
            continue;
        }
        if let Some(p) = meta.packages.iter().find(|p| p.id == dep.pkg) {
            if meta.workspace_members.contains(&p.id) {
                out.insert(p.name.clone());
            }
        }
    }
    out
}

fn reaches(graph: &BTreeMap<String, BTreeSet<String>>, from: &str, target: &str) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(cur) = stack.pop() {
        if cur == target {
            return true;
        }
        if !seen.insert(cur.to_string()) {
            continue;
        }
        if let Some(deps) = graph.get(cur) {
            for d in deps {
                stack.push(d);
            }
        }
    }
    false
}
