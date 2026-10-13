#!/usr/bin/env bash
# ci-lib-sweep.sh — POSIX counterpart of tests/scripts/ci-lib-sweep.ps1.
# Full workspace lib-test sweep, chunked per crate to fit the 8 GB test
# budget. Nightly-only in CI.

set -euo pipefail
cd "$(dirname "$0")/../.."

crates=$(ls crates | while read d; do
  if [ -f "crates/$d/Cargo.toml" ]; then
    name=$(grep -E '^name\s*=\s*"[^"]+"' "crates/$d/Cargo.toml" | head -n1 | sed -E 's/^name\s*=\s*"([^"]+)".*/\1/')
    echo "$name"
  fi
done)

failures=()
for crate in $crates; do
  echo "=== $crate ==="
  if cargo test -p "$crate" --lib --ignore-rust-version -- --test-threads=1; then
    echo "  ok"
  else
    echo "  FAIL"
    failures+=("$crate")
  fi
done

if [ ${#failures[@]} -gt 0 ]; then
  echo
  echo "ci-lib-sweep: ${#failures[@]} crate(s) failed: ${failures[*]}"
  exit 1
fi

echo
echo "ci-lib-sweep: all crates clean."
exit 0
