#!/usr/bin/env bash
# ci-strict-clippy.sh — POSIX counterpart of tests/scripts/ci-strict-clippy.ps1.
# Used by the GitHub Actions matrix when running on Linux runners.

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
  if cargo clippy -p "$crate" --all-targets --no-deps --ignore-rust-version -- -D warnings; then
    echo "  ok"
  else
    echo "  FAIL"
    failures+=("$crate")
  fi
done

if [ ${#failures[@]} -gt 0 ]; then
  echo
  echo "ci-strict-clippy: ${#failures[@]} crate(s) failed: ${failures[*]}"
  exit 1
fi

echo
echo "ci-strict-clippy: all crates clean."
exit 0
