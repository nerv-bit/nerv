#!/usr/bin/env bash
# ci-test-quick.sh — POSIX counterpart of tests/scripts/ci-test-quick.ps1.

set -euo pipefail
cd "$(dirname "$0")/../.."

suites=(
  "nerv-proofs:layout_pins measured_families"
  "nerv-state:"
  "nerv-net:"
  "nerv-consensus:"
  "nerv-registry:"
  "nerv-wallet:"
  "nerv-wallet-core:"
)

failures=()
for spec in "${suites[@]}"; do
  crate="${spec%%:*}"
  filter="${spec#*:}"
  echo "=== $crate ==="
  args=(test -p "$crate" --lib -- --test-threads=1)
  if [ -n "$filter" ]; then
    args+=(-- "$filter")
  fi
  if cargo "${args[@]}"; then
    echo "  ok"
  else
    echo "  FAIL"
    failures+=("$crate")
  fi
done

if [ ${#failures[@]} -gt 0 ]; then
  echo
  echo "ci-test-quick: ${#failures[@]} crate(s) failed: ${failures[*]}"
  exit 1
fi

echo
echo "ci-test-quick: all suites clean."
exit 0
