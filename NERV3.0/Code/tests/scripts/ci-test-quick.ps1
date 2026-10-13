# ci-test-quick.ps1
#
# D1 CI: quick regression-test gate. Runs the small, fast subset of
# `cargo test --lib` for each newly-cleaned crate. The full
# `cargo test --workspace --lib -- --test-threads=1` takes hours on
# the 8 GB box; this script runs only the suites the carry-forward
# session flagged as the regression guard.
#
# Usage (from the repo root):
#     pwsh -NoProfile -NonInteractive -File tests/scripts/ci-test-quick.ps1
#
# Default regression set (each is a single crate's --lib target with
# a TESTNAME filter where appropriate):
#   nerv-proofs      -- layout_pins measured_families  (8/8)
#   nerv-state       -- all lib tests                  (73/73)
#   nerv-net         -- all lib tests                  (66/66)
#   nerv-consensus   -- all lib tests                  (53/53)
#   nerv-registry    -- all lib tests                  (31/31)
#   nerv-wallet      -- all lib tests                  (TBD)
#   nerv-wallet-core -- all lib tests                  (TBD)
#
# Each crate is run serially with --test-threads=1; total budget is
# under 10 minutes on the 8 GB test box.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Prepend the dev-box MinGW + pinned-rustc + cargo paths only when they
# exist. On CI runners the toolchain is already on PATH (installed via
# dtolnay/rust-toolchain), so the prepend is a no-op there.
$toolchainPaths = @(
    'C:\Users\prosp\Documents\MiniMax NERV\.mingw\mingw64\bin',
    (Join-Path $env:USERPROFILE '.rustup\toolchains\1.85.0-x86_64-pc-windows-gnu\bin'),
    (Join-Path $env:USERPROFILE '.cargo\bin')
) | Where-Object { $_ -and (Test-Path $_) }
if ($toolchainPaths.Count -gt 0) {
    $env:Path = ($toolchainPaths -join ';') + ';' + $env:Path
}
Set-Location (Join-Path $PSScriptRoot '..\..')

$suites = @(
    @{ Crate = 'nerv-proofs';       Filter = 'layout_pins measured_families' }
    @{ Crate = 'nerv-state';        Filter = $null }
    @{ Crate = 'nerv-net';          Filter = $null }
    @{ Crate = 'nerv-consensus';    Filter = $null }
    @{ Crate = 'nerv-registry';     Filter = $null }
    @{ Crate = 'nerv-wallet';       Filter = $null }
    @{ Crate = 'nerv-wallet-core';  Filter = $null }
)

$failures = @()
foreach ($suite in $suites) {
    $crate = $suite.Crate
    $filter = $suite.Filter
    Write-Host "=== $crate ===" -ForegroundColor Cyan
    $args = @('test', '-p', $crate, '--lib', '--', '--test-threads=1')
    if ($filter) { $args += @('--', $filter) }
    $proc = Start-Process -FilePath 'cargo' -ArgumentList $args -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        $failures += $crate
        Write-Host "  FAIL ($($proc.ExitCode))" -ForegroundColor Red
    } else {
        Write-Host "  ok" -ForegroundColor Green
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "ci-test-quick: $($failures.Count) crate(s) failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "ci-test-quick: all suites clean." -ForegroundColor Green
exit 0
