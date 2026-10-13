# ci-strict-clippy.ps1
#
# D1 CI: per-crate strict clippy sweep.
#
# Iterates every `crates/*/Cargo.toml` and runs
#     cargo clippy -p <crate> --all-targets --no-deps --ignore-rust-version -- -D warnings
# against it, exiting non-zero on any failure. Used as the canonical
# "did the last session leave the tree in a clippy-clean state?" gate.
#
# Usage (from the repo root):
#     pwsh -NoProfile -NonInteractive -File tests/scripts/ci-strict-clippy.ps1
#
# Notes:
# - Prepends the MinGW + rustc-1.85 paths so the toolchain matches the
#   rest of the testnet CI.
# - The `apps/nerv-gui` workspace member is excluded — that crate
#   requires rustc 1.88+ and is built under a separate toolchain.
# - Run on a per-session basis before claiming a phase's clippy goal.

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

$crates = Get-ChildItem -Path 'crates' -Recurse -Filter 'Cargo.toml' -Depth 2 |
    ForEach-Object { (Select-String -Path $_.FullName -Pattern '^name\s*=\s*"([^"]+)"' -List).Matches[0].Groups[1].Value }

if ($crates.Count -eq 0) {
    Write-Host "ci-strict-clippy: no crates found under crates/" -ForegroundColor Red
    exit 1
}

$failures = @()
foreach ($crate in $crates) {
    Write-Host "=== $crate ===" -ForegroundColor Cyan
    $proc = Start-Process -FilePath 'cargo' -ArgumentList @(
        'clippy',
        '-p', $crate,
        '--all-targets',
        '--no-deps',
        '--ignore-rust-version',
        '--', '-D', 'warnings'
    ) -NoNewWindow -PassThru -Wait
    if ($proc.ExitCode -ne 0) {
        $failures += $crate
        Write-Host "  FAIL ($($proc.ExitCode))" -ForegroundColor Red
    } else {
        Write-Host "  ok" -ForegroundColor Green
    }
}

if ($failures.Count -gt 0) {
    Write-Host ""
    Write-Host "ci-strict-clippy: $($failures.Count) crate(s) failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "ci-strict-clippy: $($crates.Count) crates clean." -ForegroundColor Green
exit 0
