# ci-lib-sweep.ps1
#
# D1 CI: full workspace lib-test sweep, chunked per crate to fit the 8 GB
# test budget. Iterates every `crates/*/Cargo.toml` and runs
#     cargo test -p <crate> --lib --ignore-rust-version -- --test-threads=1
# against it. This is the multi-hour long-tail run, so in CI it is a
# nightly job (not PR-blocking).
#
# Usage (from the repo root):
#     pwsh -NoProfile -NonInteractive -File tests/scripts/ci-lib-sweep.ps1

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
    Write-Host "ci-lib-sweep: no crates found under crates/" -ForegroundColor Red
    exit 1
}

$failures = @()
foreach ($crate in $crates) {
    Write-Host "=== $crate ===" -ForegroundColor Cyan
    $proc = Start-Process -FilePath 'cargo' -ArgumentList @(
        'test',
        '-p', $crate,
        '--lib',
        '--ignore-rust-version',
        '--', '--test-threads=1'
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
    Write-Host "ci-lib-sweep: $($failures.Count) crate(s) failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "ci-lib-sweep: $($crates.Count) crates clean." -ForegroundColor Green
exit 0
