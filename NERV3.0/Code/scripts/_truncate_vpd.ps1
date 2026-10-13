$ErrorActionPreference = 'Stop'
$path = 'C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-seal/src/vpd.rs'
$lines = [System.IO.File]::ReadAllLines($path, [System.Text.Encoding]::UTF8)
if ($lines.Length -gt 598) {
    [System.IO.File]::WriteAllLines($path, $lines[0..597], [System.Text.UTF8Encoding]::new($false))
    Write-Host "Truncated from $($lines.Length) to 598 lines"
} else {
    Write-Host "Already short enough: $($lines.Length) lines"
}
