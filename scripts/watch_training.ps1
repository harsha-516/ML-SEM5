# Shows model training progress live. Close the window or press Ctrl+C to stop watching; training keeps running.
$log = Join-Path $PSScriptRoot "..\logs\training.log"
if (-not (Test-Path $log)) {
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    New-Item -ItemType File $log | Out-Null
}
Write-Host "Watching training progress. Ctrl+C stops watching only." -ForegroundColor Cyan
Get-Content -Path $log -Tail 40 -Wait
