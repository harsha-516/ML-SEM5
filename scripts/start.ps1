# One command to bring Tarang up: fresh data, live refresh loop, Telegram bot.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\start.ps1
# Add -Dashboard to also start the local dashboard on http://localhost:5173
# Stop everything:    powershell -ExecutionPolicy Bypass -File scripts\start.ps1 -Stop
param([switch]$Dashboard, [switch]$Stop)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null

function Stop-Tarang {
    # Only processes started from this project's venv or dashboard folder
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'hazards\.(bot|refresh)' -or
        ($_.CommandLine -match 'vite' -and $_.CommandLine -match [regex]::Escape("$root\dashboard"))
    } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Stop-Tarang
if ($Stop) { Write-Host "Stopped bot, refresh loop and dashboard."; return }

if (-not (Test-Path $py)) { Write-Host "Missing .venv. See the setup guide first."; exit 1 }
if (-not (Select-String -Path ".env" -Pattern "TELEGRAM_BOT_TOKEN=\S" -Quiet -ErrorAction SilentlyContinue)) {
    Write-Host "TELEGRAM_BOT_TOKEN missing in .env"; exit 1
}

Write-Host "1/3 Pulling fresh data (takes a few minutes)..."
& $py -m hazards.refresh
if ($LASTEXITCODE -ne 0) { Write-Host "Refresh had errors, check logs\refresh.err.log. Continuing." }

Write-Host "2/3 Starting live refresh loop (every 5 min)..."
Start-Process $py -ArgumentList "-m", "hazards.refresh", "--loop", "5" -WorkingDirectory $root `
    -RedirectStandardOutput "$logs\refresh.out.log" -RedirectStandardError "$logs\refresh.err.log" -WindowStyle Hidden

Write-Host "3/3 Starting Telegram bot..."
Start-Process $py -ArgumentList "-m", "hazards.bot" -WorkingDirectory $root `
    -RedirectStandardOutput "$logs\bot.out.log" -RedirectStandardError "$logs\bot.err.log" -WindowStyle Hidden

if ($Dashboard) {
    Write-Host "Starting dashboard on http://localhost:5173 ..."
    Start-Process "cmd.exe" -ArgumentList "/c", "npm run dev" -WorkingDirectory (Join-Path $root "dashboard") `
        -RedirectStandardOutput "$logs\dashboard.out.log" -RedirectStandardError "$logs\dashboard.err.log" -WindowStyle Hidden
}

Start-Sleep 6
$up = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'hazards\.(bot|refresh)' -and $_.Name -eq 'python.exe' }
Write-Host ("Running: " + (($up | ForEach-Object { if ($_.CommandLine -match 'bot') { 'bot' } else { 'refresh' } }) | Sort-Object -Unique) -join ", ")
Write-Host "Ready. Message the bot on Telegram. Logs in $logs"
