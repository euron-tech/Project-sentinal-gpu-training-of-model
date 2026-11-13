# Keep the SENTINEL dashboard always running (Docker, auto-restarts with Docker Desktop).
#   & ui\run_service.ps1                                   # start (build if needed)
#   & ui\run_service.ps1 -LlmUrl http://<VM_IP>:8000 -LlmKey <key>   # point at the Hyperstack LLM
#   & ui\run_service.ps1 -Rebuild                          # after code changes
#   & ui\run_service.ps1 -Stop
# Settings persist in ui\.env (git-ignored). The XGBoost key is taken from xgboost_benchmark\.env.
param([string]$LlmUrl, [string]$LlmKey, [string]$XgbUrl, [switch]$Rebuild, [switch]$Stop, [int]$Port = 8501)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$name = "sentinel-ui"

if (docker ps -aq --filter "name=^${name}$") { docker rm -f $name | Out-Null }
if ($Stop) { Write-Host "Stopped $name"; exit 0 }

$envFile = Join-Path $here ".env"
$cfg = [ordered]@{ LLM_URL = "http://host.docker.internal:8000"; LLM_API_KEY = ""; XGB_URL = "http://host.docker.internal:8001"; XGB_API_KEY = "" }
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) { if ($line -match '^\s*([A-Z_]+)=(.*)$') { $cfg[$matches[1]] = $matches[2].Trim() } }
}
$xgbEnv = Join-Path $root "xgboost_benchmark\.env"
if (-not $cfg.XGB_API_KEY -and (Test-Path $xgbEnv)) {
    $m = Select-String -Path $xgbEnv -Pattern '^XGB_API_KEY=(.*)$'; if ($m) { $cfg.XGB_API_KEY = $m.Matches[0].Groups[1].Value.Trim() }
}
if ($LlmUrl) { $cfg.LLM_URL = $LlmUrl }
if ($LlmKey) { $cfg.LLM_API_KEY = $LlmKey }
if ($XgbUrl) { $cfg.XGB_URL = $XgbUrl }
($cfg.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) | Out-File -Encoding ascii $envFile

if ($Rebuild -or -not (docker images -q $name)) { docker build -t $name $here; if ($LASTEXITCODE) { exit 1 } }
docker run -d --name $name --restart unless-stopped -p "${Port}:8501" --env-file $envFile `
    --add-host "host.docker.internal:host-gateway" $name | Out-Null
for ($i = 0; $i -lt 40; $i++) {
    try { $h = Invoke-WebRequest "http://localhost:$Port/_stcore/health" -UseBasicParsing -TimeoutSec 2; break } catch { Start-Sleep 1 }
}
Write-Host "Dashboard : http://localhost:$Port   (restarts automatically)"
Write-Host "LLM       : $($cfg.LLM_URL)"
Write-Host "XGBoost   : $($cfg.XGB_URL)"
Write-Host "Share     : ngrok http $Port"
