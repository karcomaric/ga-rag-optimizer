$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
& ".venv\Scripts\Activate.ps1"
$env:HF_HUB_OFFLINE = "1"

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method default --smoke
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method grid --smoke
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method random --smoke
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method ga --smoke
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Alle vier Smoke-Läufe sind erfolgreich durchgelaufen."
