$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
& ".venv\Scripts\Activate.ps1"
$env:HF_HUB_OFFLINE = "1"

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method grid --seed 42
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Block grid ist vollständig durchgelaufen (nur Seed 42, Grid Search ist deterministisch)."
