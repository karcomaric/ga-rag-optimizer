$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
& ".venv\Scripts\Activate.ps1"
$env:HF_HUB_OFFLINE = "1"

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method default --seed 42
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Block default ist vollständig durchgelaufen (nur Seed 42, die Default-Konfiguration ist deterministisch)."
