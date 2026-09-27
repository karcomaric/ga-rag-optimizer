$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
& ".venv\Scripts\Activate.ps1"
$env:HF_HUB_OFFLINE = "1"

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method ga --seed 42
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method ga --seed 43
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method ga --seed 44
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Block ga ist vollständig durchgelaufen (Seeds 42, 43, 44)."
