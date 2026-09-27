$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
& ".venv\Scripts\Activate.ps1"
$env:HF_HUB_OFFLINE = "1"

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method default --smoke --qa 20
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method grid --smoke --qa 20 --trials 24
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method random --smoke --qa 20 --trials 24
if ($LASTEXITCODE -ne 0) { exit 1 }

python src\experiment_runner\run.py --config configs\experiment_config.yaml --method ga --smoke --qa 20 --pop 6 --gen 5
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "Alle vier Medium-Smoke-Läufe sind erfolgreich durchgelaufen."
