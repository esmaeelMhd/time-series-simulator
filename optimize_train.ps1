param(
  [string]$Config = "configs/wastewater.yaml",
  [string[]]$Models = @("tft","transformer","dlinear","nlinear"),
  [string]$RunDir = "runs/wastewater/full"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$modelsArg = $Models -join " "
$outLog = Join-Path $RunDir "night_run.log"
$errLog = Join-Path $RunDir "night_run.err.log"

$env:PYTHONUNBUFFERED = "1"

python -u scripts/optimize.py --config $Config --models $modelsArg *>> $outLog 2>> $errLog
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -u scripts/train.py --config $Config --models $modelsArg --use-optuna-best-params *>> $outLog 2>> $errLog
exit $LASTEXITCODE

# Start-Process powershell -WorkingDirectory "c:\Users\smoha\Desktop\Github\time-series-simulator" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File scripts/optimize_train.ps1"
# Get-Content runs/wastewater/full/night_run.log -Wait