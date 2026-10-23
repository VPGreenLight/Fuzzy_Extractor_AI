# activate.ps1 - Kich hoat moi truong Python 3.11 cho du an Fuzzy Extractor AI
# Chay: .\activate.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvActivate = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"

if (Test-Path $VenvActivate) {
    & $VenvActivate
    Write-Host "[OK] Virtual environment activated: Python" (python --version)
    Write-Host "     python -> $((Get-Command python).Source)"
} else {
    Write-Host "[ERROR] .venv not found. Recreating..."
    $py311 = "C:\Users\Kyn33sa\AppData\Local\Programs\Python\Python311\python.exe"
    & $py311 -m venv .venv --system-site-packages
    & $VenvActivate
    Write-Host "[OK] Virtual environment created and activated."
}
