$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv311"
$cache = Join-Path $root ".cache"
$mplConfig = Join-Path $cache "matplotlib"
$localPython311 = Join-Path $root ".runtime\python311\python.exe"
$bundledPython = "C:\Users\26550\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (Test-Path -LiteralPath $localPython311) {
    $python = $localPython311
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($version -eq "3.11") {
        $python = "python"
    } else {
        throw "Python 3.11 was not found. Install Python 3.11 or install it locally under .runtime\python311."
    }
} elseif (Test-Path -LiteralPath $bundledPython) {
    throw "Only bundled Python was found, but it is not Python 3.11. Install Python 3.11 under .runtime\python311."
} else {
    throw "Python was not found. Install Python 3.11, then rerun this script."
}

if (-not (Test-Path -LiteralPath $venv)) {
    & $python -E -m venv $venv
}

New-Item -ItemType Directory -Force -Path $mplConfig | Out-Null
$env:MPLCONFIGDIR = $mplConfig

$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $root "requirements-rl.txt")

Write-Host "Environment ready: $venv"
Write-Host "Activate with: .\.venv311\Scripts\Activate.ps1"
Write-Host "For plotting, set: `$env:MPLCONFIGDIR='$mplConfig'"
