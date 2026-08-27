param(
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv-sustain-cluster"
$cache = Join-Path $root ".cache"
$mplConfig = Join-Path $cache "matplotlib"

$pythonCommand = Get-Command $PythonPath -ErrorAction SilentlyContinue
if (-not $pythonCommand -and -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "未找到 Python：$PythonPath。请安装 Python 3.10+，或用 -PythonPath 指定 python.exe。"
}
$versionOk = & $PythonPath -c "import sys; print(int(sys.version_info >= (3, 10)))"
if ($versionOk -ne "1") {
    throw "项目要求 Python 3.10 或更高版本。"
}

if (-not (Test-Path -LiteralPath $venv)) {
    & $PythonPath -m venv $venv
}

New-Item -ItemType Directory -Force -Path $mplConfig | Out-Null
$env:MPLCONFIGDIR = $mplConfig

$venvPython = Join-Path $venv "Scripts\python.exe"
$externalRequirements = Join-Path $root "references\external_repos\sustain-cluster\requirements.txt"
if (-not (Test-Path -LiteralPath $externalRequirements -PathType Leaf)) {
    throw "未找到 SustainCluster 依赖清单：$externalRequirements"
}
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r $externalRequirements
& $venvPython -m pip install -e "$root[research,test]"

Write-Host "环境已就绪：$venv"
Write-Host "激活命令：.\.venv-sustain-cluster\Scripts\Activate.ps1"
