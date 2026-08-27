param(
    [string]$PythonPath = "python",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv-sustain-cluster"
$cache = Join-Path $root ".cache"
$mplConfig = Join-Path $cache "matplotlib"
$externalRequirements = Join-Path $root "references\external_repos\sustain-cluster\requirements.txt"

$pythonCommand = Get-Command $PythonPath -ErrorAction SilentlyContinue
if (-not $pythonCommand -and -not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "未找到 Python：$PythonPath。请安装 Python 3.10+，或用 -PythonPath 指定 python.exe。"
}
$versionOk = & $PythonPath -c "import sys; print(int(sys.version_info >= (3, 10)))"
if ($versionOk -ne "1") {
    throw "项目要求 Python 3.10 或更高版本。"
}
if (-not (Test-Path -LiteralPath $externalRequirements -PathType Leaf)) {
    throw "未找到 SustainCluster 依赖清单：$externalRequirements"
}
if ($DryRun) {
    Write-Host "[DryRun] 将创建或复用环境：$venv"
    Write-Host "[DryRun] 将安装 SustainCluster 依赖：$externalRequirements"
    Write-Host "[DryRun] 将以 research,test 模式安装本项目：$root"
    exit 0
}

if (-not (Test-Path -LiteralPath $venv)) {
    & $PythonPath -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "创建 Python 环境失败：$venv"
    }
}

New-Item -ItemType Directory -Force -Path $mplConfig | Out-Null
$env:MPLCONFIGDIR = $mplConfig

$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "升级 pip/setuptools/wheel 失败。"
}
& $venvPython -m pip install -r $externalRequirements
if ($LASTEXITCODE -ne 0) {
    throw "安装 SustainCluster 依赖失败：$externalRequirements"
}
& $venvPython -m pip install -e "$root[research,test]"
if ($LASTEXITCODE -ne 0) {
    throw "安装主项目依赖失败：$root"
}

Write-Host "环境已就绪：$venv"
Write-Host "激活命令：.\.venv-sustain-cluster\Scripts\Activate.ps1"
