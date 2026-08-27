param(
    [string]$GitPath = "git",
    [switch]$AllReferences,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "references/external_repos"
$gitCommand = Get-Command $GitPath -ErrorAction SilentlyContinue
if (-not $gitCommand -and -not (Test-Path -LiteralPath $GitPath -PathType Leaf)) {
    throw "未找到 Git：$GitPath。请安装 Git，或用 -GitPath 指定 git.exe。"
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $target | Out-Null
}

$coreRepo = @{
    Name = "sustain-cluster"
    Url = "https://github.com/HewlettPackard/sustain-cluster.git"
    Commit = "3f6ea95cb835b89ba50b0ef76d66d14b8037643e"
}
$optionalRepos = @(
    @{ Name = "sustaindc"; Url = "https://github.com/HewlettPackard/dc-rl.git" },
    @{ Name = "compopt"; Url = "https://github.com/HewlettPackard/compopt.git" },
    @{ Name = "clusterdata"; Url = "https://github.com/alibaba/clusterdata.git" },
    @{ Name = "CarbonScaler"; Url = "https://github.com/umassos/CarbonScaler.git" }
)
$repos = @($coreRepo)
if ($AllReferences) {
    $repos += $optionalRepos
}

foreach ($repo in $repos) {
    $destination = Join-Path $target $repo.Name
    if (Test-Path -LiteralPath $destination) {
        if (-not (Test-Path -LiteralPath (Join-Path $destination ".git"))) {
            throw "目标目录已存在但不是 Git 仓库：$destination"
        }
        if ($repo.Commit) {
            $status = & $GitPath -C $destination status --porcelain
            if ($LASTEXITCODE -ne 0) {
                throw "无法检查 SustainCluster 状态：$destination"
            }
            if ($status) {
                throw "SustainCluster 工作区存在本地修改，请先人工处理：$destination"
            }
            $currentCommit = & $GitPath -C $destination rev-parse HEAD
            if ($LASTEXITCODE -ne 0 -or $currentCommit -ne $repo.Commit) {
                throw "SustainCluster commit 不符合要求。期望：$($repo.Commit)；实际：$currentCommit"
            }
        }
        Write-Host "参考仓库已存在并通过检查 $($repo.Name)：$destination"
        continue
    }
    if ($DryRun) {
        Write-Host "[DryRun] 将克隆 $($repo.Url) 到 $destination"
        if ($repo.Commit) {
            Write-Host "[DryRun] 将固定到 commit $($repo.Commit)"
        }
        continue
    }
    if ($repo.Commit) {
        & $GitPath clone $repo.Url $destination
    } else {
        & $GitPath clone --depth 1 $repo.Url $destination
    }
    if ($LASTEXITCODE -ne 0) {
        throw "下载参考仓库失败：$($repo.Name)"
    }
    if ($repo.Commit) {
        & $GitPath -C $destination checkout --detach $repo.Commit
        if ($LASTEXITCODE -ne 0) {
            throw "无法固定 SustainCluster commit：$($repo.Commit)"
        }
    }
}

if (-not $AllReferences) {
    Write-Host "核心依赖已准备。其他四个研究参考仓库可使用 -AllReferences 下载。"
}
