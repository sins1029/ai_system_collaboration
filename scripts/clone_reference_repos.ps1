param(
    [string]$GitPath = "git"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "references/external_repos"
$gitCommand = Get-Command $GitPath -ErrorAction SilentlyContinue
if (-not $gitCommand -and -not (Test-Path -LiteralPath $GitPath -PathType Leaf)) {
    throw "未找到 Git：$GitPath。请安装 Git，或用 -GitPath 指定 git.exe。"
}

New-Item -ItemType Directory -Force -Path $target | Out-Null

$repos = @(
    @{ Name = "sustaindc"; Url = "https://github.com/HewlettPackard/dc-rl.git" },
    @{ Name = "compopt"; Url = "https://github.com/HewlettPackard/compopt.git" },
    @{ Name = "clusterdata"; Url = "https://github.com/alibaba/clusterdata.git" },
    @{ Name = "CarbonScaler"; Url = "https://github.com/umassos/CarbonScaler.git" },
    @{ Name = "sustain-cluster"; Url = "https://github.com/HewlettPackard/sustain-cluster.git" }
)

foreach ($repo in $repos) {
    $destination = Join-Path $target $repo.Name
    if (Test-Path -LiteralPath $destination) {
        Write-Host "跳过已存在的参考仓库 $($repo.Name)：$destination"
        continue
    }
    & $GitPath clone --depth 1 $repo.Url $destination
    if ($LASTEXITCODE -ne 0) {
        throw "下载参考仓库失败：$($repo.Name)"
    }
}
