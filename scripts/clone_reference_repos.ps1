$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root "references/external_repos"
$bundledGit = "C:\Users\26550\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe"
$git = "git"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    if (Test-Path -LiteralPath $bundledGit) {
        $git = $bundledGit
    } else {
        throw "Git was not found in PATH. Install Git or update this script with your git.exe path."
    }
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
        Write-Host "Skip existing $($repo.Name): $destination"
        continue
    }
    & $git clone --depth 1 $repo.Url $destination
}
