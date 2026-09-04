[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ImagePath,

    [Parameter(Mandatory = $true)]
    [string]$WorkspacePath,

    [string]$ColmapLauncher = "D:\Tools\COLMAP-3.11.1\COLMAP.bat"
)

$ErrorActionPreference = "Stop"

foreach ($requiredPath in @($ImagePath, $ColmapLauncher)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

$databasePath = Join-Path $WorkspacePath "database.db"
$sparsePath = Join-Path $WorkspacePath "sparse"

if (Test-Path -LiteralPath $databasePath) {
    throw "Workspace already contains a database: $databasePath"
}

if ((Test-Path -LiteralPath $sparsePath) -and
    (Get-ChildItem -LiteralPath $sparsePath -Force | Select-Object -First 1)) {
    throw "Sparse output directory is not empty: $sparsePath"
}

New-Item -ItemType Directory -Force $WorkspacePath, $sparsePath | Out-Null

function Invoke-ColmapStage {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    Write-Host "=== $Name ==="
    $timer = [Diagnostics.Stopwatch]::StartNew()
    & $Command
    $exitCode = $LASTEXITCODE
    $timer.Stop()

    $timePath = Join-Path $WorkspacePath ($Name + "_time.txt")
    $timer.Elapsed.ToString() | Set-Content -LiteralPath $timePath

    if ($exitCode -ne 0) {
        throw "COLMAP stage '$Name' failed with exit code $exitCode"
    }
}

Invoke-ColmapStage -Name "feature_extraction" -Command {
    & $ColmapLauncher feature_extractor `
        --database_path $databasePath `
        --image_path $ImagePath `
        --ImageReader.single_camera 1
}

Invoke-ColmapStage -Name "feature_matching" -Command {
    & $ColmapLauncher exhaustive_matcher `
        --database_path $databasePath
}

Invoke-ColmapStage -Name "mapping" -Command {
    & $ColmapLauncher mapper `
        --database_path $databasePath `
        --image_path $ImagePath `
        --output_path $sparsePath
}

Get-ChildItem -LiteralPath $sparsePath -Directory | ForEach-Object {
    Write-Host "=== Model $($_.Name) ==="
    & $ColmapLauncher model_analyzer --path $_.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Could not analyze COLMAP model: $($_.FullName)"
    }
}
