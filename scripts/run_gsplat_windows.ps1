[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ResultDirectory,

    [int]$MaxSteps = 30000,
    [int[]]$EvalSteps = @(7000, 30000),
    [int[]]$SaveSteps = @(7000, 30000),
    [int]$DataFactor = 1,
    [ValidateRange(0, 2147483647)]
    [int]$RefineStopIter = 15000,
    [switch]$DisableVideo,
    [string]$EnvironmentPrefix = "D:\conda-envs\cp4281-as1",
    [string]$GsplatSource = "F:\CP4281-data\gsplat-1.5.3",
    [string]$VsDevCmd = "D:\Visual Studio\Visual Studio 2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CacheRoot = "F:\CP4281-data\cache"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $EnvironmentPrefix "python.exe"
$cudaRoot = Join-Path $EnvironmentPrefix "Library"
$trainer = Join-Path $GsplatSource "examples\simple_trainer.py"

foreach ($requiredPath in @($DataDirectory, $python, $trainer, $VsDevCmd)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

# Import the Visual Studio x64 compiler environment needed by gsplat's JIT build.
$environmentLines = & $env:ComSpec /d /s /c "call `"$VsDevCmd`" -arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize the Visual Studio developer environment."
}

foreach ($line in $environmentLines) {
    $separator = $line.IndexOf("=")
    if ($separator -gt 0) {
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

$extensionCache = Join-Path $CacheRoot "torch_extensions"
$torchCache = Join-Path $CacheRoot "torch"
$tempPath = Join-Path $CacheRoot "temp"
New-Item -ItemType Directory -Force `
    $ResultDirectory, $extensionCache, $torchCache, $tempPath | Out-Null

$env:PATH = "$EnvironmentPrefix;$(Join-Path $EnvironmentPrefix 'Scripts');$(Join-Path $EnvironmentPrefix 'Library\bin');$env:PATH"
$env:LIB = "$(Join-Path $cudaRoot 'lib');$env:LIB"
$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:TORCH_EXTENSIONS_DIR = $extensionCache
$env:TORCH_HOME = $torchCache
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS = "2"

& $python (Join-Path $PSScriptRoot "patch_gsplat_windows.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python (Join-Path $PSScriptRoot "patch_pycolmap_windows.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python (Join-Path $PSScriptRoot "patch_gsplat_colmap_paths_windows.py") `
    $GsplatSource
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$trainerArguments = @(
    $trainer,
    "default",
    "--data-dir", $DataDirectory,
    "--data-factor", $DataFactor.ToString(),
    "--result-dir", $ResultDirectory,
    "--max-steps", $MaxSteps.ToString(),
    "--strategy.refine-stop-iter", $RefineStopIter.ToString(),
    "--packed",
    "--disable-viewer"
)

$trainerArguments += "--eval-steps"
$trainerArguments += @($EvalSteps | ForEach-Object { $_.ToString() })
$trainerArguments += "--save-steps"
$trainerArguments += @($SaveSteps | ForEach-Object { $_.ToString() })

if ($DisableVideo) {
    $trainerArguments += "--disable-video"
}

$timer = [Diagnostics.Stopwatch]::StartNew()
& $python @trainerArguments
$exitCode = $LASTEXITCODE
$timer.Stop()
$timer.Elapsed.ToString() |
    Set-Content -LiteralPath (Join-Path $ResultDirectory "total_wall_time.txt")

exit $exitCode
