[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDirectory,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [int]$Frames = 720,
    [int]$Fps = 30,
    [ValidateSet("ellipse", "captured")]
    [string]$PathType = "ellipse",
    [double]$RadiusScale = 0.9,
    [double]$SmoothingSigma = 8.0,
    [string]$EnvironmentPrefix = "D:\conda-envs\cp4281-as1",
    [string]$GsplatSource = "F:\CP4281-data\gsplat-1.5.3",
    [string]$VsDevCmd = "D:\Visual Studio\Visual Studio 2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CacheRoot = "F:\CP4281-data\cache"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $EnvironmentPrefix "python.exe"
$cudaRoot = Join-Path $EnvironmentPrefix "Library"
$renderer = Join-Path $PSScriptRoot "render_smooth_orbit.py"

foreach ($requiredPath in @(
    $DataDirectory, $Checkpoint, $python, $renderer, $GsplatSource, $VsDevCmd
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

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
New-Item -ItemType Directory -Force $extensionCache, $torchCache, $tempPath |
    Out-Null

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

& $python $renderer `
    --data-dir $DataDirectory `
    --checkpoint $Checkpoint `
    --output $OutputPath `
    --gsplat-source $GsplatSource `
    --frames $Frames `
    --fps $Fps `
    --path-type $PathType `
    --radius-scale $RadiusScale `
    --smoothing-sigma $SmoothingSigma

exit $LASTEXITCODE
