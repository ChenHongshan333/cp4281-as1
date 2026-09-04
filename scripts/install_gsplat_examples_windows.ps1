[CmdletBinding()]
param(
    [string]$EnvironmentPrefix = "D:\conda-envs\cp4281-as1",
    [string]$GsplatSource = "F:\CP4281-data\gsplat-1.5.3",
    [string]$VsDevCmd = "D:\Visual Studio\Visual Studio 2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CacheRoot = "F:\CP4281-data\cache"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $EnvironmentPrefix "python.exe"
$cudaRoot = Join-Path $EnvironmentPrefix "Library"
$requirements = Join-Path $PSScriptRoot "..\requirements-step3.txt"

foreach ($requiredPath in @($python, $VsDevCmd, $requirements)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required path does not exist: $requiredPath"
    }
}

$pipCache = Join-Path $CacheRoot "pip"
$tempPath = Join-Path $CacheRoot "temp"
$extensionCache = Join-Path $CacheRoot "torch_extensions"
New-Item -ItemType Directory -Force $pipCache, $tempPath, $extensionCache | Out-Null

# Import the Visual Studio x64 compiler environment into this PowerShell process.
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

$env:PATH = "$EnvironmentPrefix;$(Join-Path $EnvironmentPrefix 'Scripts');$(Join-Path $EnvironmentPrefix 'Library\bin');$env:PATH"
$env:LIB = "$(Join-Path $cudaRoot 'lib');$env:LIB"
$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:PIP_CACHE_DIR = $pipCache
$env:TEMP = $tempPath
$env:TMP = $tempPath
$env:TORCH_EXTENSIONS_DIR = $extensionCache
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS = "2"

& $python -m pip install --no-build-isolation --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python (Join-Path $PSScriptRoot "patch_pycolmap_windows.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -c "import cv2, fused_ssim, imageio, matplotlib, nerfview, pycolmap, sklearn, splines, tensorboard, tensorly, torchmetrics, tqdm, tyro, viser, yaml; print('gsplat example dependencies: OK')"
exit $LASTEXITCODE
