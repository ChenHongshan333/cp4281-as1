[CmdletBinding()]
param(
    [string]$EnvironmentPrefix = "D:\conda-envs\cp4281-as1",
    [string]$VsDevCmd = "D:\Visual Studio\Visual Studio 2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$ColmapDirectory = "D:\Tools\COLMAP-3.11.1"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $EnvironmentPrefix "python.exe"
$cudaRoot = Join-Path $EnvironmentPrefix "Library"
$extensionCache = Join-Path $EnvironmentPrefix "torch_extensions"

foreach ($requiredPath in @($python, $VsDevCmd, (Join-Path $ColmapDirectory "COLMAP.bat"))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required file was not found: $requiredPath"
    }
}

# Import the x64 Visual Studio compiler environment into this PowerShell process.
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

$env:PATH = "$ColmapDirectory;$EnvironmentPrefix;$(Join-Path $EnvironmentPrefix 'Scripts');$(Join-Path $EnvironmentPrefix 'Library\bin');$env:PATH"
$env:LIB = "$(Join-Path $cudaRoot 'lib');$env:LIB"
$env:CUDA_HOME = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:TORCH_EXTENSIONS_DIR = $extensionCache
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS = "2"

& $python (Join-Path $PSScriptRoot "patch_gsplat_windows.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python (Join-Path $PSScriptRoot "verify_installation.py")
exit $LASTEXITCODE
