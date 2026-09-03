# CP4281 Assignment 1: Photos to Gaussian Splats

This repository contains the reproducible environment and verification for a
pipeline that uses COLMAP for camera reconstruction and `gsplat` for 3D
Gaussian Splatting.

## Tested system

- Windows with an NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)
- NVIDIA driver 560.92 (`nvidia-smi` reports CUDA compatibility up to 12.6)
- Visual Studio 2022 Community with Desktop development with C++
- Miniconda, with the project environment and package caches stored on drive D
- Python 3.11.16, PyTorch 2.7.1+cu126, CUDA Toolkit 12.6.3
- gsplat 1.5.3 and COLMAP 3.11.1 CUDA build

The CUDA version printed by `nvidia-smi` is the maximum CUDA version supported
by the installed driver. The CUDA runtime bundled with PyTorch and the Toolkit
used to compile gsplat are both version 12.6 in this setup.

## Step 1: create the environment on drive D

The following are the exact environment creation commands. Using an explicit
prefix keeps the large environment off drive C.

```powershell
cd "D:\NUS CS\Y3 S1\CP4281\cp4281-as1"
conda env create --prefix "D:\conda-envs\cp4281-as1" --file environment.yml
conda activate "D:\conda-envs\cp4281-as1"
python --version
python -c "import sys; print(sys.executable)"
```

Expected executable:

```text
D:\conda-envs\cp4281-as1\python.exe
```

To synchronize an environment that already exists:

```powershell
conda env update --prefix "D:\conda-envs\cp4281-as1" --file environment.yml --prune
```

`environment.yml` contains the Conda-managed compiler tools and CUDA Toolkit.
`requirements.txt` contains the exact pip package versions.

## Step 2: install PyTorch, gsplat, and COLMAP

### Python packages

With the D-drive environment activated, install the pinned packages:

```powershell
python -m pip install --requirement requirements.txt
```

Save the CUDA and JIT cache locations in this Conda environment, then reactivate
it so that the values take effect:

```powershell
conda env config vars set --prefix "D:\conda-envs\cp4281-as1" `
  "CUDA_HOME=D:\conda-envs\cp4281-as1\Library" `
  "CUDA_PATH=D:\conda-envs\cp4281-as1\Library" `
  "TORCH_EXTENSIONS_DIR=D:\conda-envs\cp4281-as1\torch_extensions" `
  "TORCH_CUDA_ARCH_LIST=8.9"
conda deactivate
conda activate "D:\conda-envs\cp4281-as1"
```

`gsplat==1.5.3` passes the GCC-only `-Wno-attributes` option to Microsoft's C++
compiler. `scripts/patch_gsplat_windows.py` applies the small Windows-specific
compiler-flag correction before the first CUDA build. It refuses to modify an
unrecognized gsplat source version.

### COLMAP

The official CUDA-enabled Windows archive used here is
`colmap-x64-windows-cuda.zip` from the COLMAP 3.11.1 release. These commands
download and extract it entirely on drive D:

```powershell
$zip = "D:\Downloads\colmap-x64-windows-cuda.zip"
$install = "D:\Tools\COLMAP-3.11.1"
New-Item -ItemType Directory -Force "D:\Downloads", "D:\Tools" | Out-Null
Invoke-WebRequest `
  "https://github.com/colmap/colmap/releases/download/3.11.1/colmap-x64-windows-cuda.zip" `
  -OutFile $zip
Expand-Archive -LiteralPath $zip -DestinationPath $install
```

Add the folder containing `COLMAP.bat` to the user PATH once:

```powershell
$colmap = "D:\Tools\COLMAP-3.11.1"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (($userPath -split ";") -notcontains $colmap) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$colmap", "User")
}
```

Close and reopen VS Code after changing PATH. Existing terminals do not receive
the updated user PATH automatically.

### Build and verify

Run the wrapper from the repository root. It loads the Visual Studio x64 build
environment, applies the safe gsplat 1.5.3 Windows patch, and performs a real
CUDA rasterization. The first run can take several minutes; later runs reuse the
compiled extension on drive D.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_windows.ps1
```

The wrapper paths can be overridden if software is installed elsewhere:

```powershell
.\scripts\verify_windows.ps1 `
  -EnvironmentPrefix "D:\conda-envs\cp4281-as1" `
  -VsDevCmd "D:\Visual Studio\Visual Studio 2022\Community\Common7\Tools\VsDevCmd.bat" `
  -ColmapDirectory "D:\Tools\COLMAP-3.11.1"
```

The successful output is saved in `step2_verification.txt`. It demonstrates
that PyTorch sees the NVIDIA GPU, `gsplat.rasterization` executes without error,
and COLMAP reports its version and CUDA-enabled build.
