# CP4281 Assignment 1: Photos to Gaussian Splats

This repository contains the reproducible pipeline for CP4281 Assignment 1.
It uses COLMAP to estimate camera poses and an initial point cloud, followed by
`gsplat` to train and render a 3D Gaussian Splatting scene.

## Step 1: Environment setup

### Tested system

- Operating system: Windows
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Dedicated GPU memory: 8188 MiB (approximately 8 GB)
- NVIDIA driver: 560.92
- Driver-supported CUDA version reported by `nvidia-smi`: 12.6
- Environment manager: Miniconda/Conda
- Python: 3.11.16

The CUDA version shown by `nvidia-smi` is the maximum version supported by the
installed driver. The PyTorch CUDA runtime and CUDA Toolkit used by `gsplat`
will be documented separately in Step 2.

### Create the environment

Run the following commands from the repository root:

```powershell
cd "D:\NUS CS\Y3 S1\CP4281\cp4281-as1"
conda env create -f environment.yml
conda activate cp4281-as1
```

If PowerShell cannot find `conda`, initialize it from the Miniconda Prompt:

```powershell
conda init powershell
```

Then close and reopen PowerShell before activating the environment.

### Verify the environment

```powershell
python --version
python -c "import sys; print(sys.executable)"
conda env list
```

Expected Python version:

```text
Python 3.11.16
```

The Python executable should be inside the `cp4281-as1` Conda environment, for
example:

```text
C:\Users\<username>\miniconda3\envs\cp4281-as1\python.exe
```

### Update an existing environment

If the environment already exists, synchronize it with the specification using:

```powershell
conda activate cp4281-as1
conda env update -f environment.yml --prune
```

### Dependency files

- `environment.yml` records Python and the Conda-managed build tools.
- `requirements.txt` records packages installed with `pip`.

Step 1 does not yet require any pip-installed runtime packages. PyTorch and
`gsplat` will be added during Step 2, after selecting a compatible CUDA setup.
Whenever pip dependencies change, regenerate their exact versions while the
environment is active:

```powershell
python -m pip freeze > requirements.txt
```
