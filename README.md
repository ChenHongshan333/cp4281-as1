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
`requirements.txt` contains the core pip package versions, while
`requirements-step3.txt` contains the additional pinned training dependencies.

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

## Step 3: reconstruct three sample scenes

### Scene selection and storage

The selected scenes are `truck`, `train`, and `drjohnson`. This gives two
different outdoor scenes and one indoor scene while requiring only the combined
Tanks and Temples + Deep Blending download.

Large inputs and generated files are kept outside this Git repository on drive
F:

```text
F:\CP4281-data\sample_scenes\
|- tandt_db\                 # extracted source bundle
`- our_reconstruction\       # COLMAP and training outputs produced by us
```

The dataset was downloaded and extracted with:

```powershell
Invoke-WebRequest `
  -Uri "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip" `
  -OutFile "F:\CP4281-data\sample_scenes\tandt_db.zip"

Expand-Archive `
  -LiteralPath "F:\CP4281-data\sample_scenes\tandt_db.zip" `
  -DestinationPath "F:\CP4281-data\sample_scenes\tandt_db"
```

Each downloaded scene contains an `images` directory and a supplied `sparse`
model. Only `images` is used as input. The supplied model is deliberately
ignored because the assignment requires recovering the camera poses with
COLMAP ourselves.

### COLMAP sparse reconstruction

The first scene processed was `truck`, which contains 251 images. Its new
workspace was created separately from the supplied reconstruction:

```powershell
New-Item -ItemType Directory -Force `
  "F:\CP4281-data\sample_scenes\our_reconstruction\truck\sparse" |
  Out-Null
```

The following commands were then run in order.

Feature extraction:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" feature_extractor `
  --database_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\database.db" `
  --image_path "F:\CP4281-data\sample_scenes\tandt_db\tandt\truck\images" `
  --ImageReader.single_camera 1
```

Exhaustive feature matching:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" exhaustive_matcher `
  --database_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\database.db"
```

Incremental mapping:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" mapper `
  --database_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\database.db" `
  --image_path "F:\CP4281-data\sample_scenes\tandt_db\tandt\truck\images" `
  --output_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\sparse"
```

The reusable wrapper [scripts/run_colmap_sparse_windows.ps1](scripts/run_colmap_sparse_windows.ps1)
runs these same three stages for a new scene and saves the wall-clock time for
each stage. It intentionally refuses to reuse a non-empty workspace, preventing
an old reconstruction from being mistaken for a new run.

### Current `truck` result

COLMAP produced two models. They were inspected with:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" model_analyzer `
  --path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\sparse\0"

& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" model_analyzer `
  --path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\sparse\1"
```

| Stage or model | Result |
|---|---:|
| Feature extraction time | 00:00:11.4588259 |
| Exhaustive matching time | 00:01:13.8169435 |
| Incremental mapping time | 00:07:26.9746256 |
| Model 0 | 2 registered images, 535 points |
| Model 1 | 251/251 registered images, 63,813 points |
| Model 1 mean track length | 7.823343 |
| Model 1 mean reprojection error | 0.588350 px |

Model 1 is the complete reconstruction and will be used for Gaussian Splatting
training. Model 0 is a redundant two-image reconstruction. Thus COLMAP emitted
more than one model, but the main model itself is not fragmented: it contains
all 251 input images.

The selected model was then undistorted into a separate gsplat-ready dataset.
The maximum image dimension was limited to 1600 pixels to reduce training VRAM
usage on the 8 GB GPU:

```powershell
New-Item -ItemType Directory -Force `
  "F:\CP4281-data\sample_scenes\our_reconstruction\truck\processed" |
  Out-Null

& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" image_undistorter `
  --image_path "F:\CP4281-data\sample_scenes\tandt_db\tandt\truck\images" `
  --input_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\sparse\1" `
  --output_path "F:\CP4281-data\sample_scenes\our_reconstruction\truck\processed" `
  --output_type COLMAP `
  --max_image_size 1600
```

Undistortion took `00:00:03.7252068`. The resulting `processed/images`
directory contains all 251 images, and `processed/sparse` contains the adjusted
`cameras.bin`, `images.bin`, and `points3D.bin` files. The generated `stereo`
directory is not needed for Gaussian Splatting training.

### Training dependency setup

The official gsplat example source matching the installed package version was
checked out on drive F:

```powershell
git clone --branch v1.5.3 --depth 1 `
  https://github.com/nerfstudio-project/gsplat.git `
  "F:\CP4281-data\gsplat-1.5.3"
```

The first attempt to install `examples/requirements.txt` failed while compiling
the optional `fused_bilagrid` CUDA extension. Its `dim3` list initialization
caused narrowing-conversion errors with MSVC. This extension is imported only
when the experimental `--use_fused_bilagrid` option is enabled, which is not
part of the baseline configuration. The Windows installer therefore excludes
only that optional dependency and retains the standard bilateral-grid Python
implementation and all dependencies needed by the default trainer:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\install_gsplat_examples_windows.ps1"
```

The pinned legacy `pycolmap` reader also used the platform-native unsigned-long
format for eight-byte fields in COLMAP binary files. Since unsigned long is
only four bytes on Windows, the initial trainer smoke test failed with
`struct.error: unpack requires a buffer of 4 bytes`. The installer now runs
`scripts/patch_pycolmap_windows.py`, which changes those readers to explicit
little-endian COLMAP field sizes and refuses to patch unrecognized source.

After fixing the reader, the next smoke-test attempt loaded the reconstruction
but failed at the first rasterization because `cl.exe` was not available in the
interactive PowerShell environment. `scripts/run_gsplat_windows.ps1` now loads
the Visual Studio x64 build environment before launching the trainer. It also
places Torch downloads, temporary files, and JIT-compiled CUDA extensions under
`F:\CP4281-data\cache`.

### `truck` training smoke test

After applying both Windows fixes, a ten-step smoke test was run to validate the
entire path from our COLMAP model to CUDA rasterization and held-out evaluation:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_gsplat_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\truck\processed" `
  -ResultDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\truck\smoke_test" `
  -MaxSteps 10 `
  -EvalSteps 10 `
  -SaveSteps 10 `
  -DisableVideo
```

The parser loaded all 251 cameras and initialized the model from the 63,813
COLMAP points. The first rasterization compiled the gsplat CUDA extension in
691.57 seconds; subsequent runs reuse the compiled extension on drive F. The
test then completed all ten optimization steps and evaluated 32 held-out views.

| Smoke-test measurement | Result |
|---|---:|
| Total wall-clock time | 00:12:55.9871051 |
| Peak allocated VRAM | 0.1542 GiB |
| Gaussian count | 63,813 |
| Held-out PSNR | 12.3387 dB |
| Held-out SSIM | 0.49527 |
| Held-out LPIPS | 0.86504 |

These quality metrics are only a pipeline check after ten steps and are not the
final `truck` results. The message `Warning: image_path not found for
reconstruction` comes from the legacy parser looking for an optional COLMAP
`project.ini`; the trainer obtains the actual image directory from `data_dir`,
as confirmed by successfully loading and evaluating all images.

### `truck` final training

The final model was trained for 30,000 steps with the following command. Only
the final step was evaluated and saved, avoiding an unnecessary intermediate
evaluation and video while retaining the final metrics and a trajectory preview:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_gsplat_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\truck\processed" `
  -ResultDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\truck\training" `
  -MaxSteps 30000 `
  -EvalSteps 30000 `
  -SaveSteps 30000
```

The baseline used SfM initialization, degree-3 spherical harmonics, world-space
normalization, batch size 1, and packed rasterization. Every eighth image was
held out, giving 32 validation views. The default densification strategy began
refinement at step 500, split or duplicated Gaussians every 100 steps, reset
opacity every 3,000 steps, pruned low-opacity Gaussians, and stopped refinement
at step 15,000.

| Final `truck` measurement | Result |
|---|---:|
| Training steps | 30,000 |
| Initial Gaussians (COLMAP points) | 63,813 |
| Final Gaussians | 2,480,237 |
| Held-out views | 32 |
| Held-out PSNR | 26.0812 dB |
| Held-out SSIM | 0.89677 |
| Held-out LPIPS | 0.09587 |
| Peak allocated VRAM | 3.632 GiB |
| Trainer time | 2,357.86 s (39 min 17.86 s) |
| Total wall-clock time | 00:40:48.7897638 |
| Final checkpoint size | 558.22 MiB |
| Trajectory preview size | 6.43 MiB |
| Trajectory preview | 240 frames, 30 FPS, 8.0 s |

The run produced `ckpt_29999_rank0.pt`, 32 paired validation images, the final
statistics JSON files, and `traj_29999.mp4`. The checkpoint was loaded back
successfully and contains step 29,999 with a `(2480237, 3)` Gaussian-center
tensor. The MP4 was also decoded successfully. Its 8-second duration makes it a
preview rather than the required 20--30 second submission video; a compliant
sample-scene video will be produced separately. A visual check of
`val_step29999_0000.png` confirmed that the rendered half contains the complete
truck, trees, and background geometry and closely matches the held-out image;
the ground and some fine textures are smoother than the reference. Large
outputs remain on drive F and are excluded from Git as required.

### Stable submission video

The trainer's default `interp` trajectory follows the original handheld camera
poses. It generated only 240 frames (8 seconds at 30 FPS) and retained visible
camera wobble. `scripts/render_smooth_orbit.py` instead constructs a closed
ellipse, resamples it to near-constant speed, fixes the up direction, and keeps
the camera aimed at one focus point. It also outputs RGB only rather than the
trainer's side-by-side RGB and depth preview.

The Windows wrapper loads the existing final checkpoint without retraining and
renders 720 frames, producing a 24-second video at 30 FPS:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\render_smooth_orbit_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\truck\processed" `
  -Checkpoint "F:\CP4281-data\sample_scenes\our_reconstruction\truck\training\ckpts\ckpt_29999_rank0.pt" `
  -OutputPath "F:\CP4281-data\sample_scenes\our_reconstruction\truck\training\videos\truck_smooth_24s.mp4" `
  -Frames 720 `
  -Fps 30 `
  -RadiusScale 0.9
```

The renderer writes a JSON file beside the MP4 containing the frame count,
duration, resolution, orbit radii, focus point, and rendering time.

The completed `truck_smooth_24s.mp4` was visually inspected and has smooth,
stable framing. It contains 720 readable frames at 30 FPS (24.0 seconds), took
20.21 seconds to render, and occupies 5.37 MiB. The RGB render resolution was
977x544; the H.264 encoder padded the stored video width to 984 pixels. This
video satisfies the assignment's 20--30 second duration requirement.

### `train` COLMAP reconstruction

The second sample scene was reconstructed from its 301 input images with the
reusable sparse-reconstruction wrapper:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_colmap_sparse_windows.ps1" `
  -ImagePath "F:\CP4281-data\sample_scenes\tandt_db\tandt\train\images" `
  -WorkspacePath "F:\CP4281-data\sample_scenes\our_reconstruction\train"
```

COLMAP produced one connected model (`sparse/0`) and registered all 301 images,
so no scene fragmentation occurred.

| `train` COLMAP measurement | Result |
|---|---:|
| Feature extraction time | 00:00:10.4806269 |
| Exhaustive matching time | 00:02:25.1030334 |
| Incremental mapping time | 00:09:11.1440604 |
| Registered images | 301/301 |
| Sparse points | 97,753 |
| Mean track length | 7.043600 |
| Mean observations per image | 2,287.485050 |
| Mean reprojection error | 0.518577 px |

The complete model was prepared for gsplat with:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" image_undistorter `
  --image_path "F:\CP4281-data\sample_scenes\tandt_db\tandt\train\images" `
  --input_path "F:\CP4281-data\sample_scenes\our_reconstruction\train\sparse\0" `
  --output_path "F:\CP4281-data\sample_scenes\our_reconstruction\train\processed" `
  --output_type COLMAP `
  --max_image_size 1600
```

Undistortion took `00:00:04.5943167`. The processed dataset contains all 301
images and its adjusted `cameras.bin`, `images.bin`, and `points3D.bin` files.

### `train` final training

The scene was trained with the same 30,000-step baseline used for `truck`:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_gsplat_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\train\processed" `
  -ResultDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\train\training" `
  -MaxSteps 30000 `
  -EvalSteps 30000 `
  -SaveSteps 30000
```

| Final `train` measurement | Result |
|---|---:|
| Training steps | 30,000 |
| Initial Gaussians (COLMAP points) | 97,753 |
| Final Gaussians | 1,325,402 |
| Held-out views | 38 |
| Held-out PSNR | 22.0377 dB |
| Held-out SSIM | 0.83673 |
| Held-out LPIPS | 0.13903 |
| Peak allocated VRAM | 2.008 GiB |
| Trainer time | 1,560.92 s (26 min 0.92 s) |
| Total wall-clock time | 00:27:32.3807221 |
| Final checkpoint size | 298.31 MiB |
| Default trajectory preview size | 7.96 MiB |

The run produced a final checkpoint, 38 paired validation renders, evaluation
statistics, and the default trajectory preview.

### `train` stable submission video and artifacts

The ellipse trajectory that worked for the compact `truck` scene was not
suitable for the long `train` scene. It moved outside the region covered by the
captured cameras; later views passed through poorly reconstructed ground and
showed noticeable camera roll. The renderer was therefore extended with a
`captured` path mode. This mode follows the original camera route, applies
circular Gaussian smoothing to position and viewing direction, resamples by
arc length for near-constant speed, and uses the capture's average up vector.

The replacement video was rendered without retraining:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\render_smooth_orbit_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\train\processed" `
  -Checkpoint "F:\CP4281-data\sample_scenes\our_reconstruction\train\training\ckpts\ckpt_29999_rank0.pt" `
  -OutputPath "F:\CP4281-data\sample_scenes\our_reconstruction\train\training\videos\train_captured_smooth_24s.mp4" `
  -Frames 720 `
  -Fps 30 `
  -PathType captured `
  -SmoothingSigma 8
```

The resulting RGB video was visually inspected and has broadly stable camera
motion without the underground path failure. It contains 720 readable frames
at 30 FPS (24.0 seconds), took 16.57 seconds to render, occupies 8.85 MiB, and
is stored at 984x544 pixels after codec padding.

Some views contain dark, fog-like floaters. These are not real fog: they are
semi-transparent Gaussians with inaccurate geometry or poorly constrained dark
colors in shadowed and sparsely observed regions. Where accumulated opacity is
insufficient, the renderer's black background can also show through. The train
undercarriage, thin structures, and novel views between captured cameras are
particularly susceptible. This is retained as an honest reconstruction
weakness for the report's failure analysis.
