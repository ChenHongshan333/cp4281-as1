# CP4281 Assignment 1: Photos to Gaussian Splats

This repository contains the reproducible environment and verification for a
pipeline that uses COLMAP for camera reconstruction and `gsplat` for 3D
Gaussian Splatting.

## Tested system

- AMD Ryzen 7 8745H CPU and 15.31 GiB physical RAM
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

### `drjohnson` COLMAP reconstruction

The third sample is an indoor scene containing 263 images. Its sparse
reconstruction was run with:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_colmap_sparse_windows.ps1" `
  -ImagePath "F:\CP4281-data\sample_scenes\tandt_db\db\drjohnson\images" `
  -WorkspacePath "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson"
```

COLMAP emitted a redundant two-image model (`sparse/0`) and a complete model
(`sparse/1`) containing all 263 input images. The complete model is used for
subsequent processing.

| `drjohnson` COLMAP measurement | Result |
|---|---:|
| Feature extraction time | 00:00:11.6271974 |
| Exhaustive matching time | 00:01:39.8161231 |
| Incremental mapping time | 00:05:36.2886419 |
| Model 0 | 2 registered images, 37 points |
| Model 1 | 263/263 registered images, 79,299 points |
| Model 1 mean track length | 4.240344 |
| Model 1 mean observations per image | 1,278.536122 |
| Model 1 mean reprojection error | 0.581401 px |

The complete model was prepared for training with:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" image_undistorter `
  --image_path "F:\CP4281-data\sample_scenes\tandt_db\db\drjohnson\images" `
  --input_path "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\sparse\1" `
  --output_path "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\processed" `
  --output_type COLMAP `
  --max_image_size 1600
```

Undistortion took `00:00:06.7060572`. The processed dataset contains all 263
images and the adjusted COLMAP binary model.

### `drjohnson` first training attempt: out of memory

The first 30,000-step attempt used the same baseline densification schedule as
the outdoor scenes, with refinement planned through step 15,000. It terminated
after `00:21:22.2733220` before producing a checkpoint. The last TensorBoard
entry was at step 10,600: the model had grown from 79,299 to 3,947,098 Gaussians
and PyTorch reported 5.827 GiB of allocated GPU memory. The next densification
operation also requires temporary memory, so this growth left insufficient
headroom on the 8 GiB GPU.

Other open applications, especially hardware-accelerated browser tabs, may
have consumed additional GPU memory and contributed to the failure. However,
the rapidly increasing Gaussian count was the main reproducible risk: the
original schedule would have continued densification for another 4,400 steps.
The failed output directory is retained as diagnostic evidence.

For the retry, `scripts/run_gsplat_windows.ps1` exposes the strategy's
`RefineStopIter` setting. Densification is stopped at step 8,000, where the
failed run had approximately 3.07 million Gaussians and 4.45 GiB allocated.
Optimization then continues to step 30,000 without further Gaussian growth.
Final evaluation and checkpointing are performed at step 30,000, and video
rendering is deferred to the separate stable-path renderer.

### `drjohnson` memory-safe training result

The scene was restarted from the beginning after closing other GPU-heavy
applications. The successful retry stopped densification at step 8,000 and
disabled the trainer's default trajectory video:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_gsplat_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\processed" `
  -ResultDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\training_memory_safe" `
  -MaxSteps 30000 `
  -EvalSteps 30000 `
  -SaveSteps 30000 `
  -RefineStopIter 8000 `
  -DisableVideo
```

| Final `drjohnson` measurement | Result |
|---|---:|
| Training steps | 30,000 |
| Initial Gaussians (COLMAP points) | 79,299 |
| Final Gaussians | 3,081,715 |
| Held-out views | 33 |
| Held-out PSNR | 28.6254 dB |
| Held-out SSIM | 0.89897 |
| Held-out LPIPS | 0.17974 |
| Peak allocated VRAM | 4.465 GiB |
| Trainer time | 2,867.41 s (47 min 47.41 s) |
| Total wall-clock time | 00:49:38.4779878 |
| Final checkpoint size | 693.60 MiB |

The final checkpoint is `ckpts/ckpt_29999_rank0.pt`. It contains 3,081,715
Gaussians and was used successfully for subsequent rendering. The run also
produced 33 paired held-out validation renders and final training and validation
statistics. Compared with the failed attempt, stopping refinement at step 8,000
kept the Gaussian count and memory usage bounded while allowing the remaining
optimization steps to improve the fixed representation.

### `drjohnson` stable handheld video

The first stable-path attempt used one fixed up vector for every frame. This is
invalid for this indoor capture because several cameras look nearly parallel to
that vector, and the renderer stopped with `Estimated view direction is too
close to the up direction`. A second `captured` attempt transported the up
direction smoothly, but its view trajectory was still visually unsuitable.

Inspection of the COLMAP poses explained the problem: ordering all 263 cameras
by image name does not produce one continuous video-like route. Adjacent source
views change by a median of 35.8 degrees and by as much as 153.4 degrees, and
camera positions contain several large jumps between separate capture runs.
Smoothing all of those original viewing directions therefore caused the video
to alternate between the ceiling and floor.

The renderer was extended with a `handheld` path mode. It splits the input at
large positional jumps, selects the longest continuous capture run, smooths and
constant-speed resamples its real camera positions, estimates the room's
vertical direction from the camera-center distribution, and keeps the view
aimed at the reconstructed scene focus. For this dataset it selected camera
indices `[0, 79)`, corresponding to `IMG_6292.jpg` through `IMG_6380.jpg`.

After checking a 180-frame preview, the final video was rendered with:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\render_smooth_orbit_windows.ps1" `
  -DataDirectory "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\processed" `
  -Checkpoint "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\training_memory_safe\ckpts\ckpt_29999_rank0.pt" `
  -OutputPath "F:\CP4281-data\sample_scenes\our_reconstruction\drjohnson\training_memory_safe\videos\drjohnson_handheld_24s.mp4" `
  -Frames 720 `
  -Fps 30 `
  -PathType handheld `
  -SmoothingSigma 8
```

The final RGB video was visually inspected and accepted. It contains 720
readable frames at 30 FPS (24.0 seconds), took 34.10 seconds to render, and
occupies 4.61 MiB. The requested render size was 1330x874; H.264 macroblock
padding gives a stored size of 1336x880. The accompanying JSON records the
checkpoint, path type, selected capture segment, focus point, resolution,
duration, and rendering wall time.

## Step 4: reconstruct our own scene

### Capture and input checks

The self-captured scene is a static arrangement of plush toys on a bed. It was
photographed handheld from different sides and at multiple heights, with the
camera translated around the subject to provide parallax rather than only
rotated in place. The dataset contains 136 readable iPhone 15 JPG images from
`IMG_2315.jpg` through `IMG_2457.jpg` and occupies 664.19 MiB.

An evenly spaced visual inspection found good viewpoint variation, stable
subject placement, and useful texture on the bed and surrounding objects. The
main capture weaknesses are shallow depth of field in some views, a fingertip
visible at the edge of one image, background clutter, and two mixed image
resolutions: 110 images at 4032x3024 and 26 at 5712x4284. Next time, one camera
resolution and focus/exposure setting should be locked for the entire capture,
and every frame should be checked for hands and subject blur.

### COLMAP failure and mixed-resolution fix

The first COLMAP attempt incorrectly forced all images to share one camera
model. It reconstructed only the first 22 images (`IMG_2315.jpg` through
`IMG_2336.jpg`) as one camera, 8,474 sparse points, and a 1.251378 px mean
reprojection error. Registration stopped exactly where the source resolution
changed from 5712x4284 to 4032x3024. This identified incompatible shared
intrinsics, rather than generally poor image matching, as the cause.

To preserve the originals without consuming another 664 MiB, the images were
grouped into resolution-named folders using same-volume hard links:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\group_images_by_resolution.ps1" `
  -SourceDirectory "F:\CP4281-data\own_scene\plush_toys\images" `
  -OutputDirectory "F:\CP4281-data\own_scene\plush_toys\images_by_resolution"
```

`scripts/run_colmap_sparse_windows.ps1` was extended with a `CameraGrouping`
option. The reconstruction was then rerun with one shared calibration per
resolution folder:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_colmap_sparse_windows.ps1" `
  -ImagePath "F:\CP4281-data\own_scene\plush_toys\images_by_resolution" `
  -WorkspacePath "F:\CP4281-data\own_scene\plush_toys\reconstruction_by_resolution" `
  -CameraGrouping per-folder
```

This produced one complete model (`sparse/0`) with both camera calibrations.

| `plush_toys` COLMAP measurement | Result |
|---|---:|
| Feature extraction time | 00:00:35.9737272 |
| Exhaustive matching time | 00:02:00.1717634 |
| Incremental mapping time | 00:04:16.4842778 |
| Total measured SfM time | 00:06:52.6297684 |
| Registered images | 136/136 |
| Scene fragments | 1 complete model |
| Camera calibrations | 2 |
| Sparse points | 46,232 |
| Mean track length | 4.352829 |
| Mean observations per image | 1,479.705882 |
| Mean reprojection error | 1.260762 px |

The complete model was undistorted and resized to a maximum dimension of 1600
pixels for safer training on the 8 GiB GPU:

```powershell
& "D:\Tools\COLMAP-3.11.1\COLMAP.bat" image_undistorter `
  --image_path "F:\CP4281-data\own_scene\plush_toys\images_by_resolution" `
  --input_path "F:\CP4281-data\own_scene\plush_toys\reconstruction_by_resolution\sparse\0" `
  --output_path "F:\CP4281-data\own_scene\plush_toys\processed" `
  --output_type COLMAP `
  --max_image_size 1600
```

The processed dataset contains all 136 images at 1600x1200. Its creation took
approximately 25 seconds based on the output timestamps.

### gsplat path compatibility fix and training

The initial smoke test loaded both cameras but failed before training with
`KeyError: '4032x3024/IMG_2337.jpg'`. COLMAP records nested paths with forward
slashes, whereas Python's Windows relative paths use backslashes. The local
`patch_gsplat_colmap_paths_windows.py` normalizes both representations, and the
training and rendering wrappers apply this compatibility patch automatically.
The repeated smoke test then completed successfully.

Formal training used the memory-safe schedule established on `drjohnson`:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\run_gsplat_windows.ps1" `
  -DataDirectory "F:\CP4281-data\own_scene\plush_toys\processed" `
  -ResultDirectory "F:\CP4281-data\own_scene\plush_toys\training_memory_safe" `
  -MaxSteps 30000 `
  -EvalSteps 30000 `
  -SaveSteps 30000 `
  -RefineStopIter 8000 `
  -DisableVideo
```

| Final `plush_toys` measurement | Result |
|---|---:|
| Training steps | 30,000 |
| Initial Gaussians (COLMAP points) | 46,232 |
| Final Gaussians | 1,022,274 |
| Held-out views | 17 |
| Held-out PSNR | 22.6270 dB |
| Held-out SSIM | 0.69292 |
| Held-out LPIPS | 0.30819 |
| Peak allocated VRAM | 1.582 GiB |
| Trainer time | 1,827.88 s (30 min 27.88 s) |
| Total wall-clock time | 00:31:51.5940183 |
| Final checkpoint size | 230.08 MiB |

The final checkpoint is `ckpts/ckpt_29999_rank0.pt`. Visual inspection of three
widely separated held-out views showed that the plush toys retain recognizable
shape, color, and facial details. The main reconstruction weakness is a
streaked or smeared appearance on the fine, repetitive bed-sheet texture,
especially in novel views. Small toys and occlusion boundaries also soften or
blend together. These effects are consistent with limited view coverage,
repetitive texture, shallow focus in some source images, and interpolation
between captured viewpoints.

### Final `plush_toys` video

Because this is a compact object-centered capture, a constant-speed ellipse is
appropriate. The radius was reduced to 0.85 of the estimated capture extent to
keep the camera near well-observed regions:

```powershell
powershell -ExecutionPolicy Bypass `
  -File ".\scripts\render_smooth_orbit_windows.ps1" `
  -DataDirectory "F:\CP4281-data\own_scene\plush_toys\processed" `
  -Checkpoint "F:\CP4281-data\own_scene\plush_toys\training_memory_safe\ckpts\ckpt_29999_rank0.pt" `
  -OutputPath "F:\CP4281-data\own_scene\plush_toys\training_memory_safe\videos\plush_toys_orbit_24s.mp4" `
  -Frames 720 `
  -Fps 30 `
  -PathType ellipse `
  -RadiusScale 0.85
```

The final video was visually inspected and accepted. It is an RGB-only H.264
video with 720 readable frames at 30 FPS (24.0 seconds), a resolution of
1600x1200, and a size of 12.50 MiB. Rendering took 28.65 seconds. The scripted
trajectory looks continuously at the reconstructed focus point and is
arc-length resampled for smooth, near-constant camera speed.
