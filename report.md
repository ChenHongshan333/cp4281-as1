# CP4281 Assignment 1: Photos to Gaussian Splats

**Name:** [NAME PLACEHOLDER]  
**Student ID:** [STUDENT ID PLACEHOLDER]  
**Hardware:** AMD Ryzen 7 8745H, 15.31 GiB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)

## 1. Pipeline and use of third-party libraries

I built a reproducible photograph-to-3D-Gaussian-Splatting pipeline around
COLMAP 3.11.1 and gsplat 1.5.3. All large datasets, reconstructions, and model
checkpoints were kept outside the Git repository, while the repository contains
the environment specifications, verification evidence, Windows compatibility
patches, and scripts used to reproduce each stage.

### COLMAP: camera reconstruction and sparse initialization

For every scene, I gave COLMAP only the source photographs rather than using
the camera poses supplied with the sample datasets. The pipeline called:

1. `feature_extractor` to detect local SIFT features in every image;
2. `exhaustive_matcher` to identify corresponding features across image pairs;
3. `mapper` to estimate camera intrinsics, camera poses, and a sparse 3D point
   cloud through incremental structure from motion and bundle adjustment;
4. `model_analyzer` to measure image registration, fragmentation, sparse-point
   count, track length, and reprojection error; and
5. `image_undistorter` to produce gsplat-ready images and adjusted camera
   parameters, with maximum image dimension 1600 pixels.

The sparse COLMAP points initialized the Gaussian centers. The reconstructed
camera poses and intrinsics supplied the training and held-out views. For the
self-captured scene, two source resolutions required two camera calibrations;
both calibrations and all images remained connected in one reconstruction.

### gsplat: optimization and rendering

I used gsplat's `examples/simple_trainer.py` baseline rather than implementing
Gaussian splatting itself. Training used SfM initialization, packed
rasterization, batch size one, normalized world coordinates, and degree-3
spherical harmonics. Every eighth image was held out for evaluation. Each
baseline was optimized for 30,000 steps. The default densification strategy
began refinement at step 500, split or duplicated Gaussians every 100 steps,
reset opacity every 3,000 steps, pruned low-opacity Gaussians, and normally
stopped refinement at step 15,000. On memory-heavy scenes I stopped refinement
at step 8,000, while continuing parameter optimization to step 30,000.

The training script reported final PSNR, SSIM, LPIPS, peak allocated VRAM,
Gaussian count, and training time. I wrote a separate renderer that reloads a
checkpoint and calls gsplat rasterization for an RGB-only video. Compact,
object-centered scenes use a constant-speed elliptical trajectory aimed at a
reconstruction-derived focus point. A captured-path mode was used for the long
train scene. For the indoor `drjohnson` scene, a handheld mode selects a
continuous segment of real camera positions, smooths it, estimates the room's
vertical direction, and keeps the camera aimed into the scene.

### Reproducibility and Windows configuration

The repository provides `environment.yml`, pinned pip requirements, installation
verification, and PowerShell wrappers. `verify_installation.py` demonstrates
that PyTorch detects the GPU, `gsplat.rasterization` executes successfully, and
COLMAP reports its version. The wrappers also initialize the Visual Studio C++
environment and move compilation caches and temporary files off the system
drive. Small documented compatibility patches handle MSVC compiler flags,
COLMAP's binary field sizes on Windows, and slash differences in nested COLMAP
image paths.

## 2. Results and screenshots

The following placeholders should be replaced with screenshots from clearly
different viewpoints. Captions should remain concise and should identify both
successful reconstruction details and visible weaknesses.

### 2.1 Sample scene: `truck`

COLMAP registered all 251 input images in the main model. The trained model
reproduces the truck, surrounding trees, and background geometry well, while
fine ground and vegetation textures are smoother than the photographs.

> **[IMAGE PLACEHOLDER T1 — Truck front/three-quarter viewpoint]**  
> Suggested caption: The truck body and high-contrast edges are reconstructed
> cleanly from a well-observed viewpoint.

> **[IMAGE PLACEHOLDER T2 — Truck side or rear viewpoint]**  
> Suggested caption: A substantially different viewpoint showing consistent
> vehicle geometry and background structure.

> **[IMAGE PLACEHOLDER T3 — Truck weakness viewpoint]**  
> Suggested caption: Fine ground or vegetation detail becomes smooth or
> smeared where geometry and appearance are less strongly constrained.

### 2.2 Sample scene: `train`

COLMAP registered all 301 images in one connected model. The large scene and
thin undercarriage structures are more difficult than the compact truck. Some
novel views contain dark, fog-like floaters: these are semi-transparent
Gaussians with inaccurate geometry or color, compounded by the black background
showing through regions with insufficient accumulated opacity.

> **[IMAGE PLACEHOLDER R1 — Train front/side viewpoint]**  
> Suggested caption: The train's main body and large-scale geometry remain
> recognizable from a well-covered view.

> **[IMAGE PLACEHOLDER R2 — Train viewpoint from the opposite side]**  
> Suggested caption: A different viewpoint showing reconstruction of the train
> and its outdoor surroundings.

> **[IMAGE PLACEHOLDER R3 — Train black-floater weakness]**  
> Suggested caption: Dark floaters appear near the undercarriage or sparsely
> observed background because poorly constrained Gaussians accumulate opacity.

### 2.3 Sample scene: `drjohnson`

The complete COLMAP model registered all 263 images; COLMAP also emitted a
redundant two-image fragment that was not used. Stopping densification at step
8,000 prevented the indoor reconstruction from exhausting the 8 GiB GPU. The
result has the highest PSNR of the four scenes, although some novel indoor
views still expose weakly observed surfaces.

> **[IMAGE PLACEHOLDER D1 — Dr Johnson room viewpoint 1]**  
> Suggested caption: A broad indoor view showing reconstructed furniture and
> room layout.

> **[IMAGE PLACEHOLDER D2 — Dr Johnson room viewpoint 2]**  
> Suggested caption: A clearly different camera position demonstrating
> multi-view consistency.

> **[IMAGE PLACEHOLDER D3 — Dr Johnson weakness viewpoint]**  
> Suggested caption: A weakly observed boundary, reflective surface, hole, or
> floater visible from a novel viewpoint.

### 2.4 Self-captured scene: `plush_toys`

I captured 136 photographs of a static plush-toy arrangement on a bed. I moved
around the subject rather than pivoting from one point and included front,
rear, side, high, and low viewpoints. The bed texture and background objects
provided features for matching. However, some photographs used shallow depth
of field, one included a fingertip at the frame edge, and two image resolutions
were mixed. Next time I would lock resolution, lens, focus, and exposure for the
whole sequence; maintain approximately 70% overlap; inspect every frame for
blur or hands; and capture additional low-angle views around occlusion
boundaries between the smaller toys.

The final model preserves the toys' shapes, colors, and facial features. Its
main weakness is the streaked or smeared reconstruction of the bed sheet's
fine repetitive pattern. Small toys also soften or merge near occlusion
boundaries. These artifacts are consistent with repetitive texture, shallow
focus, incomplete local coverage, and interpolation between captured views.

> **[IMAGE PLACEHOLDER O1 — Plush toys frontal viewpoint]**  
> Suggested caption: Frontal view showing the main duck and small toys with
> recognizable colors and facial details.

> **[IMAGE PLACEHOLDER O2 — Plush toys rear viewpoint]**  
> Suggested caption: Rear view demonstrating coverage around the large duck and
> the arrangement behind it.

> **[IMAGE PLACEHOLDER O3 — Plush toys high-angle viewpoint]**  
> Suggested caption: High-angle view showing the spatial layout of the toys on
> the bed.

> **[IMAGE PLACEHOLDER O4 — Plush toys low or side viewpoint]**  
> Suggested caption: Low/side view demonstrating parallax and reconstruction
> across a substantially different camera height.

> **[IMAGE PLACEHOLDER O5 — Plush toys weakness: smeared bed sheet]**  
> Suggested caption: The repetitive bed-sheet texture produces streaked
> Gaussians and softened occlusion boundaries in a novel view.

## 3. Quantitative results and computational cost

All experiments used the same computer described above. SfM time is the sum of
feature extraction, exhaustive matching, and incremental mapping. Training
time is the trainer's measured optimization time; total wall time is slightly
larger because it includes initialization and final evaluation. Output size
reports the final checkpoint and the selected 24-second RGB video.

| Scene | Registered images | SfM time | Training configuration | Training time | Peak VRAM | Final Gaussians | PSNR / SSIM | Checkpoint / video size |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `truck` | 251/251; plus one unused 2-image fragment | 8:52.25 | 30k steps; refine to 15k | 39:17.86 (40:48.79 wall) | 3.632 GiB | 2,480,237 | 26.0812 / 0.89677 | 558.22 / 5.37 MiB |
| `train` | 301/301; one model | 11:46.73 | 30k steps; refine to 15k | 26:00.92 (27:32.38 wall) | 2.008 GiB | 1,325,402 | 22.0377 / 0.83673 | 298.31 / 8.85 MiB |
| `drjohnson` | 263/263; plus one unused 2-image fragment | 7:27.73 | 30k steps; refine to 8k | 47:47.41 (49:38.48 wall) | 4.465 GiB | 3,081,715 | 28.6254 / 0.89897 | 693.60 / 4.61 MiB |
| `plush_toys` | 136/136; one model, two calibrations | 6:52.63 | 30k steps; refine to 8k | 30:27.88 (31:51.59 wall) | 1.582 GiB | 1,022,274 | 22.6270 / 0.69292 | 230.08 / 12.50 MiB |

The sample scenes demonstrate that scene type affects both cost and failure
mode. The compact truck achieves strong perceptual quality but densifies to
2.48 million Gaussians. The longer train uses fewer Gaussians yet has lower
PSNR and visible floaters around thin or occluded structures. The indoor
`drjohnson` scene grows most aggressively and requires the most memory despite
early stopping. The self-captured scene uses the least memory, but repetitive
fabric and capture inconsistencies reduce SSIM.

## 4. What went wrong and how it was diagnosed

Several failures were useful for understanding the pipeline:

- **Optional CUDA dependency failed to compile.** The official examples tried
  to build `fused_bilagrid`, whose source triggered MSVC narrowing-conversion
  errors. The default trainer does not use this optional extension, so I pinned
  and installed the required baseline dependencies without it.
- **COLMAP binary reader failed on Windows.** A legacy Python reader interpreted
  COLMAP eight-byte fields using a platform-native unsigned long, which is four
  bytes on Windows. Explicit little-endian field sizes fixed the parser.
- **gsplat JIT could not find `cl.exe`.** The first CUDA rasterization failed
  because the Visual Studio compiler environment was absent. The wrapper now
  loads the x64 developer environment before Python and caches the compiled
  extension on drive F.
- **`drjohnson` ran out of GPU memory.** The first run reached about 3.95
  million Gaussians and 5.83 GiB allocated memory around step 10,600, leaving
  insufficient temporary memory for the next densification. Stopping refinement
  at step 8,000 bounded the successful run at 3.08 million Gaussians.
- **Naive video trajectories failed.** The default video was too short and
  shaky; an ellipse left the covered region of the long train scene; and
  smoothing all `drjohnson` image-name-ordered poses made the camera alternate
  between ceiling and floor. Scene-specific constant-speed, captured, and
  handheld trajectories solved these problems without retraining.
- **The first self-captured COLMAP run registered only 22/136 images.** The
  failure occurred exactly where resolution changed, because all images had
  incorrectly been forced to share one calibration. Grouping the photographs
  by resolution and using one camera per folder registered all 136 images in a
  single reconstruction.
- **Nested images failed in gsplat on Windows.** COLMAP stored nested names with
  forward slashes while the gsplat parser indexed Windows backslash paths.
  Normalizing both path representations fixed the smoke test without rerunning
  COLMAP.

These failures show that reconstruction quality and reliability depend not
only on the optimizer, but also on capture consistency, calibration grouping,
coordinate conventions, memory growth during densification, and render-path
coverage.

## 5. Optional extension: lighter training at matched quality

I evaluated a lighter densification schedule on `truck`. The baseline refined
until step 15,000; the experimental run stopped refinement at step 8,000. All
other variables—including input data, COLMAP initialization, train/validation
split, packed rendering, spherical harmonics, and 30,000 total steps—were held
constant.

| Measurement | Baseline: refine to 15k | Lightweight: refine to 8k | Change |
|---|---:|---:|---:|
| Final Gaussians | 2,480,237 | 1,907,386 | -23.10% |
| Peak VRAM | 3.632 GiB | 2.797 GiB | -22.98% |
| Trainer time | 2,357.86 s | 2,152.00 s | -8.73% |
| Checkpoint size | 558.22 MiB | 429.29 MiB | -23.10% |
| PSNR | 26.081238 dB | 26.080711 dB | -0.000526 dB |
| SSIM | 0.896770 | 0.896792 | +0.000022 |
| LPIPS (lower is better) | 0.095868 | 0.096785 | +0.000917 |

Stopping densification early reduced Gaussian count, peak memory, and model
size by about 23% and reduced trainer time by 8.7%, while PSNR and SSIM were
effectively unchanged. LPIPS became slightly worse, so the saving has a small
perceptual cost. Across all 32 corresponding held-out renders, the baseline and
lightweight outputs had a mean absolute RGB difference of 0.01685 and a mean
between-model PSNR of 30.81 dB. The experiment therefore demonstrates a
measurably lighter model at matched reconstruction quality.

## 6. Submitted videos

- **Sample scene:** `truck_smooth_24s.mp4` — 720 frames, 30 FPS, 24 seconds,
  scripted constant-speed elliptical trajectory, 984x544 encoded resolution.
- **Self-captured scene:** `plush_toys_orbit_24s.mp4` — 720 frames, 30 FPS, 24
  seconds, scripted constant-speed elliptical trajectory, 1600x1200 resolution.

Both videos use RGB-only rendering and were decoded and visually inspected
after generation.
