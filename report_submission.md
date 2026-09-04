# CP4281 Assignment 1: Photos to Gaussian Splats

**Name:** Chen Hongshan<br>**Student ID:** A0311136W <br>**Hardware:** AMD Ryzen 7 8745H, 15.31 GiB RAM, NVIDIA RTX 4060 Laptop GPU (8188 MiB VRAM)

## 1. Pipeline

I built a reproducible photograph-to-Gaussian-Splatting pipeline using COLMAP 3.11.1 and gsplat 1.5.3. I supplied COLMAP only the photographs, not the sample datasets' existing poses. `feature_extractor` detected SIFT features, `exhaustive_matcher` established image correspondences, and `mapper` jointly estimated camera intrinsics, poses, and a sparse point cloud through incremental structure from motion and bundle adjustment. I used `model_analyzer` to check registration and fragmentation, then `image_undistorter --max_image_size 1600` to prepare images and adjusted cameras while controlling training cost.

The COLMAP points initialized Gaussian centers, while its cameras supplied training and held-out views. I trained gsplat's `simple_trainer.py` for 30,000 steps with packed rasterization, batch size one, degree-3 spherical harmonics, normalized coordinates, and every eighth image held out. Default densification began at step 500, split/duplicated Gaussians every 100 steps, reset opacity every 3,000 steps, pruned low-opacity Gaussians, and stopped at step 15,000; memory-heavy scenes stopped at 8,000 but continued optimizing to step 30,000. My renderer reloads the final checkpoint and calls gsplat rasterization along scene-specific, constant-speed camera paths. Environment files, exact commands, verification output, and Windows compatibility patches are documented in the README.

## 2. Reconstruction results

### Sample 1: `truck`

COLMAP registered 251/251 images in the main model (plus an unused two-image fragment). The compact orbit reconstructs the vehicle well, but close novel views smooth fine surfaces and boundaries.

<table><tr>
<td width="33%"><img src="images/truck_side.png" width="100%"><br><small><b>Fig. 1.</b> Coherent cab, cargo bed, wheels, and pavement in a broadside view.</small></td>
<td width="33%"><img src="images/truck_back.png" width="100%"><br><small><b>Fig. 2.</b> Consistent tail lights, bed panels, hitch, and street geometry from behind.</small></td>
<td width="33%"><img src="images/truck_weakness.png" width="100%"><br><small><b>Fig. 3.</b> Close-view weakness: smoothed hood/windscreen and translucent edge detail.</small></td>
</tr></table>

### Sample 2: `train`

COLMAP registered 301/301 images in one model. The long locomotive is recognizable, but thin undercarriage structures and sparsely observed background regions produce blur and dark floaters.

<table><tr>
<td width="33%"><img src="images/train_front.png" width="100%"><br><small><b>Fig. 4.</b> Preserved locomotive body, number, lettering, and handrails.</small></td>
<td width="33%"><img src="images/train_side.png" width="100%"><br><small><b>Fig. 5.</b> Broadside view retaining vents, lettering, and weathered appearance.</small></td>
<td width="33%"><img src="images/train_failure.png" width="100%"><br><small><b>Fig. 6.</b> Weak rear view with wispy floaters and blurred rails and hillside.</small></td>
</tr></table>

### Sample 3: `drjohnson`

The main COLMAP model registered 263/263 images (plus an unused two-image fragment). Early densification stopping avoided OOM and produced the highest PSNR, although weakly observed interior details still contain holes and floaters.

<table><tr>
<td width="33%"><img src="images/room_front.png" width="100%"><br><small><b>Fig. 7.</b> Wide view preserving doors, furniture, pictures, and room layout.</small></td>
<td width="33%"><img src="images/room_close.png" width="100%"><br><small><b>Fig. 8.</b> Closer view with consistent table, fireplace, trim, and picture placement.</small></td>
<td width="33%"><img src="images/room_failure.png" width="100%"><br><small><b>Fig. 9.</b> Smeared wall picture with holes and black floaters near the window.</small></td>
</tr></table>

### Self-captured scene: `plush_toys`

I took 136 photographs while translating around a static plush-toy arrangement at front, rear, side, high, and low viewpoints. The bed and background supplied useful matching texture. Next time I would lock resolution, lens, focus, and exposure; maintain about 70% overlap; remove blurred frames and hands; and add low views around occlusions. This capture mixed 110 images at 4032x3024 and 26 at 5712x4284. Forcing one calibration initially registered only 22/136 images; grouping by resolution and using two calibrations connected all 136 images in one model.

<table><tr>
<td width="33%"><img src="images/toy_front.png" width="100%"><br><small><b>Fig. 10.</b> Frontal view preserving toy shapes, colors, and facial details.</small></td>
<td width="33%"><img src="images/toy_back.png" width="100%"><br><small><b>Fig. 11.</b> Rear view demonstrating coverage around the large duck.</small></td>
<td width="33%"><img src="images/toy_side.png" width="100%"><br><small><b>Fig. 12.</b> Side view showing parallax and a recognizable duck profile.</small></td>
</tr><tr>
<td width="50%" colspan="1"><img src="images/toy_close.png" width="100%"><br><small><b>Fig. 13.</b> Close view retaining fur, robe folds, and printed blanket details.</small></td>
<td width="50%" colspan="2"><img src="images/toy_failure.png" width="100%"><br><small><b>Fig. 14.</b> Failure: repetitive bed texture streaks and small toys merge at occlusions.</small></td>
</tr></table>

## 3. Computational cost

SfM time sums feature extraction, exhaustive matching, and mapping. Training time is measured optimization time; parenthesized values include initialization and final evaluation. Output sizes are final checkpoint/video.

| Scene | Registration | SfM | Configuration | Training (wall) | VRAM | Gaussians | PSNR / SSIM | Output MiB |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `truck` | 251/251 + 2-image fragment | 8:52 | 30k; refine 15k | 39:18 (40:49) | 3.632 GiB | 2,480,237 | 26.081 / .8968 | 558.22 / 5.37 |
| `train` | 301/301; one model | 11:47 | 30k; refine 15k | 26:01 (27:32) | 2.008 GiB | 1,325,402 | 22.038 / .8367 | 298.31 / 8.85 |
| `drjohnson` | 263/263 + 2-image fragment | 7:28 | 30k; refine 8k | 47:47 (49:38) | 4.465 GiB | 3,081,715 | 28.625 / .8990 | 693.60 / 4.61 |
| `plush_toys` | 136/136; two calibrations | 6:53 | 30k; refine 8k | 30:28 (31:52) | 1.582 GiB | 1,022,274 | 22.627 / .6929 | 230.08 / 12.50 |

## 4. Failures and diagnosis

| Failure | Diagnosis and change |
|---|---|
| Optional `fused_bilagrid` failed under MSVC | It is unused by the baseline, so I installed pinned required dependencies without it. |
| COLMAP binary reader failed on Windows | Platform-native unsigned long was four bytes; explicit little-endian COLMAP field sizes fixed it. |
| gsplat JIT could not find `cl.exe` | The wrapper now loads the Visual Studio x64 environment and caches the CUDA extension on F. |
| `drjohnson` OOM at ~3.95M Gaussians | Densification needed temporary headroom; stopping refinement at 8k bounded the successful model at 3.08M. |
| Default/naive videos were short, shaky, underground, or vertically unstable | I used constant-speed ellipse, captured, or handheld trajectories according to scene geometry and camera coverage. |
| Own scene initially registered 22/136 images | The break coincided with a resolution change; two resolution-specific calibrations registered all images. |
| Nested own-scene paths raised a gsplat `KeyError` | Normalizing forward/backward slashes made Windows and COLMAP path keys agree. |

## 5. Optional extension: lighter training

On `truck`, I changed only the densification stop from 15k to 8k; data, initialization, validation split, renderer, SH degree, and 30k total steps were fixed.

| Metric | Refine 15k | Refine 8k | Change |
|---|---:|---:|---:|
| Gaussians | 2,480,237 | 1,907,386 | -23.10% |
| Peak VRAM | 3.632 GiB | 2.797 GiB | -22.98% |
| Trainer time | 2,357.86 s | 2,152.00 s | -8.73% |
| Checkpoint | 558.22 MiB | 429.29 MiB | -23.10% |
| PSNR / SSIM | 26.081238 / .896770 | 26.080711 / .896792 | effectively unchanged |
| LPIPS (lower is better) | .095868 | .096785 | +.000917 |

The lightweight model cut count, memory, and storage by about 23% and time by 8.7%, with effectively identical PSNR/SSIM and only a small LPIPS cost. Corresponding renders had 30.81 dB mean between-model PSNR, supporting matched quality.

## 6. Videos

I submitted two RGB-only scripted fly-throughs: `truck_smooth_24s.mp4` uses a constant-speed ellipse, and `plush_toys_orbit_24s.mp4` uses a tighter object-centered ellipse. Both contain 720 frames at 30 FPS (24 seconds) and were decoded and visually inspected.
