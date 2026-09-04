# CP4281 Assignment 1: Photos to Gaussian Splats

**Name:** Chen Hongshan<br>
**Student ID:** A0311136W<br>
**Hardware:** AMD Ryzen 7 8745H, 15.31 GiB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB VRAM)

## 1. Pipeline and use of third-party libraries

I built a reproducible photograph-to-3D-Gaussian-Splatting pipeline around COLMAP 3.11.1 and gsplat 1.5.3. All large datasets, reconstructions, and model checkpoints were kept outside the Git repository, while the repository contains the environment specifications, verification evidence, Windows compatibility patches, and scripts used to reproduce each stage.

### COLMAP: camera reconstruction and sparse initialization

For every scene, I gave COLMAP only the source photographs rather than using the camera poses supplied with the sample datasets. The pipeline called:

1. `feature_extractor` to detect local SIFT features in every image;
2. `exhaustive_matcher` to identify corresponding features across image pairs;
3. `mapper` to estimate camera intrinsics, camera poses, and a sparse 3D point cloud through incremental structure from motion and bundle adjustment;
4. `model_analyzer` to measure image registration, fragmentation, sparse-point count, track length, and reprojection error; and
5. `image_undistorter` to produce gsplat-ready images and adjusted camera parameters, with maximum image dimension 1600 pixels.

The sparse COLMAP points initialized the Gaussian centers. The reconstructed camera poses and intrinsics supplied the training and held-out views. For the self-captured scene, two source resolutions required two camera calibrations; both calibrations and all images remained connected in one reconstruction.

### gsplat: optimization and rendering

I used gsplat's `examples/simple_trainer.py` baseline rather than implementing Gaussian splatting itself. Training used SfM initialization, packed rasterization, batch size one, normalized world coordinates, and degree-3 spherical harmonics. Every eighth image was held out for evaluation. Each baseline was optimized for 30,000 steps. The default densification strategy began refinement at step 500, split or duplicated Gaussians every 100 steps, reset opacity every 3,000 steps, pruned low-opacity Gaussians, and normally stopped refinement at step 15,000. On memory-heavy scenes I stopped refinement at step 8,000, while continuing parameter optimization to step 30,000.

The training script reported final PSNR, SSIM, LPIPS, peak allocated VRAM, Gaussian count, and training time. I wrote a separate renderer that reloads a checkpoint and calls gsplat rasterization for an RGB-only video. Compact, object-centered scenes use a constant-speed elliptical trajectory aimed at a reconstruction-derived focus point. A captured-path mode was used for the long train scene. For the indoor `drjohnson` scene, a handheld mode selects a continuous segment of real camera positions, smooths it, estimates the room's vertical direction, and keeps the camera aimed into the scene.

### Reproducibility and Windows configuration

The repository provides `environment.yml`, pinned pip requirements, installation verification, and PowerShell wrappers. `verify_installation.py` demonstrates that PyTorch detects the GPU, `gsplat.rasterization` executes successfully, and COLMAP reports its version. The wrappers also initialize the Visual Studio C++ environment and move compilation caches and temporary files off the system drive. Small documented compatibility patches handle MSVC compiler flags, COLMAP's binary field sizes on Windows, and slash differences in nested COLMAP image paths.

## 2. Results and screenshots

### 2.1 Sample scene: `truck`

COLMAP registered all 251 input images in the main model. The trained model reproduces the truck, surrounding trees, and background geometry well, while fine ground and vegetation textures are smoother than the photographs.

![Reconstructed truck viewed broadside](images/truck_side.png)

*Figure 1. Broadside view of the reconstructed truck. The cab, wooden cargo bed, wheels, and pavement markings remain geometrically coherent.*

![Reconstructed truck viewed from behind](images/truck_back.png)

*Figure 2. Rear viewpoint showing consistent tail lights, cargo-bed panels, hitch, and surrounding street geometry from the opposite side of the orbit.*

![Close frontal view showing weaknesses in the truck reconstruction](images/truck_weakness.png)

*Figure 3. Weakness in a close novel view: the hood and windscreen are over-smoothed, and fine boundaries near the vegetation and right image edge become translucent and distorted.*

### 2.2 Sample scene: `train`

COLMAP registered all 301 images in one connected model. The large scene and thin undercarriage structures are more difficult than the compact truck. Some novel views contain dark, fog-like floaters: these are semi-transparent Gaussians with inaccurate geometry or color, compounded by the black background showing through regions with insufficient accumulated opacity.

![Reconstructed train from a front three-quarter viewpoint](images/train_front.png)

*Figure 4. Front three-quarter view. The locomotive body, number 713, lettering, handrails, and large-scale outdoor geometry are clearly reconstructed.*

![Close broadside view of the reconstructed train](images/train_side.png)

*Figure 5. Close broadside view showing preserved lettering, vents, railings, and weathered surface appearance across the long locomotive body.*

![Rear train viewpoint showing reconstruction artifacts](images/train_failure.png)

*Figure 6. Weak rear viewpoint. Wispy dark floaters appear above and beside the locomotive, while rails, thin structures, and the distant hillside blur in sparsely observed regions.*

### 2.3 Sample scene: `drjohnson`

The complete COLMAP model registered all 263 images; COLMAP also emitted a redundant two-image fragment that was not used. Stopping densification at step 8,000 prevented the indoor reconstruction from exhausting the 8 GiB GPU. The result has the highest PSNR of the four scenes, although some novel indoor views still expose weakly observed surfaces.

![Wide hallway view of the reconstructed Dr Johnson room](images/room_front.png)

*Figure 7. Wide hallway view. Doors, wall panels, dining furniture, framed pictures, and the room layout remain recognizable across a large depth range.*

![Closer viewpoint inside the reconstructed Dr Johnson room](images/room_close.png)

*Figure 8. Closer interior view from a different position, showing consistent table, chairs, fireplace, wall trim, and picture placement.*

![Dr Johnson room viewpoint showing a damaged wall picture](images/room_failure.png)

*Figure 9. Indoor reconstruction weakness. The picture near the left window is smeared and surrounded by holes and black floaters, indicating insufficient or inconsistent observations around this reflective, high-detail region.*

### 2.4 Self-captured scene: `plush_toys`

I captured 136 photographs of a static plush-toy arrangement on a bed. I moved around the subject rather than pivoting from one point and included front, rear, side, high, and low viewpoints. The bed texture and background objects provided features for matching. However, some photographs used shallow depth of field, one included a fingertip at the frame edge, and two image resolutions were mixed. Next time I would lock resolution, lens, focus, and exposure for the whole sequence; maintain approximately 70% overlap; inspect every frame for blur or hands; and capture additional low-angle views around occlusion boundaries between the smaller toys.

The final model preserves the toys' shapes, colors, and facial features. Its main weakness is the streaked or smeared reconstruction of the bed sheet's fine repetitive pattern. Small toys also soften or merge near occlusion boundaries. These artifacts are consistent with repetitive texture, shallow focus, incomplete local coverage, and interpolation between captured views.

![Frontal view of the self-captured plush-toy scene](images/toy_front.png)

*Figure 10. Frontal view of the self-captured scene. The large duck, smaller toys, yellow blanket, and surrounding bedroom retain recognizable color and overall structure.*

![Rear view of the reconstructed plush-toy arrangement](images/toy_back.png)

*Figure 11. Rear viewpoint demonstrating full capture coverage around the large duck; folds in the blue fabric and the smaller foreground toys remain visible.*

![Side view of the reconstructed plush toys](images/toy_side.png)

*Figure 12. Side viewpoint with substantial parallax from the frontal view. The duck profile and yellow blanket are preserved, while distant bedroom objects are softer.*

![Close frontal view of the plush-toy reconstruction](images/toy_close.png)

*Figure 13. Close view showing successful reconstruction of the duck's fur, facial features, robe folds, small toys, and printed blanket outline.*

![Plush-toy viewpoint showing bed-sheet and occlusion artifacts](images/toy_failure.png)

*Figure 14. Failure example from behind the arrangement. The repetitive bed sheet becomes streaked, and several small toys blur together at occlusion boundaries where view coverage is weaker.*

## 3. Quantitative results and computational cost

All experiments used the same computer described above. SfM time is the sum of feature extraction, exhaustive matching, and incremental mapping. Training time is the trainer's measured optimization time; total wall time is slightly larger because it includes initialization and final evaluation. Output size reports the final checkpoint and the selected 24-second RGB video.

| Scene | Registered images | SfM time | Training configuration | Training time | Peak VRAM | Final Gaussians | PSNR / SSIM | Checkpoint / video size |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `truck` | 251/251; plus one unused 2-image fragment | 8:52.25 | 30k steps; refine to 15k | 39:17.86 (40:48.79 wall) | 3.632 GiB | 2,480,237 | 26.0812 / 0.89677 | 558.22 / 5.37 MiB |
| `train` | 301/301; one model | 11:46.73 | 30k steps; refine to 15k | 26:00.92 (27:32.38 wall) | 2.008 GiB | 1,325,402 | 22.0377 / 0.83673 | 298.31 / 8.85 MiB |
| `drjohnson` | 263/263; plus one unused 2-image fragment | 7:27.73 | 30k steps; refine to 8k | 47:47.41 (49:38.48 wall) | 4.465 GiB | 3,081,715 | 28.6254 / 0.89897 | 693.60 / 4.61 MiB |
| `plush_toys` | 136/136; one model, two calibrations | 6:52.63 | 30k steps; refine to 8k | 30:27.88 (31:51.59 wall) | 1.582 GiB | 1,022,274 | 22.6270 / 0.69292 | 230.08 / 12.50 MiB |

The sample scenes demonstrate that scene type affects both cost and failure mode. The compact truck achieves strong perceptual quality but densifies to 2.48 million Gaussians. The longer train uses fewer Gaussians yet has lower PSNR and visible floaters around thin or occluded structures. The indoor `drjohnson` scene grows most aggressively and requires the most memory despite early stopping. The self-captured scene uses the least memory, but repetitive fabric and capture inconsistencies reduce SSIM.

## 4. What went wrong and how it was diagnosed

Several failures were useful for understanding the pipeline:

- **Optional CUDA dependency failed to compile.** The official examples tried to build `fused_bilagrid`, whose source triggered MSVC narrowing-conversion errors. The default trainer does not use this optional extension, so I pinned and installed the required baseline dependencies without it.
- **COLMAP binary reader failed on Windows.** A legacy Python reader interpreted COLMAP eight-byte fields using a platform-native unsigned long, which is four bytes on Windows. Explicit little-endian field sizes fixed the parser.
- **gsplat JIT could not find `cl.exe`.** The first CUDA rasterization failed because the Visual Studio compiler environment was absent. The wrapper now loads the x64 developer environment before Python and caches the compiled extension on drive F.
- **`drjohnson` ran out of GPU memory.** The first run reached about 3.95 million Gaussians and 5.83 GiB allocated memory around step 10,600, leaving insufficient temporary memory for the next densification. Stopping refinement at step 8,000 bounded the successful run at 3.08 million Gaussians.
- **Naive video trajectories failed.** The default video was too short and shaky; an ellipse left the covered region of the long train scene; and smoothing all `drjohnson` image-name-ordered poses made the camera alternate between ceiling and floor. Scene-specific constant-speed, captured, and handheld trajectories solved these problems without retraining.
- **The first self-captured COLMAP run registered only 22/136 images.** The failure occurred exactly where resolution changed, because all images had incorrectly been forced to share one calibration. Grouping the photographs by resolution and using one camera per folder registered all 136 images in a single reconstruction.
- **Nested images failed in gsplat on Windows.** COLMAP stored nested names with forward slashes while the gsplat parser indexed Windows backslash paths. Normalizing both path representations fixed the smoke test without rerunning COLMAP.

These failures show that reconstruction quality and reliability depend not only on the optimizer, but also on capture consistency, calibration grouping, coordinate conventions, memory growth during densification, and render-path coverage.

## 5. Optional extension: lighter training at matched quality

I evaluated a lighter densification schedule on `truck`. The baseline refined until step 15,000; the experimental run stopped refinement at step 8,000. All other variables—including input data, COLMAP initialization, train/validation split, packed rendering, spherical harmonics, and 30,000 total steps—were held constant.

| Measurement | Baseline: refine to 15k | Lightweight: refine to 8k | Change |
|---|---:|---:|---:|
| Final Gaussians | 2,480,237 | 1,907,386 | -23.10% |
| Peak VRAM | 3.632 GiB | 2.797 GiB | -22.98% |
| Trainer time | 2,357.86 s | 2,152.00 s | -8.73% |
| Checkpoint size | 558.22 MiB | 429.29 MiB | -23.10% |
| PSNR | 26.081238 dB | 26.080711 dB | -0.000526 dB |
| SSIM | 0.896770 | 0.896792 | +0.000022 |
| LPIPS (lower is better) | 0.095868 | 0.096785 | +0.000917 |

Stopping densification early reduced Gaussian count, peak memory, and model size by about 23% and reduced trainer time by 8.7%, while PSNR and SSIM were effectively unchanged. LPIPS became slightly worse, so the saving has a small perceptual cost. Across all 32 corresponding held-out renders, the baseline and lightweight outputs had a mean absolute RGB difference of 0.01685 and a mean between-model PSNR of 30.81 dB. The experiment therefore demonstrates a measurably lighter model at matched reconstruction quality.

## 6. Submitted videos

- **Sample scene:** `truck_smooth_24s.mp4` — 720 frames, 30 FPS, 24 seconds, scripted constant-speed elliptical trajectory, 984x544 encoded resolution.
- **Self-captured scene:** `plush_toys_orbit_24s.mp4` — 720 frames, 30 FPS, 24 seconds, scripted constant-speed elliptical trajectory, 1600x1200 resolution.

Both videos use RGB-only rendering and were decoded and visually inspected after generation.
