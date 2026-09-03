# Assignment 1 — Photos to Gaussian Splats

# Submission: .zip file to Canvas. Deadline: 23:59pm, Sep 4th, 2026

## Goal

Build a working pipeline that turns a collection of ordinary photographs into an
interactive 3D Gaussian Splatting scene, using **gsplat** and **COLMAP**.

You are not implementing Gaussian splatting. You are learning to use the state-of-the-art libraries, and to report the experience.

---

## Before you start
Mordern CPU/GPUs. Candidates:
* Your own computers.
* SoC servers.
* External servers. You have S$250 (**for this full semester**) for external servers and you can reimburse it by submitting your claims via the SoC Claims Hub (https://mysoc.nus.edu.sg/app/claims) from Week 1 onwards. 

Start early.

---

## Step 1 — Create an environment (5 marks)

Create an isolated Python environment using `conda`, `uv`, `venv`, or equivalent.

**Deliverables**

- The environment specification (`environment.yml`, `requirements.txt`,
  `uv.lock`, …) committed to your repository.
- The exact commands you ran, in your README.

Marks are for reproducibility. Someone else should be able to recreate your
environment from what you submit, without guessing.

---

## Step 2 — Install the packages (5 marks)

Install PyTorch, gsplat and COLMAP.

Three things reliably cause trouble; plan for them:

- **PyTorch must match your CUDA version.** Verify with
  `torch.cuda.is_available()` and `torch.version.cuda`.
- **gsplat compiles CUDA kernels on first call**, not at install time. Both
  `nvcc` and a host C++ compiler must be on `PATH` when you first use it. The
  first call therefore takes minutes; later ones are instant.
- **COLMAP is a separate program**, not a Python package. Install a prebuilt
  binary or use your package manager.

**Deliverables**

- A short verification script and its output, showing (a) PyTorch sees your GPU,
  (b) `gsplat.rasterization` runs without error, and (c) COLMAP reports its
  version.

---

## Step 3 — Reconstruct three sample scenes (40 marks)

*10 marks per scene.*

Download the datasets below and pick **any three** of the six scenes.

| Scene | Photos | Character | Bundle |
|---|---|---|---|
| `truck` | 251 | Outdoor, object-centric orbit | Tanks and Temples |
| `train` | 301 | Outdoor, large object plus background | Tanks and Temples |
| `drjohnson` | 263 | Indoor room, furnished | Deep Blending |
| `playroom` | 225 | Indoor room, strong lighting variation | Deep Blending |
| `south-building` | 128 | Building facade, 3072×2304 originals | COLMAP samples |
| `gerrard-hall` | 100 | Building exterior, 5616×3744 originals | COLMAP samples |

**Downloads**

- Tanks and Temples + Deep Blending (651 MB, contains the first four scenes):
  <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip>
- COLMAP South Building (400 MB):
  <https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip>
- COLMAP Gerrard Hall (959 MB):
  <https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip>

Choose scenes that differ from each other — at least one indoor and one outdoor —
so your report can say something about how scene type affects the result.

> **Run COLMAP yourself.** These bundles ship a `sparse/` folder with camera
> poses already solved. Recovering those poses from the images is part of this
> assignment, so use only the `images/` folder as input. You may compare against
> the supplied poses in your report, but not substitute them.

**For each scene, report**

- COLMAP report: how many images are registered out of the total, and any scene fragmentation.
- Training configuration: steps, densification strategy, final gaussian count.
- Final PSNR and SSIM on held-out views.
- Wall-clock time per stage.

---

## Step 4 — Reconstruct your own scene (40 marks)

Photograph something yourself and reconstruct it. Pick something interesting in your daily life. Reconstruction quality is
decided by your capture far more than by any training setting.

**Requirements**

- At least 100 photographs, taken by you. 150–300 for a room.
- The subject must be static; no moving people or vehicles.
- Describe your capture strategy, and what you would do differently next time.

**Capture guidance**

- Aim for roughly 70% overlap between neighbouring shots.
- **Step sideways; do not only pivot on the spot.** Splatting needs parallax, and a
  panorama shot from one point has none. This is the single most common failure.
- Cover the subject at two or three different heights.
- Lock exposure and focus before you start. Auto-exposure drift between frames
  bakes brightness seams into the model.
- Point deliberately into corners — they anchor the geometry.
- Avoid mirrors, plain glass, blank untextured walls, and changing light.
- Do not photograph ceilings or floors flat-on. A frame filled with repetitive
  ceiling tiles shares almost nothing with its neighbours and will fail to
  register. Angle the shot so walls or furniture stay in frame.

---

## What to submit

### 1. Code

- Scripts or notebooks that run your pipeline end to end.
- A README with the exact commands, in order.
- **Do not commit** datasets, checkpoints, `.ply` or `.splat` files. Commit the
  code that produces them.

### 2. Report

PDF (at most 3 pages).

- **How you used the third-party libraries.** What COLMAP does, what gsplat
  does, which APIs or commands you called, what you had to configure and why.
  Explain the pipeline, do not just paste commands.
- **Screenshots.** Three per sample scene (nine total) and **five of your own
  scene**, from clearly different viewpoints. Include at least one that shows a
  weakness — a hole, a floater, a smeared region — and say why it is there.
- **Computational cost.** A table covering, per scene: SfM time, training time,
  peak VRAM, final gaussian count, output file size. State your CPU, GPU and RAM.
- **What went wrong.** What failed, how you diagnosed it, what you changed. A
  report with no failures in it is not more impressive; it is less believable.

### 3. Videos

- One camera fly-through of **your own** scene.
- One of **a sample** scene.
- 20–60 seconds each, smooth camera motion. State how you produced them: a
  scripted trajectory, a screen recording of an interactive viewer, or something
  else.

---

## Optional extensions (10 bonus marks, included in 100 marks)

**A sensible initial camera.** Most splat viewers open at a hardcoded
default pose, which on your scene may sit inside a wall — the screen fills with
noise and the reconstruction looks broken when it is fine. Derive an opening
camera from the reconstruction itself. Explain the coordinate convention your
viewer expects, and beware: if your trainer normalises world space, the
gaussians are not in raw COLMAP coordinates.

**Faster or lighter training.** Make training measurably faster or use
less memory, at matched quality. Report a before/after measurement — time, peak
VRAM, PSNR — not just a claim. Ideas: a different densification strategy, a cap
on the gaussian count, packed rasterisation, fewer spherical-harmonic bands,
lower input resolution. State the quality cost honestly.

**Other initialization than COLMAP** COLMAP can be used as the primary tool to generate initial GS points. Can you find some other way?

---

## Marking

| Item | Marks |
|---|---|
| Step 1 — Environment | 5 |
| Step 2 — Package installation | 5 |
| Step 3 — Three sample scenes | 40 |
| Step 4 — Your own scene | 40 |
| **Core total** | **90** |
| Optional extensions | 10 |
| **Maximum** | **100** |

Within Steps 3 and 4, marks are awarded for a working reconstruction, evidence
that you understand what each stage did, and honest reporting — not for the
highest PSNR. A modest result you can explain scores better than a good one you
cannot.

* Code Score (30%): Includes source code of step 1 (5pt), 2 (5pt), 3/4 (20pt)
* Report Score (50%): Includes report for step 3 (20pt) and 4 (20pt) and optional parts (10pt)
* Oral Score (20%): Includes questions for step 3 (10pt) and 4 (10pt)

---

## Academic integrity

The photographs for Step 4 must be taken by you. You may use any open-source
library, including complete reference pipelines, provided you say clearly in your
report what you used, what you wrote, and what you changed. Using a third-party
tool is expected; misrepresenting it as your own work is not.

---


