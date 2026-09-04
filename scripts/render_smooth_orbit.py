"""Render a stable, constant-speed RGB orbit from a trained gsplat checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import tqdm


def build_constant_speed_orbit(
    poses: np.ndarray,
    frame_count: int,
    radius_scale: float,
    focus_point_fn,
    viewmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a closed ellipse with stable up direction and near-constant speed."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if radius_scale <= 0:
        raise ValueError("radius_scale must be positive")

    center = focus_point_fn(poses)
    camera_positions = poses[:, :3, 3]
    height = float(camera_positions[:, 2].mean())
    offset = np.array([center[0], center[1], height])

    radii = np.percentile(np.abs(camera_positions - offset), 90, axis=0)[:2]
    radii = np.maximum(radii * radius_scale, 1e-6)

    # First sample densely, then resample by accumulated arc length so that the
    # camera does not speed up and slow down around an elliptical path.
    dense_count = max(frame_count * 10, 4096)
    dense_theta = np.linspace(0.0, 2.0 * np.pi, dense_count + 1)
    dense_positions = np.stack(
        [
            offset[0] + radii[0] * np.cos(dense_theta),
            offset[1] + radii[1] * np.sin(dense_theta),
            np.full_like(dense_theta, height),
        ],
        axis=-1,
    )
    segment_lengths = np.linalg.norm(np.diff(dense_positions, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_lengths = np.linspace(
        0.0, cumulative_lengths[-1], frame_count, endpoint=False
    )
    theta = np.interp(target_lengths, cumulative_lengths, dense_theta)
    positions = np.stack(
        [
            offset[0] + radii[0] * np.cos(theta),
            offset[1] + radii[1] * np.sin(theta),
            np.full_like(theta, height),
        ],
        axis=-1,
    )

    average_up = poses[:, :3, 1].mean(axis=0)
    average_up /= np.linalg.norm(average_up)
    up_axis = int(np.argmax(np.abs(average_up)))
    stable_up = np.eye(3)[up_axis] * np.sign(average_up[up_axis])

    orbit = np.stack([viewmatrix(center - p, stable_up, p) for p in positions])
    return orbit, center, radii


def build_smoothed_capture_path(
    poses: np.ndarray,
    frame_count: int,
    smoothing_sigma: float,
    focus_point_fn,
    viewmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth and resample the captured camera path without leaving its coverage."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if smoothing_sigma <= 0:
        raise ValueError("smoothing_sigma must be positive")

    positions = poses[:, :3, 3]
    center = focus_point_fn(poses)
    focus_distance = float(np.median(np.linalg.norm(center - positions, axis=1)))
    # viewmatrix() stores the camera's forward direction in column 2. Preserve
    # that direction through an auxiliary point behind each camera: p - d*z.
    # After smoothing, viewmatrix(p - auxiliary, up, p) reconstructs +z.
    orientation_points = positions - focus_distance * poses[:, :3, 2]

    # Smooth along the recorded order without assuming that the first and last
    # images form a loop. Some datasets contain a large pose change at that
    # boundary, so circular smoothing would invent a bad camera move there.
    smooth_positions = gaussian_filter1d(
        positions, sigma=smoothing_sigma, axis=0, mode="nearest"
    )
    smooth_orientation = gaussian_filter1d(
        orientation_points, sigma=smoothing_sigma, axis=0, mode="nearest"
    )
    smooth_up = gaussian_filter1d(
        poses[:, :3, 1], sigma=smoothing_sigma, axis=0, mode="nearest"
    )

    segment_lengths = np.linalg.norm(np.diff(smooth_positions, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_lengths = np.linspace(0.0, cumulative_lengths[-1], frame_count)

    sampled_positions = np.stack(
        [
            np.interp(target_lengths, cumulative_lengths, smooth_positions[:, i])
            for i in range(3)
        ],
        axis=-1,
    )
    sampled_orientation = np.stack(
        [
            np.interp(
                target_lengths, cumulative_lengths, smooth_orientation[:, i]
            )
            for i in range(3)
        ],
        axis=-1,
    )
    sampled_up = np.stack(
        [
            np.interp(target_lengths, cumulative_lengths, smooth_up[:, i])
            for i in range(3)
        ],
        axis=-1,
    )

    # A single world-up vector is not valid for captures that look almost
    # vertically up or down. Start from the captured up direction, then
    # parallel-transport it from frame to frame by projecting the previous up
    # vector onto the new view plane. This creates the minimum necessary roll
    # and avoids both singularities and sudden 180-degree flips. The smoothed
    # captured up and least-aligned Cartesian axis are only fallbacks.
    path = []
    previous_up = None
    for position, orientation, source_up in zip(
        sampled_positions, sampled_orientation, sampled_up
    ):
        backward = position - orientation
        backward /= np.linalg.norm(backward)
        if previous_up is None:
            local_up = source_up - np.dot(source_up, backward) * backward
        else:
            local_up = previous_up - np.dot(previous_up, backward) * backward
        if np.linalg.norm(local_up) < 1e-6:
            local_up = source_up - np.dot(source_up, backward) * backward
        if np.linalg.norm(local_up) < 1e-6:
            fallback_axis = np.eye(3)[np.argmin(np.abs(backward))]
            local_up = fallback_axis - np.dot(fallback_axis, backward) * backward

        local_up /= np.linalg.norm(local_up)

        pose = viewmatrix(backward, local_up, position)
        path.append(pose)
        previous_up = pose[:, 1]

    path = np.stack(path)
    return path, center, path[0, :3, 1]


def build_handheld_path(
    poses: np.ndarray,
    frame_count: int,
    smoothing_sigma: float,
    focus_point_fn,
    viewmatrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow a continuous captured route while looking steadily into the scene."""
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if smoothing_sigma <= 0:
        raise ValueError("smoothing_sigma must be positive")

    positions = poses[:, :3, 3]
    center = focus_point_fn(poses)

    # COLMAP image-name order can contain separate capture runs. Split at
    # unusually large camera jumps and retain the longest continuous run rather
    # than interpolating through walls or across the room.
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    positive_steps = steps[steps > 1e-8]
    if len(positive_steps):
        jump_threshold = max(
            5.0 * float(np.median(positive_steps)),
            2.0 * float(np.percentile(positive_steps, 75)),
        )
        cuts = np.flatnonzero(steps > jump_threshold) + 1
    else:
        jump_threshold = 0.0
        cuts = np.empty(0, dtype=int)
    boundaries = np.concatenate([[0], cuts, [len(positions)]])
    segment_index = int(np.argmax(np.diff(boundaries)))
    segment_start = int(boundaries[segment_index])
    segment_end = int(boundaries[segment_index + 1])
    route = positions[segment_start:segment_end]
    if len(route) < 2:
        raise ValueError("Could not find a continuous captured camera route")

    smooth_positions = gaussian_filter1d(
        route, sigma=smoothing_sigma, axis=0, mode="nearest"
    )
    route_steps = np.linalg.norm(np.diff(smooth_positions, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(route_steps)])
    if cumulative_lengths[-1] <= 1e-8:
        raise ValueError("The selected captured camera route has no movement")
    target_lengths = np.linspace(0.0, cumulative_lengths[-1], frame_count)
    sampled_positions = np.stack(
        [
            np.interp(target_lengths, cumulative_lengths, smooth_positions[:, i])
            for i in range(3)
        ],
        axis=-1,
    )

    # Camera centers in an indoor handheld capture are approximately coplanar.
    # The least-varying PCA axis therefore provides a data-driven vertical axis,
    # without assuming that normalized scene Z is always up. Match its sign to
    # the captured camera Y axes so the rendered image is not upside down.
    centered_positions = positions - positions.mean(axis=0)
    _, principal_axes = np.linalg.eigh(np.cov(centered_positions.T))
    stable_up = principal_axes[:, 0]
    average_camera_up = poses[:, :3, 1].mean(axis=0)
    if np.dot(stable_up, average_camera_up) < 0.0:
        stable_up = -stable_up

    look_directions = center[None] - sampled_positions
    look_norms = np.linalg.norm(look_directions, axis=1)
    if np.min(look_norms) < 1e-6:
        raise ValueError("Handheld path passes too close to the scene focus point")
    normalized_look = look_directions / look_norms[:, None]
    if np.max(np.abs(normalized_look @ stable_up)) > 0.98:
        raise ValueError("Handheld view direction is too close to scene vertical")

    path = np.stack(
        [viewmatrix(center - position, stable_up, position) for position in sampled_positions]
    )
    detail = np.array([segment_start, segment_end, jump_threshold])
    return path, center, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsplat-source", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--path-type", choices=("ellipse", "captured", "handheld"), default="ellipse"
    )
    parser.add_argument("--radius-scale", type=float, default=0.9)
    parser.add_argument("--smoothing-sigma", type=float, default=8.0)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    for required_path in (args.data_dir, args.checkpoint, args.gsplat_source):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )

    examples_dir = args.gsplat_source / "examples"
    if not (examples_dir / "simple_trainer.py").is_file():
        raise FileNotFoundError(examples_dir / "simple_trainer.py")
    sys.path.insert(0, str(examples_dir))

    from datasets.traj import focus_point_fn, viewmatrix
    from simple_trainer import Config, Runner

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        data_dir=str(args.data_dir),
        data_factor=args.data_factor,
        result_dir=str(args.output.parent / "render_workspace"),
        disable_viewer=True,
        disable_video=True,
        packed=True,
    )
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)

    checkpoint = torch.load(
        args.checkpoint, map_location=runner.device, weights_only=True
    )
    for key in runner.splats.keys():
        if key not in checkpoint["splats"]:
            raise KeyError(f"Checkpoint is missing splat parameter: {key}")
        runner.splats[key].data = checkpoint["splats"][key]

    source_poses = runner.parser.camtoworlds
    if args.path_type == "ellipse":
        orbit, center, path_detail = build_constant_speed_orbit(
            source_poses[5:-5],
            frame_count=args.frames,
            radius_scale=args.radius_scale,
            focus_point_fn=focus_point_fn,
            viewmatrix=viewmatrix,
        )
    elif args.path_type == "captured":
        orbit, center, path_detail = build_smoothed_capture_path(
            source_poses,
            frame_count=args.frames,
            smoothing_sigma=args.smoothing_sigma,
            focus_point_fn=focus_point_fn,
            viewmatrix=viewmatrix,
        )
    else:
        orbit, center, path_detail = build_handheld_path(
            source_poses,
            frame_count=args.frames,
            smoothing_sigma=args.smoothing_sigma,
            focus_point_fn=focus_point_fn,
            viewmatrix=viewmatrix,
        )
    orbit_4x4 = np.concatenate(
        [
            orbit,
            np.repeat(
                np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(orbit), axis=0
            ),
        ],
        axis=1,
    )
    orbit_tensor = torch.from_numpy(orbit_4x4).float().to(runner.device)
    intrinsics = torch.from_numpy(list(runner.parser.Ks_dict.values())[0]).float()
    intrinsics = intrinsics[None].to(runner.device)
    width, height = list(runner.parser.imsize_dict.values())[0]

    start_time = time.perf_counter()
    writer = imageio.get_writer(args.output, fps=args.fps, macro_block_size=8)
    try:
        with torch.inference_mode():
            for index in tqdm.trange(args.frames, desc="Rendering smooth orbit"):
                renders, _, _ = runner.rasterize_splats(
                    camtoworlds=orbit_tensor[index : index + 1],
                    Ks=intrinsics,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    render_mode="RGB",
                )
                frame = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
                writer.append_data((frame.cpu().numpy() * 255).astype(np.uint8))
    finally:
        writer.close()

    elapsed = time.perf_counter() - start_time
    metadata = {
        "checkpoint": str(args.checkpoint),
        "frames": args.frames,
        "fps": args.fps,
        "duration_seconds": args.frames / args.fps,
        "render_wall_time_seconds": elapsed,
        "path_type": args.path_type,
        "radius_scale": args.radius_scale,
        "smoothing_sigma": args.smoothing_sigma,
        "focus_point": center.tolist(),
        "path_detail": path_detail.tolist(),
        "width": width,
        "height": height,
        "rgb_only": True,
        "constant_speed": True,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Video saved to: {args.output}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Duration: {metadata['duration_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
