"""Verify the CUDA PyTorch, gsplat, and COLMAP installation."""

import os
import shutil
import subprocess

import torch
from gsplat import rasterization


def verify_torch_and_gsplat() -> None:
    print("=== PyTorch and gsplat ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the CUDA GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")

    means = torch.tensor([[0.0, 0.0, 3.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    opacities = torch.tensor([1.0], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        device=device,
    ).unsqueeze(0)

    rendered_colors, rendered_alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width=64,
        height=64,
    )
    torch.cuda.synchronize()

    print(f"Rendered colors: {tuple(rendered_colors.shape)}")
    print(f"Rendered alphas: {tuple(rendered_alphas.shape)}")
    print("gsplat.rasterization: OK")


def verify_colmap() -> None:
    print("\n=== COLMAP ===")
    launcher = shutil.which("COLMAP.bat") or shutil.which("colmap.exe")
    if launcher is None:
        raise RuntimeError("COLMAP was not found on PATH")

    if launcher.lower().endswith(".bat"):
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", launcher, "-h"]
    else:
        command = [launcher, "-h"]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    print("\n".join((result.stdout or result.stderr).splitlines()[:3]))


if __name__ == "__main__":
    verify_torch_and_gsplat()
    verify_colmap()
