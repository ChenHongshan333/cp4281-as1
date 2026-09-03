"""Patch gsplat 1.5.3's JIT compiler flags for Microsoft Visual C++."""

from pathlib import Path

import gsplat


backend_path = Path(gsplat.__file__).resolve().parent / "cuda" / "_backend.py"
source = backend_path.read_text(encoding="utf-8")

original = """        opt_level = "-O0" if FAST_COMPILE else "-O3"
        extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
"""

replacement = """        opt_level = "-O0" if FAST_COMPILE else "-O3"
        if os.name == "nt":
            # MSVC does not support GCC's -Wno-attributes or the -O3 level.
            extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]
        else:
            extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
"""

if replacement in source:
    print(f"gsplat Windows patch already applied: {backend_path}")
elif original in source:
    backend_path.write_text(source.replace(original, replacement, 1), encoding="utf-8")
    print(f"Applied gsplat Windows patch: {backend_path}")
else:
    raise RuntimeError(
        "Expected gsplat 1.5.3 compiler-flag block was not found. "
        "Refusing to patch an unknown gsplat version."
    )
