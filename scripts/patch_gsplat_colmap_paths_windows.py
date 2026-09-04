"""Patch gsplat 1.5.3's COLMAP parser for mixed path separators on Windows."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gsplat_source", type=Path)
    args = parser.parse_args()

    parser_path = args.gsplat_source / "examples" / "datasets" / "colmap.py"
    source = parser_path.read_text(encoding="utf-8")

    original = """        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
"""
    replacement = """        # COLMAP stores nested image names with forward slashes, while
        # os.path.relpath uses backslashes on Windows. Normalize both sides.
        colmap_to_image = {
            os.path.normpath(colmap_file): image_file
            for colmap_file, image_file in zip(colmap_files, image_files)
        }
        image_paths = [
            os.path.join(image_dir, colmap_to_image[os.path.normpath(f)])
            for f in image_names
        ]
"""

    if replacement in source:
        print(f"gsplat COLMAP path patch already applied: {parser_path}")
    elif original in source:
        parser_path.write_text(
            source.replace(original, replacement, 1), encoding="utf-8"
        )
        print(f"Applied gsplat COLMAP path patch: {parser_path}")
    else:
        raise RuntimeError(
            "Expected gsplat 1.5.3 COLMAP path block was not found. "
            "Refusing to patch an unknown gsplat version."
        )


if __name__ == "__main__":
    main()
