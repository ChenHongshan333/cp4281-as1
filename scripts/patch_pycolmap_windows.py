"""Patch the pinned legacy pycolmap binary reader for Windows.

The rmbrualla/pycolmap revision used by gsplat v1.5.3 reads COLMAP uint64
fields with the native ``L`` format. On Windows, native unsigned long is four
bytes, while the COLMAP binary format stores these fields as eight-byte
little-endian integers. This patch makes only the binary readers portable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPLACEMENTS = (
    ("struct.unpack('L', f.read(8))", "struct.unpack('<Q', f.read(8))", 3),
    ("struct.unpack('IiLL', f.read(24))", "struct.unpack('<IiQQ', f.read(24))", 1),
    (
        "struct.unpack('d' * num_params, f.read(8 * num_params))",
        "struct.unpack('<' + 'd' * num_params, f.read(8 * num_params))",
        1,
    ),
    ("struct.unpack('Q', f.read(8))", "struct.unpack('<Q', f.read(8))", 1),
    (
        "struct.unpack(f'{2*track_len}I', f.read(2 * track_len * 4))",
        "struct.unpack(f'<{2*track_len}I', f.read(2 * track_len * 4))",
        1,
    ),
)


def main() -> None:
    spec = importlib.util.find_spec("pycolmap.scene_manager")
    if spec is None or spec.origin is None:
        raise RuntimeError("The pinned pycolmap scene_manager module was not found")

    source_path = Path(spec.origin)
    source = source_path.read_text(encoding="utf-8")

    if all(old not in source and new in source for old, new, _ in REPLACEMENTS):
        print(f"pycolmap Windows binary-reader patch already applied: {source_path}")
        return

    for old, new, expected_count in REPLACEMENTS:
        actual_count = source.count(old)
        if actual_count != expected_count:
            raise RuntimeError(
                f"Refusing to patch unrecognized pycolmap source: expected "
                f"{expected_count} occurrence(s) of {old!r}, found {actual_count}"
            )
        source = source.replace(old, new)

    source_path.write_text(source, encoding="utf-8")
    print(f"Applied pycolmap Windows binary-reader patch: {source_path}")


if __name__ == "__main__":
    main()
