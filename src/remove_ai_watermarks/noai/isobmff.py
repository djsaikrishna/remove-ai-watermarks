"""Minimal ISOBMFF box walker for stripping C2PA from AVIF / HEIF / MP4 / JPEG-XL.

The ISO Base Media File Format wraps content in nested ``[size:4][type:4][...]``
boxes. C2PA stores its manifest in a top-level ``uuid`` box keyed by the
C2PA UUID; JPEG-XL uses a ``jumb`` box (JUMBF) instead. To strip provenance
without re-encoding the image, we walk the top-level box list, drop boxes that
carry C2PA, and emit the rest verbatim. The codestream (``mdat`` for ISOBMFF,
``jxlc`` / ``jxlp`` for JPEG-XL) is untouched, so pixel data is preserved
bit-for-bit.

This file intentionally avoids dependencies on format-specific libraries
(pillow-heif, pillow-jxl, pymp4) so it works on systems where they aren't
installed.

Reference: ISO/IEC 14496-12 (ISOBMFF) and C2PA 2.1 spec §11.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from remove_ai_watermarks.metadata import C2PA_UUID

# Top-level box types that carry C2PA payload. ``uuid`` boxes are checked
# against ``C2PA_UUID`` before being stripped; ``jumb`` boxes are always
# stripped (JPEG-XL uses them exclusively for JUMBF).
C2PA_BOX_TYPES: frozenset[bytes] = frozenset({b"uuid", b"jumb"})


def _iter_top_level_boxes(data: bytes) -> Iterator[tuple[int, int, bytes, int]]:
    """Yield ``(start, end, type, payload_offset)`` for each top-level box.

    Handles all three ISOBMFF box-size encodings:
    - ``size > 1``: 32-bit size field is the total box length.
    - ``size == 1``: 64-bit ``largesize`` follows after the type field.
    - ``size == 0``: box runs to end of file.
    """
    pos = 0
    n = len(data)
    while pos + 8 <= n:
        size32 = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]
        if size32 == 1:
            if pos + 16 > n:
                return
            size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_off = pos + 16
        elif size32 == 0:
            size = n - pos
            payload_off = pos + 8
        else:
            size = size32
            payload_off = pos + 8
        if size < (payload_off - pos) or pos + size > n:
            return
        yield pos, pos + size, box_type, payload_off
        pos += size


def is_isobmff(data: bytes) -> bool:
    """Cheap sniff: ISOBMFF files start with an ``ftyp`` box."""
    return len(data) >= 8 and data[4:8] == b"ftyp"


def strip_c2pa_boxes(data: bytes) -> tuple[bytes, int]:
    """Return ``(cleaned_bytes, stripped_count)``.

    Walks top-level boxes; drops any ``uuid`` box whose UUID equals
    ``C2PA_UUID`` and any ``jumb`` box (JPEG-XL JUMBF container). All other
    boxes are emitted verbatim. If the input is not ISOBMFF-shaped, returns
    it unchanged.
    """
    if not is_isobmff(data):
        return data, 0

    out = bytearray()
    stripped = 0
    for start, end, box_type, payload_off in _iter_top_level_boxes(data):
        if box_type in C2PA_BOX_TYPES:
            if box_type == b"uuid":
                # uuid boxes carry the 16-byte UUID immediately after the type.
                if payload_off + 16 <= end and data[payload_off : payload_off + 16] == C2PA_UUID:
                    stripped += 1
                    continue
            else:  # b"jumb"
                stripped += 1
                continue
        out.extend(data[start:end])
    return bytes(out), stripped
