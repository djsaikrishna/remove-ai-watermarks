"""High-level convenience API: clean an image (or array) in one call.

The low-level building blocks live in ``watermark_registry`` (localize -> fill) and
``image_io`` (Unicode-safe, alpha-preserving IO). This module ties them into the two
calls a caller usually wants, so a library user does not have to decode images, wire
up metadata provenance, or preserve the alpha channel by hand:

    import remove_ai_watermarks as raiw
    raiw.remove_visible("in.png", "out.png")                 # path -> file, provenance auto
    result, removed = raiw.remove_visible(bgr_array)         # array -> array
    raiw.remove_visible("shot.png", "out.png", sensitivity="assume_ai")
    raiw.visible_provenance("in.png")                        # -> frozenset({"gemini"})

Imports stay lazy (inside the functions), so ``import remove_ai_watermarks`` is cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from remove_ai_watermarks.watermark_registry import Backend, Sensitivity


def visible_provenance(source: str | Path) -> frozenset[str]:
    """Vendor keys that the file's local metadata confirms, the evidence that drives
    the ``auto`` sensitivity (relaxing a corroborated mark's detection trust gate).

    Mapping: a Google/Gemini C2PA issuer -> ``"gemini"``; a China-AIGC (TC260) label
    -> ``"doubao"``/``"jimeng"``; a ``samsung_genai`` marker -> ``"samsung"``.
    Best-effort: any read error yields an empty set (no relaxation). Metadata-only, so
    it never loads cv2/torch.
    """
    import contextlib

    keys: set[str] = set()
    with contextlib.suppress(Exception):
        from remove_ai_watermarks import identify, metadata

        rep = identify.identify(Path(source), check_visible=False, check_invisible=False)
        platform = (rep.platform or "").lower()
        if "google" in platform or "gemini" in platform:
            keys.add("gemini")
        if metadata.aigc_label(Path(source)):
            keys |= {"doubao", "jimeng"}
        if metadata.samsung_genai(Path(source)):
            keys.add("samsung")
    return frozenset(keys)


def remove_visible(
    source: str | Path | NDArray[Any],
    output: str | Path | None = None,
    *,
    sensitivity: Sensitivity = "auto",
    backend: Backend = "auto",
    strip_metadata: bool = True,
) -> tuple[NDArray[Any], list[str]]:
    """Remove every detected known visible AI mark (Gemini sparkle, Doubao/Jimeng/
    Samsung text, the Jimeng pill) via localize -> fill, returning ``(result_bgr,
    [labels removed])``.

    ``source`` is a file path OR a BGR ndarray. For a PATH, metadata provenance is read
    automatically (so ``sensitivity="auto"`` recovers a moved/faint mark whenever the
    file still carries its provenance) and the alpha channel is preserved on write; for
    an ARRAY there is no metadata to read and no separate alpha plane. When ``output``
    is given the cleaned image is written there (alpha rejoined for a path source); the
    array is always returned as well, so an empty ``removed`` list tells a caller nothing
    known was found (e.g. route to the diffusion ``all`` path or ``erase``).

    ``sensitivity`` (``auto``/``strict``/``assume_ai``) and ``backend``
    (``auto``/``cv2``/``migan``/``lama``) are the same knobs as the CLI. Pass
    ``sensitivity="assume_ai"`` for a metadata-stripped screenshot the caller knows is
    AI-generated (best recall, at the cost of a small near-lossless fill on a clean
    corner if the guess is wrong).

    ``strip_metadata`` (default True, matching the CLI ``visible --strip-metadata``)
    also strips AI provenance metadata (C2PA/EXIF/XMP/IPTC) from the written output via
    the lossless :func:`metadata.remove_ai_metadata`, so a library call does exactly
    what the CLI does. Only applies when ``output`` is given.
    """
    from remove_ai_watermarks import image_io, watermark_registry

    alpha: NDArray[Any] | None = None
    provenance: frozenset[str] = frozenset()
    if isinstance(source, (str, Path)):
        path = Path(source)
        bgr, alpha = image_io.read_bgr_and_alpha(path)
        if bgr is None:
            raise ValueError(f"Could not read image: {source}")
        provenance = visible_provenance(path)
    else:
        bgr = source

    result, removed = watermark_registry.remove_auto_marks(
        bgr, sensitivity=sensitivity, provenance=provenance, backend=backend
    )
    if output is not None:
        same_format = isinstance(source, (str, Path)) and Path(source).suffix.lower() == Path(output).suffix.lower()
        if not removed and same_format:
            # Nothing was removed: copy the ORIGINAL bytes verbatim instead of a lossy
            # re-encode of its decode, so the pixels stay bit-identical (the metadata
            # strip below is lossless, so it does not disturb them either).
            import shutil

            shutil.copyfile(source, output)  # type: ignore[arg-type]
        else:
            image_io.write_bgr_with_alpha(output, result, alpha)
        if strip_metadata:
            from remove_ai_watermarks import metadata

            metadata.remove_ai_metadata(Path(output), Path(output))
    return result, removed
