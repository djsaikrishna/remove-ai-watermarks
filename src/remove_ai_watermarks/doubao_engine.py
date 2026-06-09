"""Doubao visible watermark removal engine.

Doubao (ByteDance) stamps every generated image with a visible "豆包AI生成"
(Doubao AI generated) text strip in the bottom-right corner -- the explicit AIGC
label mandated by China's TC260 standard, a near-white semi-transparent overlay.

Removal is **reverse-alpha blending** against a captured alpha map
(``original = (wm - a*logo)/(1-a)``), always NCC-aligned to the actual mark plus a
thin residual inpaint over the glyph footprint. This is one of the three text-mark
engines that share :class:`remove_ai_watermarks._text_mark_engine.TextMarkEngine`;
this module supplies only Doubao's tuned :class:`TextMarkConfig` (bottom-right corner,
``assets/doubao_alpha.png`` rebuilt by ``scripts/visible_alpha_solve.py``). Arbitrary-
region inpainting still lives in ``region_eraser`` / the ``erase`` command.
"""
# The module-level _alpha_template / _glyph_silhouette / _template_match_score below
# are thin test-facing shims (imported by tests/), so pyright's src-only pass sees them
# as unused; the use is cross-module.
# pyright: reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import _text_mark_engine
from remove_ai_watermarks._text_mark_engine import TextMarkConfig, TextMarkDetection, TextMarkEngine

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# Locate geometry as a fraction of image WIDTH (the mark scales with width, anchored
# bottom-right). The box is GENEROUSLY wider than the mark and reaches close to the
# corner so a per-image re-rasterization shift stays inside the NCC alignment search.
WM_WIDTH_FRAC = 0.22
WM_HEIGHT_FRAC = 0.075
MARGIN_RIGHT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.004

# Glyph appearance: a light, low-saturation gray rendered brighter than the local
# background (white top-hat), so a white-paper document is left untouched.
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 150  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 12  # glyph must exceed the local background by this many levels

# Shape-consistent detection: match the bundled alpha glyph silhouette against the
# corner candidate via TM_CCOEFF_NORMED (keys on glyph SHAPE, not coverage; #23).
DETECT_MIN_COVERAGE = 0.04
DETECT_NCC_THRESHOLD = 0.4

# Reverse-alpha geometry, emitted by scripts/visible_alpha_solve.py at the captured
# width. Removal always tries fixed AND NCC-aligned placement and keeps the lower
# residual, then a thin footprint inpaint clears the leftover edges.
_ALPHA_NATIVE_WIDTH = 2048
_ALPHA_LOGO_BGR: tuple[float, float, float] = (255.0, 255.0, 255.0)
_ALPHA_WIDTH_FRAC = 0.1636  # asset width / image width -- the alignment scale seed
_ALPHA_HEIGHT_FRAC = 0.0405
_ALPHA_MARGIN_RIGHT_FRAC = 0.0132
_ALPHA_MARGIN_BOTTOM_FRAC = 0.0166
_ALPHA_ALIGN_SEARCH = (0.88, 1.12, 25)
_RESIDUAL_ALPHA_FLOOR = 0.05
_RESIDUAL_DILATE = 5
_RESIDUAL_INPAINT_RADIUS = 2

_CONFIG = TextMarkConfig(
    name="Doubao",
    asset_name="doubao_alpha.png",
    corner="br",
    margin_floor=4,
    width_frac=WM_WIDTH_FRAC,
    height_frac=WM_HEIGHT_FRAC,
    margin_x_frac=MARGIN_RIGHT_FRAC,
    margin_bottom_frac=MARGIN_BOTTOM_FRAC,
    max_saturation=MAX_SATURATION,
    logo_min_luma=LOGO_MIN_LUMA,
    tophat_delta=TOPHAT_DELTA,
    morph_open_size=5,
    detect_min_coverage=DETECT_MIN_COVERAGE,
    detect_ncc_threshold=DETECT_NCC_THRESHOLD,
    alpha_width_frac=_ALPHA_WIDTH_FRAC,
    alpha_height_frac=_ALPHA_HEIGHT_FRAC,
    alpha_margin_x_frac=_ALPHA_MARGIN_RIGHT_FRAC,
    alpha_margin_bottom_frac=_ALPHA_MARGIN_BOTTOM_FRAC,
    alpha_align_search=_ALPHA_ALIGN_SEARCH,
    min_gw=8,
    alpha_logo_bgr=_ALPHA_LOGO_BGR,
    residual_alpha_floor=_RESIDUAL_ALPHA_FLOOR,
    residual_dilate=_RESIDUAL_DILATE,
    residual_inpaint_radius=_RESIDUAL_INPAINT_RADIUS,
)

# Doubao-specific aliases for the shared detection result/engine.
DoubaoDetection = TextMarkDetection


def _alpha_template() -> NDArray[Any] | None:
    """The bundled Doubao alpha template (float [0,1]), or None."""
    return _text_mark_engine.load_alpha_template(_CONFIG.asset_name)


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary "豆包AI生成" silhouette (255 = glyph) from the alpha map, or None."""
    return _text_mark_engine.glyph_silhouette(_CONFIG.asset_name)


def _template_match_score(box_mask: NDArray[Any], image_width: int) -> float:
    """TM_CCOEFF_NORMED of the Doubao glyph silhouette against ``box_mask``."""
    return _text_mark_engine.template_match_score(box_mask, image_width, _CONFIG)


class DoubaoEngine(TextMarkEngine):
    """Remove the visible Doubao "豆包AI生成" watermark (locate -> mask -> reverse-alpha)."""

    def __init__(self) -> None:
        super().__init__(_CONFIG)


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
