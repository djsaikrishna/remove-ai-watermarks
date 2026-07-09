"""Shared base for the visible text-mark detectors/localizers (localize -> fill).

The Doubao "豆包AI生成", Jimeng "★ 即梦AI", and Samsung "✦ Contenuti generati
dall'AI" marks are the SAME algorithm: anchor a bottom-corner box by width-relative
geometry, extract the light low-saturation glyph candidate (white top-hat), detect
by matching the bundled alpha-glyph silhouette via ``TM_CCOEFF_NORMED``, and build a
removal MASK from the glyph blob's bounding box (:meth:`footprint_mask`) for the
shared fill (region_eraser). The mask is template-FREE -- the top-hat glyph bbox, not
a fixed alpha-template placement -- so a re-rendered or differently-placed mark (e.g.
a non-Italian Samsung string) is still masked. The old reverse-alpha pixel recovery
(``original = (wm - a*logo)/(1-a)``) is gone.

They differ ONLY in a bounded set of tuned values captured by :class:`TextMarkConfig`:
the constants, the bundled silhouette asset, the corner (Doubao/Jimeng bottom-right,
Samsung bottom-left), and a few structural knobs. Each engine module is a thin
:class:`TextMarkEngine` subclass plus the test-facing module constants/helpers.

Gemini stays a SEPARATE engine (``gemini_engine``): its multi-size sparkle model is
genuinely different, not a tuned variant of this one.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Minimum image short side (px) for text-mark DETECTION. Below this the glyph
# template degrades to the ``min_gw`` floor (~8 px) and TM_CCOEFF_NORMED on a few
# pixels is noise, so an unrelated small geometric shape can spuriously correlate
# with the CJK silhouette (2026-06-26 FP: a 48x48 app icon -- a blue chevron --
# scored Doubao 0.41 / Jimeng 0.47, both above their thresholds). The FP is purely
# a small-size artifact: the same icon upscaled collapses to ~0.06-0.10 NCC at 256
# px and above. A real AI-generation text label is stamped on a full-resolution
# render (the captured samples are 1086-2048 px wide), so 200 px sits far below any
# genuine mark while killing the icon/thumbnail noise band (<=96 px). Detection is
# skipped (verdict stays "unknown", the safe default) rather than risk a false
# positive; removal is gated on detection, so it is suppressed too.
_MIN_DETECT_SHORT_SIDE = 200

# Provenance-confirmed NCC relaxation. When external metadata already confirms the
# vendor (so the mark is present with high prior), a faint or slightly re-rendered
# glyph that scores just below the standard NCC gate is still trusted. 0.7 recovers
# the near-threshold marks without dropping so low that an unrelated corner texture
# on a (provenance-confirmed) image would match -- the coverage gate still applies.
_PROVENANCE_NCC_FACTOR = 0.7


@dataclass(frozen=True)
class TextMarkConfig:
    """All per-mark tuning for a text-mark detector/localizer."""

    name: str  # short label for log lines (e.g. "Doubao")
    asset_name: str  # bundled alpha PNG under assets/ (e.g. "doubao_alpha.png")
    corner: Literal["br", "bl"]  # bottom-right (Doubao/Jimeng) or bottom-left (Samsung)
    margin_floor: int  # min margin in px for locate (4 for br marks, 2 for Samsung)
    # locate geometry (fraction of image WIDTH)
    width_frac: float
    height_frac: float
    margin_x_frac: float  # right margin (br) or left margin (bl)
    margin_bottom_frac: float
    # glyph appearance
    max_saturation: float
    logo_min_luma: float
    tophat_delta: float
    morph_open_size: int  # MORPH_OPEN kernel side (5 for br marks, 3 for Samsung)
    # detection
    detect_min_coverage: float
    detect_ncc_threshold: float
    # alpha-map glyph geometry (fraction of WIDTH) emitted by
    # scripts/visible_alpha_solve.py, sizing the detection silhouette for
    # template_match_score
    alpha_width_frac: float
    alpha_height_frac: float
    min_gw: int  # minimum glyph width for the template match (8 br, 16 Samsung)


@dataclass
class TextMarkLocation:
    """Located watermark box, in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class TextMarkDetection:
    """Result of visible text-mark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


# Alpha / silhouette templates, cached per asset name (the originals cached per
# module global; this keys by asset so the three engines share the loader without
# re-reading). Only SUCCESSFUL loads are cached, so a missing asset is retried.
_alpha_cache: dict[str, NDArray[Any]] = {}
_silhouette_cache: dict[str, NDArray[Any]] = {}


def load_alpha_template(asset_name: str) -> NDArray[Any] | None:
    """Lazily load the bundled alpha template (float [0,1]) for ``asset_name``, or None."""
    cached = _alpha_cache.get(asset_name)
    if cached is not None:
        return cached
    path = Path(__file__).parent / "assets" / asset_name
    img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _alpha_cache[asset_name] = img.astype(np.float32) / 255.0
    return _alpha_cache[asset_name]


def glyph_silhouette(asset_name: str) -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the bundled alpha map, or None."""
    cached = _silhouette_cache.get(asset_name)
    if cached is not None:
        return cached
    at = load_alpha_template(asset_name)
    if at is None:
        return None
    _silhouette_cache[asset_name] = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache[asset_name]


def template_match_score(box_mask: NDArray[Any], image_width: int, config: TextMarkConfig) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``.

    ``TM_CCOEFF_NORMED`` keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual glyph shape does.
    """
    sil = glyph_silhouette(config.asset_name)
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(config.min_gw, int(config.alpha_width_frac * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(config.alpha_height_frac * image_width)))
    if gw < config.min_gw or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class TextMarkEngine:
    """Visible text-mark detector/localizer (locate -> mask -> detect; mask feeds the fill)."""

    def __init__(self, config: TextMarkConfig) -> None:
        self.config = config

    # ── Templates (delegate to the asset-keyed module cache) ────────────

    def _alpha_template(self) -> NDArray[Any] | None:
        return load_alpha_template(self.config.asset_name)

    def _glyph_silhouette(self) -> NDArray[Any] | None:
        return glyph_silhouette(self.config.asset_name)

    def _template_match_score(self, box_mask: NDArray[Any], image_width: int) -> float:
        return template_match_score(box_mask, image_width, self.config)

    # ── Locate ──────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> TextMarkLocation:
        """Anchor the watermark box in the configured bottom corner by geometry."""
        c = self.config
        h, w = image.shape[:2]
        wm_w = max(40, int(w * c.width_frac))
        wm_h = max(16, int(w * c.height_frac))
        margin_x = max(c.margin_floor, int(w * c.margin_x_frac))
        margin_b = max(c.margin_floor, int(w * c.margin_bottom_frac))
        x = max(0, w - margin_x - wm_w) if c.corner == "br" else min(margin_x, max(0, w - wm_w))
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return TextMarkLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ────────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any]:
        """Build a box-sized uint8 mask (255 = watermark glyph) for ``loc``.

        Returns just the glyph mask of the located box (shape ``(loc.h, loc.w)``),
        not a full-frame array: every caller immediately crops to ``loc.bbox``, so
        allocating a full ``(h, w)`` mask and embedding the box was O(image) work
        and memory for an O(box) result -- a wasted full-frame uint8 allocation on
        each detect (~12 MB on a 12 MP frame, recomputed per text-mark detector on
        the memory-tight identify path). The box mask is byte-identical to the old
        full-frame mask cropped to ``loc.bbox``.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is left
        untouched (nothing brighter than its surroundings is masked there).
        """
        c = self.config
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault native code on some platforms. Skip the cv2 pipeline.
        if bh < 16 or bw < 16:
            return np.zeros((bh, bw), np.uint8)
        # Normalize the ROI to 3-channel BGR (grayscale / BGRA would break axis=2).
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < c.max_saturation

        # Local background model: a strong Gaussian blur (sigma ~ box height); the
        # white top-hat (luma - local_bg) lights up bright thin strokes regardless
        # of the absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > c.tophat_delta) & (luma > c.logo_min_luma)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        k = c.morph_open_size
        return cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    # ── Detect ──────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> TextMarkDetection:
        """Detect the mark by matching the alpha-template glyph silhouette against
        the corner candidate (``TM_CCOEFF_NORMED``); keys on glyph SHAPE, not coverage.

        ``provenance`` signals that external metadata already confirms this vendor
        (China-AIGC / byteimg for Doubao/Jimeng, ``samsung_genai`` for Samsung); the
        NCC gate exists to keep a corner texture on an UNRELATED image from matching
        the glyph silhouette, so when provenance confirms the vendor it is relaxed by
        ``_PROVENANCE_NCC_FACTOR`` to recover a faint or slightly re-rendered mark.
        """
        c = self.config
        det = TextMarkDetection()
        if image is None or image.size == 0:
            return det
        # Guard against the small-image NCC-noise false positive (see
        # _MIN_DETECT_SHORT_SIDE): an icon/thumbnail is too small to carry a real
        # text label, and the degraded few-pixel template spuriously correlates.
        if min(image.shape[:2]) < _MIN_DETECT_SHORT_SIDE:
            logger.debug(
                "%s detect: image short side %d < %d; too small to carry the mark, skipping.",
                c.name,
                min(image.shape[:2]),
                _MIN_DETECT_SHORT_SIDE,
            )
            return det
        loc = self.locate(image)
        box = self.extract_mask(image, loc)  # box-sized mask (== old full-frame cropped to bbox)
        _x, _y, bw, bh = loc.bbox
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        if coverage >= c.detect_min_coverage:
            score = self._template_match_score(box, image.shape[1])
            threshold = c.detect_ncc_threshold * (_PROVENANCE_NCC_FACTOR if provenance else 1.0)
            det.confidence = score
            det.detected = score >= threshold
            logger.debug(
                "%s detect: coverage=%.3f ncc=%.2f thr=%.2f detected=%s",
                c.name,
                coverage,
                score,
                threshold,
                det.detected,
            )
        return det

    # ── Inpaint footprint (for the inpaint-fallback removal path) ────────

    # Minimum glyph pixels for a template-free footprint. Below this the corner has
    # no real wordmark (a few top-hat specks), so without ``force`` there is nothing
    # to mask. A real strip covers hundreds of pixels.
    _MIN_GLYPH_PIXELS = 20

    def footprint_mask(
        self, image: NDArray[Any], *, force: bool = False, dilate: int | None = None
    ) -> NDArray[Any] | None:
        """Full-frame uint8 mask (255 = mark) of the mark footprint, for the shared
        fill removal path (cv2 / MI-GAN / LaMa), or None if no glyph is found.

        Template-FREE: localize the glyph blob with the top-hat :meth:`extract_mask`,
        take its bounding box in the corner, and fill that box solid (plus a small
        margin + dilation). Filling the enclosing rectangle -- not the sparse glyph
        strokes -- is what makes it robust: the top-hat under-segments individual
        strokes (which used to leave a "三包"-style residual ghost when the strokes
        themselves were the mask), but the inpaint reconstructs the whole wordmark
        rectangle from its surroundings, so a stroke missed by the top-hat is still
        covered. This drops the fixed alpha-template dependency, so a re-rendered or
        differently-localized mark (e.g. a non-Italian Samsung string) is still masked.

        With ``force`` and no glyph found, falls back to the whole geometry box (the
        ``--no-detect`` path). The caller gates on detection.
        """
        image = image_io.to_bgr(image)
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return None
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        glyph = self.extract_mask(image, loc)  # box-sized, 255 = glyph
        ys, xs = np.where(glyph > 0)
        if xs.size >= self._MIN_GLYPH_PIXELS:
            pad = max(4, int(0.10 * bh))
            rx1 = max(0, bx + int(xs.min()) - pad)
            rx2 = min(w, bx + int(xs.max()) + 1 + pad)
            ry1 = max(0, by + int(ys.min()) - pad)
            ry2 = min(h, by + int(ys.max()) + 1 + pad)
        elif force:
            rx1, ry1, rx2, ry2 = bx, by, min(w, bx + bw), min(h, by + bh)
        else:
            return None
        if rx1 >= rx2 or ry1 >= ry2:
            return None
        # Rectangular footprint + dilation is exactly region_eraser.boxes_to_mask (the
        # same primitive the shared fill uses); reuse it instead of re-inlining the
        # zeros/fill/MORPH_ELLIPSE-dilate here.
        from remove_ai_watermarks import region_eraser

        d = dilate if dilate is not None else max(3, int(0.02 * bw))
        return region_eraser.boxes_to_mask((h, w), [(rx1, ry1, rx2 - rx1, ry2 - ry1)], dilate=d)
