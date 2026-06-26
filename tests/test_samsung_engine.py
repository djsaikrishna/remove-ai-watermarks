"""Tests for the Samsung Galaxy AI visible-watermark engine.

No real Samsung sample is committed (the real-photo captures are gitignored, repo
is public), so detection/removal is exercised against a watermark synthesized from
the bundled alpha asset itself -- self-consistent and download-free. The mark is
anchored bottom-LEFT (unlike the bottom-right Doubao/Jimeng marks).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.samsung_engine import (
    _ALPHA_HEIGHT_FRAC,
    _ALPHA_LOGO_BGR,
    _ALPHA_MARGIN_BOTTOM_FRAC,
    _ALPHA_MARGIN_LEFT_FRAC,
    _ALPHA_NATIVE_WIDTH,
    _ALPHA_WIDTH_FRAC,
    DETECT_NCC_THRESHOLD,
    SamsungEngine,
    _alpha_template,
    _glyph_silhouette,
    _template_match_score,
)


def _compose(w: int, h: int, bg: float = 100.0):
    """Composite the real alpha (scaled to width ``w``) onto a flat bg by the
    engine's fixed bottom-left geometry. Returns ``(watermarked_uint8, mark_bool_mask)``."""
    img = np.full((h, w, 3), bg, np.float32)
    at = _alpha_template()
    gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
    ax = int(_ALPHA_MARGIN_LEFT_FRAC * w)
    ay = h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh
    amap = np.zeros((h, w), np.float32)
    amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
    a3 = amap[:, :, None]
    wm = (a3 * np.array(_ALPHA_LOGO_BGR, np.float32) + (1 - a3) * img).clip(0, 255).astype(np.uint8)
    return wm, amap > 0.15


class TestLocate:
    def test_box_anchored_bottom_left(self):
        eng = SamsungEngine()
        img = np.zeros((1448, 1086, 3), np.uint8)
        loc = eng.locate(img)
        assert loc.x < int(1086 * 0.03)  # hugs the left edge
        assert 1448 - (loc.y + loc.h) < int(1086 * 0.03)  # hugs the bottom

    def test_box_scales_with_width(self):
        eng = SamsungEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


class TestDetect:
    def test_clean_gradient_not_detected(self):
        eng = SamsungEngine()
        ramp = np.tile(np.linspace(0, 255, 1086, dtype=np.uint8), (1086, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert not eng.detect(img).detected

    def test_solid_blob_corner_not_detected(self):
        """A bright blob is not the glyph shape -> low correlation, not detected."""
        eng = SamsungEngine()
        img = np.zeros((1086, 1086, 3), np.uint8)
        x, y, bw, bh = eng.locate(img).bbox
        img[y + bh // 4 : y + bh * 3 // 4, x : x + bw // 2] = 200
        assert not eng.detect(img).detected

    def test_silhouette_loads(self):
        sil = _glyph_silhouette()
        assert sil is not None
        assert set(np.unique(sil)).issubset({0, 255})

    def test_match_score_shape_sensitive(self):
        """The glyph silhouette correlates with itself, not with a filled block."""
        sil = _glyph_silhouette()
        h, w = sil.shape
        box = np.zeros((h + 8, int(w / _ALPHA_WIDTH_FRAC * 0.2) + w), np.uint8)
        box[4 : 4 + h, 4 : 4 + w] = sil
        assert _template_match_score(box, _ALPHA_NATIVE_WIDTH) >= DETECT_NCC_THRESHOLD
        solid = np.full_like(box, 255)
        assert _template_match_score(solid, _ALPHA_NATIVE_WIDTH) < DETECT_NCC_THRESHOLD

    def test_small_image_guarded_from_false_positive(self):
        """Below the minimum short side a tiny geometric shape spuriously NCC-matches
        the glyph silhouette (the 2026-06-26 small-icon FP class). The size guard
        suppresses detection there. Bracket it: a real mark is detected at native
        size, but the same content downscaled below the guard is not."""
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        eng = SamsungEngine()
        assert eng.detect(wm).detected  # native: real mark detected
        assert not eng.detect(cv2.resize(wm, (150, 150))).detected  # below guard: suppressed

    def test_synthetic_mark_detected(self):
        """A watermark composed from the real alpha is detected at its threshold."""
        eng = SamsungEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        det = eng.detect(wm)
        assert det.detected
        assert det.confidence >= DETECT_NCC_THRESHOLD


class TestReverseAlpha:
    def test_alpha_asset_loads(self):
        at = _alpha_template()
        assert at is not None
        assert at.dtype.kind == "f"
        assert float(at.min()) >= 0.0
        assert float(at.max()) <= 1.0

    def test_logo_is_white(self):
        assert _ALPHA_LOGO_BGR == (255.0, 255.0, 255.0)

    def test_available_whenever_asset_present(self):
        eng = SamsungEngine()
        assert eng.reverse_alpha_available(np.zeros((1086, 1086, 3), np.uint8))
        assert eng.reverse_alpha_available(np.zeros((4054, 2958, 3), np.uint8))
        assert not eng.reverse_alpha_available(np.zeros((0, 0, 3), np.uint8))

    def test_removes_synthetic_mark(self):
        """Reverse-alpha + residual inpaint clears the composed mark (re-detect no
        longer fires)."""
        eng = SamsungEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        assert eng.detect(wm).detected
        out = eng.remove_watermark_reverse_alpha(wm)
        assert not eng.detect(out).detected

    @pytest.mark.parametrize(
        ("w", "h", "max_err"),
        [
            (_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33), 5.0),  # captured width
            (2958, 4054, 10.0),  # real-photo width (~2.7x native) -> NCC alignment generalizes
        ],
    )
    def test_recovers_flat_background(self, w, h, max_err):
        eng = SamsungEngine()
        wm, mark = _compose(w, h)
        assert float(np.abs(wm.astype(np.float32)[mark] - 100.0).mean()) > 15  # mark visible
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - 100.0).mean()) < max_err

    def test_far_region_untouched(self):
        """The residual inpaint only touches the bottom-left footprint; the
        opposite (top-right) corner stays pixel-identical."""
        eng = SamsungEngine()
        wm, _mark = _compose(_ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33))
        out = eng.remove_watermark_reverse_alpha(wm)
        h, w = wm.shape[:2]
        assert np.array_equal(wm[: h // 2, w // 2 :], out[: h // 2, w // 2 :])

    def test_recovers_shifted_mark_on_texture(self):
        """A real mark is re-rasterized a few px off its fixed slot, so removal must
        NCC-align to it (a too-tight locate box would let a corner-ward shift escape
        the search and leave a readable outline). Composes the real alpha SHIFTED on
        a known texture and asserts the texture is recovered."""
        eng = SamsungEngine()
        w, h = _ALPHA_NATIVE_WIDTH, int(_ALPHA_NATIVE_WIDTH * 1.33)
        at = _alpha_template()
        gw, gh = int(_ALPHA_WIDTH_FRAC * w), int(_ALPHA_HEIGHT_FRAC * w)
        ax = max(0, int(_ALPHA_MARGIN_LEFT_FRAC * w) + 9)  # shift right of the fixed slot
        ay = h - int(_ALPHA_MARGIN_BOTTOM_FRAC * w) - gh - 7  # shift up
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh))
        a3 = amap[:, :, None]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 120 + 40 * np.sin(xx / 90.0) + 30 * np.cos(yy / 70.0)
        bg = np.clip(np.stack([base, base * 0.95, base * 1.05], axis=-1), 0, 255)
        wm = (a3 * np.array(_ALPHA_LOGO_BGR, np.float32) + (1 - a3) * bg).clip(0, 255).astype(np.uint8)
        mark = amap > 0.15
        assert float(np.abs(wm.astype(np.float32)[mark] - bg[mark]).mean()) > 20  # mark clearly visible
        out = eng.remove_watermark_reverse_alpha(wm).astype(np.float32)
        assert float(np.abs(out[mark] - bg[mark]).mean()) < 10.0  # texture recovered, no outline


class TestDegenerateAndChannelInputs:
    """Removal must not crash on degenerate sizes or non-3-channel inputs."""

    @pytest.mark.parametrize(("w", "h"), [(2048, 1), (1, 2048), (2048, 8)])
    def test_wide_short_does_not_raise(self, w, h):
        eng = SamsungEngine()
        img = np.zeros((h, w, 3), np.uint8)
        out = eng.remove_watermark_reverse_alpha(img)
        assert out.shape == img.shape

    def test_grayscale_2d_does_not_raise(self):
        eng = SamsungEngine()
        gray = np.zeros((1448, 1086), np.uint8)
        out = eng.remove_watermark_reverse_alpha(gray)
        assert out.shape == (1448, 1086, 3)

    def test_bgra_4channel_does_not_raise(self):
        eng = SamsungEngine()
        bgra = np.zeros((1448, 1086, 4), np.uint8)
        out = eng.remove_watermark_reverse_alpha(bgra)
        assert out.shape == (1448, 1086, 3)
