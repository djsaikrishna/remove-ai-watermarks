"""Tests for the known-visible-watermark registry (localize -> fill)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from remove_ai_watermarks import watermark_registry as reg

DOUBAO_SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


class TestCatalog:
    def test_keys(self):
        assert reg.mark_keys() == ["gemini", "doubao", "jimeng", "samsung", "jimeng_pill"]

    def test_all_in_auto(self):
        assert all(m.in_auto for m in reg.known_marks())

    def test_marks_expose_detect_and_mask(self):
        # Every mark drives the uniform localize -> fill contract: a detect callable
        # (verdict + bbox, no mask) and a mask callable (full-frame footprint).
        for m in reg.known_marks():
            assert callable(m._detect)
            assert callable(m._mask)

    def test_locations(self):
        by_key = {m.key: m for m in reg.known_marks()}
        assert by_key["gemini"].location == "bottom-right"
        assert by_key["doubao"].location == "bottom-right"
        assert by_key["jimeng"].location == "bottom-right"
        assert by_key["samsung"].location == "bottom-left"
        assert by_key["jimeng_pill"].location == "top-left"

    def test_get_mark_unknown_raises(self):
        with pytest.raises(KeyError):
            reg.get_mark("nope")


class TestScan:
    def test_detect_marks_scans_all(self):
        img = np.zeros((256, 256, 3), np.uint8)
        keys = {d.key for d in reg.detect_marks(img)}
        assert keys == {"gemini", "doubao", "jimeng", "samsung", "jimeng_pill"}

    def test_blank_image_no_auto_mark(self):
        dets = reg.detect_marks(np.zeros((256, 256, 3), np.uint8), include_explicit=False)
        assert not any(d.detected for d in dets)


class TestBackendResolution:
    def test_auto_resolves_to_installed_backend(self):
        assert reg.resolve_backend("auto") in ("cv2", "migan")

    def test_explicit_backend_passes_through(self):
        assert reg.resolve_backend("cv2") == "cv2"
        assert reg.resolve_backend("lama") == "lama"


class TestFill:
    def test_fill_erases_masked_region(self):
        # A bright square on a flat field, masked, is inpainted away (cv2 backend).
        img = np.full((128, 128, 3), 60, np.uint8)
        img[40:70, 40:70] = 240
        mask = np.zeros((128, 128), np.uint8)
        mask[36:74, 36:74] = 255
        out = reg.fill(img, mask, backend="cv2")
        assert out.shape == img.shape
        # the masked bright square is pulled toward the surrounding field
        assert int(out[55, 55].mean()) < 160

    def test_fill_empty_mask_is_noop(self):
        img = np.full((64, 64, 3), 100, np.uint8)
        out = reg.fill(img, np.zeros((64, 64), np.uint8), backend="cv2")
        assert np.array_equal(out, img)


class TestProvenanceGate:
    """The Gemini trust gate relaxes from 0.5 to 0.35 when provenance confirms Google;
    tested deterministically by stubbing the engine's raw detection confidence."""

    def _stub(self, monkeypatch: pytest.MonkeyPatch, conf: float) -> None:
        from remove_ai_watermarks.gemini_engine import DetectionResult

        def fake_detect(image, force_size=None, *, trust_provenance=False):
            return DetectionResult(detected=conf >= 0.35, confidence=conf, region=(10, 10, 48, 48))

        monkeypatch.setattr(reg._engine("gemini"), "detect_watermark", fake_detect)

    def test_midband_conf_needs_provenance(self, monkeypatch: pytest.MonkeyPatch):
        # conf 0.42 sits in [0.35, 0.5): demoted without provenance, trusted with it.
        self._stub(monkeypatch, 0.42)
        img = np.zeros((256, 256, 3), np.uint8)
        assert reg.get_mark("gemini").detect(img).detected is False
        assert reg.get_mark("gemini").detect(img, provenance=True).detected is True

    def test_high_conf_detected_either_way(self, monkeypatch: pytest.MonkeyPatch):
        self._stub(monkeypatch, 0.72)
        img = np.zeros((256, 256, 3), np.uint8)
        assert reg.get_mark("gemini").detect(img).detected is True
        assert reg.get_mark("gemini").detect(img, provenance=True).detected is True


@pytest.mark.skipif(not DOUBAO_SAMPLE.exists(), reason="doubao sample not present")
class TestRealSample:
    def test_doubao_sample_detected(self):
        from remove_ai_watermarks.image_io import imread

        fired = [d.key for d in reg.detect_marks(imread(DOUBAO_SAMPLE), include_explicit=False) if d.detected]
        assert "doubao" in fired

    def test_doubao_remove_returns_region(self):
        from remove_ai_watermarks.image_io import imread

        img = imread(DOUBAO_SAMPLE)
        result, region = reg.get_mark("doubao").remove(img, backend="cv2")
        assert region is not None
        assert result.shape == img.shape


class TestLocalizeFill:
    def test_clean_corner_is_untouched(self):
        # No glyph in the corner -> no mask -> remove is a no-op copy.
        img = np.zeros((512, 512, 3), np.uint8)
        result, region = reg.get_mark("doubao").remove(img, backend="cv2")
        assert region is None
        assert np.array_equal(result, img)


class TestSensitivity:
    """``resolve_relax`` turns the sensitivity policy + evidence into the per-mark
    relaxation boolean the engines consume."""

    def test_strict_never_relaxes(self):
        # even with metadata provenance, strict keeps the conservative gate
        assert (
            reg.resolve_relax("gemini", sensitivity="strict", provenance=frozenset({"gemini"}), strict_keys=set())
            is False
        )

    def test_assume_ai_always_relaxes(self):
        assert reg.resolve_relax("gemini", sensitivity="assume_ai", provenance=frozenset(), strict_keys=set()) is True

    def test_auto_relaxes_on_own_metadata(self):
        assert (
            reg.resolve_relax("gemini", sensitivity="auto", provenance=frozenset({"gemini"}), strict_keys=set()) is True
        )

    def test_auto_strict_without_evidence(self):
        assert reg.resolve_relax("gemini", sensitivity="auto", provenance=frozenset(), strict_keys=set()) is False

    def test_auto_cross_mark_same_product(self):
        # a detected Jimeng wordmark relaxes the Jimeng pill (same product, other corner)
        assert (
            reg.resolve_relax("jimeng_pill", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"}) is True
        )

    def test_auto_no_cross_mark_across_products(self):
        # a detected Jimeng wordmark must NOT relax Doubao (distinct products, same corner)
        assert reg.resolve_relax("doubao", sensitivity="auto", provenance=frozenset(), strict_keys={"jimeng"}) is False

    def test_remove_auto_marks_accepts_all_sensitivities(self):
        blank = np.zeros((256, 256, 3), np.uint8)
        for s in ("auto", "strict", "assume_ai"):
            _, removed = reg.remove_auto_marks(blank, sensitivity=s, backend="cv2")
            assert removed == []


class TestArbiter:
    """``decide`` is the PURE removal arbiter: (candidates, context) -> ordered
    winners, no image / no I/O. Tested in isolation by handing it fabricated
    Candidates -- this is the payoff of separating decision from perception."""

    @staticmethod
    def _c(key, *, strict=False, relaxed=False, flat=False):
        feats = {"footprint_flat": 1.0} if flat else {}
        return reg.Candidate(key, f"L:{key}", strict, relaxed, feats)

    def _keys(self, cands, ctx):
        return {d.candidate.key for d in reg.decide(cands, ctx)}

    def test_empty(self):
        assert reg.decide([], reg.Context()) == []

    def test_strict_uses_strict_verdict(self):
        # relaxed-only detection must NOT fire under strict
        assert self._keys([self._c("gemini", relaxed=True)], reg.Context(sensitivity="strict")) == set()

    def test_assume_ai_uses_relaxed(self):
        fired = reg.decide([self._c("gemini", relaxed=True)], reg.Context(sensitivity="assume_ai"))
        assert [d.candidate.key for d in fired] == ["gemini"]
        assert fired[0].relax is True

    def test_auto_relaxes_on_provenance(self):
        c = [self._c("gemini", relaxed=True)]
        assert self._keys(c, reg.Context(provenance=frozenset({"gemini"}))) == {"gemini"}
        assert self._keys(c, reg.Context()) == set()  # no evidence -> strict verdict (not fired)

    def test_cross_mark_relaxes_pill_via_jimeng(self):
        cands = [self._c("jimeng", strict=True, relaxed=True), self._c("jimeng_pill", relaxed=True, flat=True)]
        assert self._keys(cands, reg.Context()) == {"jimeng", "jimeng_pill"}

    def test_pill_dropped_on_doubao(self):
        cands = [
            self._c("doubao", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=True),
        ]
        keys = self._keys(cands, reg.Context(provenance=frozenset({"jimeng"})))
        assert "doubao" in keys
        assert "jimeng_pill" not in keys

    def test_pill_metadata_arm_gated_on_flatness(self):
        ctx = reg.Context(provenance=frozenset({"jimeng"}))
        assert self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=True)], ctx) == {"jimeng_pill"}
        assert self._keys([self._c("jimeng_pill", strict=True, relaxed=True, flat=False)], ctx) == set()

    def test_pill_wordmark_arm_ignores_flatness(self):
        # wordmark present -> pill removed even on a textured (non-flat) footprint
        cands = [
            self._c("jimeng", strict=True, relaxed=True),
            self._c("jimeng_pill", strict=True, relaxed=True, flat=False),
        ]
        assert "jimeng_pill" in self._keys(cands, reg.Context())


class TestProvenanceMaskThreading:
    """Regression for the provenance-relaxed Gemini no-op (#1) and the false 'removed'
    label (#2). Before the fix, footprint_mask re-detected WITHOUT trust_provenance, the
    FP gate demoted the sparkle to detected=False, the mask came back None, yet
    remove_auto_marks still reported the mark as removed."""

    def test_relaxed_sparkle_yields_mask(self, monkeypatch: pytest.MonkeyPatch):
        # A sparkle a strict re-detect would demote (detected False) but a
        # provenance-relaxed detect accepts must still produce a removal mask.
        from remove_ai_watermarks.gemini_engine import DetectionResult

        def fake(image, force_size=None, *, trust_provenance=False):
            return DetectionResult(
                detected=trust_provenance, confidence=0.42 if trust_provenance else 0.30, region=(400, 400, 60)
            )

        monkeypatch.setattr(reg._engine("gemini"), "detect_watermark", fake)
        img = np.full((512, 512, 3), 90, np.uint8)
        assert reg.get_mark("gemini").localize(img, provenance=True).mask is not None
        assert reg.get_mark("gemini").localize(img, provenance=False).mask is None

    def test_no_label_when_mask_none(self, monkeypatch: pytest.MonkeyPatch):
        # A decided mark whose mask comes back None must NOT be reported as removed.
        from remove_ai_watermarks.gemini_engine import DetectionResult

        eng = reg._engine("gemini")
        monkeypatch.setattr(
            eng,
            "detect_watermark",
            lambda image, force_size=None, *, trust_provenance=False: DetectionResult(True, 0.9, (10, 10, 40)),
        )
        monkeypatch.setattr(eng, "footprint_mask", lambda image, *, force=False, region=None, dilate=None: None)
        _, removed = reg.remove_auto_marks(np.zeros((256, 256, 3), np.uint8), sensitivity="strict", backend="cv2")
        assert "Google Gemini sparkle" not in removed
