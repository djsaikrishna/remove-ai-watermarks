"""Tests for the --auto pipeline planner (content-adaptive mode selection).

Detection runs on synthetic images; the face-present routing is exercised by
monkeypatching ``detect_face`` (a real detectable face fixture is private, never
committed). The planner is cv2-only and torch-free.
"""

from __future__ import annotations

import cv2
import numpy as np

from remove_ai_watermarks import auto_config, image_io


def _write(img, tmp_path, name="x.png"):
    p = tmp_path / name
    image_io.imwrite(p, img)
    return p


class TestDetectors:
    def test_detect_face_false_on_flat(self):
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert auto_config.detect_face(flat) is False

    def test_edge_density_flat_near_zero(self):
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        assert auto_config.edge_density(flat) < 0.001

    def test_edge_density_text_higher_than_blank(self):
        blank = np.full((200, 400, 3), 255, dtype=np.uint8)
        text = blank.copy()
        cv2.putText(text, "HELLO AI TEXT", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        assert auto_config.edge_density(text) > auto_config.edge_density(blank)

    def test_dbnet_detects_text_card(self):
        """The bundled PP-OCRv3 DBNet model fires on a clear text card and not on flat."""
        card = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.putText(card, "INVOICE TOTAL 1234", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
        assert auto_config._detect_text_dbnet(card) is True
        assert auto_config._detect_text_dbnet(np.full((300, 500, 3), 128, dtype=np.uint8)) is False

    def test_detect_text_falls_back_to_mser_when_dbnet_unavailable(self, monkeypatch):
        """If DBNet can't load (returns None), detect_text uses the MSER heuristic."""
        monkeypatch.setattr(auto_config, "_detect_text_dbnet", lambda _img: None)
        called = {}

        def _fake_mser(_img):
            called["mser"] = True
            return True

        monkeypatch.setattr(auto_config, "_detect_text_mser", _fake_mser)
        assert auto_config.detect_text(np.full((100, 100, 3), 128, dtype=np.uint8)) is True
        assert called.get("mser") is True


class TestPlan:
    def test_unreadable_returns_none(self, tmp_path):
        assert auto_config.plan(tmp_path / "does_not_exist.png") is None

    def test_flat_image_is_default_pipeline_no_polish(self, tmp_path):
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        cfg = auto_config.plan(_write(flat, tmp_path))
        assert cfg is not None
        assert cfg.pipeline == "default"  # structure-less -> plain SDXL
        assert cfg.restore_faces is False
        assert cfg.adaptive_polish is False  # no smoothing pass -> no polish
        assert cfg.unsharp == 0.0
        assert cfg.humanize == 0.0
        assert cfg.min_resolution == 1024

    def test_text_image_uses_controlnet(self, tmp_path):
        img = np.full((300, 500, 3), 255, dtype=np.uint8)
        cv2.putText(img, "INVOICE TOTAL 1234", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
        cfg = auto_config.plan(_write(img, tmp_path))
        assert cfg is not None
        # Text creates edges above the structure-less floor -> controlnet preserves them.
        assert cfg.pipeline == "controlnet"

    def test_face_routes_to_restore_and_controlnet_and_polish(self, tmp_path, monkeypatch):
        monkeypatch.setattr(auto_config, "detect_face", lambda _img: True)
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        cfg = auto_config.plan(_write(flat, tmp_path))
        assert cfg is not None
        assert cfg.has_face
        assert cfg.restore_faces
        assert cfg.pipeline == "controlnet"
        assert cfg.adaptive_polish  # smoothing pass ran -> adaptive polish on
        assert cfg.unsharp == 0.0  # fixed knobs off; the adaptive polish replaces them
        assert cfg.humanize == 0.0

    def test_text_signal_forces_controlnet_on_flat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(auto_config, "detect_text", lambda _img: True)
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        cfg = auto_config.plan(_write(flat, tmp_path))
        assert cfg is not None
        assert cfg.has_text
        assert cfg.pipeline == "controlnet"


class TestReason:
    def test_reason_summarizes_plan(self):
        cfg = auto_config.AutoConfig(
            pipeline="controlnet",
            restore_faces=True,
            adaptive_polish=True,
            unsharp=0.0,
            humanize=0.0,
            min_resolution=1024,
            has_face=True,
            has_text=False,
            edge_density=0.05,
            width=800,
            height=600,
        )
        r = cfg.reason
        assert "controlnet" in r
        assert "face" in r
        assert "face-restore on" in r
        assert "adaptive polish" in r
