"""Tests for the optional Real-ESRGAN upscaler (no model download).

The model-running path is exercised manually (it downloads ~67 MB of BSD-3-Clause
weights on first use); these tests cover the availability guard and the no-model
control flow, mirroring the repo convention for ML-adjacent modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks import upscaler


class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(upscaler.is_available(), bool)


class TestUpscaleGuard:
    def test_raises_without_extra(self, monkeypatch):
        monkeypatch.setattr(upscaler, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="esrgan"):
            upscaler.upscale(np.full((32, 32, 3), 128, dtype=np.uint8))


class TestModelCachePath:
    def test_cache_path_uses_model_filename(self):
        if not upscaler.is_available():
            pytest.skip("esrgan extra (torch) not installed")
        assert upscaler._model_cache_path().name == upscaler._MODEL_FILENAME
