"""Tests for AI metadata detection and removal."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from remove_ai_watermarks.metadata import (
    _is_ai_key,
    get_ai_metadata,
    has_ai_metadata,
    remove_ai_metadata,
)

# ── Key detection ───────────────────────────────────────────────────


class TestIsAiKey:
    """Tests for _is_ai_key helper."""

    def test_exact_match_lowercase(self):
        assert _is_ai_key("parameters")

    def test_exact_match_mixed_case(self):
        assert _is_ai_key("Parameters")

    def test_keyword_substring(self):
        assert _is_ai_key("stable_diffusion_model_v2")

    def test_c2pa_detected(self):
        assert _is_ai_key("c2pa_chunk")

    def test_standard_key_not_flagged(self):
        assert not _is_ai_key("Author")

    def test_innocuous_key_not_flagged(self):
        assert not _is_ai_key("Title")

    def test_dpi_not_flagged(self):
        assert not _is_ai_key("dpi")


# ── has_ai_metadata / get_ai_metadata ───────────────────────────────


class TestHasAiMetadata:
    """Tests for detecting AI metadata in images."""

    def test_detects_ai_metadata(self, tmp_png_with_ai_metadata):
        assert has_ai_metadata(tmp_png_with_ai_metadata)

    def test_clean_image_no_ai(self, tmp_clean_png):
        assert not has_ai_metadata(tmp_clean_png)

    def test_detects_c2pa_uuid_in_isobmff_container(self, tmp_path: Path):
        """C2PA in AVIF/HEIF/MP4 lives in a ``uuid`` box identified by a fixed UUID.

        Real AVIF/HEIF fixtures aren't shipped, so simulate the container by
        prepending an ISOBMFF-shaped ftyp box and the C2PA UUID bytes.
        """
        from remove_ai_watermarks.metadata import C2PA_UUID

        path = tmp_path / "fake.avif"
        # ftyp box: size(4) + 'ftyp' + 'avif' + minor_version(4) + 'avif'
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"jumb-payload"
        path.write_bytes(ftyp + uuid_box + b"\x00" * 64)
        assert has_ai_metadata(path)

    def test_strip_c2pa_boxes_removes_uuid_box(self, tmp_path: Path):
        """ISOBMFF strip should drop the C2PA uuid box and keep everything else."""
        from remove_ai_watermarks.metadata import C2PA_UUID
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        # uuid box: size(4) + 'uuid' + 16-byte UUID + minimal payload (8 bytes -> total 32)
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        cleaned, stripped = strip_c2pa_boxes(ftyp + uuid_box + mdat)
        assert stripped == 1
        assert cleaned == ftyp + mdat

    def test_strip_c2pa_boxes_passthrough_for_non_isobmff(self):
        """Non-ISOBMFF input must be returned unchanged."""
        from remove_ai_watermarks.noai.isobmff import strip_c2pa_boxes

        data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100
        cleaned, stripped = strip_c2pa_boxes(data)
        assert stripped == 0
        assert cleaned == data

    def test_remove_ai_metadata_strips_c2pa_in_avif(self, tmp_path: Path):
        """End-to-end: ``remove_ai_metadata`` on a fake .avif drops the C2PA box."""
        from remove_ai_watermarks.metadata import C2PA_UUID, remove_ai_metadata

        src = tmp_path / "in.avif"
        ftyp = b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00avifmif1"
        uuid_box = b"\x00\x00\x00\x20uuid" + C2PA_UUID + b"payload!"
        mdat = b"\x00\x00\x00\x10mdat" + b"pixeldat"
        src.write_bytes(ftyp + uuid_box + mdat)

        out = tmp_path / "out.avif"
        result = remove_ai_metadata(src, out)
        assert result == out
        assert out.read_bytes() == ftyp + mdat
        # And after stripping, detection must no longer flag the cleaned file.
        from remove_ai_watermarks.metadata import has_ai_metadata

        assert not has_ai_metadata(out)

    def test_detects_iptc_trained_algorithmic_media_marker(self, tmp_path: Path):
        """Some pipelines embed only the IPTC AI marker in XMP, no C2PA manifest."""
        path = tmp_path / "fake.jpg"
        # Minimal JPEG-ish bytes containing the IPTC AI marker in an XMP-like blob.
        xmp = (
            b"<x:xmpmeta><Iptc4xmpExt:DigitalSourceType>"
            b"trainedAlgorithmicMedia"
            b"</Iptc4xmpExt:DigitalSourceType></x:xmpmeta>"
        )
        path.write_bytes(b"\xff\xd8\xff\xe1" + xmp + b"\xff\xd9")
        assert has_ai_metadata(path)


class TestGetAiMetadata:
    """Tests for extracting AI metadata."""

    def test_extracts_parameters_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "parameters" in meta
        assert "Euler" in meta["parameters"]

    def test_extracts_prompt_key(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "prompt" in meta

    def test_does_not_extract_author(self, tmp_png_with_ai_metadata):
        meta = get_ai_metadata(tmp_png_with_ai_metadata)
        assert "Author" not in meta

    def test_clean_image_empty_dict(self, tmp_clean_png):
        meta = get_ai_metadata(tmp_clean_png)
        assert meta == {}


# ── remove_ai_metadata ──────────────────────────────────────────────


class TestRemoveAiMetadata:
    """Tests for stripping AI metadata."""

    def test_removes_ai_keys(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output)

        with Image.open(output) as img:
            assert "parameters" not in img.info
            assert "prompt" not in img.info

    def test_keeps_standard_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=True)

        with Image.open(output) as img:
            assert "Author" in img.info
            assert img.info["Author"] == "Test Author"

    def test_remove_all_metadata(self, tmp_png_with_ai_metadata):
        output = tmp_png_with_ai_metadata.parent / "cleaned.png"
        remove_ai_metadata(tmp_png_with_ai_metadata, output, keep_standard=False)
        with Image.open(output) as img:
            assert "Author" not in img.info
            assert "parameters" not in img.info

    def test_overwrite_in_place(self, tmp_path):
        """When output_path is None, should overwrite source."""
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test data")
        path = tmp_path / "inplace.png"
        img.save(path, pnginfo=pnginfo)

        result = remove_ai_metadata(path)
        assert result == path

        with Image.open(path) as cleaned:
            assert "parameters" not in cleaned.info

    def test_jpeg_output(self, tmp_path):
        """Test metadata removal for JPEG format."""
        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", "test")
        png_path = tmp_path / "source.png"
        img.save(png_path, pnginfo=pnginfo)

        jpg_path = tmp_path / "output.jpg"
        result = remove_ai_metadata(png_path, jpg_path)
        assert result == jpg_path
        assert jpg_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        img = Image.new("RGB", (32, 32))
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", "test")
        path = tmp_path / "source.png"
        img.save(path, pnginfo=pnginfo)

        output = tmp_path / "sub" / "dir" / "cleaned.png"
        remove_ai_metadata(path, output)
        assert output.exists()

    def test_returns_path(self, tmp_clean_png):
        output = tmp_clean_png.parent / "out.png"
        result = remove_ai_metadata(tmp_clean_png, output)
        assert isinstance(result, Path)
        assert result == output
