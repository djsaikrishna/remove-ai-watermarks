# Remove-AI-Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI watermarks from images.

## How to run

- `uv run remove-ai-watermarks all <image.png> -o <output.png>`
- `uv run remove-ai-watermarks metadata <image.png> --check` — inspect AI metadata (C2PA, EXIF, PNG chunks)
- `uv run remove-ai-watermarks metadata <image.png> --remove -o <out.png>` — strip all AI metadata

## Configuration

- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)

## Key modules

- `noai/c2pa.py` — PNG chunk parser; use `extract_c2pa_chunk(path)` to get raw caBX payload, `has_c2pa_metadata(path)` to detect. Do not reimplement chunk parsing.
- `noai/constants.py` — PNG_SIGNATURE, C2PA_CHUNK_TYPE, C2PA_SIGNATURES constants
- `face_protector.py` — YOLO detect + soft-blend pattern; mirror this for any "protect region during diffusion" features

## Known limitations

- `invisible` pipeline downscales to model-native resolution (1024 px for SDXL) before diffusion. Degrades fine text in infographics. Tracked; fix is tile-based diffusion.
- Pyright first run is slow (2-3 min) due to ML deps (torch/diffusers/transformers stubs)
- `ultralytics` monkey-patches `PIL.Image.open` and tries to autoload `pi_heif`. When `pi_heif` is missing, opening files raises `ModuleNotFoundError`, not `UnidentifiedImageError`. Code that opens user-supplied or unknown-format files should `except Exception`, not just `OSError`/`UnidentifiedImageError`.
- Metadata detection for AVIF/HEIF/JPEG-XL relies on a binary scan for `C2PA_UUID` + `IPTC_AI_MARKERS`. C2PA removal in those containers is implemented via `noai/isobmff.py` (top-level ``uuid`` / ``jumb`` box stripper, no re-encoding). EXIF/XMP boxes inside those containers are not yet scrubbed.
- **SynthID v2 vs default pipeline:** the SDXL-based default profile (since May 2026) defeats SynthID v2 per the deployment in raiw-app (`fal-ai/fast-sdxl` at native ~1024 px, strength 0.05, steps 50). SD-1.5 dreamshaper at 768 px was previously the default and does NOT defeat v2 — verified empirically against Gemini app's "Verify with SynthID" feature (strength 0.04, 0.10, and elastic warp α∈{5,8} all flagged positive). That SD-1.5 path was removed; only `default` (SDXL) and `ctrlregen` profiles remain.
