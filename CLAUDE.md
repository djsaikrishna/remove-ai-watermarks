# Remove-AI-Watermarks

You are a **principal Python engineer** maintaining a CLI tool and library for removing visible and invisible AI watermarks from images.

## How to run

- `uv run remove-ai-watermarks all <image.png> -o <output.png>`
- `uv run remove-ai-watermarks metadata <image.png> --check` — inspect AI metadata (C2PA, EXIF, PNG chunks)
- `uv run remove-ai-watermarks metadata <image.png> --remove -o <out.png>` — strip all AI metadata

## Test and lint

- `bash maintain.sh` — uv-outdated, uv-secure, ruff check/fix, ruff format, pyright, pytest -n auto

## Configuration

- GPU/ML modules (invisible_engine, ctrlregen, watermark_remover) are optional — guard imports with `is_available()` checks
- Tests for ML modules are limited to availability checks (require multi-GB downloads)

## Key modules

- `noai/c2pa.py` — PNG chunk parser; use `extract_c2pa_chunk(path)` to get raw caBX payload, `has_c2pa_metadata(path)` to detect. Do not reimplement chunk parsing. `extract_c2pa_info(path)` sets `synthid_watermark`/`synthid_vendors` when the manifest is signed by a SynthID-using vendor.
- `noai/constants.py` — PNG_SIGNATURE, C2PA_CHUNK_TYPE, C2PA_SIGNATURES, C2PA_ISSUERS, and `SYNTHID_C2PA_ISSUERS` (issuers that pair SynthID with C2PA: Google, OpenAI). Add a new issuer here, not inline.
- `metadata.py` — `synthid_source(path)` returns the vendor name(s) if the C2PA manifest implies a SynthID pixel watermark, else None; `get_ai_metadata` surfaces the verdict, and `metadata --check` prints it as a callout.
- `face_protector.py` — YOLO detect + soft-blend pattern; mirror this for any "protect region during diffusion" features

## Known limitations

- `invisible` pipeline downscales to model-native resolution (1024 px for SDXL) before diffusion. Degrades fine text in infographics. Tracked; fix is tile-based diffusion.
- Pyright first run is slow (2-3 min) due to ML deps (torch/diffusers/transformers stubs)
- `ultralytics` monkey-patches `PIL.Image.open` and tries to autoload `pi_heif`. When `pi_heif` is missing, opening files raises `ModuleNotFoundError`, not `UnidentifiedImageError`. Code that opens user-supplied or unknown-format files should `except Exception`, not just `OSError`/`UnidentifiedImageError`.
- Metadata detection for AVIF/HEIF/JPEG-XL relies on a binary scan for `C2PA_UUID` + `IPTC_AI_MARKERS`. C2PA removal in those containers is implemented via `noai/isobmff.py` (top-level ``uuid`` / ``jumb`` box stripper, no re-encoding). EXIF/XMP boxes inside those containers are not yet scrubbed.
- **SynthID detection is metadata-only.** There is no reliable *local* detector of the SynthID *pixel* watermark — Google's decoder is proprietary, no public spec or API (only a waitlisted portal). We detect SynthID by its C2PA companion (`synthid_source` / `SYNTHID_C2PA_ISSUERS`), which is reliable while the manifest is intact but says nothing once C2PA is stripped. Google→SynthID is long-standing; OpenAI→SynthID is confirmed by OpenAI's Help Center (ChatGPT/Codex/API "include both C2PA metadata and SynthID watermarks", updated 2026-05-21) but time-gated (pre-rollout OpenAI images carry C2PA without SynthID), so the OpenAI verdict is hedged "likely". Oracles: Gemini app "Verify with SynthID" (Google), openai.com/verify (OpenAI). The spectral phase-coherence approach from `github.com/aloshdenny/reverse-SynthID` was evaluated (May 2026) and **does not work for real-content detection**: on its own shipped codebook + validation set, watermarked and cleaned images were indistinguishable (conf within noise, cleaned often higher); it only fires on pure-black 1024x1024 reference images at exact resolution (the controlled case it was calibrated on). The README's "90% / conf=0.91" reproduces only in that lab condition. Do not build a production detector on it; if revisited, it is experimental/diagnostic only and needs a per-resolution, per-model reference corpus.
- **SynthID v2 vs default pipeline:** the SDXL-based default profile (since May 2026) defeats SynthID v2. **Verified end-to-end (May 2026):** local SDXL run on a Gemini 3 Pro output, checked via the Gemini app's "Verify with SynthID" feature, returned "no SynthID watermark detected". The same configuration is used in raiw-app production (`fal-ai/fast-sdxl` at native ~1024 px, strength 0.05, steps 50). SD-1.5 dreamshaper at 768 px was previously the default and does NOT defeat v2 — verified empirically against the same feature (strength 0.04, 0.10, and elastic warp α∈{5,8} all flagged positive). That SD-1.5 path was removed; only `default` (SDXL) and `ctrlregen` profiles remain.
