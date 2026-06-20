# qwen_in — pipeline-fidelity eval set

A small, **stable** set of AI-generated images used to compare the diffusion
removal pipelines (`controlnet` / `sdxl` / `qwen`) for fidelity with
`scripts/fidelity_metrics.py`. Fixing the set in the repo keeps comparisons
reproducible across runs and pipelines.

All four are AI-generated test content (they carry SynthID + C2PA from their
generator — verify with `remove-ai-watermarks identify`), same class as the
`data/samples/` fixtures. No real-person photos.

| file | vendor (SynthID) | content | exercises |
|---|---|---|---|
| `openai_1_original.png` | OpenAI | typography sheet (EN + RU + ZH) | text (multi-script) |
| `openai_2_original.png` | OpenAI | Raiw.cc poster | text (EN, small) |
| `gemini_1_original.png` | Google | landscape + Chinese sign | text (CJK) |
| `gemini_3_original.png` | Google | 3x3 portrait grid | faces (identity / skin texture) |

## Text ground truth

`ground_truth.json` (`{basename: text}`) is the **hand-verified** OCR of the
text-bearing originals, seeded by `fidelity_metrics.py ocr` and corrected by
hand (PaddleOCR mis-reads stylized Cyrillic in particular). It is the reference
for the text CER metric — much cleaner than OCR-vs-OCR. Regenerate the seed with:

    uv run scripts/fidelity_metrics.py ocr data/qwen_in/openai_1_original.png \
        data/qwen_in/openai_2_original.png data/qwen_in/gemini_1_original.png \
        --langs en,ru,ch --out data/qwen_in/ground_truth.json
    # then re-verify by hand before trusting it.

## Compare

    uv run scripts/fidelity_metrics.py compare \
        --original data/qwen_in/gemini_3_original.png \
        --variant controlnet=<out>.png --variant qwen=<out>.png --ocr-langs ""
