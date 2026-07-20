"""Render SYNTHETIC detection silhouettes for the CJK vendor text marks (data-safe).

Adding a mark needs only a DETECTION silhouette, and it must be font-rendered rather
than derived from user uploads: the corpus is real user content and may never reach a
tracked asset (see the repo CLAUDE.md data-safety rule). Seeing real samples to learn
the glyphs, weight and layout is fine; the committed template stays synthetic.

Covered here:
  qwen    "千问AI生成"  -- Alibaba Tongyi Qianwen, bottom-right, 3-lobed logo + text
  xinghui "星绘AI生成"  -- ByteDance 星绘, bottom-right, 4-point sparkle + text

The leading LOGO is deliberately NOT rendered. It is the part that varies most between
releases and is hardest to reproduce synthetically, while the CJK run is stable and is
what actually discriminates one vendor from another (the shared `AI生成` tail is exactly
what does NOT discriminate -- see the rival-margin mechanism in _text_mark_engine).

Regenerate with:  uv run python scripts/render_vendor_silhouettes.py

STATUS 2026-07-18: these two marks are NOT registered, and this script is kept as the
method + the record of why. Measured on 14 hand-verified 千问 positives from the corpus,
the current detect architecture (top-hat glyph blob -> binary TM_CCOEFF_NORMED) cannot
see this mark AT ALL:

    same pipeline, each mark scored with its OWN template, on real positives
      doubao   n=40   mean NCC 0.723   median 0.835   >= 0.40 gate: 82%
      qwen     n=14   mean NCC 0.170   median 0.179   >= 0.40 gate:  0%

Three checks ruled out the obvious explanations, in order:
  1. NOT the synthetic render. A template cut from an ACTUAL Qwen mark scores the same
     as the font-rendered one (real-vs-real 0.307 vs synthetic 0.308) -- and real masks
     do not even match EACH OTHER.
  2. NOT the morphology kernel size. Scaling MORPH_OPEN/CLOSE with the box height (they
     are fixed 5px, ~9% of a 57px-tall box) gained only +0.014 mean and moved nothing
     across the gate.
  3. NOT the appearance thresholds. Sweeping tophat_delta / logo_min_luma / kernel
     reached at best mean 0.35 with 4/14 over the gate.

The blocker is SEGMENTATION on a faint mark: Doubao is stamped bold and opaque, so the
white top-hat returns a clean glyph blob; the Qwen mark is a thin translucent overlay
that shatters into specks, and no template can match a blob that is not there. Adding
it therefore needs a detection front-end that does not depend on binarizing the glyph
(grayscale/edge correlation on the raw top-hat, or a learned patch classifier) -- not a
new silhouette. Shipping it on the current front-end would mean a detector that finds
almost nothing and, at any threshold low enough to fire, fires on arbitrary corner text.

星绘 additionally has only ONE confirmed example in the corpus, so even a working
front-end could not have its threshold calibrated yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ASSETS = Path(__file__).resolve().parents[1] / "src" / "remove_ai_watermarks" / "assets"
# STHeiti Medium approximates the semibold CJK sans these marks are set in; the exact
# family is unpublished for every vendor (GB 45438-2025 only requires a legible face).
_FONT = "/System/Library/Fonts/STHeiti Medium.ttc"

MARKS = {
    "qwen_alpha.png": "千问AI生成",
    "xinghui_alpha.png": "星绘AI生成",
}


def render(text: str, width: int = 335) -> np.ndarray:
    """Binary glyph silhouette (255 = glyph), sized to the doubao asset's convention.

    Matching doubao's 335px asset width keeps the `alpha_*_frac` numbers transferable,
    since these marks are the same house style and scale.
    """
    probe = Image.new("L", (10, 10))
    d0 = ImageDraw.Draw(probe)
    size = 8
    while size < 200:  # grow until the run fills the target width
        f = ImageFont.truetype(_FONT, size)
        if d0.textbbox((0, 0), text, font=f)[2] >= width * 0.98:
            break
        size += 1
    font = ImageFont.truetype(_FONT, size)
    bb = d0.textbbox((0, 0), text, font=font)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pad = max(2, int(h * 0.12))
    im = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), text, font=font, fill=255)
    return np.array(im)


def main() -> None:
    try:
        for name, text in MARKS.items():
            sil = render(text)
            Image.fromarray(sil).save(_ASSETS / name)
            print(f"wrote {_ASSETS / name}  ({sil.shape[1]}x{sil.shape[0]})  text={text!r}")
    except OSError as e:
        print(f"Font not found ({e}); install a CJK font or edit _FONT.", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
