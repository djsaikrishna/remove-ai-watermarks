# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "click",
#   "numpy",
#   "opencv-python-headless",
#   "pillow",
#   "scikit-image",
#   "rapidfuzz",
#   "torch",
#   "lpips",
#   "easyocr",
#   "insightface",
#   "onnxruntime",
# ]
# ///
"""Objective fidelity metrics for comparing watermark-removal outputs.

Given an ORIGINAL (the reference) and one or more cleaned VARIANTS that have all
ALREADY passed the scrub oracle, this scores how much real detail each variant
preserved -- so "closer to the original" is the right axis here (between two
equally-scrubbed outputs, the one that deviates less from the original wins).

It is a standalone eval tool, NOT part of the package: PEP 723 inline deps let
``uv run`` build a throwaway env so the heavy models (EasyOCR, insightface,
LPIPS) never touch uv.lock or the shipped library. Metrics self-gate: face
metrics run only where faces are detected, text metrics only where text is.

Four metric groups (all reference = original):
  1. Text  -- EasyOCR character error rate (CER) of each variant vs the original's
              OCR string. Lower = text better preserved. OCR is noisy, so treat it
              as a RELATIVE comparison (every variant scored against the same ref).
  2. Face identity -- insightface (buffalo_l) ArcFace cosine similarity, original
              face vs the geometrically-matched variant face. Higher = identity kept.
  3. Face texture  -- LPIPS + Laplacian-variance ratio (variant/original) on face
              crops. Catches "plastication" (lost high-frequency skin detail):
              lapvar ratio < 1 = smoother than the original.
  4. Whole image   -- LPIPS / SSIM / PSNR vs the original (context: background too).

Usage:
    uv run scripts/fidelity_metrics.py --original O.png \
        --variant controlnet=C.png --variant qwen=Q.png --ocr-langs en,ru,ch_sim
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import cv2
import numpy as np
from _plain_console import Console, Table

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)
console = Console()


# ── helpers ──────────────────────────────────────────────────────────


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise click.ClickException(f"cannot read image: {path}")
    return img


def _match_size(variant: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Resize a variant to the reference size (outputs differ by a grid-round)."""
    if variant.shape[:2] != ref.shape[:2]:
        variant = cv2.resize(variant, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return variant


# ── text: OCR CER ────────────────────────────────────────────────────

# EasyOCR rejects some language combos in one Reader, so group into compatible
# readers and union the detections. Cyrillic and Chinese cannot share a reader.
_OCR_GROUPS = {
    "en": ["en"],
    "ru": ["ru", "en"],
    "ch_sim": ["ch_sim", "en"],
}


def _ocr_string(readers: list, bgr: np.ndarray) -> str:
    """Union all readers' detections into one position-sorted, whitespace-free string."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    dets: list[tuple[float, float, str]] = []
    for reader in readers:
        for box, text, conf in reader.readtext(rgb):
            if conf < 0.3 or not text.strip():
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            dets.append((min(ys), min(xs), text.strip()))
    # Sort top-to-bottom, then left-to-right (coarse reading order).
    dets.sort(key=lambda d: (round(d[0] / 20.0), d[1]))
    return "".join(t for _, _, t in dets).replace(" ", "")


def _build_ocr_readers(langs: list[str]) -> list:
    import easyocr

    seen: set[tuple[str, ...]] = set()
    readers = []
    for lang in langs:
        group = tuple(_OCR_GROUPS.get(lang, [lang]))
        if group in seen:
            continue
        seen.add(group)
        readers.append(easyocr.Reader(list(group), gpu=False, verbose=False))
    return readers


# ── face: detection + ArcFace + texture ──────────────────────────────


@dataclass
class FaceStats:
    n_faces: int = 0
    identity: list[float] = field(default_factory=list)
    lpips: list[float] = field(default_factory=list)
    lapvar_ratio: list[float] = field(default_factory=list)


def _lap_var(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _match_face(orig_face: Any, variant_faces: list[Any]) -> Any:
    """Nearest variant face to an original face by bbox-center distance (geometry kept)."""
    ox, oy = (orig_face.bbox[0] + orig_face.bbox[2]) / 2, (orig_face.bbox[1] + orig_face.bbox[3]) / 2
    best, best_d = None, 1e18
    for vf in variant_faces:
        vx, vy = (vf.bbox[0] + vf.bbox[2]) / 2, (vf.bbox[1] + vf.bbox[3]) / 2
        d = (ox - vx) ** 2 + (oy - vy) ** 2
        if d < best_d:
            best, best_d = vf, d
    return best


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _crop(bgr: np.ndarray, bbox: Any) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3])))
    return bgr[y1:y2, x1:x2]


# ── whole image: LPIPS / SSIM / PSNR ─────────────────────────────────


def _lpips_model() -> tuple[Any, Any]:
    import lpips
    import torch

    model = lpips.LPIPS(net="alex", verbose=False)
    model.eval()
    return model, torch


def _lpips_distance(model_torch: tuple[Any, Any], a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    model, torch = model_torch

    def _t(img: np.ndarray) -> Any:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        return float(model(_t(a_bgr), _t(b_bgr)).item())


def _ssim_psnr(a_bgr: np.ndarray, b_bgr: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    ssim = float(structural_similarity(a, b))
    psnr = float(peak_signal_noise_ratio(a, b))
    return ssim, psnr


# ── main ─────────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _fmt(v: float | None, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


@click.command()
@click.option("--original", required=True, type=click.Path(exists=True), help="Reference (unprocessed) image.")
@click.option(
    "--variant",
    "variants",
    multiple=True,
    required=True,
    help="LABEL=PATH of a cleaned output (repeatable).",
)
@click.option("--ocr-langs", default="en", help="Comma list of EasyOCR langs (en,ru,ch_sim). Empty = skip text.")
@click.option("--no-faces", is_flag=True, help="Skip face metrics.")
def main(original: str, variants: tuple[str, ...], ocr_langs: str, no_faces: bool) -> None:
    """Score each VARIANT against ORIGINAL across the four fidelity groups."""
    ref = _load_bgr(original)
    parsed: list[tuple[str, np.ndarray]] = []
    for spec in variants:
        if "=" not in spec:
            raise click.ClickException(f"--variant must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        parsed.append((label, _match_size(_load_bgr(path), ref)))

    langs = [x.strip() for x in ocr_langs.split(",") if x.strip()]
    lp = _lpips_model()  # AlexNet LPIPS, loaded once and reused for face crops + whole image

    # ── text ──
    ocr_cer: dict[str, float | None] = {label: None for label, _ in parsed}
    if langs:
        console.print(f"  OCR ({','.join(langs)})...")
        from rapidfuzz.distance import Levenshtein

        readers = _build_ocr_readers(langs)
        ref_text = _ocr_string(readers, ref)
        if ref_text:
            for label, img in parsed:
                hyp = _ocr_string(readers, img)
                ocr_cer[label] = Levenshtein.normalized_distance(ref_text, hyp)
        else:
            console.print("  (no text detected in the original; skipping text metric)")

    # ── faces ──
    face_stats: dict[str, FaceStats] = {label: FaceStats() for label, _ in parsed}
    if not no_faces:
        console.print("  Faces (insightface buffalo_l)...")
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        ref_faces = app.get(ref)
        if ref_faces:
            for label, img in parsed:
                vfaces = app.get(img)
                st = face_stats[label]
                for of in ref_faces:
                    vf = _match_face(of, vfaces)
                    if vf is None:
                        continue
                    st.n_faces += 1
                    st.identity.append(_cosine(of.normed_embedding, vf.normed_embedding))
                    oc, vc = _crop(ref, of.bbox), _crop(img, of.bbox)
                    if oc.size == 0 or vc.size == 0:
                        continue
                    vc_r = cv2.resize(vc, (oc.shape[1], oc.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                    st.lpips.append(_lpips_distance(lp, oc, vc_r))
                    ov = _lap_var(oc)
                    st.lapvar_ratio.append(_lap_var(vc_r) / ov if ov > 1e-6 else 0.0)
        else:
            console.print("  (no faces detected in the original; skipping face metrics)")

    # ── whole image ──
    console.print("  Whole-image LPIPS/SSIM/PSNR...")
    whole: dict[str, tuple[float, float, float]] = {}
    for label, img in parsed:
        ssim, psnr = _ssim_psnr(ref, img)
        whole[label] = (_lpips_distance(lp, ref, img), ssim, psnr)

    # ── report ──
    table = Table(title=f"Fidelity vs {Path(original).name} (reference)")
    for col in ("variant", "text CER↓", "faces", "ID cos↑", "face LPIPS↓", "lapvar↑", "img LPIPS↓", "SSIM↑", "PSNR↑"):
        table.add_column(col)
    for label, _ in parsed:
        st = face_stats[label]
        wl, ws, wp = whole[label]
        table.add_row(
            label,
            _fmt(ocr_cer[label]),
            str(st.n_faces),
            _fmt(_mean(st.identity)),
            _fmt(_mean(st.lpips)),
            _fmt(_mean(st.lapvar_ratio)),
            _fmt(wl),
            _fmt(ws),
            _fmt(wp, 1),
        )
    console.print(table)
    console.print(
        "  Legend: CER lower=better; ID cos higher=better; face LPIPS lower=better; "
        "lapvar ratio ~1=detail kept, <1=smoothed/plastic; img LPIPS lower=better; SSIM/PSNR higher=closer."
    )


if __name__ == "__main__":
    main()
