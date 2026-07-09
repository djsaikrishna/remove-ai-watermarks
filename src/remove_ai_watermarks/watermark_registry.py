"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Localize -> fill.** A known mark is removed by LOCALIZING it (a template-free,
version-robust detector that returns a binary footprint MASK) and then handing
that mask to ONE shared, swappable fill backend (``region_eraser``: cv2 Telea/NS,
MI-GAN, or big-LaMa). No mark carries a reverse-alpha step any more: the old
``original = (wm - a*logo)/(1-a)`` recovery depended on a fixed captured alpha map
at a fixed position, broke whenever a vendor re-rendered or moved its mark, and was
not colour-lossless even with the right map (it amplifies quantization/JPEG-chroma
error by ``1/(1-a)`` -- the "the color just changed, not removed" reports). The
localizer stays cheap (cv2/numpy, CPU) so a memory-tight caller can run it on a
small worker; the heavy fill (MI-GAN / LaMa) is opt-in and chosen by the caller.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
  - ``jimeng_pill`` -- Jimeng-basic "AI生成" pill, top-left (capture-less).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

Region = tuple[int, int, int, int]

# Fill backend for the shared removal path. ``auto`` resolves to the preferred
# installed ONNX model (MI-GAN) or cv2 (see ``resolve_backend``); the others force
# a specific backend (mirrors the ``erase`` command's ``--backend``).
Backend = Literal["auto", "cv2", "migan", "lama"]

# Detection sensitivity for the removal path -- how much to trust a borderline mark.
#   * ``strict``: high-precision visual gate only; never relaxed, so a clean image is
#     never touched (the gate demotes a sparkle-shaped content match, so it never fills
#     a clean corner). Lowest recall on faint/moved marks.
#   * ``auto`` (default): relax a mark's gate ONLY when the image carries same-product
#     evidence the mark is there -- metadata provenance for that vendor, or a confidently
#     detected sibling mark of the same product (see ``_resolve_relax``). No evidence ->
#     stays strict. Safe: it only escalates where the mark is corroborated.
#   * ``assume_ai``: relax every mark's gate regardless of evidence -- the caller asserts
#     the image is AI and wants the mark gone (e.g. a metadata-stripped screenshot uploaded
#     to a watermark remover). Recovers the faint/moved marks the strict gate demotes
#     (~49% -> ~89% Gemini recall, corpus-measured), at the cost of a harmless small fill
#     on some clean corners. The library CANNOT infer this from a stripped image -- only the
#     caller's out-of-band context (the user uploaded to remove a mark) justifies it.
Sensitivity = Literal["auto", "strict", "assume_ai"]

# Product family per mark, for the ``auto`` cross-mark corroboration: a confidently
# detected mark relaxes only OTHER marks of the SAME product (different corners, one
# product -- the Jimeng wordmark + the Jimeng pill). Doubao and Jimeng are BOTH ByteDance
# but distinct products in the SAME bottom-right corner, so they must NOT cross-relax
# (relaxing Doubao on a Jimeng wordmark would spuriously fire Doubao on it).
_PRODUCT_OF: dict[str, str] = {
    "gemini": "gemini",
    "doubao": "doubao",
    "jimeng": "jimeng",
    "jimeng_pill": "jimeng",  # same product as the Jimeng wordmark
    "samsung": "samsung",
}


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region


@dataclass(frozen=True)
class Localization:
    """A located mark: its detection verdict plus the full-frame removal mask.

    ``mask`` is a full-frame uint8 array (255 = remove) sized to the image, or None
    when nothing should be removed (no detection and not forced, or the footprint
    could not be placed). ``region`` is the mark's bbox (for logging / residual
    positioning)."""

    detected: bool
    confidence: float
    region: Region
    mask: NDArray[Any] | None


@dataclass(frozen=True)
class Context:
    """The evidence + policy the removal arbiter decides against (perception is
    kept separate from this decision). ``sensitivity`` is the caller's intent
    (see :data:`Sensitivity`); ``provenance`` is the vendor keys local metadata
    confirms, the evidence that drives ``auto``. Bundling them into one object is
    why the arbiter can be a pure function of ``(candidates, context)``."""

    sensitivity: Sensitivity = "auto"
    provenance: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Candidate:
    """One mark's PERCEPTION output -- what the engine sees, with NO policy applied.

    Carries the mark's verdict at BOTH trust levels (``detected_strict`` = the
    conservative gate, ``detected_relaxed`` = the gate the engine relaxes to under
    provenance/assume), so the arbiter can pick per mark without re-running detection.
    ``features`` is a generic bag of physical measurements a mark's gate may need (the
    mark owns which it reports via ``KnownMark._features``); e.g. the pill supplies
    ``footprint_flat`` (0/1). Empty for marks whose gate needs no extra evidence."""

    key: str
    label: str
    location: str
    region: Region
    detected_strict: bool
    detected_relaxed: bool
    confidence: float
    features: dict[str, float]  # generic; both construction sites always supply it (empty when none)


@dataclass(frozen=True)
class Decision:
    """The arbiter's verdict for one fired mark: remove it, at the resolved trust
    level (``relax`` feeds the mark's mask build so the fill footprint matches the
    level the mark was accepted at)."""

    candidate: Candidate
    relax: bool


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and mask it.

    Removal is uniform (:meth:`remove`): localize the mark to a mask, then fill that
    mask with the chosen backend. Each mark supplies two cheap cv2/numpy callables --
    ``_detect`` (verdict + bbox, no mask; used by the identify scan) and ``_mask``
    (the full-frame footprint mask; used by removal)."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    _detect: Callable[..., MarkDetection]
    _mask: Callable[..., NDArray[Any] | None]
    # Optional physical-feature probe: the mark's OWN measurements its gate needs
    # (e.g. the pill's footprint flatness), so the perception pass stays uniform and
    # does not special-case any mark. None = the mark's gate needs no extra evidence.
    _features: Callable[..., dict[str, float]] | None = None

    def features(self, image: NDArray[Any]) -> dict[str, float]:
        """Physical features the mark reports for the arbiter's gate (empty if none)."""
        return self._features(image) if self._features is not None else {}

    def detect(self, image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        """Detect the mark (verdict + bbox, no mask). ``provenance`` signals that
        external metadata already confirms this vendor, so the engine may relax its
        trust threshold (a mark it would otherwise demote as a content false positive
        is trusted when provenance says the vendor is present)."""
        return self._detect(image, provenance=provenance)

    def localize(self, image: NDArray[Any], *, provenance: bool = False, force: bool = False) -> Localization:
        """Detect and build the removal mask in one call. Returns a
        :class:`Localization`; ``mask`` is None unless the mark is detected (or
        ``force`` bypasses detection for the mark's usual footprint)."""
        det = self.detect(image, provenance=provenance)
        if not (det.detected or force):
            return Localization(det.detected, det.confidence, det.region, None)
        mask = self._mask(image, force=force)
        return Localization(det.detected, det.confidence, det.region, mask)

    def remove(
        self,
        image: NDArray[Any],
        *,
        backend: Backend = "auto",
        provenance: bool = False,
        force: bool = False,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by localize -> fill; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox, or None if nothing was removed.

        ``backend`` picks the fill (``auto`` = MI-GAN if installed else cv2; or force
        ``cv2``/``migan``/``lama``). ``provenance`` relaxes the detector's trust gate
        when metadata already confirms the vendor. ``force`` removes at the mark's
        usual footprint even without a positive detection (the ``--no-detect`` path).
        NB: the CLI does NOT use ``region`` to clear alpha on save -- that zeroing
        caused the issue-#30 white box."""
        loc = self.localize(image, provenance=provenance, force=force)
        if loc.mask is None or not loc.mask.any():
            return image.copy(), None
        return fill(image, loc.mask, backend=backend), (loc.region if loc.detected else None)


# Single source of truth for the Gemini-sparkle "trust this as a real mark"
# confidence, shared by BOTH the removal arbitration here (`_gemini_detect`) and
# the provenance detector in `identify` (which imports it as its sparkle threshold).
# Defining it once removes the detect-vs-remove
# threshold drift the retained-corpus mining surfaced (2026-06-20): identify
# would report a sparkle while removal declined it, or vice versa, whenever the
# two independently-maintained 0.5 constants fell out of step. Now they cannot.
#
# Value 0.5 is corpus-validated: the gemini engine's own `detected` flag uses a
# looser internal threshold (0.35) and weakly fires (~0.36-0.42) on unrelated
# bottom-right text -- a real Doubao mark scores ~0.40-0.42 as a gemini match,
# and its core-ring brightness margin is HIGHER than a genuine faint sparkle's,
# so neither confidence nor the brightness gate separates them in the [0.35, 0.5)
# band. Lowering this gate to recover faint sparkles was evaluated against that
# band (2026-06-20) and REJECTED for the no-provenance case: it cannot be done
# without re-admitting the Doubao-text / content false positives. The band below
# the gate is therefore left to the metadata-confirmed path below.
GEMINI_SPARKLE_TRUST_CONF = 0.5
_GEMINI_AUTO_MIN_CONF = GEMINI_SPARKLE_TRUST_CONF

# Provenance-confirmed Gemini trust gate. When external metadata already proves the
# image is a Google generation (C2PA issuer "Google"/"Gemini"), the [0.35, 0.5)
# band that the no-provenance gate leaves out is no longer ambiguous with Doubao
# text: a Doubao image carries ByteDance provenance, not Google, so it never reaches
# this relaxed gate. The vendor moving/re-rendering the sparkle (bigger, lighter,
# shifted north-west) drops a real sparkle into this band, and the fixed-slot
# detector demotes it -- provenance is exactly the extra evidence that lets us trust
# it. Set to the engine's own internal `detected` floor (0.35); combined with the
# engine's FP-gate being skipped under provenance (see gemini_engine), this recovers
# the moved-mark misses without touching the no-provenance precision.
_GEMINI_PROVENANCE_MIN_CONF = 0.35

# ── Engine adapters (lazy singletons; engines are cv2-only, no model load) ──

_engines: dict[str, Any] = {}


def _engine(key: str) -> Any:
    if key not in _engines:
        if key == "gemini":
            from remove_ai_watermarks.gemini_engine import GeminiEngine

            _engines[key] = GeminiEngine()
        elif key == "doubao":
            from remove_ai_watermarks.doubao_engine import DoubaoEngine

            _engines[key] = DoubaoEngine()
        elif key == "jimeng":
            from remove_ai_watermarks.jimeng_engine import JimengEngine

            _engines[key] = JimengEngine()
        elif key == "samsung":
            from remove_ai_watermarks.samsung_engine import SamsungEngine

            _engines[key] = SamsungEngine()
        elif key == "jimeng_pill":
            from remove_ai_watermarks.pill_engine import PillEngine

            _engines[key] = PillEngine()
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def inpaint_model_available() -> bool:
    """True when any ONNX inpaint-model backend (MI-GAN or big-LaMa) can run."""
    from remove_ai_watermarks import region_eraser

    return region_eraser.migan_available() or region_eraser.lama_available()


def preferred_inpaint_backend() -> str:
    """Backend used by the ``auto`` fill: MI-GAN (light, droplet-friendly, the
    default) when its ONNX runtime is available, else cv2. big-LaMa is NOT auto-
    selected -- it is a heavier explicit opt-in via ``--backend lama`` (both models
    run on onnxruntime, so availability alone cannot express the user's intent; the
    light model is the safe default)."""
    from remove_ai_watermarks import region_eraser

    return "migan" if region_eraser.migan_available() else "cv2"


def resolve_backend(backend: Backend) -> Literal["cv2", "migan", "lama"]:
    """Resolve ``auto`` to the preferred installed backend; pass the rest through."""
    if backend == "auto":
        return "migan" if preferred_inpaint_backend() == "migan" else "cv2"
    return backend


def fill(image: NDArray[Any], mask: NDArray[Any], *, backend: Backend = "auto") -> NDArray[Any]:
    """The ONE shared, mark-agnostic removal: erase ``mask`` (255 = remove) via the
    chosen inpaint backend. Delegates to :func:`region_eraser.erase`; ``auto``
    resolves to MI-GAN when installed else cv2 (see :func:`resolve_backend`)."""
    from remove_ai_watermarks import region_eraser

    return region_eraser.erase(image, mask=mask, backend=resolve_backend(backend))


# ── Detection adapters (verdict + bbox; no mask work on this path) ──
# The identify scan calls `detect_marks`, which must stay cheap (it runs every
# detector on the memory-tight identify host), so detection never builds a mask.


def _gemini_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image, trust_provenance=provenance)
    gate = _GEMINI_PROVENANCE_MIN_CONF if provenance else _GEMINI_AUTO_MIN_CONF
    detected = bool(d.detected) and d.confidence >= gate
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_mask(image: NDArray[Any], *, force: bool = False) -> NDArray[Any] | None:
    return _engine("gemini").footprint_mask(image, force=force)


# The three text-mark engines (Doubao/Jimeng/Samsung) share the TextMarkEngine
# interface, so one parameterized adapter pair drives all of them -- a new
# text mark is one `_text_mark(...)` row below, not another copy-paste of these
# bodies. Detection matches the glyph silhouette; the mask is the template-free
# glyph-bbox footprint (see TextMarkEngine.footprint_mask).
def _text_mark_detect(key: str, label: str, location: str) -> Callable[..., MarkDetection]:
    def detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
        d = _engine(key).detect(image, provenance=provenance)
        return MarkDetection(key, label, location, d.detected, d.confidence, d.region)

    return detect


def _text_mark_mask(key: str) -> Callable[..., NDArray[Any] | None]:
    def mask(image: NDArray[Any], *, force: bool = False) -> NDArray[Any] | None:
        return _engine(key).footprint_mask(image, force=force)

    return mask


def _text_mark(key: str, label: str, location: str) -> KnownMark:
    """A text-mark registry row (Doubao/Jimeng/Samsung): glyph-silhouette detect +
    template-free glyph-bbox mask."""
    return KnownMark(key, label, location, True, _text_mark_detect(key, label, location), _text_mark_mask(key))


# ── Capture-less mark: the Jimeng-basic "AI生成" pill (top-left) ──
# Detection is edge-NCC of a synthetic silhouette; the mask is a fixed top-left
# geometry box (see pill_engine). Removal is the same localize -> fill as the rest.
def _pill_detect(image: NDArray[Any], *, provenance: bool = False) -> MarkDetection:
    del provenance  # the pill detector is provenance-independent; its relaxation lives entirely in _keep_pill
    d = _engine("jimeng_pill").detect(image)
    return MarkDetection("jimeng_pill", "Jimeng AI生成 pill", "top-left", d.detected, d.confidence, d.region)


def _pill_mask(image: NDArray[Any], *, force: bool = False) -> NDArray[Any] | None:
    return _engine("jimeng_pill").footprint_mask(image, force=force)


def _pill_features(image: NDArray[Any]) -> dict[str, float]:
    """The pill's own gate feature: top-left footprint flatness (1.0 = flat enough for
    an invisible fill), read by the metadata/assume arm of :func:`_keep_pill`."""
    return {"footprint_flat": float(_engine("jimeng_pill").footprint_is_flat(image))}


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, _gemini_detect, _gemini_mask),
    _text_mark("doubao", "Doubao 豆包AI生成 text", "bottom-right"),
    _text_mark("jimeng", "Jimeng 即梦AI wordmark", "bottom-right"),
    _text_mark("samsung", "Samsung Galaxy AI text", "bottom-left"),
    KnownMark("jimeng_pill", "Jimeng AI生成 pill", "top-left", True, _pill_detect, _pill_mask, _pill_features),
)


def known_marks() -> tuple[KnownMark, ...]:
    """All registered known visible watermarks."""
    return _REGISTRY


def mark_keys() -> list[str]:
    """Keys of all registered marks (for CLI choices)."""
    return [m.key for m in _REGISTRY]


def get_mark(key: str) -> KnownMark:
    """Look up a known mark by key (raises KeyError if unknown)."""
    for m in _REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def detect_marks(
    image: NDArray[Any],
    *,
    include_explicit: bool = True,
    provenance: frozenset[str] = frozenset(),
) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    Returns one MarkDetection per scanned mark (``detected`` flags which fired).
    ``include_explicit=False`` scans only the ``in_auto`` marks -- the set used
    by ``--mark auto``. ``provenance`` names the vendor keys that external metadata
    already confirms, so each named mark's detector may relax its trust gate."""
    return [m.detect(image, provenance=m.key in provenance) for m in _REGISTRY if include_explicit or m.in_auto]


def _resolve_relax(
    key: str,
    *,
    sensitivity: Sensitivity,
    provenance: frozenset[str],
    strict_keys: set[str],
) -> bool:
    """Whether mark ``key``'s detection gate is relaxed (strict -> assume level).

    The single place that turns the ``sensitivity`` policy + evidence into a per-mark
    boolean (which the engines consume): ``strict`` never relaxes, ``assume_ai`` always
    relaxes, and ``auto`` relaxes only on same-product evidence -- the vendor confirmed
    by metadata (``key in provenance``) or a confidently strict-detected sibling of the
    same product (``_PRODUCT_OF``)."""
    if sensitivity == "strict":
        return False
    if sensitivity == "assume_ai":
        return True
    if key in provenance:
        return True
    product = _PRODUCT_OF[key]
    return any(_PRODUCT_OF[k] == product for k in strict_keys if k != key)


def _keep_pill(keys: set[str], *, provenance: frozenset[str], sensitivity: Sensitivity, footprint_flat: bool) -> bool:
    """Whether to auto-remove the capture-less 'AI生成' pill given the fired marks.

    Pure decision (the flatness feature is precomputed at perception time and passed
    in). The pill detector is weak (~7% raw false-fire) and metadata/intent confirms
    the platform, not pill presence, so a naive confirmation-OR gate over-fires: on a
    32k real-upload corpus (2026-07) the metadata-only arm was only ~27% precise and
    its false fires were textured ceilings/walls that the fill visibly SMEARS. Arms:
      * bottom-right "★ 即梦AI" wordmark fired -> ~94% precise, and it survives
        metadata-STRIPPED uploads: remove the pill unrestricted;
      * TC260 metadata confirms Jimeng (``"jimeng" in provenance``, no wordmark) OR the
        caller asserts AI (``sensitivity == "assume_ai"``) -> remove ONLY when the
        top-left footprint is flat enough for an invisible fill (``footprint_flat``),
        so real flat-scene pills (and harmless flat false fires) are cleaned while the
        damaging textured false fires are left untouched even under assume_ai.
    A Doubao image is TC260 too but is not Jimeng-basic, so the pill never rides on a
    Doubao detection. No confirmation at all -> never remove (blocks false fires on
    non-Jimeng content)."""
    if "doubao" in keys:
        return False
    if "jimeng" in keys:
        return True
    if "jimeng" in provenance or sensitivity == "assume_ai":
        return footprint_flat
    return False


def _build_candidates(image: NDArray[Any]) -> list[Candidate]:
    """PERCEPTION pass: run every ``in_auto`` mark's detector at both trust levels and
    package the raw verdicts + physical features into :class:`Candidate` objects. No
    policy here -- the arbiter (:func:`decide`) makes every keep/drop call.

    Each mark is detected at the strict AND the relaxed (``provenance=True``) level so
    :func:`decide` can pick per mark without re-running detection; a relaxed gate is
    monotonically more permissive, so this reproduces the old strict-then-relax pass
    exactly. The loop is uniform -- it knows nothing about any specific mark: each mark
    reports its own gate features via :meth:`KnownMark.features` (computed only when the
    mark is detected, so a clean image pays nothing extra)."""
    cands: list[Candidate] = []
    for m in _REGISTRY:
        if not m.in_auto:
            continue
        strict = m.detect(image, provenance=False)
        relaxed = m.detect(image, provenance=True)
        feats = m.features(image) if (strict.detected or relaxed.detected) else {}
        cands.append(
            Candidate(
                m.key, m.label, m.location, strict.region, strict.detected, relaxed.detected, strict.confidence, feats
            )
        )
    return cands


def decide(candidates: list[Candidate], context: Context) -> list[Decision]:
    """The removal ARBITER: a pure function turning perception + context into the
    ordered list of marks to remove (and the trust level each was accepted at).

    All policy lives here, in one place: per-mark relaxation (:func:`_resolve_relax`,
    which needs the strict-detected siblings for ``auto`` cross-mark corroboration) and
    the capture-less pill gate (:func:`_keep_pill`). No image, no I/O -- so it is
    unit-testable in isolation and the same decision drives every caller."""
    strict_keys = {c.key for c in candidates if c.detected_strict}
    fired: list[Decision] = []
    for c in candidates:
        relax = _resolve_relax(
            c.key, sensitivity=context.sensitivity, provenance=context.provenance, strict_keys=strict_keys
        )
        if c.detected_relaxed if relax else c.detected_strict:
            fired.append(Decision(c, relax))
    keys = {d.candidate.key for d in fired}
    if "jimeng_pill" in keys:
        pill = next(d for d in fired if d.candidate.key == "jimeng_pill")
        flat = bool(pill.candidate.features.get("footprint_flat", 0.0))
        if not _keep_pill(
            keys,
            provenance=context.provenance,
            sensitivity=context.sensitivity,
            footprint_flat=flat,
        ):
            fired = [d for d in fired if d.candidate.key != "jimeng_pill"]
    return fired


def remove_auto_marks(
    image: NDArray[Any],
    *,
    sensitivity: Sensitivity = "auto",
    provenance: frozenset[str] = frozenset(),
    backend: Backend = "auto",
) -> tuple[NDArray[Any], list[str]]:
    """Remove EVERY decided ``in_auto`` mark in one pass, chaining the result.

    The three stages are separated: PERCEPTION (:func:`_build_candidates` -- engines
    report what they see, no policy), DECISION (:func:`decide` -- the pure arbiter over
    ``(candidates, Context)``), ACTION (localize -> :func:`fill` per winner). Marks
    coexist in different corners -- a Jimeng-basic image carries BOTH the top-left pill
    AND the bottom-right wordmark -- so every decided mark is removed, chained on the
    progressively-cleaned image (order does not matter, each re-localizes its corner).

    Three orthogonal knobs: ``sensitivity`` (how hard to trust a borderline mark --
    see :data:`Sensitivity`), ``provenance`` (vendor keys external metadata confirms,
    the evidence that drives ``auto``; the TC260 label maps to ``jimeng``/``doubao``),
    and ``backend`` (the shared fill). Returns ``(result, [labels removed])``; empty
    means nothing fired."""
    context = Context(sensitivity=sensitivity, provenance=provenance)
    result = image
    labels: list[str] = []
    for d in decide(_build_candidates(image), context):
        result, _ = get_mark(d.candidate.key).remove(result, backend=backend, provenance=d.relax, force=False)
        labels.append(d.candidate.label)
    return result, labels
