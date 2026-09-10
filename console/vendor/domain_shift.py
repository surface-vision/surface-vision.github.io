"""Photometric domain normalisation for frames that were not shot on the NEU-DET rig.

The detector was fine-tuned on one camera, one lighting rig and one steel grade.
Anything else arrives with a different exposure: measured here, NEU-DET's pooled
grey level is mean 129.2 sigma 54.3 while real Severstal strip, over 1,400 frames,
sits at mean 87.3.
That offset alone costs false alarms, and it is the one part of the domain gap
that costs nothing to remove -- unlike the textural part, which needs data.

This module owns three things:

* **the reference distribution** -- the 256-bin grey histogram of the NEU-DET
  *training* split (never val or test: the reference is part of the model, so it
  may only see what the model saw), cached to JSON next to the weights so the
  deployed pipeline does not need the dataset on disk. Two targets are stored,
  because which one you match onto turns out to matter more than the method:
  the **population** histogram (every training pixel pooled, sigma 54.3) and the
  **typical frame** histogram (per-frame histograms averaged after each frame is
  shifted onto the population mean, sigma 30.1). Pooling folds the across-frame
  exposure spread into the target, so matching a single frame onto it
  over-stretches contrast; the typical-frame target dominates it on every
  measurement below and is the default;
* **the mapping** -- classical histogram matching of an incoming frame's
  luminance onto that reference, plus a cheaper mean/sigma affine variant and a
  ``strength`` dial for blending back towards the identity;
* **the honest description** -- the before/after statistics and a one-line,
  operator-readable note, so a console can say *why* the pixels changed.

Deliberately depends only on numpy and OpenCV. Normalisation is a preprocessing
step; a dataset-preparation script must be able to import it without pulling in
torch. ``_to_rgb`` therefore restates ``inference._to_rgb``'s contract locally,
and ``tests/test_smoke.py`` asserts the two agree so they cannot drift.

**Measured, and it is not the free win the audit reported.** Full protocol and
numbers in ``reports/ood_guard.md``; the short version:

* On 3,200 crops of genuinely defect-free real Severstal strip at conf 0.25, the
  naive recipe (full match onto the pooled population) does cut crop false
  alarms 44.3 % -> 34.1 %, reproducing the audit. But **defect-crop recall falls
  with it, 58.3 % -> 43.1 %**. Held at equal recall, that recipe is *worse* than
  doing nothing: 45.8 % false alarms against a 44.3 % baseline, and cross-domain
  AUC drops 0.615 -> 0.593. The whole apparent gain was a threshold artefact.
* The typical-frame target at ``strength=0.25`` -- the module defaults -- is a
  real gain: cross-domain AUC 0.615 -> **0.641**, and at matched 58.3 % recall
  the clean-strip false-alarm rate goes 44.3 % -> **38.0 %** (-14 % relative).
* It is **not free in-domain**. NEU-DET held-out test mAP50 falls monotonically
  with strength -- 0.7524 (off) / 0.7249 (0.25) / 0.6503 (0.50) / 0.4912 (1.00,
  population target) -- because the shipped weights were trained on
  un-normalised frames, so any per-frame remapping is train/serve skew.

So normalisation is **off by default in the deployed pipeline** and the 0.7524
headline is untouched. It is a per-source commissioning setting -- switched on
for a camera that is not the NEU-DET rig -- not a per-frame decision, because
``histogram_distance`` provably cannot tell the two populations apart (see its
docstring). The way to collect the benefit without the in-domain cost is to
normalise the training data and re-fit; that is a training change and is out of
this module's scope.

CLI::

    .venv/bin/python src/domain_shift.py --rebuild
    .venv/bin/python src/domain_shift.py --stats data/neu-det/test/images/scratches_271.jpg
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/domain_shift.py` from the Jindal Stainless surface-defect
# repository, carried into the Hugging Face Space so the app imports nothing from
# outside this directory (the repository's `src/` does not exist on the Space
# host). The upstream file is the source of truth; re-vendor rather than patch.
#
# `PROJECT_ROOT` in these modules is `Path(__file__).resolve().parent.parent`,
# which on the Space resolves to the Space root -- so `models/`, `reports/`,
# `data/` and `assets/` are the trimmed copies shipped beside this directory and
# every path the upstream code builds still resolves.
#
# Deviations from upstream are marked `# SPACE:` inline and listed in
# `vendor/README.md`. There are none in this file unless such a marker appears.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "neu-det"

# The reference ships with the model, not with the dataset: a deployed pipeline
# has weights and this JSON, and never needs 1,440 training JPEGs on disk.
REFERENCE_PATH = PROJECT_ROOT / "models" / "domain_reference.json"

REFERENCE_SPLIT = "train"
SCHEMA = "jsw-domain-reference/2"
LEVELS = 256

# Methods accepted by `match_to_reference`.
METHODS = ("histogram", "affine", "none")

# Which stored distribution a frame is mapped onto. "frame" is the average
# single training frame; "population" is every training pixel pooled. Measured
# on real Severstal strip, "frame" wins at every strength (cross-domain AUC
# 0.641 / 0.634 / 0.631 / 0.624 at strength 0.25 / 0.50 / 0.75 / 1.00, against
# 0.629 / 0.614 / 0.604 / 0.593 for "population", from a 0.615 baseline).
TARGETS = ("frame", "population")

# Defaults chosen by measurement, not by taste: strength 0.25 on the frame
# target is the peak of the cross-domain dose-response curve, and the point
# beyond which in-domain mAP50 falls faster than false alarms do.
RECOMMENDED_TARGET = "frame"
RECOMMENDED_STRENGTH = 0.25

_CACHE: dict[str, "ReferenceStats"] = {}
_CACHE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #


def _to_rgb(image: str | Path | np.ndarray | Any) -> np.ndarray:
    """Normalise any accepted input to a contiguous HWC RGB uint8 array.

    Mirrors `inference._to_rgb` exactly (tested), restated here so that this
    module stays importable without torch.
    """
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB array, got shape {image.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    if hasattr(image, "convert") and hasattr(image, "size"):
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))

    raise TypeError(
        f"Unsupported image type {type(image)!r}; expected path, PIL.Image or ndarray."
    )


def to_grey(image: str | Path | np.ndarray | Any) -> np.ndarray:
    """ITU-R BT.601 luminance, the same weighting OpenCV and PIL both use."""
    return cv2.cvtColor(_to_rgb(image), cv2.COLOR_RGB2GRAY)


def grey_histogram(grey: np.ndarray) -> np.ndarray:
    """256-bin count vector of a uint8 luminance plane."""
    return np.bincount(np.asarray(grey, dtype=np.uint8).ravel(), minlength=LEVELS).astype(
        np.float64
    )


def _validate_histogram(counts: Any, what: str) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.float64)
    if arr.shape != (LEVELS,):
        raise ValueError(f"{what} must have {LEVELS} bins, got {arr.shape}")
    if arr.min() < 0:
        raise ValueError(f"{what} has a negative bin")
    if arr.sum() <= 0:
        raise ValueError(f"{what} is empty")
    return arr


def _cdf(counts: np.ndarray) -> np.ndarray:
    total = float(counts.sum())
    if total <= 0.0:
        # Degenerate input (an empty array). A flat CDF maps everything to 0,
        # which is the honest answer: there is no distribution to match.
        return np.zeros(LEVELS, dtype=np.float64)
    return np.cumsum(counts / total)


# --------------------------------------------------------------------------- #
# the reference distribution
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class ReferenceStats:
    """The photometric distribution the detector was trained on.

    Two 256-bin grey-level count vectors are held:

    * `histogram` -- every pixel of every training frame pooled. Its sigma
      (54.3) is inflated by the exposure differences *between* frames.
    * `frame_histogram` -- the per-frame histograms averaged after each frame is
      shifted so its own mean sits on the population mean. Sigma 30.1, close to
      the mean per-frame sigma of 26.8. This is what a single frame is supposed
      to look like, and therefore the right thing to map a single frame onto.

    Everything else is derived from them and stored so a reader of the JSON does
    not have to recompute anything to sanity-check it.
    """

    histogram: np.ndarray
    mean: float
    std: float
    n_images: int
    n_pixels: int
    dataset: str = "NEU-DET"
    split: str = REFERENCE_SPLIT
    luma: str = "ITU-R BT.601"
    created_utc: str = ""
    per_frame_mean_std: float = 0.0
    per_frame_std_mean: float = 0.0
    percentiles: dict[str, float] | None = None
    frame_histogram: np.ndarray | None = None
    frame_mean: float = 0.0
    frame_std: float = 0.0

    def __post_init__(self) -> None:
        self.histogram = _validate_histogram(self.histogram, "reference histogram")
        if self.frame_histogram is None:
            # A schema-1 file, or a caller that only supplied the pooled target.
            self.frame_histogram = self.histogram
            self.frame_mean, self.frame_std = self.mean, self.std
        else:
            self.frame_histogram = _validate_histogram(
                self.frame_histogram, "reference frame histogram"
            )
        self._cdf_cache: dict[str, np.ndarray] = {}

    @property
    def cdf(self) -> np.ndarray:
        """Non-decreasing 256-point CDF of the pooled population."""
        return self.cdf_for("population")

    @property
    def frame_cdf(self) -> np.ndarray:
        """Non-decreasing 256-point CDF of the average single training frame."""
        return self.cdf_for("frame")

    def cdf_for(self, target: str) -> np.ndarray:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
        cached = self._cdf_cache.get(target)
        if cached is None:
            counts = self.frame_histogram if target == "frame" else self.histogram
            cached = _cdf(np.asarray(counts, dtype=np.float64))
            self._cdf_cache[target] = cached
        return cached

    def moments_for(self, target: str) -> tuple[float, float]:
        """(mean, sigma) of the chosen target, for the affine method."""
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
        return (self.frame_mean, self.frame_std) if target == "frame" else (self.mean, self.std)

    @property
    def pdf(self) -> np.ndarray:
        return self.histogram / float(self.histogram.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "source": {
                "dataset": self.dataset,
                "split": self.split,
                "n_images": int(self.n_images),
                "n_pixels": int(self.n_pixels),
            },
            "luma": self.luma,
            "created_utc": self.created_utc,
            "grey": {
                "mean": round(float(self.mean), 4),
                "std": round(float(self.std), 4),
                "percentiles": {k: round(float(v), 2) for k, v in (self.percentiles or {}).items()},
            },
            "per_frame": {
                "std_of_frame_means": round(float(self.per_frame_mean_std), 4),
                "mean_of_frame_stds": round(float(self.per_frame_std_mean), 4),
                "mean": round(float(self.frame_mean), 4),
                "std": round(float(self.frame_std), 4),
            },
            "histogram": [int(v) for v in self.histogram],
            "frame_histogram": [int(round(float(v))) for v in np.asarray(self.frame_histogram)],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceStats":
        schema = payload.get("schema")
        if schema != SCHEMA:
            raise ValueError(f"unsupported reference schema {schema!r}, expected {SCHEMA!r}")
        source = payload.get("source", {})
        grey = payload.get("grey", {})
        per_frame = payload.get("per_frame", {})
        return cls(
            histogram=np.asarray(payload["histogram"], dtype=np.float64),
            mean=float(grey.get("mean", 0.0)),
            std=float(grey.get("std", 0.0)),
            n_images=int(source.get("n_images", 0)),
            n_pixels=int(source.get("n_pixels", 0)),
            dataset=str(source.get("dataset", "NEU-DET")),
            split=str(source.get("split", REFERENCE_SPLIT)),
            luma=str(payload.get("luma", "ITU-R BT.601")),
            created_utc=str(payload.get("created_utc", "")),
            per_frame_mean_std=float(per_frame.get("std_of_frame_means", 0.0)),
            per_frame_std_mean=float(per_frame.get("mean_of_frame_stds", 0.0)),
            frame_histogram=(
                np.asarray(payload["frame_histogram"], dtype=np.float64)
                if "frame_histogram" in payload
                else None
            ),
            frame_mean=float(per_frame.get("mean", grey.get("mean", 0.0))),
            frame_std=float(per_frame.get("std", grey.get("std", 0.0))),
        )

    def describe(self) -> str:
        return (
            f"{self.dataset} {self.split} split, {self.n_images} frames, "
            f"grey mean {self.mean:.1f} sigma {self.std:.1f} pooled / "
            f"sigma {self.frame_std:.1f} per typical frame"
        )


def compute_reference(
    images: Iterable[str | Path] | None = None,
    *,
    split: str = REFERENCE_SPLIT,
    data_root: Path = DATA_ROOT,
) -> ReferenceStats:
    """Build both reference targets from the training split in one pass.

    The pooled target sums raw histograms. The typical-frame target sums each
    frame's histogram *after shifting it onto the population mean*, so the
    result carries the shape of a single frame rather than the shape of a
    population that happens to span a range of exposures.
    """
    paths = (
        [Path(p) for p in images]
        if images is not None
        else sorted((Path(data_root) / split / "images").glob("*.jpg"))
    )
    if not paths:
        raise FileNotFoundError(
            f"no images to build a reference from (split={split!r}, root={data_root})"
        )

    counts = np.zeros(LEVELS, dtype=np.float64)
    frame_means: list[float] = []
    frame_stds: list[float] = []
    per_frame: list[np.ndarray] = []
    for path in paths:
        grey = to_grey(path)
        histogram = grey_histogram(grey)
        counts += histogram
        per_frame.append(histogram)
        frame_means.append(float(grey.mean()))
        frame_stds.append(float(grey.std()))

    levels = np.arange(LEVELS, dtype=np.float64)
    pdf = counts / counts.sum()
    mean = float((pdf * levels).sum())
    std = float(np.sqrt((pdf * (levels - mean) ** 2).sum()))
    cdf = np.cumsum(pdf)

    # np.roll wraps, which would move bright pixels to the bottom of the range.
    # Frames are shifted by at most a few tens of levels and the extremes are
    # sparse, so shift with zero fill instead and let the mass fall off the end.
    frame_counts = np.zeros(LEVELS, dtype=np.float64)
    for histogram, frame_mean in zip(per_frame, frame_means):
        shift = int(round(mean - frame_mean))
        if shift == 0:
            frame_counts += histogram
        elif shift > 0:
            frame_counts[shift:] += histogram[: LEVELS - shift]
        else:
            frame_counts[: LEVELS + shift] += histogram[-shift:]
    frame_pdf = frame_counts / frame_counts.sum()
    frame_mean_level = float((frame_pdf * levels).sum())
    frame_std_level = float(np.sqrt((frame_pdf * (levels - frame_mean_level) ** 2).sum()))
    percentiles = {
        f"p{q}": float(np.searchsorted(cdf, q / 100.0)) for q in (1, 5, 25, 50, 75, 95, 99)
    }

    return ReferenceStats(
        histogram=counts,
        frame_histogram=frame_counts,
        frame_mean=frame_mean_level,
        frame_std=frame_std_level,
        mean=mean,
        std=std,
        n_images=len(paths),
        n_pixels=int(counts.sum()),
        split=split,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        per_frame_mean_std=float(np.std(frame_means)),
        per_frame_std_mean=float(np.mean(frame_stds)),
        percentiles=percentiles,
    )


def save_reference(stats: ReferenceStats, path: Path = REFERENCE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n")
    return path


def load_reference(
    path: Path = REFERENCE_PATH, *, build_if_missing: bool = True
) -> ReferenceStats:
    """Process-level cached reference, rebuilt from the training split on demand.

    The console reruns its script on every widget change; re-reading and
    re-parsing a 256-bin JSON each time is cheap but pointless, and rebuilding
    from 1,440 JPEGs is not cheap at all.
    """
    key = str(Path(path).resolve())
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    file_path = Path(path)
    if file_path.is_file():
        stats = ReferenceStats.from_dict(json.loads(file_path.read_text()))
    elif build_if_missing:
        stats = compute_reference()
        save_reference(stats, file_path)
    else:
        raise FileNotFoundError(f"no cached reference at {file_path}")

    with _CACHE_LOCK:
        _CACHE[key] = stats
    return stats


def clear_reference_cache() -> None:
    """Drop the process-level cache; only tests and `--rebuild` need this."""
    with _CACHE_LOCK:
        _CACHE.clear()


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PhotometricStats:
    """What a frame looks like photometrically, and how far that is from training."""

    mean: float
    std: float
    p1: float
    p99: float
    saturation: float
    hist_distance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "p1": round(self.p1, 1),
            "p99": round(self.p99, 1),
            "saturation": round(self.saturation, 3),
            "hist_distance": round(self.hist_distance, 2),
        }


def histogram_distance(
    image: str | Path | np.ndarray | Any, reference: ReferenceStats | None = None
) -> float:
    """Wasserstein-1 distance between the frame's grey CDF and the reference, in grey levels.

    Reported in the same units an engineer already thinks in: a value of 40 means
    "on average this frame's pixels sit 40 grey levels away from where a training
    pixel of the same rank would sit".

    Measured spread (see `reports/ood_guard.md`): NEU-DET train frames span
    5.7-116.5 and real Severstal frames span 4.3-126.1, so this number is a good
    description of exposure drift and a *bad* out-of-distribution test -- a blank
    white frame scores 125.8, inside the range real steel occupies.
    """
    ref = reference if reference is not None else load_reference()
    grey = to_grey(image)
    return float(np.abs(_cdf(grey_histogram(grey)) - ref.cdf).sum())


def frame_statistics(
    image: str | Path | np.ndarray | Any, reference: ReferenceStats | None = None
) -> PhotometricStats:
    """Everything the console needs to explain a normalisation in one call."""
    ref = reference if reference is not None else load_reference()
    rgb = _to_rgb(image)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    counts = grey_histogram(grey)
    cdf = _cdf(counts)
    saturation = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].mean())
    return PhotometricStats(
        mean=float(grey.mean()),
        std=float(grey.std()),
        p1=float(np.searchsorted(cdf, 0.01)),
        p99=float(np.searchsorted(cdf, 0.99)),
        saturation=saturation,
        hist_distance=float(np.abs(cdf - ref.cdf).sum()),
    )


# --------------------------------------------------------------------------- #
# the mapping
# --------------------------------------------------------------------------- #


def build_lut(
    source_counts: np.ndarray, reference_cdf: np.ndarray, *, strength: float = 1.0
) -> np.ndarray:
    """Classical histogram-matching LUT, blended `strength` of the way from identity.

    `lut[v]` is the lowest reference level whose cumulative mass reaches the
    source's cumulative mass at `v`. Monotone non-decreasing by construction, so
    it can never invert the contrast of a defect against its background.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    identity = np.arange(LEVELS, dtype=np.float64)
    source_cdf = _cdf(np.asarray(source_counts, dtype=np.float64))
    if source_cdf.max() <= 0.0:
        return identity.astype(np.uint8)
    matched = np.searchsorted(np.asarray(reference_cdf, dtype=np.float64), source_cdf, side="left")
    matched = np.clip(matched, 0, LEVELS - 1).astype(np.float64)
    blended = (1.0 - strength) * identity + strength * matched
    return np.clip(np.rint(blended), 0, LEVELS - 1).astype(np.uint8)


def affine_lut(
    source_mean: float,
    source_std: float,
    reference: ReferenceStats,
    *,
    target: str = RECOMMENDED_TARGET,
    strength: float = 1.0,
    min_std: float = 1e-3,
) -> np.ndarray:
    """Mean/sigma match: cheaper than histogram matching, and shape-preserving.

    Useful when the frame's histogram shape is trusted and only exposure and gain
    have drifted -- and as the control that says how much of the measured benefit
    came from the two moments alone rather than from the full distribution.
    Measured, it is the weaker control: at full strength on Severstal it lands at
    AUC 0.556 against histogram matching's 0.566 and a 0.615 baseline.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    identity = np.arange(LEVELS, dtype=np.float64)
    ref_mean, ref_std = reference.moments_for(target)
    gain = ref_std / max(float(source_std), min_std)
    mapped = (identity - float(source_mean)) * gain + ref_mean
    blended = (1.0 - strength) * identity + strength * mapped
    return np.clip(np.rint(blended), 0, LEVELS - 1).astype(np.uint8)


def match_to_reference(
    image: str | Path | np.ndarray | Any,
    reference: ReferenceStats | None = None,
    *,
    method: str = "histogram",
    target: str = RECOMMENDED_TARGET,
    strength: float = RECOMMENDED_STRENGTH,
) -> np.ndarray:
    """Map a frame onto the training photometric distribution. Returns RGB uint8.

    The LUT is derived from the frame's **luminance** histogram and applied to
    all three channels. For the greyscale imagery this system is built for
    (measured mean HSV saturation 0.000 on all 1,800 NEU-DET frames and all 1,400
    Severstal frames) that is exact histogram matching; for colour input it is a
    monotone tone curve, which is the conservative thing to do -- it will not
    invent chroma that the detector has never seen.

    Args:
        image: path, PIL image or HWC array, in the same set `DefectDetector`
            accepts.
        reference: distribution to match onto; defaults to the cached NEU-DET
            training reference.
        method: ``"histogram"`` (full CDF match), ``"affine"`` (mean/sigma only)
            or ``"none"`` (a validated no-op, so callers can A/B without
            branching).
        target: ``"frame"`` (the average single training frame -- the default,
            and the measured winner) or ``"population"`` (every training pixel
            pooled, which over-stretches contrast).
        strength: 0.0 leaves the frame untouched, 1.0 lands it fully on the
            reference. Intermediate values blend the LUT, not the pixels, so the
            result stays monotone. Defaults to the measured optimum, 0.25.

    Measured on 3,200 crops of defect-free real Severstal strip plus 1,079
    mask-confirmed defect crops, imgsz 256, at the defaults: cross-domain AUC
    0.615 -> 0.641, and at matched 58.3 % defect recall the clean-strip false
    alarm rate 44.3 % -> 38.0 %. In-domain cost on held-out NEU-DET, mAP50
    0.7524 -> 0.7249; do not switch this on for the NEU-DET rig.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")

    rgb = _to_rgb(image)
    if method == "none" or strength <= 0.0:
        return rgb.copy()

    ref = reference if reference is not None else load_reference()
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if method == "histogram":
        lut = build_lut(grey_histogram(grey), ref.cdf_for(target), strength=strength)
    else:
        lut = affine_lut(
            float(grey.mean()), float(grey.std()), ref, target=target, strength=strength
        )
    return np.ascontiguousarray(cv2.LUT(rgb, lut))


def normalisation_note(
    before: PhotometricStats, after: PhotometricStats, reference: ReferenceStats
) -> str:
    """One operator-readable line explaining what the normalisation did."""
    return (
        f"Photometric normalisation: frame grey mean {before.mean:.0f} sigma "
        f"{before.std:.0f} -> {after.mean:.0f} sigma {after.std:.0f}, matched to the "
        f"{reference.dataset} training distribution (mean {reference.mean:.0f} sigma "
        f"{reference.std:.0f}). Distance from that distribution "
        f"{before.hist_distance:.0f} -> {after.hist_distance:.0f} grey levels."
    )


def normalise_frame(
    image: str | Path | np.ndarray | Any,
    reference: ReferenceStats | None = None,
    *,
    method: str = "histogram",
    target: str = RECOMMENDED_TARGET,
    strength: float = RECOMMENDED_STRENGTH,
) -> tuple[np.ndarray, dict[str, Any]]:
    """`match_to_reference` plus the evidence, for a console that must justify itself.

    Returns the normalised RGB frame and a JSON-serialisable record holding the
    before/after statistics and a ready-to-render caption.
    """
    ref = reference if reference is not None else load_reference()
    rgb = _to_rgb(image)
    before = frame_statistics(rgb, ref)
    out = match_to_reference(rgb, ref, method=method, target=target, strength=strength)
    after = frame_statistics(out, ref)
    return out, {
        "method": method,
        "target": target,
        "strength": float(np.clip(strength, 0.0, 1.0)),
        "reference": ref.describe(),
        "before": before.to_dict(),
        "after": after.to_dict(),
        "note": normalisation_note(before, after, ref),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rebuild", action="store_true", help="recompute the reference from the training split"
    )
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--split", default=REFERENCE_SPLIT)
    parser.add_argument("--stats", type=Path, nargs="*", help="print statistics for these frames")
    parser.add_argument("--method", default="histogram", choices=list(METHODS))
    parser.add_argument("--target", default=RECOMMENDED_TARGET, choices=list(TARGETS))
    parser.add_argument("--strength", type=float, default=RECOMMENDED_STRENGTH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.rebuild:
        stats = compute_reference(split=args.split)
        path = save_reference(stats, args.reference)
        clear_reference_cache()
        print(f"reference written to {path}")
        print(f"  {stats.describe()}")
        print(f"  pixels {stats.n_pixels:,}  percentiles {stats.percentiles}")
        print(f"  typical-frame target: mean {stats.frame_mean:.2f} sigma {stats.frame_std:.2f}")
    else:
        stats = load_reference(args.reference)
        print(f"reference: {stats.describe()} ({args.reference})")

    for path in args.stats or []:
        _, record = normalise_frame(
            path, stats, method=args.method, target=args.target, strength=args.strength
        )
        print(f"\n{path}")
        print(f"  before {record['before']}")
        print(f"  after  {record['after']}")
        print(f"  {record['note']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
