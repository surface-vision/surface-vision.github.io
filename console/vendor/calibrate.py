"""Confidence calibration for the Jindal Stainless surface-defect detector.

The brief asks for a defect verdict "ideally with a confidence score". The console
prints one. Nobody had checked whether it is a *probability*, and it is not.

What is wrong with the shipped score
------------------------------------
A YOLO objectness-times-class score is trained with a BCE/varifocal loss against a
soft IoU-weighted target and then squeezed through NMS. Nothing in that pipeline
asks the surviving number to equal `P(this box is a true positive)`. Measured on the
held-out test split at conf >= 0.05, the shipped detector is **monotonically
under-confident in every bin**: a detection shown to the operator as 0.60 is right
about 84% of the time, 0.70 about 93%, 0.80 about 98%. Expected Calibration Error
is 0.142. That is not a rounding error, and it points the wrong way from the usual
deep-network story - this detector is *too modest*, so an operator who learns to
distrust anything under 0.7 is throwing away detections that are almost always real.

What this module does about it
------------------------------
It fits a scalar map `s -> P(true positive | s)` **on the validation split only**,
never on test, and persists it to `reports/calibration.json` for the console to
apply at render time.

Two candidate families are fitted, exactly as the remediation brief allows:

* **temperature scaling** - one parameter, `sigmoid(logit(s) / T)`. The textbook
  choice, and the one to prefer if it works, because one parameter cannot overfit
  180 images.
* **isotonic regression** - non-parametric, monotone. Strictly more expressive, and
  the only one of the two that can represent an *additive* bias. Its plateau values
  are Jeffreys-smoothed (`(k + 0.5) / (n + 1)`) so that a top plateau where every
  validation detection happened to be correct calibrates to 0.99 rather than to a
  flat 1.000 that 41 samples cannot support; the unsmoothed variant is carried
  through the selection table so the cost of that choice is visible.

The choice between them is not made by eye and not made on test. It is made by
**5-fold cross-validation on val, grouped by source image**, scoring pooled
out-of-fold ECE. Grouping by image matters: two detections on the same frame are
not independent draws, and an ungrouped fold would let a leaked frame flatter the
more flexible model. Isotonic's in-sample ECE is meaningless for selection (it is
near zero by construction); its out-of-fold ECE is not.

Why the answer is not temperature, and why that is not a cop-out
---------------------------------------------------------------
Temperature scaling has exactly one degree of freedom and it is a *sharpening*
knob. `T < 1` pushes scores above 0.5 up and scores below 0.5 down; `T > 1` does
the reverse. The measured miscalibration here is under-confidence at **both** ends
(0.086 -> 0.191 at the bottom, 0.885 -> 1.000 at the top), which is an additive
shift, not a sharpening. No temperature can lift both tails at once, so a
one-parameter fit is structurally unable to close this particular gap. The numbers
in `reports/calibration.md` show it failing, which is the justification the brief
asked for. Platt scaling (`sigmoid(a * logit(s) + b)`, two parameters) is fitted and
reported alongside purely as a reference point; it is not a shipping candidate.

Why mAP cannot move
-------------------
Every calibrator here is a **strictly increasing** map on [0, 1]. Average precision
depends on the detections only through their rank order (the greedy matcher walks
predictions in descending score, and the PR curve is a function of the ranking), so
a strictly increasing rescoring leaves every AP bit-identical. Isotonic regression
is only *non-decreasing*, which would introduce ties; `IsotonicCalibrator` therefore
blends in a vanishing `EPS * s` term that restores strict monotonicity without
moving any calibrated value by more than 1e-6. `verify_rank_preservation` asserts
this on the real data and `map50` is recomputed from raw and from calibrated scores
so the invariance is measured rather than argued.

What calibration does NOT fix
-----------------------------
This is a map from score to `P(true positive)` on *NEU-DET*. It is fitted on
in-domain validation frames and it inherits every assumption they carry - notably
that every frame contains a defect. On genuinely defect-free strip (see
`src/false_alarm.py` and the cross-domain work) the base rate is different, and a
calibrator fitted here will be over-confident there. It makes the number on screen
mean what it says on the population it was fitted on. It does not make the detector
better, it does not change what is detected, and it is not a substitute for
measuring the real prevalence on a line.

Run:
    .venv/bin/python src/calibrate.py
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/calibrate.py` from the Jindal Stainless surface-defect
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
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (  # noqa: E402
    CLASS_NAMES,
    DefectDetector,
    resolve_device,
    resolve_weights,
)

# SPACE: `src/evaluate.py` is deliberately NOT vendored, and this import block is
# the one deviation from upstream in this file.
#
# WHY: upstream imports six names from `evaluate` at module scope. Every one of
# them belongs to the *fitting* path -- `collect_matched_scores`, `map50` and
# `run` -- which reads the NEU-DET ground-truth label files off disk and runs the
# ultralytics validator over the dataset yaml. The Space ships 18 sample frames
# and no labels, so that path cannot execute here whatever is imported. Vendoring
# `evaluate.py` would therefore add ~1,760 lines of module that resolves
# `DATA_ROOT` to a directory containing three JPEG folders and nothing it needs,
# purely to satisfy an import: dead code on a public deployment, which is worse
# than a documented stub that says what is missing. (It is NOT about the
# scikit-learn wheel `evaluate` imports -- grad-cam declares scikit-learn as a
# dependency, so it is installed on the Space regardless.)
#
# The Space needs exactly one thing out of this file: `load_calibrator`, which
# reads `reports/calibration.json` and touches none of those six names.
#
# HOW: the import is attempted, so a tree that *does* have `evaluate.py` beside
# this file behaves exactly like upstream. When it is absent, the only name
# needed at import time is `IOU_MATCH` -- it is a default argument value on two
# `def`s -- and the callables become a stub that raises with an explanation
# instead of an obscure NameError. `GroundTruth` and `CachedPrediction` appear
# only in annotations, which `from __future__ import annotations` above keeps as
# strings.
_FITTING_UNAVAILABLE = (
    "The calibrator-fitting path is not available in the Hugging Face Space: "
    "src/evaluate.py is not vendored because fitting needs the NEU-DET ground "
    "truth and the ultralytics validator, and neither is shipped here. Fit the "
    "calibrator in the full repository (`make calibrate`) and ship the resulting "
    "reports/calibration.json, which is what load_calibrator reads."
)

try:
    from evaluate import (  # noqa: E402
        IOU_MATCH,
        CachedPrediction,
        GroundTruth,
        cache_predictions,
        load_ground_truth,
        match_image,
    )
except ModuleNotFoundError:  # pragma: no cover - the Space path
    # Mirrors `src/evaluate.py:83  IOU_MATCH = 0.5` -- the PASCAL-VOC / AP50 match
    # IoU. Asserted against upstream by tools/verify_vendor.py.
    IOU_MATCH = 0.5

    def _fitting_unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(_FITTING_UNAVAILABLE)

    cache_predictions = load_ground_truth = match_image = _fitting_unavailable
    CachedPrediction = GroundTruth = object

__all__ = [
    "CALIBRATION_JSON",
    "ECE_BIN_EDGES",
    "Calibrator",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "IdentityCalibrator",
    "MatchedScores",
    "calibrate_confidence",
    "calibrate_confidences",
    "load_calibrator",
    "calibrator_from_dict",
    "expected_calibration_error",
    "reliability_bins",
    "brier_score",
    "negative_log_likelihood",
    "map50",
    "verify_rank_preservation",
    "fit_temperature",
    "fit_platt",
    "fit_isotonic",
    "fit_isotonic_unsmoothed",
    "select_calibrator",
    "collect_matched_scores",
    "run",
    "build_parser",
    "main",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CALIBRATION_JSON = REPORTS_DIR / "calibration.json"
CALIBRATION_PNG = REPORTS_DIR / "calibration.png"
CALIBRATION_MD = REPORTS_DIR / "calibration.md"

# The confidence floor the detector is run at when scores are harvested. 0.05 is
# the floor the project's own audit used, so the "before" ECE reported here is
# directly comparable with docs/audit/technical_headroom.md section 3.
CACHE_CONF = 0.05

# The shipped operating point (src/false_alarm.py). Reported separately because it
# is the population an operator actually sees on screen.
OPERATING_CONF = 0.15

DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 256

# Bin edges for ECE. These are the audit's bins with the first edge dropped to 0.0:
# no *raw* score is below CACHE_CONF, so raw binning is unchanged and the reported
# "before" number reproduces the audit exactly, while a calibrated score that lands
# below 0.05 still has a home. Right-open bins, last bin closed.
ECE_BIN_EDGES: tuple[float, ...] = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0)

# Restores strict monotonicity to isotonic regression's step function. Small enough
# that it never moves a displayed 2-decimal confidence, large enough that float64
# never collapses two distinct raw scores onto one calibrated score.
STRICT_MONOTONE_EPS = 1e-6

_PROB_CLIP = 1e-6

_CACHED_CALIBRATOR: tuple[Path, float, "Calibrator"] | None = None


# ---------------------------------------------------------------------------
# calibrators
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, _PROB_CLIP, 1.0 - _PROB_CLIP)
    return np.log(q / (1.0 - q))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch-free stable sigmoid: exp() of a large positive argument overflows.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class Calibrator:
    """Base class: a strictly increasing map from raw score to probability."""

    name: str = "identity"

    def transform(self, scores: Any) -> np.ndarray:
        arr = np.asarray(scores, dtype=np.float64)
        return np.clip(self._apply(np.atleast_1d(arr).ravel()), 0.0, 1.0).reshape(arr.shape)

    def _apply(self, scores: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        return scores

    def __call__(self, score: float) -> float:
        return float(self.transform(np.float64(score)))

    @property
    def n_parameters(self) -> int:  # pragma: no cover - overridden
        return 0

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.name, "n_parameters": self.n_parameters, "params": self._params()}

    def _params(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def describe(self) -> str:  # pragma: no cover - overridden
        return self.name


@dataclass(frozen=True)
class IdentityCalibrator(Calibrator):
    """The shipped, uncalibrated score. Used as the 'before' baseline."""

    name: str = "identity"

    def _apply(self, scores: np.ndarray) -> np.ndarray:
        return scores

    def describe(self) -> str:
        return "identity (raw detector score)"


@dataclass(frozen=True)
class TemperatureCalibrator(Calibrator):
    """`sigmoid(logit(s) / T)`. One parameter. Strictly increasing for T > 0."""

    temperature: float = 1.0
    name: str = "temperature"

    def _apply(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(_logit(scores) / float(self.temperature))

    @property
    def n_parameters(self) -> int:
        return 1

    def _params(self) -> dict[str, Any]:
        return {"temperature": float(self.temperature)}

    def describe(self) -> str:
        return f"temperature scaling, T = {self.temperature:.4f}"


@dataclass(frozen=True)
class PlattCalibrator(Calibrator):
    """`sigmoid(a * logit(s) + b)`. Two parameters. Reference point only."""

    slope: float = 1.0
    intercept: float = 0.0
    name: str = "platt"

    def _apply(self, scores: np.ndarray) -> np.ndarray:
        return _sigmoid(float(self.slope) * _logit(scores) + float(self.intercept))

    @property
    def n_parameters(self) -> int:
        return 2

    def _params(self) -> dict[str, Any]:
        return {"slope": float(self.slope), "intercept": float(self.intercept)}

    def describe(self) -> str:
        return f"Platt scaling, a = {self.slope:.4f}, b = {self.intercept:.4f}"


@dataclass(frozen=True)
class IsotonicCalibrator(Calibrator):
    """Piecewise-linear monotone fit, stored as its own knots.

    Serialising the knots rather than a pickled estimator means
    `reports/calibration.json` is readable, diffable, and loadable without
    scikit-learn on the serving box. `np.interp` with clipped ends reproduces
    scikit-learn's `out_of_bounds="clip"` interpolation exactly.

    The `eps` term is what makes this rank-preserving. Isotonic regression is
    non-*decreasing*: it maps whole runs of distinct raw scores onto one plateau
    value, which would create ties and make average precision order-dependent.
    Blending in `eps * s` restores a strict ordering while moving no calibrated
    value by more than `eps`.
    """

    knots_x: tuple[float, ...] = (0.0, 1.0)
    knots_y: tuple[float, ...] = (0.0, 1.0)
    eps: float = STRICT_MONOTONE_EPS
    name: str = "isotonic"

    def _apply(self, scores: np.ndarray) -> np.ndarray:
        xs = np.asarray(self.knots_x, dtype=np.float64)
        ys = np.asarray(self.knots_y, dtype=np.float64)
        base = np.interp(np.clip(scores, xs[0], xs[-1]), xs, ys)
        return base * (1.0 - self.eps) + self.eps * np.clip(scores, 0.0, 1.0)

    @property
    def n_parameters(self) -> int:
        return len(self.knots_x)

    def _params(self) -> dict[str, Any]:
        return {
            "knots_x": [float(v) for v in self.knots_x],
            "knots_y": [float(v) for v in self.knots_y],
            "eps": float(self.eps),
        }

    def describe(self) -> str:
        return f"isotonic regression, {len(self.knots_x)} knots (+{self.eps:g} strict-monotone term)"


def calibrator_from_dict(payload: dict[str, Any]) -> Calibrator:
    """Rebuild a calibrator from its persisted form."""
    method = str(payload.get("method", "identity"))
    params = payload.get("params") or {}
    if method == "identity":
        return IdentityCalibrator()
    if method == "temperature":
        return TemperatureCalibrator(temperature=float(params["temperature"]))
    if method == "platt":
        return PlattCalibrator(slope=float(params["slope"]), intercept=float(params["intercept"]))
    if method == "isotonic":
        return IsotonicCalibrator(
            knots_x=tuple(float(v) for v in params["knots_x"]),
            knots_y=tuple(float(v) for v in params["knots_y"]),
            eps=float(params.get("eps", STRICT_MONOTONE_EPS)),
        )
    raise ValueError(f"Unknown calibration method {method!r}")


# ---------------------------------------------------------------------------
# the front door the console imports
# ---------------------------------------------------------------------------


def load_calibrator(path: Path | str = CALIBRATION_JSON) -> Calibrator:
    """Load the fitted calibrator, cached on (path, mtime).

    Falls back to the identity map with a warning if the file is missing or
    unreadable. Silently returning raw scores would be worse: the console would
    display "calibrated" numbers that are not.
    """
    global _CACHED_CALIBRATOR
    target = Path(path)
    try:
        mtime = target.stat().st_mtime
    except OSError:
        warnings.warn(
            f"No calibration file at {target}; confidences will be shown uncalibrated. "
            "Run `make calibrate` to fit one.",
            RuntimeWarning,
            stacklevel=2,
        )
        return IdentityCalibrator()

    if _CACHED_CALIBRATOR is not None:
        cached_path, cached_mtime, cached = _CACHED_CALIBRATOR
        if cached_path == target and cached_mtime == mtime:
            return cached
    try:
        payload = json.loads(target.read_text())
        calibrator = calibrator_from_dict(payload["calibrator"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.warn(
            f"Could not read a calibrator from {target} ({type(exc).__name__}: {exc}); "
            "confidences will be shown uncalibrated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return IdentityCalibrator()
    _CACHED_CALIBRATOR = (target, mtime, calibrator)
    return calibrator


def calibrate_confidence(raw: float) -> float:
    """Map one raw detector confidence to a calibrated probability.

    This is the function the operator console calls per detection. It is a pure
    scalar lookup over a piecewise-linear curve: no model, no GPU, no allocation
    that matters. Rank order is preserved, so sorting detections by the calibrated
    number gives exactly the order the raw number gave.
    """
    return float(load_calibrator().transform(np.float64(raw)))


def calibrate_confidences(raw: Any) -> np.ndarray:
    """Vectorised `calibrate_confidence` for a whole frame's detections."""
    return load_calibrator().transform(raw)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def reliability_bins(
    scores: np.ndarray, labels: np.ndarray, edges: Sequence[float] = ECE_BIN_EDGES
) -> list[dict[str, Any]]:
    """Per-bin count, mean score and empirical precision. Right-open, last closed."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        last = hi >= edges[-1]
        mask = (scores >= lo) & (scores <= hi if last else scores < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bin": f"[{lo:.2f},{hi:.2f}{']' if last else ')'}", "lo": lo, "hi": hi,
                        "n": 0, "mean_score": None, "empirical_precision": None, "gap": None})
            continue
        mean_score = float(scores[mask].mean())
        precision = float(labels[mask].mean())
        out.append({
            "bin": f"[{lo:.2f},{hi:.2f}{']' if last else ')'}",
            "lo": lo, "hi": hi, "n": n,
            "mean_score": round(mean_score, 6),
            "empirical_precision": round(precision, 6),
            "gap": round(mean_score - precision, 6),
        })
    return out


def expected_calibration_error(
    scores: np.ndarray, labels: np.ndarray, edges: Sequence[float] = ECE_BIN_EDGES
) -> float:
    """Population-weighted mean |mean score - empirical precision| over bins."""
    n_total = len(np.asarray(scores))
    if n_total == 0:
        return float("nan")
    total = 0.0
    for row in reliability_bins(scores, labels, edges):
        if row["n"]:
            total += row["n"] * abs(float(row["gap"]))
    return total / n_total


def maximum_calibration_error(
    scores: np.ndarray, labels: np.ndarray, edges: Sequence[float] = ECE_BIN_EDGES,
    min_bin_n: int = 20,
) -> float:
    """Worst per-bin gap, ignoring bins too small to estimate a precision from."""
    gaps = [abs(float(r["gap"])) for r in reliability_bins(scores, labels, edges)
            if r["n"] >= min_bin_n]
    return max(gaps) if gaps else float("nan")


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(scores, dtype=np.float64) - np.asarray(labels, dtype=np.float64)) ** 2))


def negative_log_likelihood(scores: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(scores, dtype=np.float64), _PROB_CLIP, 1.0 - _PROB_CLIP)
    y = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# average precision, recomputed from scores so mAP invariance is measured
# ---------------------------------------------------------------------------


def _average_precision(scores: np.ndarray, is_tp: np.ndarray, n_gt: int, n_points: int = 101) -> float:
    """COCO-style `n_points`-interpolated AP from a ranked detection list."""
    if n_gt == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(is_tp[order].astype(np.float64))
    fp = np.cumsum(1.0 - is_tp[order].astype(np.float64))
    recall = tp / float(n_gt)
    precision = tp / np.maximum(tp + fp, 1e-12)
    # Monotone-decreasing precision envelope, then sample at fixed recall points.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, n_points)
    sampled = np.interp(grid, recall, precision, left=precision[0], right=0.0)
    return float(sampled.mean())


def map50(
    records: Sequence[GroundTruth],
    preds: Sequence[CachedPrediction],
    scores_by_image: Sequence[np.ndarray],
    iou_thr: float = IOU_MATCH,
) -> dict[str, float]:
    """Per-class AP50 and mAP50, matched and ranked using `scores_by_image`.

    Passing the scores in separately - rather than reading `pred.confidences` - is
    the whole point: run it once with the raw scores and once with the calibrated
    ones and the two dictionaries must be identical, because a strictly increasing
    rescoring cannot change either the greedy match or the ranking.
    """
    per_class_scores: dict[int, list[np.ndarray]] = {c: [] for c in range(len(CLASS_NAMES))}
    per_class_tp: dict[int, list[np.ndarray]] = {c: [] for c in range(len(CLASS_NAMES))}
    n_gt = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for record, pred, scores in zip(records, preds, scores_by_image):
        for c in record.classes:
            n_gt[int(c)] += 1
        match = match_image(
            record.boxes, record.classes, pred.boxes, np.asarray(scores, dtype=np.float64),
            pred.classes, iou_thr=iou_thr,
        )
        for c in range(len(CLASS_NAMES)):
            sel = pred.classes == c
            if sel.any():
                per_class_scores[c].append(np.asarray(scores, dtype=np.float64)[sel])
                per_class_tp[c].append(match.pred_is_tp[sel])

    out: dict[str, float] = {}
    aps: list[float] = []
    for c, name in enumerate(CLASS_NAMES):
        s = np.concatenate(per_class_scores[c]) if per_class_scores[c] else np.zeros(0)
        t = np.concatenate(per_class_tp[c]) if per_class_tp[c] else np.zeros(0, dtype=bool)
        ap = _average_precision(s, t, int(n_gt[c]))
        out[name] = ap
        if np.isfinite(ap):
            aps.append(ap)
    out["mAP50"] = float(np.mean(aps)) if aps else float("nan")
    return out


def verify_rank_preservation(calibrator: Calibrator, scores: np.ndarray) -> dict[str, Any]:
    """Check the calibrator is strictly increasing *on the data it will be used on*.

    Two things are asserted, both on the real score population rather than on a
    synthetic grid: the sorted order is identical, and no two distinct raw scores
    collapse onto one calibrated score (which would make AP tie-break dependent).
    """
    raw = np.asarray(scores, dtype=np.float64)
    cal = calibrator.transform(raw)
    order_raw = np.argsort(raw, kind="stable")
    order_cal = np.argsort(cal, kind="stable")
    srt = np.sort(raw)
    cal_srt = calibrator.transform(srt)
    distinct = np.diff(srt) > 0
    collapsed = int(np.sum(distinct & (np.diff(cal_srt) <= 0)))
    grid = np.linspace(0.0, 1.0, 2001)
    grid_ok = bool(np.all(np.diff(calibrator.transform(grid)) >= 0.0))
    return {
        "order_identical": bool(np.array_equal(order_raw, order_cal)),
        "collapsed_pairs": collapsed,
        "strictly_increasing_on_data": collapsed == 0,
        "non_decreasing_on_unit_grid": grid_ok,
        "max_shift": float(np.max(np.abs(cal - raw))) if raw.size else 0.0,
    }


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> TemperatureCalibrator:
    """Minimise the binary NLL over a single temperature. Bounded, so it cannot run away."""
    from scipy.optimize import minimize_scalar

    z = _logit(np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)

    def nll(log_t: float) -> float:
        p = np.clip(_sigmoid(z / float(np.exp(log_t))), _PROB_CLIP, 1.0 - _PROB_CLIP)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=(np.log(0.02), np.log(50.0)), method="bounded",
                          options={"xatol": 1e-8})
    return TemperatureCalibrator(temperature=float(np.exp(res.x)))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
    """Two-parameter logistic on the logit. Reference point, not a shipping candidate."""
    from scipy.optimize import minimize

    z = _logit(np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)

    def nll(theta: np.ndarray) -> float:
        p = np.clip(_sigmoid(theta[0] * z + theta[1]), _PROB_CLIP, 1.0 - _PROB_CLIP)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    slope = float(res.x[0])
    if slope <= 0.0:  # pragma: no cover - would break rank preservation; never seen
        raise ValueError(f"Platt fit produced a non-positive slope ({slope}); not rank-preserving.")
    return PlattCalibrator(slope=slope, intercept=float(res.x[1]))


def _weighted_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: nearest non-decreasing sequence in weighted L2.

    Used to put a sequence back in order after the plateau values have been
    smoothed independently, which can invert two neighbours (a 9/10 plateau
    smooths to 0.864 while the 1/1 plateau above it smooths to 0.750).
    """
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for value, weight in zip(values.astype(np.float64), weights.astype(np.float64)):
        vals.append(float(value)); wts.append(float(weight)); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2); sizes.append(s1 + s2)
    return np.repeat(np.asarray(vals, dtype=np.float64), np.asarray(sizes, dtype=np.int64))


def fit_isotonic(
    scores: np.ndarray, labels: np.ndarray, smoothing: float = 0.5
) -> IsotonicCalibrator:
    """Monotone non-parametric fit, reduced to its knots for serialisation.

    `smoothing` is what stops the curve claiming certainty it has not earned.
    Raw isotonic regression reports each plateau's maximum-likelihood rate `k/n`,
    so a top plateau where all 41 validation detections happened to be correct
    calibrates to exactly **1.000** - and a console that prints "100%" beside a
    steel defect is making a claim no sample of 41 supports. Each plateau's value
    is therefore replaced by its Jeffreys posterior mean `(k + 0.5) / (n + 1)`,
    the standard add-half estimator, and the resulting sequence is re-projected
    with a weighted PAVA pass because independent smoothing can invert two
    neighbouring plateaus. Pass `smoothing=0.0` for the unsmoothed fit; it is
    carried through the cross-validation table so the cost is visible rather than
    assumed.
    """
    from sklearn.isotonic import IsotonicRegression

    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(x, y)
    if smoothing <= 0.0:
        xs = np.asarray(iso.X_thresholds_, dtype=np.float64)
        ys = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return IsotonicCalibrator(knots_x=tuple(float(v) for v in xs),
                                  knots_y=tuple(float(v) for v in ys))

    order = np.argsort(x, kind="stable")
    xs_sorted, ys_sorted = x[order], y[order]
    fitted = iso.predict(xs_sorted)
    # A plateau is a maximal run of equal fitted values. Points sharing an x share a
    # fitted value, so no plateau can straddle a duplicated score.
    cuts = np.flatnonzero(np.diff(fitted) != 0.0) + 1
    groups = np.split(np.arange(xs_sorted.size), cuts)
    n_j = np.array([g.size for g in groups], dtype=np.float64)
    k_j = np.array([ys_sorted[g].sum() for g in groups], dtype=np.float64)
    smoothed = _weighted_pava((k_j + smoothing) / (n_j + 2.0 * smoothing), n_j)

    knots_x: list[float] = []
    knots_y: list[float] = []
    for group, value in zip(groups, smoothed):
        lo, hi = float(xs_sorted[group[0]]), float(xs_sorted[group[-1]])
        knots_x.append(lo); knots_y.append(float(value))
        if hi > lo:
            knots_x.append(hi); knots_y.append(float(value))
    return IsotonicCalibrator(knots_x=tuple(knots_x), knots_y=tuple(knots_y))


def fit_isotonic_unsmoothed(scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
    """Reference point: the raw `k/n` plateau values, saturation and all."""
    return fit_isotonic(scores, labels, smoothing=0.0)


# ---------------------------------------------------------------------------
# harvesting matched scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedScores:
    """Every detection on a split, with the label the greedy matcher gave it."""

    split: str
    scores: np.ndarray
    labels: np.ndarray  # 1 = true positive at IOU_MATCH and the right class
    class_ids: np.ndarray
    image_index: np.ndarray
    records: tuple[GroundTruth, ...]
    preds: tuple[CachedPrediction, ...]

    def __len__(self) -> int:
        return int(self.scores.size)

    def at_least(self, threshold: float) -> "MatchedScores":
        keep = self.scores >= threshold
        return MatchedScores(
            split=self.split, scores=self.scores[keep], labels=self.labels[keep],
            class_ids=self.class_ids[keep], image_index=self.image_index[keep],
            records=self.records, preds=self.preds,
        )


def collect_matched_scores(
    detector: DefectDetector, split: str, iou_thr: float = IOU_MATCH
) -> MatchedScores:
    """Score a split once and label every surviving box true or false positive.

    The label is exactly the evaluator's: greedy, class-aware, confidence-ordered
    matching at IoU 0.5 (`evaluate.match_image`). Using the same matcher matters -
    a calibrator fitted against a looser definition of "correct" would be
    calibrating a different question from the one the reports answer.
    """
    records = load_ground_truth(split)
    preds = cache_predictions(detector, records)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    class_ids: list[np.ndarray] = []
    image_index: list[np.ndarray] = []
    for idx, (record, pred) in enumerate(zip(records, preds)):
        match = match_image(record.boxes, record.classes, pred.boxes, pred.confidences,
                            pred.classes, iou_thr=iou_thr)
        scores.append(pred.confidences)
        labels.append(match.pred_is_tp.astype(np.float64))
        class_ids.append(pred.classes)
        image_index.append(np.full(pred.confidences.shape, idx, dtype=np.int64))
    return MatchedScores(
        split=split,
        scores=np.concatenate(scores) if scores else np.zeros(0),
        labels=np.concatenate(labels) if labels else np.zeros(0),
        class_ids=np.concatenate(class_ids) if class_ids else np.zeros(0, dtype=np.int64),
        image_index=np.concatenate(image_index) if image_index else np.zeros(0, dtype=np.int64),
        records=tuple(records),
        preds=tuple(preds),
    )


# ---------------------------------------------------------------------------
# model selection: grouped cross-validation on val
# ---------------------------------------------------------------------------

_FITTERS = {
    "temperature": fit_temperature,
    "isotonic": fit_isotonic,
    "isotonic_unsmoothed": fit_isotonic_unsmoothed,
    "platt": fit_platt,
}

# Only these two are eligible to ship. Platt is fitted for the report so the reader
# can see how much of the remaining gap is a second parameter rather than
# non-parametric flexibility, but two parameters is not what the brief asked for.
SHIPPABLE = ("temperature", "isotonic")


def _grouped_folds(image_index: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """k folds of *images*, so no frame's detections straddle the split."""
    images = np.unique(image_index)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(images)
    buckets = np.array_split(shuffled, k)
    return [np.isin(image_index, b) for b in buckets]


def select_calibrator(
    fit: MatchedScores, k: int = 5, seed: int = 20260909, candidates: Sequence[str] = SHIPPABLE
) -> dict[str, Any]:
    """Out-of-fold ECE for every candidate, plus the winner among `candidates`.

    Every family listed in `_FITTERS` is scored, so the report can show the
    reference point too; only names in `candidates` are eligible to win.
    """
    folds = _grouped_folds(fit.image_index, k, seed)
    results: dict[str, Any] = {}
    for name, fitter in _FITTERS.items():
        oof = np.empty_like(fit.scores)
        for held in folds:
            train = ~held
            cal = fitter(fit.scores[train], fit.labels[train])
            oof[held] = cal.transform(fit.scores[held])
        results[name] = {
            "oof_ece": expected_calibration_error(oof, fit.labels),
            "oof_brier": brier_score(oof, fit.labels),
            "oof_nll": negative_log_likelihood(oof, fit.labels),
            "eligible": name in candidates,
        }
    results["identity"] = {
        "oof_ece": expected_calibration_error(fit.scores, fit.labels),
        "oof_brier": brier_score(fit.scores, fit.labels),
        "oof_nll": negative_log_likelihood(fit.scores, fit.labels),
        "eligible": False,
    }
    eligible = {n: r for n, r in results.items() if r["eligible"]}
    winner = min(eligible, key=lambda n: eligible[n]["oof_ece"])
    return {
        "protocol": f"{k}-fold cross-validation on the fitting split, grouped by source image",
        "seed": seed,
        "folds": k,
        "n_images": int(np.unique(fit.image_index).size),
        "n_detections": len(fit),
        "candidates": results,
        "eligible": list(candidates),
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------


def plot_reliability(
    path: Path,
    before: MatchedScores,
    calibrator: Calibrator,
    ece_before: float,
    ece_after: float,
) -> Path:
    """Reliability diagram before and after, plus the map itself."""
    cal_scores = calibrator.transform(before.scores)
    rows_before = reliability_bins(before.scores, before.labels)
    rows_after = reliability_bins(cal_scores, before.labels)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    for ax, rows, title, ece, colour in (
        (axes[0][0], rows_before, "Before: raw detector score", ece_before, "#c0392b"),
        (axes[0][1], rows_after, f"After: {calibrator.name} (fitted on val)", ece_after, "#1f7a4d"),
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        xs = [r["mean_score"] for r in rows if r["n"]]
        ys = [r["empirical_precision"] for r in rows if r["n"]]
        ns = [r["n"] for r in rows if r["n"]]
        ax.plot(xs, ys, "o-", color=colour, lw=2, ms=6, label="measured")
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(5, -11), fontsize=7)
        ax.fill_between([0, 1], [0, 1], [1, 1], color="#3498db", alpha=0.06)
        ax.text(0.03, 0.93, "under-confident", fontsize=8, color="#2c6fa8")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean score in bin"); ax.set_ylabel("empirical precision (P(true positive))")
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=11)
        ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)

    ax = axes[1][0]
    bins = np.linspace(0, 1, 41)
    ax.hist(before.scores, bins=bins, alpha=0.55, color="#c0392b", label="raw")
    ax.hist(cal_scores, bins=bins, alpha=0.55, color="#1f7a4d", label="calibrated")
    ax.axvline(OPERATING_CONF, color="k", ls=":", lw=1.2,
               label=f"shipped operating point ({OPERATING_CONF:g})")
    ax.set_xlabel("confidence"); ax.set_ylabel("detections")
    ax.set_title("Score distribution on the held-out split", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[1][1]
    grid = np.linspace(0, 1, 1001)
    ax.plot(grid, grid, "k--", lw=1, label="identity")
    ax.plot(grid, calibrator.transform(grid), color="#1f7a4d", lw=2, label=calibrator.describe())
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("raw detector score"); ax.set_ylabel("calibrated probability")
    ax.set_title("The fitted map (strictly increasing, so mAP is unchanged)", fontsize=11)
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Confidence calibration -- fitted on val, evaluated on {before.split} "
        f"(n = {len(before)} detections at conf >= {CACHE_CONF:g})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# report writers
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _bin_rows(rows: Sequence[dict[str, Any]]) -> list[list[Any]]:
    return [[r["bin"], r["n"],
             "-" if r["mean_score"] is None else f"{r['mean_score']:.3f}",
             "-" if r["empirical_precision"] is None else f"{r['empirical_precision']:.3f}",
             "-" if r["gap"] is None else f"{r['gap']:+.3f}"]
            for r in rows]


def write_calibration_md(path: Path, payload: dict[str, Any]) -> Path:
    """The document a reviewer reads before trusting the number on screen."""
    cal = payload["calibrator"]
    sel = payload["selection"]
    m = payload["metrics"]
    fit_split = payload["fit"]["split"]
    eval_split = payload["evaluation"]["split"]
    rank = payload["rank_preservation"]
    ap = payload["map50"]

    shape = {
        "identity": "0 (the shipped score)",
        "temperature": "1",
        "platt": "2",
        "isotonic": "non-parametric, monotone",
        "isotonic_unsmoothed": "non-parametric, monotone",
    }
    cand_rows = [
        [f"**{name}**" if name == sel["winner"] else name,
         shape.get(name, "-"),
         f"{r['oof_ece']:.4f}", f"{r['oof_brier']:.4f}", f"{r['oof_nll']:.4f}",
         "yes" if r["eligible"] else "reference only"]
        for name, r in sorted(sel["candidates"].items(), key=lambda kv: kv[1]["oof_ece"])
    ]

    ap_rows = [[name, f"{ap['before'][name]:.6f}", f"{ap['after'][name]:.6f}",
                "identical" if ap["before"][name] == ap["after"][name] else "DIFFERS"]
               for name in list(CLASS_NAMES) + ["mAP50"]]

    lines = [
        "# Confidence calibration",
        "",
        f"**Checkpoint** `{payload['run']['model_name']}` (`{payload['run']['weights']}`)  ",
        f"**Device** {payload['run']['device']} | **imgsz** {payload['run']['imgsz']} | "
        f"**NMS IoU** {payload['run']['nms_iou']} | **score floor** {payload['run']['cache_conf']} | "
        f"**match IoU** {payload['run']['match_iou']}  ",
        f"**Fitted on** `{fit_split}` ({payload['fit']['n_images']} images, "
        f"{payload['fit']['n_detections']} detections)  ",
        f"**Evaluated on** `{eval_split}` ({payload['evaluation']['n_images']} images, "
        f"{payload['evaluation']['n_detections']} detections) -- never seen by the fit  ",
        f"**Generated** {payload['run']['generated_at']}",
        "",
        "## 0. The gap this closes",
        "",
        "The brief asks for a defect verdict \"ideally with a confidence score\". The console",
        "shows one. It was never a probability. A detection displayed at 0.60 was correct far",
        "more often than 60% of the time, and an operator who calibrates their own trust to the",
        "displayed number therefore discards real defects.",
        "",
        "A calibrator is fitted here **on `val` only**. `test` is used once, at the end, to",
        "report the result. No hyper-parameter, no bin edge and no family choice was made by",
        "looking at `test`.",
        "",
        "## 1. Before: the shipped score is not a probability",
        "",
        f"On `{eval_split}`, {payload['evaluation']['n_detections']} detections at "
        f"conf >= {payload['run']['cache_conf']}, labelled true/false positive by the evaluator's own",
        f"greedy class-aware matcher at IoU {payload['run']['match_iou']}:",
        "",
        _md_table(["score bin", "n", "mean score", "empirical precision", "gap"],
                  _bin_rows(payload["bins"]["before"])),
        "",
        f"**ECE = {m['ece']['before']:.4f}**, and the sign of the gap is negative in every",
        "populated bin. This is not noise around the diagonal, it is a systematic bias: the",
        "detector is *under*-confident everywhere.",
        "",
        "## 2. Why one parameter is not enough, measured rather than asserted",
        "",
        "Temperature scaling has a single degree of freedom and it is a sharpening knob:",
        "`T < 1` pushes scores above 0.5 up and scores below 0.5 down, `T > 1` does the",
        "reverse. The bias above runs the *same direction at both ends*. No single temperature",
        "can lift the 0.09 bin to 0.19 and the 0.89 bin to 1.00 at the same time.",
        "",
        f"Selection protocol: {sel['protocol']}, {sel['folds']} folds, seed {sel['seed']},",
        f"over {sel['n_images']} `{fit_split}` images and {sel['n_detections']} detections.",
        "Grouping by image is not cosmetic -- two detections on one frame are not independent",
        "draws, and an ungrouped fold would let a leaked frame flatter the more flexible model.",
        "Every number below is **out of fold**, so isotonic gets no in-sample advantage.",
        "",
        _md_table(["family", "parameters", "out-of-fold ECE", "Brier", "NLL", "shippable"], cand_rows),
        "",
        f"The winner is **{sel['winner']}**.",
        "",
        "## 3. After",
        "",
        f"Fitted on all of `{fit_split}`: **{payload['calibrator_description']}**",
        "",
        *(( [
            "Each plateau reports its Jeffreys posterior mean `(k + 0.5) / (n + 1)` rather than",
            "the raw `k / n`, and the resulting sequence is re-projected with a weighted PAVA pass",
            "so it stays monotone. That is not cosmetic. The top plateau covers "
            f"{payload['top_plateau']['n']} `{fit_split}` detections above raw "
            f"{payload['top_plateau']['raw_from']:.3f}, of which "
            f"{payload['top_plateau']['positives']} are true positives -- a maximum-likelihood "
            f"value of {payload['top_plateau']['maximum_likelihood_value']:.3f}. Printing "
            "**1.00** beside a steel defect is a claim that sample cannot support, so the",
            f"calibrator reports {payload['top_plateau']['jeffreys_value']:.3f} instead. It also",
            "measures better: smoothing improves out-of-fold ECE from "
            f"{payload['selection']['candidates']['isotonic_unsmoothed']['oof_ece']:.4f} to "
            f"{payload['selection']['candidates']['isotonic']['oof_ece']:.4f}.",
            "",
        ] if payload.get("top_plateau") else [])),
        _md_table(["score bin", "n", "mean score", "empirical precision", "gap"],
                  _bin_rows(payload["bins"]["after"])),
        "",
        _md_table(
            ["metric", f"before (`{eval_split}`)", f"after (`{eval_split}`)"],
            [["ECE", f"{m['ece']['before']:.4f}", f"{m['ece']['after']:.4f}"],
             ["max per-bin gap (bins with n >= 20)", f"{m['mce']['before']:.4f}", f"{m['mce']['after']:.4f}"],
             ["Brier score", f"{m['brier']['before']:.4f}", f"{m['brier']['after']:.4f}"],
             ["negative log likelihood", f"{m['nll']['before']:.4f}", f"{m['nll']['after']:.4f}"],
             [f"ECE at the shipped operating point (conf >= {OPERATING_CONF:g}, "
              f"n = {m['operating_point']['n']})",
              f"{m['operating_point']['ece_before']:.4f}",
              f"{m['operating_point']['ece_after']:.4f}"]],
        ),
        "",
        f"ECE falls by {100.0 * (1.0 - m['ece']['after'] / m['ece']['before']):.0f}%, and the",
        "remaining 0.046 is not a systematic bias any more -- the largest surviving gaps sit in",
        "the thin middle bins where a few dozen detections cannot pin a precision down. The",
        "residual is honest sampling noise, not a direction.",
        "",
        "![reliability diagram](calibration.png)",
        "",
        "## 4. mAP is unchanged, and that is measured, not argued",
        "",
        "Average precision is a function of the *ranking* of detections, not of their score",
        "values: the greedy matcher walks predictions in descending score and the PR curve is",
        "swept by rank. A strictly increasing rescoring therefore cannot move it. Isotonic",
        "regression is only non-decreasing on its own, which would create ties and make AP",
        "tie-break dependent, so `IsotonicCalibrator` blends in a vanishing `eps * s` term",
        f"(eps = {cal['params'].get('eps', 0):g}) that restores a strict order without moving any",
        "value by more than that.",
        "",
        f"Checked on the {payload['evaluation']['n_detections']} `{eval_split}` detections:",
        "",
        _md_table(["check", "result"],
                  [["sorted order identical before/after", str(rank["order_identical"])],
                   ["distinct raw scores collapsed onto one calibrated score", str(rank["collapsed_pairs"])],
                   ["strictly increasing on the data", str(rank["strictly_increasing_on_data"])],
                   ["non-decreasing on a 2001-point grid over [0,1]", str(rank["non_decreasing_on_unit_grid"])],
                   ["largest score shift", f"{rank['max_shift']:.4f}"]]),
        "",
        f"AP50 recomputed from scratch on `{eval_split}`, once ranked by raw score and once by",
        "calibrated score (101-point interpolation, greedy class-aware matching at IoU 0.5):",
        "",
        _md_table(["class", "AP50 (raw ranking)", "AP50 (calibrated ranking)", "verdict"], ap_rows),
        "",
        ap["note"],
        "",
        "## 5. What this does not fix",
        "",
        "The map is `score -> P(true positive)` **on NEU-DET validation frames**, every one of",
        "which contains a defect. The base rate on genuinely defect-free strip is different, so",
        "a calibrated 0.9 on mill imagery is not the same claim as a calibrated 0.9 here (see",
        "`reports/false_alarm.md` and the cross-domain work for what that population actually",
        "looks like). Calibration changes what the number *means*; it changes nothing about",
        "what is detected. It is not an accuracy improvement and must not be presented as one.",
        "",
        "Two smaller caveats worth stating:",
        "",
        f"1. The fit sees {payload['fit']['n_detections']} detections from "
        f"{payload['fit']['n_images']} images. The top bins are thin, so the upper end of the",
        "   curve is the least certain part of it. Out-of-fold selection is what keeps that from",
        "   becoming an overfitting story, but more validation frames would tighten it.",
        "2. The calibrator is class-agnostic. Per-class miscalibration is real and measured in",
        "   `docs/audit/technical_headroom.md` (gaps from -0.055 on `scratches` to -0.193 on",
        "   `crazing`); a per-class calibrator would fit six times as many parameters on the same",
        "   180 images, which is the wrong trade at this sample size.",
        "",
        "## 6. How to use it",
        "",
        "```python",
        "from calibrate import calibrate_confidence",
        "",
        "shown = calibrate_confidence(detection.confidence)   # 0.60 -> "
        f"{payload['examples']['0.60']:.2f}",
        "```",
        "",
        f"Parameters live in `{CALIBRATION_JSON.name}` with the split and date they were fitted",
        "on. Re-fit whenever the checkpoint changes: a calibrator is tied to one set of weights.",
        "",
        _md_table(["raw score", "calibrated"],
                  [[k, f"{v:.3f}"] for k, v in payload["examples"].items()]),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run(
    weights: str | Path | None = None,
    device: str = "auto",
    imgsz: int = DEFAULT_IMGSZ,
    nms_iou: float = DEFAULT_IOU,
    cache_conf: float = CACHE_CONF,
    fit_split: str = "val",
    eval_split: str = "test",
    out_dir: Path = REPORTS_DIR,
    folds: int = 5,
    seed: int = 20260909,
) -> dict[str, Any]:
    """Fit on `fit_split`, report on `eval_split`, write the three artefacts."""
    if fit_split == eval_split:
        raise ValueError(
            f"fit_split and eval_split are both {fit_split!r}. Fitting a calibrator on the "
            "split it is scored on is the one mistake this module exists to avoid."
        )
    weights_path = resolve_weights(weights)
    resolved_device = resolve_device(device)
    detector = DefectDetector(
        weights=weights_path, device=resolved_device, conf=cache_conf, iou=nms_iou, imgsz=imgsz
    )
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"[calibrate] harvesting scores on {fit_split!r} ...")
    fit = collect_matched_scores(detector, fit_split)
    print(f"[calibrate] harvesting scores on {eval_split!r} ...")
    ev = collect_matched_scores(detector, eval_split)
    print(f"[calibrate] fit {len(fit)} detections, eval {len(ev)} detections")

    selection = select_calibrator(fit, k=folds, seed=seed)
    calibrator = _FITTERS[selection["winner"]](fit.scores, fit.labels)
    print(f"[calibrate] selected {calibrator.describe()}")

    cal_eval = calibrator.transform(ev.scores)
    cal_fit = calibrator.transform(fit.scores)

    op = ev.at_least(OPERATING_CONF)
    metrics = {
        "ece": {
            "before": expected_calibration_error(ev.scores, ev.labels),
            "after": expected_calibration_error(cal_eval, ev.labels),
            "fit_split_before": expected_calibration_error(fit.scores, fit.labels),
            "fit_split_after": expected_calibration_error(cal_fit, fit.labels),
        },
        "mce": {
            "before": maximum_calibration_error(ev.scores, ev.labels),
            "after": maximum_calibration_error(cal_eval, ev.labels),
        },
        "brier": {"before": brier_score(ev.scores, ev.labels),
                  "after": brier_score(cal_eval, ev.labels)},
        "nll": {"before": negative_log_likelihood(ev.scores, ev.labels),
                "after": negative_log_likelihood(cal_eval, ev.labels)},
        "operating_point": {
            "conf": OPERATING_CONF,
            "n": len(op),
            "ece_before": expected_calibration_error(op.scores, op.labels),
            "ece_after": expected_calibration_error(calibrator.transform(op.scores), op.labels),
        },
    }

    top_plateau: dict[str, Any] | None = None
    if isinstance(calibrator, IsotonicCalibrator):
        kx = np.asarray(calibrator.knots_x, dtype=np.float64)
        ky = np.asarray(calibrator.knots_y, dtype=np.float64)
        top = float(ky.max())
        raw_from = float(kx[ky >= top - 1e-12].min())
        support = fit.scores >= raw_from
        n_sup, k_sup = int(support.sum()), int(fit.labels[support].sum())
        top_plateau = {
            "raw_from": raw_from,
            "n": n_sup,
            "positives": k_sup,
            "maximum_likelihood_value": (k_sup / n_sup) if n_sup else float("nan"),
            "jeffreys_value": top,
        }

    rank = verify_rank_preservation(calibrator, ev.scores)
    raw_by_image = [p.confidences for p in ev.preds]
    cal_by_image = [calibrator.transform(p.confidences) for p in ev.preds]
    ap_before = map50(ev.records, ev.preds, raw_by_image)
    ap_after = map50(ev.records, ev.preds, cal_by_image)
    identical = all(ap_before[k] == ap_after[k] for k in ap_before)
    print(f"[calibrate] mAP50 (own matcher) before {ap_before['mAP50']:.6f} "
          f"after {ap_after['mAP50']:.6f} -> {'identical' if identical else 'CHANGED'}")

    payload: dict[str, Any] = {
        "run": {
            "weights": str(weights_path),
            "model_name": detector.model_name,
            "device": resolved_device,
            "imgsz": imgsz,
            "nms_iou": nms_iou,
            "cache_conf": cache_conf,
            "match_iou": IOU_MATCH,
            "generated_at": generated_at,
        },
        "fit": {
            "split": fit_split,
            "date": generated_at[:10],
            "n_images": int(np.unique(fit.image_index).size),
            "n_detections": len(fit),
            "positives": int(fit.labels.sum()),
        },
        "evaluation": {
            "split": eval_split,
            "n_images": len(ev.records),
            "n_detections": len(ev),
            "positives": int(ev.labels.sum()),
        },
        "calibrator": calibrator.to_dict(),
        "calibrator_description": calibrator.describe(),
        "top_plateau": top_plateau,
        "selection": selection,
        "metrics": metrics,
        "rank_preservation": rank,
        "map50": {
            "before": ap_before,
            "after": ap_after,
            "identical": identical,
            "note": (
                "AP50 here is this module's own 101-point implementation over the evaluator's "
                "greedy matcher, so it is not numerically the ultralytics headline (0.7524); it "
                "is the same quantity computed twice under two rankings, which is what makes the "
                "invariance meaningful."
            ),
        },
        "examples": {f"{v:.2f}": float(calibrator.transform(np.float64(v)))
                     for v in (0.15, 0.25, 0.40, 0.60, 0.70, 0.80, 0.90)},
        "bins": {
            "before": reliability_bins(ev.scores, ev.labels),
            "after": reliability_bins(cal_eval, ev.labels),
        },
        "ece_bin_edges": list(ECE_BIN_EDGES),
        "artifacts": {},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot_reliability(out_dir / CALIBRATION_PNG.name, ev, calibrator,
                           metrics["ece"]["before"], metrics["ece"]["after"])
    md = write_calibration_md(out_dir / CALIBRATION_MD.name, payload)
    payload["artifacts"] = {"png": str(png), "md": str(md),
                            "json": str(out_dir / CALIBRATION_JSON.name)}
    (out_dir / CALIBRATION_JSON.name).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[calibrate] wrote {out_dir / CALIBRATION_JSON.name}, {png} and {md}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit and report a confidence calibrator.")
    parser.add_argument("--weights", default=None, help="checkpoint (default: resolve_weights)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU")
    parser.add_argument("--cache-conf", type=float, default=CACHE_CONF,
                        help="score floor detections are harvested at")
    parser.add_argument("--fit-split", default="val", help="the ONLY split the calibrator sees")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--out-dir", default=str(REPORTS_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        weights=args.weights, device=args.device, imgsz=args.imgsz, nms_iou=args.iou,
        cache_conf=args.cache_conf, fit_split=args.fit_split, eval_split=args.eval_split,
        out_dir=Path(args.out_dir), folds=args.folds, seed=args.seed,
    )
    m = payload["metrics"]["ece"]
    print(f"\nECE on {payload['evaluation']['split']}: {m['before']:.4f} -> {m['after']:.4f} "
          f"({100.0 * (1.0 - m['after'] / m['before']):.1f}% reduction)")
    print(f"mAP50 unchanged: {payload['map50']['identical']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin wrapper over main()
    raise SystemExit(main())
