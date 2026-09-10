"""A plausibility gate that runs before the verdict: is this frame even strip imagery?

The detector is a six-class defect head. It has no seventh class for "not steel",
so it answers *every* image, and the audit measured what that costs: a company
logo scores ``scratches 0.47``; a plain white frame scores ``pitted_surface
0.739, severity 86.3 critical``; a cartoon landscape gives nine detections at
severity 97.2 and four cards of corrective actions. Flat grey and Gaussian noise,
by contrast, give nothing at all -- so the failure is specific: the model fires on
*unfamiliar texture that resembles steel*, not on anything at all.

This module answers a narrower question than "is this in distribution", because
that question is not cheaply answerable and pretending otherwise would be the
same overclaim. It answers: **does this frame have the physical properties every
strip capture has?** Six checks, each one number against one threshold, each with
a sentence an operator can read:

===============  ==========================================================
resolution       shorter side must be at least 64 px
aspect           longest:shortest side must not exceed 64:1
colour           mean HSV saturation must be under 40/255 -- mill imagery is
                 monochrome
noise            fewer than 70 % of pixels may be pixel-level edges
focus            Laplacian variance of the 256 px-normalised luminance must
                 reach 1.5
tonal range      the unclipped grey histogram must carry at least 1.8 bits,
                 i.e. more than about 3.5 effective grey levels
===============  ==========================================================

A frame is REJECTED if any check fails; the failing checks *are* the explanation.
No classifier is trained, nothing is fitted, and every threshold is a constant a
mill engineer can read off the table in ``reports/ood_guard.md`` and change.

**Measured** (protocol and full tables in ``reports/ood_guard.md``):

* **0 false rejections out of 3,200 genuine steel frames** -- all 1,800 NEU-DET
  frames (train + val + test) and all 1,400 real Severstal strip frames.
* **18/18 of the tuning negative suite rejected**, and -- the number that
  matters, because the tuning suite was used to pick thresholds --
  **13/15 of a held-out negative suite built afterwards with a different seed
  and different constructions**, with no threshold changed.
* The two held-out misses are both *greyscale textured* images (a sinusoidal
  wood grain and a radial gradient). This is the honest limit: **the gate tests
  physics, not semantics.** Anything monochrome, in focus and textured passes,
  because on these six signals it is indistinguishable from rough strip. Closing
  that needs a texture model or real negatives, not another threshold.

One signal is deliberately **reported but not allowed to gate**: the distance
from the NEU-DET training grey histogram. It cannot separate -- real steel spans
4.3-126.1 grey levels of distance and a blank white frame scores 125.8, inside
that range (see ``domain_shift.histogram_distance``). It is instead used for the
third decision, ``REVIEW``: a frame that passes all six checks but sits outside
the exposure range the model was trained on is worth scoring *and* worth
flagging, and is exactly the case ``domain_shift.match_to_reference`` exists for.

Depends only on numpy, OpenCV and ``domain_shift``. No torch: the gate must be
callable from a data-preparation script, and it must never be the reason a
console is slow to say "no". Measured cost is ~1 ms on a 200x200 frame.

Wiring (the console owns the UI; this module owns the decision)::

    from ood_guard import inspect

    verdict = inspect(frame)
    if not verdict.ok:
        show_banner(verdict.headline, verdict.reason)   # and do NOT run the model
    else:
        result = detector.predict(frame)
        if verdict.review:
            show_note(verdict.reason)

CLI::

    .venv/bin/python src/ood_guard.py path/to/frame.png
    .venv/bin/python src/ood_guard.py --calibrate            # margins on real steel
    .venv/bin/python src/ood_guard.py --suite reports/ood_guard.json
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/ood_guard.py` from the Jindal Stainless surface-defect
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from domain_shift import ReferenceStats, _to_rgb, load_reference

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Decisions, worst first. `REVIEW` still runs the model; `REJECT` does not.
REJECT = "reject"
REVIEW = "review"
PASS = "pass"

# Longest side the luminance plane is reduced to before the focus statistic is
# taken. 256 is DEFAULT_IMGSZ: focus has to be judged at the scale the detector
# actually sees, or a 4x upscale of a sharp frame reads as sharper than it is.
FOCUS_NORM_PX = 256

# Pixels at or beyond these values carry no information about the surface -- the
# sensor clipped. They are excluded from the tonal-range statistic, which is the
# whole reason that statistic works: a 92 %-blown-out but genuine Severstal frame
# and a white-background logo have nearly the same raw histogram entropy (1.75
# vs 1.60 bits, a 9 % gap), and 2.08 vs 1.60 bits once the clipped mass is
# dropped (a 30 % gap, and the one real frame below 2.5 bits is an outlier -- the
# next lowest of 3,200 sits above it).
CLIP_LOW = 2
CLIP_HIGH = 253

# Laplacian magnitude above which a pixel counts as a "pixel-level edge" for the
# noise check. Swept over 4/8/16/24/32 on 3,200 real frames against three noise
# images: the separation widens monotonically (1.05x -> 1.56x), so 32 is the
# usable end of the sweep, not a taste.
NOISE_LAPLACIAN = 32.0


@dataclass(frozen=True)
class GuardThresholds:
    """Every constant the gate uses, in one auditable place.

    Each default is quoted with the measurement that set it: the extreme value
    over 3,200 genuine steel frames (1,800 NEU-DET + 1,400 real Severstal), and
    the extreme value over the negatives the threshold has to catch. A mill
    commissioning a new camera should re-run ``--calibrate`` on a few hundred of
    its own frames and widen anything whose margin has closed.
    """

    # Below 64 px the detector's 256 px input is more than 4x upsampling: there
    # is no evidence left to detect from. Real frames: 200 px (NEU-DET),
    # 256 px (Severstal). Negatives: 8, 10, 40 px.
    min_side_px: int = 64

    # A strip frame can legitimately be very wide -- Severstal is 6.25:1 and a
    # line-scan camera feeding `predict_tiled` is wider still. This is a
    # backstop against degenerate slices only. Real frames: 6.25 max.
    max_aspect: float = 64.0

    # Mill strip imagery is monochrome. Measured mean HSV saturation is exactly
    # 0.00/255 on every one of the 3,200 real frames; a deliberately warm-cast
    # steel frame (a plausible colour-camera artefact) reads 28.9. Colour
    # negatives read 118-255. 40 sits between with 1.4x margin below and 2.9x
    # above -- the tightest colour margin, and it is against a synthetic case.
    max_saturation: float = 40.0

    # Fraction of pixels whose |Laplacian| exceeds NOISE_LAPLACIAN. Real frames
    # reach 0.538 (p99 0.519); Gaussian and uniform noise reach 0.840-0.895.
    max_edge_density: float = 0.70

    # Laplacian variance of the 256 px-normalised luminance. Real frames: 7.76
    # minimum over 3,200. Blur negatives: 0.0-1.1. A synthetic 15-tap motion
    # blur of a real frame -- which *should* pass -- sits at 2.3, so this is the
    # narrowest gate in the module and the one with a genuine grey zone.
    min_focus: float = 1.5

    # Shannon entropy (bits) of the grey histogram over unclipped pixels only.
    # 1.8 bits is about 3.5 effective grey levels. Real frames: 2.08 minimum,
    # and only 1 of 3,200 sits below 2.5. Flat-graphic negatives: 0.00-1.61.
    min_tonal_bits: float = 1.8

    # REVIEW, not reject. The NEU-DET *training* split spans 5.7-116.5 grey
    # levels of Wasserstein-1 distance from its own pooled histogram; beyond
    # that maximum the frame's exposure is outside anything the model saw.
    # Held-out NEU-DET test never reaches it (max 111.0); real Severstal
    # exceeds it often, which is the correct answer.
    review_hist_distance: float = 116.5

    def to_dict(self) -> dict[str, float]:
        return {
            "min_side_px": float(self.min_side_px),
            "max_aspect": self.max_aspect,
            "max_saturation": self.max_saturation,
            "max_edge_density": self.max_edge_density,
            "min_focus": self.min_focus,
            "min_tonal_bits": self.min_tonal_bits,
            "review_hist_distance": self.review_hist_distance,
        }


DEFAULT_THRESHOLDS = GuardThresholds()


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FrameSignals:
    """The raw measurements, before any threshold is applied.

    Kept separate from the decision on purpose: a console can plot these over a
    shift and watch a camera drift towards a threshold long before it trips one.
    """

    height: int
    width: int
    min_side: int
    aspect: float
    saturation: float
    edge_density: float
    focus: float
    tonal_bits: float
    grey_mean: float
    grey_std: float
    clipped_fraction: float
    hist_distance: float

    def to_dict(self) -> dict[str, float]:
        return {
            "height": self.height,
            "width": self.width,
            "min_side": self.min_side,
            "aspect": round(self.aspect, 2),
            "saturation": round(self.saturation, 2),
            "edge_density": round(self.edge_density, 4),
            "focus": round(self.focus, 2),
            "tonal_bits": round(self.tonal_bits, 3),
            "grey_mean": round(self.grey_mean, 1),
            "grey_std": round(self.grey_std, 1),
            "clipped_fraction": round(self.clipped_fraction, 4),
            "hist_distance": round(self.hist_distance, 1),
        }


def _focus_plane(grey: np.ndarray) -> np.ndarray:
    """Downscale-only to FOCUS_NORM_PX on the longest side.

    Never upscales: interpolating a small frame up adds no detail and would
    inflate the variance of an already-soft capture.
    """
    height, width = grey.shape[:2]
    scale = FOCUS_NORM_PX / max(height, width)
    if scale >= 1.0:
        return grey
    return cv2.resize(
        grey,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _tonal_bits_from_counts(counts: np.ndarray) -> float:
    """Shared implementation, so the full histogram is only ever built once."""
    # Levels CLIP_LOW+1 .. CLIP_HIGH-1 are exactly the pixels satisfying
    # `grey > CLIP_LOW and grey < CLIP_HIGH`, so slicing the histogram is
    # identical to masking the pixels, and does not touch the image again.
    unclipped = np.asarray(counts, dtype=np.float64)[CLIP_LOW + 1 : CLIP_HIGH]
    total = float(unclipped.sum())
    if total <= 0.0:
        # Every pixel is crushed or blown. Zero bits is the literal truth.
        return 0.0
    probabilities = unclipped / total
    nonzero = probabilities[probabilities > 0.0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def tonal_bits(grey: np.ndarray) -> float:
    """Entropy in bits of the grey histogram, counting unclipped pixels only.

    ``2 ** tonal_bits`` is the effective number of grey levels the surface is
    described with, which is the form to quote to an operator: genuine strip runs
    at 70-150 effective levels, a printed logo at 3.

    Clipped pixels are excluded because they are the one part of the frame that
    carries no surface information, and because including them makes a
    blown-out-but-real frame look like a graphic. Measured on the two extremes
    that matter: a 92 %-blown-out but genuine Severstal frame and a
    white-background logo sit at 1.75 vs 1.60 bits on the raw histogram (a 9 %
    gap, unusable) and 2.08 vs 1.60 once the clipped mass is dropped.
    """
    return _tonal_bits_from_counts(
        np.bincount(np.asarray(grey, dtype=np.uint8).ravel(), minlength=256)
    )


def frame_signals(
    image: str | Path | np.ndarray | Any, reference: ReferenceStats | None = None
) -> FrameSignals:
    """Measure all six gating signals plus the reported-only diagnostics. ~1 ms."""
    ref = reference if reference is not None else load_reference()
    rgb = _to_rgb(image)
    height, width = rgb.shape[:2]
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    plane = _focus_plane(grey)
    # CV_16S, not CV_64F: the Laplacian of a uint8 plane is integer-valued and
    # well inside int16, so the thresholded count is bit-identical (verified on
    # 633 frames, 0 differ) at 1.8x the speed on a 1600x256 frame.
    laplacian = cv2.Laplacian(grey, cv2.CV_16S)
    counts = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(counts / counts.sum())
    return FrameSignals(
        height=int(height),
        width=int(width),
        min_side=int(min(height, width)),
        aspect=float(max(height, width) / max(1, min(height, width))),
        saturation=float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].mean()),
        edge_density=float((np.abs(laplacian) > NOISE_LAPLACIAN).mean()),
        focus=float(cv2.Laplacian(plane, cv2.CV_64F).var()),
        tonal_bits=_tonal_bits_from_counts(counts),
        grey_mean=float(grey.mean()),
        grey_std=float(grey.std()),
        clipped_fraction=float(((grey <= CLIP_LOW) | (grey >= CLIP_HIGH)).mean()),
        hist_distance=float(np.abs(cdf - ref.cdf).sum()),
    )


# --------------------------------------------------------------------------- #
# checks and verdict
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Check:
    """One named signal against one threshold, with the sentence that explains it."""

    name: str
    value: float
    threshold: float
    comparison: str  # ">=" means "value must be >= threshold to pass"
    passed: bool
    units: str
    explanation: str

    @property
    def margin(self) -> float:
        """How far the value sits on the passing side, as a ratio. >1 means safe."""
        if self.comparison == ">=":
            return float(self.value / self.threshold) if self.threshold else float("inf")
        return float(self.threshold / self.value) if self.value else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "comparison": self.comparison,
            "passed": self.passed,
            "units": self.units,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class GuardVerdict:
    """What the gate decided, why, and everything needed to argue with it."""

    decision: str
    signals: FrameSignals
    checks: tuple[Check, ...]
    thresholds: GuardThresholds = field(default=DEFAULT_THRESHOLDS)

    @property
    def ok(self) -> bool:
        """True when the detector should be run at all."""
        return self.decision != REJECT

    @property
    def review(self) -> bool:
        """True when the frame should be scored but the verdict carries a caveat."""
        return self.decision == REVIEW

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def headline(self) -> str:
        """Six words for a banner."""
        if self.decision == REJECT:
            return "Not a strip capture - verdict withheld"
        if self.decision == REVIEW:
            return "Outside the trained exposure range"
        return "Frame accepted as strip imagery"

    @property
    def reason(self) -> str:
        """One line an operator can act on."""
        failed = self.failed
        if failed:
            return (
                "The detector was not run: "
                + "; ".join(c.explanation for c in failed)
                + ". A six-class defect model has no 'not steel' answer, so any "
                "verdict on this image would be meaningless rather than wrong."
            )
        if self.decision == REVIEW:
            return (
                f"Scored, with a caveat: this frame's grey histogram sits "
                f"{self.signals.hist_distance:.0f} grey levels from the NEU-DET training "
                f"distribution, past the {self.thresholds.review_hist_distance:.0f} that "
                "split spans. The exposure is outside what the model was trained on, so "
                "confidences are less trustworthy than the held-out numbers suggest; "
                "domain_shift.match_to_reference is the commissioning fix."
            )
        return (
            f"Monochrome ({self.signals.saturation:.0f}/255 saturation), in focus "
            f"(Laplacian variance {self.signals.focus:.0f}), "
            f"{2 ** self.signals.tonal_bits:.0f} effective grey levels, "
            f"{self.signals.min_side} px shortest side. Consistent with strip imagery."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "ok": self.ok,
            "review": self.review,
            "headline": self.headline,
            "reason": self.reason,
            "failed": [c.name for c in self.failed],
            "signals": self.signals.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "thresholds": self.thresholds.to_dict(),
        }


def _build_checks(signals: FrameSignals, t: GuardThresholds) -> tuple[Check, ...]:
    """The whole rule set, in the order a person would ask the questions."""
    return (
        Check(
            name="resolution",
            value=float(signals.min_side),
            threshold=float(t.min_side_px),
            comparison=">=",
            passed=signals.min_side >= t.min_side_px,
            units="px",
            explanation=(
                f"the shortest side is {signals.min_side} px, under the {t.min_side_px} px "
                "floor, so scoring it at 256 px would be more than 4x upsampling"
            ),
        ),
        Check(
            name="aspect",
            value=signals.aspect,
            threshold=t.max_aspect,
            comparison="<=",
            passed=signals.aspect <= t.max_aspect,
            units=":1",
            explanation=(
                f"the frame is {signals.aspect:.0f}:1, past the {t.max_aspect:.0f}:1 limit; "
                "that is a degenerate slice, not a strip frame"
            ),
        ),
        Check(
            name="colour",
            value=signals.saturation,
            threshold=t.max_saturation,
            comparison="<=",
            passed=signals.saturation <= t.max_saturation,
            units="/255",
            explanation=(
                f"mean colour saturation is {signals.saturation:.0f}/255 against a limit of "
                f"{t.max_saturation:.0f}; mill strip imagery is monochrome (measured 0/255 "
                "on all 3,200 reference frames)"
            ),
        ),
        Check(
            name="noise",
            value=signals.edge_density,
            threshold=t.max_edge_density,
            comparison="<=",
            passed=signals.edge_density <= t.max_edge_density,
            units="fraction",
            explanation=(
                f"{signals.edge_density * 100:.0f}% of pixels are pixel-level edges, past the "
                f"{t.max_edge_density * 100:.0f}% limit; that is sensor noise or a synthetic "
                "pattern, not surface texture"
            ),
        ),
        Check(
            name="focus",
            value=signals.focus,
            threshold=t.min_focus,
            comparison=">=",
            passed=signals.focus >= t.min_focus,
            units="Laplacian variance",
            explanation=(
                f"Laplacian variance is {signals.focus:.1f} against a floor of {t.min_focus:.1f}; "
                "the frame is out of focus, blank or a smooth gradient, and carries no "
                "surface detail to judge"
            ),
        ),
        Check(
            name="tonal_range",
            value=signals.tonal_bits,
            threshold=t.min_tonal_bits,
            comparison=">=",
            passed=signals.tonal_bits >= t.min_tonal_bits,
            units="bits",
            explanation=(
                f"the unclipped image uses {2 ** signals.tonal_bits:.1f} effective grey levels "
                f"({signals.tonal_bits:.2f} bits, floor {t.min_tonal_bits:.1f}); printed "
                "graphics, logos and screenshots look like this, steel does not"
            ),
        ),
    )


def inspect(
    image: str | Path | np.ndarray | Any,
    *,
    thresholds: GuardThresholds = DEFAULT_THRESHOLDS,
    reference: ReferenceStats | None = None,
) -> GuardVerdict:
    """Decide whether this frame is worth showing the detector.

    Args:
        image: path, PIL image or HWC array -- the same set `DefectDetector`
            accepts, so a console can call this on exactly what it is about to
            score, before any normalisation.
        thresholds: override any constant; the defaults are the measured ones.
        reference: NEU-DET training distribution, for the reported histogram
            distance. Defaults to the cached one.

    Returns:
        A `GuardVerdict`. Branch on `.ok` (run the model or not) and `.review`
        (run it, but caveat the answer); render `.headline` and `.reason`.

    Raises:
        FileNotFoundError, ValueError, TypeError: only from decoding the input.
        A frame that decodes always gets a verdict -- the gate never throws
        because it dislikes an image.
    """
    signals = frame_signals(image, reference)
    checks = _build_checks(signals, thresholds)
    if any(not c.passed for c in checks):
        decision = REJECT
    elif signals.hist_distance > thresholds.review_hist_distance:
        decision = REVIEW
    else:
        decision = PASS
    return GuardVerdict(decision=decision, signals=signals, checks=checks, thresholds=thresholds)


def is_steel_like(
    image: str | Path | np.ndarray | Any, *, thresholds: GuardThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """`inspect(...).ok`, for callers that only want the boolean."""
    return inspect(image, thresholds=thresholds).ok


# --------------------------------------------------------------------------- #
# calibration on a new camera
# --------------------------------------------------------------------------- #


def calibrate(
    images: Sequence[str | Path],
    *,
    thresholds: GuardThresholds = DEFAULT_THRESHOLDS,
    reference: ReferenceStats | None = None,
) -> dict[str, Any]:
    """Report how much headroom a set of known-good frames has on every check.

    Commissioning tool. Point it at a few hundred frames from a camera you know
    are good strip; anything with `worst_margin` near 1.0 is a threshold that
    will start rejecting real product, and anything with a rejection is a
    threshold to widen before the line runs.
    """
    ref = reference if reference is not None else load_reference()
    rows = [inspect(p, thresholds=thresholds, reference=ref) for p in images]
    if not rows:
        raise ValueError("calibrate() needs at least one image")
    names = [c.name for c in rows[0].checks]
    out: dict[str, Any] = {
        "n_images": len(rows),
        "rejected": sum(1 for r in rows if r.decision == REJECT),
        "review": sum(1 for r in rows if r.decision == REVIEW),
        "checks": {},
    }
    for i, name in enumerate(names):
        values = np.array([r.checks[i].value for r in rows], dtype=np.float64)
        margins = np.array([r.checks[i].margin for r in rows], dtype=np.float64)
        check = rows[0].checks[i]
        out["checks"][name] = {
            "threshold": check.threshold,
            "comparison": check.comparison,
            # The extreme that matters is the one nearest the threshold.
            "worst_value": float(values.min() if check.comparison == ">=" else values.max()),
            "p01": float(np.percentile(values, 1)),
            "p50": float(np.median(values)),
            "p99": float(np.percentile(values, 99)),
            "worst_margin": float(np.min(margins)),
            "n_failed": int(sum(1 for r in rows if not r.checks[i].passed)),
        }
    hist = np.array([r.signals.hist_distance for r in rows], dtype=np.float64)
    out["hist_distance"] = {
        "min": float(hist.min()),
        "p50": float(np.median(hist)),
        "max": float(hist.max()),
        "review_threshold": thresholds.review_hist_distance,
    }
    return out


def suggest_thresholds(
    images: Sequence[str | Path],
    *,
    safety: float = 2.0,
    base: GuardThresholds = DEFAULT_THRESHOLDS,
    reference: ReferenceStats | None = None,
) -> GuardThresholds:
    """Loosen -- never tighten -- the four texture thresholds for a new camera.

    Takes the worst value each check reaches on the supplied known-good frames
    and backs off by `safety`. Only ever moves a threshold in the permissive
    direction, because the failure this module exists to avoid is rejecting real
    steel; a camera whose frames are *cleaner* than NEU-DET does not earn a
    tighter gate on this evidence alone.
    """
    if safety < 1.0:
        raise ValueError("safety must be >= 1.0; a safety factor below 1 tightens the gate")
    stats = calibrate(images, thresholds=base, reference=reference)["checks"]
    return replace(
        base,
        max_saturation=max(base.max_saturation, stats["colour"]["worst_value"] * safety),
        max_edge_density=min(
            1.0, max(base.max_edge_density, stats["noise"]["worst_value"] * safety)
        ),
        min_focus=min(base.min_focus, stats["focus"]["worst_value"] / safety),
        min_tonal_bits=min(base.min_tonal_bits, stats["tonal_range"]["worst_value"] / safety),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(verdict: GuardVerdict, label: str) -> str:
    mark = {REJECT: "REJECT", REVIEW: "REVIEW", PASS: "PASS  "}[verdict.decision]
    lines = [f"{mark}  {label}", f"        {verdict.reason}"]
    for check in verdict.checks:
        flag = "ok  " if check.passed else "FAIL"
        lines.append(
            f"        [{flag}] {check.name:12s} {check.value:10.3f} {check.comparison} "
            f"{check.threshold:<8.3g} ({check.units})"
        )
    lines.append(f"        [    ] hist_distance {verdict.signals.hist_distance:8.1f} (reported only)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("images", nargs="*", type=Path, help="frames to inspect")
    parser.add_argument(
        "--calibrate",
        type=Path,
        nargs="*",
        help="directories of known-good frames; report headroom on every check",
    )
    parser.add_argument("--json", type=Path, help="write the machine-readable result here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload: dict[str, Any] = {"thresholds": DEFAULT_THRESHOLDS.to_dict()}

    if args.calibrate is not None:
        paths: list[Path] = []
        for root in args.calibrate:
            paths.extend(
                sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".jpg", ".png"})
            )
        if not paths:
            print("no images found for --calibrate")
            return 1
        report = calibrate(paths)
        payload["calibration"] = report
        print(f"calibration over {report['n_images']} frames: "
              f"{report['rejected']} rejected, {report['review']} review")
        for name, row in report["checks"].items():
            print(
                f"  {name:12s} threshold {row['threshold']:<8.3g} worst {row['worst_value']:10.3f} "
                f"margin {row['worst_margin']:6.2f}x  failed {row['n_failed']}"
            )

    verdicts = []
    for path in args.images:
        verdict = inspect(path)
        verdicts.append({"image": str(path), **verdict.to_dict()})
        print(_format(verdict, str(path)))
    if verdicts:
        payload["images"] = verdicts

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwritten {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
