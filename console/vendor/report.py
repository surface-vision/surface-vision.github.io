"""Per-coil inspection report for the Jindal Stainless surface-defect detector.

A single frame verdict is not a decision. What the finishing line acts on is the
*coil*: several hundred frames scored together, rolled up into a defect rate, a
per-class breakdown and one disposition - ACCEPT, DOWNGRADE or HOLD.

This module runs the detector over a list of frames, aggregates them, applies a
configurable rule set, and renders two artefacts:

* `reports/coil_report.html` - fully self-contained (inline CSS, base64 PNGs, no
  external requests), so it can be attached to an email or archived against the coil
  record and still render years later,
* `reports/coil_report.json` - the same numbers without the pictures, for the MES /
  quality database.

The disposition rules live in `DispositionRules` rather than in the reporting code,
because they are a commercial decision that the quality department owns and will
re-tune. Every rule that fired is written into the report, so the reason a coil was
held is auditable.

Run it directly:

    python src/report.py --images data/neu-det/test/images --limit 60 --coil-id C-1042
"""

# ---------------------------------------------------------------------------
# VENDORED COPY -- do not edit here.
#
# This file is a copy of `src/report.py` from the Jindal Stainless surface-defect
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
import base64
import glob as globlib
import html
import json
import platform
import random
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import (  # noqa: E402  (path bootstrap must run first)
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_IMGSZ,
    DEFECT_INFO,
    SEVERITY_BANDS,
    DefectDetector,
    InferenceResult,
    # Both deliberate: the report must band scores with the engine's own bands and
    # must see exactly the pixels the detector saw.
    _severity_bucket,
    _to_rgb,
    load_detector,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports"

# Embedded-figure sizing. A NEU-DET crop is only 200 px, so figures are enlarged for
# legibility, but not without bound - and a mill-resolution frame is shrunk, so the
# report stays small enough to email.
_MAX_FIGURE_UPSCALE = 3.0
_CROP_DISPLAY_PX = 220.0
_BAND_COLORS = {
    "clean": (206, 212, 218),
    "low": (99, 190, 123),
    "medium": (240, 190, 0),
    "high": (247, 133, 45),
    "critical": (220, 30, 45),
}
_DISPOSITION_COLORS = {
    "ACCEPT": (34, 139, 84),
    "DOWNGRADE": (198, 130, 0),
    "HOLD": (200, 30, 45),
}

__all__ = [
    "DispositionRules",
    "FrameRecord",
    "CoilStats",
    "CoilReport",
    "inspect_coil",
    "render_html",
    "write_json",
    "build_coil_report",
    "collect_images",
]


# ------------------------------------------------------------------ data model


@dataclass
class FrameRecord:
    """One inspected frame, reduced to what the coil roll-up and the HTML need."""

    index: int
    source: str
    width: int
    height: int
    verdict: str
    defect_count: int
    dominant_class: str | None
    max_confidence: float
    severity_score: float
    severity_band: str
    class_counts: dict[str, int]
    detections: list[dict[str, Any]]
    total_ms: float

    @classmethod
    def from_result(cls, index: int, source: str, result: InferenceResult) -> "FrameRecord":
        counts: dict[str, int] = {}
        for det in result.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return cls(
            index=index,
            source=source,
            width=int(result.image_size[0]),
            height=int(result.image_size[1]),
            verdict=result.verdict,
            defect_count=int(result.defect_count),
            dominant_class=result.dominant_class,
            max_confidence=float(result.max_confidence),
            severity_score=float(result.severity_score),
            severity_band=(
                _severity_bucket(result.severity_score) if result.detections else "clean"
            ),
            class_counts=counts,
            detections=[d.to_dict() for d in result.detections],
            total_ms=float(result.total_ms),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source": self.source,
            "image_size": [self.width, self.height],
            "verdict": self.verdict,
            "defect_count": self.defect_count,
            "dominant_class": self.dominant_class,
            "max_confidence": round(self.max_confidence, 4),
            "severity_score": round(self.severity_score, 2),
            "severity_band": self.severity_band,
            "class_counts": dict(self.class_counts),
            "detections": self.detections,
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class CoilStats:
    """Coil-level aggregates. All rates are fractions of total frames."""

    total_frames: int
    defect_frames: int
    clean_frames: int
    defect_rate: float
    total_detections: int
    detections_per_frame: float
    class_counts: dict[str, int]
    class_frame_counts: dict[str, int]
    band_counts: dict[str, int]
    mean_severity: float
    max_severity: float
    p95_severity: float
    mean_latency_ms: float
    throughput_fps: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "defect_frames": self.defect_frames,
            "clean_frames": self.clean_frames,
            "defect_rate": round(self.defect_rate, 4),
            "total_detections": self.total_detections,
            "detections_per_frame": round(self.detections_per_frame, 3),
            "class_counts": dict(self.class_counts),
            "class_frame_counts": dict(self.class_frame_counts),
            "band_counts": dict(self.band_counts),
            "severity": {
                "mean": round(self.mean_severity, 2),
                "max": round(self.max_severity, 2),
                "p95": round(self.p95_severity, 2),
            },
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
        }


@dataclass
class DispositionRules:
    """Thresholds that turn coil statistics into a disposition.

    Tuned to be conservative: a coil is only released when the evidence is clean on
    every axis, and any single hard trigger holds it. The quality department owns
    these numbers; nothing else in this module hard-codes a threshold.
    """

    # ACCEPT requires all of these.
    accept_max_defect_rate: float = 0.02
    accept_max_severity_p95: float = 30.0
    accept_forbid_bands: tuple[str, ...] = ("high", "critical")

    # Any one of these forces HOLD.
    hold_min_defect_rate: float = 0.25
    hold_min_severity_p95: float = 70.0
    hold_min_critical_frames: int = 1
    # Classes that cannot be rolled, pickled or ground out downstream.
    hold_classes: tuple[str, ...] = ("inclusion",)
    hold_class_frame_limit: int = 0

    def evaluate(self, stats: CoilStats) -> tuple[str, list[str]]:
        """Return the disposition and the human-readable reasons behind it."""
        hold: list[str] = []
        if stats.defect_rate > self.hold_min_defect_rate:
            hold.append(
                f"Defect rate {stats.defect_rate:.1%} exceeds the hold limit of "
                f"{self.hold_min_defect_rate:.1%}."
            )
        if stats.p95_severity >= self.hold_min_severity_p95:
            hold.append(
                f"95th-percentile severity {stats.p95_severity:.1f} reaches the hold "
                f"limit of {self.hold_min_severity_p95:.1f}."
            )
        critical_frames = stats.band_counts.get("critical", 0)
        if critical_frames >= self.hold_min_critical_frames:
            hold.append(
                f"{critical_frames} frame(s) in the critical severity band "
                f"(limit {self.hold_min_critical_frames - 1})."
            )
        for name in self.hold_classes:
            n = stats.class_frame_counts.get(name, 0)
            if n > self.hold_class_frame_limit:
                hold.append(
                    f"{n} frame(s) contain '{name}', a zero-tolerance defect "
                    f"(limit {self.hold_class_frame_limit}) - it cannot be removed "
                    "downstream."
                )
        if hold:
            return "HOLD", hold

        missed: list[str] = []
        if stats.defect_rate > self.accept_max_defect_rate:
            missed.append(
                f"Defect rate {stats.defect_rate:.1%} above the accept limit of "
                f"{self.accept_max_defect_rate:.1%}."
            )
        if stats.p95_severity > self.accept_max_severity_p95:
            missed.append(
                f"95th-percentile severity {stats.p95_severity:.1f} above the accept "
                f"limit of {self.accept_max_severity_p95:.1f}."
            )
        banded = [
            f"{stats.band_counts.get(b, 0)} frame(s) in the {b} band"
            for b in self.accept_forbid_bands
            if stats.band_counts.get(b, 0) > 0
        ]
        if banded:
            missed.append("; ".join(banded) + " - not releasable as prime.")

        if missed:
            return "DOWNGRADE", missed
        accepted = [
            f"Defect rate {stats.defect_rate:.1%} within the accept limit of "
            f"{self.accept_max_defect_rate:.1%}.",
            f"95th-percentile severity {stats.p95_severity:.1f} within the accept "
            f"limit of {self.accept_max_severity_p95:.1f}.",
        ]
        if self.accept_forbid_bands:
            accepted.append(
                "No frames in the "
                + " or ".join(self.accept_forbid_bands)
                + " severity band(s)."
            )
        return "ACCEPT", accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "accept_max_defect_rate": self.accept_max_defect_rate,
            "accept_max_severity_p95": self.accept_max_severity_p95,
            "accept_forbid_bands": list(self.accept_forbid_bands),
            "hold_min_defect_rate": self.hold_min_defect_rate,
            "hold_min_severity_p95": self.hold_min_severity_p95,
            "hold_min_critical_frames": self.hold_min_critical_frames,
            "hold_classes": list(self.hold_classes),
            "hold_class_frame_limit": self.hold_class_frame_limit,
        }


@dataclass
class CoilReport:
    """Everything needed to render the HTML and to persist the JSON."""

    coil_id: str
    generated_at: str
    model_name: str
    weights_path: str
    device: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    stats: CoilStats
    frames: list[FrameRecord]
    worst: list[FrameRecord]
    disposition: str
    reasons: list[str]
    rules: DispositionRules
    elapsed_s: float
    # Rendering payload only; deliberately absent from `to_dict` so the JSON stays
    # small enough for a database column.
    figures: dict[int, dict[str, str]] = field(default_factory=dict, repr=False)
    explain_note: str | None = None
    # Where the frames came from. Carried into both artefacts because the headline
    # defect rate is only interpretable next to it.
    source_note: str | None = None
    # How the frame sequence was ordered. The position map is read as strip
    # geometry, so a synthetic ordering has to say so on the map itself: a cluster
    # in a shuffled sequence is chance, not a process upset.
    order_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "coil_id": self.coil_id,
            "generated_at": self.generated_at,
            "disposition": self.disposition,
            "reasons": list(self.reasons),
            "model": {
                "name": self.model_name,
                "weights": self.weights_path,
                "device": self.device,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "imgsz": self.imgsz,
            },
            "rules": self.rules.to_dict(),
            "stats": self.stats.to_dict(),
            "worst_frames": [f.index for f in self.worst],
            # The figures themselves stay out of the JSON, but the reason one could
            # not be produced must not: the HTML is the only other place it appears,
            # and the JSON is the copy the quality database keeps.
            "explain_note": self.explain_note,
            "source_note": self.source_note,
            "order_note": self.order_note,
            "frames": [f.to_dict() for f in self.frames],
            "elapsed_s": round(self.elapsed_s, 3),
        }


# ------------------------------------------------------------------- inspection


def collect_images(sources: Sequence[str | Path]) -> list[Path]:
    """Expand directories, globs and plain paths into a sorted image file list."""
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    found: list[Path] = []
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            found.extend(
                p for p in sorted(path.iterdir()) if p.suffix.lower() in suffixes
            )
        elif path.is_file():
            found.append(path)
        else:
            matches = [Path(p) for p in sorted(globlib.glob(str(source)))]
            if not matches:
                raise FileNotFoundError(f"No images matched: {source}")
            found.extend(p for p in matches if p.suffix.lower() in suffixes)
    if not found:
        raise FileNotFoundError(f"No images found in: {list(map(str, sources))}")
    return found


def _header_size(path: Path) -> tuple[int, int] | None:
    """(width, height) read from the image header, without decoding the pixels."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return (int(im.width), int(im.height))
    except (OSError, ValueError, ImportError):
        return None


def _predict_in_size_groups(
    detector: DefectDetector, paths: Sequence[Path], batch_size: int
) -> list[InferenceResult]:
    """Score frames in same-size groups, returning results in the input order.

    ultralytics letterboxes a batch of identically sized frames to the minimum
    stride-multiple rectangle, but pads a *mixed*-size batch out to a full square
    (measured: one 240x800 frame alone reaches the network as 192x640; the same
    frame batched with a 200x200 one reaches it as 640x640). Batching mixed sizes
    together would therefore make a frame's detections depend on which other frames
    happened to land in its chunk - so the same coil scored with a different
    `--batch-size`, or after inserting one frame, would score differently. It would
    also put the boxes in a different geometry from the one `explain.generate_cam`
    reconstructs, so the CAM overlay and the drawn boxes would disagree on a wide
    strip. Grouping by size removes both problems and is a no-op on a uniform coil.
    """
    groups: dict[Any, list[int]] = {}
    for i, path in enumerate(paths):
        size = _header_size(path)
        # An unreadable header gets a singleton group rather than being pooled with
        # every other unknown, which would reintroduce the mixed-size batch.
        groups.setdefault(size if size is not None else ("unknown", i), []).append(i)

    out: list[InferenceResult | None] = [None] * len(paths)
    for indices in groups.values():
        scored = detector.predict_batch([paths[i] for i in indices], batch_size=batch_size)
        for index, result in zip(indices, scored):
            out[index] = result
    if any(r is None for r in out):
        raise RuntimeError("Internal error: not every frame was scored.")
    return [r for r in out if r is not None]


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def _aggregate(records: Sequence[FrameRecord]) -> CoilStats:
    """Roll per-frame records up to coil level."""
    total = len(records)
    defect_frames = sum(1 for r in records if r.verdict == "DEFECT")
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_frame_counts = {name: 0 for name in CLASS_NAMES}
    band_counts = {"clean": 0, **{b: 0 for b in SEVERITY_BANDS}}

    for record in records:
        band_counts[record.severity_band] = band_counts.get(record.severity_band, 0) + 1
        for name, count in record.class_counts.items():
            class_counts[name] = class_counts.get(name, 0) + count
            class_frame_counts[name] = class_frame_counts.get(name, 0) + 1

    severities = [r.severity_score for r in records]
    latencies = [r.total_ms for r in records]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return CoilStats(
        total_frames=total,
        defect_frames=defect_frames,
        clean_frames=total - defect_frames,
        defect_rate=(defect_frames / total) if total else 0.0,
        total_detections=sum(r.defect_count for r in records),
        detections_per_frame=(sum(r.defect_count for r in records) / total) if total else 0.0,
        class_counts=class_counts,
        class_frame_counts=class_frame_counts,
        band_counts=band_counts,
        mean_severity=float(np.mean(severities)) if severities else 0.0,
        max_severity=float(np.max(severities)) if severities else 0.0,
        p95_severity=_percentile(severities, 95),
        mean_latency_ms=mean_latency,
        throughput_fps=(1000.0 / mean_latency) if mean_latency > 0 else 0.0,
    )


def _scaled_result(result: InferenceResult, factor: float) -> InferenceResult:
    """Copy a result with every box scaled, so annotation can be drawn on an
    upscaled frame at a legible font size instead of being blurred afterwards."""
    scaled = [
        replace(det, bbox_xyxy=tuple(float(v) * factor for v in det.bbox_xyxy))
        for det in result.detections
    ]
    return replace(result, detections=scaled)


def _data_uri(rgb: np.ndarray, quality: int = 86) -> str:
    """Embed a figure as a base64 JPEG data URI.

    JPEG rather than PNG on purpose: these are photographs of steel texture, where
    PNG cannot find any redundancy to exploit. On a six-frame report the same
    figures cost 2.3 MB as PNG against roughly 0.2 MB as JPEG, which is the
    difference between a report that survives a mail gateway and one that does not.
    Quality 86 keeps the annotation chips and box edges clean.
    """
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed while embedding a report figure.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _build_figures(
    detector: DefectDetector,
    path: Path,
    result: InferenceResult,
    include_cam: bool,
    display_px: int = 400,
) -> tuple[dict[str, str], str | None]:
    """Annotated frame, defect crop and (optionally) an EigenCAM overlay, as base64
    JPEG data URIs. Returns the figures and a note if the CAM had to be skipped."""
    rgb = _to_rgb(path)
    h, w = rgb.shape[:2]
    # `display_px` is the long edge of the embedded figure in both directions: a
    # 200 px NEU-DET crop is enlarged so the annotation is legible, and a
    # mill-resolution 3000 px strip is shrunk. Enlarging only would embed a full
    # native-resolution frame per worst-case entry (measured: 450 KB each on a
    # 2600x1400 frame), which is exactly the mail-gateway problem `_data_uri`
    # avoids. The zoomed crop below is what preserves defect detail.
    factor = min(_MAX_FIGURE_UPSCALE, display_px / float(max(h, w)))
    interp = cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA
    big = (
        rgb
        if factor == 1.0
        else cv2.resize(
            rgb,
            (max(1, int(round(w * factor))), max(1, int(round(h * factor)))),
            interpolation=interp,
        )
    )
    annotated = detector.annotate(big, _scaled_result(result, factor))
    figures = {"annotated": _data_uri(annotated)}

    if result.detections:
        # Crop from the native-resolution frame rather than from the display copy:
        # on a downscaled mill frame the display copy has already thrown away the
        # texture this figure exists to show.
        det = max(result.detections, key=lambda d: d.confidence)
        x1, y1, x2, y2 = det.bbox_xyxy
        pad = 0.35 * max(x2 - x1, y2 - y1, 8.0)
        cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        cx2, cy2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        crop = rgb[cy1:cy2, cx1:cx2]
        if crop.size:
            zoom = float(
                np.clip(_CROP_DISPLAY_PX / max(crop.shape[0], crop.shape[1]), 1.0, 8.0)
            )
            crop_big = (
                crop
                if zoom == 1.0
                else cv2.resize(
                    crop,
                    (int(crop.shape[1] * zoom), int(crop.shape[0] * zoom)),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            # Only the target box, translated into crop coordinates: annotate()
            # clamps label chips into the frame, so a neighbouring detection that
            # falls outside the crop would still stamp a misleading label on it.
            local = replace(
                result,
                detections=[
                    replace(
                        det,
                        bbox_xyxy=(
                            (x1 - cx1) * zoom,
                            (y1 - cy1) * zoom,
                            (x2 - cx1) * zoom,
                            (y2 - cy1) * zoom,
                        ),
                    )
                ],
            )
            figures["crop"] = _data_uri(detector.annotate(crop_big, local))

    note = None
    if include_cam:
        try:
            from explain import DegenerateExplanationError, generate_cam, overlay_cam

            heatmap = generate_cam(detector, rgb)
            overlay = overlay_cam(big, heatmap)
            figures["cam"] = _data_uri(overlay)
        except DegenerateExplanationError as exc:
            note = f"EigenCAM overlay omitted: {exc}"
        except Exception as exc:  # explainability must never break the report
            note = f"EigenCAM overlay unavailable ({type(exc).__name__}: {exc})."
    return figures, note


def _derive_source_note(paths: Sequence[Path]) -> str | None:
    """Say where the frames came from, when we can tell, without being asked.

    NEU-DET has no defect-free image: all 1800 frames carry at least one labelled
    defect, verified across every split. A coil assembled from it therefore has a
    defect rate near 100% by construction, and quoting that figure as if it were a
    mill's reject rate would be the single most misleading number this report can
    produce. Deriving the caveat from the frame paths means it appears whenever it
    applies, rather than whenever whoever ran the script remembered to pass it.
    """
    if not paths:
        return None
    try:
        marker = "neu-det"
        if not all(marker in str(p.resolve()).lower() for p in paths):
            return None
    except OSError:  # pragma: no cover - unresolvable path
        return None
    return (
        "Frames are real held-out NEU-DET images. Every image in that dataset "
        "carries at least one labelled defect -- there is no clean frame in any "
        "split -- so this coil's defect rate is a property of the source data, "
        "not a forecast of a mill's reject rate. What this report demonstrates is "
        "the aggregation, disposition and evidence path over a coil-length "
        "sequence; the per-frame detections, severities and latencies are real "
        "measurements on unseen images."
    )


def inspect_coil(
    images: Sequence[str | Path],
    detector: DefectDetector | None = None,
    *,
    coil_id: str = "UNSPECIFIED",
    worst_n: int = 6,
    batch_size: int = 16,
    rules: DispositionRules | None = None,
    include_cam: bool = True,
    weights: str | Path | None = None,
    device: str = "auto",
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = DEFAULT_IMGSZ,
    source_note: str | None = None,
    order_note: str | None = None,
) -> CoilReport:
    """Score every frame of a coil and roll the results up into a `CoilReport`."""
    started = time.perf_counter()
    paths = collect_images(images)
    detector = detector or load_detector(
        weights=weights, device=device, conf=conf, iou=iou, imgsz=imgsz
    )
    rules = rules or DispositionRules()

    results = _predict_in_size_groups(detector, paths, batch_size)
    records = [
        FrameRecord.from_result(i, str(p), r) for i, (p, r) in enumerate(zip(paths, results))
    ]

    stats = _aggregate(records)
    disposition, reasons = rules.evaluate(stats)

    # Worst = highest severity, then most detections, then highest confidence.
    ranked = sorted(
        (r for r in records if r.defect_count > 0),
        key=lambda r: (r.severity_score, r.defect_count, r.max_confidence),
        reverse=True,
    )
    worst = ranked[: max(0, int(worst_n))]

    figures: dict[int, dict[str, str]] = {}
    notes: list[str] = []
    for record in worst:
        figure, note = _build_figures(
            detector, Path(record.source), results[record.index], include_cam
        )
        figures[record.index] = figure
        if note and note not in notes:
            notes.append(note)

    return CoilReport(
        coil_id=coil_id,
        generated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        model_name=detector.model_name,
        weights_path=str(detector.weights_path),
        device=detector.device,
        conf_threshold=detector.conf,
        iou_threshold=detector.iou,
        imgsz=detector.imgsz,
        stats=stats,
        frames=records,
        worst=worst,
        disposition=disposition,
        reasons=reasons,
        rules=rules,
        elapsed_s=time.perf_counter() - started,
        figures=figures,
        # Every distinct reason a CAM was skipped, not just the first: two frames can
        # fail for two different reasons and a reviewer needs to see both.
        explain_note=" ".join(notes) if notes else None,
        source_note=source_note or _derive_source_note(paths),
        order_note=order_note,
    )


# --------------------------------------------------------------------- rendering


def _rgb_css(colour: tuple[int, int, int]) -> str:
    return f"rgb({colour[0]},{colour[1]},{colour[2]})"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


_STYLE = """
:root{--ink:#1c2128;--muted:#5b6672;--line:#e2e6ea;--bg:#f6f7f9;--card:#ffffff;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:28px 22px 56px;}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.2px;}
h2{font-size:15px;margin:0 0 14px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);}
h3{font-size:15px;margin:0 0 6px;}
.sub{color:var(--muted);font-size:13px;margin:0;}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:20px 22px;margin-bottom:18px;}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;}
.badge{color:#fff;padding:10px 20px;border-radius:8px;font-weight:700;font-size:19px;
 letter-spacing:.06em;white-space:nowrap;}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;}
.kpi{border:1px solid var(--line);border-radius:8px;padding:12px 14px;background:var(--card);}
.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.2;}
.kpi .l{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-top:3px;}
ul.reasons{margin:0;padding-left:20px;}
ul.reasons li{margin-bottom:5px;}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);font-size:13px;}
th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600;}
td.n,th.n{text-align:right;}
.bar{height:9px;border-radius:5px;min-width:2px;}
.barcell{width:38%;}
.map{display:flex;flex-wrap:wrap;gap:2px;}
.map i{width:11px;height:16px;border-radius:2px;display:block;}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px;color:var(--muted);font-size:12px;}
.legend span{display:flex;align-items:center;gap:6px;}
.legend i{width:11px;height:11px;border-radius:2px;display:block;}
.frame{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;
 padding:18px 0;border-bottom:1px solid var(--line);}
.frame:last-child{border-bottom:none;padding-bottom:0;}
.figs{display:flex;gap:10px;flex-wrap:wrap;}
.fig{text-align:center;}
.fig img{border:1px solid var(--line);border-radius:6px;display:block;max-width:100%;}
.fig figcaption{font-size:10.5px;color:var(--muted);margin-top:4px;text-transform:uppercase;
 letter-spacing:.06em;}
.pill{display:inline-block;color:#fff;border-radius:999px;padding:2px 10px;font-size:11px;
 font-weight:650;letter-spacing:.04em;}
.kb{background:#fbfcfd;border:1px solid var(--line);border-left:3px solid var(--muted);
 border-radius:0 6px 6px 0;padding:10px 14px;margin-top:12px;font-size:12.5px;}
.kb b{display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
 color:var(--muted);margin-bottom:2px;}
.kb p{margin:0 0 9px;}
.kb p:last-child{margin-bottom:0;}
.meta{font-size:12px;color:var(--muted);}
.meta code{background:#eef1f4;padding:1px 5px;border-radius:3px;font-size:11.5px;}
.note{font-size:12px;color:#8a5a00;background:#fff8e6;border:1px solid #f0dfae;
 border-radius:6px;padding:8px 12px;margin-top:12px;}
@media print{body{background:#fff;}.card{break-inside:avoid;}.frame{break-inside:avoid;}}
@media (max-width:720px){.frame{grid-template-columns:1fr;}}
"""


def _kpi(value: str, label: str) -> str:
    return f'<div class="kpi"><div class="v">{_esc(value)}</div><div class="l">{_esc(label)}</div></div>'


def _bar_row(label: str, count: int, total: int, colour: tuple[int, int, int], extra: str) -> str:
    pct = (count / total * 100.0) if total else 0.0
    return (
        f"<tr><td>{_esc(label)}</td><td class='n'>{count}</td><td class='n'>{extra}</td>"
        f"<td class='barcell'><div class='bar' style='width:{max(pct, 0.6):.1f}%;"
        f"background:{_rgb_css(colour)}'></div></td>"
        f"<td class='n'>{pct:.1f}%</td></tr>"
    )


def _frame_block(report: CoilReport, record: FrameRecord) -> str:
    figures = report.figures.get(record.index, {})
    band_colour = _BAND_COLORS.get(record.severity_band, (120, 120, 120))
    captions = (("annotated", "detections"), ("crop", "worst defect, zoomed"), ("cam", "EigenCAM attention"))
    figs = "".join(
        f"<figure class='fig'><img src='{figures[key]}' alt='{_esc(caption)}'>"
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
        for key, caption in captions
        if key in figures
    )

    rows = "".join(
        f"<tr><td><span class='pill' style='background:"
        f"{_rgb_css(CLASS_COLORS.get(d['class_name'], (100, 100, 100)))}'>"
        f"{_esc(d['class_name'])}</span></td>"
        f"<td class='n'>{d['confidence']:.2f}</td>"
        f"<td class='n'>{d['area_frac'] * 100:.2f}%</td>"
        f"<td>{_esc(d['severity'])}</td></tr>"
        for d in record.detections[:8]
    )
    more = (
        f"<p class='meta'>+{len(record.detections) - 8} further detection(s) not listed.</p>"
        if len(record.detections) > 8
        else ""
    )

    info = DEFECT_INFO.get(record.dominant_class or "", {})
    kb = (
        f"<div class='kb'><b>Probable cause</b><p>{_esc(info['cause'])}</p>"
        f"<b>Recommended action</b><p>{_esc(info['action'])}</p></div>"
        if info
        else ""
    )

    return (
        "<div class='frame'>"
        f"<div><div class='figs'>{figs}</div></div>"
        "<div>"
        f"<h3>Frame {record.index} &middot; {_esc(Path(record.source).name)}</h3>"
        f"<p class='sub'>severity <b>{record.severity_score:.1f}</b> "
        f"<span class='pill' style='background:{_rgb_css(band_colour)}'>"
        f"{_esc(record.severity_band)}</span> &middot; {record.defect_count} detection(s)</p>"
        "<table><thead><tr><th>Class</th><th class='n'>Conf</th>"
        f"<th class='n'>Area</th><th>Severity</th></tr></thead><tbody>{rows}</tbody></table>"
        f"{more}{kb}</div></div>"
    )


def render_html(report: CoilReport, out_path: str | Path) -> Path:
    """Write the self-contained HTML report and return its path."""
    stats = report.stats
    disp_colour = _DISPOSITION_COLORS.get(report.disposition, (90, 90, 90))

    kpis = "".join([
        _kpi(str(stats.total_frames), "frames inspected"),
        _kpi(f"{stats.defect_rate:.1%}", "defect rate"),
        _kpi(str(stats.defect_frames), "defective frames"),
        _kpi(str(stats.total_detections), "total detections"),
        _kpi(f"{stats.mean_severity:.1f}", "mean severity"),
        _kpi(f"{stats.p95_severity:.1f}", "p95 severity"),
        _kpi(f"{stats.max_severity:.1f}", "max severity"),
        _kpi(f"{stats.throughput_fps:.0f} fps", "throughput"),
    ])

    class_rows = "".join(
        _bar_row(
            name,
            stats.class_counts.get(name, 0),
            max(stats.total_detections, 1),
            CLASS_COLORS.get(name, (120, 120, 120)),
            str(stats.class_frame_counts.get(name, 0)),
        )
        for name in sorted(CLASS_NAMES, key=lambda n: -stats.class_counts.get(n, 0))
    )
    band_rows = "".join(
        _bar_row(
            band,
            stats.band_counts.get(band, 0),
            max(stats.total_frames, 1),
            _BAND_COLORS[band],
            "-",
        )
        for band in ("clean",) + SEVERITY_BANDS
    )

    coil_map = "".join(
        f"<i title='frame {r.index}: {_esc(r.severity_band)} "
        f"({r.severity_score:.0f})' style='background:"
        f"{_rgb_css(_BAND_COLORS.get(r.severity_band, (150, 150, 150)))}'></i>"
        for r in report.frames
    )
    legend = "".join(
        f"<span><i style='background:{_rgb_css(_BAND_COLORS[b])}'></i>{_esc(b)}</span>"
        for b in ("clean",) + SEVERITY_BANDS
    )

    frames_html = (
        "".join(_frame_block(report, r) for r in report.worst)
        or "<p class='sub'>No defective frames on this coil.</p>"
    )
    reasons = "".join(f"<li>{_esc(reason)}</li>" for reason in report.reasons)
    note = f"<div class='note'>{_esc(report.explain_note)}</div>" if report.explain_note else ""

    rules = report.rules
    source_note = (
        f"<div class='note' style='margin-top:14px'>Frame provenance &mdash; "
        f"{_esc(report.source_note)}</div>"
        if report.source_note
        else ""
    )
    # Rendered inline in the map caption rather than as a separate note: a reader
    # who skips a footnote must not be left believing a shuffled sequence is strip
    # geometry.
    order_note = f" <b>{_esc(report.order_note)}</b>" if report.order_note else ""
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Coil inspection report {_esc(report.coil_id)}</title>
<style>{_STYLE}</style></head><body><div class="wrap">

<div class="card"><div class="head">
<div><h1>Coil inspection report &middot; {_esc(report.coil_id)}</h1>
<p class="sub">{_esc(stats.total_frames)} frames &middot; generated {_esc(report.generated_at)}
 &middot; model <code>{_esc(report.model_name)}</code></p></div>
<div class="badge" style="background:{_rgb_css(disp_colour)}">{_esc(report.disposition)}</div>
</div></div>

<div class="card"><h2>Summary</h2><div class="kpis">{kpis}</div>{source_note}</div>

<div class="card"><h2>Disposition rationale</h2>
<ul class="reasons">{reasons}</ul>
<p class="meta" style="margin-top:14px">Rules applied &mdash;
accept below {rules.accept_max_defect_rate:.0%} defect rate and
p95 severity {rules.accept_max_severity_p95:.0f} with no
{_esc("/".join(rules.accept_forbid_bands))} frames;
hold above {rules.hold_min_defect_rate:.0%} defect rate,
p95 severity {rules.hold_min_severity_p95:.0f},
{rules.hold_min_critical_frames} critical frame(s), or any frame containing
{_esc("/".join(rules.hold_classes))}; otherwise downgrade.</p></div>

<div class="card"><h2>Coil position map</h2>
<div class="map">{coil_map}</div>
<div class="legend">{legend}</div>
<p class="meta" style="margin-top:10px">One cell per frame in strip order, coloured by
frame severity band. When the frames arrive in true strip order, a cluster indicates a
sustained process upset rather than a sporadic defect.{order_note}</p></div>

<div class="card"><h2>Defect classes</h2>
<table><thead><tr><th>Class</th><th class="n">Detections</th><th class="n">Frames</th>
<th class="barcell">Share of detections</th><th class="n">%</th></tr></thead>
<tbody>{class_rows}</tbody></table></div>

<div class="card"><h2>Severity distribution</h2>
<table><thead><tr><th>Band</th><th class="n">Frames</th><th class="n"></th>
<th class="barcell">Share of frames</th><th class="n">%</th></tr></thead>
<tbody>{band_rows}</tbody></table></div>

<div class="card"><h2>Worst {len(report.worst)} frames</h2>{frames_html}{note}</div>

<div class="card"><h2>Provenance</h2>
<p class="meta">Weights <code>{_esc(report.weights_path)}</code><br>
Device <code>{_esc(report.device)}</code> &middot;
imgsz <code>{report.imgsz}</code> &middot;
conf <code>{report.conf_threshold}</code> &middot;
iou <code>{report.iou_threshold}</code><br>
Mean latency {stats.mean_latency_ms:.1f} ms/frame &middot;
coil scored in {report.elapsed_s:.1f} s &middot;
{_esc(platform.platform())}<br>
This report is self-contained: all images are embedded and it makes no network
requests.</p></div>

</div></body></html>
"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def write_json(report: CoilReport, out_path: str | Path) -> Path:
    """Write the machine-readable report (no images) and return its path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return out


def build_coil_report(
    images: Sequence[str | Path],
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    stem: str = "coil_report",
    **kwargs: Any,
) -> tuple[CoilReport, Path, Path]:
    """Inspect a coil and write both artefacts. Returns (report, html, json)."""
    report = inspect_coil(images, **kwargs)
    directory = Path(out_dir)
    return (
        report,
        render_html(report, directory / f"{stem}.html"),
        write_json(report, directory / f"{stem}.json"),
    )


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a per-coil inspection report.")
    parser.add_argument("--images", nargs="+", required=True, help="dirs, globs or files")
    parser.add_argument("--coil-id", default="UNSPECIFIED")
    parser.add_argument("--limit", type=int, default=0, help="cap the frame count (0 = all)")
    parser.add_argument(
        "--shuffle-seed", type=int, default=None,
        help="shuffle the frames into strip order with this seed before scoring. "
             "A directory listing is alphabetical, which for NEU-DET means sorted "
             "by defect class, and the coil position map then shows six clean "
             "blocks that read as sustained process upsets. Shuffling is what makes "
             "the map honest; the seed is what makes it reproducible.",
    )
    parser.add_argument("--worst-n", type=int, default=6)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--stem", default="coil_report")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                        help="network input size (default: inference.DEFAULT_IMGSZ, "
                             "the size src/model_study.py chose on val)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-cam", action="store_true", help="skip EigenCAM overlays")
    parser.add_argument(
        "--source-note", default=None,
        help="override the frame-provenance caveat printed on the report "
             "(default: derived from the frame paths)",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = list(collect_images(args.images))
    order_note: str | None = None
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(paths)
        order_note = (
            f"This coil's frame order is a reproducible shuffle (seed "
            f"{args.shuffle_seed}) of a directory listing, not a real strip "
            f"sequence, so clustering in this map carries no process meaning."
        )
    if args.limit > 0:
        paths = paths[: args.limit]

    report, html_path, json_path = build_coil_report(
        list(paths),
        out_dir=args.out_dir,
        stem=args.stem,
        coil_id=args.coil_id,
        worst_n=args.worst_n,
        batch_size=args.batch_size,
        include_cam=not args.no_cam,
        weights=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        source_note=args.source_note,
        order_note=order_note,
    )
    stats = report.stats
    print(
        f"coil {report.coil_id}: {stats.total_frames} frames, "
        f"defect rate {stats.defect_rate:.1%}, {stats.total_detections} detections, "
        f"p95 severity {stats.p95_severity:.1f} -> {report.disposition}"
    )
    for reason in report.reasons:
        print(f"  - {reason}")
    print(f"html: {html_path} ({html_path.stat().st_size / 1024:.0f} KB)")
    print(f"json: {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
