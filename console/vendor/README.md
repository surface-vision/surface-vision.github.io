# vendor/ -- copies, not a fork

Every `.py` file in this directory is a copy of the file of the same name in
`src/` of the Jindal Stainless surface-defect repository. They are here because
the Space host has no `src/`: an import that reached outside this directory would
be an `ImportError` on the first click, so the console carries what it needs.

`app.py` puts this directory on `sys.path` and imports these modules flat, which
is exactly how `demo/app.py` imports `src/` upstream -- same module names, same
import lines, no package wrapper. Each module resolves its own
`PROJECT_ROOT = Path(__file__).resolve().parent.parent`, which from here is the
Space root, so `models/`, `reports/`, `data/` and `assets/` all point at the
trimmed copies committed beside this directory and no path in the upstream code
had to be touched.

## What is here, and who imports it

| module | pulled in by | when |
|---|---|---|
| `inference.py` | `app.py` | at import -- the detector, severity model, tiling and `DEFECT_INFO` |
| `ood_guard.py` | `app.py` | at import -- the plausibility gate; numpy + OpenCV, no torch |
| `domain_shift.py` | `ood_guard.py` | at import -- supplies the training-grey reference distribution |
| `calibrate.py` | `app.py` | lazily, on first render -- `load_calibrator` only |
| `explain.py` | `app.py` | lazily, on the Explain button -- EigenCAM |
| `report.py` | `app.py` | lazily, on the Build report button -- coil disposition artefact |

`domain_shift.load_reference()` reads `models/domain_reference.json`, which is
committed here. Without it the function rebuilds the reference from 1,440
training JPEGs, which do not exist on this host -- that file is not optional.

## Deviations from upstream

Marked `# SPACE:` at the point they occur. There is exactly one:

1. **`calibrate.py`** -- the module-scope `from evaluate import ...` is wrapped so
   its absence is survivable. `src/evaluate.py` is not vendored: every name
   `calibrate.py` takes from it belongs to the calibrator-*fitting* path, which
   reads NEU-DET ground-truth label files and runs the ultralytics validator over
   the dataset yaml, and this Space ships 18 sample frames and no labels. The
   Space needs one function out of this file, `load_calibrator`, which reads
   `reports/calibration.json`. `IOU_MATCH` is the only name needed at import time
   (it is a default argument on two `def`s) and is mirrored with its upstream
   value, asserted by `tools/verify_vendor.py`; the callables become a stub that
   raises with an explanation rather than a `NameError`.

Nothing else was edited. The `# SPACE:` marker is the only thing to grep for.

## Re-vendoring

    cp src/{inference,domain_shift,ood_guard,calibrate,explain,report}.py hf_space/vendor/

then re-apply the header note and the one `calibrate.py` deviation, and run
`hf_space/tools/verify_vendor.py`, which diffs each vendored file against its
upstream and fails on any difference outside the marked regions.
