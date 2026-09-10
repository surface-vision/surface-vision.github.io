/**
 * Operator-facing defect knowledge base.
 *
 * Copied verbatim, by script, from DEFECT_INFO in src/inference.py -- restricted to
 * the six NEU-DET classes the shipped ONNX head actually predicts. Not one word of
 * this metallurgy was written for the web page: the cause, base severity tier and
 * recommended action are the same strings the Streamlit console and the coil report
 * show, so an operator reading the demo and an operator reading the console are
 * reading the same instruction.
 *
 * "severity" is the BASE TIER of the defect type, not the severity of one detection.
 * src/inference.py -> score_detection scales that tier by confidence and by the
 * fraction of the frame the box covers; scoreDetection() below reproduces that rule.
 */

export const SEVERITY_BANDS = ["low", "medium", "high", "critical"];

export const DEFECT_INFO = {
  "crazing": {
    "cause": "Network of fine surface cracks from low hot ductility at the strip surface: over-soaking or excessive reheat temperature coarsening the grain, finishing or coiling temperature off schedule, or tramp copper and tin from the scrap charge causing hot shortness (Cu above roughly 0.2 wt% without compensating nickel).",
    "severity": "high",
    "action": "Check reheat furnace soak temperature and residence time against the practice sheet, verify finishing and coiling temperatures on the pyrometer trend, pull the heat chemistry for Cu/Sn residuals, and hold the coil for surface conditioning or grinding before it is released."
  },
  "inclusion": {
    "cause": "Non-metallic inclusion (oxide, silicate or sulphide) trapped below the slab surface and exposed and elongated by rolling. Sources are slag or mould-flux carryover in the caster, deoxidation products that failed to float out, and eroded ladle or tundish refractory.",
    "severity": "critical",
    "action": "Trace the coil back to its heat and cast sequence and quarantine the affected strand length. Review tundish level control and slag carryover at ladle change, inspect ladle and shroud refractory wear, and check calcium treatment and argon stirring practice. Crop or downgrade the affected length; inclusions cannot be rolled or pickled out."
  },
  "patches": {
    "cause": "Irregular light or dark areas from non-uniform surface condition: patchy descaling that leaves residual scale which is then flattened by the rolls, temperature streaks from unbalanced reheat furnace burners, or uneven roll-coolant and emulsion pickup on the strip.",
    "severity": "medium",
    "action": "Verify descaler header pressure and check for blocked or misaligned nozzles across the strip width, balance reheat furnace burners to remove skid and burner temperature streaks, and confirm pickling line acid concentration and line speed so the patches are not carried forward."
  },
  "pitted_surface": {
    "cause": "Dense field of small depressions left where scale nodules grew into the base metal and then spalled off, driven by prolonged high-temperature oxidation in the reheat furnace with an oxidising atmosphere, or by over-pickling and acid attack on the pickle line.",
    "severity": "high",
    "action": "Reduce furnace residence time and trim the air/fuel ratio towards stoichiometric to cut secondary scale growth, confirm descaling is effective ahead of the finishing stands, and audit pickle line acid concentration, bath temperature and line speed. Reject for exposed or coated applications: pits retain acid and moisture."
  },
  "rolled-in_scale": {
    "cause": "Iron oxide scale pressed into the strip surface by the work rolls. Caused by incomplete primary or secondary descaling (low header pressure, blocked nozzles, wrong standoff), heavy furnace scaling, or scale build-up baked onto the work roll barrel.",
    "severity": "high",
    "action": "Restore descaler pressure to practice (typically above 180 bar) and clear or replace blocked nozzles, inspect work roll surface condition and shorten the roll change interval, and lower reheat temperature or residence time. Flag the coil: embedded oxide breaks out in cold rolling and ruins coating adhesion, and pickling may not fully remove it."
  },
  "scratches": {
    "cause": "Linear mechanical gouging from contact with fixed or damaged plant: seized or non-rotating table rollers, worn side guides and side guards, damaged pinch rolls or coiler mandrel, hard scale debris dragged along the strip, or mishandling during coil transport and strapping.",
    "severity": "medium",
    "action": "Walk the run-out table and inspect for seized or scored rollers, dress or replace damaged side guides, guards and pinch rolls, clear scale debris from the table, and review coil handling, tong and strapping practice. Deep scratches act as crack initiators in forming and must be ground out or the length cropped."
  }
};

/** src/inference.py -> _TIER_BASE_SCORE */
const TIER_BASE_SCORE = { low: 22.0, medium: 42.0, high: 64.0, critical: 84.0 };

/** src/inference.py -> _SEVERITY_CUTOFFS */
const SEVERITY_CUTOFFS = [30.0, 55.0, 80.0];

/** src/inference.py -> _AREA_SATURATION, _AREA_GAIN */
const AREA_SATURATION = 0.25;
const AREA_GAIN = 0.55;

/**
 * src/inference.py -> score_detection, transcribed.
 *
 *   base      = tier anchor for the class
 *   conf_mult = 0.5 + 0.5 * confidence          a marginal hit is halved
 *   area_mult = 1 + 0.55 * min(1, area/0.25)    a quarter-frame defect is fully weighted
 *   score     = clip(base * conf_mult * area_mult, 0, 100)
 *
 * The console feeds this the RAW model confidence (src/inference.py line 817), not
 * the calibrated probability, so this page does the same. Changing it here would
 * make the demo's severity band disagree with the coil report's.
 */
export function scoreDetection(className, confidence, areaFrac) {
  const info = DEFECT_INFO[className];
  const base = TIER_BASE_SCORE[info ? info.severity : 'medium'];
  const confMult = 0.5 + 0.5 * Math.min(Math.max(confidence, 0), 1);
  const areaMult = 1 + AREA_GAIN * Math.min(1, Math.max(areaFrac, 0) / AREA_SATURATION);
  return Math.min(Math.max(base * confMult * areaMult, 0), 100);
}

/** src/inference.py -> _severity_bucket */
export function severityBucket(score) {
  for (let i = 0; i < SEVERITY_CUTOFFS.length; i++) {
    if (score < SEVERITY_CUTOFFS[i]) return SEVERITY_BANDS[i];
  }
  return SEVERITY_BANDS[SEVERITY_BANDS.length - 1];
}
