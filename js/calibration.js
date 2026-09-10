/**
 * Isotonic confidence calibration, transcribed verbatim from
 * reports/calibration.json -> calibrator.params. Nothing here is re-fitted or
 * re-rounded: this file is generated from that JSON so the browser reports the same
 * probability the Python console reports.
 *
 * Why it exists. A YOLO class score is not a probability. On the held-out test split
 * the shipped model's raw scores carry an expected calibration error of
 * 0.1418: a box shown at "0.40" was right far less often than 40% of the
 * time. src/calibrate.py fits a monotone isotonic map on the VAL split (grouped
 * 5-fold CV by source image, seed 20260909), which drops held-out ECE to
 * 0.0461 -- a 67.5% reduction. Because the map is monotone it cannot
 * reorder detections, so mAP50 is numerically unchanged at 0.707986 before and
 * after (reports/calibration.json -> map50.identical = true).
 *
 * The eps term is what makes it strictly monotone. Isotonic regression is only
 * non-decreasing: it maps runs of distinct raw scores onto one plateau, creating ties
 * that would make average precision depend on sort order. Blending in eps * s
 * restores a strict order while moving no value by more than 1e-06.
 *
 * Method chosen out of fold from 5 candidates (reports/calibration.json ->
 * selection): isotonic OOF ECE 0.0250 beat temperature
 * 0.0961 and identity 0.1008.
 */

/** reports/calibration.json -> calibrator.params.knots_x */
const KNOTS_X = [
  0.050164446234703064, 0.05081210657954216, 0.05097289755940437, 0.052958663552999496,
  0.053003329783678055, 0.13377390801906586, 0.13395541906356812, 0.1742050051689148,
  0.17491646111011505, 0.2399325966835022, 0.24104474484920502, 0.31750497221946716,
  0.3183032274246216, 0.3576990067958832, 0.35778093338012695, 0.3634984791278839,
  0.3669489920139313, 0.39388763904571533, 0.39440345764160156, 0.4545809030532837,
  0.46039879322052, 0.4709504544734955, 0.47620758414268494, 0.5732674598693848,
  0.5750333070755005, 0.594139814376831, 0.5976260304450989, 0.6975067853927612,
  0.6998829245567322, 0.7374289035797119, 0.7380223870277405, 0.7797176837921143,
  0.7800125479698181, 0.9425139427185059
];

/** reports/calibration.json -> calibrator.params.knots_y */
const KNOTS_Y = [
  0.09210526315789473, 0.09210526315789473, 0.09210526315789473, 0.09210526315789473,
  0.1570796460176991, 0.1570796460176991, 0.25510204081632654, 0.25510204081632654,
  0.26744186046511625, 0.26744186046511625, 0.32978723404255317, 0.32978723404255317,
  0.35714285714285715, 0.35714285714285715, 0.4166666666666667, 0.4166666666666667, 0.46875,
  0.46875, 0.5, 0.5, 0.6428571428571429, 0.6428571428571429, 0.7051282051282052,
  0.7051282051282052, 0.7222222222222222, 0.7222222222222222, 0.8015873015873016,
  0.8015873015873016, 0.8913043478260869, 0.8913043478260869, 0.9444444444444444,
  0.9444444444444444, 0.995, 0.995
];

/** reports/calibration.json -> calibrator.params.eps */
const EPS = 1e-06;

export const CALIBRATION = {
  method: 'isotonic',
  nKnots: KNOTS_X.length,
  eceBefore: 0.1418430734177215,
  eceAfter: 0.04608132784810127,
  brierBefore: 0.1753679834518315,
  brierAfter: 0.15761263525448946,
  fitSplit: 'val',
  evalSplit: 'test',
  source: 'reports/calibration.json',
};

/**
 * np.interp with clipped ends, which is what scikit-learn's
 * out_of_bounds="clip" does, plus the strict-monotone eps blend. Mirrors
 * IsotonicCalibrator._apply in src/calibrate.py line for line.
 */
export function calibrate(raw) {
  const lo = KNOTS_X[0];
  const hi = KNOTS_X[KNOTS_X.length - 1];
  const s = Math.min(Math.max(raw, lo), hi);

  let base;
  if (s <= lo) {
    base = KNOTS_Y[0];
  } else if (s >= hi) {
    base = KNOTS_Y[KNOTS_Y.length - 1];
  } else {
    // np.interp uses the LAST matching interval for repeated x values; searching
    // from the right reproduces that on the duplicated plateau edges.
    let i = KNOTS_X.length - 1;
    while (i > 0 && KNOTS_X[i - 1] >= s) i--;
    const xa = KNOTS_X[i - 1];
    const xb = KNOTS_X[i];
    const ya = KNOTS_Y[i - 1];
    const yb = KNOTS_Y[i];
    base = xb === xa ? yb : ya + ((yb - ya) * (s - xa)) / (xb - xa);
  }

  return base * (1 - EPS) + EPS * Math.min(Math.max(raw, 0), 1);
}

/**
 * reports/calibration.json -> examples. Checked at load in the parity harness, so a
 * transcription slip in the knots above fails loudly instead of quietly shifting
 * every probability on the page.
 */
export const CALIBRATION_EXAMPLES = {
  "0.15": 0.2551019357142857,
  "0.25": 0.32978715425531907,
  "0.40": 0.4999999,
  "0.60": 0.8015871,
  "0.70": 0.891304156521739,
  "0.80": 0.994999805,
  "0.90": 0.9949999049999999
};
