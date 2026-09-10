/**
 * Side-by-side comparison of the Python ultralytics detections and the JS decode.
 *
 * Greedy matching by IoU within the same class, highest-confidence first. Prints a
 * table and exits non-zero if any detection is unmatched or any matched box moves
 * more than --tol pixels or any confidence moves more than --conf-tol.
 *
 * Usage: node site/verify/compare.mjs [--tol 2.0] [--conf-tol 0.01]
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const arg = (n, d) => {
  const i = process.argv.indexOf(n);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d;
};
const TOL = parseFloat(arg('--tol', '2.0'));
const CTOL = parseFloat(arg('--conf-tol', '0.01'));

const py = JSON.parse(readFileSync(path.join(HERE, 'parity_python.json'), 'utf8'));
const js = JSON.parse(readFileSync(path.join(HERE, 'parity_js.json'), 'utf8'));
const jsBy = new Map(js.images.map((i) => [i.name, i]));

function iou(a, b) {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih;
  const ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return inter / ua;
}
const boxDelta = (a, b) => Math.max(
  Math.abs(a.x1 - b.x1), Math.abs(a.y1 - b.y1),
  Math.abs(a.x2 - b.x2), Math.abs(a.y2 - b.y2),
);

const pad = (s, n) => String(s).padEnd(n);
const lpad = (s, n) => String(s).padStart(n);

let fails = 0;
let matched = 0;
let worstBox = 0;
let worstConf = 0;
let worstPre = null;
let worstJpeg = null;
let worstJpegMean = null;
let worstPreMean = null;

const rows = [];
for (const p of py.images) {
  const j = jsBy.get(p.name);
  if (!j) { console.error(`no JS result for ${p.name}`); fails++; continue; }
  if (j.preprocess_vs_ultralytics) {
    worstPre = Math.max(worstPre ?? 0, j.preprocess_vs_ultralytics.max_abs_diff);
    worstPreMean = Math.max(worstPreMean ?? 0, j.preprocess_vs_ultralytics.mean_abs_diff);
  }
  if (j.jpegjs_vs_libjpeg) {
    worstJpeg = Math.max(worstJpeg ?? 0, j.jpegjs_vs_libjpeg.max_abs_diff);
    worstJpegMean = Math.max(worstJpegMean ?? 0, j.jpegjs_vs_libjpeg.mean_abs_diff);
  }

  const used = new Set();
  for (const d of p.detections) {
    let best = -1, bestI = 0;
    j.detections.forEach((c, k) => {
      if (used.has(k) || c.className !== d.class_name) return;
      const v = iou(d, c);
      if (v > bestI) { bestI = v; best = k; }
    });
    if (best < 0) {
      rows.push([p.name, d.class_name, d.raw, '--', '--', '--', 'MISSING in JS']);
      fails++;
      continue;
    }
    used.add(best);
    const c = j.detections[best];
    const bd = boxDelta(d, c);
    const cd = Math.abs(d.raw - c.raw);
    worstBox = Math.max(worstBox, bd);
    worstConf = Math.max(worstConf, cd);
    matched++;
    const ok = bd <= TOL && cd <= CTOL;
    if (!ok) fails++;
    rows.push([
      p.name, d.class_name, d.raw, c.raw, bestI, bd,
      ok ? 'match' : 'OUT OF TOLERANCE',
      `[${d.x1.toFixed(1)},${d.y1.toFixed(1)},${d.x2.toFixed(1)},${d.y2.toFixed(1)}]`,
      `[${c.x1.toFixed(1)},${c.y1.toFixed(1)},${c.x2.toFixed(1)},${c.y2.toFixed(1)}]`,
      c.calibrated,
    ]);
  }
  j.detections.forEach((c, k) => {
    if (used.has(k)) return;
    rows.push([p.name, c.className, '--', c.raw, '--', '--', 'EXTRA in JS']);
    fails++;
  });
}

const H = ['image', 'class', 'py conf', 'js conf', 'IoU', 'dpx', 'verdict', 'python xyxy', 'js xyxy', 'js calib'];
const W = [26, 16, 8, 8, 8, 8, 18, 30, 30, 9];
console.log(`\nPython ultralytics ${py.ultralytics} (conf ${py.conf}, iou ${py.iou}, imgsz ${py.imgsz})`);
console.log(`JS  ${js.runtime} on ${js.node}, shared site/js/detect.js\n`);
console.log(H.map((h, i) => pad(h, W[i])).join(' '));
console.log(W.map((w) => '-'.repeat(w)).join(' '));
for (const r of rows) {
  console.log(r.map((v, i) => {
    const s = typeof v === 'number' ? (i === 2 || i === 3 || i === 9 ? v.toFixed(4) : v.toFixed(i === 4 ? 5 : 4)) : v;
    return i >= 2 && i <= 5 ? lpad(s, W[i]) : pad(s, W[i]);
  }).join(' ').trimEnd());
}

const pyTotal = py.images.reduce((a, i) => a + i.detections.length, 0);
const jsTotal = js.images.reduce((a, i) => a + i.detections.length, 0);
const meds = js.images.map((i) => i.infer_ms_median).sort((a, b) => a - b);
console.log('');
console.log(`images                     ${py.images.length}`);
console.log(`python detections          ${pyTotal}`);
console.log(`js detections              ${jsTotal}`);
console.log(`matched pairs              ${matched}`);
console.log(`max box delta              ${worstBox.toFixed(4)} px  (tolerance ${TOL})`);
console.log(`max confidence delta       ${worstConf.toExponential(3)}  (tolerance ${CTOL})`);
console.log(`pixel source (JS side)     ${js.pixel_source ?? 'unknown'}`);
console.log(`max preprocess pixel delta ${worstPre === null ? 'not measured (run with --pixels)' : worstPre + ' / 255  (our fixed-point bilinear vs the tensor ultralytics built), mean ' + worstPreMean.toFixed(5)}`);
console.log(`max jpeg-js vs libjpeg     ${worstJpeg === null ? 'not measured (run with --pixels)' : worstJpeg + ' / 255  (source-pixel disagreement, absent in a browser), mean ' + worstJpegMean.toFixed(5)}`);
console.log(`median session.run         ${meds[Math.floor(meds.length / 2)].toFixed(2)} ms  (${js.timed_runs_per_image} timed runs/image, ${js.warmup_runs} warm-ups discarded)`);
console.log(`verdict                    ${fails === 0 ? 'PASS -- identical detections' : `FAIL -- ${fails} discrepancies`}`);

// Machine-readable summary, so the page can render this table from the same numbers
// instead of a human retyping them into HTML.
const out = arg('--json', null);
if (out) {
  writeFileSync(out, JSON.stringify({
    generated_at: new Date().toISOString().slice(0, 19) + 'Z',
    python: { ultralytics: py.ultralytics, weights: py.weights, conf: py.conf, iou: py.iou, imgsz: py.imgsz },
    js: { runtime: js.runtime, node: js.node, model: js.model, pixel_source: js.pixel_source },
    tolerance: { box_px: TOL, conf: CTOL },
    summary: {
      images: py.images.length,
      python_detections: pyTotal,
      js_detections: jsTotal,
      matched_pairs: matched,
      max_box_delta_px: worstBox,
      max_conf_delta: worstConf,
      max_preprocess_pixel_delta: worstPre,
      mean_preprocess_pixel_delta: worstPreMean,
      max_jpegjs_vs_libjpeg: worstJpeg,
      mean_jpegjs_vs_libjpeg: worstJpegMean,
      median_session_run_ms: meds[Math.floor(meds.length / 2)],
      timed_runs_per_image: js.timed_runs_per_image,
      warmup_runs: js.warmup_runs,
      discrepancies: fails,
      verdict: fails === 0 ? 'PASS' : 'FAIL',
    },
    rows: rows.map((r) => ({
      image: r[0], class: r[1], python_conf: r[2], js_conf: r[3], iou: r[4],
      box_delta_px: r[5], verdict: r[6], python_xyxy: r[7], js_xyxy: r[8], js_calibrated: r[9],
    })),
  }, null, 2));
  console.log(`\nwrote ${out}`);
}
process.exit(fails === 0 ? 0 : 1);
