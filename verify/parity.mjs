/**
 * Runs the browser's EXACT decode path in Node, against the same ONNX file the page
 * downloads, and compares detection for detection with the Python ultralytics model.
 *
 * The only thing this file does that the browser does not is get pixels: the browser
 * uses <canvas> + drawImage at natural size, this uses jpeg-js. Everything after that
 * -- the resize, the /255 NCHW packing, the [1,10,1344] decode, the NMS, the
 * calibration -- is imported from ../js/detect.js and ../js/calibration.js, the same
 * two modules the page loads. There is no second implementation to drift.
 *
 * To rule the JPEG decoder out as a variable, --pixels <dir> feeds Node the raw RGB
 * bytes that cv2 handed ultralytics (dumped by reference_ultralytics.py
 * --dump-pixels), so the two paths start from byte-identical source pixels.
 *
 * Usage (from the project root):
 *   ORT_MODULES_BASE=/path/with/node_modules \
 *   node site/verify/parity.mjs --pixels /tmp/refpx --out site/verify/parity_js.json
 */

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import { preprocess, decode, IMGSZ, SHIPPED_CONF, NMS_IOU, CLASS_NAMES } from '../js/detect.js';
import { calibrate, CALIBRATION_EXAMPLES } from '../js/calibration.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');

// onnxruntime-node and jpeg-js are dev-only: they never ship to the page. Resolve
// them from wherever they were installed rather than vendoring node_modules into a
// directory that gets committed to GitHub Pages.
const base = process.env.ORT_MODULES_BASE
  ? path.join(path.resolve(process.env.ORT_MODULES_BASE), 'package.json')
  : path.join(HERE, 'package.json');
const req = createRequire(base);
const ort = req('onnxruntime-node');
const jpeg = req('jpeg-js');

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const MODEL = path.join(ROOT, 'site', 'model', 'yolov8n_neudet_best.onnx');
const PY = JSON.parse(readFileSync(arg('--python', path.join(HERE, 'parity_python.json')), 'utf8'));
const PIXELS = arg('--pixels', null);
// Forces the jpeg-js source path while keeping the cv2 dumps available for the delta
// measurements, so run B can attribute its own residual instead of asserting it.
const FORCE_JPEG = process.argv.includes('--force-jpeg');
const OUT = arg('--out', path.join(HERE, 'parity_js.json'));
const WARMUP = 3;
const TIMED = 20;

// Guard: a transcription slip in the calibration knots must fail here, loudly,
// rather than quietly shifting every probability the page prints.
for (const [rawStr, expected] of Object.entries(CALIBRATION_EXAMPLES)) {
  const got = calibrate(parseFloat(rawStr));
  if (Math.abs(got - expected) > 1e-12) {
    throw new Error(`calibration mismatch at ${rawStr}: ${got} != ${expected}`);
  }
}

/** Original-size RGBA, either from the cv2 dump or from the JPEG itself. */
function loadRGBA(name) {
  const stem = name.replace(/\.[^.]+$/, '');
  if (PIXELS && !FORCE_JPEG) {
    const meta = path.join(PIXELS, `${stem}.meta.json`);
    const raw = path.join(PIXELS, `${stem}.orig.rgb`);
    if (existsSync(meta) && existsSync(raw)) {
      const [h, w] = JSON.parse(readFileSync(meta, 'utf8')).orig;
      const rgb = readFileSync(raw);
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0, n = w * h; i < n; i++) {
        rgba[i * 4] = rgb[i * 3];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
      }
      return { rgba, width: w, height: h, source: 'cv2-dump' };
    }
  }
  const buf = readFileSync(path.join(ROOT, 'data', 'neu-det', 'test', 'images', name));
  const img = jpeg.decode(buf, { useTArray: true });
  return {
    rgba: new Uint8ClampedArray(img.data.buffer, img.data.byteOffset, img.data.length),
    width: img.width,
    height: img.height,
    source: 'jpeg-js',
  };
}

/**
 * How far jpeg-js' IDCT is from the libjpeg-turbo one cv2 used. This is the ONE
 * variable the browser does not share with this harness: a browser decodes JPEG with
 * libjpeg-turbo, the same library cv2 uses, while Node has to use a pure-JS decoder.
 * Measuring it here is what lets the jpeg-js run's residual be attributed rather
 * than hand-waved.
 */
function jpegDecoderDelta(name) {
  if (!PIXELS) return null;
  const stem = name.replace(/\.[^.]+$/, '');
  const metaP = path.join(PIXELS, `${stem}.meta.json`);
  const rawP = path.join(PIXELS, `${stem}.orig.rgb`);
  if (!existsSync(metaP) || !existsSync(rawP)) return null;
  const [h, w] = JSON.parse(readFileSync(metaP, 'utf8')).orig;
  const ref = readFileSync(rawP);
  const img = jpeg.decode(
    readFileSync(path.join(ROOT, 'data', 'neu-det', 'test', 'images', name)),
    { useTArray: true },
  );
  if (img.width !== w || img.height !== h) return null;
  let max = 0;
  let sum = 0;
  const n = w * h * 3;
  for (let i = 0, px = 0; px < w * h; px++) {
    for (let c = 0; c < 3; c++, i++) {
      const d = Math.abs(img.data[px * 4 + c] - ref[i]);
      if (d > max) max = d;
      sum += d;
    }
  }
  return { max_abs_diff: max, mean_abs_diff: sum / n };
}

/** Max |a - b| between our resized uint8 tensor and the one ultralytics built. */
function preprocessDelta(resized, stem) {
  if (!PIXELS) return null;
  const p = path.join(PIXELS, `${stem}.rgb`);
  if (!existsSync(p)) return null;
  const ref = readFileSync(p); // HWC RGB uint8, IMGSZ x IMGSZ x 3
  let max = 0;
  let sum = 0;
  const n = IMGSZ * IMGSZ * 3;
  for (let i = 0, px = 0; i < n; px++) {
    for (let c = 0; c < 3; c++, i++) {
      const d = Math.abs(resized[px * 4 + c] - ref[i]);
      if (d > max) max = d;
      sum += d;
    }
  }
  return { max_abs_diff: max, mean_abs_diff: sum / n };
}

const session = await ort.InferenceSession.create(MODEL, {
  executionProviders: ['cpu'],
  graphOptimizationLevel: 'all',
});

const results = [];
let warmDone = false;

for (const img of PY.images) {
  const { rgba, width, height, source } = loadRGBA(img.name);
  const { tensor, resized } = preprocess(rgba, width, height, IMGSZ);
  const feeds = { images: new ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]) };

  if (!warmDone) {
    for (let i = 0; i < WARMUP; i++) await session.run(feeds);
    warmDone = true;
  }

  const times = [];
  let out = null;
  for (let i = 0; i < TIMED; i++) {
    const t0 = performance.now();
    out = await session.run(feeds);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);

  const o = out.output0;
  const dets = decode(o.data, {
    conf: SHIPPED_CONF, iou: NMS_IOU, origW: width, origH: height, size: IMGSZ,
    numAnchors: o.dims[2],
  }).map((d) => ({ ...d, calibrated: calibrate(d.raw) }));
  dets.sort((a, b) => b.raw - a.raw);

  results.push({
    name: img.name,
    width, height,
    pixel_source: source,
    output_dims: o.dims,
    preprocess_vs_ultralytics: preprocessDelta(resized, img.name.replace(/\.[^.]+$/, '')),
    jpegjs_vs_libjpeg: jpegDecoderDelta(img.name),
    infer_ms_median: times[Math.floor(times.length / 2)],
    infer_ms_min: times[0],
    infer_ms_max: times[times.length - 1],
    detections: dets,
  });
}

const payload = {
  runtime: `onnxruntime-node ${req('onnxruntime-node/package.json').version}`,
  node: process.version,
  model: path.relative(ROOT, MODEL),
  conf: SHIPPED_CONF,
  iou: NMS_IOU,
  imgsz: IMGSZ,
  class_names: CLASS_NAMES,
  pixel_source: PIXELS && !FORCE_JPEG
    ? 'cv2 dump (JPEG decoder held constant)'
    : 'jpeg-js (Node has no libjpeg; a browser does not have this gap)',
  warmup_runs: WARMUP,
  timed_runs_per_image: TIMED,
  images: results,
};
writeFileSync(OUT, JSON.stringify(payload, null, 2));

const total = results.reduce((a, r) => a + r.detections.length, 0);
const med = results.map((r) => r.infer_ms_median).sort((a, b) => a - b);
console.log(
  `wrote ${OUT}: ${results.length} images, ${total} detections, ` +
  `median session.run ${med[Math.floor(med.length / 2)].toFixed(2)} ms`
);
