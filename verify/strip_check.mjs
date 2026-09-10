/**
 * Wide-strip check: what the two 2048x1000 GC10 strip samples actually return.
 *
 * Runs the browser's own modules (../js/detect.js) under onnxruntime-node against the
 * same ONNX file the page downloads, two ways:
 *
 *   squash  one 2048x1000 -> 256x256 resize, aspect ratio destroyed (8.00x in x,
 *           3.91x in y). This is what site/js/app.js did before this change.
 *   tile    square windows of side = frame height, stepped across the width with
 *           overlap, each resized 1000 -> 256 (uniform 3.91x), boxes mapped back to
 *           strip pixels, then ONE class-aware global NMS over the union.
 *
 * Dev-only. Nothing here is served to the page.
 *
 * Usage, from site/verify:
 *   node strip_check.mjs --out strip_check.json
 */

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

import {
  preprocess, decode, nms, IMGSZ, SHIPPED_CONF, NMS_IOU, CLASS_NAMES,
} from '../js/detect.js';
import { calibrate } from '../js/calibration.js';
import { planTiles } from '../js/tiling.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SITE = path.resolve(HERE, '..');
const req = createRequire(path.join(HERE, 'package.json'));
const ort = req('onnxruntime-node');
const jpeg = req('jpeg-js');

const arg = (n, d = null) => {
  const i = process.argv.indexOf(n);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d;
};
const OUT = arg('--out', path.join(HERE, 'strip_check.json'));
const STRIPS = ['strip_rollmark.jpg', 'strip_weldline.jpg'];

function loadRGBA(name) {
  const img = jpeg.decode(readFileSync(path.join(SITE, 'samples', name)), { useTArray: true });
  return {
    rgba: new Uint8ClampedArray(img.data.buffer, img.data.byteOffset, img.data.length),
    width: img.width,
    height: img.height,
  };
}

/** Copy one axis-aligned window out of an RGBA buffer. Mirrors app.js' canvas crop. */
function crop(rgba, w, x0, y0, cw, ch) {
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const src = ((y0 + y) * w + x0) * 4;
    out.set(rgba.subarray(src, src + cw * 4), y * cw * 4);
  }
  return out;
}

const session = await ort.InferenceSession.create(
  path.join(SITE, 'model', 'yolov8n_neudet_best.onnx'),
  { executionProviders: ['cpu'], graphOptimizationLevel: 'all' },
);

async function run(tensor) {
  const out = await session.run({
    images: new ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]),
  });
  return { data: out.output0.data, numAnchors: out.output0.dims[2] };
}

const brief = (d) => ({
  className: d.className,
  raw: Number(d.raw.toFixed(4)),
  calibrated: Number(calibrate(d.raw).toFixed(4)),
  box: [d.x1, d.y1, d.x2, d.y2].map((v) => Number(v.toFixed(1))),
});

const report = { generated_at: new Date().toISOString(), conf: SHIPPED_CONF, iou: NMS_IOU, imgsz: IMGSZ, strips: [] };

for (const name of STRIPS) {
  const { rgba, width, height } = loadRGBA(name);

  // ---- squash: exactly one forward pass over a distorted whole frame ----------
  const tSq = performance.now();
  const { tensor } = preprocess(rgba, width, height, IMGSZ);
  const o = await run(tensor);
  const squash = decode(o.data, {
    conf: SHIPPED_CONF, iou: NMS_IOU, origW: width, origH: height,
    size: IMGSZ, numAnchors: o.numAnchors,
  });
  const squashMs = performance.now() - tSq;

  // ---- tile: the shared plan from ../js/tiling.js, one global NMS -------------
  const tiles = planTiles(width, height, IMGSZ);
  const tTi = performance.now();
  const boxes = [];
  const scores = [];
  const classes = [];
  for (const t of tiles) {
    const px = crop(rgba, width, t.x, t.y, t.w, t.h);
    const { tensor: tt } = preprocess(px, t.w, t.h, IMGSZ);
    const to = await run(tt);
    // Decode inside the tile's own pixel space, then translate to strip pixels.
    for (const d of decode(to.data, {
      conf: SHIPPED_CONF, iou: NMS_IOU, origW: t.w, origH: t.h,
      size: IMGSZ, numAnchors: to.numAnchors,
    })) {
      boxes.push([d.x1 + t.x, d.y1 + t.y, d.x2 + t.x, d.y2 + t.y]);
      scores.push(d.raw);
      classes.push(d.classId);
    }
  }
  const kept = nms(boxes, scores, classes, NMS_IOU);
  const tileMs = performance.now() - tTi;
  const tiled = kept.map((i) => {
    const [x1, y1, x2, y2] = boxes[i];
    return {
      classId: classes[i], className: CLASS_NAMES[classes[i]], raw: scores[i],
      x1, y1, x2, y2, areaFrac: ((x2 - x1) * (y2 - y1)) / (width * height),
    };
  }).sort((a, b) => b.raw - a.raw);

  report.strips.push({
    file: name,
    size: [width, height],
    squash: {
      scale_x: Number((width / IMGSZ).toFixed(3)),
      scale_y: Number((height / IMGSZ).toFixed(3)),
      aspect_distortion: Number(((width / IMGSZ) / (height / IMGSZ)).toFixed(3)),
      forward_passes: 1,
      ms: Number(squashMs.toFixed(1)),
      n: squash.length,
      detections: squash.map(brief),
    },
    tiled: {
      tiles: tiles.map((t) => [t.x, t.y, t.w, t.h]),
      scale: Number((tiles[0].h / IMGSZ).toFixed(3)),
      forward_passes: tiles.length,
      ms: Number(tileMs.toFixed(1)),
      candidates_before_global_nms: boxes.length,
      n: tiled.length,
      detections: tiled.map(brief),
    },
  });

  console.log(`\n=== ${name}  ${width}x${height} ===`);
  console.log(`squash: 1 pass, ${squashMs.toFixed(1)} ms, ${squash.length} detections`
    + `  (resize ${(width / IMGSZ).toFixed(2)}x in x vs ${(height / IMGSZ).toFixed(2)}x in y)`);
  for (const d of squash.map(brief)) console.log(`   ${d.className.padEnd(16)} raw ${d.raw.toFixed(3)}  cal ${d.calibrated.toFixed(3)}  [${d.box.join(', ')}]`);
  console.log(`tiled : ${tiles.length} passes, ${tileMs.toFixed(1)} ms, `
    + `${boxes.length} candidates -> ${tiled.length} after global NMS  (uniform ${(tiles[0].h / IMGSZ).toFixed(2)}x)`);
  for (const d of tiled.map(brief)) console.log(`   ${d.className.padEnd(16)} raw ${d.raw.toFixed(3)}  cal ${d.calibrated.toFixed(3)}  [${d.box.join(', ')}]`);
}

writeFileSync(OUT, `${JSON.stringify(report, null, 2)}\n`);
console.log(`\nwrote ${path.relative(SITE, OUT)}`);
