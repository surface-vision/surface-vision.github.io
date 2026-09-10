/**
 * Browser detector for the Jindal Stainless surface-defect model.
 *
 * Real inference, in the tab, with no backend: onnxruntime-web on the WASM backend
 * runs the same 12.1 MB yolov8n_neudet_best.onnx the Python console runs. Every
 * numeric step -- resize, normalise, decode, NMS, calibrate -- comes from
 * ./detect.js and ./calibration.js, and the window geometry for non-square frames
 * comes from ./tiling.js. site/verify runs those same modules under onnxruntime-node
 * against the Python model. See site/verify/parity_report.json for the result, and
 * site/verify/strip_check.json for the wide-strip comparison.
 */

import {
  CLASS_NAMES, CLASS_COLORS, IMGSZ, SHIPPED_CONF, NMS_IOU,
  preprocess, decode, nms,
} from './detect.js';
import { planTiles } from './tiling.js';
import { calibrate, CALIBRATION } from './calibration.js';
import { DEFECT_INFO, scoreDetection, severityBucket } from './defect-info.js';
import { wireConsole } from './console-embed.js';

/* onnxruntime-web, pinned to an exact version on both mirrors.
 *
 * ort.wasm.* is the WASM-ONLY build. The default ort.min.mjs pulls the JSEP/WebGPU
 * binary, which is 23.8 MB; this one pulls the plain 11.9 MB binary and nothing else.
 *
 * Two bases, tried in order, because a demo whose only job is to open reliably should
 * not have a single point of failure in a CDN. Both host the identical 1.23.2 files. */
const ORT_VERSION = '1.23.2';
const ORT_BASES = [
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  `https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/${ORT_VERSION}/`,
];
const MODEL_URL = new URL('../model/yolov8n_neudet_best.onnx', import.meta.url).href;

/**
 * The twelve NEU-DET test frames, then the two wide GC10 strip frames.
 *
 * Order is deliberate: the first chip a visitor clicks should be a class the model is
 * genuinely good at, so the classes run strongest first by held-out AP50 -- patches
 * 0.949, scratches 0.909, inclusion 0.827, pitted_surface 0.756, rolled-in_scale
 * 0.630, crazing 0.444 (reports/evaluation.json -> detection_metrics.per_class).
 * Leading with crazing showed the weakest class in the page's first impression.
 *
 * `file` is unchanged and is what the run reports and the parity harness matches on;
 * `label` is only what the chip says.
 */
const SAMPLES = [
  { file: 'patches_271.jpg', cls: 'patches', label: 'Patches 1' },
  { file: 'patches_272.jpg', cls: 'patches', label: 'Patches 2' },
  { file: 'scratches_271.jpg', cls: 'scratches', label: 'Scratches 1' },
  { file: 'scratches_272.jpg', cls: 'scratches', label: 'Scratches 2' },
  { file: 'inclusion_271.jpg', cls: 'inclusion', label: 'Inclusion 1' },
  { file: 'inclusion_272.jpg', cls: 'inclusion', label: 'Inclusion 2' },
  { file: 'pitted_surface_271.jpg', cls: 'pitted_surface', label: 'Pitted surface 1' },
  { file: 'pitted_surface_272.jpg', cls: 'pitted_surface', label: 'Pitted surface 2' },
  { file: 'rolled-in_scale_271.jpg', cls: 'rolled-in_scale', label: 'Rolled-in scale 1' },
  { file: 'rolled-in_scale_272.jpg', cls: 'rolled-in_scale', label: 'Rolled-in scale 2' },
  { file: 'crazing_271.jpg', cls: 'crazing', label: 'Crazing 1' },
  { file: 'crazing_272.jpg', cls: 'crazing', label: 'Crazing 2' },
  { file: 'strip_rollmark.jpg', cls: null, label: 'Wide strip: roll mark', wide: true },
  { file: 'strip_weldline.jpg', cls: null, label: 'Wide strip: weld line', wide: true },
];

const $ = (id) => document.getElementById(id);
const rgb = (name) => `rgb(${(CLASS_COLORS[name] || [255, 255, 255]).join(',')})`;
const fmt = (v, n = 3) => v.toFixed(n);

const state = {
  ort: null,
  session: null,
  loading: null,
  /**
   * One entry per forward pass of the last run -- { tile, data, numAnchors } -- kept
   * so the confidence slider re-decodes without touching the session. A square frame
   * has exactly one entry; a 2048x1000 strip has one per window.
   */
  passes: [],
  image: null,      // { bitmapOrImg, width, height, name }
  dets: [],
  selected: -1,
  conf: SHIPPED_CONF,
  timing: { infer: null, pre: null, post: null, first: null, tiles: null },
};

/* ---------------------------------------------------------------- status ---- */

function status(msg, isError = false) {
  const el = $('status');
  if (!el) return;
  el.textContent = msg;
  el.classList.toggle('err', isError);
}

function progress(frac) {
  const el = $('progbar');
  if (el) el.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
}

/** The bar is only shown while there is something to report. */
function progressVisible(on) {
  const el = $('progbar');
  if (el && el.parentElement) el.parentElement.classList.toggle('on', on);
}

/* ------------------------------------------------------------ model load ---- */

/**
 * Started eagerly on load (see preload() at the bottom) so that a visitor's first
 * click returns a result rather than beginning a 12.1 MB download -- but never
 * awaited by anything on the page, so nothing is blocked on it. A click that lands
 * mid-download joins this same promise instead of starting a second one.
 *
 * The ONNX file is streamed with fetch so the progress bar is real byte progress,
 * not a spinner pretending.
 */
async function ensureSession() {
  if (state.session) return state.session;
  if (state.loading) return state.loading;

  state.loading = (async () => {
    progressVisible(true);
    status('starting the detector...');
    let ort = null;
    let base = null;
    const tried = [];
    for (const candidate of ORT_BASES) {
      try {
        ort = await import(/* @vite-ignore */ `${candidate}ort.wasm.min.mjs`);
        base = candidate;
        break;
      } catch (err) {
        tried.push(`${new URL(candidate).host}: ${err.message}`);
      }
    }
    if (!ort) throw new Error(`no CDN reachable for onnxruntime-web -- ${tried.join(' | ')}`);

    // Without this the runtime looks for its .wasm next to THIS page and fails,
    // usually silently: no exception, just a session that never resolves. It must
    // point at the CDN directory the loader itself came from, trailing slash included.
    ort.env.wasm.wasmPaths = base;
    // GitHub Pages cannot send COOP/COEP, so SharedArrayBuffer is unavailable and
    // the threaded build would fall back anyway. Asking for 1 thread up front keeps
    // the failure out of the console and the start-up deterministic.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    state.ort = ort;

    status('downloading the model -- 12.1 MB, once...');
    const buf = await fetchWithProgress(MODEL_URL);
    progress(1);

    status('preparing the model...');
    const session = await ort.InferenceSession.create(buf, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    state.session = session;

    // Warm-up. The first run pays for WASM code-gen and arena allocation and is
    // reported separately; it is never mixed into the quoted inference time.
    const zeros = new Float32Array(3 * IMGSZ * IMGSZ);
    const t0 = performance.now();
    await session.run({ images: new ort.Tensor('float32', zeros, [1, 3, IMGSZ, IMGSZ]) });
    state.timing.first = performance.now() - t0;

    progressVisible(false);
    status('ready -- the model is loaded and running on your CPU. Click a sample.');
    return session;
  })().catch((err) => {
    state.loading = null;
    progressVisible(false);
    status(`the model could not load: ${err.message}`, true);
    throw err;
  });

  return state.loading;
}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !total) return await res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    progress(got / total);
    status(`downloading the model -- ${(got / 1048576).toFixed(1)} of ${(total / 1048576).toFixed(1)} MB`);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

/* ------------------------------------------------------------- inference ---- */

/** Original-size RGBA out of any drawable source. drawImage at natural size only. */
function pixelsOf(src, w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingEnabled = false;
  g.drawImage(src, 0, 0);
  return g.getImageData(0, 0, w, h).data;
}

/** Copy one axis-aligned window out of an RGBA buffer. Same code as the harness. */
function cropRGBA(rgba, w, x0, y0, cw, ch) {
  const out = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    const src = ((y0 + y) * w + x0) * 4;
    out.set(rgba.subarray(src, src + cw * 4), y * cw * 4);
  }
  return out;
}

/**
 * One run over one frame.
 *
 * The network input is a fixed 256x256. A square frame goes straight in. A frame that
 * is not square is cut into square windows by ./tiling.js and each window goes in on
 * its own, because squashing a 2048x1000 strip into the square compresses it 8.00x
 * across the width against 3.91x down the height -- an anisotropy the weights never
 * saw in training, and one that measurably costs detections. site/verify/strip_check.json
 * has strip_weldline returning 0 detections squashed against 3 tiled, and strip_rollmark
 * returning six boxes all labelled inclusion squashed against a rolled-in_scale call on
 * the mark itself tiled.
 *
 * Pixels are read ONCE at natural size and the windows are cut out of that buffer, so
 * the browser and site/verify/strip_check.mjs crop identically.
 */
async function runOn(source, width, height, name) {
  const session = await ensureSession();
  state.image = { source, width, height, name };

  const tiles = planTiles(width, height, IMGSZ);
  if (tiles.length > 1) {
    status(`${name} -- ${width}x${height}, ${tiles.length} tiled passes...`);
  }

  let pre = 0;
  let infer = 0;
  const passes = [];

  const tRead = performance.now();
  const rgba = pixelsOf(source, width, height);
  pre += performance.now() - tRead;

  for (const t of tiles) {
    const t0 = performance.now();
    const px = tiles.length === 1 ? rgba : cropRGBA(rgba, width, t.x, t.y, t.w, t.h);
    const { tensor } = preprocess(px, t.w, t.h, IMGSZ);
    pre += performance.now() - t0;

    const feeds = { images: new state.ort.Tensor('float32', tensor, [1, 3, IMGSZ, IMGSZ]) };
    const t1 = performance.now();
    const out = await session.run(feeds);
    infer += performance.now() - t1;

    passes.push({ tile: t, data: out.output0.data, numAnchors: out.output0.dims[2] });
  }

  state.passes = passes;
  state.timing.pre = pre;
  state.timing.infer = infer;
  state.timing.tiles = tiles.length;

  refilter();
  // The verdict line above states the ANSWER; this line states how it was computed, so
  // the two do not say the same thing twice.
  status(`${name} -- ${width}x${height}, `
    + (tiles.length > 1
      ? `${tiles.length} tiled passes merged by one global NMS`
      : 'single pass')
    + `, conf ${fmt(state.conf, 2)}`);
}

/**
 * Re-decode from the cached output tensors. No session.run, so moving the slider is
 * free -- but it is a genuine re-decode, not a filter over an old detection list:
 * the confidence threshold changes which candidates enter NMS, so a post-hoc filter
 * would show boxes the model would not have emitted at that threshold.
 *
 * Each window is decoded in its OWN pixel space, so decode()'s internal NMS runs at the
 * uniform scale ultralytics would have used, and the boxes are then translated into
 * frame pixels. Only when there is more than one window does a second, global,
 * class-aware NMS run over the union, to merge the duplicate calls that the deliberate
 * window overlap produces. With a single window that pass would be a no-op -- nothing
 * survives decode()'s own NMS still holding IoU > 0.45 against a same-class survivor --
 * so it is skipped outright and the single-frame path stays identical to the one
 * site/verify checks against Python.
 */
function refilter() {
  if (!state.passes.length || !state.image) return;
  const { width, height } = state.image;
  const t0 = performance.now();

  let merged;
  if (state.passes.length === 1) {
    const { data, numAnchors, tile } = state.passes[0];
    merged = decode(data, {
      conf: state.conf, iou: NMS_IOU,
      origW: tile.w, origH: tile.h, size: IMGSZ, numAnchors,
    });
  } else {
    const boxes = [];
    const scores = [];
    const classes = [];
    for (const { data, numAnchors, tile } of state.passes) {
      for (const d of decode(data, {
        conf: state.conf, iou: NMS_IOU,
        origW: tile.w, origH: tile.h, size: IMGSZ, numAnchors,
      })) {
        boxes.push([d.x1 + tile.x, d.y1 + tile.y, d.x2 + tile.x, d.y2 + tile.y]);
        scores.push(d.raw);
        classes.push(d.classId);
      }
    }
    merged = nms(boxes, scores, classes, NMS_IOU).map((i) => {
      const [x1, y1, x2, y2] = boxes[i];
      return {
        classId: classes[i], className: CLASS_NAMES[classes[i]], raw: scores[i],
        x1, y1, x2, y2, width: x2 - x1, height: y2 - y1,
        areaFrac: ((x2 - x1) * (y2 - y1)) / (width * height),
      };
    });
  }

  state.dets = merged.map((d) => {
    const cal = calibrate(d.raw);
    const score = scoreDetection(d.className, d.raw, d.areaFrac);
    return { ...d, calibrated: cal, score, band: severityBucket(score) };
  }).sort((a, b) => b.raw - a.raw);
  state.timing.post = performance.now() - t0;
  state.selected = -1;
  render();

  // Public integration point, and the hook the browser half of the parity check uses.
  // Everything the table shows, at full precision, plus the timings, so the page's own
  // output can be diffed against the Python model from outside the page. Also what an
  // embedder would listen to.
  document.dispatchEvent(new CustomEvent('surface:detections', {
    detail: {
      image: state.image.name,
      width: state.image.width,
      height: state.image.height,
      conf: state.conf,
      iou: NMS_IOU,
      imgsz: IMGSZ,
      // The window plan, so an audit can tell a tiled run from a single-pass one
      // without inferring it from the frame size.
      tiles: state.passes.map(({ tile }) => ({ x: tile.x, y: tile.y, w: tile.w, h: tile.h })),
      timing: { ...state.timing },
      detections: state.dets.map((d) => ({
        className: d.className, classId: d.classId,
        raw: d.raw, calibrated: d.calibrated,
        x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2,
        areaFrac: d.areaFrac, score: d.score, band: d.band,
      })),
    },
  }));
}

/* --------------------------------------------------------------- drawing ---- */

function render() {
  drawCanvas();
  drawVerdict();
  drawTable();
  drawGuidance();
  drawTiming();
}

/**
 * The one-line answer, directly under the frame and above the fold: how many
 * detections, the strongest one, its severity band and what the forward pass cost.
 * The six-column table below the fold is the same answer in full.
 */
function drawVerdict() {
  const el = $('verdict');
  if (!el) return;
  if (!state.image) {
    el.className = 'verdict none';
    el.textContent = 'Nothing scored yet.';
    return;
  }
  const n = state.dets.length;
  if (!n) {
    el.className = 'verdict none';
    el.textContent = `No defect above the threshold in ${state.image.name}.`;
    return;
  }
  const top = state.dets[0];
  const t = state.timing;
  el.className = 'verdict';
  el.style.setProperty('--cc', rgb(top.className));
  el.innerHTML = `<span class="dot"></span>`
    + `<span class="nm">${n} detection${n === 1 ? '' : 's'}</span>`
    + `<span class="sep">&middot;</span>`
    // The calibrated probability, said the way a person would say it. Floored rather
    // than rounded so the wording can never claim more than the number does.
    + `<span><span class="nm">${top.className.replace(/_/g, ' ')}</span>, `
    + `${Math.floor(top.calibrated * 100)}% confidence</span>`
    + `<span class="sep">&middot;</span><span class="pill ${top.band}">${top.band}</span>`
    // The timing sits at the far end with no separator in front of it, so that when
    // the line wraps on a narrow panel it does not leave a dangling middot behind.
    + (t.infer === null ? ''
      : `<span class="ms mono">${t.infer.toFixed(0)} ms`
        + `${t.tiles > 1 ? ` over ${t.tiles} passes` : ''}</span>`);
}

function drawCanvas() {
  const { source, width, height } = state.image;
  const cvs = $('view');
  const hint = $('dropHint');
  if (hint) hint.style.display = 'none';
  cvs.style.display = 'block';
  // The dashed "drop something here" border has done its job once there is a frame in
  // the box; leaving it up makes a real result look like an empty placeholder.
  cvs.parentElement.classList.add('has-image');

  // DISPLAY size first. The panel width is one bound; the canvas's CSS max-height is
  // the other, and it is the one that matters now that the detector sits above the
  // fold -- without it a 200x200 sample would blow up to the full panel width and
  // push the sample chips off the screen. Reading the cap back from the stylesheet
  // keeps the two in one place: CSS decides how tall the frame may be, and every
  // number below is derived from the size actually shown.
  const avail = Math.max(1, cvs.parentElement.clientWidth - 2);
  const capH = parseFloat(getComputedStyle(cvs).maxHeight);
  const capW = Number.isFinite(capH) ? capH * (width / height) : Infinity;
  const box = Math.max(1, Math.min(avail, capW));

  // Render at up to 2x the displayed box for crisp boxes on a 200 px source, capped so
  // a 2048 px strip does not allocate a needless 4096 px canvas.
  const scale = Math.min(Math.max(box / width, 1), 2048 / Math.max(width, height), 6);
  cvs.width = Math.round(width * scale);
  cvs.height = Math.round(height * scale);
  cvs.style.width = `${Math.round(box)}px`;
  cvs.style.height = 'auto';

  const g = cvs.getContext('2d');
  g.imageSmoothingEnabled = scale < 1;
  g.clearRect(0, 0, cvs.width, cvs.height);
  g.drawImage(source, 0, 0, cvs.width, cvs.height);

  // Chip and stroke sizes are chosen in DISPLAY pixels, then converted into canvas
  // pixels, because the two differ in both directions here: a 200 px frame is drawn
  // into a 541 px canvas (magnified), while a 2048 px strip is drawn 1:1 into a canvas
  // the browser then shows at 541 CSS px (shrunk 3.8x). Sizing off the render scale
  // alone made the strip's labels three CSS pixels tall and unreadable.
  const k = cvs.width / Math.max(box, 1);
  const lw = Math.max(1, 1.6 * k);
  const fs = Math.max(9, Math.round(13 * k));
  const mono = getComputedStyle(document.body).getPropertyValue('--mono').trim();
  g.font = `600 ${fs}px ${mono}`;
  g.textBaseline = 'top';

  const pad = Math.round(fs * 0.34);
  const chipH = fs + pad * 2;

  // Boxes first, then every chip on top, so a later box's edge never cuts an
  // earlier box's label in half.
  const chips = [];

  state.dets.forEach((d, i) => {
    const sel = i === state.selected;
    const col = rgb(d.className);
    const x = d.x1 * scale;
    const y = d.y1 * scale;
    const w = (d.x2 - d.x1) * scale;
    const h = (d.y2 - d.y1) * scale;

    g.lineWidth = sel ? lw * 2 : lw;
    g.strokeStyle = col;
    g.globalAlpha = state.selected === -1 || sel ? 1 : 0.45;
    g.strokeRect(x, y, w, h);
    if (sel) {
      g.fillStyle = col.replace('rgb(', 'rgba(').replace(')', ',0.14)');
      g.fillRect(x, y, w, h);
    }
    g.globalAlpha = 1;
    chips.push({ d, sel, col, x, y, h });
  });

  for (const { d, sel, col, x, y, h } of chips) {
    let text = `${d.className} ${fmt(d.calibrated, 2)}`;
    let tw = g.measureText(text).width;
    // A chip wider than the frame is useless; drop to the score alone, then to
    // nothing, rather than painting over the defect the judge is trying to see.
    if (tw + pad * 2 > cvs.width) {
      text = fmt(d.calibrated, 2);
      tw = g.measureText(text).width;
      if (tw + pad * 2 > cvs.width) continue;
    }
    const chipW = tw + pad * 2;
    // Keep it inside the canvas on both axes: right-aligned if the box runs off the
    // right edge, below the box if there is no room above.
    const cx = Math.max(0, Math.min(x, cvs.width - chipW));
    // Above the box by preference. With no room above -- a detection touching the top
    // edge of the frame, which is common on a strip crop -- drop the chip INSIDE the
    // box, but only if the box is tall enough to still show as a box afterwards. A
    // 70 px box under a 78 px chip would be swallowed whole and the judge would see a
    // label floating over bare steel, so in that case put the chip below the box.
    const cy = y - chipH >= 0 ? y - chipH
      : (h >= chipH * 1.25 ? y
        : Math.min(y + h, cvs.height - chipH));
    g.globalAlpha = state.selected === -1 || sel ? 1 : 0.5;
    g.fillStyle = col;
    g.fillRect(cx, cy, chipW, chipH);
    g.fillStyle = '#0B0E12';
    g.fillText(text, cx + pad, cy + pad);
    g.globalAlpha = 1;
  }
}

function drawTable() {
  const body = $('resBody');
  const empty = $('resEmpty');
  body.innerHTML = '';
  const n = state.dets.length;
  $('detCount').textContent = `${n} detection${n === 1 ? '' : 's'}`;
  empty.style.display = n ? 'none' : 'block';
  $('resTable').style.display = n ? 'table' : 'none';
  if (!n) return;

  state.dets.forEach((d, i) => {
    const tr = document.createElement('tr');
    if (i === state.selected) tr.className = 'sel';
    // Column order is the reading order of the answer: what, how sure, how bad, how
    // big -- then the two engineering columns, which are the ones allowed to scroll
    // off the right edge of a narrow panel.
    tr.innerHTML = `
      <td class="mono"><span class="swatch" style="background:${rgb(d.className)}"></span>${d.className}</td>
      <td class="num">${fmt(d.calibrated)}</td>
      <td><span class="pill ${d.band}">${d.band}</span></td>
      <td class="num">${(d.areaFrac * 100).toFixed(1)}%</td>
      <td class="num" style="color:var(--muted)">${fmt(d.raw)}</td>
      <td class="num" title="${Math.round(d.width)} x ${Math.round(d.height)} px">${Math.round(d.x1)}, ${Math.round(d.y1)}, ${Math.round(d.x2)}, ${Math.round(d.y2)}</td>`;
    tr.addEventListener('click', () => {
      state.selected = state.selected === i ? -1 : i;
      render();
    });
    body.appendChild(tr);
  });
}

function drawGuidance() {
  const host = $('guide');
  host.innerHTML = '';
  if (!state.dets.length) {
    host.innerHTML = '<p class="empty">No detection above the threshold. Operator guidance appears here, one card per detection.</p>';
    return;
  }
  // One card per distinct class -- the guidance is per defect type, and six copies of
  // the same paragraph is not more informative than one.
  const seen = new Map();
  for (const d of state.dets) {
    const cur = seen.get(d.className);
    if (!cur || d.raw > cur.raw) seen.set(d.className, d);
  }
  for (const [name, d] of seen) {
    const info = DEFECT_INFO[name];
    const count = state.dets.filter((x) => x.className === name).length;
    const card = document.createElement('div');
    card.className = 'gcard';
    card.style.setProperty('--cc', rgb(name));
    card.innerHTML = `
      <div class="top">
        <span class="nm">${name}</span>
        <span class="pill ${severityBucket(scoreDetection(name, d.raw, d.areaFrac))}">${info.severity} base tier</span>
        <span class="cf">${count} box${count === 1 ? '' : 'es'} &middot; best p=${fmt(d.calibrated, 2)}</span>
      </div>
      <dl>
        <dt>Likely cause</dt><dd>${info.cause}</dd>
        <dt>Recommended action</dt><dd>${info.action}</dd>
      </dl>`;
    host.appendChild(card);
  }
}

function drawTiming() {
  const t = state.timing;
  // With tiling the session.run figure is a sum over windows, and saying so is the
  // difference between an honest number and one that looks like a regression.
  $('tInfer').textContent = t.infer === null
    ? '--'
    : `${t.infer.toFixed(1)} ms${t.tiles > 1 ? ` / ${t.tiles} passes` : ''}`;
  $('tPre').textContent = t.pre === null ? '--' : `${t.pre.toFixed(1)} ms`;
  $('tPost').textContent = t.post === null ? '--' : `${t.post.toFixed(2)} ms`;
  $('tFirst').textContent = t.first === null ? '--' : `${t.first.toFixed(0)} ms`;
}

/* ----------------------------------------------------------------- input ---- */

function loadImageFromURL(url, name) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${name}`));
    img.decoding = 'sync';
    img.src = url;
  });
}

async function useSample(s, chip) {
  document.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
  if (chip) chip.classList.add('active');
  try {
    status(`loading ${s.file}...`);
    const img = await loadImageFromURL(`samples/${s.file}`, s.file);
    await runOn(img, img.naturalWidth, img.naturalHeight, s.file);
  } catch (err) {
    status(err.message, true);
  }
}

async function useFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    status('that is not an image file', true);
    return;
  }
  document.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImageFromURL(url, file.name);
    await runOn(img, img.naturalWidth, img.naturalHeight, file.name);
  } catch (err) {
    status(err.message, true);
  } finally {
    URL.revokeObjectURL(url);
  }
}

/* ------------------------------------------------------------------- init ---- */

function buildChips() {
  const host = $('chips');
  if (!host) return;
  let noted = false;
  SAMPLES.forEach((s, i) => {
    // The two wide frames take a different route through the model, so they are
    // announced rather than dropped in among the square ones.
    if (s.wide && !noted) {
      noted = true;
      const note = document.createElement('div');
      note.className = 'row-note';
      note.textContent = '2048 x 1000 line-scan frames -- these are cut into three windows:';
      host.appendChild(note);
    }
    const b = document.createElement('button');
    b.className = `chip${s.wide ? ' wide' : ''}${i === 0 ? ' suggest' : ''}`;
    b.type = 'button';
    b.dataset.sample = s.file;
    if (s.cls) b.style.setProperty('--cc', rgb(s.cls));
    b.textContent = s.label || s.file.replace(/\.jpg$/, '');
    b.addEventListener('click', () => useSample(s, b));
    host.appendChild(b);
  });
}

/**
 * Buttons elsewhere on the page that run a named sample -- the strip section's
 * "run the roll-mark strip". They drive the chip itself rather than duplicating the
 * run path, so there is exactly one way a sample gets scored.
 */
function wireSampleButtons() {
  for (const el of document.querySelectorAll('[data-run-sample]')) {
    el.addEventListener('click', () => {
      const chip = document.querySelector(`#chips .chip[data-sample="${el.dataset.runSample}"]`);
      if (!chip) return;
      chip.click();
      document.getElementById('drop')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    });
  }
}

function wireDrop() {
  const drop = $('drop');
  if (!drop) return;
  ['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => {
    ev.preventDefault();
    drop.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, (ev) => {
    ev.preventDefault();
    drop.classList.remove('over');
  }));
  drop.addEventListener('drop', (ev) => {
    const f = ev.dataTransfer?.files?.[0];
    if (f) useFile(f);
  });
  $('file').addEventListener('change', (ev) => useFile(ev.target.files[0]));
  $('pick').addEventListener('click', () => $('file').click());
  window.addEventListener('paste', (ev) => {
    const f = [...(ev.clipboardData?.files || [])][0];
    if (f) useFile(f);
  });
}

function wireSlider() {
  const sl = $('conf');
  if (!sl) return;
  sl.value = String(SHIPPED_CONF);
  const paint = () => {
    state.conf = parseFloat(sl.value);
    const shipped = Math.abs(state.conf - SHIPPED_CONF) < 1e-9;
    $('confVal').textContent = fmt(state.conf, 2) + (shipped ? ' *' : '');
    $('confNote').textContent = shipped
      ? 'the shipped operating point (reports/operating_point.json)'
      : `moved off the shipped 0.15; p(defect) at this raw score is ${fmt(calibrate(state.conf), 3)}`;
  };
  sl.addEventListener('input', () => { paint(); refilter(); });
  paint();
}

function fillFacts() {
  const set = (id, text) => { const el = $(id); if (el) el.textContent = text; };
  set('calFacts',
    `${CALIBRATION.method}, ${CALIBRATION.nKnots} knots, fitted on ${CALIBRATION.fitSplit}, `
    + `ECE ${CALIBRATION.eceBefore.toFixed(4)} -> ${CALIBRATION.eceAfter.toFixed(4)} on ${CALIBRATION.evalSplit}`);
  set('ortVer', `onnxruntime-web ${ORT_VERSION} / wasm`);
  set('classList', CLASS_NAMES.join('  '));
}

/**
 * The recorded latency band inside the timing disclosure is read from the artefact
 * rather than typed into the HTML, so the page cannot drift away from the run that
 * produced it. Fetched on first open, not on load: nothing on the front page should
 * pay for a 25 KB report that a reader may never expand.
 */
function wireLatencyFact() {
  const host = $('latencyFact');
  const det = host && host.closest('details');
  if (!det) return;
  let done = false;
  det.addEventListener('toggle', async () => {
    if (done || !det.open) return;
    done = true;
    try {
      const r = await fetch('verify/parity_browser.json');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const p = await r.json();
      const t = p.summary.session_run_ms;
      host.textContent = `min ${t.min.toFixed(1)}, median ${t.median.toFixed(1)}, `
        + `max ${t.max.toFixed(1)} ms over ${p.summary.images} frames, in ${p.chrome}`;
    } catch (err) {
      done = false;
      host.textContent = `(verify/parity_browser.json unreadable: ${err.message})`;
    }
  });
}

/* ------------------------------------------------------------------ boot ---- */

buildChips();
wireSampleButtons();
wireLatencyFact();
wireDrop();
wireSlider();
wireConsole();
fillFacts();
drawTiming();
drawVerdict();
status('starting the detector...');

/**
 * Warm the runtime and the 12.1 MB model as soon as the browser is idle.
 *
 * The old page fetched nothing until the first click, which meant the first click
 * bought a CDN round trip, a 12.1 MB download and a session build before it returned
 * anything. Starting it here turns that into a click that answers. It is deliberately
 * not awaited and nothing renders behind it: the page is fully usable while it runs,
 * the progress bar under the frame reports real byte progress, and a click that
 * arrives mid-download joins the same promise rather than starting a second one.
 */
function preload() {
  ensureSession().catch(() => { /* status() has already said so, in the panel */ });
}
if ('requestIdleCallback' in window) requestIdleCallback(preload, { timeout: 1500 });
else setTimeout(preload, 250);
