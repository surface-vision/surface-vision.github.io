/**
 * The browser half of the parity check.
 *
 * parity.mjs proves that the shared decode module agrees with Python under Node. That
 * leaves one honest gap: the page also has to get its pixels out of a <canvas> and its
 * arithmetic out of a WASM build of the runtime, and neither of those is exercised by
 * Node. So this drives the actual published page in real Chrome, catches the
 * `surface:detections` event the page dispatches after every run, and diffs the result
 * against the same Python reference.
 *
 * It also reports genuine browser timings: performance.now() around session.run inside
 * the page, with the warm-up call excluded.
 *
 * Requires a served copy of site/ and a Chrome binary:
 *   python -m http.server 8777 --bind 127.0.0.1    # from inside site/
 *   npm i puppeteer-core@24
 *   CHROME=/path/to/chrome node site/verify/browser_parity.mjs \
 *       --url http://127.0.0.1:8777/ --json site/verify/parity_browser.json
 *
 * CHROME defaults to the Puppeteer cache, then to Google Chrome's normal macOS path.
 */

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const base = process.env.ORT_MODULES_BASE
  ? path.join(path.resolve(process.env.ORT_MODULES_BASE), 'package.json')
  : path.join(HERE, 'package.json');
const puppeteer = createRequire(base)('puppeteer-core');

const arg = (n, d = null) => {
  const i = process.argv.indexOf(n);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d;
};

const URL_BASE = arg('--url', 'http://127.0.0.1:8777/');
const OUT = arg('--json', path.join(HERE, 'parity_browser.json'));
const PY = JSON.parse(readFileSync(arg('--python', path.join(HERE, 'parity_python.json')), 'utf8'));
const TOL_BOX = parseFloat(arg('--tol', '2.0'));
const TOL_CONF = parseFloat(arg('--conf-tol', '0.01'));

const CHROME_CANDIDATES = [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
];

function findChrome() {
  // Prefer a Puppeteer-managed build if one is cached: its version is pinned and
  // printed in the report, which a system Chrome's is not.
  const cache = `${process.env.HOME}/.cache/puppeteer/chrome`;
  if (existsSync(cache)) {
    for (const dir of readdirSync(cache).sort().reverse()) {
      for (const rel of [
        'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        'chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
        'chrome-linux64/chrome',
      ]) {
        const p = path.join(cache, dir, rel);
        if (existsSync(p)) return p;
      }
    }
  }
  for (const c of CHROME_CANDIDATES) if (existsSync(c)) return c;
  throw new Error('no Chrome found; set CHROME=/path/to/chrome');
}

const executablePath = process.env.CHROME && existsSync(process.env.CHROME)
  ? process.env.CHROME
  : findChrome();

const browser = await puppeteer.launch({ executablePath, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1000 });

const pageErrors = [];
const consoleErrors = [];
const badRequests = [];
page.on('pageerror', (e) => pageErrors.push(String(e.message)));
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('requestfailed', (r) => badRequests.push(['FAILED', r.url()]));
page.on('requestfinished', (r) => {
  const s = r.response()?.status();
  if (s && s >= 400) badRequests.push([s, r.url()]);
});

// Installed before any page script runs, so no event can be missed.
await page.evaluateOnNewDocument(() => {
  window.__caught = [];
  document.addEventListener('surface:detections', (e) => window.__caught.push(e.detail));
});
await page.goto(URL_BASE, { waitUntil: 'networkidle2', timeout: 120000 });

const chips = await page.$$('#chips .chip');
if (!chips.length) throw new Error('no sample chips on the page');
for (let i = 0; i < chips.length; i++) {
  await chips[i].click();
  // The first click also pays for the CDN runtime and the 12 MB model download.
  await page.waitForFunction((n) => window.__caught.length > n, { timeout: 180000, polling: 60 }, i);
}
const caught = await page.evaluate(() => window.__caught);
const chromeVersion = await browser.version();
await browser.close();

/* ------------------------------------------------------------------ compare ---- */

const byName = new Map(caught.map((c) => [c.image, c]));
const iou = (a, b) => {
  const iw = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const ih = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (iw <= 0 || ih <= 0) return 0;
  const it = iw * ih;
  return it / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - it);
};
const boxDelta = (a, b) => Math.max(
  Math.abs(a.x1 - b.x1), Math.abs(a.y1 - b.y1), Math.abs(a.x2 - b.x2), Math.abs(a.y2 - b.y2),
);

const pad = (s, n) => String(s).padEnd(n);
const lp = (s, n) => String(s).padStart(n);
const H = ['image', 'class', 'py raw', 'browser raw', 'IoU', 'box dpx', 'p(defect)', 'verdict'];
const W = [24, 16, 9, 12, 9, 9, 10, 12];

console.log(`\nPython ultralytics ${PY.ultralytics} (conf ${PY.conf}, iou ${PY.iou}, imgsz ${PY.imgsz})`);
console.log(`Browser ${chromeVersion}, onnxruntime-web on WASM, page at ${URL_BASE}\n`);
console.log(H.map((h, i) => pad(h, W[i])).join(' '));
console.log(W.map((w) => '-'.repeat(w)).join(' '));

let fails = 0; let matched = 0; let worstBox = 0; let worstConf = 0; let pyTot = 0; let jsTot = 0;
const rows = [];
for (const p of PY.images) {
  const j = byName.get(p.name);
  if (!j) { console.log(`${pad(p.name, W[0])} NO BROWSER RESULT`); fails++; continue; }
  pyTot += p.detections.length;
  jsTot += j.detections.length;
  const used = new Set();
  for (const d of p.detections) {
    let best = -1; let bi = 0;
    j.detections.forEach((c, k) => {
      if (used.has(k) || c.className !== d.class_name) return;
      const v = iou(d, c);
      if (v > bi) { bi = v; best = k; }
    });
    if (best < 0) {
      console.log([pad(p.name, W[0]), pad(d.class_name, W[1]), lp(d.raw.toFixed(4), W[2]),
        lp('--', W[3]), lp('--', W[4]), lp('--', W[5]), lp('--', W[6]), 'MISSING'].join(' '));
      rows.push({ image: p.name, class: d.class_name, python_raw: d.raw, verdict: 'MISSING' });
      fails++; continue;
    }
    used.add(best); matched++;
    const c = j.detections[best];
    const db = boxDelta(d, c);
    const dc = Math.abs(d.raw - c.raw);
    worstBox = Math.max(worstBox, db);
    worstConf = Math.max(worstConf, dc);
    const ok = db <= TOL_BOX && dc <= TOL_CONF;
    if (!ok) fails++;
    console.log([pad(p.name, W[0]), pad(d.class_name, W[1]), lp(d.raw.toFixed(4), W[2]),
      lp(c.raw.toFixed(4), W[3]), lp(bi.toFixed(5), W[4]), lp(db.toFixed(4), W[5]),
      lp(c.calibrated.toFixed(4), W[6]), ok ? 'match' : 'OUT OF TOL'].join(' '));
    rows.push({
      image: p.name, class: d.class_name, python_raw: d.raw, browser_raw: c.raw,
      iou: bi, box_delta_px: db, browser_calibrated: c.calibrated,
      python_xyxy: [d.x1, d.y1, d.x2, d.y2], browser_xyxy: [c.x1, c.y1, c.x2, c.y2],
      verdict: ok ? 'match' : 'OUT OF TOL',
    });
  }
  j.detections.forEach((c, k) => {
    if (used.has(k)) return;
    console.log([pad(p.name, W[0]), pad(c.className, W[1]), lp('--', W[2]), lp(c.raw.toFixed(4), W[3]),
      lp('--', W[4]), lp('--', W[5]), lp(c.calibrated.toFixed(4), W[6]), 'EXTRA'].join(' '));
    rows.push({ image: p.name, class: c.className, browser_raw: c.raw, verdict: 'EXTRA' });
    fails++;
  });
}

const inSet = caught.filter((c) => PY.images.some((p) => p.name === c.image));
const stat = (key) => {
  const v = inSet.map((c) => c.timing[key]).sort((a, b) => a - b);
  return { min: v[0], median: v[Math.floor(v.length / 2)], max: v[v.length - 1] };
};
const infer = stat('infer');
const pre = stat('pre');
const post = stat('post');

console.log('');
console.log(`images                    ${PY.images.length}`);
console.log(`python detections         ${pyTot}`);
console.log(`browser detections        ${jsTot}`);
console.log(`matched pairs             ${matched}`);
console.log(`max box delta             ${worstBox.toFixed(4)} px  (tolerance ${TOL_BOX})`);
console.log(`max raw confidence delta  ${worstConf.toExponential(3)}  (tolerance ${TOL_CONF})`);
console.log(`browser session.run       min ${infer.min.toFixed(1)} / median ${infer.median.toFixed(1)} / max ${infer.max.toFixed(1)} ms`);
console.log(`browser preprocess        min ${pre.min.toFixed(2)} / median ${pre.median.toFixed(2)} ms`);
console.log(`browser decode + NMS      min ${post.min.toFixed(3)} / median ${post.median.toFixed(3)} ms`);
console.log(`warm-up run (excluded)    ${caught[0].timing.first.toFixed(0)} ms`);
console.log(`page errors               ${pageErrors.length ? pageErrors.join(' | ') : 'none'}`);
console.log(`console errors            ${consoleErrors.length ? consoleErrors.slice(0, 3).join(' | ') : 'none'}`);
console.log(`failed / 4xx requests     ${badRequests.length ? JSON.stringify(badRequests) : 'none'}`);
console.log(`verdict                   ${fails === 0 ? 'PASS -- identical detections' : `FAIL -- ${fails} discrepancies`}`);

const strips = caught.filter((c) => !PY.images.some((p) => p.name === c.image));
if (strips.length) {
  // No Python reference for these: reference_ultralytics.py runs the twelve square
  // NEU-DET frames. What this shows is that the tiled path fires in the real browser
  // and on how many windows -- strip_check.mjs is where tiled is measured against
  // squashed on the same two frames.
  console.log('\nout-of-domain strip frames (GC10-DET, tiled, no Python reference):');
  for (const s of strips) {
    const t = s.tiles ? `${s.tiles.length} tile${s.tiles.length === 1 ? '' : 's'}` : 'tiles n/a';
    console.log(`  ${s.image}  ${s.width}x${s.height}  ${t}  ${s.detections.length} det  `
      + (s.detections.map((d) => `${d.className} ${d.raw.toFixed(3)}`).join(', ') || '(none)'));
  }
}

writeFileSync(OUT, JSON.stringify({
  generated_at: new Date().toISOString().slice(0, 19) + 'Z',
  chrome: chromeVersion,
  url: URL_BASE,
  python: { ultralytics: PY.ultralytics, conf: PY.conf, iou: PY.iou, imgsz: PY.imgsz },
  tolerance: { box_px: TOL_BOX, conf: TOL_CONF },
  summary: {
    images: PY.images.length,
    python_detections: pyTot,
    browser_detections: jsTot,
    matched_pairs: matched,
    max_box_delta_px: worstBox,
    max_conf_delta: worstConf,
    session_run_ms: infer,
    preprocess_ms: pre,
    decode_nms_ms: post,
    warmup_ms: caught[0].timing.first,
    page_errors: pageErrors,
    console_errors: consoleErrors,
    bad_requests: badRequests,
    discrepancies: fails,
    verdict: fails === 0 ? 'PASS' : 'FAIL',
  },
  rows,
  out_of_domain: strips.map((s) => ({
    tiles: s.tiles || null,
    image: s.image, width: s.width, height: s.height,
    detections: s.detections.map((d) => ({ className: d.className, raw: d.raw, calibrated: d.calibrated })),
  })),
}, null, 2));
console.log(`\nwrote ${OUT}`);
process.exit(fails === 0 ? 0 : 1);
