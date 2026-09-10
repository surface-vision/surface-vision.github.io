/**
 * The verification tables on technical.html.
 *
 * These render site/verify/parity_report.json and site/verify/parity_browser.json --
 * the two runs that compare this site's decode path with Python ultralytics, under
 * onnxruntime-node and inside real Chrome. They read the artefacts at run time rather
 * than repeating their numbers in the HTML, so the page cannot drift away from the
 * run that produced them.
 *
 * Split out of js/app.js when the front page became the detector alone: nothing here
 * touches the model, and the detector page no longer carries this code.
 */

import { CLASS_NAMES } from './detect.js';
import { wireConsole } from './console-embed.js';

const $ = (id) => document.getElementById(id);

/** Renders site/verify/parity_report.json, so the page cannot drift from the run. */
async function loadParity() {
  const host = $('parityBody');
  if (!host) return;
  try {
    const r = await fetch('verify/parity_report.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const p = await r.json();
    const s = p.summary;
    const row = (k, v, n) => `<tr><td>${k}</td><td class="num">${v}</td><td style="color:var(--muted);font-size:0.82rem">${n}</td></tr>`;
    host.innerHTML = [
      row('images compared', s.images, 'two per class, held-out NEU-DET test split'),
      row('detections, Python', s.python_detections, `ultralytics ${p.python.ultralytics}, conf ${p.python.conf}, iou ${p.python.iou}, imgsz ${p.python.imgsz}`),
      row('detections, JS', s.js_detections, p.js.runtime + ', shared site/js/detect.js'),
      row('matched pairs', s.matched_pairs, 'same class, greedy IoU match'),
      row('max box delta', `${s.max_box_delta_px.toFixed(4)} px`, `tolerance ${p.tolerance.box_px} px`),
      row('max confidence delta', s.max_conf_delta.toExponential(2), `tolerance ${p.tolerance.conf}`),
      row('max preprocess pixel delta', `${s.max_preprocess_pixel_delta} / 255`, `mean ${s.mean_preprocess_pixel_delta.toFixed(5)}; our fixed-point resize vs the tensor ultralytics built`),
      row('median session.run', `${s.median_session_run_ms.toFixed(2)} ms`, `onnxruntime-node on CPU, ${s.timed_runs_per_image} timed runs per image, ${s.warmup_runs} warm-ups discarded`),
      row('verdict', `<span class="tag ${s.verdict === 'PASS' ? 'pass' : 'warn'}">${s.verdict}</span>`, s.discrepancies === 0 ? 'no discrepancies' : `${s.discrepancies} discrepancies`),
    ].join('');
    const st = $('parityStamp');
    if (st) st.textContent = `run ${p.generated_at}`;
  } catch (err) {
    host.innerHTML = `<tr><td colspan="3" style="color:var(--muted)">parity_report.json not reachable (${err.message}). Regenerate it with the two commands above.</td></tr>`;
  }
}

/** Renders site/verify/parity_browser.json -- the same page, checked in real Chrome. */
async function loadBrowserParity() {
  const host = $('parityBrowserBody');
  if (!host) return;
  try {
    const r = await fetch('verify/parity_browser.json');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const p = await r.json();
    const s = p.summary;
    const ms = (o) => `${o.min.toFixed(1)} / ${o.median.toFixed(1)} / ${o.max.toFixed(1)} ms`;
    const row = (k, v, n) => `<tr><td>${k}</td><td class="num">${v}</td><td style="color:var(--muted);font-size:0.82rem">${n}</td></tr>`;
    host.innerHTML = [
      row('images compared', s.images, `driven headless in ${p.chrome}`),
      row('detections, Python', s.python_detections, `ultralytics ${p.python.ultralytics}, conf ${p.python.conf}, iou ${p.python.iou}, imgsz ${p.python.imgsz}`),
      row('detections, this page', s.browser_detections, 'onnxruntime-web on WASM, 1 thread'),
      row('matched pairs', s.matched_pairs, 'same class, greedy IoU match'),
      row('max box delta', `${s.max_box_delta_px.toFixed(4)} px`, `tolerance ${p.tolerance.box_px} px`),
      row('max raw confidence delta', s.max_conf_delta.toExponential(2), `tolerance ${p.tolerance.conf}`),
      row('session.run, min / median / max', ms(s.session_run_ms), 'performance.now() inside the page, warm-up excluded'),
      row('preprocess', ms(s.preprocess_ms), 'canvas read, fixed-point resize, NCHW pack'),
      row('decode + NMS', ms(s.decode_nms_ms), 'the part the confidence slider re-runs'),
      row('warm-up run', `${s.warmup_ms.toFixed(0)} ms`, 'first call only, WASM code-gen and arena allocation; never counted above'),
      row('page and console errors', s.page_errors.length + s.console_errors.length, s.page_errors.length + s.console_errors.length === 0 ? 'clean load' : (s.page_errors.concat(s.console_errors).join(' | ')),),
      row('failed or 4xx requests', s.bad_requests.length, s.bad_requests.length === 0 ? 'every asset returned 200' : JSON.stringify(s.bad_requests)),
      row('verdict', `<span class="tag ${s.verdict === 'PASS' ? 'pass' : 'warn'}">${s.verdict}</span>`, s.discrepancies === 0 ? 'no discrepancies' : `${s.discrepancies} discrepancies`),
    ].join('');
    const st = $('parityBrowserStamp');
    if (st) st.textContent = `run ${p.generated_at}`;

    // The hero quotes a latency. Fill it from this artefact rather than from the
    // HTML, so the headline number cannot drift away from the measurement the
    // moment the parity run is repeated on different hardware.
    const hero = $('heroLatency');
    if (hero) {
      const r = s.session_run_ms;
      hero.textContent = `(min ${r.min.toFixed(1)}, median ${r.median.toFixed(1)}, max ${r.max.toFixed(1)} ms)`;
    }
  } catch (err) {
    host.innerHTML = `<tr><td colspan="3" style="color:var(--muted)">parity_browser.json not reachable (${err.message}). Regenerate it with <code>npm run serve</code> then <code>npm run parity:browser</code> in site/verify.</td></tr>`;
  }
}

/** Class list in head-index order, for the footer. */
function fillFacts() {
  const el = $('classList');
  if (el) el.textContent = CLASS_NAMES.join('  ');
}

fillFacts();
wireConsole();
loadParity();
loadBrowserParity();
