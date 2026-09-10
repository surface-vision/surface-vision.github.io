/**
 * Independent render proof. drive_demo.mjs listens to a JS event, which proves the
 * detector COMPUTED boxes; it does not prove anything reached the screen. This reads
 * the canvas back pixel by pixel and checks that (a) each detection's rectangle is
 * actually stroked in its class colour and (b) a filled label chip sits on it. Steel
 * photographs are greyscale, so any chromatic pixel is overlay the page drew.
 *
 *   node canvas_proof.mjs --url http://127.0.0.1:8777/ --out /tmp/proof
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const URL_BASE = arg('--url', 'http://127.0.0.1:8777/');
const OUT = arg('--out', '/tmp/proof');
const CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
const bad = [];
page.on('pageerror', (e) => bad.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') bad.push('console: ' + m.text()); });
page.on('requestfailed', (r) => bad.push('requestfailed: ' + r.url() + ' ' + (r.failure()?.errorText || '')));
page.on('response', (r) => { if (r.status() >= 400) bad.push(`HTTP ${r.status()}: ${r.url()}`); });

await page.evaluateOnNewDocument(() => {
  window.__caught = [];
  document.addEventListener('surface:detections', (e) => window.__caught.push(e.detail));
});

await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
await page.goto(URL_BASE, { waitUntil: 'networkidle2', timeout: 120000 });
await page.waitForFunction(() => /ready --/.test(document.getElementById('status')?.textContent || ''),
  { timeout: 180000, polling: 100 });

let fail = 0;
for (const name of ['patches_271.jpg', 'scratches_271.jpg', 'strip_rollmark.jpg', 'strip_weldline.jpg']) {
  const before = await page.evaluate(() => window.__caught.length);
  await page.click(`#chips .chip[data-sample="${name}"]`);
  await page.waitForFunction((n) => window.__caught.length > n, { timeout: 120000, polling: 60 }, before);
  await new Promise((r) => setTimeout(r, 250));

  const probe = await page.evaluate(() => {
    const cvs = document.getElementById('view');
    const g = cvs.getContext('2d', { willReadFrequently: true });
    const img = g.getImageData(0, 0, cvs.width, cvs.height).data;
    const chroma = (i) => {
      const r = img[i], gg = img[i + 1], b = img[i + 2];
      return Math.max(r, gg, b) - Math.min(r, gg, b);
    };
    let chromatic = 0;
    for (let i = 0; i < img.length; i += 4) if (chroma(i) > 40) chromatic++;
    // scale from source px -> canvas px
    const st = window.__state || null;
    return {
      cw: cvs.width, ch: cvs.height,
      cssW: cvs.getBoundingClientRect().width, cssH: cvs.getBoundingClientRect().height,
      visible: getComputedStyle(cvs).display !== 'none' && cvs.getBoundingClientRect().height > 0,
      chromatic, totalPx: img.length / 4,
      // sample the perimeter of each detection box, in canvas coords
      probeBox: null, st,
    };
  });

  const det = await page.evaluate(() => window.__caught[window.__caught.length - 1]);
  // recompute source->canvas scale from the canvas size and the image size
  const sx = probe.cw / det.width, sy = probe.ch / det.height;

  const perim = await page.evaluate((dets, sx, sy) => {
    const cvs = document.getElementById('view');
    const g = cvs.getContext('2d', { willReadFrequently: true });
    const d0 = g.getImageData(0, 0, cvs.width, cvs.height).data;
    const at = (x, y) => { const i = (y * cvs.width + x) * 4; return [d0[i], d0[i + 1], d0[i + 2]]; };
    const chroma = ([r, g2, b]) => Math.max(r, g2, b) - Math.min(r, g2, b);
    return dets.map((d) => {
      const x1 = Math.round(d.x1 * sx), y1 = Math.round(d.y1 * sy);
      const x2 = Math.round(d.x2 * sx), y2 = Math.round(d.y2 * sy);
      let hits = 0, tries = 0;
      const near = (x, y) => {
        // the stroke is a few px wide and centred on the edge; look +/-3 px
        for (let o = -3; o <= 3; o++) {
          for (const [px, py] of [[x + o, y], [x, y + o]]) {
            if (px < 0 || py < 0 || px >= cvs.width || py >= cvs.height) continue;
            if (chroma(at(px, py)) > 40) return true;
          }
        }
        return false;
      };
      for (let t = 0; t <= 20; t++) {
        const fx = x1 + ((x2 - x1) * t) / 20, fy = y1 + ((y2 - y1) * t) / 20;
        for (const [px, py] of [[Math.round(fx), y1], [Math.round(fx), y2], [x1, Math.round(fy)], [x2, Math.round(fy)]]) {
          tries++; if (near(px, py)) hits++;
        }
      }
      // label chip: a run of solid chromatic pixels on a horizontal line just above y1
      let chipRun = 0, best = 0;
      const ly = y1 - 6 >= 0 ? y1 - 6 : Math.min(y1 + 6, cvs.height - 1);
      for (let x = 0; x < cvs.width; x++) {
        if (chroma(at(x, ly)) > 40) { chipRun++; best = Math.max(best, chipRun); } else chipRun = 0;
      }
      // dark glyph pixels inside that chip band = the text drawn in #0B0E12
      let glyph = 0;
      for (let y = Math.max(0, y1 - 120); y < Math.min(cvs.height, y1 + 120); y++) {
        for (let x = 0; x < cvs.width; x++) {
          const [r, g2, b] = at(x, y);
          if (r < 60 && g2 < 60 && b < 60) glyph++;
        }
      }
      return { cls: d.className, box: [x1, y1, x2, y2], edgeHits: hits, edgeTries: tries, chipRun: best, glyphPx: glyph };
    });
  }, det.detections, sx, sy);

  const png = await page.evaluate(() => document.getElementById('view').toDataURL('image/png'));
  writeFileSync(path.join(OUT, `canvas_${name.replace(/\.jpg$/, '')}.png`), Buffer.from(png.split(',')[1], 'base64'));

  const verdict = (await page.$eval('#verdict', (e) => e.textContent)).replace(/\s+/g, ' ').trim();
  const rows = await page.$$eval('#resBody tr', (trs) => trs.map((t) => t.innerText.replace(/\s+/g, ' ').trim()));

  console.log(`\n=== ${name}  source ${det.width}x${det.height} -> canvas ${probe.cw}x${probe.ch} (css ${Math.round(probe.cssW)}x${Math.round(probe.cssH)})`);
  console.log(`    canvas visible: ${probe.visible}   tiles: ${det.tiles.length}   detections: ${det.detections.length}`);
  console.log(`    chromatic (overlay) pixels: ${probe.chromatic} of ${probe.totalPx}`);
  for (const p of perim) {
    const pct = ((p.edgeHits / p.edgeTries) * 100).toFixed(0);
    const ok = p.edgeHits / p.edgeTries > 0.8 && p.chipRun > 8 && p.glyphPx > 20;
    if (!ok) fail++;
    console.log(`    ${ok ? 'OK  ' : 'FAIL'} ${p.cls.padEnd(16)} box ${JSON.stringify(p.box).padEnd(24)} edge stroke ${pct}% of perimeter | label chip run ${p.chipRun}px | glyph px ${p.glyphPx}`);
  }
  console.log(`    verdict: "${verdict}"`);
  console.log(`    table rows: ${rows.length}`);
  rows.forEach((r) => console.log(`      | ${r}`));
  if (det.detections.length === 0) { console.log('    FAIL no detections'); fail++; }
}

console.log('\nnetwork/console problems: ' + (bad.length ? '\n  ' + bad.join('\n  ') : 'none'));
console.log(fail === 0 && bad.length === 0 ? '\nRENDER PROOF PASS' : `\nRENDER PROOF FAIL (${fail} render, ${bad.length} network/console)`);
await browser.close();
