/**
 * Drives the front page the way a judge would: click a sample, look at what the
 * detector returned, click a wide strip, check the tiled path engaged. Screenshots
 * each state at both viewports. Dev-only; nothing here is served to the page.
 *
 *   node drive_demo.mjs --url http://127.0.0.1:8788/ --shots /tmp/ui_drive
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, mkdirSync } from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const URL_BASE = arg('--url', 'http://127.0.0.1:8788/');
const SHOTS = arg('--shots', '/tmp/ui_drive');
const CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
mkdirSync(SHOTS, { recursive: true });

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
page.on('requestfailed', (r) => errors.push('requestfailed: ' + r.url()));
page.on('requestfinished', (r) => { const s = r.response()?.status(); if (s >= 400) errors.push(`${s}: ${r.url()}`); });

await page.evaluateOnNewDocument(() => {
  window.__caught = [];
  document.addEventListener('surface:detections', (e) => window.__caught.push(e.detail));
});

async function run(vp, tag) {
  await page.setViewport(vp);
  await page.goto(URL_BASE, { waitUntil: 'networkidle2', timeout: 120000 });
  await page.waitForFunction(() => document.getElementById('status')
    && /ready --/.test(document.getElementById('status').textContent), { timeout: 180000, polling: 100 });
  console.log(`\n[${tag}] model ready before any click: ${await page.$eval('#status', (e) => e.textContent)}`);
  await page.screenshot({ path: `${SHOTS}/${tag}_1_idle.png` });

  for (const [label, name] of [['square sample', 'patches_271.jpg'], ['wide strip', 'strip_rollmark.jpg']]) {
    const before = await page.evaluate(() => window.__caught.length);
    await page.click(`#chips .chip[data-sample="${name}"]`);
    await page.waitForFunction((n) => window.__caught.length > n, { timeout: 120000, polling: 60 }, before);
    const d = await page.evaluate(() => window.__caught[window.__caught.length - 1]);
    console.log(`[${tag}] ${label}: ${d.image} ${d.width}x${d.height}  tiles=${d.tiles.length}  `
      + `detections=${d.detections.length}  session.run=${d.timing.infer.toFixed(1)} ms`);
    for (const x of d.detections) {
      console.log(`        ${x.className.padEnd(16)} raw ${x.raw.toFixed(4)}  p ${x.calibrated.toFixed(4)}  `
        + `[${x.x1.toFixed(1)}, ${x.y1.toFixed(1)}, ${x.x2.toFixed(1)}, ${x.y2.toFixed(1)}]  ${x.band}`);
    }
    console.log(`[${tag}] verdict line: "${(await page.$eval('#verdict', (e) => e.textContent)).replace(/\s+/g, ' ').trim()}"`);
    const shot = name.startsWith('strip') ? 'strip' : 'sample';
    await page.screenshot({ path: `${SHOTS}/${tag}_2_${shot}_fold.png` });
    if (shot === 'strip') await page.screenshot({ path: `${SHOTS}/${tag}_3_${shot}_full.png`, fullPage: true });
  }
}

await run({ width: 1440, height: 900, deviceScaleFactor: 1 }, 'desktop');
await run({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true }, 'mobile');

console.log('\nerrors: ' + (errors.length ? errors.join('\n  ') : 'none'));
await browser.close();
