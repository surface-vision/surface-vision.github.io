// Drives the page in real Chrome: loads it, runs samples, records the verdicts and any console errors,
// and saves screenshots. Dev-only; not served.
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');
const OUT = process.argv[2] || '/tmp';
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
const p = await b.newPage();
const errors = [];
p.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
p.on('pageerror', (e) => errors.push(String(e)));
await p.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
await p.goto('http://127.0.0.1:8777/', { waitUntil: 'networkidle0' });
const run = async (label) => {
  await p.evaluate((l) => [...document.querySelectorAll('#chips .chip')].find((c) => c.textContent === l).click(), label);
  await p.waitForFunction((l) => { const v = document.getElementById('verdict'); return v && !/Nothing analysed/.test(v.textContent) && document.getElementById('status').textContent.indexOf('loading') < 0; }, { timeout: 60000 }, label);
  await new Promise((r) => setTimeout(r, 400));
  return p.evaluate(() => document.getElementById('verdict').textContent.replace(/\s+/g, ' ').trim());
};
const out = {};
for (const l of ['Scratches 1', 'Inclusion 1', 'Mill strip: defect 1', 'Mill strip: defect 2', 'Mill strip: clean 1', 'Mill strip: clean 2']) out[l] = await run(l);
await run('Patches 1');
await p.screenshot({ path: path.join(OUT, 'site_hero.png') });
await p.screenshot({ path: path.join(OUT, 'site_full.png'), fullPage: true });
await p.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 });
await p.screenshot({ path: path.join(OUT, 'site_mobile.png') });
console.log(JSON.stringify({ verdicts: out, errors }, null, 1));
await b.close();
