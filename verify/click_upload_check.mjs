// Checks that clicking the drop zone (and pressing Enter on it) opens the file picker, and that the
// chosen photo is analysed. Also checks that choosing the same file twice runs twice. Dev-only; not served.
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');
const OUT = process.argv[2] || '/tmp';
const URL = process.argv[3] || 'http://127.0.0.1:8777/';
const IMG = path.join(HERE, '..', 'samples', 'mill_defect_1.jpg');
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: 'new' });
const p = await b.newPage();
const errors = [];
p.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
p.on('pageerror', (e) => errors.push(String(e)));
await p.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
await p.goto(URL, { waitUntil: 'networkidle0' });
await p.waitForFunction(() => /Ready/.test(document.getElementById('status').textContent), { timeout: 60000 });
const cursor = await p.$eval('#drop', (d) => getComputedStyle(d).cursor);
const hint = await p.$eval('#dropHint', (d) => d.textContent.replace(/\s+/g, ' ').trim());
const analyse = async (open) => {
  await p.evaluate(() => { document.getElementById('status').textContent = ''; });
  const [chooser] = await Promise.all([p.waitForFileChooser({ timeout: 5000 }), open()]);
  await chooser.accept([IMG]);
  await p.waitForFunction(() => /^Analysed/.test(document.getElementById('status').textContent), { timeout: 60000 });
  return p.evaluate(() => document.getElementById('verdict').textContent.replace(/\s+/g, ' ').trim());
};
const byClick = await analyse(() => p.click('#drop'));
const byClickOnImage = await analyse(() => p.click('#drop')); // second time: canvas now showing, same file
const byEnter = await analyse(async () => { await p.focus('#drop'); await p.keyboard.press('Enter'); });
const status = await p.$eval('#status', (s) => s.textContent);
await p.screenshot({ path: path.join(OUT, 'site_click_upload.png') });
console.log(JSON.stringify({ cursor, hint, byClick, byClickOnImage, byEnter, status, errors }, null, 1));
await b.close();
