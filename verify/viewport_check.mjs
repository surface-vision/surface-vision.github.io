/**
 * Narrow-viewport check. Judges open this on a phone.
 *
 * The rule the page must satisfy: the BODY never scrolls sideways. Wide content
 * (tables, the command block, the canvas) is allowed to scroll inside its own
 * container -- that is deliberate -- so this reports the offenders that push the
 * document itself wider than the viewport, and nothing else.
 *
 * Usage:  node viewport_check.mjs --url http://127.0.0.1:8777/
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');

const arg = (n, d) => { const i = process.argv.indexOf(n); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };
const BASE = arg('--url', 'http://127.0.0.1:8777/');

const CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!CHROME) throw new Error('no Chrome found');

const VIEWPORTS = [
  { name: 'iPhone SE      ', width: 320, height: 568, dsf: 2 },
  { name: 'iPhone 12/13   ', width: 390, height: 844, dsf: 3 },
  { name: 'Pixel 7        ', width: 412, height: 915, dsf: 2.6 },
  { name: 'iPad mini port ', width: 768, height: 1024, dsf: 2 },
];
const PAGES = ['', 'technical.html', 'evidence/', 'console/'];

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
let fails = 0;

for (const p of PAGES) {
  console.log(`\n=== /${p || ''} ===`);
  for (const vp of VIEWPORTS) {
    const page = await browser.newPage();
    await page.setViewport({ width: vp.width, height: vp.height, deviceScaleFactor: vp.dsf, isMobile: true, hasTouch: true });
    await page.goto(BASE + p, { waitUntil: 'networkidle2', timeout: 60000 });

    const r = await page.evaluate(() => {
      const de = document.documentElement;
      const vw = de.clientWidth;
      const over = [];
      for (const el of document.querySelectorAll('body *')) {
        const b = el.getBoundingClientRect();
        if (b.width === 0 && b.height === 0) continue;
        // Only elements that actually stick out of the viewport box, and only when no
        // ancestor is a scroll container -- an element wider than its own scrolling
        // parent is the intended design, not an overflow bug.
        if (b.right > vw + 1 || b.left < -1) {
          let scrollable = false;
          for (let a = el.parentElement; a; a = a.parentElement) {
            const ov = getComputedStyle(a).overflowX;
            if (ov === 'auto' || ov === 'scroll' || ov === 'hidden') { scrollable = true; break; }
          }
          if (!scrollable) {
            over.push(`${el.tagName.toLowerCase()}${el.id ? '#' + el.id : ''}${el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\s+/).join('.') : ''} right=${Math.round(b.right)}`);
          }
        }
      }
      return {
        vw,
        docScrollWidth: de.scrollWidth,
        bodyScrollWidth: document.body.scrollWidth,
        horizontalScroll: de.scrollWidth > vw + 1,
        offenders: [...new Set(over)].slice(0, 6),
      };
    });

    const ok = !r.horizontalScroll && r.offenders.length === 0;
    if (!ok) fails++;
    console.log(`  ${vp.name} ${String(vp.width).padStart(4)}px  doc scrollWidth ${String(r.docScrollWidth).padStart(4)} vs viewport ${String(r.vw).padStart(4)}  ${ok ? 'OK -- no horizontal scroll' : 'OVERFLOW'}`);
    if (r.offenders.length) console.log(`      offenders: ${r.offenders.join(' | ')}`);
    await page.close();
  }
}

await browser.close();
console.log(`\n${fails === 0 ? 'PASS -- no page scrolls sideways at any tested width' : `FAIL -- ${fails} viewport/page combinations overflow`}`);
process.exit(fails === 0 ? 0 : 1);
