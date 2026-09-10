/**
 * Page-weight metric for the site: rendered height and VISIBLE word count.
 *
 * "Visible" means what a judge actually reads without opening anything: text nodes
 * whose parent element has a client rect (so the inside of a closed <details>, a
 * `hidden` element or a `display:none` block is excluded, exactly as the reader
 * sees it). Nothing here is served to the page; it is a measurement tool.
 *
 *   node page_metrics.mjs --url http://127.0.0.1:8788/ [--shots /tmp/out] [--label new]
 */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync, mkdirSync } from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const puppeteer = createRequire(path.join(HERE, 'package.json'))('puppeteer-core');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : d; };

const URL_BASE = arg('--url', 'http://127.0.0.1:8788/');
const SHOTS = arg('--shots', null);
const LABEL = arg('--label', 'page');
const CHROME = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find(existsSync);
if (!CHROME) throw new Error('no Chrome found');
if (SHOTS) mkdirSync(SHOTS, { recursive: true });

const MEASURE = () => {
  const de = document.documentElement;
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let words = 0;      // everything laid out, disclosures counted as if open
  let visible = 0;    // what the reader actually sees before opening anything
  const seen = new WeakMap();
  // Text inside a CLOSED <details> is in the document but is not read: that is the
  // whole point of progressive disclosure. It is counted separately, not silently.
  const disclosed = (el) => {
    for (let a = el; a; a = a.parentElement) {
      if (a.tagName === 'DETAILS' && !a.open) return false;
      if (a.tagName === 'SUMMARY') return true;
    }
    return true;
  };
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const t = n.nodeValue.trim();
    if (!t) continue;
    const el = n.parentElement;
    if (!el) continue;
    if (el.closest('script,style,noscript')) continue;
    let vis = seen.get(el);
    if (vis === undefined) {
      const cs = getComputedStyle(el);
      vis = el.getClientRects().length > 0 && cs.visibility !== 'hidden' && cs.display !== 'none';
      seen.set(el, vis);
    }
    if (!vis) continue;
    const w = t.split(/\s+/).filter(Boolean).length;
    words += w;
    if (disclosed(el)) visible += w;
  }
  return {
    words,
    visible,
    height: Math.max(document.body.scrollHeight, de.scrollHeight),
    docScrollWidth: de.scrollWidth,
    viewportWidth: de.clientWidth,
    horizontalScroll: de.scrollWidth > de.clientWidth + 1,
  };
};

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });
await page.goto(URL_BASE, { waitUntil: 'networkidle2', timeout: 120000 });
await new Promise((r) => setTimeout(r, 2500));

const desktop = await page.evaluate(MEASURE);
console.log(`${LABEL} @1440x900   height ${desktop.height} px   words on screen ${desktop.visible}   words incl. collapsed detail ${desktop.words}   h-scroll ${desktop.horizontalScroll}`);
if (SHOTS) {
  await page.screenshot({ path: path.join(SHOTS, `${LABEL}_1440_fold.png`) });
  await page.screenshot({ path: path.join(SHOTS, `${LABEL}_1440_full.png`), fullPage: true });
}

await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
await new Promise((r) => setTimeout(r, 1200));
const mobile = await page.evaluate(MEASURE);
console.log(`${LABEL} @390x844    height ${mobile.height} px   words on screen ${mobile.visible}   words incl. collapsed detail ${mobile.words}   h-scroll ${mobile.horizontalScroll}`);
if (SHOTS) {
  await page.screenshot({ path: path.join(SHOTS, `${LABEL}_390_fold.png`) });
  await page.screenshot({ path: path.join(SHOTS, `${LABEL}_390_full.png`), fullPage: true });
}
await browser.close();
