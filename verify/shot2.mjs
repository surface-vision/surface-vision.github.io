import puppeteer from 'puppeteer-core';
const CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const b=await puppeteer.launch({executablePath:CHROME,headless:'new',args:['--no-sandbox']});
const p=await b.newPage();
await p.setViewport({width:1440,height:900});
await p.goto('https://surface-vision.github.io',{waitUntil:'networkidle2',timeout:90000});
await new Promise(r=>setTimeout(r,6000));
console.log('height:', await p.evaluate(()=>document.body.scrollHeight));
await p.screenshot({path:'/tmp/new_fold.png'});
// click a sample and capture the result
const clicked = await p.evaluate(()=>{
  const c=document.querySelector('[data-sample],.sample-chip,button[data-file]');
  if(c){c.click();return c.textContent.trim().slice(0,40);} return null;});
console.log('clicked sample:', clicked);
await new Promise(r=>setTimeout(r,5000));
await p.screenshot({path:'/tmp/new_result.png'});
await p.setViewport({width:390,height:844});
await new Promise(r=>setTimeout(r,1500));
await p.screenshot({path:'/tmp/new_mobile.png'});
await b.close();
