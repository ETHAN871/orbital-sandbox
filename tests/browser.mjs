import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = resolve(process.env.SERVE_DIR || '.');
const runtimeOnly = !!process.env.BASE_URL || !!process.env.SERVE_DIR;
await mkdir('artifacts', { recursive: true });
const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    if (url.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = resolve(root, '.' + decodeURIComponent(url.pathname) + (url.pathname.endsWith('/') ? 'index.html' : ''));
    if (!file.startsWith(root + '/')) throw new Error('outside root');
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': ({'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.css':'text/css','.wgsl':'text/plain'})[extname(file)] || 'application/octet-stream', 'Cache-Control':'no-store' });
    res.end(data);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = process.env.BASE_URL || `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({ headless:true, ...(process.env.BROWSER_PATH ? {executablePath:process.env.BROWSER_PATH} : {}), args:process.env.CI ? ['--use-angle=swiftshader','--enable-unsafe-swiftshader'] : [] });
const results=[];
async function check(name, fn) { const result=await fn(); results.push({name,status:'passed',detail:result}); console.log(`PASS ${name}`,result?.summary ?? result ?? ''); }
async function app(options={}) {
  const p=await browser.newPage({viewport:{width:1280,height:900},...options});
  p.errors=[];p.on('pageerror',e=>p.errors.push(e.message));p.on('console',m=>{if(m.type()==='error'&&!m.text().includes('favicon'))p.errors.push(m.text())});
  await p.goto(base);await p.evaluate(async()=>{window.appState=(await import('./src/state.js')).state});await p.waitForFunction(()=>window.appState.backendName!==null);
  return p;
}
async function glProbe(p) { return p.evaluate(async()=>{
 const r=await import('./src/renderer-webgl.js?v=20261102-maintenance');r.prepareFrameRenderer();r.drawScene();r.drawField();r.drawUI();
 const gl=document.querySelector('#stage').getContext('webgl2');const pixels=new Uint8Array(16*16*4);gl.readPixels(0,0,16,16,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
 return {error:gl.getError(),nonzero:pixels.some(v=>v>0),lost:gl.isContextLost()};
}); }
try {
 if(!runtimeOnly) for(const file of ['potential.test.html','rapier-acceptance.test.html?engine=rapier&backend=force-cpu','physics-audit.test.html?engine=rapier&backend=force-cpu','gpu-mutation.test.html']) await check(file,async()=>{
  const p=await browser.newPage();const errors=[];p.on('pageerror',e=>errors.push(e.message));await p.goto(base+'tests/'+file);
  await p.waitForFunction(()=>{const e=document.querySelector('.summary');return e&&!e.textContent.includes('Running')},null,{timeout:30000});
  const s=await p.locator('.summary').innerText();const details=await p.locator('#output').innerText();assert.ok(await p.locator('.summary.ok').count(),s+'\n'+details);assert.deepEqual(errors,[]);await p.close();return {summary:s,details};
 });
 const p=await app();
 await check('default Rapier and WebGL render',async()=>{assert.equal(await p.evaluate(async()=>(await import('./src/state.js')).state.backendName),'rapier');const x=await glProbe(p);assert.equal(x.error,0);assert.ok(x.nonzero);return x});
 await check('place, edit, pause and resume',async()=>{
  await p.mouse.move(240,200);await p.mouse.down();await p.mouse.move(200,200);await p.mouse.up();
  await p.waitForFunction(()=>window.appState.entities.length===1);
  await p.locator('#edit-btn').click();await p.locator('#pause-btn').click();await p.waitForTimeout(100);
  const a=await p.evaluate(async()=>{const e=(await import('./src/state.js')).state.entities[0];return [e.x,e.y]});
  await p.waitForTimeout(300);const b=await p.evaluate(async()=>{const e=(await import('./src/state.js')).state.entities[0];return [e.x,e.y]});assert.deepEqual(a,b);
  await p.locator('#pause-btn').click();await p.waitForTimeout(300);assert.ok(await p.evaluate(async()=>{const e=(await import('./src/state.js')).state.entities[0];return e.vx>0}));
  await p.locator('#pause-btn').click();await p.locator('#edit-btn').click();
 });
 await check('cancelled pointer does not leave drag or spawn',async()=>{
  await p.mouse.move(400,300);await p.mouse.down();await p.evaluate(()=>document.querySelector('#stage').dispatchEvent(new PointerEvent('pointercancel',{pointerId:1,bubbles:true})));await p.mouse.up();
  assert.equal(await p.evaluate(async()=>(await import('./src/state.js')).state.drag),null);assert.equal(await p.evaluate(async()=>(await import('./src/state.js')).state.entities.length),1);
 });
 await check('wrap mirror selection and property edits',async()=>{
  await p.evaluate(async()=>{const {state}=await import('./src/state.js');const {createEntity}=await import('./src/entities.js');state.entities.length=0;state.entities.push({...createEntity({type:'planet',x:2,y:150,vx:0,vy:0,mass:100,radius:20,charge:1}),embed:1});state.boundaryMode='wrap';});
  await p.locator('#edit-btn').click();const box=await p.locator('#stage').boundingBox();await p.mouse.click(box.x+box.width-3,box.y+150);
  assert.ok(await p.evaluate(async()=>(await import('./src/state.js')).state.selectedId!==null));
  await p.locator('#mass').evaluate(e=>{e.value='240';e.dispatchEvent(new Event('input',{bubbles:true}))});assert.equal(await p.evaluate(async()=>(await import('./src/state.js')).state.entities[0].mass),240);
 });
 await check('all field styles, wrap, resize and context restoration twice',async()=>{
  for(const style of ['mesh','screen','2d','3d','legacy','curvilinear','rubber-sheet']){
   await p.evaluate(async style=>{const {state}=await import('./src/state.js');state.showField=true;state.fieldStyle=style},style);
   const x=await glProbe(p);assert.equal(x.error,0,style);
  }
  for(let i=0;i<2;i++){
   await p.evaluate(()=>{window.loss=document.querySelector('#stage').getContext('webgl2').getExtension('WEBGL_lose_context');loss.loseContext()});await p.waitForTimeout(100);
   await p.setViewportSize({width:1100+i*100,height:800});await p.evaluate(()=>loss.restoreContext());await p.waitForTimeout(300);
   const x=await glProbe(p);assert.equal(x.error,0);assert.ok(!x.lost&&x.nonzero);assert.ok(await p.locator('#webgl-error').isHidden());
  }
  await p.evaluate(async()=>{const {state}=await import('./src/state.js');state.fieldStyle='mesh';});await p.waitForTimeout(100);await glProbe(p);await p.locator('#panel').evaluate(e=>e.scrollTop=0);await p.screenshot({path:'artifacts/desktop.png'});
 });
 await check('no renderer or application errors',async()=>{assert.deepEqual(p.errors,[])});
 await p.close();
 await check('mobile touch placement and panel layout',async()=>{
  const m=await app({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:2});await m.tap('#stage',{position:{x:120,y:120}});
  assert.equal(await m.evaluate(async()=>(await import('./src/state.js')).state.entities.length),1);
  assert.ok(await m.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));const x=await glProbe(m);assert.equal(x.error,0);assert.deepEqual(m.errors,[]);await m.screenshot({path:'artifacts/mobile.png'});await m.close();
 });
 if(!runtimeOnly) await check('CDN failure falls back to functional local solver',async()=>{
  const c=await browser.newContext();await c.route('https://esm.sh/**',r=>r.abort());const q=await c.newPage();await q.goto(base);await q.evaluate(async()=>{window.appState=(await import('./src/state.js')).state});await q.waitForFunction(()=>window.appState.backendName==='cpu');
  await q.mouse.click(200,200);await q.waitForTimeout(100);assert.equal(await q.evaluate(async()=>(await import('./src/state.js')).state.entities.length),1);assert.equal((await glProbe(q)).error,0);await c.close();
 });
 if(!runtimeOnly) await check('GPU K1 parity with CPU, including partial workgroups',async()=>{
  const q=await browser.newPage();await q.goto(base+'tests/potential.test.html');
  const x=await q.evaluate(async()=>{
   const {detectBackend,loadKernel}=await import('../src/gpu-init.js');const d=await detectBackend();if(d.backend!=='webgpu')return{skipped:d.reason};
   const {createGravityGPU}=await import('../src/physics-gpu-gravity.js');const {computeAccelerations}=await import('../src/physics.js');const{state}=await import('../src/state.js');const g=await createGravityGPU(d.device,await loadKernel('./kernels/gravity_accel.wgsl'));
   let maxError=0;try{for(const n of [2,257,513])for(const mode of ['destroy','wrap']){
    state.viewport={width:800,height:600};state.boundaryMode=mode;
    const es=Array.from({length:n},(_,i)=>({id:i,x:(i*71)%800,y:(i*47)%600,mass:10+i%90,radius:2+i%10,charge:(i%3)-1,absorbing:i%11===0}));
    const expected=es.map(()=>({ax:0,ay:0}));computeAccelerations(es,expected);g.growIfNeeded(n);g.uploadPositions(es,n);g.uploadMetaAll(es,n);g.uploadParams(n,state.G,state.epsilon,mode==='wrap'?800:0,mode==='wrap'?600:0);const enc=d.device.createCommandEncoder();g.recordDispatch(enc,n,0);d.device.queue.submit([enc.finish()]);const actual=await g.readbackStaging(0,n);
    for(let i=0;i<n;i++)for(let j=0;j<2;j++){const want=expected[i][j?'ay':'ax'];const err=Math.abs(actual[2*i+j]-want)/(1+Math.abs(want));maxError=Math.max(maxError,err);}
   }}finally{g.destroy();d.device.destroy()}return{maxError};
  });if(!x.skipped)assert.ok(x.maxError<5e-5,JSON.stringify(x));await q.close();return x;
 });
} catch(error) { results.push({status:'failed',error:error.stack});process.exitCode=1;console.error(error); }
finally {await writeFile(runtimeOnly ? 'artifacts/release-results.json' : 'artifacts/browser-results.json',JSON.stringify(results,null,2));await browser.close();await new Promise(r=>server.close(r));}
