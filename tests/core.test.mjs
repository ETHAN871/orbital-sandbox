import test from 'node:test';
import assert from 'node:assert/strict';
import { state, SIM_DT, effectiveTimeRatio } from '../src/state.js';
import { advanceClock } from '../src/simulation-clock.js';
import { minImageDelta, wrapCoordinate } from '../src/boundary.js';
import { computeAccelerations, applyBoundary, predictTrajectory, updateAbsorptions } from '../src/physics.js';
import { buildSpatialHash, forEachCollisionPair } from '../src/physics-spatial-hash.js';
import { computePotentialAt, computeForceDirAt } from '../src/potential.js';
const close = (a,b,tol=1e-9) => assert.ok(Math.abs(a-b)<tol, `${a} != ${b}`);

test('fixed timestep honors time scales at 60, 120 and 144 Hz without discarding fractions',()=>{
 for(const fps of [60,120,144]) for(const ratio of [0.4,1,2,6]) {
  let rem=0,steps=0;
  for(let i=0;i<fps*10;i++){const t=advanceClock(rem,1/fps,ratio,8);rem=t.remainder;steps+=t.steps;}
  close(steps*SIM_DT+rem,10*ratio,1e-8);
 }
});
test('overload drops whole backlog only; pause clears residual time',()=>{
 const t=advanceClock(0.005,0.1,6,8);assert.equal(t.steps,8);close(t.remainder,0.005);
 assert.deepEqual(advanceClock(0.015,0.02,0,8),{steps:0,remainder:0});
 state.isEditMode=true;state.timeScale=0;assert.equal(effectiveTimeRatio(),0);
 state.timeScale=3;assert.equal(effectiveTimeRatio(),0.2);state.isEditMode=false;
});
test('wrap normalizes multiple widths, exact edge, and retains odd min-image ties',()=>{
 for(const [x,want] of [[3000,0],[-3001,999],[1000,0]]) assert.equal(wrapCoordinate(x,1000),want);
 assert.equal(minImageDelta(3010,1000),10);assert.equal(minImageDelta(-3010,1000),-10);
 assert.equal(minImageDelta(500,1000),-minImageDelta(-500,1000));
 const es=[{x:3000,y:-1801}];applyBoundary(es,{width:1000,height:600},'wrap');assert.deepEqual(es,[{x:0,y:599}]);
});
test('small wrap hash returns each pair once and includes predicted positions beyond boundary',()=>{
 state.boundaryMode='wrap';state.viewport={width:50,height:50};
 const es=[{id:1,x:51,y:10,radius:20},{id:2,x:49,y:10,radius:20}];
 buildSpatialHash(es);const pairs=[];forEachCollisionPair(es,(a,b,dx)=>pairs.push([a.id,b.id,dx]));
 assert.deepEqual(pairs,[[1,2,-2]]);
});
test('randomized wrap broadphase covers all overlapping pairs once',()=>{
 let seed=7;const random=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
 for(const [W,H] of [[345,514],[64,80],[800,600]]){
  state.viewport={width:W,height:H};state.boundaryMode='wrap';
  const es=Array.from({length:80},(_,id)=>({id,x:(random()*1.2-0.1)*W,y:(random()*1.2-0.1)*H,radius:2+random()*25}));
  buildSpatialHash(es);const seen=new Set();forEachCollisionPair(es,(a,b)=>{const k=`${a.id}:${b.id}`;assert.ok(!seen.has(k));seen.add(k)});
  for(let i=0;i<es.length;i++)for(let j=i+1;j<es.length;j++){const a=es[i],b=es[j];if(Math.hypot(minImageDelta(a.x-b.x,W),minImageDelta(a.y-b.y,H))<a.radius+b.radius)assert.ok(seen.has(`${i}:${j}`));}
 }
});
test('symmetric charges conserve pair momentum; source-only charge semantics remain explicit',()=>{
 state.boundaryMode='destroy';state.G=80;state.epsilon=4;
 const a={x:0,y:0,radius:10,mass:40,charge:1},b={x:100,y:30,radius:20,mass:70,charge:1};
 const acc=[{},{}];computeAccelerations([a,b],acc);close(40*acc[0].ax+70*acc[1].ax,0);
 const expected=80*70*100/(100**2+30**2+30**2)**1.5;close(acc[0].ax,expected);
 a.charge=0;computeAccelerations([a,b],acc);assert.equal(acc[1].ax,0);assert.ok(acc[0].ax>0);
});
test('potential reference force equals its numerical gradient (not the finite-radius dynamics)',()=>{
 const es=[{x:70,y:80,mass:40,charge:1},{x:130,y:10,mass:10,charge:-1}];const h=1e-3;
 const f=computeForceDirAt(0,0,es,80,4);
 const dx=-(computePotentialAt(h,0,es,80,4)-computePotentialAt(-h,0,es,80,4))/(2*h);
 const dy=-(computePotentialAt(0,h,es,80,4)-computePotentialAt(0,-h,es,80,4))/(2*h);
 close(f.x*f.mag,dx,1e-8);close(f.y*f.mag,dy,1e-8);
});
test('cross-edge absorption follows the short route; prediction wraps large jumps',()=>{
 state.viewport={width:100,height:100};state.boundaryMode='wrap';state.absorptionDuration=1;
 const es=[{id:1,x:2,y:50},{id:2,x:98,y:50,radius:5,absorbing:{blackHoleId:1,elapsedSim:0,startX:98,startY:50,startRadius:5}}];
 updateAbsorptions(es,0.5);close(es[1].x,0);
 state.predictHorizon=1;const path=predictTrajectory({x:10,y:10,vx:18000,vy:0,radius:2},[]);
 for(let i=0;i<path.length;i++)assert.ok(path.data[i*2]>=0&&path.data[i*2]<100);
});
