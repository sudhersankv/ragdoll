// FALLGUY viewer page. World frame = MuJoCo: metres, +Z up. MuJoCo quats are (w, x, y, z).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const COLORS = { ok: new THREE.Color('#3ccf7a'), warn: new THREE.Color('#f2b632'), bad: new THREE.Color('#ff4d4d') };
const TAKE_COLORS = ['#6aa7ff', '#ff8a3d', '#c77dff', '#3ccf7a'];

const S = {
  info: null, view: params.get('view') || 'ref', t: 0, dur: 0, playing: false, speed: 1,
  takes: [], perf: {}, bio: {}, crit: {}, active: null, critTab: null,
  worldMesh: null, collGroup: new THREE.Group(), visGroup: new THREE.Group(),
};

const status = (msg) => { const el = $('status'); el.textContent = msg || ''; el.hidden = !msg; };
const fetchJSON = async (u) => { const r = await fetch(u); const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j; };
const fetchBin = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); };
const mjq = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ------------------------------------------------------------------ three setup
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setScissorTest(true);
const scene = new THREE.Scene();
const BG = new THREE.Color('#0b0d10');
scene.background = BG;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(4, -3, 8); scene.add(sun);
scene.add(S.collGroup, S.visGroup);
const camera = new THREE.PerspectiveCamera(60, 1, 0.03, 300);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;

// ------------------------------------------------------------------ geometry builders
function geomGeometry(g, meshes, blob) {
  const s = g.size;
  let geo;
  switch (g.type) {
    case 'box': geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case 'sphere': geo = new THREE.SphereGeometry(s[0], 24, 16); break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0], s[1], s[2]); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16); geo.rotateX(Math.PI / 2); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24); geo.rotateX(Math.PI / 2); break;
    case 'plane': {
      const w = s[0] > 0 ? 2 * s[0] : 80, h = s[1] > 0 ? 2 * s[1] : 80;
      geo = new THREE.PlaneGeometry(w, h); break;
    }
    case 'mesh': {
      const m = meshes[g.mesh]; if (!m || !blob) return null;
      geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(blob, m.vofs, m.vnum * 3), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, m.fofs, m.fnum * 3), 1));
      geo.computeVertexNormals();
      break;
    }
    default: return null;
  }
  return geo;
}

// ------------------------------------------------------------------ world
async function loadWorld() {
  const w = await fetchJSON('/api/world');
  const blob = Object.keys(w.meshes || {}).length ? await fetchBin('/api/world.bin') : null;
  for (const g of w.geoms) {
    if (g.type === 'plane') {
      const grid = new THREE.GridHelper(40, 40, 0x445060, 0x262c34);
      grid.rotation.x = Math.PI / 2; grid.position.set(g.pos[0], g.pos[1], g.pos[2] + 0.002);
      grid.userData.floor = true;
      S.visGroup.add(grid);
      if (!S.info.glb) {
        const pl = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x3a3d42 }));
        pl.position.fromArray(g.pos); S.visGroup.add(pl);
      }
      continue;
    }
    const geo = geomGeometry(g, w.meshes, blob); if (!geo) continue;
    const pos = new THREE.Vector3().fromArray(g.pos), quat = mjq(g.quat);
    if (g.collide) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x38b6ff, transparent: true, opacity: 0.22, depthWrite: false });
      const mesh = new THREE.Mesh(geo, mat); mesh.position.copy(pos); mesh.quaternion.copy(quat);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: 0x7fd3ff }));
      mesh.add(edges); mesh.name = g.name;
      S.collGroup.add(mesh);
    } else {
      const c = g.rgba;
      const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), transparent: c[3] < 1, opacity: c[3] });
      const mesh = new THREE.Mesh(geo, mat); mesh.position.copy(pos); mesh.quaternion.copy(quat);
      S.visGroup.add(mesh);
    }
  }
  if (S.info.glb) {
    status('loading reconstructed mesh...');
    const gltf = await new GLTFLoader().loadAsync(S.info.glb);
    const root = gltf.scene;
    orientGLB(root, w.visual_bbox);
    root.traverse((o) => {
      if (o.isMesh) {
        o.userData.litMat = o.material;
        const m = o.material;
        // untextured (game-style replica) materials stay lit: "unlit texture" only applies to photo textures
        o.userData.unlitMat = !m.map ? m : new THREE.MeshBasicMaterial({ map: m.map, color: 0xffffff,
          vertexColors: !!o.geometry.attributes.color, side: THREE.DoubleSide });
        if (m) m.side = THREE.DoubleSide;
      }
    });
    S.worldMesh = root; scene.add(root);
    applyUnlit();
  }
}

function orientGLB(root, expect) {
  // glTF is Y-up by convention; the contract says env.glb shares scene.xml coordinates (Z-up). Pick whichever
  // orientation matches the scene.xml visual mesh bbox (if any); ?glbyup=1 / 0 forces it.
  const force = params.get('glbyup');
  const box = new THREE.Box3().setFromObject(root);
  let yup = force != null ? force === '1' : true;  // world/reconstruct.py writes env.glb Y-up
  if (force == null && expect) {
    const e0 = new THREE.Vector3().fromArray(expect[0]), e1 = new THREE.Vector3().fromArray(expect[1]);
    const asIs = box.min.distanceTo(e0) + box.max.distanceTo(e1);
    root.rotation.x = Math.PI / 2; root.updateMatrixWorld(true);
    const b2 = new THREE.Box3().setFromObject(root);
    const rot = b2.min.distanceTo(e0) + b2.max.distanceTo(e1);
    root.rotation.x = 0; root.updateMatrixWorld(true);
    yup = rot < asIs;
  }
  if (yup) root.rotation.x = Math.PI / 2;
  S.glbYup = yup;
}

function applyUnlit() {
  if (!S.worldMesh) return;
  const unlit = $('tUnlit').checked;
  S.worldMesh.traverse((o) => { if (o.isMesh) o.material = unlit ? o.userData.unlitMat : o.userData.litMat; });
}

// ------------------------------------------------------------------ performer (H1) per take
async function loadTake(name, idx) {
  const [p, blob, traj] = await Promise.all([
    fetchJSON(`/api/performer?take=${name}`), fetchBin(`/api/performer.bin?take=${name}`), fetchJSON(`/api/traj?take=${name}`)]);
  const group = new THREE.Group(); group.name = `performer_${name}`;
  const bodies = {};
  for (const b of p.bodies) { const o = new THREE.Group(); o.name = b.name; bodies[b.name] = o; group.add(o); }
  const geoCache = {};
  const mats = [];
  for (const g of p.geoms) {
    const key = g.mesh || null;
    const geo = (key && geoCache[key]) || geomGeometry(g, p.meshes, blob);
    if (!geo) continue;
    if (key) geoCache[key] = geo;
    const base = new THREE.Color(g.rgba[0], g.rgba[1], g.rgba[2]);
    const mat = new THREE.MeshStandardMaterial({ color: base.clone(), roughness: 0.55, metalness: 0.1 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.fromArray(g.pos); mesh.quaternion.copy(mjq(g.quat));
    (bodies[g.body] || group).add(mesh);
    mats.push({ body: g.body, mat, base });
  }
  scene.add(group);
  const nb = traj.bodies.length;
  const perf = { name, idx, group, bodies, mats, traj, nb, color: TAKE_COLORS[idx % TAKE_COLORS.length] };
  S.perf[name] = perf;
  S.dur = Math.max(S.dur, traj.t[traj.t.length - 1] || 0);
  return perf;
}

function frameOf(series_t, t) {
  // index of the last sample <= t
  let lo = 0, hi = series_t.length - 1;
  if (hi < 0) return 0;
  if (t <= series_t[0]) return 0;
  if (t >= series_t[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (series_t[m] <= t) lo = m; else hi = m; }
  return lo;
}

function poseTake(perf, t) {
  const tr = perf.traj;
  const k = frameOf(tr.t, t);
  const k2 = Math.min(k + 1, tr.t.length - 1);
  const a = k2 > k ? Math.min(1, Math.max(0, (t - tr.t[k]) / (tr.t[k2] - tr.t[k]))) : 0;
  const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
  tr.bodies.forEach((name, i) => {
    const o = perf.bodies[name]; if (!o) return;
    const p0 = (k * perf.nb + i) * 3, p1 = (k2 * perf.nb + i) * 3;
    o.position.set(tr.pos[p0] + a * (tr.pos[p1] - tr.pos[p0]), tr.pos[p0 + 1] + a * (tr.pos[p1 + 1] - tr.pos[p0 + 1]),
      tr.pos[p0 + 2] + a * (tr.pos[p1 + 2] - tr.pos[p0 + 2]));
    const q0 = (k * perf.nb + i) * 4, q1 = (k2 * perf.nb + i) * 4;
    qa.set(tr.quat[q0 + 1], tr.quat[q0 + 2], tr.quat[q0 + 3], tr.quat[q0]);
    qb.set(tr.quat[q1 + 1], tr.quat[q1 + 2], tr.quat[q1 + 3], tr.quat[q1]);
    o.quaternion.slerpQuaternions(qa, qb, a);
  });
  colourTake(perf, t);
}

// ------------------------------------------------------------------ stress colours
const PART_MAP = [
  ['knee', ['knee']], ['hip', ['hip']], ['ankle', ['ankle']], ['foot', ['ankle']], ['shoulder', ['shoulder']],
  ['elbow', ['elbow']], ['arm', ['shoulder', 'elbow']], ['head', ['torso_link']], ['neck', ['torso_link']],
  ['lumbar', ['torso_link', 'pelvis']], ['spine', ['torso_link', 'pelvis']], ['torso', ['torso_link']],
  ['chest', ['torso_link']], ['pelvis', ['pelvis']],
];
function keyMatchesBody(key, body) {
  const k = key.toLowerCase();
  if (k === body || k + '_link' === body) return true;
  const side = /(^|_)(left|l)(_|$)/.test(k) ? 'left' : /(^|_)(right|r)(_|$)/.test(k) ? 'right' : null;
  if (side && !body.startsWith(side)) return false;
  for (const [part, targets] of PART_MAP) {
    if (k.includes(part)) {
      if (!side && (part === 'hip') && body !== 'pelvis' && !k.includes('hip')) continue;
      if (targets.some((t) => body.includes(t))) return true;
    }
  }
  return false;
}
function bodyStressMap(perf) {
  if (perf.stressMap) return perf.stressMap;
  const bio = S.bio[perf.name]; const map = {};
  if (bio) {
    for (const body of Object.keys(perf.bodies)) {
      const keys = Object.keys(bio.stress || {}).filter((k) => keyMatchesBody(k, body));
      if (keys.length) map[body] = keys;
    }
  }
  perf.stressMap = map; return map;
}
function stressColour(x) { return x >= 1 ? COLORS.bad : x >= 0.5 ? COLORS.warn : COLORS.ok; }
function colourTake(perf, t) {
  const on = $('tStress').checked; const bio = S.bio[perf.name];
  const map = on && bio ? bodyStressMap(perf) : {};
  const st = bio?.series?.t;
  const kser = st ? frameOf(st, t) : 0;
  for (const m of perf.mats) {
    const keys = map[m.body];
    if (!keys) { m.mat.color.copy(m.base); m.mat.emissive.setRGB(0, 0, 0); continue; }
    let v = 0;
    for (const k of keys) {
      const ser = bio.stress_series?.[k];
      v = Math.max(v, ser ? ser[Math.min(kser, ser.length - 1)] : bio.stress[k]);
    }
    const c = stressColour(v);
    m.mat.color.copy(m.base).lerp(c, 0.85);
    m.mat.emissive.copy(c).multiplyScalar(v >= 1 ? 0.35 : 0.08);
  }
}

// ------------------------------------------------------------------ viewports + camera
function shownTakes() {
  const withTraj = S.takes.filter((t) => S.perf[t.name]);
  if ($('tSplit').checked && withTraj.length >= 2) return withTraj.slice(-2).map((t) => t.name);
  return S.active ? [S.active] : [];
}
function fitRect(x, y, w, h, aspect) {
  let vw = w, vh = w / aspect;
  if (vh > h) { vh = h; vw = h * aspect; }
  return { x: x + (w - vw) / 2, y: y + (h - vh) / 2, w: vw, h: vh };
}
function layout() {
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const shown = shownTakes();
  const n = Math.max(1, shown.length);
  const rects = [];
  const ref = S.view === 'ref' && S.info?.camera;
  for (let i = 0; i < n; i++) {
    const x = (W / n) * i, w = W / n;
    rects.push(ref ? fitRect(x + 4, 4, w - 8, H - 8, S.info.camera.aspect || 9 / 16) : { x, y: 0, w, h: H });
  }
  return { rects, shown, W, H };
}
function setRefCamera() {
  const c = S.info.camera;
  camera.position.fromArray(c.pos); camera.quaternion.copy(mjq(c.quat));
  camera.fov = c.fovy; camera.updateProjectionMatrix();
}
function setView(v) {
  S.view = v;
  if (v === 'orbit') {
    controls.enabled = true;
    if (!S.orbitInit) {
      S.orbitInit = true;
      const c = S.info.camera;
      const tgt = focusPoint();
      if (c) {
        camera.position.fromArray(c.pos);
        const back = camera.position.clone().sub(tgt).setZ(0).normalize().multiplyScalar(2.5);
        camera.position.add(back).add(new THREE.Vector3(0, 0, 2.0));
      } else camera.position.copy(tgt).add(new THREE.Vector3(-6, -6, 4));
      camera.fov = 50; camera.updateProjectionMatrix();
      controls.target.copy(tgt);
    }
    controls.update();
  } else {
    controls.enabled = false;
  }
  document.querySelectorAll('input[name=view]').forEach((r) => { r.checked = r.value === v; });
  $('hud').textContent = v === 'orbit' ? 'drag: orbit, right-drag: pan, wheel: zoom' : 'reference camera (matches the video)';
}
function focusPoint() {
  const p = S.active && S.perf[S.active]?.bodies?.pelvis;
  if (p) return p.getWorldPosition(new THREE.Vector3());
  const box = new THREE.Box3();
  if (S.worldMesh) box.setFromObject(S.worldMesh); else box.setFromObject(S.collGroup);
  return box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
}

function placeOverlays(rects, shown) {
  const op = parseFloat($('bdOp').value);
  for (let i = 0; i < 2; i++) {
    const img = $('bd' + i), lbl = $('lbl' + i), r = rects[i];
    const plate = $('tPlate').checked && S.view === 'ref';
    const showImg = r && S.view === 'ref' && S.info?.ref_image && (op > 0 || plate);
    img.style.opacity = showImg ? (plate ? 1 : op) : 0;
    img.style.zIndex = plate ? 0 : 2; canvas.style.zIndex = 1;
    if (r) Object.assign(img.style, { left: r.x + 'px', top: r.y + 'px', width: r.w + 'px', height: r.h + 'px' });
    const take = shown[i];
    lbl.hidden = !(r && take && shown.length > 1);
    if (!lbl.hidden) {
      lbl.textContent = take; lbl.style.left = (r.x + 8) + 'px'; lbl.style.top = (r.y + 8) + 'px';
      lbl.style.color = S.perf[take].color;
    }
  }
}

let lastFrame = performance.now();
let lastW = 0, lastH = 0;
function tick(now) {
  try { frame(now); } catch (e) { console.error(e); status('render error: ' + e.message); }
  requestAnimationFrame(tick);
}
function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
  if (S.playing && S.dur > 0) { S.t += dt * S.speed; if (S.t > S.dur) S.t = 0; syncTime(false); }
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (W !== lastW || H !== lastH) { renderer.setSize(W, H, false); lastW = W; lastH = H; }
  for (const p of Object.values(S.perf)) poseTake(p, S.t);
  if (S.view === 'orbit') {
    if ($('tFollow').checked && S.active && S.perf[S.active]) {
      const tgt = S.perf[S.active].bodies.pelvis?.getWorldPosition(new THREE.Vector3());
      if (tgt) { const d = tgt.clone().sub(controls.target); controls.target.add(d); camera.position.add(d); }
    }
    controls.update();
  } else if (S.info?.camera) setRefCamera();
  const { rects, shown } = layout();
  const plate = $('tPlate').checked && S.view === 'ref' && !!S.info?.ref_image;
  scene.background = plate ? null : BG;
  if (S.worldMesh) S.worldMesh.visible = !plate && $('tMesh').checked;
  S.collGroup.visible = !plate && $('tColl').checked; S.visGroup.visible = !plate;
  renderer.setScissor(0, 0, W, H); renderer.setViewport(0, 0, W, H); renderer.setClearColor(0x050607, plate ? 0 : 1); renderer.clear();
  rects.forEach((r, i) => {
    for (const p of Object.values(S.perf)) p.group.visible = shown.length ? p.name === shown[i] : false;
    renderer.setViewport(r.x, H - r.y - r.h, r.w, r.h);
    renderer.setScissor(r.x, H - r.y - r.h, r.w, r.h);
    camera.aspect = r.w / r.h; camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  });
  placeOverlays(rects, shown);
  drawCharts();
}

// ------------------------------------------------------------------ timeline + charts
function syncTime(fromScrub) {
  if (!fromScrub) $('scrub').value = S.dur ? S.t / S.dur : 0;
  $('time').textContent = `${S.t.toFixed(2)} / ${S.dur.toFixed(2)} s`;
  const vid = $('video');
  if (!$('videoBox').hidden && vid.readyState >= 1 && Math.abs(vid.currentTime - S.t) > 0.25 && !S.playing) vid.currentTime = S.t;
}

function chart(cv, key, label, unit, limits) {
  const ctx = cv.getContext('2d');
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const pad = { l: 34, r: 8, t: 16, b: 14 };
  let ymax = 0; const series = [];
  for (const t of S.takes) {
    const b = S.bio[t.name]; const y = b?.series?.[key]; const x = b?.series?.t;
    if (y && x) { series.push({ take: t.name, x, y, color: S.perf[t.name]?.color || '#aaa' }); ymax = Math.max(ymax, ...y); }
  }
  for (const l of limits) if (l.v < ymax * 3 || !series.length) ymax = Math.max(ymax, l.v * 1.1);
  ymax = ymax || 1;
  const X = (t) => pad.l + (t / (S.dur || 1)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v / ymax) * (h - pad.t - pad.b);
  ctx.font = '10px -apple-system, sans-serif'; ctx.fillStyle = '#8b929c';
  ctx.fillText(`${label} (${unit})`, pad.l, 11);
  ctx.strokeStyle = '#2a2f37'; ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) { const v = (ymax * i) / 2; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(v >= 10 ? 0 : 1), 2, Y(v) + 3); }
  const rel = S.bio[S.active]?.t_release;
  if (typeof rel === 'number' && rel > 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.04)'; ctx.fillRect(pad.l, pad.t, X(rel) - pad.l, h - pad.t - pad.b);
    ctx.fillStyle = '#5a616b'; ctx.fillText('scripted: inverse dynamics', pad.l + 3, h - pad.b - 3);
    ctx.fillText('free: measured', X(rel) + 3, h - pad.b - 3);
  }
  for (const l of limits) {
    if (l.v > ymax) { ctx.fillStyle = l.c; ctx.fillText(`${l.name}: off scale`, w - pad.r - ctx.measureText(`${l.name}: off scale`).width, 11); continue; }
    ctx.setLineDash([4, 3]); ctx.strokeStyle = l.c; ctx.beginPath(); ctx.moveTo(pad.l, Y(l.v)); ctx.lineTo(w - pad.r, Y(l.v)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = l.c; ctx.fillText(l.name, w - pad.r - ctx.measureText(l.name).width, Y(l.v) - 3);
  }
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.take === S.active || series.length === 1 ? 1.8 : 1.2;
    ctx.globalAlpha = s.take === S.active || shownTakes().includes(s.take) ? 1 : 0.45;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(s.x.length / (w * 2)));
    for (let i = 0; i < s.x.length; i += step) { const px = X(s.x[i]), py = Y(s.y[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (!series.length) { ctx.fillStyle = '#5a616b'; ctx.fillText('no data', w / 2 - 20, h / 2); }
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(S.t), pad.t); ctx.lineTo(X(S.t), h - pad.b); ctx.stroke();
  cv.onclick = (e) => { const r = cv.getBoundingClientRect(); S.t = Math.max(0, Math.min(S.dur, ((e.clientX - r.left - pad.l) / (w - pad.l - pad.r)) * S.dur)); syncTime(false); };
}
function limitOf(keys, dflt) {
  for (const t of S.takes) {
    const l = S.bio[t.name]?.limits || {};
    for (const k of keys) { const v = typeof l[k] === 'object' && l[k] ? l[k].value : l[k]; if (typeof v === 'number') return v; }
  }
  return dflt;
}
function drawCharts() {
  if ($('stuntPanel').hidden) return;
  chart($('chartG'), 'head_g', 'head acceleration', 'G', [
    { name: 'blueprint target 8 G', v: limitOf(['head_g_target', 'head_target_g', 'target_g'], 8), c: '#f2b632' },
    { name: 'severe 80 g', v: limitOf(['head_g_severe', 'head_g', 'head_severe_g'], 80), c: '#ff4d4d' }]);
  chart($('chartK'), 'knee_torque', 'knee torque (max L/R)', 'N·m', [
    { name: 'limit ~200 N·m', v: limitOf(['knee_moment', 'knee_Nm', 'knee', 'knee_torque_Nm'], 200), c: '#ff4d4d' }]);
}

// ------------------------------------------------------------------ side panels
function fmt(v) {
  if (typeof v === 'number') return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
  if (v && typeof v === 'object') { const n = v.value ?? v.peak ?? v.max ?? v.index; if (typeof n === 'number') return fmt(n) + (v.source ? ` <span class="dim" title="${esc(v.source)}">${String(v.source).startsWith('inv') ? 'ID' : 'meas'}</span>` : ''); return esc(JSON.stringify(v)).slice(0, 60); }
  return esc(v);
}
function renderPeaks() {
  const names = S.takes.map((t) => t.name).filter((n) => S.bio[n]?.available);
  if (!names.length) { $('peaks').innerHTML = '<tr><td>no biomech.json yet</td></tr>'; return; }
  const rows = new Set();
  names.forEach((n) => Object.keys(S.bio[n].peaks || {}).forEach((k) => rows.add(k)));
  let html = `<tr><th></th>${names.map((n) => `<th style="color:${S.perf[n]?.color || ''}">${n}</th>`).join('')}</tr>`;
  for (const r of rows) html += `<tr><td>${esc(r)}</td>${names.map((n) => `<td>${fmt(S.bio[n].peaks[r])}</td>`).join('')}</tr>`;
  const sk = new Set(); names.forEach((n) => Object.keys(S.bio[n].stress || {}).forEach((k) => sk.add(k)));
  if (sk.size) html += `<tr><th>stress index (load / limit)</th>${names.map(() => '<th></th>').join('')}</tr>`;
  for (const k of sk) html += `<tr><td>${esc(k)}</td>${names.map((n) => { const v = S.bio[n].stress[k]; if (v == null) return '<td>-</td>';
    const c = v >= 1 ? 'var(--bad)' : v >= 0.5 ? 'var(--warn)' : 'var(--ok)'; return `<td style="color:${c}">${v.toFixed(2)}</td>`; }).join('')}</tr>`;
  for (const n of names) {
    const f = S.bio[n].flags;
    if (f && f.length) html += `<tr><td colspan="${names.length + 1}" style="text-align:left;color:var(--warn)">${n}: ${f.map(esc).join('<br>')}</td></tr>`;
  }
  $('peaks').innerHTML = html;
}
function renderList(v) {
  if (!Array.isArray(v)) return esc(typeof v === 'object' ? JSON.stringify(v) : v);
  return '<ul>' + v.map((x) => `<li>${typeof x === 'object' && x ? esc(Object.entries(x).map(([k, y]) => `${k}: ${typeof y === 'object' ? JSON.stringify(y) : y}`).join(', ')) : esc(x)}</li>`).join('') + '</ul>';
}
function renderCritique() {
  const names = S.takes.map((t) => t.name);
  $('critTabs').innerHTML = names.map((n) => `<button data-t="${n}" class="${n === S.critTab ? 'on' : ''}">${n}</button>`).join('')
    + (S.compare ? `<button data-t="__compare" class="${S.critTab === '__compare' ? 'on' : ''}">compare</button>` : '');
  $('critTabs').querySelectorAll('button').forEach((b) => { b.onclick = () => { S.critTab = b.dataset.t; renderCritique(); }; });
  if (S.critTab === '__compare') { $('crit').innerHTML = `<pre>${esc(JSON.stringify(S.compare, null, 1))}</pre>`; return; }
  const c = S.crit[S.critTab]?.critique;
  if (!c) { $('crit').innerHTML = '<span class="dim">no critique.json for this take yet</span>'; return; }
  if (typeof c === 'string') { $('crit').innerHTML = `<p>${esc(c)}</p>`; return; }
  let html = '';
  const score = c.score ?? c.fidelity ?? c.rating;
  if (score != null) html += `<div class="score">${esc(score)}<span class="dim" style="font-size:12px"> / 100</span></div>`;
  const used = new Set(['score', 'fidelity', 'rating']);
  for (const k of ['verdict', 'summary', 'critique', 'text', 'assessment', 'notes', 'comment']) {
    if (typeof c[k] === 'string') { html += `<p>${k === 'verdict' ? '<b>' : ''}${esc(c[k])}${k === 'verdict' ? '</b>' : ''}</p>`; used.add(k); }
  }
  for (const [k, v] of Object.entries(c)) {
    if (used.has(k)) continue;
    if (Array.isArray(v) && v.length) html += `<div class="dim">${esc(k)}</div>${renderList(v)}`;
    else if (typeof v === 'string' && v.length < 400) html += `<p><span class="dim">${esc(k)}:</span> ${esc(v)}</p>`;
    else if (typeof v === 'number' || typeof v === 'boolean') html += `<p><span class="dim">${esc(k)}:</span> ${esc(v)}</p>`;
    else if (v && typeof v === 'object' && Object.keys(v).length) html += `<div class="dim">${esc(k)}</div><pre>${esc(JSON.stringify(v, null, 1)).slice(0, 1500)}</pre>`;
  }
  $('crit').innerHTML = html || `<pre>${esc(JSON.stringify(c, null, 1))}</pre>`;
}
function renderWorldInfo() {
  const i = S.info; const parts = [];
  if (i.world_name) parts.push(`<b>${esc(i.world_name)}</b>`);
  const v = (i.verify || []).filter(Boolean); const last = v[v.length - 1];
  if (last) parts.push(`verify round ${esc(last.round ?? v.length)}: fidelity <b>${esc(last.score ?? '?')}</b>`);
  if (i.objects?.length) parts.push('objects: ' + i.objects.map((o) => `${esc(o.label || o.id)}${o.hazard ? ' <span style="color:var(--warn)">(hazard)</span>' : ''}`).join(', '));
  if (i.camera) parts.push(`ref camera from ${esc(i.camera.source)}, fovy ${i.camera.fovy.toFixed(1)}°, aspect ${i.camera.aspect.toFixed(3)}`);
  if (i.glb) parts.push(`mesh ${esc(i.glb.split('/').slice(-1)[0])}${S.glbYup ? ' (Y-up, rotated)' : ''}`); else parts.push('no reconstructed mesh (env.glb) found');
  if (i.edits?.length) parts.push('edits: ' + i.edits.map((e) => esc(e.prompt || JSON.stringify(e))).join('; '));
  $('worldInfo').innerHTML = parts.join('<br>');
}
function renderTakeRow() {
  const names = S.takes.filter((t) => S.perf[t.name]).map((t) => t.name);
  $('takeRow').innerHTML = names.length ? 'take: ' + names.map((n) => `<button data-t="${n}" class="${n === S.active ? 'on' : ''}" style="border-color:${S.perf[n].color}">${n}</button>`).join('') : '';
  $('takeRow').querySelectorAll('button').forEach((b) => { b.onclick = () => { S.active = b.dataset.t; S.critTab = b.dataset.t; renderTakeRow(); renderCritique(); setVideo(); }; });
  $('legend').innerHTML = names.map((n) => `<span><span class="sw" style="background:${S.perf[n].color}"></span>${n}</span>`).join('');
}
function setVideo() {
  const t = S.takes.find((x) => x.name === S.active);
  $('videoBox').hidden = !(t && t.video);
  if (t && t.video && $('video').getAttribute('src') !== t.video) $('video').src = t.video;
}

// ------------------------------------------------------------------ main
async function main() {
  try {
    S.info = await fetchJSON('/api/info');
    const st = S.info.stunt || {};
    $('brief').textContent = st.brief ? `"${st.brief}"${st.duration || st.duration_sec ? ` · ${st.duration || st.duration_sec} s` : ''}` : 'world preview';
    $('worldname').textContent = S.info.world_name || '';
    if (S.info.ref_image) { $('bd0').src = S.info.ref_image; $('bd1').src = S.info.ref_image; }
    if (!S.info.camera) setView('orbit');
    await loadWorld();
    S.takes = S.info.takes || [];
    S.compare = S.info.compare;
    const loadable = S.takes.filter((t) => t.has_traj && t.has_scene);
    status(loadable.length ? `loading ${loadable.length} take(s)...` : '');
    for (const [i, t] of S.takes.entries()) {
      if (t.has_traj && t.has_scene) { try { await loadTake(t.name, i); } catch (e) { console.error(e); status(`take ${t.name}: ${e.message}`); } }
      try { S.bio[t.name] = await fetchJSON(`/api/biomech?take=${t.name}`); } catch (e) { console.warn(e); }
      try { S.crit[t.name] = await fetchJSON(`/api/critique?take=${t.name}`); } catch (e) { console.warn(e); }
    }
    const withTraj = S.takes.filter((t) => S.perf[t.name]);
    S.active = withTraj.length ? withTraj[withTraj.length - 1].name : null;
    S.critTab = S.active || S.takes[0]?.name;
    $('stuntPanel').hidden = !S.takes.length;
    if (withTraj.length < 2) $('tSplit').parentElement.hidden = true;
    renderTakeRow(); renderPeaks(); renderCritique(); renderWorldInfo(); setVideo();
    setView(S.info.camera ? S.view : 'orbit');
    if (params.get('t')) S.t = parseFloat(params.get('t'));
    if (params.get('bd')) { $('bdOp').value = params.get('bd'); $('bdOp').dispatchEvent(new Event('input')); }
    if (params.get('split') === '0') $('tSplit').checked = false;
    if (params.get('plate') === '1') $('tPlate').checked = true;
    syncTime(false);
    if (!$('status').textContent.startsWith('take')) status('');
    if (params.get('play') === '1') togglePlay();
    window.__fallguyReady = true;
  } catch (e) {
    console.error(e); status('error: ' + e.message);
  }
}

function togglePlay() {
  S.playing = !S.playing; $('play').textContent = S.playing ? 'Pause' : 'Play';
  const vid = $('video');
  if (!$('videoBox').hidden) { if (S.playing) { vid.currentTime = S.t; vid.playbackRate = S.speed; vid.play().catch(() => {}); } else vid.pause(); }
}
$('play').onclick = togglePlay;
$('scrub').oninput = (e) => { S.t = parseFloat(e.target.value) * S.dur; syncTime(true); };
$('speed').onchange = (e) => { S.speed = parseFloat(e.target.value); $('video').playbackRate = S.speed; };
document.querySelectorAll('input[name=view]').forEach((r) => { r.onchange = () => setView(r.value); });
$('tColl').onchange = (e) => { S.collGroup.visible = e.target.checked; };
$('tMesh').onchange = (e) => { if (S.worldMesh) S.worldMesh.visible = e.target.checked; };
$('tUnlit').onchange = applyUnlit;
$('bdOp').oninput = (e) => { $('bdOpV').textContent = Math.round(e.target.value * 100) + '%'; };
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range' && e.target.type !== 'checkbox') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') { S.t = Math.min(S.dur, S.t + 1 / 60); syncTime(false); }
  if (e.key === 'ArrowLeft') { S.t = Math.max(0, S.t - 1 / 60); syncTime(false); }
});
window.fallguy = { S, setView, THREE };
requestAnimationFrame(tick);
main();
