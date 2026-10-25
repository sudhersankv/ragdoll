// FALLGUY director app. World frame = MuJoCo: metres, +Z up. MuJoCo quats are (w, x, y, z).
// Viewport code adapted from fallguy/viewer/static/app.js (world GLB, collision primitives, reference camera,
// H1 trajectory playback with per-link stress colours).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const COLORS = { ok: new THREE.Color('#3ccf7a'), warn: new THREE.Color('#f2b632'), bad: new THREE.Color('#ff4d4d') };
const TAKE_COLORS = ['#6aa7ff', '#ff8a3d', '#c77dff', '#4fd1c5', '#e8eaed'];
const takeColor = (name) => TAKE_COLORS[(Math.max(1, parseInt(String(name).slice(1), 10) || 1) - 1) % TAKE_COLORS.length];
const takeNum = (n) => parseInt(String(n).replace(/^v/, ''), 10) || 0;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const mjq = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);
const num = (v) => (typeof v === 'number' && isFinite(v) ? v : (v && typeof v === 'object' && typeof v.value === 'number' ? v.value : null));
const fx = (v, d = 1) => (v == null || !isFinite(v) ? '-' : Number(v).toFixed(d));
const store = {
  get(k) { try { return localStorage.getItem('fallguy.' + k); } catch { return null; } },
  set(k, v) { try { v == null ? localStorage.removeItem('fallguy.' + k) : localStorage.setItem('fallguy.' + k, v); } catch { /* private mode */ } },
};

const S = {
  worlds: [], worldId: null, worldDetail: null, scene: null, stunts: [], stuntId: null, stunt: null,
  takes: [], perf: {}, traj: {}, active: null, critTab: null, view: 'ref', t: 0, dur: 0, playing: false, speed: 1,
  jobs: {}, worldRoot: new THREE.Group(), collGroup: new THREE.Group(), visGroup: new THREE.Group(), worldMesh: null,
  sceneToken: 0, stuntToken: 0,
};

// ------------------------------------------------------------------ api
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const r = await fetch(path, init);
  const txt = await r.text();
  let j = null;
  try { j = txt ? JSON.parse(txt) : null; } catch { j = { error: txt.slice(0, 200) }; }
  if (!r.ok) throw new Error((j && (j.detail || j.error)) || `${path}: ${r.status}`);
  return j;
}
const fetchBin = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return r.arrayBuffer(); };
const status = (msg) => { const el = $('status'); el.textContent = msg || ''; el.hidden = !msg; };

// ------------------------------------------------------------------ three setup
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setScissorTest(true);
const scene = new THREE.Scene();
const BG = new THREE.Color('#07080a');
scene.background = BG;
scene.add(new THREE.HemisphereLight(0xffffff, 0x404048, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(4, -3, 8); scene.add(sun);
S.worldRoot.add(S.collGroup, S.visGroup);
scene.add(S.worldRoot);
const camera = new THREE.PerspectiveCamera(60, 1, 0.03, 300);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;

function geomGeometry(g, meshes, blob) {
  const s = g.size || [0.1, 0.1, 0.1];
  let geo;
  switch (g.type) {
    case 'box': geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
    case 'sphere': geo = new THREE.SphereGeometry(s[0], 24, 16); break;
    case 'ellipsoid': geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(s[0], s[1], s[2]); break;
    case 'capsule': geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16); geo.rotateX(Math.PI / 2); break;
    case 'cylinder': geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24); geo.rotateX(Math.PI / 2); break;
    case 'plane': geo = new THREE.PlaneGeometry(s[0] > 0 ? 2 * s[0] : 80, s[1] > 0 ? 2 * s[1] : 80); break;
    case 'mesh': {
      const m = meshes && meshes[g.mesh]; if (!m || !blob) return null;
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

function disposeTree(o) {
  o.traverse((c) => {
    if (c.geometry) c.geometry.dispose();
    if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach((m) => m.dispose && m.dispose());
  });
}
function clearWorld() {
  for (const g of [S.collGroup, S.visGroup]) { disposeTree(g); g.clear(); }
  if (S.worldMesh) { scene.remove(S.worldMesh); disposeTree(S.worldMesh); S.worldMesh = null; }
  S.scene = null; S.orbitInit = false;
}
function clearPerformers() {
  for (const p of Object.values(S.perf)) { scene.remove(p.group); disposeTree(p.group); }
  S.perf = {}; S.traj = {}; S.dur = 0; S.t = 0;
}

// normalise /api/worlds/{id}/scene: viewer info + viewer world geoms, possibly nested
function normScene(sc, id) {
  sc = sc || {};
  const w = sc.geoms && !Array.isArray(sc.geoms) && typeof sc.geoms === 'object' ? sc.geoms
    : sc.world && Array.isArray(sc.world.geoms) ? sc.world : sc;
  let geoms = w.geoms || [];
  if (!geoms.length && Array.isArray(sc.collision)) {
    geoms = sc.collision.map((c, i) => ({ name: c.name || `c${i}`, type: c.type || 'box', size: c.size, pos: c.pos,
      quat: c.quat || [1, 0, 0, 0], collide: true }));
  }
  const meshes = w.meshes || {};
  const binUrl = sc.geoms_bin_url || sc.bin_url || sc.world_bin || w.bin_url || (Object.keys(meshes).length ? `/api/worlds/${id}/scene.bin` : null);
  return { glb: sc.glb || sc.mesh_url || sc.glb_url || null, ref_image: sc.ref_image || sc.ref_frame || sc.ref_frame_url || null,
    camera: sc.camera || sc.ref_camera || null, geoms, meshes, binUrl, visual_bbox: w.visual_bbox || sc.visual_bbox || null,
    objects: sc.objects || [] };
}

async function loadScene(id) {
  const token = ++S.sceneToken;
  clearWorld();
  status('loading world...');
  const sc = normScene(await api(`/api/worlds/${encodeURIComponent(id)}/scene`), id);
  if (token !== S.sceneToken) return;
  S.scene = sc;
  const blob = sc.binUrl && Object.keys(sc.meshes).length ? await fetchBin(sc.binUrl).catch(() => null) : null;
  for (const g of sc.geoms) {
    if (g.type === 'plane') {
      const grid = new THREE.GridHelper(40, 40, 0x445060, 0x22272e);
      grid.rotation.x = Math.PI / 2; grid.position.set(g.pos[0], g.pos[1], g.pos[2] + 0.002);
      S.visGroup.add(grid);
      if (!sc.glb) {
        const pl = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), new THREE.MeshStandardMaterial({ color: 0x34373c }));
        pl.position.fromArray(g.pos); S.visGroup.add(pl);
      }
      continue;
    }
    const geo = geomGeometry(g, sc.meshes, blob); if (!geo) continue;
    const pos = new THREE.Vector3().fromArray(g.pos || [0, 0, 0]), quat = mjq(g.quat || [1, 0, 0, 0]);
    if (g.collide) {
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x38b6ff, transparent: true, opacity: 0.2, depthWrite: false }));
      mesh.position.copy(pos); mesh.quaternion.copy(quat);
      mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: 0x7fd3ff })));
      S.collGroup.add(mesh);
    } else {
      const c = g.rgba || [0.6, 0.6, 0.6, 1];
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), transparent: c[3] < 1, opacity: c[3] }));
      mesh.position.copy(pos); mesh.quaternion.copy(quat);
      S.visGroup.add(mesh);
    }
  }
  if (sc.ref_image) { $('bd0').src = sc.ref_image; $('bd1').src = sc.ref_image; } else { $('bd0').removeAttribute('src'); $('bd1').removeAttribute('src'); }
  if (sc.glb) {
    status('loading game-style replica mesh...');
    try {
      const gltf = await new GLTFLoader().loadAsync(sc.glb);
      if (token !== S.sceneToken) return;
      const root = gltf.scene;
      orientGLB(root, sc.visual_bbox);
      root.traverse((o) => { if (o.isMesh && o.material) o.material.side = THREE.DoubleSide; });
      S.worldMesh = root; scene.add(root);
    } catch (e) { console.warn(e); }
  }
  // no mesh at all: show collision boxes so the world is not empty
  if (!S.worldMesh && S.collGroup.children.length) $('tColl').checked = true;
  setView(sc.camera ? S.view : 'orbit', true);
  status('');
}

function orientGLB(root, expect) {
  // glTF is Y-up; world/reconstruct.py writes env.glb Y-up. Match against the scene.xml visual bbox when known.
  const box = new THREE.Box3().setFromObject(root);
  let yup = true;
  if (expect) {
    const e0 = new THREE.Vector3().fromArray(expect[0]), e1 = new THREE.Vector3().fromArray(expect[1]);
    const asIs = box.min.distanceTo(e0) + box.max.distanceTo(e1);
    root.rotation.x = Math.PI / 2; root.updateMatrixWorld(true);
    const b2 = new THREE.Box3().setFromObject(root);
    const rot = b2.min.distanceTo(e0) + b2.max.distanceTo(e1);
    root.rotation.x = 0; root.updateMatrixWorld(true);
    yup = rot < asIs;
  }
  if (yup) root.rotation.x = Math.PI / 2;
}

// ------------------------------------------------------------------ performer (H1) per take
async function loadTake(stuntId, name, token) {
  const sid = encodeURIComponent(stuntId);
  const traj = await api(`/api/stunts/${sid}/trajectory?take=${name}`);
  if (token !== S.stuntToken) return;
  let p = traj.performer && traj.performer.geoms ? traj.performer : null;
  let blob = null;
  try {
    if (!p) p = await api(traj.performer_url || `/api/stunts/${sid}/performer?take=${name}`);
    if (p.meshes && Object.keys(p.meshes).length) blob = await fetchBin(p.bin_url || traj.performer_bin_url || `/api/stunts/${sid}/performer.bin?take=${name}`);
  } catch (e) { console.warn('performer meshes unavailable, using markers', e); p = null; }
  if (token !== S.stuntToken) return;
  const group = new THREE.Group(); group.name = `performer_${name}`;
  const bodies = {}; const mats = [];
  const tbodies = traj.bodies || traj.links || [];
  for (const b of (p ? p.bodies.map((x) => x.name) : tbodies)) { const o = new THREE.Group(); o.name = b; bodies[b] = o; group.add(o); }
  if (p) {
    const geoCache = {};
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
  } else {
    const geo = new THREE.SphereGeometry(0.055, 14, 10);
    for (const b of tbodies) {
      const base = new THREE.Color('#c9ced6');
      const mat = new THREE.MeshStandardMaterial({ color: base.clone() });
      bodies[b].add(new THREE.Mesh(geo, mat)); mats.push({ body: b, mat, base });
    }
  }
  scene.add(group);
  const nb = tbodies.length;
  const pos = traj.pos && Array.isArray(traj.pos[0]) ? traj.pos.flat(2) : traj.pos;
  const quat = traj.quat && Array.isArray(traj.quat[0]) ? traj.quat.flat(2) : traj.quat;
  const perf = { name, group, bodies, mats, traj: { ...traj, bodies: tbodies, pos, quat }, nb, color: takeColor(name) };
  S.perf[name] = perf; S.traj[name] = traj;
  S.dur = Math.max(S.dur, traj.t[traj.t.length - 1] || 0);
}

function frameOf(ts, t) {
  let lo = 0, hi = ts.length - 1;
  if (hi < 0 || t <= ts[0]) return 0;
  if (t >= ts[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m; }
  return lo;
}
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion();
function poseTake(perf, t) {
  const tr = perf.traj;
  const k = frameOf(tr.t, t), k2 = Math.min(k + 1, tr.t.length - 1);
  const a = k2 > k ? Math.min(1, Math.max(0, (t - tr.t[k]) / (tr.t[k2] - tr.t[k]))) : 0;
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
    if (k.includes(part) && targets.some((t) => body.includes(t))) return true;
  }
  return false;
}
function stressOf(traj) {
  // accept {stress: {link: idx}}, {stress_series: {link: [..]}}, {link_names, link_index[[..]]} or {colors: {body: c}}
  const st = traj.stress || traj.link_stress || {};
  const ss = { ...(traj.stress_series || {}) };
  const ser = traj.biomech_series || traj.series || {};
  if (Array.isArray(ser.link_names) && Array.isArray(ser.link_index)) {
    ser.link_names.forEach((n, c) => { ss[n] = ser.link_index.map((row) => row[c]); });
  }
  return { st, ss, colors: traj.stress_colour || traj.colors || traj.link_colors || traj.stress_colors || null, ts: traj.stress_t || ser.t || traj.t };
}
function colourOf(c) {
  if (Array.isArray(c)) return new THREE.Color(c[0], c[1], c[2]);
  if (c === 'red') return COLORS.bad; if (c === 'amber' || c === 'yellow') return COLORS.warn; if (c === 'green') return COLORS.ok;
  try { return new THREE.Color(c); } catch { return null; }
}
const stressColour = (x) => (x >= 1 ? COLORS.bad : x >= 0.5 ? COLORS.warn : COLORS.ok);
function colourTake(perf, t) {
  const on = $('tStress').checked;
  if (!perf.stress) {
    const s = stressOf(perf.traj); const map = {};
    const keys = [...new Set([...Object.keys(s.st), ...Object.keys(s.ss)])];
    for (const body of Object.keys(perf.bodies)) { const ks = keys.filter((k) => keyMatchesBody(k, body)); if (ks.length) map[body] = ks; }
    perf.stress = { ...s, map };
  }
  const { st, ss, map, colors, ts } = perf.stress;
  const kf = ts ? frameOf(ts, t) : 0;
  for (const m of perf.mats) {
    let c = null, v = 0;
    if (on && map[m.body]) {
      for (const k of map[m.body]) { const ser = ss[k]; v = Math.max(v, ser ? ser[Math.min(kf, ser.length - 1)] : (num(st[k]) ?? 0)); }
      c = stressColour(v);
    } else if (on && colors && colors[m.body] != null) {
      const cc = colors[m.body];
      c = colourOf(Array.isArray(cc) && Array.isArray(cc[0]) ? cc[Math.min(kf, cc.length - 1)] : cc);
    }
    if (!c) { m.mat.color.copy(m.base); m.mat.emissive.setRGB(0, 0, 0); continue; }
    m.mat.color.copy(m.base).lerp(c, 0.85);
    m.mat.emissive.copy(c).multiplyScalar(v >= 1 ? 0.35 : 0.08);
  }
}

// ------------------------------------------------------------------ viewports + camera
function shownTakes() {
  const withTraj = S.takes.filter((t) => S.perf[t]);
  if ($('tSplit').checked && withTraj.length >= 2) {
    const i = withTraj.indexOf(S.active);
    if (i > 0) return [withTraj[i - 1], withTraj[i]];
    return withTraj.slice(0, 2);
  }
  return S.active && S.perf[S.active] ? [S.active] : [];
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
  const ref = S.view === 'ref' && S.scene?.camera;
  const rects = [];
  for (let i = 0; i < n; i++) {
    const x = (W / n) * i, w = W / n;
    rects.push(ref ? fitRect(x + 4, 4, w - 8, H - 8, S.scene.camera.aspect || 16 / 9) : { x, y: 0, w, h: H });
  }
  return { rects, shown, W, H };
}
function setRefCamera() {
  const c = S.scene.camera;
  camera.position.fromArray(c.pos); camera.quaternion.copy(mjq(c.quat));
  camera.fov = c.fovy || 60; camera.updateProjectionMatrix();
}
function focusPoint() {
  const p = S.active && S.perf[S.active]?.bodies?.pelvis;
  if (p) return p.getWorldPosition(new THREE.Vector3());
  const box = new THREE.Box3();
  if (S.worldMesh) box.setFromObject(S.worldMesh); else box.setFromObject(S.worldRoot);
  return box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
}
function setView(v, reset) {
  if (reset) S.orbitInit = false;
  S.view = v;
  if (v === 'orbit') {
    controls.enabled = true;
    if (!S.orbitInit) {
      S.orbitInit = true;
      const c = S.scene?.camera; let tgt = focusPoint();
      if (c) {
        // start at the reference camera pose (always in free space); orbit about a point 4 m down its view axis
        camera.position.fromArray(c.pos);
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(mjq(c.quat));
        tgt = camera.position.clone().addScaledVector(fwd, 4);
      } else camera.position.copy(tgt).add(new THREE.Vector3(-7, -7, 5));
      camera.fov = 50; camera.updateProjectionMatrix();
      controls.target.copy(tgt);
    }
    controls.update();
  } else controls.enabled = false;
  document.querySelectorAll('.seg button[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  $('hud').textContent = v === 'orbit' ? 'drag: orbit · right-drag: pan · wheel: zoom' : (S.scene?.camera ? 'camera match: the reference camera from the video' : '');
}
function placeOverlays(rects, shown) {
  for (let i = 0; i < 2; i++) {
    const img = $('bd' + i), lbl = $('lbl' + i), r = rects[i];
    const show = r && S.view === 'ref' && S.scene?.ref_image && $('tFrame').checked;
    img.style.opacity = show ? 0.5 : 0; img.style.zIndex = 2;
    if (r) Object.assign(img.style, { left: r.x + 'px', top: r.y + 'px', width: r.w + 'px', height: r.h + 'px' });
    const take = shown[i];
    lbl.hidden = !(r && take && shown.length > 1);
    if (!lbl.hidden) { lbl.textContent = take; lbl.style.left = (r.x + 8) + 'px'; lbl.style.top = (r.y + 74) + 'px'; lbl.style.color = S.perf[take].color; }
  }
}

let lastFrame = performance.now(), lastW = 0, lastH = 0;
function tick(now) {
  try { frame(now); } catch (e) { console.error(e); }
  requestAnimationFrame(tick);
}
function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
  if (S.playing && S.dur > 0) { S.t += dt * S.speed; if (S.t > S.dur) S.t = 0; syncTime(false); }
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (W !== lastW || H !== lastH) { renderer.setSize(W, H, false); lastW = W; lastH = H; }
  for (const p of Object.values(S.perf)) poseTake(p, S.t);
  if (S.view === 'orbit') {
    if ($('tFollow').checked && S.active && S.perf[S.active]) {
      const tgt = S.perf[S.active].bodies.pelvis?.getWorldPosition(new THREE.Vector3());
      if (tgt) { const d = tgt.clone().sub(controls.target); controls.target.add(d); camera.position.add(d); }
    }
    controls.update();
  } else if (S.scene?.camera) setRefCamera();
  const { rects, shown } = layout();
  if (S.worldMesh) S.worldMesh.visible = $('tMesh').checked;
  S.collGroup.visible = $('tColl').checked;
  renderer.setScissor(0, 0, W, H); renderer.setViewport(0, 0, W, H); renderer.setClearColor(0x050607, 1); renderer.clear();
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

function syncTime(fromScrub) {
  if (!fromScrub) $('scrub').value = S.dur ? S.t / S.dur : 0;
  $('time').textContent = `${S.t.toFixed(2)} / ${S.dur.toFixed(2)} s`;
}
function togglePlay(force) {
  S.playing = force ?? !S.playing; $('play').textContent = S.playing ? 'Pause' : 'Play';
}

// ------------------------------------------------------------------ jobs
const JOB_LABEL = { build: 'Build world', edit: 'Edit world', verify: 'Verify', plan: 'Plan', answer: 'Re-plan', run: 'Run', improve: 'Improve' };
function trackJob(jobId, kind, ctx = {}) {
  S.jobs[jobId] = { id: jobId, kind, ctx, status: 'running', step: 'starting', log_tail: [], started: Date.now() / 1000 };
  saveJobs(); renderJobs(); refreshButtons();
}
function saveJobs() {
  const keep = Object.values(S.jobs).filter((j) => j.status === 'running').map((j) => ({ id: j.id, kind: j.kind, ctx: j.ctx }));
  store.set('jobs', JSON.stringify(keep));
}
function jobsFor(kind, key, val) { return Object.values(S.jobs).filter((j) => j.status === 'running' && (!kind || [].concat(kind).includes(j.kind)) && (!key || j.ctx[key] === val)); }
function renderJobs() {
  const js = Object.values(S.jobs).sort((a, b) => (b.started || 0) - (a.started || 0)).slice(0, 4);
  $('jobPills').innerHTML = js.map((j) => {
    const cls = j.status === 'running' ? 'run' : j.status === 'done' ? 'done' : 'fail';
    const label = `${JOB_LABEL[j.kind] || j.kind}${j.ctx.take ? ' ' + j.ctx.take : ''}${j.ctx.to ? ' ' + j.ctx.to : ''}`;
    const tail = j.status === 'running' ? `: ${j.step || '...'}` : j.status === 'done' ? ': done' : ': failed';
    return `<span class="pill ${cls}" title="${esc((j.log_tail || []).slice(-6).join('\n'))}">${esc(label + tail).slice(0, 60)}</span>`;
  }).join('');
  renderBuild();
}
function renderBuild() {
  const b = Object.values(S.jobs).filter((j) => j.kind === 'build').sort((a, c) => (c.started || 0) - (a.started || 0))[0];
  const show = b && (b.status === 'running' || b.status === 'failed' || (Date.now() / 1000 - (b.ended || 0) < 20));
  $('buildCard').hidden = !show;
  if (!show) return;
  $('buildStep').textContent = b.status === 'failed' ? 'failed' : b.status === 'done' ? 'done' : (b.step || '...');
  const el = (b.ended || Date.now() / 1000) - (b.started || Date.now() / 1000);
  $('buildBar').style.width = (b.status === 'done' ? 100 : Math.min(95, 4 + (el / 420) * 91)) + '%';
  $('buildBar').style.background = b.status === 'failed' ? 'var(--bad)' : b.status === 'done' ? 'var(--ok)' : 'var(--accent)';
  $('buildCard').querySelector('.spin').hidden = b.status !== 'running';
  const log = $('buildLog');
  log.textContent = (b.log_tail || []).join('\n') + (b.error ? `\n${b.error}` : '');
  log.scrollTop = log.scrollHeight;
}
async function pollJobs() {
  const running = Object.values(S.jobs).filter((j) => j.status === 'running');
  await Promise.all(running.map(async (j) => {
    try {
      const r = await api(`/api/jobs/${encodeURIComponent(j.id)}`);
      const st = r.status === 'queued' ? 'running' : r.status;
      Object.assign(j, { status: st, step: r.step, log_tail: r.log_tail || [], result: r.result, started: r.started || j.started, ended: r.ended,
        steps: r.steps, step_index: r.step_index, error: r.error });
      if (j.kind === 'build' && r.world_id && !j.ctx.world_id) {
        j.ctx.world_id = r.world_id; saveJobs();
        refreshWorlds().then(() => { if (!S.worldId || S.worldId === j.ctx.prev_world || !S.worlds.find((w) => w.id === S.worldId)) selectWorld(r.world_id); });
      }
      if (st !== 'running') { saveJobs(); onJobDone(j).catch((e) => console.error(e)); }
    } catch (e) {
      if (/no job|404/i.test(e.message)) { j.status = 'failed'; j.step = 'job not found'; saveJobs(); }
    }
  }));
  renderJobs(); refreshButtons();
  if (S.worldId && jobsFor('build').length) renderBuild();
}
async function onJobDone(j) {
  const ok = j.status === 'done';
  const res = j.result || {};
  if (j.kind === 'build') {
    await refreshWorlds();
    const wid = (res.world_dir ? String(res.world_dir).split('/').filter(Boolean).pop() : null) || j.ctx.world_id || (res.world_dir ? String(res.world_dir).split('/').filter(Boolean).pop() : null) || res.world_id;
    if (ok && wid) await selectWorld(wid);
  } else if (j.kind === 'edit' || j.kind === 'verify') {
    await refreshWorlds();
    if (j.ctx.world_id === S.worldId) { await loadWorldDetail(S.worldId); if (j.kind === 'edit' && ok) await loadScene(S.worldId); }
  } else if (j.kind === 'plan' || j.kind === 'answer') {
    const sid = res.stunt_id || (res.stunt_dir ? String(res.stunt_dir).split('/').filter(Boolean).pop() : null) || j.ctx.stunt_id;
    if (!ok) { msg(`Argon plan failed: ${(j.log_tail || []).slice(-1)[0] || ''}`, true); return; }
    if (j.ctx.world_id === S.worldId) { await refreshStunts(); if (sid) await selectStunt(sid, { questions: res.questions }); }
    msg(res.questions?.length ? `Argon has ${res.questions.length} question(s). Answer them or run v1 with its assumptions.` : 'Choreography ready. Run v1.');
  } else if (j.kind === 'run' || j.kind === 'improve') {
    if (j.ctx.stunt_id === S.stuntId) {
      const take = j.ctx.take || j.ctx.to;
      await selectStunt(S.stuntId, { keepQuestions: true, activate: take });
    }
    await refreshStunts(true);
    msg(ok ? `${j.ctx.to || j.ctx.take} done. Argon's critique is in the right panel.` : `${j.kind} failed: ${(j.log_tail || []).slice(-1)[0] || ''}`, !ok);
  }
}
function msg(text, bad) { const el = $('stuntMsg'); el.textContent = text || ''; el.style.color = bad ? 'var(--bad)' : ''; }

// ------------------------------------------------------------------ worlds
async function refreshWorlds() {
  try { S.worlds = await api('/api/worlds'); } catch (e) { console.warn(e); return; }
  S.worlds.sort((a, b) => String(b.created || b.id).localeCompare(String(a.created || a.id)));
  const sel = $('worldSel');
  sel.innerHTML = (S.worlds.length ? '' : '<option value="">no worlds yet</option>') + S.worlds.map((w) => {
    const st = w.status === 'ready' ? (w.fidelity_score != null ? `fidelity ${Math.round(w.fidelity_score)}` : 'ready') : w.status;
    const media = w.media ? String(w.media).split('/').pop() : '';
    return `<option value="${esc(w.id)}">${esc(w.name || w.id)}${media ? ' · ' + esc(media) : ''} · ${esc(st)} · ${esc(w.id)}</option>`;
  }).join('');
  if (S.worldId) sel.value = S.worldId;
  $('empty').hidden = !!S.worldId;
}
async function loadWorldDetail(id) {
  let d;
  try { d = await api(`/api/worlds/${encodeURIComponent(id)}`); } catch (e) { d = { error: e.message }; }
  if (id !== S.worldId) return d;
  S.worldDetail = d;
  renderWorld();
  return d;
}
function worldEntry() { return S.worlds.find((w) => w.id === S.worldId) || {}; }
function renderWorld() {
  const d = S.worldDetail || {}; const w = d.world || {}; const e = worldEntry();
  const st = e.status || (d.world ? 'ready' : 'building');
  $('worldStatus').textContent = st; $('worldStatus').className = 'pill ' + (st === 'ready' ? 'done' : st === 'failed' ? 'fail' : 'run');
  const objs = (w.objects || []).map((o) => `${esc(o.label || o.id)}${o.hazard ? ' <span class="warnc">⚠</span>' : ''}`);
  const media = w.source_media || e.media;
  const edits = (w.edits || []).map((x) => esc(x.prompt || x.text || JSON.stringify(x)));
  $('worldMeta').innerHTML = [`<b>${esc(e.name || S.worldId)}</b> <span class="dim">${esc(S.worldId)}</span>`,
    media ? `source: ${esc(String(media).split('/').pop())}` : '',
    objs.length ? `<details><summary>${objs.length} objects · ${(w.objects || []).filter((o) => o.hazard).length} hazards</summary>${objs.join(', ')}</details>` : (st !== 'ready' ? 'Still building: objects and mesh appear when the build finishes.' : ''),
    edits.length ? `edits: ${edits.join('; ')}` : '', d.error ? `<span class="bad">${esc(d.error)}</span>` : ''].filter(Boolean).join('<br>');
  // verify rounds
  let ver = Array.isArray(d.verify) && d.verify.length ? d.verify : (Array.isArray(w.verify) ? w.verify : []);
  ver = ver.filter(Boolean);
  const last = ver[ver.length - 1];
  const score = num(last?.score) ?? num(last?.fidelity) ?? e.fidelity_score;
  $('fidScore').textContent = score != null ? Math.round(score) : '-';
  const ring = $('fidRing');
  ring.style.setProperty('--p', score ?? 0);
  ring.style.setProperty('--ringc', score == null ? 'var(--mute)' : score >= 80 ? 'var(--ok)' : score >= 60 ? 'var(--warn)' : 'var(--bad)');
  $('verifyRounds').innerHTML = ver.length ? ver.map((v, i) => `<li>round ${esc(v.round ?? i + 1)}: <b>${fx(num(v.score) ?? num(v.fidelity), 0)}</b>${(v.issues || []).length ? ` · ${v.issues.length} issue(s)` : ''}${(v.fixes || v.corrections || []).length ? ` · ${(v.fixes || v.corrections).length} fix(es)` : ''}</li>`).join('')
    : '<li>no Argon verify yet</li>';
  const allIssues = last?.issues || [];
  const issues = allIssues.slice(0, 5);
  $('verifyIssues').innerHTML = issues.map((x) => {
    if (typeof x === 'string') return `<li>${esc(x)}</li>`;
    let obj = x.object || x.id || x.object_id || x.target;
    if (obj && typeof obj === 'object') obj = obj.label || obj.id || obj.name || '';
    const sev = String(x.severity || '').toLowerCase();
    const txt = x.issue || x.problem || x.text || x.description || JSON.stringify(x);
    return `<li class="sev-${esc(sev)}" title="${esc(txt)}">${obj ? `<span class="obj">${esc(obj)}</span> ` : ''}<span class="itxt">${esc(txt)}</span></li>`;
  }).join('') + (allIssues.length > 5 ? `<li class="dim">+${allIssues.length - 5} more</li>` : '');
  // views
  const views = (d.views || []).slice();
  const rank = (u) => (/side|vs|compare/.test(u) ? 0 : /ref/.test(u) ? 1 : /^r\d/.test(u.split('/').pop()) ? 2 : 3);
  views.sort((a, b) => rank(a) - rank(b) || b.localeCompare(a));
  $('views').innerHTML = views.slice(0, 8).map((u) => `<figure data-u="${esc(u)}"><img loading="lazy" src="${esc(u)}" alt=""><figcaption>${esc(u.split('/').pop().replace(/\.png$/, ''))}</figcaption></figure>`).join('');
  $('views').querySelectorAll('figure').forEach((f) => { f.onclick = () => openLightbox(f.dataset.u); });
  refreshButtons();
}
function openLightbox(u) { const lb = $('lightbox'); lb.querySelector('img').src = u; lb.querySelector('.cap').textContent = u.split('/').pop(); lb.hidden = false; }
$('lightbox').onclick = () => { $('lightbox').hidden = true; };

let worldPoll = null;
async function selectWorld(id) {
  if (!id) return;
  S.worldId = id; store.set('world', id);
  $('worldSel').value = id; $('empty').hidden = true;
  clearPerformers(); S.stunt = null; S.stuntId = null; S.takes = []; renderStunt();
  clearTimeout(worldPoll);
  await loadWorldDetail(id);
  const e = worldEntry();
  const ready = e.status ? e.status === 'ready' : !!S.worldDetail?.world;
  if (ready) {
    try { await loadScene(id); } catch (err) { console.error(err); status('could not load world scene: ' + err.message); }
  } else {
    clearWorld();
    status(e.status === 'failed' ? 'This world build failed. See the log on the left.' : 'This world is still building. Views appear on the left as Argon checks them.');
    const poll = async () => {
      if (S.worldId !== id) return;
      await refreshWorlds(); await loadWorldDetail(id);
      if (worldEntry().status === 'ready') { status(''); await loadScene(id); await refreshStunts(); } else worldPoll = setTimeout(poll, 6000);
    };
    worldPoll = setTimeout(poll, 6000);
  }
  await refreshStunts();
}

// ------------------------------------------------------------------ stunts
async function refreshStunts(keep) {
  if (!S.worldId) return;
  try { S.stunts = await api(`/api/stunts?world=${encodeURIComponent(S.worldId)}`); } catch (e) { S.stunts = []; }
  S.stunts.sort((a, b) => String(b.created || b.id).localeCompare(String(a.created || a.id)));
  const sel = $('stuntSel');
  sel.innerHTML = '<option value="">+ new stunt</option>' + S.stunts.map((s) => {
    const best = s.best && s.best.stunt_score != null ? ` · best ${s.best.take} ${fx(s.best.stunt_score, 0)}` : '';
    return `<option value="${esc(s.id)}">${esc((s.brief || s.id).slice(0, 48))} · ${(s.takes || []).length} take(s)${esc(best)}</option>`;
  }).join('');
  sel.value = S.stuntId || '';
  if (!keep && !S.stuntId && S.stunts.length && !params.has('nostunt')) await selectStunt(S.stunts[0].id);
}

async function selectStunt(id, opts = {}) {
  const token = ++S.stuntToken;
  if (!id) { S.stuntId = null; S.stunt = null; S.takes = []; clearPerformers(); renderStunt(); return; }
  S.stuntId = id; $('stuntSel').value = id;
  let d;
  try { d = await api(`/api/stunts/${encodeURIComponent(id)}`); } catch (e) { msg('could not load stunt: ' + e.message, true); return; }
  if (token !== S.stuntToken) return;
  S.stunt = d;
  const st = d.stunt || {};
  $('brief').value = st.brief || '';
  if (st.duration_sec || st.duration) $('duration').value = st.duration_sec || st.duration;
  const takesObj = d.takes || {};
  S.takes = Object.keys(takesObj).filter((k) => /^v\d+$/.test(k)).sort((a, b) => takeNum(a) - takeNum(b));
  const prevT = S.t;
  clearPerformers();
  S.active = opts.activate && S.takes.includes(opts.activate) ? opts.activate : S.takes[S.takes.length - 1] || null;
  S.critTab = S.active;
  if (!opts.keepQuestions || opts.questions) renderQuestions(opts.questions || (S.takes.length ? [] : st.questions) || []);
  renderStunt();
  if (S.takes.length) {
    status(`loading ${S.takes.length} take(s)...`);
    await Promise.all(S.takes.map((t) => loadTake(id, t, token).catch((e) => { console.warn(t, e); })));
    if (token !== S.stuntToken) return;
    status('');
    S.t = Math.min(prevT, S.dur);
    renderTakeRow(); syncTime(false);
    if (S.view === 'orbit') setView('orbit', true);
  }
}

function takeData(t) { return (S.stunt?.takes || {})[t] || {}; }
function loopRows() {
  const d = S.stunt; if (!d) return [];
  if (Array.isArray(d.loop) && d.loop.length) return d.loop;
  return S.takes.map((t) => {
    const td = takeData(t); const m = td.metrics || {}; const g1 = m.gate1 || {}; const b = td.biomech_summary || m.biomech || {}; const c = td.critique || {};
    return { take: t, stunt_score: m.score?.stunt_score, head_g_peak: g1.head_g_peak ?? num(b.head_g), hic15: g1.hic15 ?? num(b.hic15),
      knee_torque_peak_nm: num(b.knee_torque) != null ? Math.abs(num(b.knee_torque)) : null, peak_grf_bw: b.grf?.x_bw,
      min_clearance_cm: g1.min_clearance_cm, landing_error_m: g1.landing?.error_m, gate1_passed: g1.passed,
      gate2_passed: td.gate2?.passed ?? m.gate2?.passed ?? (td.critique ? c.approve === true : null), argon_style_score: c.overall_score, critique_summary: c.summary };
  });
}

function renderStunt() {
  renderTakeRow(); renderGates(); renderConvergence(); renderLoopTable(); renderBio(); renderCritique(); renderVideo(); refreshButtons();
}

function renderQuestions(qs) {
  S.questions = (qs || []).filter((q) => q && (q.question || typeof q === 'string'));
  $('questions').hidden = !S.questions.length;
  $('qList').innerHTML = S.questions.map((q, i) => {
    const text = typeof q === 'string' ? q : q.question;
    const opts = (q.options || []).map((o) => `<button type="button" data-i="${i}" data-o="${esc(o)}">${esc(o)}</button>`).join('');
    return `<div class="q"><p>${esc(text)}</p>${q.assumed ? `<div class="assumed">Argon assumed: ${esc(q.assumed)}</div>` : ''}
      ${opts ? `<div class="opts">${opts}</div>` : ''}<input data-i="${i}" placeholder="your answer (blank keeps the assumption)"></div>`;
  }).join('');
  $('qList').querySelectorAll('.opts button').forEach((b) => { b.onclick = () => { $('qList').querySelector(`input[data-i="${b.dataset.i}"]`).value = b.dataset.o; }; });
}

function renderTakeRow() {
  const names = S.takes.filter((t) => S.perf[t]);
  $('takeRow').innerHTML = names.map((n) => `<button data-t="${n}" class="${n === S.active ? 'on' : ''}" style="border-color:${takeColor(n)};${n === S.active ? `background:${takeColor(n)}` : ''}">${n}</button>`).join('');
  $('takeRow').querySelectorAll('button').forEach((b) => { b.onclick = () => setActive(b.dataset.t); });
  $('stressLegend').hidden = !names.length || !$('tStress').checked;
  $('tSplit').parentElement.style.opacity = names.length >= 2 ? 1 : 0.4;
}
function setActive(t) {
  S.active = t; S.critTab = t;
  renderTakeRow(); renderGates(); renderLoopTable(); renderBio(); renderCritique(); renderVideo(); renderConvergence();
}

// ---- gates
function renderGates() {
  const t = S.active; const row = loopRows().find((r) => r.take === t) || {};
  const td = takeData(t); const m = td.metrics || {};
  const lock = S.stuntId ? store.get('lock.' + S.stuntId) : null;
  const g = (name, title, state, sub) => `<div class="gate ${state === true ? 'pass' : state === false ? 'fail' : 'wait'}" ${name === 'G3' ? 'id="gate3" style="cursor:pointer" title="Director: click to lock / unlock this take"' : ''}>
    <div class="g">${name}</div><div class="n">${title}</div><div class="st">${state === true ? '✓ PASS' : state === false ? '✕ FAIL' : '· pending'}</div><div class="small dim">${sub || ''}</div></div>`;
  const g1 = row.gate1_passed ?? m.gate1?.passed;
  const g2 = row.gate2_passed ?? td.gate2?.passed ?? m.gate2?.passed;
  const g3 = lock ? (lock === t ? true : null) : null;
  $('gateTake').textContent = t ? `take ${t}` : '';
  $('gates').innerHTML = !t ? '<div class="small dim" style="grid-column:1/4">Run a take to evaluate the gates.</div>'
    : g('G1', 'Physics + biomech', g1, 'head G, clearance, landing') + g('G2', 'Argon visual critic', g2, row.argon_style_score != null ? `style ${fx(row.argon_style_score, 0)}` : 'style + framing')
      + g('G3', 'Director lock', g3, lock ? (lock === t ? 'locked' : `locked: ${lock}`) : 'click to lock');
  const g2d = td.gate2 || m.gate2 || {};
  const fails = [...new Set([...(m.gate1?.failures || []),
    ...(g2d.passed === false ? (g2d.reasons || g2d.failures || []).filter((r) => !/^gate 1 not passed/i.test(r)) : [])])];
  $('gateNotes').innerHTML = fails.length ? `<ul>${fails.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>` : '';
  const g3el = $('gate3');
  if (g3el) g3el.onclick = () => { store.set('lock.' + S.stuntId, lock === t ? null : t); renderGates(); };
}

// ---- convergence chart (SVG; one y-axis 0-100)
function renderConvergence() {
  const rows = loopRows(); const svg = $('conv');
  const W = svg.clientWidth || 360, H = 150, pad = { l: 28, r: 14, t: 12, b: 22 };
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const series = [{ key: 'stunt_score', name: 'Stunt score', color: '#e8eaed' }, { key: 'argon_style_score', name: 'Argon style score', color: '#c77dff', dash: '5 3' }];
  if (!rows.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="#5d646e" font-size="11" text-anchor="middle">No takes yet: plan a stunt and run v1</text>`;
    $('convLegend').innerHTML = ''; return;
  }
  const n = rows.length;
  const X = (i) => (n === 1 ? (pad.l + W - pad.r) / 2 : pad.l + (i / (n - 1)) * (W - pad.l - pad.r));
  const vals = rows.flatMap((r) => series.map((s) => r[s.key])).filter((v) => v != null);
  const lo = Math.max(0, Math.floor((Math.min(85, ...vals) - 10) / 10) * 10);
  const Y = (v) => H - pad.b - ((Math.max(lo, Math.min(100, v)) - lo) / (100 - lo)) * (H - pad.t - pad.b);
  let h = '';
  for (const v of [lo, Math.round((lo + 100) / 2), 100]) h += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#252a32"/><text x="${pad.l - 5}" y="${Y(v) + 3}" fill="#8b929c" font-size="10" text-anchor="end">${v}</text>`;
  h += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(85)}" y2="${Y(85)}" stroke="#5d646e" stroke-dasharray="3 3"/><text x="${pad.l + 4}" y="${Y(85) - 3}" fill="#8b929c" font-size="9.5">gate 2 needs 85</text>`;
  rows.forEach((r, i) => { h += `<text x="${X(i)}" y="${H - 6}" fill="${r.take === S.active ? '#e8eaed' : '#8b929c'}" font-size="10.5" font-weight="${r.take === S.active ? 700 : 400}" text-anchor="middle">${esc(r.take)}</text>`; });
  for (const s of series) {
    const pts = rows.map((r, i) => (r[s.key] != null ? [X(i), Y(r[s.key])] : null)).filter(Boolean);
    if (pts.length > 1) h += `<polyline points="${pts.map((p) => p.join(',')).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`;
    rows.forEach((r, i) => {
      if (r[s.key] == null) return;
      h += `<circle cx="${X(i)}" cy="${Y(r[s.key])}" r="4.5" fill="${s.color}" stroke="#15181d" stroke-width="2"/>`;
    });
  }
  // direct label for the last stunt score
  const lastR = rows[n - 1];
  if (lastR.stunt_score != null) h += `<text x="${X(n - 1) - 8}" y="${Y(lastR.stunt_score) + (lastR.stunt_score > 92 ? 16 : -8)}" fill="#e8eaed" font-size="11" font-weight="700" text-anchor="${n === 1 ? 'middle' : 'end'}">${fx(lastR.stunt_score, 0)}</text>`;
  // hover targets
  rows.forEach((r, i) => { h += `<rect class="hit" data-i="${i}" x="${X(i) - 22}" y="${pad.t}" width="44" height="${H - pad.t - pad.b + 14}" fill="transparent"/>`; });
  svg.innerHTML = h;
  $('convLegend').innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
  const tip = $('convTip');
  svg.querySelectorAll('rect.hit').forEach((el) => {
    const r = rows[+el.dataset.i];
    el.onmouseenter = () => {
      tip.innerHTML = `<b>${esc(r.take)}</b> · stunt ${fx(r.stunt_score, 1)} · style ${fx(r.argon_style_score, 0)}<br>head ${fx(r.head_g_peak, 2)} G · HIC ${fx(r.hic15, 1)} · knee ${fx(r.knee_torque_peak_nm, 0)} N·m<br>G1 ${r.gate1_passed ? '✓' : '✕'} · G2 ${r.gate2_passed ? '✓' : r.gate2_passed === false ? '✕' : '-'}`;
      tip.hidden = false;
      const x = X(+el.dataset.i) * (svg.clientWidth / W);
      tip.style.left = Math.max(0, Math.min(svg.clientWidth - 220, x - 100)) + 'px'; tip.style.top = '-6px';
    };
    el.onmouseleave = () => { tip.hidden = true; };
    el.onclick = () => { if (S.takes.includes(r.take)) setActive(r.take); };
    el.style.cursor = 'pointer';
  });
}
function renderLoopTable() {
  const rows = loopRows();
  if (!rows.length) { $('loopTable').innerHTML = ''; return; }
  const cell = (v, d, cls) => `<td class="${cls || ''}">${fx(v, d)}</td>`;
  const gate = (v) => `<td class="${v ? 'ok' : v === false ? 'bad' : 'dim'}">${v ? '✓' : v === false ? '✕' : '-'}</td>`;
  $('loopTable').innerHTML = `<tr><th>take</th><th title="stunt score 0-100">score</th><th title="peak head G">head G</th><th>HIC15</th><th title="peak knee torque N·m">knee</th><th title="peak GRF x body weight">GRF</th><th title="min clearance cm">clear</th><th title="landing error m">land</th><th>G1</th><th>G2</th><th title="Argon style score">style</th></tr>`
    + rows.map((r) => `<tr class="${r.take === S.active ? 'active' : ''}" data-t="${esc(r.take)}" style="cursor:pointer"><td style="color:${takeColor(r.take)};font-weight:700">${esc(r.take)}</td>${cell(r.stunt_score, 1)}${cell(r.head_g_peak, 2, r.head_g_peak >= 8 ? 'bad' : '')}${cell(r.hic15, 0, r.hic15 >= 700 ? 'bad' : '')}${cell(r.knee_torque_peak_nm, 0, r.knee_torque_peak_nm >= 200 ? 'warn' : '')}${cell(r.peak_grf_bw, 1, r.peak_grf_bw >= 10 ? 'bad' : '')}${cell(r.min_clearance_cm, 1)}${cell(r.landing_error_m, 2)}${gate(r.gate1_passed)}${gate(r.gate2_passed)}${cell(r.argon_style_score, 0)}</tr>`).join('');
  $('loopTable').querySelectorAll('tr[data-t]').forEach((tr) => { tr.onclick = () => { if (S.takes.includes(tr.dataset.t)) setActive(tr.dataset.t); }; });
}

// ---- anatomical telemetry
function bioOf(t) {
  const td = takeData(t); const mb = td.metrics?.biomech; const bs = td.biomech_summary || {};
  if (mb && (mb.head_g || mb.knee_torque)) return { worst_link: bs.worst_link, flags: bs.flags, ...mb };
  if (bs.peaks) return { ...bs.peaks, worst_link: bs.worst_link, flags: bs.flags };
  return bs;
}
const idxColor = (x) => (x == null ? '#5d646e' : x >= 1 ? 'var(--bad)' : x >= 0.5 ? 'var(--warn)' : 'var(--ok)');
function sideStress(t, part, side) {
  const tr = S.traj[t]; if (!tr) return null;
  const { st, ss } = stressOf(tr); let v = null;
  for (const k of new Set([...Object.keys(st), ...Object.keys(ss)])) {
    const kl = k.toLowerCase(); if (!kl.includes(part) || (side && !kl.includes(side))) continue;
    const x = num(st[k]) ?? (ss[k] ? Math.max(...ss[k]) : null);
    if (x != null) v = Math.max(v ?? 0, x);
  }
  return v;
}
function renderBio() {
  const t = S.active; const b = bioOf(t);
  $('bioTake').textContent = t ? `take ${t}` : '';
  const items = [
    ['head', 'Head G (15 ms)', num(b.head_g), 'G', (v) => v / (b.head_g?.target || 8), `target ${b.head_g?.target || 8} G`, 2],
    ['hic', 'HIC15', num(b.hic15), '', (v) => v / (b.hic15?.limit || 700), `limit ${b.hic15?.limit || 700}`, 0],
    ['neck', 'Neck compression*', num(b.neck_compression), 'N', (v) => b.neck_compression?.index ?? v / 4000, 'limit 4.0 kN', 0],
    ['lumbar', 'Lumbar compression*', num(b.lumbar_compression), 'N', (v) => b.lumbar_compression?.index ?? v / 3400, 'limit 3.4 kN', 0],
    ['hip', 'Hip torque', num(b.hip_torque), 'N·m', (v) => b.hip_torque?.index ?? Math.abs(v) / (b.hip_torque?.limit || 300), `limit ${b.hip_torque?.limit || 300}`, 0],
    ['knee', 'Knee torque', num(b.knee_torque), 'N·m', (v) => b.knee_torque?.index ?? Math.abs(v) / (b.knee_torque?.limit || 200), `limit ${b.knee_torque?.limit || 200}`, 0],
    ['ankle', 'Ankle torque', num(b.ankle_torque), 'N·m', (v) => b.ankle_torque?.index ?? Math.abs(v) / (b.ankle_torque?.limit || 250), `limit ${b.ankle_torque?.limit || 250}`, 0],
    ['grf', 'Ground reaction', b.grf?.x_bw ?? null, '× BW', (v) => v / 10, 'flag > 10 × BW', 1],
  ];
  const idx = {};
  $('meters').innerHTML = !t ? '<div class="small dim">No take selected.</div>' : items.map(([key, label, v, unit, f, lim, d]) => {
    const x = v == null ? null : f(v); idx[key] = x;
    const src = b[{ head: 'head_g', hic: 'hic15', neck: 'neck_compression', lumbar: 'lumbar_compression', hip: 'hip_torque', knee: 'knee_torque', ankle: 'ankle_torque', grf: 'grf' }[key]]?.source;
    const srcTag = src ? `<span class="src" title="${esc(src)}">${String(src).startsWith('inv') ? 'ID' : 'meas'}</span>` : '';
    return `<div class="meter" title="${esc(lim)}${x != null ? ` · ${(x * 100).toFixed(0)}% of limit` : ''}"><div class="top"><span class="lab">${label}${srcTag}</span><b>${v == null ? '-' : fx(Math.abs(v), d)} <span class="dim small">${unit}</span></b></div>
      <div class="trk"><i style="width:${x == null ? 0 : Math.min(100, (x / 1.5) * 100)}%;background:${idxColor(x)}"></i><u style="left:${100 / 1.5}%"></u></div></div>`;
  }).join('');
  // body figure: per-side where the trajectory carries per-link stress
  const sv = (part, side) => sideStress(t, part, side) ?? idx[part];
  const node = (x, y, r, v, lab) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${idxColor(v)}" fill-opacity="${v == null ? 0.3 : 0.9}" stroke="#15181d" stroke-width="1.5"><title>${lab}: ${v == null ? 'n/a' : (v * 100).toFixed(0) + '% of limit'}</title></circle>`;
  const bone = 'stroke="#3a414c" stroke-width="5" stroke-linecap="round"';
  $('body').innerHTML = `<g>
    <line x1="50" y1="30" x2="50" y2="95" ${bone}/><line x1="50" y1="42" x2="26" y2="80" ${bone}/><line x1="50" y1="42" x2="74" y2="80" ${bone}/>
    <line x1="40" y1="100" x2="38" y2="150" ${bone}/><line x1="60" y1="100" x2="62" y2="150" ${bone}/>
    <line x1="38" y1="150" x2="36" y2="196" ${bone}/><line x1="62" y1="150" x2="64" y2="196" ${bone}/>
    <line x1="36" y1="198" x2="26" y2="200" ${bone}/><line x1="64" y1="198" x2="74" y2="200" ${bone}/></g>
    ${node(50, 16, 11, idx.head == null ? null : Math.max(idx.head, idx.hic ?? 0), 'head (G / HIC)')}
    ${node(50, 34, 5, idx.neck, 'neck')}${node(50, 80, 6, idx.lumbar, 'lumbar')}
    ${node(40, 100, 6, sv('hip', 'right'), 'right hip')}${node(60, 100, 6, sv('hip', 'left'), 'left hip')}
    ${node(38, 150, 6, sv('knee', 'right'), 'right knee')}${node(62, 150, 6, sv('knee', 'left'), 'left knee')}
    ${node(36, 196, 5, sv('ankle', 'right'), 'right ankle')}${node(64, 196, 5, sv('ankle', 'left'), 'left ankle')}`;
  const flags = b.flags || [];
  const worst = b.worst_link;
  $('bioFlags').innerHTML = (worst ? `worst link: <b>${esc(worst.name)}</b> ${fx(worst.index, 2)}× limit at ${fx(worst.t, 2)} s<br>` : '')
    + flags.slice(0, 3).map(esc).join('<br>') + (t ? '<div class="dim">* neck/lumbar are H1 load proxies. ID = inverse dynamics (scripted phase), meas = free physics.</div>' : '');
}

// ---- timeline charts (canvas), take colours = identity
function chart(cv, key, label, unit, limits) {
  const ctx = cv.getContext('2d'); const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const pad = { l: 34, r: 8, t: 16, b: 14 };
  let ymax = 0; const series = [];
  for (const t of S.takes) {
    const tr = S.traj[t]; if (!tr) continue;
    const ser = tr.biomech_series || tr.series || {};
    const y = key === 'head_g' ? (ser.head_g || tr.head_g) : (ser[key] || tr[key]);
    const x = (ser.t && y && ser.t.length === y.length) ? ser.t : tr.t;
    if (y && x) { series.push({ take: t, x, y, color: takeColor(t) }); for (const v of y) if (v > ymax) ymax = v; }
  }
  for (const l of limits) if (l.v < ymax * 3 || !series.length) ymax = Math.max(ymax, l.v * 1.1);
  ymax = ymax || 1;
  const X = (t) => pad.l + (t / (S.dur || 1)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v / ymax) * (h - pad.t - pad.b);
  ctx.font = '10px -apple-system, sans-serif'; ctx.fillStyle = '#8b929c';
  ctx.fillText(`${label} (${unit})`, pad.l, 11);
  ctx.strokeStyle = '#22262d'; ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) { const v = (ymax * i) / 2; ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(v >= 10 ? 0 : 1), 2, Y(v) + 3); }
  for (const l of limits) {
    if (l.v > ymax) continue;
    ctx.setLineDash([4, 3]); ctx.strokeStyle = '#8b929c'; ctx.beginPath(); ctx.moveTo(pad.l, Y(l.v)); ctx.lineTo(w - pad.r, Y(l.v)); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = '#8b929c'; ctx.fillText(l.name, w - pad.r - ctx.measureText(l.name).width, Y(l.v) - 3);
  }
  const shown = shownTakes();
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.globalAlpha = s.take === S.active || shown.includes(s.take) ? 1 : 0.35;
    ctx.beginPath();
    const step = Math.max(1, Math.floor(s.x.length / (w * 2)));
    for (let i = 0; i < s.x.length; i += step) { const px = X(s.x[i]), py = Y(Math.abs(s.y[i])); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (!series.length) { ctx.fillStyle = '#5d646e'; ctx.fillText('no time series for this stunt', w / 2 - 60, h / 2); }
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(S.t), pad.t); ctx.lineTo(X(S.t), h - pad.b); ctx.stroke();
  if (series.length) {
    // value readout at the playhead for each take (text in ink, swatch carries identity)
    let yy = pad.t + 10;
    for (const s of series) {
      const v = s.y[frameOf(s.x, S.t)];
      ctx.fillStyle = s.color; ctx.fillRect(pad.l + 4, yy - 7, 8, 8);
      ctx.fillStyle = '#e8eaed'; ctx.fillText(`${s.take} ${Math.abs(v).toFixed(1)}`, pad.l + 16, yy); yy += 12;
    }
  }
  cv.onclick = (e) => { const r = cv.getBoundingClientRect(); S.t = Math.max(0, Math.min(S.dur, ((e.clientX - r.left - pad.l) / (w - pad.l - pad.r)) * S.dur)); syncTime(false); };
}
let chartTick = 0;
function drawCharts() {
  if (++chartTick % 2) return;
  chart($('chartG'), 'head_g', 'head acceleration', 'G', [{ name: 'target 8 G', v: 8 }]);
  chart($('chartK'), 'knee_torque', 'knee torque (max L/R)', 'N·m', [{ name: 'limit ~200 N·m', v: 200 }]);
}

// ---- critique
function renderList(v) {
  return '<ul>' + v.map((x) => {
    if (typeof x !== 'object' || !x) return `<li>${esc(x)}</li>`;
    const t = x.t_sec ?? x.t;
    const sev = String(x.severity || '').toLowerCase();
    const main = x.finding || x.issue || x.note || x.reason || x.event || x.text || '';
    let extra = '';
    if (x.name || x.path) extra = `<b>${esc(x.name || x.path)}</b>${x.value != null ? ` → ${esc(x.value)}` : ''}${x.old != null ? ` <span class="dim">(was ${esc(x.old)})</span>` : ''} `;
    if (x.event && (x.note)) extra = `<b>${esc(x.event)}</b> `;
    return `<li class="sev-${esc(sev)}">${t != null ? `<span class="t">${fx(t, 2)}s</span>` : ''}${extra}${esc(x.event && x.note ? x.note : main)}</li>`;
  }).join('') + '</ul>';
}
function renderCritique() {
  const names = S.takes;
  const cmp = S.stunt?.compare;
  $('critTabs').innerHTML = names.map((n) => `<button data-t="${n}" class="${n === S.critTab ? 'on' : ''}">${n}</button>`).join('')
    + (cmp ? `<button data-t="__compare" class="${S.critTab === '__compare' ? 'on' : ''}">${esc(cmp.from || 'v1')} → ${esc(cmp.to || 'v2')}</button>` : '');
  $('critTabs').querySelectorAll('button').forEach((b) => { b.onclick = () => { S.critTab = b.dataset.t; renderCritique(); }; });
  if (S.critTab === '__compare' && cmp) {
    const rows = cmp.rows || [];
    let h = `<table><tr><th>metric</th><th>${esc(cmp.from)}</th><th>${esc(cmp.to)}</th><th>Δ</th></tr>` + rows.map((r) => `<tr><td>${esc(r.metric)}</td><td>${fx(r[cmp.from], 2)}</td><td>${fx(r[cmp.to], 2)}</td><td>${r.delta == null ? '-' : (r.delta > 0 ? '+' : '') + fx(r.delta, 2)}</td></tr>`).join('') + '</table>';
    const ch = cmp.changes?.applied || [];
    if (ch.length) h += `<div class="hd">Changes Argon applied</div>${renderList(ch)}`;
    $('crit').innerHTML = h; return;
  }
  const c = takeData(S.critTab).critique;
  if (!S.critTab) { $('crit').innerHTML = '<span class="dim">Run a take to get Argon\'s critique.</span>'; return; }
  if (!c) { $('crit').innerHTML = `<span class="dim">No critique for ${esc(S.critTab)} yet.</span>`; return; }
  if (typeof c === 'string') { $('crit').innerHTML = `<p>${esc(c)}</p>`; return; }
  let h = '';
  const score = c.overall_score ?? c.score;
  if (score != null) h += `<div class="row" style="margin:0 0 6px"><span class="score">${esc(score)}</span><span class="dim">/ 100</span><span class="grow"></span>${c.approve != null ? `<span class="pill ${c.approve ? 'done' : 'fail'}">${c.approve ? 'Argon approves' : 'Argon wants changes'}</span>` : ''}</div>`;
  if (c.summary) h += `<p>${esc(c.summary)}</p>`;
  if (c.brief_match) h += `<div class="hd">Brief match: <span class="${c.brief_match.matches ? 'ok' : 'bad'}">${c.brief_match.matches ? 'yes' : 'no'}</span></div><p>${esc(c.brief_match.notes || '')}</p>${(c.brief_match.missing || []).length ? renderList(c.brief_match.missing) : ''}`;
  if (c.motion_plausibility) h += `<div class="hd">Motion plausibility ${fx(c.motion_plausibility.score, 0)}</div>${renderList(c.motion_plausibility.issues || [])}`;
  if ((c.safety_findings || []).length) h += `<div class="hd">Safety findings</div>${renderList(c.safety_findings)}`;
  if (c.framing) h += `<div class="hd">Framing ${c.framing.in_frame_pct != null ? fx(c.framing.in_frame_pct, 0) + '% in frame' : ''}</div><p>${esc(c.framing.notes || '')}</p>`;
  if ((c.changes || []).length) h += `<div class="hd">Changes for the next take</div>${renderList(c.changes)}`;
  if ((c.events || []).length) h += `<div class="hd">What Argon saw</div>${renderList(c.events)}`;
  const known = new Set(['overall_score', 'score', 'approve', 'summary', 'brief_match', 'motion_plausibility', 'safety_findings', 'framing', 'changes', 'events', 'take', 'video']);
  for (const [k, v] of Object.entries(c)) {
    if (known.has(k) || v == null) continue;
    if (typeof v === 'string' && v.length < 600) h += `<p><span class="dim">${esc(k)}:</span> ${esc(v)}</p>`;
    else if (typeof v === 'number' || typeof v === 'boolean') h += `<p><span class="dim">${esc(k)}:</span> ${esc(v)}</p>`;
  }
  $('crit').innerHTML = h || `<pre>${esc(JSON.stringify(c, null, 1))}</pre>`;
}
function renderVideo() {
  const t = S.active; const u = takeData(t).video_url;
  $('videoCard').hidden = !u;
  $('videoTake').textContent = t ? `take ${t} · server render` : '';
  if (u && $('video').getAttribute('src') !== u) $('video').src = u;
}

// ------------------------------------------------------------------ buttons + actions
function refreshButtons() {
  const wbusy = S.worldId && jobsFor(['edit', 'verify', 'build'], 'world_id', S.worldId).length;
  const ready = worldEntry().status === 'ready' || (!worldEntry().status && !!S.worldDetail?.world);
  $('editBtn').disabled = !S.worldId || !ready || !!wbusy;
  $('verifyBtn').disabled = !S.worldId || !ready || !!wbusy;
  const planBusy = S.worldId && jobsFor(['plan', 'answer'], 'world_id', S.worldId).length;
  const sbusy = S.stuntId && jobsFor(['run', 'improve', 'answer'], 'stunt_id', S.stuntId).length;
  $('planBtn').disabled = !S.worldId || !ready || !!planBusy;
  $('planBtn').textContent = planBusy ? 'Argon is planning...' : 'Plan with Argon';
  const last = S.takes[S.takes.length - 1];
  $('runBtn').disabled = !S.stuntId || !!sbusy;
  $('runBtn').textContent = S.takes.includes('v1') ? 'Re-run v1' : 'Run v1';
  const to = last ? `v${takeNum(last) + 1}` : null;
  $('improveBtn').disabled = !S.stuntId || !last || !takeData(last).critique || !!sbusy;
  $('improveBtn').textContent = last ? `Improve ${last} → ${to}` : 'Improve';
  $('answerBtn').disabled = !S.stuntId || !!sbusy;
  $('steerBtn').disabled = !S.stuntId || !!sbusy;
}

async function startUpload(file) {
  if (!file) return;
  if (!(file.type.startsWith('video/') || /\.(mp4|mov|webm|m4v)$/i.test(file.name))) { status('Choose a video of the location.'); return; }
  const fd = new FormData(); fd.append('video', file, file.name);
  status(`uploading ${file.name} (${(file.size / 1048576).toFixed(1)} MB)...`);
  try {
    const r = await api('/api/worlds', { method: 'POST', body: fd });
    status('');
    trackJob(r.job_id, 'build', { world_id: r.world_id || null, file: file.name, prev_world: S.worldId });
    await refreshWorlds();
    if (r.world_id) await selectWorld(r.world_id);
  } catch (e) { status('upload failed: ' + e.message); }
}
$('fileIn').onchange = (e) => { startUpload(e.target.files[0]); e.target.value = ''; };
$('fileIn2').onchange = (e) => { startUpload(e.target.files[0]); e.target.value = ''; };
$('worldSel').onchange = (e) => selectWorld(e.target.value);
$('stuntSel').onchange = (e) => {
  const id = e.target.value;
  if (!id) { selectStunt(null); $('brief').value = ''; renderQuestions([]); $('brief').focus(); } else selectStunt(id);
};
$('editBtn').onclick = async () => {
  const prompt = $('editPrompt').value.trim(); if (!prompt) { $('editPrompt').focus(); return; }
  try { const r = await api(`/api/worlds/${encodeURIComponent(S.worldId)}/edit`, { json: { prompt } }); trackJob(r.job_id, 'edit', { world_id: S.worldId }); $('editPrompt').value = ''; }
  catch (e) { status('edit failed: ' + e.message); }
};
$('verifyBtn').onclick = async () => {
  try { const r = await api(`/api/worlds/${encodeURIComponent(S.worldId)}/verify`, { method: 'POST' }); trackJob(r.job_id, 'verify', { world_id: S.worldId }); }
  catch (e) { status('verify failed: ' + e.message); }
};
$('planBtn').onclick = async () => {
  const brief = $('brief').value.trim(); if (!brief) { $('brief').focus(); msg('Describe the stunt first.', true); return; }
  const duration = parseFloat($('duration').value) || 6;
  try {
    const r = await api('/api/stunts', { json: { world_id: S.worldId, brief, duration } });
    trackJob(r.job_id, 'plan', { world_id: S.worldId }); msg('Argon is writing the choreography...');
  } catch (e) { msg('plan failed: ' + e.message, true); }
};
async function sendAnswers(answers, label) {
  try {
    const r = await api(`/api/stunts/${encodeURIComponent(S.stuntId)}/answer`, { json: { answers } });
    trackJob(r.job_id, 'answer', { world_id: S.worldId, stunt_id: S.stuntId }); msg(label);
  } catch (e) { msg('re-plan failed: ' + e.message, true); }
}
$('answerBtn').onclick = () => {
  const answers = (S.questions || []).map((q, i) => {
    const a = $('qList').querySelector(`input[data-i="${i}"]`)?.value.trim();
    return { question: typeof q === 'string' ? q : q.question, answer: a || (q.assumed ? `(keep assumption) ${q.assumed}` : '') };
  });
  sendAnswers(answers, 'Argon is re-planning with your answers...');
};
$('steerChips').querySelectorAll('button').forEach((b) => { b.onclick = () => { const s = $('steer'); s.value = s.value ? `${s.value}; ${b.dataset.s}` : b.dataset.s; }; });
$('steerBtn').onclick = () => {
  const s = $('steer').value.trim(); if (!s) { $('steer').focus(); return; }
  sendAnswers([{ question: 'Director steer', answer: s }], 'Argon is re-planning with your steer...'); $('steer').value = '';
};
$('runBtn').onclick = async () => {
  try { const r = await api(`/api/stunts/${encodeURIComponent(S.stuntId)}/run`, { json: { take: 'v1' } }); trackJob(r.job_id, 'run', { stunt_id: S.stuntId, take: 'v1' }); msg('Simulating v1, then Argon critiques it...'); }
  catch (e) { msg('run failed: ' + e.message, true); }
};
$('improveBtn').onclick = async () => {
  const from = S.takes[S.takes.length - 1]; const to = `v${takeNum(from) + 1}`;
  try { const r = await api(`/api/stunts/${encodeURIComponent(S.stuntId)}/improve`, { json: { from, to } }); trackJob(r.job_id, 'improve', { stunt_id: S.stuntId, from, to }); msg(`Applying Argon's changes: ${from} → ${to}...`); }
  catch (e) { msg('improve failed: ' + e.message, true); }
};
$('play').onclick = () => togglePlay();
$('scrub').oninput = (e) => { S.t = parseFloat(e.target.value) * S.dur; syncTime(true); };
$('speed').onchange = (e) => { S.speed = parseFloat(e.target.value); };
document.querySelectorAll('.seg button[data-view]').forEach((b) => { b.onclick = () => setView(b.dataset.view); });
$('tStress').onchange = () => renderTakeRow();
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName) && !['range', 'checkbox'].includes(e.target.type)) return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.key === 'ArrowRight') { S.t = Math.min(S.dur, S.t + 1 / 60); syncTime(false); }
  if (e.key === 'ArrowLeft') { S.t = Math.max(0, S.t - 1 / 60); syncTime(false); }
  if (e.key === 'Escape') $('lightbox').hidden = true;
});
window.addEventListener('resize', () => renderConvergence());

// ------------------------------------------------------------------ main
async function main() {
  try { for (const j of JSON.parse(store.get('jobs') || '[]')) S.jobs[j.id] = { ...j, status: 'running', step: 'reconnecting', log_tail: [] }; } catch { /* ignore */ }
  try {
    const KIND = { world_build: 'build', world_edit: 'edit', world_verify: 'verify', stunt_plan: 'plan', stunt_answer: 'answer', stunt_run: 'run', stunt_improve: 'improve' };
    const list = await api('/api/jobs');
    for (const r of Array.isArray(list) ? list : (list?.jobs || [])) {
      if (!['running', 'queued'].includes(r.status) || S.jobs[r.job_id]) continue;
      const kind = KIND[r.kind] || r.kind;
      S.jobs[r.job_id] = { id: r.job_id, kind: kind === 'plan' && r.parent_stunt_id ? 'answer' : kind, status: 'running', step: r.step, log_tail: r.log_tail || [], started: r.started,
        ctx: { world_id: r.world_id, stunt_id: r.stunt_id || r.parent_stunt_id, take: r.take, from: r.from, to: r.to } };
    }
  } catch (e) { console.warn('jobs list', e); }
  setInterval(() => { if (Object.values(S.jobs).some((j) => j.status === 'running')) pollJobs(); }, 1500);
  setInterval(() => refreshWorlds(), 15000);
  await refreshWorlds();
  const want = params.get('world') || store.get('world');
  const pick = S.worlds.find((w) => w.id === want) || S.worlds.find((w) => w.status === 'ready') || S.worlds[0];
  if (pick) {
    if (params.get('stunt')) S.stuntId = null;
    await selectWorld(pick.id);
    if (params.get('stunt')) await selectStunt(params.get('stunt'));
  } else $('empty').hidden = false;
  if (params.get('take') && S.takes.includes(params.get('take'))) setActive(params.get('take'));
  if (params.get('split') === '1') $('tSplit').checked = true;
  if (params.get('view')) setView(params.get('view'), true);
  if (params.get('t')) { S.t = parseFloat(params.get('t')); syncTime(false); }
  if (params.get('play') === '1') togglePlay(true);
  renderJobs();
  window.__fallguyReady = true;
}
window.fallguy = { S, setView, selectWorld, selectStunt, THREE };
requestAnimationFrame(tick);
main();
