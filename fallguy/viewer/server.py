"""FALLGUY viewer server: a stdlib HTTP server that feeds the Three.js page in static/.

Endpoints (all GET):
  /                         static/index.html (+ /static/*, vendored three in /static/vendor)
  /api/info                 world + stunt summary: reference camera, GLB url, ref frame url, takes, stunt.json, world.json
  /api/world                world geoms from the world's scene.xml (collision primitives, floor, visual primitives;
                            world-body meshes only when there is no GLB) + expected visual-mesh bbox (GLB axis check)
  /api/performer?take=v1    dynamic bodies of the take's scene.xml (the H1) with their visual geoms; meshes go in
  /api/performer.bin?take=  one binary blob (float32 verts, uint32 faces) cached in memory per take
  /api/traj?take=v1         60 Hz per-body positions/quaternions from trajectory.npz via mujoco forward kinematics
  /api/biomech?take=v1      biomech.json normalised: head G / knee torque series, per-link stress, peaks, limits
  /api/critique?take=v1     critique.json (raw) + compare.json at the stunt root
  /files/world/<rel>        files under the world dir (mesh/env.glb, textures, frames/ref.jpg)
  /files/stunt/<rel>        files under the stunt dir (v1/video.mp4, ...)
"""
import json
import math
import mimetypes
import os
import re
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
FPS = 60.0
GEOM_TYPES = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box", 7: "mesh", 8: "sdf"}


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _mat2quat(xmat):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=float).reshape(9))
    return q


def _r(a, n=5):
    return [round(float(x), n) for x in np.asarray(a).ravel()]


# ----------------------------------------------------------------------------- model loading
_compile_lock = threading.Lock()


def _compile(xml_path):
    import mujoco
    with _compile_lock:
        t0 = time.time()
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        print(f"[viewer] compiled {xml_path} in {time.time() - t0:.1f}s (nbody={m.nbody}, ngeom={m.ngeom})", flush=True)
        return m


def _name(model, objtype, i):
    import mujoco
    return mujoco.mj_id2name(model, objtype, i) or ""


def _geom_rgba(model, g):
    mat = model.geom_matid[g]
    if mat >= 0:
        rgba = model.mat_rgba[mat]
        if not np.allclose(rgba, 1.0) or np.allclose(model.geom_rgba[g], [0.5, 0.5, 0.5, 1]):
            return _r(rgba, 3)
    return _r(model.geom_rgba[g], 3)


# ----------------------------------------------------------------------------- the app state
class Viewer:
    def __init__(self, world_dir, stunt_dir=None):
        self.world_dir = Path(world_dir).resolve() if world_dir else None
        self.stunt_dir = Path(stunt_dir).resolve() if stunt_dir else None
        if self.world_dir is None and self.stunt_dir is not None:
            st = _read_json(self.stunt_dir / "stunt.json", {}) or {}
            wd = st.get("world_dir") or st.get("world")
            if isinstance(wd, str) and os.path.isdir(wd):
                self.world_dir = Path(wd).resolve()
        self._cache = {}
        self._lock = threading.Lock()

    # -- helpers
    def _cached(self, key, fn):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        val = fn()
        with self._lock:
            self._cache[key] = val
        return val

    def world_json(self):
        return (_read_json(self.world_dir / "world.json", {}) or {}) if self.world_dir else {}

    def takes(self):
        if not self.stunt_dir:
            return []
        out = []
        for d in sorted(self.stunt_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            if d.is_dir() and re.fullmatch(r"v\d+", d.name):
                out.append({"name": d.name,
                            "has_traj": (d / "trajectory.npz").is_file(),
                            "has_scene": (d / "scene.xml").is_file(),
                            "has_biomech": (d / "biomech.json").is_file(),
                            "has_critique": (d / "critique.json").is_file(),
                            "video": f"/files/stunt/{d.name}/video.mp4" if (d / "video.mp4").is_file() else None})
        return out

    def take_dir(self, take):
        if not self.stunt_dir or not re.fullmatch(r"v\d+", take or ""):
            raise FileNotFoundError(f"no take {take!r}")
        d = self.stunt_dir / take
        if not d.is_dir():
            raise FileNotFoundError(f"no take dir {d}")
        return d

    def world_glb(self):
        if not self.world_dir:
            return None
        for rel in ("mesh/env.glb", "mesh/world.glb", "env.glb"):
            if (self.world_dir / rel).is_file():
                return rel
        g = sorted((self.world_dir / "mesh").glob("*.glb")) if (self.world_dir / "mesh").is_dir() else []
        return str(g[0].relative_to(self.world_dir)) if g else None

    def ref_image(self):
        if not self.world_dir:
            return None
        frames = self.world_json().get("frames")
        if isinstance(frames, dict) and isinstance(frames.get("ref"), str) and (self.world_dir / frames["ref"]).is_file():
            return frames["ref"]
        fr = self.world_dir / "frames"
        if (fr / "ref.jpg").is_file():
            return "frames/ref.jpg"
        for pat in ("ref.*", "*.jpg", "*.png"):
            g = sorted(fr.glob(pat)) if fr.is_dir() else []
            if g:
                return str(g[0].relative_to(self.world_dir))
        return None

    def world_model(self):
        def build():
            xml = self.world_dir / "scene.xml" if self.world_dir else None
            if xml and xml.is_file():
                try:
                    return _compile(xml)
                except Exception as e:  # noqa: BLE001
                    print(f"[viewer] world scene.xml failed to compile: {e}", flush=True)
            # fall back to the first take's scene (world + H1)
            for t in self.takes():
                if t["has_scene"]:
                    try:
                        return _compile(self.stunt_dir / t["name"] / "scene.xml")
                    except Exception as e:  # noqa: BLE001
                        print(f"[viewer] take scene.xml failed to compile: {e}", flush=True)
            return None
        return self._cached("world_model", build)

    def take_model(self, take):
        d = self.take_dir(take)
        return self._cached(("take_model", take), lambda: _compile(d / "scene.xml"))

    # -- reference camera
    def ref_camera(self):
        import mujoco
        m = self.world_model()
        cam = None
        if m is not None and m.ncam > 0:
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            names = [_name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
            i = names.index("ref") if "ref" in names else 0
            cam = {"name": names[i], "pos": _r(d.cam_xpos[i]), "quat": _r(_mat2quat(d.cam_xmat[i])),
                   "fovy": float(m.cam_fovy[i]), "source": "scene.xml"}
        if cam is None:
            cam = self._camera_from_world_json()
        if cam is not None:
            cam["aspect"] = self._ref_aspect()
            cam["poses"] = self._camera_poses()
        return cam

    def _ref_aspect(self):
        wj = self.world_json().get("camera") or {}
        intr = wj.get("intrinsics") if isinstance(wj, dict) else None
        if isinstance(intr, dict):
            w, h = intr.get("width") or intr.get("w"), intr.get("height") or intr.get("h")
            if w and h:
                return float(w) / float(h)
        img = self.ref_image()
        if img:
            try:
                from PIL import Image
                with Image.open(self.world_dir / img) as im:
                    return im.width / im.height
            except Exception:
                pass
        return 9.0 / 16.0

    def _pose_from(self, p):
        """{pos, quat} | {pos, lookat} | {pos, xyaxes} -> {pos, quat (MuJoCo camera frame)}."""
        if not isinstance(p, dict) or "pos" not in p:
            return None
        import mujoco
        pos = np.asarray(p["pos"], float)
        if "quat" in p:
            q = np.asarray(p["quat"], float)
        else:
            if "xyaxes" in p:
                x, y = np.asarray(p["xyaxes"][:3], float), np.asarray(p["xyaxes"][3:], float)
            elif "lookat" in p or "target" in p:
                fwd = np.asarray(p.get("lookat", p.get("target")), float) - pos
                fwd /= np.linalg.norm(fwd) + 1e-9
                x = np.cross(fwd, [0, 0, 1.0])
                x /= np.linalg.norm(x) + 1e-9
                y = np.cross(x, fwd)
            else:
                return None
            x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
            z = np.cross(x, y)
            q = _mat2quat(np.stack([x, y, z], axis=1))
        out = {"pos": _r(pos), "quat": _r(q)}
        if "fovy" in p:
            out["fovy"] = float(p["fovy"])
        if "t" in p or "time" in p:
            out["t"] = float(p.get("t", p.get("time")))
        return out

    def _camera_from_world_json(self):
        c = self.world_json().get("camera") or {}
        for key in ("ref", "reference", "ref_pose", "reference_pose"):
            pose = self._pose_from(c.get(key)) if isinstance(c, dict) else None
            if pose:
                fovy = (pose.get("fovy") or c[key].get("fovy_deg") or (c.get("intrinsics") or {}).get("fovy")
                        or c.get("fovy") or 60.0)
                return {"name": "ref", "pos": pose["pos"], "quat": pose["quat"], "fovy": float(fovy),
                        "source": "world.json"}
        return None

    def _camera_poses(self):
        c = self.world_json().get("camera") or {}
        poses = c.get("poses") or c.get("frames") or [] if isinstance(c, dict) else []
        out = []
        for p in poses if isinstance(poses, list) else []:
            q = self._pose_from(p)
            if q and "t" in q:
                out.append(q)
        return out

    # -- endpoints
    def info(self):
        wj = self.world_json()
        st = (_read_json(self.stunt_dir / "stunt.json", {}) or {}) if self.stunt_dir else {}
        objects = [{"id": o.get("id"), "label": o.get("label"), "role": o.get("role"), "hazard": o.get("hazard")}
                   for o in (wj.get("objects") or []) if isinstance(o, dict)]
        verify = wj.get("verify") or wj.get("verify_history") or []
        glb = self.world_glb()
        img = self.ref_image()
        return {
            "world_dir": str(self.world_dir) if self.world_dir else None,
            "stunt_dir": str(self.stunt_dir) if self.stunt_dir else None,
            "world_name": self.world_dir.name if self.world_dir else None,
            "glb": f"/files/world/{glb}" if glb else None,
            "ref_image": f"/files/world/{img}" if img else None,
            "camera": self.ref_camera(),
            "frame_notes": wj.get("frame"),
            "objects": objects,
            "verify": verify if isinstance(verify, list) else [verify],
            "edits": wj.get("edits") or [],
            "takes": self.takes(),
            "stunt": {k: st.get(k) for k in ("brief", "duration", "duration_sec", "target", "phases", "clarifications")
                      if k in st},
            "compare": (_read_json(self.stunt_dir / "compare.json") if self.stunt_dir else None),
        }

    def world(self):
        return self._cached("world", self._world)

    def _world(self):
        import mujoco
        m = self.world_model()
        geoms, meshes, blob = [], {}, bytearray()
        bbox = None
        have_glb = self.world_glb() is not None
        if m is not None:
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
            for g in range(m.ngeom):
                b = m.geom_bodyid[g]
                if m.body_weldid[b] != 0:
                    continue  # dynamic (performer / props): not part of the world
                gtype = GEOM_TYPES.get(int(m.geom_type[g]), "?")
                collide = bool(m.geom_contype[g] or m.geom_conaffinity[g])
                entry = {"name": _name(m, mujoco.mjtObj.mjOBJ_GEOM, g), "type": gtype, "size": _r(m.geom_size[g]),
                         "pos": _r(d.geom_xpos[g]), "quat": _r(_mat2quat(d.geom_xmat[g])),
                         "rgba": _geom_rgba(m, g), "collide": collide, "group": int(m.geom_group[g])}
                if gtype == "mesh":
                    mid = int(m.geom_dataid[g])
                    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    v = m.mesh_vert[va:va + vn]
                    R = d.geom_xmat[g].reshape(3, 3)
                    wv = v @ R.T + d.geom_xpos[g]
                    lo, hi = wv.min(0), wv.max(0)
                    if not collide:
                        bbox = [_r(lo), _r(hi)] if bbox is None else [_r(np.minimum(bbox[0], lo)), _r(np.maximum(bbox[1], hi))]
                    if have_glb and not collide:
                        continue  # the GLB is the visual; skip the duplicate
                    name = _name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
                    if name not in meshes:
                        meshes[name] = _pack_mesh(m, mid, blob)
                    entry["mesh"] = name
                geoms.append(entry)
        else:
            # world.json collision primitives only
            for o in self.world_json().get("objects") or []:
                for i, c in enumerate(o.get("collision") or []):
                    geoms.append({"name": f"{o.get('id')}_{i}", "type": c.get("type", "box"), "size": c.get("size"),
                                  "pos": c.get("pos"), "quat": c.get("quat", [1, 0, 0, 0]),
                                  "rgba": [0.3, 0.7, 1.0, 0.5], "collide": True, "group": 0})
        return {"geoms": geoms, "meshes": meshes, "visual_bbox": bbox}, bytes(blob)

    def performer(self, take):
        return self._cached(("performer", take), lambda: self._performer(take))

    def _performer(self, take):
        import mujoco
        m = self.take_model(take)
        bodies, geoms, meshes, blob = [], [], {}, bytearray()
        dyn = [b for b in range(1, m.nbody) if m.body_weldid[b] != 0]
        for b in dyn:
            bodies.append({"id": b, "name": _name(m, mujoco.mjtObj.mjOBJ_BODY, b)})
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if m.body_weldid[b] == 0:
                continue
            visual = not (m.geom_contype[g] or m.geom_conaffinity[g])
            if m.geom_group[g] >= 3 or (not visual and m.geom_rgba[g][3] == 0):
                continue  # collision proxies
            gtype = GEOM_TYPES.get(int(m.geom_type[g]), "?")
            entry = {"body": _name(m, mujoco.mjtObj.mjOBJ_BODY, b), "type": gtype, "size": _r(m.geom_size[g]),
                     "pos": _r(m.geom_pos[g]), "quat": _r(m.geom_quat[g]), "rgba": _geom_rgba(m, g)}
            if gtype == "mesh":
                mid = int(m.geom_dataid[g])
                name = _name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or f"mesh{mid}"
                if name not in meshes:
                    meshes[name] = _pack_mesh(m, mid, blob)
                entry["mesh"] = name
            geoms.append(entry)
        return {"bodies": bodies, "geoms": geoms, "meshes": meshes}, bytes(blob)

    def traj(self, take):
        d = self.take_dir(take)
        npz = d / "trajectory.npz"
        key = ("traj", take, npz.stat().st_mtime)
        return self._cached(key, lambda: self._traj(take, npz))

    def _traj(self, take, npz):
        import mujoco
        m = self.take_model(take)
        z = np.load(npz)
        qpos = np.asarray(z["qpos"], float)
        if qpos.shape[1] != m.nq:
            raise ValueError(f"trajectory qpos has {qpos.shape[1]} columns, scene.xml nq={m.nq}")
        t = np.asarray(z["t"], float) if "t" in z.files else np.arange(len(qpos)) * m.opt.timestep
        times = np.arange(t[0], t[-1] + 1e-9, 1.0 / FPS)
        idx = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
        dyn = [b for b in range(1, m.nbody) if m.body_weldid[b] != 0]
        data = mujoco.MjData(m)
        P = np.zeros((len(idx), len(dyn), 3), np.float32)
        Q = np.zeros((len(idx), len(dyn), 4), np.float32)
        for k, i in enumerate(idx):
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(m, data)
            P[k] = data.xpos[dyn]
            Q[k] = data.xquat[dyn]
        out = {"fps": FPS, "t": _r(times - t[0], 4), "bodies": [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in dyn],
               "pos": _r(P, 4), "quat": _r(Q, 4)}
        if "head_g15" in z.files:
            out["head_g"] = _r(np.asarray(z["head_g15"])[idx], 3)
        ev = _read_json(npz.parent / "events.json")
        if ev is not None:
            out["events"] = ev
        return out

    def biomech(self, take):
        d = self.take_dir(take)
        raw = _read_json(d / "biomech.json")
        out = normalize_biomech(raw) if raw is not None else {"series": {}, "stress": {}, "peaks": {}, "limits": {}}
        if not out["series"].get("head_g") and (d / "trajectory.npz").is_file():
            try:
                tr = self.traj(take)
                if tr.get("head_g"):
                    out["series"]["t"], out["series"]["head_g"] = tr["t"], tr["head_g"]
                    out["notes"] = (out.get("notes") or []) + ["head G from trajectory.npz (no biomech series)"]
            except Exception:
                pass
        out["available"] = raw is not None
        return out

    def critique(self, take):
        d = self.take_dir(take)
        return {"critique": _read_json(d / "critique.json"), "metrics": _read_json(d / "metrics.json"),
                "params": _read_json(d / "params.json"),
                "compare": _read_json(self.stunt_dir / "compare.json") if self.stunt_dir else None}


def _pack_mesh(m, mid, blob):
    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    v = np.ascontiguousarray(m.mesh_vert[va:va + vn], dtype=np.float32)
    f = np.ascontiguousarray(m.mesh_face[fa:fa + fn], dtype=np.uint32)
    entry = {"vofs": len(blob), "vnum": vn}
    blob += v.tobytes()
    entry["fofs"] = len(blob)
    entry["fnum"] = fn
    blob += f.tobytes()
    return entry


# ----------------------------------------------------------------------------- biomech.json normaliser
_STRESS_KEYS = ("link_stress", "stress", "stress_index", "per_link", "links", "link_index")
_SCALAR_KEYS = ("index", "stress", "stress_index", "peak", "value", "ratio", "max")


def _num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return float(v)
    if isinstance(v, dict):
        for k in _SCALAR_KEYS:
            if k in v and isinstance(v[k], (int, float)):
                return float(v[k])
    return None


def _series_of(v):
    if isinstance(v, list) and len(v) > 2 and all(isinstance(x, (int, float)) or x is None for x in v[:5]):
        return [float(x) if x is not None else 0.0 for x in v]
    if isinstance(v, dict):
        for k in ("values", "series", "data", "y"):
            if k in v:
                return _series_of(v[k])
    return None


def _walk(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield path + (str(k),), v
            yield from _walk(v, path + (str(k),))


def normalize_biomech(raw):
    """Pull out (tolerantly) what the page draws: series t/head_g/knee_torque, per-link stress (peak and, if present,
    per-frame), peaks, limits. Expected shape (fallguy/biomech.py):
      {"t": [...] or "series": {"t": [...], "head_g": [...], "knee_torque": [...] | "left_knee": [...], ...},
       "stress": {"<link or body>": index | {"index": x, "series": [...], "source": "measured"|"inverse"}},
       "peaks": {...}, "limits": {...}}"""
    out = {"series": {}, "stress": {}, "stress_series": {}, "peaks": {}, "limits": {}, "notes": []}
    if not isinstance(raw, dict):
        return out
    # series
    ser = raw.get("series") or raw.get("timeseries") or raw.get("time_series") or {}
    if not isinstance(ser, dict):
        ser = {}
    pool = dict(ser)
    for k, v in raw.items():
        if k not in pool and _series_of(v) is not None:
            pool[k] = v
    t = _series_of(pool.get("t") or pool.get("time") or raw.get("t"))
    head, knee = None, []
    for k, v in pool.items():
        s = _series_of(v)
        if s is None:
            continue
        kl = k.lower()
        if head is None and "head" in kl and ("g" in kl or "acc" in kl) and "hic" not in kl:
            head = s
        elif "knee" in kl and ("torque" in kl or "moment" in kl or kl.endswith("knee") or "nm" in kl):
            knee.append(s)
        elif isinstance(v, dict) and "knee" in kl:
            knee.append(s)
    if head is None:
        for path, v in _walk(raw):
            p = "/".join(path).lower()
            s = _series_of(v)
            if s is not None and "head" in p and ("g" in p or "acc" in p) and "hic" not in p:
                head = s
                break
    if not knee:
        for path, v in _walk(raw):
            p = "/".join(path).lower()
            s = _series_of(v)
            if s is not None and "knee" in p and "stress" not in p:
                knee.append(s)
    jn, jt = ser.get("joint_names"), ser.get("joint_torque")
    if not knee and isinstance(jn, list) and isinstance(jt, list) and jt:
        cols = [i for i, j in enumerate(jn) if "knee" in str(j)]
        if cols:
            knee = [[abs(float(row[c])) for row in jt] for c in cols]
    n = len(head) if head else (len(knee[0]) if knee else 0)
    if t is None and n:
        fps = float(raw.get("fps") or raw.get("rate_hz") or 60.0)
        t = [i / fps for i in range(n)]
    if t:
        out["series"]["t"] = t
    if head:
        out["series"]["head_g"] = head
    if knee:
        L = min(len(s) for s in knee)
        out["series"]["knee_torque"] = [max(abs(s[i]) for s in knee) for i in range(L)]
    # per-link stress
    st = None
    for k in _STRESS_KEYS:
        if isinstance(raw.get(k), (dict, list)):
            st = raw[k]
            break
    if st is None:
        for path, v in _walk(raw):
            if path[-1].lower() in _STRESS_KEYS and isinstance(v, (dict, list)):
                st = v
                break
    if isinstance(st, list):
        st = {str(e.get("link") or e.get("name") or e.get("body") or i): e for i, e in enumerate(st) if isinstance(e, dict)}
    for k, v in (st or {}).items():
        val = _num(v)
        if val is not None:
            out["stress"][k] = val
        s = _series_of(v.get("series")) if isinstance(v, dict) and "series" in v else None
        if s is not None:
            out["stress_series"][k] = s
            if val is None:
                out["stress"][k] = max(s)
    ln, li = ser.get("link_names"), ser.get("link_index")
    if isinstance(ln, list) and isinstance(li, list) and li:
        for c, name in enumerate(ln):
            out["stress_series"][str(name)] = [float(row[c]) for row in li]
            out["stress"].setdefault(str(name), max(float(row[c]) for row in li))
    for k in ("flags", "worst_link", "t_release", "method"):
        if k in raw:
            out[k] = raw[k]
    for k in ("peaks", "peak", "summary"):
        if isinstance(raw.get(k), dict):
            out["peaks"] = raw[k]
            break
    if isinstance(raw.get("limits"), dict):
        out["limits"] = raw["limits"]
    return out


# ----------------------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    viewer: Viewer = None

    def log_message(self, fmt, *args):  # quiet
        if os.environ.get("FALLGUY_VIEWER_LOG"):
            super().log_message(fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)) or body is None:
            body = json.dumps(body, allow_nan=False, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, base, rel):
        if base is None:
            return self._send(404, {"error": "no dir"})
        p = (base / rel).resolve()
        if not str(p).startswith(str(base.resolve())) or not p.is_file():
            return self._send(404, {"error": f"not found: {rel}"})
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix == ".glb":
            ctype = "model/gltf-binary"
        size = p.stat().st_size
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):  # video seeking
            a, _, b = rng[6:].partition("-")
            a = int(a or 0)
            b = int(b) if b else size - 1
            with open(p, "rb") as f:
                f.seek(a)
                data = f.read(b - a + 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{a + len(data) - 1}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        with open(p, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        u = urllib.parse.urlparse(self.path)
        q = dict(urllib.parse.parse_qsl(u.query))
        path = urllib.parse.unquote(u.path)
        v = self.viewer
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC, "index.html")
            if path.startswith("/static/"):
                return self._file(STATIC, path[len("/static/"):])
            if path.startswith("/files/world/"):
                return self._file(v.world_dir, path[len("/files/world/"):])
            if path.startswith("/files/stunt/"):
                return self._file(v.stunt_dir, path[len("/files/stunt/"):])
            if path == "/api/info":
                return self._send(200, v.info())
            if path == "/api/world":
                return self._send(200, v.world()[0])
            if path == "/api/world.bin":
                return self._send(200, v.world()[1], "application/octet-stream")
            if path == "/api/performer":
                return self._send(200, v.performer(q.get("take", "v1"))[0])
            if path == "/api/performer.bin":
                return self._send(200, v.performer(q.get("take", "v1"))[1], "application/octet-stream")
            if path == "/api/traj":
                return self._send(200, v.traj(q.get("take", "v1")))
            if path == "/api/biomech":
                return self._send(200, v.biomech(q.get("take", "v1")))
            if path == "/api/critique":
                return self._send(200, v.critique(q.get("take", "v1")))
            return self._send(404, {"error": f"unknown path {path}"})
        except FileNotFoundError as e:
            return self._send(404, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def serve(world_dir, stunt_dir=None, host="127.0.0.1", port=8765, open_browser=True, warm=True):
    viewer = Viewer(world_dir, stunt_dir)
    handler = type("FallguyHandler", (Handler,), {"viewer": viewer})
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), handler)
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"no free port in {port}..{port + 19}")
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(json.dumps({"url": url, "world_dir": str(viewer.world_dir) if viewer.world_dir else None,
                      "stunt_dir": str(viewer.stunt_dir) if viewer.stunt_dir else None}), flush=True)
    if warm:  # compile + FK in the background so the first page load is fast
        def _warm():
            try:
                viewer.world()
                for t in viewer.takes():
                    if t["has_scene"] and t["has_traj"]:
                        viewer.performer(t["name"])
                        viewer.traj(t["name"])
            except Exception as e:  # noqa: BLE001
                print(f"[viewer] warm-up: {e}", flush=True)
        threading.Thread(target=_warm, daemon=True).start()
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
