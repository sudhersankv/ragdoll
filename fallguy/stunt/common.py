"""Shared helpers for fallguy.stunt: world loading, engine plan assembly, paths, JSON IO."""
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(PKG)


def say(msg):
    sys.stderr.write(f"[stunt] {msg}\n")
    sys.stderr.flush()


def rj(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wj(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def slug(s, n=40):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return (s[:n].rstrip("-") or "stunt")


def new_dir(root, brief):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(root, f"{slug(brief)}-{ts}")
    d, k = base, 2
    while os.path.exists(d):
        d, k = f"{base}_{k}", k + 1
    os.makedirs(d)
    return os.path.abspath(d)


def argon_log(stunt_dir):
    return os.path.join(stunt_dir, "logs", "argon_calls.jsonl")


# ----------------------------------------------------------------------------- world
def _ref_frame(world_dir, wjson):
    cam = (wjson.get("camera") or {})
    fr = wjson.get("frames") if isinstance(wjson.get("frames"), dict) else {}
    for c in (fr, cam.get("ref") or {}, cam.get("reference") or {}, cam):
        for k in ("ref", "frame", "image", "ref_frame"):
            v = c.get(k) if isinstance(c, dict) else None
            if isinstance(v, str):
                p = v if os.path.isabs(v) else os.path.join(world_dir, v)
                if os.path.isfile(p):
                    return p
    for pat in ("frames/ref.jpg", "frames/ref.png", "frames/*.jpg", "frames/*.png"):
        hits = sorted(glob.glob(os.path.join(world_dir, pat)))
        if hits:
            return hits[0]
    return None


def world_info(world_dir):
    """Everything stages 2-4 need from a stage-1 world dir: world.json, ref camera (from scene.xml camera "ref"),
    collision objects with world AABBs, room bounds, ref frame and the media's aspect."""
    import mujoco
    from PIL import Image
    world_dir = os.path.abspath(world_dir)
    wjson = rj(os.path.join(world_dir, "world.json")) if os.path.isfile(os.path.join(world_dir, "world.json")) else {}
    model = mujoco.MjModel.from_xml_path(os.path.join(world_dir, "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ref")
    if cid < 0 and model.ncam:
        cid = 0
    if cid >= 0:
        pos = data.cam_xpos[cid].copy()
        R = data.cam_xmat[cid].reshape(3, 3)
        fwd = -R[:, 2]
        cam = {"pos": [round(float(v), 4) for v in pos], "lookat": [round(float(v), 4) for v in pos + 6.0 * fwd],
               "fovy": round(float(model.cam_fovy[cid]), 2)}
    else:
        cam = {"pos": [0.0, 0.0, 1.5], "lookat": [-5.0, 0.0, 1.0], "fovy": 60.0}
    objs = []
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        R = data.geom_xmat[g].reshape(3, 3)
        c = data.geom_xpos[g] + R @ model.geom_aabb[g, :3]
        h = np.abs(R) @ model.geom_aabb[g, 3:]
        objs.append({"geom": name, "lo": (c - h).round(3).tolist(), "hi": (c + h).round(3).tolist(),
                     "type": int(model.geom_type[g])})
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    if not np.isfinite(lo).all():
        lo, hi = np.array([-10.0, -10.0, 0.0]), np.array([10.0, 10.0, 4.0])
    ref = _ref_frame(world_dir, wjson)
    w, h = (720, 1280)
    if ref:
        with Image.open(ref) as im:
            w, h = im.size
    src = wjson.get("source")
    media = wjson.get("source_media") or (src.get("media") if isinstance(src, dict) else src)
    return {"dir": world_dir, "json": wjson, "camera": cam, "objects": objs, "ref_frame": ref, "width": w, "height": h,
            "media": media if isinstance(media, str) else None,
            "bounds": {"x_min": float(lo[0]), "x_max": float(hi[0]), "y_min": float(lo[1]), "y_max": float(hi[1]),
                       "ceiling_z": float(max(hi[2], 3.0))}}


def engine_plan(W, stunt, brief="", duration=6.0):
    """The engine.* plan dict for a world-mode stunt: room from the world's collision bounds, camera.match = the
    world's ref camera, no plan obstacles (the world's scene.xml supplies them), stunt = the choreography."""
    b = W["bounds"]
    floor_mat = ((W["json"].get("floor") or {}).get("material")) or "concrete"
    return {
        "version": 2, "brief": brief, "duration_sec": float(duration),
        "source": {"media": W.get("media") or "", "kind": "video", "ref_frame": W.get("ref_frame") or "",
                   "width": W["width"], "height": W["height"]},
        "room": {"x_min": b["x_min"] - 0.5, "x_max": b["x_max"] + 0.5, "y_min": b["y_min"] - 0.5,
                 "y_max": b["y_max"] + 0.5, "ceiling_z": min(b["ceiling_z"], 12.0), "floor_material": floor_mat},
        "obstacles": [], "crash_mats": list(stunt.get("crash_mats") or []), "set_dressing": {},
        "camera": {"match": dict(W["camera"])},
        "stunt": {k: v for k, v in stunt.items() if k != "crash_mats"},
        "performer": {"model": "h1"},
        "world": {"dir": W["dir"]},
    }


def frame_size(W, long_side=1280):
    """Render size keeping the media's aspect: portrait -> 720x1280."""
    w, h = W["width"], W["height"]
    if h >= w:
        return (int(round(long_side * w / h / 2)) * 2, long_side)
    return (long_side, int(round(long_side * h / w / 2)) * 2)
