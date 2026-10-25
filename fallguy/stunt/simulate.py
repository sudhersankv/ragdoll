"""Stage 3: simulate one take (H1 in the world), biomechanics, render video.mp4 from the world's ref camera.

python -m fallguy.stunt simulate --stunt DIR --take v1 [--no-render]
"""
import json
import os
import time

import numpy as np

from .common import engine_plan, frame_size, rj, say, wj, world_info


def take_choreography(stunt_dir, take):
    """The take's stunt dict: <take>/choreography.json (improve writes it) else stunt.json's stunt."""
    p = os.path.join(stunt_dir, take, "choreography.json")
    return rj(p) if os.path.isfile(p) else rj(os.path.join(stunt_dir, "stunt.json"))["stunt"]


def _tint(bio):
    """callable(model, ts) -> (geom ids, rgba) colouring each H1 visual geom by its link's stress index
    (max over the last 0.25 s so short peaks stay visible for a few frames)."""
    import mujoco
    from .. import biomech
    s = bio["series"]
    t = np.asarray(s["t"], float)
    idx = np.asarray(s["link_index"], float)                     # (n, links)
    names = s["link_names"]
    cache = {}

    def fn(model, ts):
        if "ids" not in cache:
            from ..engine import performer as perf
            pf = perf.Performer(model)
            ids, cols = [], []
            for g in pf.visual_geoms:
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
                ids.append(int(g))
                cols.append(names.index(bname) if bname in names else -1)
            cache["ids"], cache["cols"] = np.array(ids, int), np.array(cols, int)
        sel = (t <= ts) & (t >= ts - 0.25)
        cur = idx[sel].max(0) if sel.any() else idx[int(np.clip(np.searchsorted(t, ts), 0, len(t) - 1))]
        rgba = np.array([biomech.rgba(cur[c]) if c >= 0 else (0.45, 0.47, 0.5, 1.0) for c in cache["cols"]])
        return cache["ids"], rgba
    return fn


def run_take(stunt_dir, take, render=True, fps=30):
    from .. import biomech
    from ..engine import metrics as M, params as P, render as R, scene, sim
    stunt_dir = os.path.abspath(stunt_dir)
    S = rj(os.path.join(stunt_dir, "stunt.json"))
    W = world_info(S["world_dir"])
    take_dir = os.path.join(stunt_dir, take)
    os.makedirs(take_dir, exist_ok=True)
    stunt = take_choreography(stunt_dir, take)
    plan = engine_plan(W, stunt, S.get("brief", ""), S.get("duration_sec", 6.0))
    nplan = scene.normalize_plan(plan)
    pp = os.path.join(take_dir, "params.json")
    params = rj(pp) if os.path.isfile(pp) else {}
    params = {**P.clamp(params, nplan), "id": take, "camera": "match_follow"}
    wj(pp, params)
    wj(os.path.join(take_dir, "plan.json"), plan)
    t0 = time.time()
    events = sim.run(plan, params, take_dir, log=say)
    say(f"{take}: sim done in {time.time() - t0:.1f} s")
    metrics = M.compute(plan, params, take_dir)
    bio = biomech.compute(take_dir)
    metrics["biomech"] = {k: bio["peaks"][k] for k in ("head_g", "hic15", "knee_torque", "hip_torque", "ankle_torque",
                                                       "grf", "neck_compression", "lumbar_compression")}
    metrics["biomech"]["worst_link"] = bio["worst_link"]
    metrics["biomech"]["flags"] = bio["flags"]
    wj(os.path.join(take_dir, "metrics.json"), metrics)
    say(f"{take}: metrics + biomech done ({time.time() - t0:.1f} s)")
    out = {"take": take, "dir": take_dir, "metrics": os.path.join(take_dir, "metrics.json"),
           "biomech": os.path.join(take_dir, "biomech.json"), "events": events.get("t_finish")}
    if render:
        size = frame_size(W)
        inputs = {"scene_xml": os.path.join(take_dir, "scene.xml"), "trajectory": os.path.join(take_dir, "trajectory.npz"),
                  "events": os.path.join(take_dir, "events.json"), "plan": nplan, "params": params, "metrics": metrics,
                  "geom_rgba": _tint(bio)}
        rep = R.render_video(inputs, os.path.join(take_dir, "video.mp4"), float(S.get("duration_sec", 6.0)), fps=fps,
                             size=size, camera="match_follow", hud=True, title=f"FALLGUY {take}")
        wj(os.path.join(take_dir, "render.json"), rep)
        say(f"{take}: video {rep['size']} {rep['duration']} s in {rep['seconds']} s")
        out["video"] = rep["path"]
        # a few stills for quick looks
        try:
            _stills(take_dir, rep, float(S.get("duration_sec", 6.0)))
        except Exception as e:                                   # noqa: BLE001
            say(f"stills skipped: {e}")
    return out


def _stills(take_dir, rep, duration, n=4):
    import imageio
    from PIL import Image
    rd = imageio.get_reader(rep["path"])
    tot = rep["frames"]
    picks = [int(tot * f) for f in (0.15, 0.4, 0.65, 0.9)][:n]
    ims = []
    for i, fr in enumerate(rd):
        if i in picks:
            ims.append(Image.fromarray(fr))
        if i > max(picks):
            break
    rd.close()
    if ims:
        w, h = ims[0].size
        sheet = Image.new("RGB", (w * len(ims) // 2, h // 2))
        for k, im in enumerate(ims):
            sheet.paste(im.resize((w // 2, h // 2)), (k * w // 2, 0))
        sheet.save(os.path.join(take_dir, "sheet.jpg"), quality=85)
