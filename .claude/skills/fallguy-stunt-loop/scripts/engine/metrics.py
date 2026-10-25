"""Gate-1 metrics of one simulated candidate (reads out_dir/{scene.xml, trajectory.npz, events.json}).

compute(plan, params, out_dir) measures everything on the MuJoCo run, lets gates.evaluate_gate1 fill
gate1.passed / failures and the score, writes out_dir/metrics.json and returns the dict.
- head G: head accelerometer |a| / 9.81, 15 ms moving average, from take-off on; HIC15 from the raw signal
- hazard collisions: performer-hazard contacts deeper than 0.5 cm, one entry per (body, object) pair, plus any
  hazard the kinematic run-up passes through
- clearance: smallest mj_geomDistance between performer collision geoms and hazard geoms, take-off to landing
- landing: first touch on mat / floor / non-hazard obstacle; point = centre of mass at that instant; target =
  landing-mat centre after mat_offset; inside_mat = touched the mat with the point over the mat
- framing: render.framing_report for the candidate's camera (the same camera the video uses); falls back to
  the scene's static cameras if the renderer cannot be imported
- rotation: pelvis rotation integrated in the travel frame between take-off and landing
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

from . import params as P
from . import performer as perf
from . import scene

PENETRATION = 0.005        # m: a hazard contact deeper than this is a collision
DIST_MAX = 2.0             # m: clearances beyond this are reported as 200 cm
NEAR_PATH = 0.6            # m: a hazard this close (horizontally) to the flight path is "under" it
RUNUP_EVERY = 10           # run-up obstruction check every 10th step (20 ms)
FRAME_SIZE = (1280, 720)


def _gates():
    try:
        import gates
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import gates
    return gates


def _hic15(g, dt):
    """HIC over windows up to 15 ms of a G trace sampled every dt."""
    if len(g) < 2:
        return 0.0
    cs = np.concatenate([[0.0], np.cumsum(g) * dt])
    best = 0.0
    for w in range(1, int(round(0.015 / dt)) + 1):
        if w >= len(cs):
            break
        avg = (cs[w:] - cs[:-w]) / (w * dt)
        best = max(best, float(np.max(w * dt * np.maximum(avg, 0.0) ** 2.5)))
    return best


def _rotvec(q1, q0):
    """World-frame rotation vector taking orientation q0 to q1."""
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, q1, np.array([q0[0], -q0[1], -q0[2], -q0[3]]))
    if dq[0] < 0:
        dq = -dq
    s = float(np.linalg.norm(dq[1:]))
    return np.zeros(3) if s < 1e-12 else 2.0 * math.atan2(s, dq[0]) * dq[1:] / s


class _Clearance:
    """Smallest signed distance between performer collision geoms and a set of static geoms, with a
    bounding-sphere prefilter so only nearby pairs reach mj_geomDistance."""

    def __init__(self, model, data, perf_geoms, other_geoms):
        self.m, self.d = model, data
        self.a = np.asarray(perf_geoms, int)
        self.b = np.asarray(other_geoms, int)
        mujoco.mj_kinematics(model, data)
        self.bpos = data.geom_xpos[self.b].copy()
        self.brad = model.geom_rbound[self.b].copy()
        self.arad = model.geom_rbound[self.a].copy()
        self.ft = np.zeros(6)

    def step(self, best):
        """(distance, perf geom, other geom) for the current kinematics, or None if nothing is closer than best."""
        if not len(self.b):
            return None
        apos = self.d.geom_xpos[self.a]
        lb = np.linalg.norm(apos[:, None, :] - self.bpos[None, :, :], axis=2) - self.arad[:, None] - self.brad[None, :]
        ia, ib = np.nonzero(lb < min(best, DIST_MAX))
        out = None
        for i, j in zip(ia, ib):
            dist = mujoco.mj_geomDistance(self.m, self.d, int(self.a[i]), int(self.b[j]), DIST_MAX, self.ft)
            if dist < best:
                best, out = dist, (dist, int(self.a[i]), int(self.b[j]))
        return out


def _framing(out_dir, plan, params, events, model, data, pf, traj_t, traj_q):
    """(framing_pct, source). Uses the renderer's own camera logic so the number matches the video."""
    cam = params.get("camera", "hero")
    try:
        from . import render
        rep = render.framing_report({"scene_xml": os.path.join(out_dir, "scene.xml"),
                                     "trajectory": os.path.join(out_dir, "trajectory.npz"),
                                     "events": events, "plan": plan, "params": params, "metrics": None},
                                    camera=cam, size=FRAME_SIZE, fps=30)
        return float(rep["framing_pct"]), f"render.framing_report ({cam})"
    except Exception as e:                       # renderer missing or failing: static scene camera fallback
        sys.stderr.write(f"warning: render.framing_report unavailable ({type(e).__name__}: {e}); "
                         f"framing from the scene's static camera\n")
    name = {"hero": "three_quarter"}.get(cam, cam)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        return 0.0, f"no camera {name!r}"
    W, H = FRAME_SIZE
    t0, t1 = events["t_runup"], min(events["t_landing"] + 1.0, events["t_end"])
    times = np.arange(t0, t1 + 1e-9, 1 / 30)
    geoms = pf.visual_geoms
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    inside = 0
    for ts in times:
        data.qpos[:] = traj_q[min(int(round(ts / 0.002)), len(traj_q) - 1)]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        R = data.geom_xmat[geoms].reshape(-1, 3, 3)
        c = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", R, model.geom_aabb[geoms, :3])
        h = np.einsum("nij,nj->ni", np.abs(R), model.geom_aabb[geoms, 3:])
        pts = (c[:, None, :] + h[:, None, :] * signs[None]).reshape(-1, 3)
        Rc = data.cam_xmat[cid].reshape(3, 3)
        v = (pts - data.cam_xpos[cid]) @ Rc
        depth = -v[:, 2]
        if np.any(depth <= 1e-6):
            continue
        f = H / (2 * math.tan(math.radians(model.cam_fovy[cid]) / 2))
        u, w = W / 2 + f * v[:, 0] / depth, H / 2 - f * v[:, 1] / depth
        inside += bool(u.min() >= 0 and u.max() <= W and w.min() >= 0 and w.max() <= H)
    return round(100.0 * inside / max(len(times), 1), 1), f"static scene camera {name!r}"


def compute(plan, params, out_dir):
    """Gate-1 metrics for the sim in out_dir; writes out_dir/metrics.json and returns the dict."""
    plan = scene.normalize_plan(plan)
    params = P.clamp(params, plan)
    with open(os.path.join(out_dir, "events.json"), encoding="utf-8") as f:
        ev = json.load(f)
    z = np.load(os.path.join(out_dir, "trajectory.npz"))
    t, q = z["t"], z["qpos"]
    dt = float(ev.get("dt", 0.002))
    model = mujoco.MjModel.from_xml_path(os.path.join(out_dir, "scene.xml"))
    data = mujoco.MjData(model)
    pf = perf.Performer(model)
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"
    obj = lambda g: ev["geom_objects"].get(str(int(g)), gname(g).split("__")[0])
    body = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(g)]))
    hazard = np.array(ev["hazard_geoms"], int)
    k0 = int(round(ev["t_takeoff"] / dt))
    k1 = min(int(round(ev["t_landing"] / dt)), len(t) - 1)

    # ---- head G (15 ms average) and HIC15, from take-off on
    g15 = z["head_g15"]
    i_pk = k0 + int(np.argmax(g15[k0:]))
    head_peak, head_t = float(g15[i_pk]), float(t[i_pk])
    hic = _hic15(z["head_g"][k0:], dt)

    # ---- hazard collisions (dynamic phase, from the contact log)
    contacts = {}
    is_haz = np.zeros(model.ngeom, bool)
    is_haz[hazard] = True
    ck, ca, cb, cd = z["con_k"], z["con_g1"], z["con_g2"], z["con_dist"]
    sel = np.nonzero(is_haz[cb] & (cd < -PENETRATION))[0] if len(cb) else []
    for i in sel:
        key = (body(ca[i]), obj(cb[i]))
        c = contacts.get(key)
        depth = -float(cd[i]) * 100.0
        if c is None:
            contacts[key] = {"t": round(float(t[ck[i]]), 3), "body": key[0], "object": key[1], "depth_cm": round(depth, 2),
                             "phase": "flight" if ck[i] <= k1 else "landing"}
        else:
            c["depth_cm"] = round(max(c["depth_cm"], depth), 2)
    # ---- run-up: the kinematic performer must not pass through a hazard
    clr = _Clearance(model, data, pf.collision_geoms, hazard)
    run_min = None
    for k in range(int(round(ev["t_runup"] / dt)), k0, RUNUP_EVERY):
        data.qpos[:] = q[k]
        mujoco.mj_kinematics(model, data)
        hit = clr.step(-PENETRATION)
        if hit is not None:
            key = (body(hit[1]), obj(hit[2]))
            if key not in contacts:
                contacts[key] = {"t": round(float(t[k]), 3), "body": key[0], "object": key[1],
                                 "depth_cm": round(-hit[0] * 100.0, 2), "phase": "runup"}
            run_min = hit[0] if run_min is None else min(run_min, hit[0])
    hazard_contacts = sorted(contacts.values(), key=lambda c: c["t"])

    # ---- minimum clearance, take-off to landing
    best, where = DIST_MAX, None
    com = z["com"]
    for k in range(k0, k1 + 1):
        data.qpos[:] = q[k]
        mujoco.mj_kinematics(model, data)
        hit = clr.step(best)
        if hit is not None:
            best, where = hit[0], (float(t[k]), hit[1], hit[2])
    # is there a hazard under the flight path (something to clear, so over-jumping costs score)?
    over_target = False
    if len(hazard):
        R = data.geom_xmat[hazard].reshape(-1, 3, 3)
        hc = data.geom_xpos[hazard] + np.einsum("nij,nj->ni", R, model.geom_aabb[hazard, :3])
        hh = np.einsum("nij,nj->ni", np.abs(R), model.geom_aabb[hazard, 3:])
        for k in range(k0, k1 + 1, 5):
            if not np.isfinite(com[k, 0]):
                continue
            gap = np.maximum(np.abs(com[k, :2] - hc[:, :2]) - hh[:, :2], 0.0)
            near = (np.hypot(gap[:, 0], gap[:, 1]) < NEAR_PATH) & (hc[:, 2] + hh[:, 2] < com[k, 2]) & (hc[:, 2] + hh[:, 2] > 0.15)
            if near.any():
                over_target = True
                break
    min_clear_cm = round(best * 100.0, 2)
    at = {"t": round(where[0], 3), "body": body(where[1]), "object": obj(where[2])} if where else None

    # ---- landing
    mat = ev.get("landing_mat")
    st = plan["stunt"]
    target = [mat["pos"][0], mat["pos"][1]] if mat else [st["landing_x"], st["lane_y"]]
    surface = ev.get("landing_surface", "none") if ev.get("landed") else "none"
    point = [float(com[k1, 0]), float(com[k1, 1])] if np.isfinite(com[k1, 0]) else [float(q[k1, 0]), float(q[k1, 1])]
    err = math.hypot(point[0] - target[0], point[1] - target[1])
    inside = bool(surface == "crash_mat" and mat is not None and
                  scene.inside_footprint(point, mat["pos"], mat["size"], mat["yaw"]))
    landing = {"t": round(float(ev["t_landing"]), 3), "surface": surface, "point": [round(v, 3) for v in point],
               "target": [round(v, 3) for v in target], "error_m": round(err, 3), "inside_mat": inside}

    # ---- rotation between take-off and landing, travel frame = world axes
    acc = np.zeros(3)
    fq = pf.free_qadr
    for k in range(k0, k1):
        acc += _rotvec(q[k + 1, fq + 3:fq + 7], q[k, fq + 3:fq + 7])
    rot = {"roll": round(math.degrees(acc[0]), 1), "pitch": round(math.degrees(acc[1]), 1), "yaw": round(math.degrees(acc[2]), 1)}

    # ---- flight summary
    fl = slice(k0, k1 + 1)
    zc = com[fl, 2]
    i_ap = k0 + int(np.nanargmax(zc)) if np.isfinite(zc).any() else k0
    v0 = z["com_vel"][k0] if "com_vel" in z.files else np.array(ev["takeoff"]["v_com"])
    sim = {"t_takeoff": round(float(ev["t_takeoff"]), 3), "t_landing": round(float(ev["t_landing"]), 3),
           "flight_time": round(float(ev["t_landing"]) - float(ev["t_takeoff"]), 3),
           "apex_m": round(float(com[i_ap, 2]), 3), "takeoff_velocity": [round(float(v0[0]), 3), round(float(v0[2]), 3)],
           "t_apex": round(float(t[i_ap]), 3), "apex_pelvis_m": round(float(np.max(q[fl, fq + 2])), 3),
           "takeoff_x": round(float(ev["choreography"]["takeoff_x_used"]), 3),
           "flight_time_est": ev["takeoff"].get("flight_time_est"),
           "landed": bool(ev.get("landed")), "warnings": list(ev["choreography"].get("warnings") or [])}

    framing, fsrc = _framing(out_dir, plan, params, ev, model, data, pf, t, q)
    gate1 = {"head_g_peak": round(head_peak, 2), "head_g_peak_t": round(head_t, 3), "hic15": round(hic, 1),
             "hazard_collisions": len(hazard_contacts), "hazard_contacts": hazard_contacts,
             "min_clearance_cm": min_clear_cm, "min_clearance_at": at, "clearance_over_target": over_target,
             "landing": landing, "framing_pct": round(framing, 1), "framing_source": fsrc,
             "rotation_deg": rot}
    metrics = {"candidate_id": params.get("id"), "ok": True, "error": None, "sim": sim, "gate1": gate1}
    metrics = _gates().evaluate_gate1(metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    return metrics
