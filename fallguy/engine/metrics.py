"""Gate-1 metrics of one simulated candidate (reads out_dir/{scene.xml, trajectory.npz, events.json}).

compute(plan, params, out_dir) measures everything on the MuJoCo run, lets gates.evaluate_gate1 fill
gate1.passed / failures and the score, writes out_dir/metrics.json and returns the dict.
- head G: |a - g| / 9.81 of the head site (finite differences, 15 ms moving average) over the whole stunt;
  HIC15 from the unaveraged signal
- lateral g: peak horizontal pelvis acceleration perpendicular to the direction of travel during move phases
- hazard collisions: kinematic phases geometrically (performer collision geoms penetrating a hazard by more than
  0.5 cm, mj_geomDistance); free phases from the contact log; one entry per (body, object) pair
- clearance: smallest mj_geomDistance between performer collision geoms and hazard geoms over the stunt
- landing against stunt.target: mat -> first support contact after take-off (centre of mass at that instant);
  zone -> pelvis at t_finish; none -> the end point with error 0
- framing: render.framing_report for the candidate's camera (the camera the video uses)
- rotation: pelvis rotation in the start-heading frame, take-off to landing (or start to t_finish on the ground)
"""
import json
import math
import os
import sys

import mujoco
import numpy as np

from . import params as P
from . import scene

PENETRATION = 0.005        # m: a hazard penetration deeper than this is a collision
DIST_MAX = 2.0             # m: clearances beyond this are reported as 200 cm
NEAR_PATH = 0.6            # m: a hazard this close (horizontally) to the flight path is "under" it
KIN_EVERY, FREE_EVERY = 5, 2   # clearance sampling (steps) in kinematic / free phases
G = 9.81


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


def _smooth(x, sigma):
    if sigma < 0.5 or len(x) < 3:
        return x
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


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

    def pairs(self, limit):
        """[(distance, perf geom, other geom)] for pairs closer than limit (current kinematics)."""
        if not len(self.b):
            return []
        apos = self.d.geom_xpos[self.a]
        lb = np.linalg.norm(apos[:, None, :] - self.bpos[None, :, :], axis=2) - self.arad[:, None] - self.brad[None, :]
        ia, ib = np.nonzero(lb < min(limit, DIST_MAX))
        out = []
        for i, j in zip(ia, ib):
            dist = mujoco.mj_geomDistance(self.m, self.d, int(self.a[i]), int(self.b[j]), DIST_MAX, self.ft)
            if dist < limit:
                out.append((dist, int(self.a[i]), int(self.b[j])))
        return out


def frame_size(plan):
    src = plan.get("source") or {}
    w, h = src.get("width"), src.get("height")
    try:
        return (720, 1280) if float(h) > float(w) else (1280, 720)
    except (TypeError, ValueError):
        return (1280, 720)


def _framing(out_dir, plan, params, events):
    cam = params.get("camera") or P.default_camera(plan)
    try:
        from . import render
        rep = render.framing_report({"scene_xml": os.path.join(out_dir, "scene.xml"),
                                     "trajectory": os.path.join(out_dir, "trajectory.npz"),
                                     "events": events, "plan": plan, "params": params, "metrics": None},
                                    camera=cam, size=frame_size(plan), fps=15)
        return float(rep["framing_pct"]), f"render.framing_report ({cam})"
    except Exception as e:                       # noqa: BLE001 - framing must not sink the whole metrics
        sys.stderr.write(f"warning: framing_report failed ({type(e).__name__}: {e})\n")
        return 0.0, f"framing failed: {type(e).__name__}: {e}"


def compute(plan, params, out_dir):
    """Gate-1 metrics for the sim in out_dir; writes out_dir/metrics.json and returns the dict."""
    plan = scene.normalize_plan(plan)
    params = P.clamp(params, plan)
    stunt = scene.normalize_stunt(P.apply(plan, params))
    with open(os.path.join(out_dir, "events.json"), encoding="utf-8") as f:
        ev = json.load(f)
    z = np.load(os.path.join(out_dir, "trajectory.npz"))
    t, q = z["t"], z["qpos"]
    dt = float(ev.get("dt", 0.002))
    n = len(t)
    model = mujoco.MjModel.from_xml_path(os.path.join(out_dir, "scene.xml"))
    data = mujoco.MjData(model)
    from . import performer as perf
    pf = perf.Performer(model)
    gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{int(g)}"
    obj = lambda g: ev["geom_objects"].get(str(int(g)), gname(g).split("__")[0])
    body = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(g)]))
    hazard = np.array(ev["hazard_geoms"], int)
    airborne = ev.get("t_takeoff") is not None
    k_rel = int(round(ev["t_takeoff"] / dt)) if airborne else n
    k_kin_end = int(round(ev.get("t_kinematic_end", t[-1]) / dt))
    t_fin = float(ev["t_finish"])
    k_fin = min(int(round(t_fin / dt)), n - 1)
    k_land = min(int(round(ev["t_landing"] / dt)), n - 1) if ev.get("t_landing") is not None else None
    fq = pf.free_qadr

    # ---- head G (15 ms average) and HIC15, whole stunt
    g15 = z["head_g15"]
    i_pk = int(np.argmax(g15))
    head_peak, head_t = float(g15[i_pk]), float(t[i_pk])
    hic = _hic15(z["head_g"], dt)

    # ---- lateral g during move phases (pelvis xy)
    xy = _smooth(q[:, fq:fq + 2].astype(float), 0.03 / dt)
    vel = np.gradient(xy, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    spd = np.linalg.norm(vel, axis=1)
    lat = np.abs(vel[:, 0] * acc[:, 1] - vel[:, 1] * acc[:, 0]) / np.maximum(spd, 1e-9) / G
    in_move = np.zeros(n, bool)
    for ph in ev.get("phases") or []:
        if ph["kind"] == "move" and ph.get("t0") is not None:
            in_move[int(round(ph["t0"] / dt)):int(round(ph["t1"] / dt)) + 1] = True
    lat_sel = in_move & (spd > 0.5)
    lat_peak = float(lat[lat_sel].max()) if lat_sel.any() else 0.0
    lat_t = float(t[np.nonzero(lat_sel)[0][np.argmax(lat[lat_sel])]]) if lat_sel.any() else None

    # ---- hazard contacts / clearance
    contacts = {}

    def note(k, pg, og, depth_m, phase):
        key = (body(pg), obj(og))
        c = contacts.get(key)
        if c is None:
            contacts[key] = {"t": round(float(t[k]), 3), "body": key[0], "object": key[1],
                             "depth_cm": round(depth_m * 100.0, 2), "phase": phase}
        else:
            c["depth_cm"] = round(max(c["depth_cm"], depth_m * 100.0), 2)

    is_haz = np.zeros(model.ngeom, bool)
    is_haz[hazard] = True
    ck, ca, cb, cd = z["con_k"], z["con_g1"], z["con_g2"], z["con_dist"]
    for i in (np.nonzero(is_haz[cb] & (cd < -PENETRATION))[0] if len(cb) else []):
        note(int(ck[i]), ca[i], cb[i], -float(cd[i]), "free")
    clr = _Clearance(model, data, pf.collision_geoms, hazard)
    best, where = DIST_MAX, None
    steps = list(range(0, min(k_kin_end, k_fin) + 1, KIN_EVERY))
    if airborne:
        steps += list(range(k_rel, k_fin + 1, FREE_EVERY))
    for k in steps:
        data.qpos[:] = q[k]
        mujoco.mj_kinematics(model, data)
        for dist, a, b in clr.pairs(max(best, 0.0) + 1e-9 if best > 0 else 1e-9):
            if dist < best:
                best, where = dist, (float(t[k]), a, b)
            if k < k_rel and dist < -PENETRATION:
                note(k, a, b, -dist, "kinematic")
    hazard_contacts = sorted(contacts.values(), key=lambda c: c["t"])
    min_clear_cm = round(best * 100.0, 2)
    at = {"t": round(where[0], 3), "body": body(where[1]), "object": obj(where[2])} if where else None
    com = z["com"]
    over_target = False                          # a hazard under the flight path: over-jumping costs score
    if airborne and len(hazard):
        R = data.geom_xmat[hazard].reshape(-1, 3, 3)
        hc = data.geom_xpos[hazard] + np.einsum("nij,nj->ni", R, model.geom_aabb[hazard, :3])
        hh = np.einsum("nij,nj->ni", np.abs(R), model.geom_aabb[hazard, 3:])
        for k in range(k_rel, (k_land or k_fin) + 1, 5):
            gap = np.maximum(np.abs(com[k, :2] - hc[:, :2]) - hh[:, :2], 0.0)
            top = hc[:, 2] + hh[:, 2]
            if ((np.hypot(gap[:, 0], gap[:, 1]) < NEAR_PATH) & (top < com[k, 2]) & (top > 0.15)).any():
                over_target = True
                break

    # ---- landing against stunt.target
    tgt = stunt["target"]
    kind = tgt.get("kind", "none")
    if kind == "mat":
        mat = scene.target_mat(plan, stunt)
        target = [mat["pos"][0], mat["pos"][1]] if mat else None
        if airborne and ev.get("landed") and k_land is not None:
            point = [float(com[k_land, 0]), float(com[k_land, 1])]
            surface, t_l = ev.get("landing_surface", "none"), float(ev["t_landing"])
        else:
            point = [float(q[k_fin, fq]), float(q[k_fin, fq + 1])]
            surface, t_l = ("none" if airborne else "floor"), t_fin
        err = math.hypot(point[0] - target[0], point[1] - target[1]) if target else None
        inside = bool(mat is not None and surface == "crash_mat" and
                      scene.inside_footprint(point, mat["pos"], mat["size"], mat["yaw"]))
    else:
        point = [float(q[k_fin, fq]), float(q[k_fin, fq + 1])]
        t_l = float(ev["t_landing"]) if (airborne and ev.get("t_landing") is not None) else t_fin
        surface = ev.get("landing_surface", "none") if airborne else "floor"
        if kind == "zone":
            target = list(tgt["pos"])
            err = math.hypot(point[0] - target[0], point[1] - target[1])
            inside = err <= float(tgt.get("radius", 0.65))
        else:
            target, err, inside = list(point), 0.0, True
    landing = {"t": round(t_l, 3), "surface": surface, "point": [round(v, 3) for v in point],
               "target": None if target is None else [round(v, 3) for v in target],
               "error_m": None if err is None else round(err, 3), "inside_mat": bool(inside), "target_kind": kind}

    # ---- rotation in the start-heading frame: integrated through the flight (counts full turns), or the net
    #      change start -> finish for a stunt that stays on the ground (yaw unwrapped along the way)
    h0 = math.radians(stunt["start"]["heading_deg"])
    c, s_ = math.cos(h0), math.sin(h0)
    to_local = np.array([[c, s_, 0.0], [-s_, c, 0.0], [0.0, 0.0, 1.0]])
    if airborne:
        k0, k1 = k_rel, (k_land if k_land is not None else k_fin)
        acc_r = np.zeros(3)
        for k in range(k0, max(k1, k0)):
            acc_r += _rotvec(q[k + 1, fq + 3:fq + 7], q[k, fq + 3:fq + 7])
        loc = np.degrees(to_local @ acc_r)
    else:
        def mat(k):
            m_ = np.zeros(9)
            mujoco.mju_quat2Mat(m_, q[k, fq + 3:fq + 7])
            return m_.reshape(3, 3)
        heads = [math.atan2(mat(k)[1, 0], mat(k)[0, 0]) for k in range(0, k_fin + 1, 10)] + \
                [math.atan2(mat(k_fin)[1, 0], mat(k_fin)[0, 0])]
        yaw = float(np.degrees(np.unwrap(heads)[-1] - np.unwrap(heads)[0]))
        Rl = mat(0).T @ mat(k_fin)                       # relative rotation in the start body frame
        roll = math.degrees(math.atan2(Rl[2, 1], Rl[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -Rl[2, 0]))))
        loc = np.array([roll, pitch, yaw])
    rot = {"roll": round(float(loc[0]), 1), "pitch": round(float(loc[1]), 1), "yaw": round(float(loc[2]), 1)}

    # ---- sim summary
    sim = {"t_takeoff": ev.get("t_takeoff"), "t_landing": ev.get("t_landing"), "t_finish": round(t_fin, 3),
           "airborne": airborne, "apex_m": None, "peak_speed": round(float(spd[:k_fin + 1].max()), 3),
           "landed": bool(ev.get("landed")) if airborne else None,
           "warnings": list((ev.get("choreography") or {}).get("warnings") or [])}
    if airborne:
        k1f = k_land if k_land is not None else n - 1
        zc = com[k_rel:k1f + 1, 2]
        i_ap = k_rel + int(np.argmax(zc))
        sim.update(apex_m=round(float(com[i_ap, 2]), 3), t_apex=round(float(t[i_ap]), 3),
                   flight_time=round(float(t[k1f] - t[k_rel]), 3),
                   flight_time_est=(ev.get("takeoff") or {}).get("flight_time_est"),
                   takeoff_velocity=(ev.get("takeoff") or {}).get("v_com"))

    framing, fsrc = _framing(out_dir, plan, params, ev)
    gate1 = {"head_g_peak": round(head_peak, 2), "head_g_peak_t": round(head_t, 3), "hic15": round(hic, 1),
             "lateral_g_peak": round(lat_peak, 3), "lateral_g_peak_t": None if lat_t is None else round(lat_t, 3),
             "hazard_collisions": len(hazard_contacts), "hazard_contacts": hazard_contacts,
             "min_clearance_cm": min_clear_cm, "min_clearance_at": at, "clearance_over_target": over_target,
             "landing": landing, "framing_pct": round(framing, 1), "framing_source": fsrc, "rotation_deg": rot}
    metrics = {"candidate_id": params.get("id"), "ok": True, "error": None, "sim": sim, "gate1": gate1}
    metrics = _gates().evaluate_gate1(metrics)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    return metrics
