"""MuJoCo simulation of one candidate (contract physics).

- 0 .. t_takeoff: kinematic run-up from the choreography. The root and joints are written on the 2 ms grid
  (exact foot placement, no sinking or skating), the mocap weld holds the pelvis, performer collisions are off.
- t_takeoff: the weld is released (data.eq_active), performer collisions turn on, and the free root gets the
  take-off state: centre-of-mass velocity [approach_speed, 0, takeoff_vz] and the angular momentum the
  choreography sized for the planned rotation.
- then free MuJoCo dynamics (2 ms step) with PD limb servos: flight, contact, landing, and at least 3 s more.

run() writes out_dir/scene.xml (the exact MJCF, absolute meshdir), trajectory.npz (t, qpos, head_g15 and the
extra arrays metrics.py reads) and events.json, and returns the events dict.
"""
import json
import math
import os
import sys
import time

import mujoco
import numpy as np

from . import choreo as C
from . import params as P
from . import performer as perf
from . import scene

DT = C.DT
POST_LANDING = 3.0        # s simulated after the first touch
MAX_AIR = 6.0             # s after take-off without a landing before the run stops
G15_SAMPLES = 8           # 15 ms moving average at 2 ms (8 samples = 16 ms window)
CATS = {"performer": -1, "none": 0, "floor": 1, "mat": 2, "hazard": 3, "obstacle": 4, "launch": 5}
SUPPORT = (1, 2, 4, 5)    # categories that count as a landing surface


def make_choreography(choreography, plan, params):
    """Accepts None (library by stunt type), a Choreo instance, a Choreo class or a callable(plan, params)."""
    if choreography is None:
        return C.get(plan["stunt"]["type"])(plan, params)
    if isinstance(choreography, C.Choreo):
        return choreography
    ch = choreography(plan, params)
    if not isinstance(ch, C.Choreo):
        raise TypeError(f"choreography(plan, params) must return an engine.choreo.Choreo, got {type(ch).__name__}")
    return ch


def build(plan, params, choreography=None):
    """Normalized plan, clamped params, choreography, compiled model + data, performer handles, scene meta, xml."""
    plan = scene.normalize_plan(plan)
    params = P.clamp(params, plan)
    ch = make_choreography(choreography, plan, params)
    pos0, quat0, _ = ch.kinematic(0)
    xml, meta = scene.build_xml(plan, params, pos0, quat0)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:
        raise RuntimeError(f"MuJoCo could not compile the stunt scene: {e}") from e
    data = mujoco.MjData(model)
    pf = perf.Performer(model)
    return plan, params, ch, model, data, pf, meta, xml


def _geom_ids(model, names):
    out = []
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        if i < 0:
            raise RuntimeError(f"scene geom {n!r} missing from the compiled model")
        out.append(i)
    return out


def _set_pose(data, pf, pos, quat, qj):
    data.qpos[pf.free_qadr:pf.free_qadr + 3] = pos
    data.qpos[pf.free_qadr + 3:pf.free_qadr + 7] = quat
    data.qpos[pf.qadr] = qj


def _takeoff_velocity(model, data, pf, v_com, L):
    """Set the free-joint velocity so the whole body's com velocity is v_com and its angular momentum about
    the com is L (world frame), keeping the joint velocities already in data.qvel."""
    fv = pf.free_vadr
    data.qvel[fv:fv + 6] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    L0 = data.subtree_angmom[pf.root].copy()
    _, I = pf.com_inertia(model, data)
    w = np.linalg.solve(I, np.asarray(L, float) - L0)
    R = data.xmat[pf.root].reshape(3, 3)
    data.qvel[fv + 3:fv + 6] = R.T @ w                      # free-joint angular velocity is in the body frame
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    data.qvel[fv:fv + 3] += np.asarray(v_com, float) - data.subtree_linvel[pf.root]
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    return data.subtree_linvel[pf.root].copy(), data.subtree_angmom[pf.root].copy(), w


def run(plan, params, out_dir, choreography=None, log=None):
    """Simulate one candidate; writes scene.xml, trajectory.npz, events.json into out_dir; returns events."""
    wall = time.perf_counter()
    log = log or (lambda msg: sys.stderr.write(msg + "\n"))
    os.makedirs(out_dir, exist_ok=True)
    plan, params, ch, model, data, pf, meta, xml = build(plan, params, choreography)
    with open(os.path.join(out_dir, "scene.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    for w_ in ch.warnings:
        log(f"warning: {w_}")

    cats = meta["categories"]
    cat = np.zeros(model.ngeom, np.int8)
    ids = {k: _geom_ids(model, v) for k, v in cats.items()}
    for k, v in ids.items():
        cat[v] = CATS[k]
    is_perf = np.zeros(model.ngeom, bool)
    is_perf[pf.collision_geoms] = True
    cat[pf.geoms] = CATS["performer"]
    sadr = lambda name: int(model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)])
    head_adr, com_adr, comv_adr, angm_adr = pf.head_adr, sadr("com"), sadr("com_vel"), sadr("angmom")

    K = ch.K
    n_max = K + int(round((MAX_AIR + POST_LANDING + 0.5) / DT)) + 2
    qlog = np.zeros((n_max, model.nq))
    head = np.zeros(n_max)
    com = np.full((n_max, 3), np.nan)
    comv = np.full((n_max, 3), np.nan)
    angm = np.full((n_max, 3), np.nan)
    con_k, con_a, con_b, con_d = [], [], [], []

    # ---------------------------------------------------------------- kinematic run-up (collisions off, weld on)
    pf.set_collisions(model, False)
    for k in range(K):
        _set_pose(data, pf, *ch.kinematic(k))
        qlog[k] = data.qpos

    # ---------------------------------------------------------------- release
    pos, quat, qj = ch.kinematic(K)
    _set_pose(data, pf, pos, quat, qj)
    data.mocap_pos[0], data.mocap_quat[0] = pos, quat
    data.qvel[:] = 0.0
    data.qvel[pf.vadr] = ch.release_joint_velocity()
    data.time = ch.t_takeoff
    ch.prepare(model, data, pf)
    _set_pose(data, pf, pos, quat, qj)
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "root_weld")
    data.eq_active[weld] = 0
    pf.set_collisions(model, True)
    mujoco.mj_forward(model, data)
    pen = min([float(d) for d, gg in zip(data.contact.dist, data.contact.geom) if is_perf[gg[0]] ^ is_perf[gg[1]]] + [0.0])
    data.qpos[pf.free_qadr + 2] += max(0.0, -pen) + 0.001        # start just clear of the ground
    v_com, L, w = _takeoff_velocity(model, data, pf, ch.v_com_target, ch.L_target)
    tgt = ch.flight_targets(0.0)
    data.ctrl[pf.act] = tgt
    takeoff = {"com": [round(float(v), 4) for v in data.subtree_com[pf.root]],
               "v_com": [round(float(v), 4) for v in v_com], "angmom": [round(float(v), 3) for v in L],
               "omega_deg_s": [round(float(math.degrees(v)), 1) for v in w], "flight_time_est": round(ch.T_f, 4),
               "landing_surface_z_est": round(ch.landing_surface_z, 3)}

    # ---------------------------------------------------------------- free dynamics
    gain_land = ch.landing_gain_scale()
    t_touch, surface, q_touch, airborne = None, None, None, False
    first_hazard = None
    k = K
    while k < n_max:
        t_prev = k * DT
        s = t_prev - ch.t_takeoff
        if t_touch is None:
            u = s / ch.T_f
            pf.set_leg_gain_scale(model, 1.0 + (gain_land - 1.0) * C.smooth((u - 0.8) / 0.2))
            tgt = ch.flight_targets(s)
        else:
            tgt = ch.landing_targets(t_prev - t_touch, q_touch)
        data.ctrl[pf.act] = tgt
        qlog[k] = data.qpos
        mujoco.mj_step(model, data)
        sd = data.sensordata
        head[k] = math.sqrt(float(sd[head_adr] ** 2 + sd[head_adr + 1] ** 2 + sd[head_adr + 2] ** 2)) / 9.81
        com[k] = sd[com_adr:com_adr + 3]
        comv[k] = sd[comv_adr:comv_adr + 3]
        angm[k] = sd[angm_adr:angm_adr + 3]
        nc = data.ncon
        touching = False
        if nc:
            gg = data.contact.geom
            pa, pb = is_perf[gg[:, 0]], is_perf[gg[:, 1]]
            m = pa ^ pb
            if m.any():
                touching = True
                pg = np.where(pa, gg[:, 0], gg[:, 1])[m]
                og = np.where(pa, gg[:, 1], gg[:, 0])[m]
                con_k.append(np.full(len(pg), k, np.int32))
                con_a.append(pg.astype(np.int32))
                con_b.append(og.astype(np.int32))
                con_d.append(data.contact.dist[m].astype(np.float64))
                if t_touch is None:
                    c = cat[og]
                    if airborne and np.isin(c, SUPPORT).any():
                        t_touch, surface = t_prev, {1: "floor", 2: "crash_mat", 4: "obstacle", 5: "obstacle"}[
                            int(c[np.isin(c, SUPPORT)][0])]
                    elif airborne and (c == CATS["hazard"]).any():
                        first_hazard = t_prev if first_hazard is None else first_hazard
                        if t_prev - first_hazard > 0.25:          # resting on a hazard: that was the landing
                            t_touch, surface = first_hazard, "obstacle"
                    elif not airborne and s > 0.3:                # never left the ground
                        t_touch, surface = ch.t_takeoff + DT, {1: "floor", 2: "crash_mat"}.get(int(c[0]), "obstacle")
                    if t_touch is not None:
                        q_touch = tgt.copy()
        if not touching:
            airborne = True
            if t_touch is None:
                first_hazard = None
        k += 1
        if t_touch is not None and k * DT >= t_touch + POST_LANDING:
            break
        if t_touch is None and s >= MAX_AIR:
            break
    N = k
    t = np.arange(N) * DT
    head = head[:N]
    g15 = np.zeros(N)
    cs = np.concatenate([[0.0], np.cumsum(head[K:])])
    idx = np.arange(N - K)
    lo = np.maximum(0, idx + 1 - G15_SAMPLES)
    g15[K:] = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)
    con = {k_: (np.concatenate(v) if v else np.zeros(0, d)) for k_, v, d in
           (("con_k", con_k, np.int32), ("con_g1", con_a, np.int32), ("con_g2", con_b, np.int32), ("con_dist", con_d, float))}
    np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=t, qpos=qlog[:N], head_g15=g15, head_g=head,
                        com=com[:N], com_vel=comv[:N], angmom=angm[:N], **con)

    t_end = float(t[-1])
    t_land = float(t_touch) if t_touch is not None else t_end
    fl = (t >= ch.t_takeoff) & (t <= t_land)
    i_apex = int(np.argmax(np.where(fl, com[:N, 2], -np.inf)))
    events = {
        "t_start": 0.0, "t_runup": round(ch.t_runup, 4), "t_takeoff": round(ch.t_takeoff, 4),
        "t_apex": round(float(t[i_apex]), 4), "t_landing": round(t_land, 4), "t_end": round(t_end, 4),
        "performer_root_body": perf.ROOT_BODY,
        "performer_geoms": [int(g) for g in pf.geoms],
        "hazard_geoms": [int(g) for g in ids.get("hazard", [])],
        "mat_geoms": [int(g) for g in ids.get("mat", [])],
        "floor_geoms": [int(g) for g in ids.get("floor", [])],
        "performer_collision_geoms": [int(g) for g in pf.collision_geoms],
        "obstacle_geoms": [int(g) for g in ids.get("obstacle", [])],
        "launch_geoms": [int(g) for g in ids.get("launch", [])],
        "landed": t_touch is not None, "landing_surface": surface or "none",
        "landing_mat": meta["landing_mat"], "takeoff": takeoff, "dt": DT,
        "stunt_type": ch.name, "choreography": ch.summary(),
        "geom_objects": {str(i): meta["objects"][mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)]
                         for i in range(model.ngeom) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) in meta["objects"]},
    }
    with open(os.path.join(out_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)
    log(f"sim: {ch.name}, take-off t={ch.t_takeoff:.2f}s, landing t={t_land:.2f}s on {surface or 'nothing'}, "
        f"{N} steps in {time.perf_counter() - wall:.1f} s wall")
    return events
