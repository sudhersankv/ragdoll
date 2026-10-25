"""MuJoCo simulation of one candidate (contract physics).

- hold / move / the gather of a jump: kinematic. The root and joints are written on the 2 ms grid (feet planted
  by leg IK), the mocap weld holds the pelvis, performer collisions are off (clearance and penetration are
  measured geometrically by engine.metrics).
- jump / fall: the weld is released (data.eq_active), performer collisions turn on, and the free root gets the
  launch state: centre-of-mass velocity and the angular momentum the choreography sized for the rotations.
- then free MuJoCo dynamics (2 ms) with PD limb servos: air pose schedule, contact, land, >= 3 s past landing.
- a stunt that never leaves the ground holds its last pose kinematically for 2 s after the last phase.

run(plan, params, out_dir, choreography=None, log=None) writes out_dir/scene.xml (exact MJCF, absolute meshdir),
trajectory.npz (t, qpos, head_g15 + extras metrics.py reads) and events.json, and returns the events dict.

choreography: None -> engine.params.apply(plan, params) interpreted by engine.choreo.Choreography;
a dict -> that stunt dict (summary / start / phases / target); an engine.choreo.Choreography instance or
subclass; a callable(plan, params) returning any of these; or a custom motion object with the attributes
documented on engine.choreo.Choreography (K, kinematic(k), released, t_takeoff, phase_events, t_finish_kin,
warnings, summary(), and when released: prepare(), v_com_target, L_target, T_f, landing_surface_z,
release_joint_velocity(), flight_targets(s), landing_targets(s, q), landing_gain_scale(), land_duration).
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
MAX_AIR = 6.0             # s after release without a landing before the run stops
G15_SAMPLES = 8           # 15 ms moving average at 2 ms (8 samples = 16 ms window)
CATS = {"performer": -1, "none": 0, "floor": 1, "mat": 2, "hazard": 3, "obstacle": 4, "launch": 5}
SUPPORT = (1, 2, 4, 5)    # categories that count as a landing surface
SURFACE = {1: "floor", 2: "crash_mat", 4: "obstacle", 5: "obstacle", 3: "obstacle"}


def _stunt_for(plan, params, stunt):
    stunt = dict(stunt)
    tgt = dict(stunt.get("target") or {"kind": "none"})
    tgt.setdefault("offset", [float(params.get("mat_offset_x", 0.0)), float(params.get("mat_offset_y", 0.0))])
    stunt["target"] = tgt
    return stunt


def make_choreography(choreography, plan, params, _depth=0):
    """Resolve the choreography argument of run()/build() (see the module docstring)."""
    if choreography is None:
        return C.Choreography(plan, P.apply(plan, params))
    if isinstance(choreography, dict):
        return C.Choreography(plan, _stunt_for(plan, params, choreography))
    if isinstance(choreography, C.Choreography):
        return choreography
    if isinstance(choreography, type) and issubclass(choreography, C.Choreography):
        return choreography(plan, P.apply(plan, params))
    if hasattr(choreography, "kinematic") and hasattr(choreography, "K"):
        return choreography
    if callable(choreography) and _depth == 0:
        return make_choreography(choreography(plan, params), plan, params, 1)
    raise TypeError(f"choreography must be None, a stunt dict, an engine.choreo.Choreography or a callable returning "
                    f"one; got {type(choreography).__name__}")


def build(plan, params, choreography=None):
    """(normalized plan, clamped params, choreography, model, data, performer handles, scene meta, xml)."""
    plan = scene.normalize_plan(plan)
    params = P.clamp(params, plan)
    ch = make_choreography(choreography, plan, params)
    stunt = getattr(ch, "stunt", None) or P.apply(plan, params)
    pos0, quat0, _ = ch.kinematic(0)
    xml, meta = scene.build_xml(plan, stunt, pos0, quat0)
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
    """Free-joint velocity so the whole body's com velocity is v_com and its angular momentum about the com is L
    (world frame), keeping the joint velocities already in data.qvel."""
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


def head_g_from_positions(head_pos, dt, breaks=()):
    """|a - g| / 9.81 of the head site from finite differences of its position; second differences that span
    a break (the release, where the velocity is set) are replaced by their neighbours."""
    n = len(head_pos)
    a = np.zeros((n, 3))
    if n >= 3:
        a[1:-1] = (head_pos[2:] - 2 * head_pos[1:-1] + head_pos[:-2]) / (dt * dt)
        a[0], a[-1] = a[1], a[-2]
    for b in breaks:
        lo, hi = max(b - 1, 0), min(b + 1, n - 1)
        left, right = a[max(lo - 1, 0)].copy(), a[min(hi + 1, n - 1)].copy()
        a[lo:b + 1] = left
        a[b + 1:hi + 1] = right
    g = np.linalg.norm(a + np.array([0.0, 0.0, 9.81]), axis=1) / 9.81
    cs = np.concatenate([[0.0], np.cumsum(g)])
    idx = np.arange(n)
    lo = np.maximum(0, idx + 1 - G15_SAMPLES)
    return g, (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)


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
    com_adr, comv_adr, angm_adr = sadr("com"), sadr("com_vel"), sadr("angmom")
    released = bool(ch.released)
    land_dur = float(getattr(ch, "land_duration", 1.0)) if released else 0.0
    post = max(3.0, land_dur + 1.0)
    K = ch.K
    n_max = K + 1 + (int(round((MAX_AIR + post + 0.5) / DT)) + 2 if released else 0)
    qlog = np.zeros((n_max, model.nq))
    comv = np.full((n_max, 3), np.nan)
    angm = np.full((n_max, 3), np.nan)
    con_k, con_a, con_b, con_d = [], [], [], []
    # free-phase loads for fallguy.biomech: actuator torques, contact force per performer body, cfrc_int
    nb = model.nbody
    tau_log = np.full((n_max, len(pf.act)), np.nan)
    bforce = np.zeros((n_max, nb))                 # |contact force| on each body (N)
    grf = np.zeros((n_max, 3))                     # total external contact force on the performer (world, N)
    cint = np.full((n_max, nb, 6), np.nan, np.float32)
    f6 = np.zeros(6)

    # ---------------------------------------------------------------- kinematic phases (collisions off, weld on)
    pf.set_collisions(model, False)
    for k in range(K + (0 if released else 1)):
        _set_pose(data, pf, *ch.kinematic(k))
        qlog[k] = data.qpos

    t_touch, surface, takeoff = None, None, None
    N = K + 1
    if released:
        # ------------------------------------------------------------ release
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
        pen = min([float(d) for d, gg in zip(data.contact.dist, data.contact.geom) if is_perf[gg[0]] ^ is_perf[gg[1]]]
                  + [0.0])
        data.qpos[pf.free_qadr + 2] += max(0.0, -pen) + 0.001        # start just clear of the support
        v_com, L, w = _takeoff_velocity(model, data, pf, ch.v_com_target, ch.L_target)
        data.ctrl[pf.act] = ch.flight_targets(0.0)
        takeoff = {"com": [round(float(v), 4) for v in data.subtree_com[pf.root]],
                   "v_com": [round(float(v), 4) for v in v_com], "angmom": [round(float(v), 3) for v in L],
                   "omega_deg_s": [round(float(math.degrees(v)), 1) for v in w],
                   "flight_time_est": round(ch.T_f, 4), "landing_surface_z_est": round(ch.landing_surface_z, 3)}
        # ------------------------------------------------------------ free dynamics
        gain_land = ch.landing_gain_scale()
        q_touch, airborne, first_hazard = None, False, None
        k = K
        while k < n_max:
            t_prev = k * DT
            s = t_prev - ch.t_takeoff
            if t_touch is None:
                pf.set_leg_gain_scale(model, 1.0 + (gain_land - 1.0) * C.smooth((s / ch.T_f - 0.8) / 0.2))
                tgt = ch.flight_targets(s)
            else:
                tgt = ch.landing_targets(t_prev - t_touch, q_touch)
            data.ctrl[pf.act] = tgt
            qlog[k] = data.qpos
            mujoco.mj_step(model, data)
            mujoco.mj_rnePostConstraint(model, data)
            tau_log[k] = data.actuator_force[pf.act]
            cint[k] = data.cfrc_int
            for ci in range(data.ncon):
                g1, g2 = data.contact.geom[ci]
                if is_perf[g1] == is_perf[g2]:
                    continue
                mujoco.mj_contactForce(model, data, ci, f6)
                fw = data.contact.frame[ci].reshape(3, 3).T @ f6[:3]
                if is_perf[g1]:
                    fw = -fw                                   # force on geom2 is +fw: flip when the performer is geom1
                b = int(model.geom_bodyid[g1 if is_perf[g1] else g2])
                bforce[k, b] += float(np.linalg.norm(fw))
                grf[k] += fw
            sd = data.sensordata
            comv[k] = sd[comv_adr:comv_adr + 3]
            angm[k] = sd[angm_adr:angm_adr + 3]
            touching = False
            if data.ncon:
                gg = data.contact.geom[:data.ncon]
                pa, pb = is_perf[gg[:, 0]], is_perf[gg[:, 1]]
                m = pa ^ pb
                if m.any():
                    touching = True
                    pg = np.where(pa, gg[:, 0], gg[:, 1])[m]
                    og = np.where(pa, gg[:, 1], gg[:, 0])[m]
                    con_k.append(np.full(len(pg), k, np.int32))
                    con_a.append(pg.astype(np.int32))
                    con_b.append(og.astype(np.int32))
                    con_d.append(data.contact.dist[:data.ncon][m].astype(np.float64))
                    if t_touch is None:
                        c = cat[og]
                        sup = np.isin(c, SUPPORT)
                        if airborne and sup.any():
                            t_touch, surface = t_prev, SURFACE[int(c[sup][0])]
                        elif airborne and (c == CATS["hazard"]).any():
                            first_hazard = t_prev if first_hazard is None else first_hazard
                            if t_prev - first_hazard > 0.25:         # resting on a hazard: that was the landing
                                t_touch, surface = first_hazard, "obstacle"
                        elif not airborne and s > 0.3:               # never left the support
                            t_touch, surface = ch.t_takeoff + DT, SURFACE.get(int(c[0]), "obstacle")
                        if t_touch is not None:
                            q_touch = tgt.copy()
            if not touching:
                airborne = True
                if t_touch is None:
                    first_hazard = None
            k += 1
            if t_touch is not None and k * DT >= t_touch + post:
                break
            if t_touch is None and s >= MAX_AIR:
                break
        N = k

    # ---------------------------------------------------------------- head G and com over every step
    t = np.arange(N) * DT
    qlog = qlog[:N]
    head = np.zeros((N, 3))
    com = np.zeros((N, 3))
    for i in range(N):
        data.qpos[:] = qlog[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        head[i] = data.site_xpos[pf.head_site]
        com[i] = data.subtree_com[pf.root]
    head_g, g15 = head_g_from_positions(head, DT, breaks=(K,) if released else ())
    comv = comv[:N]
    kin = slice(0, K + 1 if not released else K)
    comv[kin] = np.gradient(com[kin], DT, axis=0) if (K + 1) > 2 else 0.0
    con = {k_: (np.concatenate(v) if v else np.zeros(0, d)) for k_, v, d in
           (("con_k", con_k, np.int32), ("con_g1", con_a, np.int32), ("con_g2", con_b, np.int32),
            ("con_dist", con_d, float))}
    np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=t, qpos=qlog, head_g15=g15, head_g=head_g,
                        head_pos=head, com=com, com_vel=comv, angmom=angm[:N], free_tau=tau_log[:N],
                        free_body_force=bforce[:N], free_grf=grf[:N], free_cfrc_int=cint[:N], k_release=K if released else -1,
                        **con)

    # ---------------------------------------------------------------- events
    t_end = float(t[-1])
    phases = [dict(p) for p in ch.phase_events]
    t_takeoff = t_apex = t_land = None
    if released:
        t_takeoff = float(ch.t_takeoff)
        t_land = float(t_touch) if t_touch is not None else None
        fl = (t >= t_takeoff) & (t <= (t_land if t_land is not None else t_end))
        t_apex = float(t[int(np.argmax(np.where(fl, com[:, 2], -np.inf)))])
        air = getattr(ch, "air", None)
        land = getattr(ch, "land", None)
        if air is not None:
            phases.append({"name": air["name"], "kind": "air", "t0": round(t_takeoff, 4),
                           "t1": round(t_land if t_land is not None else t_end, 4)})
        if land is not None:
            phases.append({"name": land["name"], "kind": "land",
                           "t0": None if t_land is None else round(t_land, 4),
                           "t1": None if t_land is None else round(min(t_land + land_dur, t_end), 4)})
        t_finish = min(t_land + land_dur, t_end) if t_land is not None else t_end
    else:
        t_finish = ch.t_finish_kin if ch.t_finish_kin is not None else (phases[-1]["t1"] if phases else t_end)
    events = {
        "t_start": 0.0, "t_end": round(t_end, 4), "phases": phases,
        "t_takeoff": None if t_takeoff is None else round(t_takeoff, 4),
        "t_apex": None if t_apex is None else round(t_apex, 4),
        "t_landing": None if t_land is None else round(t_land, 4),
        "t_finish": round(float(t_finish), 4),
        "performer_root_body": perf.ROOT_BODY,
        "performer_geoms": [int(g) for g in pf.geoms],
        "hazard_geoms": [int(g) for g in ids.get("hazard", [])],
        "mat_geoms": [int(g) for g in ids.get("mat", [])],
        "floor_geoms": [int(g) for g in ids.get("floor", [])],
        "performer_collision_geoms": [int(g) for g in pf.collision_geoms],
        "obstacle_geoms": [int(g) for g in ids.get("obstacle", [])],
        "launch_geoms": [int(g) for g in ids.get("launch", [])],
        "t_kinematic_end": round(K * DT, 4), "airborne": released,
        "landed": t_touch is not None, "landing_surface": surface or "none",
        "target_mat": meta["landing_mat"], "takeoff": takeoff, "dt": DT,
        "choreography": ch.summary(),
        "geom_objects": {str(i): meta["objects"][nm] for i in range(model.ngeom)
                         if (nm := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)) in meta["objects"]},
    }
    with open(os.path.join(out_dir, "events.json"), "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)
    if released:
        log(f"sim: release t={t_takeoff:.2f}s, landing "
            f"{'t=%.2fs on %s' % (t_land, surface) if t_land is not None else 'none'}, {N} steps in "
            f"{time.perf_counter() - wall:.1f} s wall")
    else:
        log(f"sim: ground stunt, finish t={t_finish:.2f}s, {N} steps in {time.perf_counter() - wall:.1f} s wall")
    return events
