"""MuJoCo stunt simulation + Gate 1 metrics.

Everything reported here is measured from the simulation:
- head / spine / knee loads come from accelerometer sensors (MuJoCo reports
  proper acceleration, so free fall reads ~0 g and standing reads ~1 g);
- HIC15 is computed from the head accelerometer trace (standard formula);
- clearance is the minimum signed distance between any humanoid geom and any
  hazard geom (`mj_geomDistance`), collisions are real contacts;
- touchdown is the first contact with a landing surface after takeoff;
- framing projects head + pelvis through the scene camera.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from backend.params import clamp_params

TIMESTEP = 0.002
DEFAULT_DURATION = 2.4
RECORD_EVERY = 8  # 62.5 Hz playback frames

# Gate 1 limits. Conservative stunt-safety thresholds (HIC15 700 is the
# automotive regulatory limit; we use far stricter values for planned stunts).
LIMITS = {
    "peak_head_g": 15.0,      # 15 ms moving-average peak head acceleration
    "hic15": 250.0,
    "min_clearance_cm": 2.0,
    "landing_error_m": 0.65,
}

HUMANOID_GEOMS: List[Tuple[str, str]] = [
    ("geom_head", "head"), ("geom_neck", "neck"), ("geom_torso", "spine"), ("geom_pelvis", "spine"),
    ("geom_upper_arm_l", "arms"), ("geom_lower_arm_l", "arms"),
    ("geom_upper_arm_r", "arms"), ("geom_lower_arm_r", "arms"),
    ("geom_thigh_l", "knees"), ("geom_shin_l", "knees"), ("geom_foot_l", "knees"),
    ("geom_thigh_r", "knees"), ("geom_shin_r", "knees"), ("geom_foot_r", "knees"),
]

HUMANOID_XML = """
    <body name="pelvis" pos="{sx:.3f} {sy:.3f} {sz:.3f}">
      <freejoint name="root"/>
      <geom name="geom_pelvis" type="capsule" fromto="0 -0.10 0 0 0.10 0" size="0.09" mass="10"/>
      <body name="torso" pos="0 0 0.12">
        <joint name="abdomen_y" axis="0 1 0" range="-60 90" damping="6"/>
        <geom name="geom_torso" type="capsule" fromto="0 0 0.05 0 0 0.34" size="0.105" mass="18"/>
        <site name="torso_site" pos="0 0 0.20" size="0.02"/>
        <body name="head" pos="0 0 0.44">
          <joint name="neck_y" axis="0 1 0" range="-45 55" damping="3"/>
          <geom name="geom_neck" type="capsule" fromto="0 0 -0.06 0 0 0.02" size="0.05" mass="1.5"/>
          <geom name="geom_head" type="sphere" pos="0 0 0.11" size="0.105" mass="4.5"/>
          <site name="head_site" pos="0 0 0.11" size="0.02"/>
        </body>
        <body name="upper_arm_l" pos="0 0.19 0.30">
          <joint name="shoulder_l" axis="0 1 0" range="-170 60"/>
          <geom name="geom_upper_arm_l" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.045" mass="2.2"/>
          <body name="lower_arm_l" pos="0 0 -0.26">
            <joint name="elbow_l" axis="0 1 0" range="-140 5"/>
            <geom name="geom_lower_arm_l" type="capsule" fromto="0 0 0 0 0 -0.22" size="0.038" mass="1.5"/>
          </body>
        </body>
        <body name="upper_arm_r" pos="0 -0.19 0.30">
          <joint name="shoulder_r" axis="0 1 0" range="-170 60"/>
          <geom name="geom_upper_arm_r" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.045" mass="2.2"/>
          <body name="lower_arm_r" pos="0 0 -0.26">
            <joint name="elbow_r" axis="0 1 0" range="-140 5"/>
            <geom name="geom_lower_arm_r" type="capsule" fromto="0 0 0 0 0 -0.22" size="0.038" mass="1.5"/>
          </body>
        </body>
      </body>
      <body name="thigh_l" pos="0 0.11 -0.05">
        <joint name="hip_l" axis="0 1 0" range="-130 40" damping="5"/>
        <geom name="geom_thigh_l" type="capsule" fromto="0 0 0 0 0 -0.36" size="0.065" mass="7.5"/>
        <body name="shin_l" pos="0 0 -0.38">
          <joint name="knee_l" axis="0 1 0" range="-2 145" damping="4.5"/>
          <geom name="geom_shin_l" type="capsule" fromto="0 0 0 0 0 -0.34" size="0.05" mass="4"/>
          <geom name="geom_foot_l" type="box" pos="0.05 0 -0.38" size="0.09 0.045 0.03" mass="1"/>
          <site name="shin_l_site" pos="0 0 -0.17" size="0.02"/>
        </body>
      </body>
      <body name="thigh_r" pos="0 -0.11 -0.05">
        <joint name="hip_r" axis="0 1 0" range="-130 40" damping="5"/>
        <geom name="geom_thigh_r" type="capsule" fromto="0 0 0 0 0 -0.36" size="0.065" mass="7.5"/>
        <body name="shin_r" pos="0 0 -0.38">
          <joint name="knee_r" axis="0 1 0" range="-2 145" damping="4.5"/>
          <geom name="geom_shin_r" type="capsule" fromto="0 0 0 0 0 -0.34" size="0.05" mass="4"/>
          <geom name="geom_foot_r" type="box" pos="0.05 0 -0.38" size="0.09 0.045 0.03" mass="1"/>
          <site name="shin_r_site" pos="0 0 -0.17" size="0.02"/>
        </body>
      </body>
    </body>
"""

ACTUATORS = [
    ("abdomen_y", 180), ("neck_y", 80), ("shoulder_l", 110), ("elbow_l", 90),
    ("shoulder_r", 110), ("elbow_r", 90), ("hip_l", 200), ("knee_l", 200),
    ("hip_r", 200), ("knee_r", 200),
]


def build_mjcf(obstacles: List[Dict[str, Any]], start_pos: List[float], mat_offset_x: float = 0.0) -> str:
    geoms = []
    for obs in obstacles:
        pos = [float(v) for v in obs["pos"]]
        if obs["id"] == "crash_mat":
            pos[0] += float(mat_offset_x)
            contact = 'solref="0.06 1" solimp="0.5 0.95 0.05" friction="1.4 0.05 0.01"'  # soft foam
        else:
            contact = 'solref="0.008 1"'
        size = [max(0.005, float(v)) for v in obs["size"]]
        geoms.append(
            f'<geom name="{obs["id"]}" type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
            f'size="{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}" {contact}/>'
        )
    acts = "\n    ".join(f'<position name="act_{j}" joint="{j}" kp="{kp}"/>' for j, kp in ACTUATORS)
    sx, sy, sz = (float(v) for v in start_pos)
    return f"""
<mujoco model="fallguy">
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>
  <default>
    <joint type="hinge" damping="3" armature="0.02" limited="true"/>
    <geom condim="3" friction="0.9 0.05 0.01"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1" solref="0.008 1"/>
    {''.join(geoms)}
    {HUMANOID_XML.format(sx=sx, sy=sy, sz=sz)}
  </worldbody>
  <actuator>
    {acts}
  </actuator>
  <sensor>
    <accelerometer name="head_acc" site="head_site"/>
    <accelerometer name="torso_acc" site="torso_site"/>
    <accelerometer name="shin_l_acc" site="shin_l_site"/>
    <accelerometer name="shin_r_acc" site="shin_r_site"/>
  </sensor>
</mujoco>
"""


_MODEL_CACHE: Dict[str, mujoco.MjModel] = {}


def _get_model(obstacles, start_pos, mat_offset_x) -> mujoco.MjModel:
    key = hashlib.sha1(
        json.dumps([obstacles, start_pos, round(float(mat_offset_x), 3)], sort_keys=True, default=str).encode()
    ).hexdigest()
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = mujoco.MjModel.from_xml_string(build_mjcf(obstacles, start_pos, mat_offset_x))
        if len(_MODEL_CACHE) > 256:
            _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = model
    return model


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, ma[0]), ma])


def hic15(acc_g: np.ndarray, dt: float = TIMESTEP) -> float:
    """Head Injury Criterion over windows up to 15 ms. acc_g: resultant head acceleration in g."""
    if len(acc_g) < 2:
        return 0.0
    c = np.concatenate([[0.0], np.cumsum(acc_g) * dt])
    best = 0.0
    max_w = int(round(0.015 / dt))
    for w in range(1, max_w + 1):
        T = w * dt
        avg = (c[w:] - c[:-w]) / T
        val = float(np.max(T * np.power(np.clip(avg, 0, None), 2.5)))
        best = max(best, val)
    return best


def _camera_basis(camera: Dict[str, Any]):
    p = np.array(camera.get("pos", [0, -4.5, 1.5]), dtype=float)
    l = np.array(camera.get("lookat", [0.5, 0, 0.7]), dtype=float)
    f = l - p
    f /= max(1e-9, np.linalg.norm(f))
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = math.tan(math.radians(float(camera.get("fovy", 50))) / 2)
    return p, f, r, u, t


def _in_frame(point: np.ndarray, basis, aspect: float = 16 / 9) -> bool:
    p, f, r, u, t = basis
    d = point - p
    z = float(d @ f)
    if z <= 0.1:
        return False
    x = float(d @ r) / (z * t * aspect)
    y = float(d @ u) / (z * t)
    return abs(x) <= 0.95 and abs(y) <= 0.95


def _rad(deg: float) -> float:
    return math.radians(deg)


def simulate(
    candidate: Dict[str, Any],
    scene: Dict[str, Any],
    duration: float = DEFAULT_DURATION,
    record_frames: bool = True,
) -> Dict[str, Any]:
    """Run one candidate stunt and return Gate 1 telemetry (+ playback frames)."""
    params = clamp_params(candidate)
    obstacles = scene["scene_obstacles"]
    start_pos = scene["start_pos"]
    model = _get_model(obstacles, start_pos, params["mat_offset_x"])
    data = mujoco.MjData(model)

    name2geom = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
    ngeom = model.ngeom
    is_human = model.geom_bodyid != 0
    hazard_ids = [name2geom(o["id"]) for o in obstacles if o.get("is_hazard")]
    is_hazard = np.zeros(ngeom, dtype=bool)
    is_hazard[hazard_ids] = True
    mat_id = name2geom("crash_mat")
    head_id = name2geom("geom_head")
    head_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_site")
    is_surface = (~is_human) & (~is_hazard)  # floor, mat, platforms
    human_ids = [name2geom(n) for n, _ in HUMANOID_GEOMS]
    human_regions = [r for _, r in HUMANOID_GEOMS]
    gid2name = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(ngeom)}

    mat_obs = next((o for o in obstacles if o["id"] == "crash_mat"), None)
    mat_center = np.array(mat_obs["pos"][:2], dtype=float) if mat_obs else np.array([2.0, 0.0])
    if mat_obs:
        mat_center[0] += params["mat_offset_x"]
    hazard_far_x = max((o["pos"][0] + o["size"][0] for o in obstacles if o.get("is_hazard")), default=None)
    hazard_x_lo = min((o["pos"][0] - o["size"][0] for o in obstacles if o.get("is_hazard")), default=None)

    act_qadr = [int(model.jnt_qposadr[model.actuator_trnid[i, 0]]) for i in range(model.nu)]

    # Takeoff state: launch velocity + rotation applied to the free joint.
    mujoco.mj_forward(model, data)
    data.qvel[0] = params["takeoff_vx"]
    data.qvel[2] = params["takeoff_vz"]
    data.qvel[4] = params["pitch_rate"]

    tuck_t, ext_t = params["tuck_time"], params["extend_time"]
    reach = -_rad(params["arm_reach_deg"])
    comp = params["landing_compliance"]
    targets_a = [_rad(10), 0.0, reach, _rad(-20), reach, _rad(-20), _rad(-15), _rad(15), _rad(-15), _rad(15)]
    targets_b = [_rad(35), _rad(20), reach, _rad(-60), reach, _rad(-60),
                 -_rad(params["hip_tuck_deg"]), _rad(params["knee_tuck_deg"]),
                 -_rad(params["hip_tuck_deg"]), _rad(params["knee_tuck_deg"])]
    targets_c = [_rad(15), _rad(15), _rad(-60), _rad(-40), _rad(-60), _rad(-40), _rad(-20), _rad(20), _rad(-20), _rad(20)]
    hip_abs, knee_abs = -_rad(25 + 55 * comp), _rad(30 + 80 * comp)
    targets_d = [_rad(20 + 25 * comp), _rad(25), _rad(-50), _rad(-60), _rad(-50), _rad(-60), hip_abs, knee_abs, hip_abs, knee_abs]
    stiff_d = 1.0 - 0.7 * comp

    steps = int(duration / TIMESTEP)
    head_acc = np.zeros(steps)
    torso_acc = np.zeros(steps)
    shin_acc = np.zeros(steps)
    basis = _camera_basis(scene.get("camera", {}))
    has_geom_distance = hasattr(mujoco, "mj_geomDistance")

    frames: List[Dict[str, Any]] = []
    trajectory: List[List[float]] = []
    events: List[Dict[str, Any]] = []
    collision: Optional[Dict[str, Any]] = None
    head_strike: Optional[Dict[str, Any]] = None
    touchdown: Optional[Dict[str, Any]] = None
    min_clear = math.inf
    framed = sampled = 0
    quat = np.zeros(4)
    fromto = np.zeros(6)
    g_running = np.zeros(8)

    for step in range(steps):
        t = step * TIMESTEP
        if touchdown is not None:
            q = data.qpos
            for i in range(10):
                cur = q[act_qadr[i]]
                data.ctrl[i] = cur + (targets_d[i] - cur) * stiff_d
        else:
            tgt = targets_a if t < tuck_t else (targets_b if t < ext_t else targets_c)
            data.ctrl[:10] = tgt

        mujoco.mj_step(model, data)

        sd = data.sensordata
        head_acc[step] = math.sqrt(sd[0] ** 2 + sd[1] ** 2 + sd[2] ** 2) / 9.81
        torso_acc[step] = math.sqrt(sd[3] ** 2 + sd[4] ** 2 + sd[5] ** 2) / 9.81
        shin_acc[step] = max(math.sqrt(sd[6] ** 2 + sd[7] ** 2 + sd[8] ** 2),
                             math.sqrt(sd[9] ** 2 + sd[10] ** 2 + sd[11] ** 2)) / 9.81

        if data.ncon:
            pairs = np.asarray(data.contact.geom)[: data.ncon]
            for g1, g2 in pairs:
                h1, h2 = is_human[g1], is_human[g2]
                if h1 == h2:
                    continue
                hum, other = (g1, g2) if h1 else (g2, g1)
                if is_hazard[other] and collision is None:
                    collision = {"time_s": round(t, 3), "body": gid2name[hum], "obstacle": gid2name[other]}
                    events.append({"t": round(t, 3), "type": "collision", **collision})
                elif is_surface[other] and t > 0.12:
                    if touchdown is None:
                        touchdown = {
                            "time_s": round(t, 3),
                            "surface": gid2name[other],
                            "pelvis_xy": [float(data.qpos[0]), float(data.qpos[1])],
                            "first_contact": gid2name[hum],
                        }
                        events.append({"t": round(t, 3), "type": "touchdown", "surface": gid2name[other]})
                    if hum == head_id and head_strike is None and other != mat_id:
                        head_strike = {"time_s": round(t, 3), "surface": gid2name[other]}
                        events.append({"t": round(t, 3), "type": "head_strike", "surface": gid2name[other]})

        # Clearance: closest approach of any body part to any hazard.
        if hazard_ids and step % 2 == 0:
            px = data.qpos[0]
            if hazard_x_lo - 1.2 <= px <= hazard_far_x + 1.2:
                if has_geom_distance:
                    for hz in hazard_ids:
                        for hg in human_ids:
                            dist = mujoco.mj_geomDistance(model, data, hg, hz, 1.0, fromto)
                            if dist < min_clear:
                                min_clear = dist

        if step % RECORD_EVERY == 0:
            g_running = np.roll(g_running, 1)
            g_running[0] = head_acc[step]
            sampled += 1
            head_p = data.site_xpos[head_site]
            if _in_frame(np.asarray(data.qpos[:3]), basis) and _in_frame(np.asarray(head_p), basis):
                framed += 1
            if record_frames:
                trajectory.append([round(float(v), 3) for v in data.qpos[:3]])
                geoms = []
                for gid in human_ids:
                    mujoco.mju_mat2Quat(quat, data.geom_xmat[gid])
                    p = data.geom_xpos[gid]
                    geoms.append({
                        "p": [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)],
                        "q": [round(float(quat[1]), 4), round(float(quat[2]), 4),
                              round(float(quat[3]), 4), round(float(quat[0]), 4)],  # x,y,z,w
                    })
                frames.append({"t": round(t, 3), "geoms": geoms, "g": round(float(np.mean(head_acc[max(0, step - 7):step + 1])), 2)})

    # ---- metrics -------------------------------------------------------
    skip = 10  # ignore the first 20 ms (launch transient)
    head_f = _moving_average(head_acc, 8)[skip:]
    peak_head_g = float(np.max(head_f)) if len(head_f) else 0.0
    peak_spine_g = float(np.max(_moving_average(torso_acc, 8)[skip:]))
    peak_knee_g = float(np.max(_moving_average(shin_acc, 8)[skip:]))
    hic = hic15(_moving_average(head_acc, 3)[skip:])

    if not hazard_ids:
        min_clear_cm = 100.0
    elif min_clear is math.inf:
        min_clear_cm = 100.0
    else:
        min_clear_cm = round(min_clear * 100.0, 1)
    if collision is not None:
        min_clear_cm = min(min_clear_cm, 0.0)

    td_xy = np.array(touchdown["pelvis_xy"]) if touchdown else np.array(data.qpos[:2])
    landing_error = float(np.linalg.norm(td_xy - mat_center))
    on_mat = bool(touchdown and touchdown["surface"] == "crash_mat")
    cleared = hazard_far_x is None or float(td_xy[0]) > hazard_far_x
    framing_pct = int(round(100.0 * framed / max(1, sampled)))

    def clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    s_g = clip01((2 * LIMITS["peak_head_g"] - peak_head_g) / (2 * LIMITS["peak_head_g"] - 3.0))
    s_hic = clip01((2 * LIMITS["hic15"] - hic) / (2 * LIMITS["hic15"] - 30.0))
    safety = 100.0 * min(s_g, s_hic) * (0.6 if head_strike else 1.0)
    if collision is not None:
        clearance_s = 0.0
    elif not cleared:
        clearance_s = 10.0
    elif min_clear_cm < 5.0:
        clearance_s = 60.0 + 8.0 * max(0.0, min_clear_cm)
    elif min_clear_cm <= 42.0:
        clearance_s = 100.0
    else:
        clearance_s = max(55.0, 100.0 - 1.1 * (min_clear_cm - 42.0))
    if on_mat:
        landing_s = max(40.0, 100.0 * clip01(1.0 - landing_error / 0.9))
    else:
        landing_s = 40.0 * clip01(1.0 - landing_error / 2.0)
    score = int(round(0.35 * safety + 0.25 * clearance_s + 0.20 * landing_s + 0.20 * framing_pct))

    checks = {
        "no_collision": collision is None,
        "cleared_obstacles": bool(cleared),
        "clearance_ok": min_clear_cm >= LIMITS["min_clearance_cm"],
        "head_g_ok": peak_head_g < LIMITS["peak_head_g"],
        "hic_ok": hic < LIMITS["hic15"],
        "landed_on_mat": on_mat,
        "landing_accuracy_ok": landing_error <= LIMITS["landing_error_m"],
    }
    gate_1 = all(checks.values())

    problems = []
    if collision:
        problems.append(f"{collision['body']} hit {collision['obstacle']} at t={collision['time_s']}s")
    elif not cleared:
        problems.append("touched down before clearing the obstacle")
    elif min_clear_cm < LIMITS["min_clearance_cm"]:
        problems.append(f"clearance only {min_clear_cm}cm")
    elif min_clear_cm > 40:
        problems.append(f"over-jumped (+{min_clear_cm}cm clearance)")
    if not checks["head_g_ok"]:
        problems.append(f"peak head {peak_head_g:.1f}g > {LIMITS['peak_head_g']}g")
    if not checks["hic_ok"]:
        problems.append(f"HIC15 {hic:.0f} > {LIMITS['hic15']:.0f}")
    if head_strike:
        problems.append(f"head struck {head_strike['surface']} at t={head_strike['time_s']}s")
    if not on_mat:
        surf = touchdown["surface"] if touchdown else "nothing"
        problems.append(f"landed on {surf}, {landing_error:.2f}m from mat centre")
    elif not checks["landing_accuracy_ok"]:
        problems.append(f"landed {landing_error:.2f}m off mat centre")
    diagnostic = "GATE 1 PASS" if gate_1 else "; ".join(problems) or "failed"

    def level(v: float, lim: float) -> str:
        return "critical" if v >= lim else ("warn" if v >= 0.6 * lim else "safe")

    result: Dict[str, Any] = {
        "ghost_id": candidate.get("ghost_id", "ghost_A"),
        "label": candidate.get("label", "candidate"),
        "color": candidate.get("color", "#3b82f6"),
        "origin": candidate.get("origin", "unknown"),
        "parameters": params,
        "gate_1_passed": gate_1,
        "gate_1_checks": checks,
        "stunt_score": score,
        "score_breakdown": {
            "safety": round(safety, 1), "clearance": round(clearance_s, 1),
            "landing": round(landing_s, 1), "framing": framing_pct,
        },
        "metrics": {
            "peak_head_g": round(peak_head_g, 2),
            "hic15": round(hic, 1),
            "peak_spine_g": round(peak_spine_g, 2),
            "peak_knee_g": round(peak_knee_g, 2),
            "min_clearance_cm": min_clear_cm,
            "landing_error_m": round(landing_error, 3),
            "landed_on": touchdown["surface"] if touchdown else None,
            "touchdown_time_s": touchdown["time_s"] if touchdown else None,
            "touchdown_x": round(float(td_xy[0]), 3),
            "camera_framing_pct": framing_pct,
        },
        "joint_stress": {
            "head": level(peak_head_g, LIMITS["peak_head_g"]) if not head_strike else "critical",
            "neck": level(hic, LIMITS["hic15"]),
            "spine": level(peak_spine_g, 20.0),
            "knees": level(peak_knee_g, 30.0),
        },
        "collision": collision,
        "head_strike": head_strike,
        "events": events,
        "diagnostic": diagnostic,
    }
    if record_frames:
        result["trajectory_points"] = trajectory
        result["humanoid_geoms_meta"] = [
            {"name": n, "region": r, "type": {2: "sphere", 3: "capsule", 6: "box"}.get(int(model.geom_type[g]), "capsule"),
             "size": [round(float(s), 4) for s in model.geom_size[g]]}
            for (n, r), g in zip(HUMANOID_GEOMS, human_ids)
        ]
        result["frames"] = frames
        result["frame_dt"] = TIMESTEP * RECORD_EVERY
    return result


def compact(result: Dict[str, Any]) -> Dict[str, Any]:
    """History-friendly summary (no frames) for the critic / optimizer."""
    return {
        "ghost_id": result["ghost_id"],
        "label": result["label"],
        "origin": result.get("origin"),
        "parameters": result["parameters"],
        "stunt_score": result["stunt_score"],
        "gate_1_passed": result["gate_1_passed"],
        "metrics": result["metrics"],
        "diagnostic": result["diagnostic"],
    }
