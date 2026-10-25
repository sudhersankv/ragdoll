"""Tests for scripts/engine/render.py (retime, interpolation, cameras, HUD, encoding, timing).

Fixtures are self-contained: a small capsule-person MJCF with a kinematic run-up and a free ballistic flight,
and (skipped when the prototype is absent) the real H1 venue from <repo>/barrel_roll.py, imported in-process
and released at take-off. Outputs go to $FALLGUY_TEST_OUT if set (kept for inspection), else a pytest tmp dir.
Run: python -m pytest tests/test_render.py -q
"""
import importlib.util
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
import pytest
from PIL import Image, ImageDraw

SKILL = Path(__file__).resolve().parents[1]
REPO = SKILL.parents[2]
sys.path.insert(0, str(SKILL / "scripts"))
from engine import render  # noqa: E402

DT = 0.002
G = 9.81


# ----------------------------------------------------------------------------- shared helpers
def head_g15(acc, t, t_takeoff):
    """|a| / g, 15 ms trailing moving average, 0 before take-off (contract: trajectory.npz head_g15)."""
    g = np.linalg.norm(acc, axis=1) / G
    n = max(1, int(round(0.015 / DT)))
    c = np.cumsum(np.r_[0.0, g])
    avg = np.array([(c[i + 1] - c[max(0, i + 1 - n)]) / min(n, i + 1) for i in range(len(g))])
    avg[t < t_takeoff] = 0.0
    return avg


def save_case(out, xml, t, qpos, g15, events, plan, params):
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene.xml").write_text(xml)
    np.savez(out / "trajectory.npz", t=np.asarray(t), qpos=np.asarray(qpos), head_g15=np.asarray(g15))
    (out / "events.json").write_text(json.dumps(events, indent=1))
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "params.json").write_text(json.dumps(params, indent=1))
    return {"scene_xml": str(out / "scene.xml"), "trajectory": str(out / "trajectory.npz"),
            "events": str(out / "events.json"), "plan": plan, "params": params, "metrics": None, "dir": out}


def contact_sheet(mp4, out_png, picks=12, cols=4, tile=(480, 270)):
    """Tile `picks` evenly spaced frames of an mp4 (labelled with frame index and video time)."""
    rd = imageio.get_reader(mp4)
    frames = [f for f in rd]
    fps = rd.get_meta_data()["fps"]
    rd.close()
    idx = np.linspace(0, len(frames) - 1, picks).round().astype(int)
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile[0], rows * tile[1]), "white")
    d = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        x, y = (k % cols) * tile[0], (k // cols) * tile[1]
        sheet.paste(Image.fromarray(frames[i]).resize(tile), (x, y))
        d.rectangle([x, y + tile[1] - 18, x + 150, y + tile[1]], fill="black")
        d.text((x + 4, y + tile[1] - 15), f"#{i}  {i / fps:.2f}s", fill="white")
    sheet.save(out_png)
    return out_png


def seg_bbox(model, data, cam, geoms, size):
    """Pixel bbox (x0, y0, x1, y1) of the given geoms in a segmentation render, or None if not visible."""
    w, h = size
    with mujoco.Renderer(model, h, w) as r:
        r.enable_segmentation_rendering()
        r.update_scene(data, cam, render._scene_option())
        seg = r.render()
    mask = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], geoms)
    ys, xs = np.nonzero(mask)
    return None if xs.size == 0 else (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


# ----------------------------------------------------------------------------- capsule-person case
ROOM = {"x_min": -7.0, "x_max": 5.0, "y_min": -3.2, "y_max": 6.0, "ceiling_z": 3.4}


def capsule_xml(room=ROOM):
    x0, x1, y0, y1, top = room["x_min"], room["x_max"], room["y_min"], room["y_max"], room["ceiling_z"]
    cx, cy, hx, hy, hz = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 0.2, (y1 - y0) / 2 + 0.2, top / 2
    P = 'contype="2" conaffinity="1"'
    return f"""
<mujoco model="fallguy_capsule_case">
  <option timestep="{DT}"/>
  <visual><global offwidth="640" offheight="360"/><quality shadowsize="2048"/>
    <headlight ambient="0.35 0.35 0.37" diffuse="0.25 0.25 0.25" specular="0 0 0"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.58 0.57 0.55" rgb2="0.52 0.51 0.5" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="12 12" texuniform="true" reflectance="0.02"/>
  </asset>
  <default><geom contype="1" conaffinity="2" friction="0.9 0.05 0.01"/>
    <joint damping="3" stiffness="40" armature="0.01"/></default>
  <worldbody>
    <light name="key" directional="true" pos="-2 -3 6" dir="0.35 0.45 -1" diffuse="0.55 0.55 0.52" castshadow="true"/>
    <geom name="floor" type="plane" size="30 30 0.1" material="floor"/>
    <geom name="wall_xn" type="box" size="0.1 {hy:.2f} {hz:.2f}" pos="{x0 - 0.1:.2f} {cy:.2f} {hz:.2f}" rgba="0.82 0.82 0.8 1"/>
    <geom name="wall_xp" type="box" size="0.1 {hy:.2f} {hz:.2f}" pos="{x1 + 0.1:.2f} {cy:.2f} {hz:.2f}" rgba="0.82 0.82 0.8 1"/>
    <geom name="wall_yn" type="box" size="{hx:.2f} 0.1 {hz:.2f}" pos="{cx:.2f} {y0 - 0.1:.2f} {hz:.2f}" rgba="0.78 0.8 0.82 1"/>
    <geom name="wall_yp" type="box" size="{hx:.2f} 0.1 {hz:.2f}" pos="{cx:.2f} {y1 + 0.1:.2f} {hz:.2f}" rgba="0.78 0.8 0.82 1"/>
    <geom name="table" type="box" size="0.3 0.8 0.4" pos="0 0 0.4" rgba="0.55 0.4 0.25 1"/>
    <geom name="mat" type="box" size="0.9 1.0 0.15" pos="1.6 0 0.15" rgba="0.15 0.35 0.85 1"/>
    <geom name="chair" type="box" size="0.25 0.25 0.45" pos="2.5 -2.3 0.45" rgba="0.1 0.1 0.12 1"/>
    <camera name="overview" pos="-1 -2.9 3.0" xyaxes="1 0 0 0 0.6 0.8"/>
    <body name="pelvis" pos="-4.15 0 1.02">
      <freejoint/>
      <geom name="hips" type="capsule" fromto="0 -0.1 0 0 0.1 0" size="0.1" rgba="0.2 0.22 0.27 1" {P}/>
      <geom name="torso" type="capsule" fromto="0 0 0.08 0 0 0.42" size="0.12" rgba="0.95 0.5 0.15 1" {P}/>
      <body name="head" pos="0 0 0.56">
        <joint name="neck" type="ball" damping="2" stiffness="20" range="0 40"/>
        <geom name="head" type="sphere" size="0.1" rgba="0.85 0.7 0.6 1" {P}/>
        <site name="head_imu" pos="0 0 0.02" size="0.01"/>
      </body>
      <body name="arm_l" pos="0 0.2 0.4"><joint name="sh_l" axis="0 1 0" range="-180 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.045" rgba="0.95 0.5 0.15 1" {P}/></body>
      <body name="arm_r" pos="0 -0.2 0.4"><joint name="sh_r" axis="0 1 0" range="-180 60"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.045" rgba="0.2 0.55 0.95 1" {P}/></body>
      <body name="thigh_l" pos="0 0.1 -0.05"><joint name="hip_l" axis="0 1 0" range="-120 40"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.065" rgba="0.2 0.22 0.27 1" {P}/>
        <body name="shin_l" pos="0 0 -0.47"><joint name="knee_l" axis="0 1 0" range="0 140"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05" rgba="0.2 0.22 0.27 1" {P}/>
          <geom type="box" pos="0.05 0 -0.475" size="0.09 0.04 0.025" rgba="0.9 0.9 0.9 1" {P}/></body></body>
      <body name="thigh_r" pos="0 -0.1 -0.05"><joint name="hip_r" axis="0 1 0" range="-120 40"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.065" rgba="0.2 0.22 0.27 1" {P}/>
        <body name="shin_r" pos="0 0 -0.47"><joint name="knee_r" axis="0 1 0" range="0 140"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05" rgba="0.2 0.22 0.27 1" {P}/>
          <geom type="box" pos="0.05 0 -0.475" size="0.09 0.04 0.025" rgba="0.9 0.9 0.9 1" {P}/></body></body>
    </body>
  </worldbody>
  <sensor><accelerometer name="head_acc" site="head_imu"/></sensor>
</mujoco>"""


def build_capsule_case(out, room=ROOM):
    """Kinematic run-up (qpos set directly), then a free ballistic roll over the table onto the mat."""
    xml = capsule_xml(room)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    jadr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j] for j in range(model.njnt)}
    t_runup, t_take, v, x_take = 0.6, 2.4, 3.5, -1.0
    acc_run = v / (t_take - t_runup)
    x_start = x_take - 0.5 * v * (t_take - t_runup)
    ts, qs, accs = [], [], []

    def record():
        mujoco.mj_rnePostConstraint(model, data)
        ts.append(data.time)
        qs.append(data.qpos.copy())
        accs.append(data.sensordata[:3].copy())

    while data.time < t_take - 1e-9:                       # kinematic run-up
        t = data.time
        u = max(0.0, t - t_runup)
        x = x_start + 0.5 * acc_run * u * u
        amp = min(1.0, u / 0.6)
        ph = 2 * math.pi * 2.2 * u
        data.qpos[:3] = [x, 0.0, 1.02 - 0.03 * amp * abs(math.sin(ph))]
        lean = math.radians(10) * amp
        data.qpos[3:7] = [math.cos(lean / 2), 0, math.sin(lean / 2), 0]
        data.qpos[jadr["hip_l"]] = -0.6 * amp * math.sin(ph)
        data.qpos[jadr["hip_r"]] = 0.6 * amp * math.sin(ph)
        data.qpos[jadr["knee_l"]] = 0.9 * amp * max(0.0, math.sin(ph + 1.2))
        data.qpos[jadr["knee_r"]] = 0.9 * amp * max(0.0, -math.sin(ph + 1.2))
        data.qpos[jadr["sh_l"]] = 0.7 * amp * math.sin(ph)
        data.qpos[jadr["sh_r"]] = -0.7 * amp * math.sin(ph)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        record()
        data.time += DT
    vz = 3.8
    t_flight = 2 * vz / G
    data.qvel[:3] = [v, 0.0, vz]
    R = data.xmat[model.body("pelvis").id].reshape(3, 3)
    data.qvel[3:6] = R.T @ np.array([2 * math.pi / t_flight * 0.9, 0.0, 0.0])   # free-joint ang. vel. is local
    t_land, t_apex, zmax = None, t_take, 0.0
    perf = {g for g in range(model.ngeom) if model.geom_bodyid[g] >= model.body("pelvis").id}
    while True:
        mujoco.mj_step(model, data)
        record()
        if data.qpos[2] > zmax and t_land is None:
            zmax, t_apex = data.qpos[2], data.time
        if t_land is None and data.time > t_take + 0.05:
            if any((c.geom1 in perf) != (c.geom2 in perf) for c in data.contact[:data.ncon]):
                t_land = data.time
        if (t_land and data.time > t_land + 2.6) or data.time > t_take + 6:
            break
    t = np.array(ts)
    root = model.body("pelvis").id
    events = {"t_start": 0.0, "t_runup": t_runup, "t_takeoff": t_take, "t_apex": round(t_apex, 3),
              "t_landing": round(t_land, 3), "t_end": round(float(t[-1]), 3), "performer_root_body": "pelvis",
              "performer_geoms": sorted(int(g) for g in perf), "hazard_geoms": [model.geom("table").id],
              "mat_geoms": [model.geom("mat").id], "floor_geoms": [model.geom("floor").id]}
    plan = {"version": 1, "brief": "barrel roll over the table", "duration_sec": 9.0,
            "room": dict(room, floor_rgb=[0.55, 0.55, 0.55], wall_rgb=[0.8, 0.8, 0.8]),
            "camera": {"match": {"pos": [-6.4, -2.4, 1.7], "lookat": [0.0, 0.0, 0.8], "fovy": 50}},
            "stunt": {"type": "barrel_roll", "start_x": x_start, "takeoff_x": x_take, "landing_x": 1.6, "lane_y": 0.0},
            "performer": {"model": "capsule"}}
    params = {"id": "w1c1", "label": "test roll", "camera": "hero"}
    case = save_case(out, xml, t, qs, head_g15(np.array(accs), t, t_take), events, plan, params)
    case["root"] = root
    return case


# ----------------------------------------------------------------------------- H1 venue case (prototype)
def build_h1_case(out):
    """The prototype venue + Unitree H1 from <repo>/barrel_roll.py: kinematic run-up, weld released at
    take-off, contacts on, free flight and landing (as in the release_at_takeoff experiment)."""
    br_path, plan_path = REPO / "barrel_roll.py", REPO / "plan.json"
    h1_dir = REPO / "models" / "mujoco_menagerie" / "unitree_h1"
    if not (br_path.is_file() and plan_path.is_file() and h1_dir.is_dir()):
        pytest.skip("prototype barrel_roll.py / plan.json / H1 assets not found next to the skill")
    spec = importlib.util.spec_from_file_location("fallguy_test_barrel_roll", br_path)
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    old = json.loads(plan_path.read_text())
    ch = br.Choreo(old)
    root = ET.fromstring(br.build_xml(old, ch))
    coll = root.find(".//default[@class='collision']/geom")
    coll.set("contype", "1")
    coll.set("conaffinity", "1")
    ET.SubElement(root.find(".//body[@name='torso_link']"), "site", name="head_imu", pos="0.045 0 0.68", size="0.01")
    ET.SubElement(ET.SubElement(root, "sensor"), "accelerometer", name="head_acc", site="head_imu")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rig = br.H1Rig()
    pel = model.body("pelvis").id
    hum = np.array([g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == pel])
    hum_coll = np.array([g for g in hum if model.geom_group[g] == 3])
    floor = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][0]

    def contacts(on):
        model.geom_contype[hum_coll] = model.geom_conaffinity[hum_coll] = 1 if on else 0

    contacts(False)
    p, q = ch.root(0.0)
    data.qpos[:3], data.qpos[3:7] = p, q
    for n, val in zip(br.H1_JOINTS, rig.ctrl_for(ch.joints(0.0))):
        data.qpos[model.jnt_qposadr[model.joint(n).id]] = val
    mujoco.mj_forward(model, data)
    ts, qs, accs = [], [], []

    def step():
        data.ctrl[:] = rig.ctrl_for(ch.joints(data.time))
        mujoco.mj_step(model, data)
        ts.append(data.time)
        qs.append(data.qpos.copy())
        accs.append(data.sensordata[:3].copy())

    while data.time < br.T_TAKEOFF - 1e-9:                 # prototype run-up (mocap weld, no contacts)
        data.mocap_pos[0], data.mocap_quat[0] = ch.root(data.time)
        step()
    data.eq_active[0] = 0
    contacts(True)
    vz = math.sqrt(2 * G * (ch.apex - ch.z0))
    data.qvel[0:3] = [ch.vx, 0.0, vz]
    data.qvel[3:6] = data.xmat[pel].reshape(3, 3).T @ np.array([2 * math.pi / ch.T_f, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    pen = min([c.dist for c in data.contact[:data.ncon] if floor in (c.geom1, c.geom2)] + [0.0])
    data.qpos[2] += max(0.0, -pen) + 0.002
    t_take = data.time
    t_land, t_apex, zmax = None, t_take, 0.0
    while True:
        step()
        if t_land is None and data.qpos[2] > zmax:
            zmax, t_apex = data.qpos[2], data.time
        if t_land is None and data.time > t_take + 0.05:
            if any((c.geom1 in hum_coll) != (c.geom2 in hum_coll) for c in data.contact[:data.ncon]):
                t_land = data.time
        if (t_land and data.time > t_land + 3.0) or data.time > t_take + 6:
            break
    t = np.array(ts)
    scene = model.body("scene").id
    events = {"t_start": 0.0, "t_runup": 1.0, "t_takeoff": round(t_take, 4), "t_apex": round(t_apex, 4),
              "t_landing": round(t_land or t_take + 1.0, 4), "t_end": round(float(t[-1]), 4),
              "performer_root_body": "pelvis", "performer_geoms": hum_coll.tolist(),
              "hazard_geoms": [g for g in range(model.ngeom) if model.geom_bodyid[g] == scene],
              "mat_geoms": [], "floor_geoms": [floor]}
    env = old["environment"]["room"]
    plan = {"version": 1, "brief": "barrel roll over the shelf", "duration_sec": 9.0,
            "room": {k: env[k] for k in ("x_min", "x_max", "y_min", "y_max", "ceiling_z")},
            "camera": {"match": {"pos": [-8.0, 0.0, 1.5], "lookat": [-0.8, 0.0, 1.0], "fovy": 52}},
            "stunt": {"type": "barrel_roll", "start_x": ch.x_start, "takeoff_x": ch.x_take, "landing_x": ch.x_land,
                      "lane_y": 0.0},
            "performer": {"model": "h1"}}
    params = {"id": "w2c3", "label": "prototype release", "camera": "hero"}
    return save_case(out, xml, t, qs, head_g15(np.array(accs), t, t_take), events, plan, params)


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def out_dir(tmp_path_factory):
    env = os.environ.get("FALLGUY_TEST_OUT")
    if env:
        Path(env).mkdir(parents=True, exist_ok=True)
        return Path(env)
    return tmp_path_factory.mktemp("render")


@pytest.fixture(scope="session")
def capsule(out_dir):
    return build_capsule_case(out_dir / "capsule")


@pytest.fixture(scope="session")
def h1(out_dir):
    return build_h1_case(out_dir / "h1")


EVENTS = {"t_start": 0.0, "t_runup": 0.8, "t_takeoff": 3.1, "t_apex": 3.55, "t_landing": 4.0, "t_end": 7.0}


# ----------------------------------------------------------------------------- retime
@pytest.mark.parametrize("fps", [30, 60])
@pytest.mark.parametrize("duration", [5, 9, 15])
def test_retime_frame_counts(duration, fps):
    tl = render._timeline(EVENTS, duration, fps)
    t = render.retime(EVENTS, duration, fps)
    assert len(t) == round(duration * fps) == tl["frames"]
    assert np.allclose(t, tl["times"])
    moving = len(t) - tl["hold_frames"]
    assert np.all(np.diff(t[:moving]) > 0), "frame times must strictly increase until the hold"
    assert np.all(t[moving:] == tl["t1"]) and t[moving - 1] <= tl["t1"], "held frames show the clip's last instant"
    assert EVENTS["t_start"] <= t[0] and t[-1] <= EVENTS["t_end"]
    assert t[0] <= EVENTS["t_takeoff"] - 0.6 and t[-1] >= EVENTS["t_landing"] + 0.6, "take-off and landing on screen"
    assert render.SLOW_MIN <= tl["factor"] <= render.SLOW_MAX
    step = np.diff(t[:moving]) * fps                        # playback rate between frames
    in_flight = (t[:moving - 1] > EVENTS["t_takeoff"] - 0.05) & (t[:moving - 1] < EVENTS["t_landing"] + 0.15)
    assert np.allclose(step[in_flight], tl["factor"], atol=0.02), "flight plays at the chosen factor"
    if duration == 5:      # too long in real time: no slow motion, run-up start trimmed
        assert tl["factor"] == 1.0 and t[0] > EVENTS["t_runup"] - render.PRE_ROLL
    if duration == 9:      # fills the clip with slow motion
        assert 0.25 < tl["factor"] < 0.5 and tl["hold_frames"] == 0
        assert abs(t[-1] - tl["t1"]) < 1e-6 and t[0] == EVENTS["t_runup"] - render.PRE_ROLL
        assert np.all(np.abs(step[(t[:-1] < EVENTS["t_takeoff"] - 0.5) | (t[:-1] > EVENTS["t_landing"] + 0.6)] - 1) < 0.02)
    if duration == 15:     # slowest factor, pre-roll from t_start, settle to t_end, then hold the last frame
        assert tl["factor"] == 0.25 and t[0] == EVENTS["t_start"] and t[-1] == EVENTS["t_end"]
        assert tl["hold_frames"] > 0


def test_retime_edge_cases():
    no_land = dict(EVENTS, t_landing=None)
    assert len(render.retime(no_land, 6, 30)) == 180
    short = render.retime(EVENTS, 1.0, 30)                  # shorter than the flight: centred on it
    assert len(short) == 30 and short[0] < EVENTS["t_landing"] and short[-1] > EVENTS["t_takeoff"]
    with pytest.raises(ValueError, match="t_takeoff"):
        render.retime({"t_end": 5.0}, 5, 30)
    with pytest.raises(ValueError, match="at least 2 frames"):
        render.retime(EVENTS, 0.01, 30)


# ----------------------------------------------------------------------------- interpolation
def test_interpolation_is_smooth_in_slow_motion(capsule):
    """60 fps at x0.25 is 4.2 ms of sim per frame vs 2 ms physics steps: snapping judders, interpolation not."""
    model = mujoco.MjModel.from_xml_path(capsule["scene_xml"])
    traj = render._Trajectory.load(model, capsule["trajectory"])
    ev = json.loads(Path(capsule["events"]).read_text())
    tl = render._timeline(ev, 15, 60)                       # long clip -> factor 0.25
    assert tl["factor"] == 0.25
    sel = (tl["times"] > ev["t_takeoff"] + 0.02) & (tl["times"] < ev["t_landing"] - 0.02)
    ts = tl["times"][sel]
    assert len(ts) > 30 and np.all(np.diff(ts) > 0)
    interp = np.array([traj.qpos_at(t)[:3] for t in ts])
    snap = np.array([traj.q[np.argmin(np.abs(traj.t - t)), :3] for t in ts])
    # per-frame root displacement: speed changes slowly in flight, so consecutive steps should match closely
    di, ds = np.linalg.norm(np.diff(interp, axis=0), axis=1), np.linalg.norm(np.diff(snap, axis=0), axis=1)
    jitter_i = np.abs(np.diff(di)).max() / di.mean()
    jitter_s = np.abs(np.diff(ds)).max() / ds.mean()
    assert di.min() > 0, "no repeated poses"
    assert jitter_i < 0.06, f"interpolated motion judders: step-to-step change {jitter_i:.1%} of a step"
    assert jitter_s > 0.3, f"the snapping baseline should judder (test sanity), got {jitter_s:.1%}"


def test_quaternion_and_hinge_interpolation(capsule):
    model = mujoco.MjModel.from_xml_path(capsule["scene_xml"])
    traj = render._Trajectory.load(model, capsule["trajectory"])
    ev = json.loads(Path(capsule["events"]).read_text())
    i = int(np.searchsorted(traj.t, ev["t_apex"]))
    mid = 0.5 * (traj.t[i] + traj.t[i + 1])
    q = traj.qpos_at(mid)
    for adr in traj.quat:                                   # unit quaternions halfway between the samples
        assert abs(np.linalg.norm(q[adr:adr + 4]) - 1) < 1e-9
        a, b = traj.q[i, adr:adr + 4], traj.q[i + 1, adr:adr + 4]
        assert np.dot(q[adr:adr + 4], a) * np.dot(q[adr:adr + 4], b) > 0
    assert np.allclose(q[traj.lin], 0.5 * (traj.q[i, traj.lin] + traj.q[i + 1, traj.lin]))
    assert np.allclose(traj.qpos_at(traj.t[i]), traj.q[i]) and np.allclose(traj.qpos_at(-5), traj.q[0])


# ----------------------------------------------------------------------------- cameras
def _shot_at(case, camera, size, ts_list):
    model = render._load_model(case["scene_xml"], size)
    data = mujoco.MjData(model)
    traj = render._Trajectory.load(model, case["trajectory"])
    ev = json.loads(Path(case["events"]).read_text())
    shot = render._Shot(model, data, traj, case["plan"], camera, ev)
    perf = render._Performer(model, ev)
    out = []
    for ts in ts_list:
        cid = shot.apply(ts)
        data.qpos[:] = traj.qpos_at(ts)
        mujoco.mj_forward(model, data)
        out.append(seg_bbox(model, data, cid, perf.geoms, size))
    return shot, out


@pytest.mark.parametrize("camera", ["hero", "side"])
def test_tracking_camera_framing(capsule, camera):
    w, h = size = (640, 360)
    ev = json.loads(Path(capsule["events"]).read_text())
    upright = np.linspace(ev["t_runup"] - 0.4, ev["t_takeoff"] - 0.3, 6)
    action = np.linspace(ev["t_takeoff"], ev["t_landing"] + 1.0, 10)
    shot, boxes = _shot_at(capsule, camera, size, np.r_[upright, action])
    assert all(b is not None for b in boxes), "performer visible in every sampled frame"
    shares = [(b[3] - b[1]) / h for b in boxes[:len(upright)]]
    assert all(0.28 <= s <= 0.47 for s in shares), f"{camera}: standing performer height share {np.round(shares, 3)}"
    for b in boxes:
        assert b[0] > 2 and b[1] > 2 and b[2] < w - 2 and b[3] < h - 2, f"{camera}: performer cut by the frame edge {b}"
    r = ROOM                                                # the camera never leaves the room
    lo = np.array([r["x_min"], r["y_min"], 0.0]) + render.WALL_MARGIN - 1e-6
    hi = np.array([r["x_max"], r["y_max"], r["ceiling_z"]]) - render.WALL_MARGIN + 1e-6
    assert np.all(shot.kpos >= lo) and np.all(shot.kpos <= hi), f"{camera}: camera outside the room"
    assert np.all(np.abs(np.diff(shot.kpos, axis=0)).max(0) < 0.1), "camera moves smoothly (<10 cm per 10 ms)"


def test_hero_zooms_instead_of_cutting_through_walls(out_dir):
    """In a 3 m wide corridor the hero camera can't back off: it stays inside and widens fovy instead."""
    room = dict(ROOM, y_min=-1.6, y_max=1.6)
    case = build_capsule_case(out_dir / "capsule_narrow", room)
    ev = json.loads(Path(case["events"]).read_text())
    w, h = size = (640, 360)
    shot, boxes = _shot_at(case, "hero", size, np.linspace(ev["t_runup"], ev["t_takeoff"] - 0.3, 4))
    assert np.all(np.abs(shot.kpos[:, 1]) <= 1.6 - render.WALL_MARGIN + 1e-6)
    assert shot.kfovy.max() > render.RIGS["hero"]["fovy"] + 5, "fovy widened to compensate"
    assert all(b is not None and (b[3] - b[1]) / h >= 0.25 for b in boxes)


def test_framing_report(capsule):
    rep = render.framing_report(capsule, camera="hero", size=(640, 360))
    assert rep["framing_pct"] == 100.0 and rep["frames"] > 50
    assert 0.25 < rep["height_share"]["max"] < 0.6
    rep_match = render.framing_report(capsule, camera="match", size=(640, 360))
    assert 0 < rep_match["framing_pct"] <= 100


# ----------------------------------------------------------------------------- HUD
def test_hud_stays_in_its_corner():
    w, h = 1280, 720
    frame = np.full((h, w, 3), 90, np.uint8)
    hud = render._Hud((w, h), "FALLGUY . Wave 2 . w2c3")
    out = hud.draw(frame.copy(), 3.47, 3.12, 0.3, 9.2, 9.6)
    changed = np.any(out != frame, axis=2)
    ys, xs = np.nonzero(changed)
    assert xs.max() < w * 0.4 and ys.max() < h * 0.2, "HUD confined to the top-left corner"
    assert changed.mean() < 0.06
    assert not changed[h // 4:, :].any() and not changed[:, w // 2:].any()
    calm = hud.draw(frame.copy(), 1.0, 1.0, 1.0, 0.0, 0.0)  # no SLOW-MO badge at real time
    assert np.any(np.any(out != calm, axis=2)[:, int(xs.max() * 0.6):])
    count = lambda img, rgb: int(np.all(img == rgb, axis=2).sum())
    green = hud.draw(frame.copy(), 1.0, 1.0, 1.0, 2.0, 2.0)
    amber = hud.draw(frame.copy(), 1.0, 1.0, 1.0, 6.0, 6.0)
    red, ok, warn = (239, 68, 68), (34, 197, 94), (245, 158, 11)
    assert count(out, red) > 5 * count(green, red) > 0, "9.2 G fills the meter red (2 G: only the 8 G limit tick)"
    assert count(green, ok) > 50 and count(green, warn) == 0
    assert count(amber, warn) > 50 and count(amber, ok) == 0


# ----------------------------------------------------------------------------- video + stills
def _probe(mp4):
    rd = imageio.get_reader(mp4)
    meta = rd.get_meta_data()
    frames = [f for f in rd]
    rd.close()
    return meta, frames


@pytest.mark.parametrize("hud", [False, True])
def test_render_video_encodes_exact_frames(capsule, out_dir, hud):
    out = out_dir / f"capsule_hero_hud{int(hud)}.mp4"
    res = render.render_video(capsule, str(out), duration=3.0, fps=30, size=(321, 181), camera="hero", hud=hud)
    assert res["frames"] == 90 and res["size"] == [320, 180] and res["path"] == str(out)
    assert 0.25 <= res["slowmo_factor"] <= 1.0 and res["hud"] is hud
    meta, frames = _probe(str(out))
    assert len(frames) == 90 and frames[0].shape == (180, 320, 3)
    assert meta["codec"] == "h264" and meta["pix_fmt"].startswith("yuv420p") and abs(meta["fps"] - 30) < 1e-6
    nframes, secs = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert nframes == 90 and abs(secs - 3.0) < 0.05
    corner = np.mean([f[8:40, 8:120].astype(float).std() for f in frames[::10]])
    if hud:
        assert corner > 20, "HUD text visible in the corner"


def test_render_video_60fps_and_cameras(capsule, out_dir):
    for cam in ("side", "match", "overview"):
        res = render.render_video(capsule, str(out_dir / f"capsule_{cam}.mp4"), duration=2.0, fps=60,
                                  size=(320, 180), camera=cam)
        assert res["frames"] == 120 and res["camera"] == cam
    meta, frames = _probe(str(out_dir / "capsule_side.mp4"))
    assert len(frames) == 120 and abs(meta["fps"] - 60) < 1e-6
    with pytest.raises(ValueError, match="unknown camera"):
        render.render_video(capsule, str(out_dir / "nope.mp4"), 1.0, 30, (320, 180), camera="drone")


def test_render_still_and_errors(capsule, out_dir):
    with np.load(capsule["trajectory"]) as z:
        q0 = z["qpos"][0]
    for cam in ("match", "hero", "side", "overview"):
        png = render.render_still(capsule["scene_xml"], q0, capsule["plan"], str(out_dir / f"still_{cam}.png"), cam, (640, 360))
        img = np.asarray(Image.open(png))
        assert img.shape == (360, 640, 3) and img.std() > 10
    png = render.render_still(capsule["scene_xml"], None, {}, str(out_dir / "still_default.png"))
    assert Image.open(png).size == (1280, 720)
    with pytest.raises(FileNotFoundError):
        render.render_still(str(out_dir / "missing.xml"), None, {}, str(out_dir / "x.png"))
    with pytest.raises(ValueError, match="nq"):
        render.render_still(capsule["scene_xml"], np.zeros(3), {}, str(out_dir / "x.png"))
    bad = out_dir / "bad_traj.npz"
    np.savez(bad, t=np.arange(5) * DT, qpos=np.zeros((5, 4)))
    with pytest.raises(ValueError, match="same sim run"):
        render.render_video(dict(capsule, trajectory=str(bad)), str(out_dir / "bad.mp4"), 1.0, 30, (320, 180))


def test_composite_over(capsule, out_dir):
    with np.load(capsule["trajectory"]) as z:
        qpos, t = z["qpos"], z["t"]
    ev = json.loads(Path(capsule["events"]).read_text())
    q = qpos[np.searchsorted(t, ev["t_runup"] + 1.0)]
    ref = out_dir / "ref_frame.jpg"
    Image.fromarray(np.tile(np.linspace(40, 200, 800, dtype=np.uint8)[None, :, None], (600, 1, 3))).save(ref)
    png = render.composite_over(str(ref), capsule["scene_xml"], q, capsule["plan"], str(out_dir / "composite.png"))
    out, base = np.asarray(Image.open(png)).astype(int), np.asarray(Image.open(ref)).astype(int)
    assert out.shape == base.shape
    changed = np.any(np.abs(out - base) > 12, axis=2)
    assert 0.001 < changed.mean() < 0.2, "only the performer is pasted over the plate"
    mp4 = out_dir / "composite.mp4"                         # video mode: 4:3 plate -> 4:3 video at the given height
    res = render.render_video(dict(capsule, composite_ref=str(ref)), str(mp4), 1.5, 30, (640, 360), camera="match")
    assert res["composite"] and res["size"] == [480, 360] and res["frames"] == 45
    _, frames = _probe(str(mp4))
    plate = np.asarray(Image.open(ref).resize((480, 360))).astype(int)
    moved = [np.mean(np.any(np.abs(f.astype(int) - plate) > 25, axis=2)) for f in frames[::9]]
    assert all(0.0005 < m < 0.25 for m in moved), f"performer over the plate in every frame: {np.round(moved, 4)}"
    with pytest.raises(ValueError, match="composite_ref needs camera='match'"):
        render.render_video(dict(capsule, composite_ref=str(ref)), str(out_dir / "x.mp4"), 1.0, 30, (640, 360), "hero")


# ----------------------------------------------------------------------------- H1 venue: look + timing
def test_h1_venue_hero_timing_and_sheet(h1, out_dir):
    """9 s at 30 fps, 1280x720, hero camera + HUD on the full prototype venue: under 20 s."""
    out = out_dir / "h1_hero_hud.mp4"
    t0 = time.perf_counter()
    res = render.render_video(h1, str(out), duration=9.0, fps=30, size=(1280, 720), camera="hero", hud=True,
                              title="FALLGUY . Wave 2 . w2c3")
    wall = time.perf_counter() - t0
    assert res["frames"] == 270 and wall < 20.0, f"render took {wall:.1f} s"
    assert 0.25 <= res["slowmo_factor"] < 1.0
    nframes, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert nframes == 270
    contact_sheet(str(out), str(out_dir / "h1_hero_sheet.png"))
    rep = render.framing_report(h1, camera="hero")
    assert rep["framing_pct"] >= 95, rep


def test_h1_venue_side_and_match_sheets(h1, out_dir):
    for cam in ("side", "match"):
        out = out_dir / f"h1_{cam}.mp4"
        res = render.render_video(h1, str(out), duration=6.0, fps=30, size=(960, 540), camera=cam, hud=True)
        assert res["frames"] == 180
        contact_sheet(str(out), str(out_dir / f"h1_{cam}_sheet.png"), picks=8)
    with np.load(h1["trajectory"]) as z:
        q0 = z["qpos"][0]
    png = render.render_still(h1["scene_xml"], q0, h1["plan"], str(out_dir / "h1_preview_match.png"), "match")
    assert Image.open(png).size == (1280, 720)
