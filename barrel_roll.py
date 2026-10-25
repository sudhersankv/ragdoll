"""Build the hackathon scene from plan.json (made by argon_plan.py) in MuJoCo and render a
~9 s clip of a human doing a barrel roll over the shelf unit.

Usage: python barrel_roll.py [--cam three_quarter|front|side] [--out FILE]  (default renders/barrel_roll_<timestamp>.mp4) [--check]

The humanoid's torso is tied to a mocap body that follows the choreographed root path; the limbs are
driven by position actuators towards the pose keyframes (Argon's poses, clamped to joint limits).
The roll itself is shown in slow motion; physics always runs at the true timestep.
"""
import argparse
from datetime import datetime
import json
import math
import os

import imageio.v2 as imageio
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = 9.81
FPS = 60
VIDEO_SECONDS = 9.0
SLOWMO_RATE = 0.3          # sim seconds per video second around the flight
RUNUP = 2.2                # s of run-up before take-off
T_TAKEOFF = 1.0 + RUNUP    # s, sim time at which the feet leave the ground

HIP_BELOW = 0.1742         # Unitree H1: hip joints sit this far below the root (pelvis) body
THIGH, SHIN, FOOT = 0.40, 0.40, 0.07
H1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mujoco_menagerie", "unitree_h1")
ROOT_BODY = "pelvis"
JOINTS = {  # name: (min, max) degrees
    "l_shoulder_x": (-30, 170), "l_shoulder_y": (-60, 180), "l_elbow": (0, 140),
    "r_shoulder_x": (-30, 170), "r_shoulder_y": (-60, 180), "r_elbow": (0, 140),
    "l_hip_y": (-40, 120), "l_knee": (0, 140), "r_hip_y": (-40, 120), "r_knee": (0, 140),
}
JNAMES = list(JOINTS)

DEFAULT_POSES = {
    "stand": dict(l_shoulder_x=5, l_shoulder_y=0, l_elbow=10, r_shoulder_x=5, r_shoulder_y=0, r_elbow=10,
                  l_hip_y=0, l_knee=5, r_hip_y=0, r_knee=5),
    "crouch": dict(l_shoulder_x=10, l_shoulder_y=-30, l_elbow=45, r_shoulder_x=10, r_shoulder_y=-30, r_elbow=45,
                   l_hip_y=85, l_knee=95, r_hip_y=85, r_knee=95),
    "takeoff": dict(l_shoulder_x=15, l_shoulder_y=130, l_elbow=15, r_shoulder_x=15, r_shoulder_y=130, r_elbow=15,
                    l_hip_y=-15, l_knee=10, r_hip_y=-15, r_knee=10),
    "flight": dict(l_shoulder_x=10, l_shoulder_y=165, l_elbow=5, r_shoulder_x=10, r_shoulder_y=165, r_elbow=5,
                   l_hip_y=-10, l_knee=5, r_hip_y=-10, r_knee=5),
    "land": dict(l_shoulder_x=20, l_shoulder_y=45, l_elbow=35, r_shoulder_x=20, r_shoulder_y=45, r_elbow=35,
                 l_hip_y=65, l_knee=75, r_hip_y=65, r_knee=75),
}
CELEBRATE = dict(l_shoulder_x=35, l_shoulder_y=150, l_elbow=10, r_shoulder_x=35, r_shoulder_y=150, r_elbow=10,
                 l_hip_y=0, l_knee=5, r_hip_y=0, r_knee=5)


# ----------------------------------------------------------------------------- math helpers
def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def quat(axis, ang):
    a = np.asarray(axis, float)
    a /= np.linalg.norm(a)
    return np.r_[math.cos(ang / 2), a * math.sin(ang / 2)]


def qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def blend(p, q, u):
    return {k: p[k] + (q[k] - p[k]) * u for k in JNAMES}


def clamp_pose(pose):
    return {k: float(np.clip(pose.get(k, 0.0), *JOINTS[k])) for k in JNAMES}


def leg_drop(h, k):
    """Vertical distance from hip joint to the ground for a leg with hip flexion h, knee bend k (deg)."""
    h, k = math.radians(h), math.radians(k)
    return THIGH * math.cos(h) + SHIN * math.cos(h - k) + FOOT


def root_height(pose):
    """Torso-centre height that puts the lower foot on the floor."""
    drop = max(leg_drop(pose["l_hip_y"], pose["l_knee"]), leg_drop(pose["r_hip_y"], pose["r_knee"]))
    return HIP_BELOW + drop


# ----------------------------------------------------------------------------- choreography
class Choreo:
    def __init__(self, plan):
        c = plan["choreography"]
        sh = plan["shelf"]
        poses = {n: clamp_pose({**DEFAULT_POSES[n], **c.get("poses", {}).get(n, {})}) for n in DEFAULT_POSES}
        self.poses = poses
        prop_top = max([p["z"] + p["size"][1] for p in plan["props"]] + [sh["height"]])
        self.apex = max(c["apex_z"], prop_top + 0.85)        # torso centre must clear the tallest prop
        self.x_take = min(c["takeoff_x"], -sh["depth"] / 2 - 0.9)
        self.x_land = max(c["landing_x"], sh["depth"] / 2 + 1.5)
        self.z0 = root_height(poses["takeoff"])
        self.z1 = root_height(poses["land"])
        a, b = math.sqrt(2 * (self.apex - self.z0) / G), math.sqrt(2 * (self.apex - self.z1) / G)
        self.t_up, self.t_down = a, b
        self.T_f = a + b
        self.vx = (self.x_land - self.x_take) / self.T_f
        self.t_b = T_TAKEOFF + self.T_f
        self.x_start = self.x_take - 0.5 * self.vx * RUNUP
        self.gait_f = 2.3

    # --- joints
    def joints(self, t):
        P = self.poses
        if t < 1.0:
            return P["stand"]
        if t < T_TAKEOFF:
            u = (t - 1.0) / RUNUP
            amp = smooth(u / 0.5)
            ph = 2 * math.pi * self.gait_f * (t - 1.0)
            s = math.sin(ph)
            run = dict(P["stand"])
            a_h, a_arm = 38 * amp, 45 * amp
            run["l_hip_y"], run["r_hip_y"] = 10 * amp + a_h * s, 10 * amp - a_h * s
            run["l_knee"] = 15 + amp * (20 + 55 * max(0.0, math.sin(ph + 1.2)))
            run["r_knee"] = 15 + amp * (20 + 55 * max(0.0, -math.sin(ph + 1.2)))
            run["l_shoulder_y"], run["r_shoulder_y"] = -a_arm * s, a_arm * s
            run["l_elbow"] = run["r_elbow"] = 20 + 70 * amp
            if t < T_TAKEOFF - 0.45:
                return run
            if t < T_TAKEOFF - 0.12:
                return blend(run, P["crouch"], smooth((t - (T_TAKEOFF - 0.45)) / 0.33))
            return blend(P["crouch"], P["takeoff"], smooth((t - (T_TAKEOFF - 0.12)) / 0.12))
        if t < self.t_b:
            tau = (t - T_TAKEOFF) / self.T_f
            if tau < 0.3:
                return blend(P["takeoff"], P["flight"], smooth(tau / 0.3))
            if tau < 0.78:
                return P["flight"]
            return blend(P["flight"], P["land"], smooth((tau - 0.78) / 0.22))
        dt = t - self.t_b
        if dt < 0.25:
            return blend(P["land"], P["crouch"], smooth(dt / 0.25))
        if dt < 1.0:
            return blend(P["crouch"], P["stand"], smooth((dt - 0.25) / 0.75))
        if dt < 1.6:
            return P["stand"]
        return blend(P["stand"], CELEBRATE, smooth((dt - 1.6) / 0.6))

    # --- root pose
    def root(self, t):
        """Return (pos, quat) of the torso centre."""
        if t < T_TAKEOFF:
            u = max(0.0, (t - 1.0) / RUNUP)
            x = self.x_start + (self.x_take - self.x_start) * u * u
            speed = 2 * (self.x_take - self.x_start) * u / RUNUP
            lean = math.radians(12) * min(1.0, speed / max(self.vx, 1e-6))
            z = root_height(self.joints(t))
            if speed > 0.3:
                z -= 0.03 * abs(math.sin(2 * math.pi * self.gait_f * (t - 1.0)))   # run bob
            return np.array([x, 0.0, z]), quat([0, 1, 0], lean)
        if t < self.t_b:
            s = t - T_TAKEOFF
            tau = s / self.T_f
            x = self.x_take + self.vx * s
            if s < self.t_up:
                z = self.z0 + math.sqrt(2 * G * (self.apex - self.z0)) * s - 0.5 * G * s * s
            else:
                z = self.apex - 0.5 * G * (s - self.t_up) ** 2
            pitch = math.radians(90) * (smooth(tau / 0.3) - smooth((tau - 0.8) / 0.2))
            roll = 2 * math.pi * smooth((tau - 0.08) / 0.84)
            return np.array([x, 0.0, z]), qmul(quat([1, 0, 0], roll), quat([0, 1, 0], pitch))
        dt = t - self.t_b
        x = self.x_land + 0.6 * (1 - math.exp(-dt / 0.25))
        return np.array([x, 0.0, root_height(self.joints(t))]), quat([0, 1, 0], 0.0)


# ----------------------------------------------------------------------------- MJCF
def f3(v):
    return " ".join(f"{x:.4f}" for x in v)


KEEP_CLEAR = (-6.0, 3.2, -3.5, 3.5)     # x0, x1, y0, y1: run-up / flight / landing area, nothing may stand here


def _rot(x, y, yaw, ox, oy):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + ox * c - oy * s, y + ox * s + oy * c


def _in_clear_zone(x, y, r):
    x0, x1, y0, y1 = KEEP_CLEAR
    return x + r > x0 and x - r < x1 and y + r > y0 and y - r < y1


def env_geoms(env, sx):
    """Floor-to-ceiling surroundings from Argon's "environment" description: walls, curtains, sofas,
    chairs, seated people, speakers and ceiling pipes. Anything inside the keep-clear zone is dropped."""
    g = []
    box = lambda size, pos, rgb, yaw=0.0: g.append(
        f'<geom type="box" size="{f3(size)}" pos="{f3(pos)}" euler="0 0 {yaw:.2f}" rgba="{f3(rgb)} 1"/>')
    room = env["room"]
    x0, x1 = min(room["x_min"], -9.5), max(room["x_max"], sx + 1.0)
    y0, y1 = min(room["y_min"], -8.5), max(room["y_max"], 8.5)
    H = float(np.clip(room.get("ceiling_z", 4.5), 3.5, 8.0))
    wall = room.get("wall_rgb", (0.85, 0.85, 0.83))
    # walls (no ceiling slab: it would put the whole room in the key light's shadow)
    box((0.1, (y1 - y0) / 2 + 0.2, H / 2), ((x1 + 0.1), (y0 + y1) / 2, H / 2), wall)
    box(((x1 - x0) / 2 + 0.2, 0.1, H / 2), ((x0 + x1) / 2, y0 - 0.1, H / 2), wall)
    box(((x1 - x0) / 2 + 0.2, 0.1, H / 2), ((x0 + x1) / 2, y1 + 0.1, H / 2), wall)
    box((0.1, (y1 - y0) / 2 + 0.2, H / 2), (x0 - 0.1, (y0 + y1) / 2, H / 2), wall)
    # folded curtains along the back wall
    cu = env.get("curtain", {})
    pw = float(np.clip(cu.get("panel_width", 0.4), 0.25, 1.0))
    fd = float(np.clip(cu.get("fold_depth", 0.12), 0.05, 0.3))
    ch_ = min(float(cu.get("height", H - 0.2)), H - 0.05)
    crgb = cu.get("rgb", (0.9, 0.86, 0.78))
    for i in range(int((y1 - y0) / pw)):
        shade = 1.0 - 0.07 * (i % 2)
        box((0.05, pw / 2, ch_ / 2), (x1 - 0.1 - fd * (i % 2), y0 + (i + 0.5) * pw, ch_ / 2), [c * shade for c in crgb])
    # sofas
    for c in env.get("couches", []):
        L, D, sh_, bh = c["length"], c["depth"], c["seat_h"], c["back_h"]
        x, y, yaw = c["x"], c["y"], math.radians(c.get("yaw_deg", 0))
        if _in_clear_zone(x, y, max(L, D) / 2):
            continue
        d = math.degrees(yaw)
        box((D / 2, L / 2, sh_ / 2), (x, y, sh_ / 2), c["rgb"], d)
        bx, by = _rot(x, y, yaw, D / 2 - 0.1, 0)
        box((0.1, L / 2, bh / 2), (bx, by, bh / 2), [v * 0.95 for v in c["rgb"]], d)
        for side in (-1, 1):
            ax, ay = _rot(x, y, yaw, 0, side * (L / 2 - 0.07))
            box((D / 2, 0.07, (sh_ + 0.18) / 2), (ax, ay, (sh_ + 0.18) / 2), [v * 0.9 for v in c["rgb"]], d)
    # chairs (backrest toward +X when yaw = 0)
    for c in env.get("chairs", []):
        x, y, yaw = c["x"], c["y"], math.radians(c.get("yaw_deg", 0))
        if _in_clear_zone(x, y, 0.3):
            continue
        d, rgb = math.degrees(yaw), c.get("rgb", (0.1, 0.1, 0.1))
        box((0.22, 0.22, 0.015), (x, y, 0.45), rgb, d)
        bx, by = _rot(x, y, yaw, 0.21, 0)
        box((0.015, 0.22, 0.17), (bx, by, 0.65), rgb, d)
        for ox in (-0.19, 0.19):
            for oy in (-0.19, 0.19):
                lx, ly = _rot(x, y, yaw, ox, oy)
                box((0.015, 0.015, 0.225), (lx, ly, 0.225), rgb, d)
    # seated people (face -X when yaw = 0)
    for p in env.get("people", []):
        x, y, yaw, sz = p["x"], p["y"], math.radians(p.get("yaw_deg", 0)), p.get("seat_z", 0.45)
        if _in_clear_zone(x, y, 0.5):
            continue
        W = lambda ox, oy, z: " ".join(f"{v:.3f}" for v in (*_rot(x, y, yaw, ox, oy), z))
        shirt = f3(p.get("rgb", (0.12, 0.12, 0.15)))
        g.append(f'<geom type="capsule" fromto="{W(0,0,sz+0.15)} {W(0,0,sz+0.5)}" size="0.13" rgba="{shirt} 1"/>')
        g.append(f'<geom type="sphere" size="0.1" pos="{W(-0.02,0,sz+0.72)}" rgba="0.85 0.68 0.55 1"/>')
        for side in (-0.1, 0.1):
            g.append(f'<geom type="capsule" fromto="{W(0,side,sz+0.1)} {W(-0.4,side,sz+0.1)}" size="0.06" rgba="0.15 0.16 0.2 1"/>')
            g.append(f'<geom type="capsule" fromto="{W(-0.4,side,sz+0.1)} {W(-0.4,side,0.06)}" size="0.05" rgba="0.15 0.16 0.2 1"/>')
    # speakers on stands
    for s in env.get("speakers", []):
        x, y = s["x"], s["y"]
        if _in_clear_zone(x, y, 0.4):
            continue
        zb, w, h = s.get("z_bottom", 1.0), s.get("width", 0.35), s.get("height", 0.6)
        g.append(f'<geom type="cylinder" size="0.02 {zb/2:.3f}" pos="{x:.3f} {y:.3f} {zb/2:.3f}" rgba="0.05 0.05 0.05 1"/>')
        g.append(f'<geom type="cylinder" size="0.22 0.015" pos="{x:.3f} {y:.3f} 0.015" rgba="0.05 0.05 0.05 1"/>')
        box((w * 0.4, w / 2, h / 2), (x, y, zb + h / 2), (0.06, 0.06, 0.07))
    # ceiling pipes / beams
    for f in env.get("ceiling_fixtures", [])[:12]:
        px, py, pz = f["pos"]
        px, py, pz = float(np.clip(px, x0 + 0.5, x1 - 0.5)), float(np.clip(py, y0 + 0.5, y1 - 0.5)), float(np.clip(pz, 3.0, H - 0.05))
        half, rad, rgb = float(np.clip(f.get("length", 4), 1, 20)) / 2, float(np.clip(f.get("radius", 0.08), 0.03, 0.3)), f.get("rgb", (0.1, 0.1, 0.1))
        axis = f.get("axis", "y")
        if f.get("kind") == "beam":
            sz_ = (half, rad, rad) if axis == "x" else (rad, half, rad)
            box(sz_, (px, py, pz), rgb)
        else:
            eu = "0 90 0" if axis == "x" else "90 0 0"
            g.append(f'<geom type="cylinder" size="{rad:.3f} {half:.3f}" pos="{px:.3f} {py:.3f} {pz:.3f}" euler="{eu}" rgba="{f3(rgb)} 1"/>')
    return g



H1_JOINTS = [f"{side}_{j}" for side in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")] + \
            ["torso"] + \
            [f"{side}_{j}" for side in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
# position-servo gains (kp, kv); force limits come from the H1 motor ctrlranges
GAINS = {"hip_yaw": (400, 25), "hip_roll": (400, 25), "hip_pitch": (500, 30), "knee": (500, 30), "ankle": (120, 8),
         "torso": (400, 25), "shoulder_pitch": (120, 8), "shoulder_roll": (120, 8), "shoulder_yaw": (40, 3), "elbow": (40, 3)}
FORCE = {"hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 40, "torso": 200,
         "shoulder_pitch": 40, "shoulder_roll": 40, "shoulder_yaw": 18, "elbow": 18}
LINK_COLORS = {"torso_link": "0.30 0.33 0.40 1", "pelvis": "0.22 0.24 0.28 1"}   # arms are tinted below so the roll reads


def h1_xml(pos, q):
    """Unitree H1 from MuJoCo Menagerie (models/mujoco_menagerie/unitree_h1).
    Returns (default-class xml, mesh/material asset xml, root body xml). Joint ranges are converted to degrees
    to match this scene's compiler settings, and the collision proxies are switched off."""
    root = ET.parse(os.path.join(H1_DIR, "h1.xml")).getroot()
    dflt = root.find("default/default[@class='h1']")
    dflt.find("default[@class='collision']/geom").set("contype", "0")
    dflt.find("default[@class='collision']/geom").set("conaffinity", "0")
    assets = "".join(ET.tostring(e, encoding="unicode") for e in root.find("asset"))
    pelvis = root.find("worldbody/body[@name='pelvis']")
    pelvis.set("pos", f3(pos))
    pelvis.set("quat", f3(q))
    for j in pelvis.iter("joint"):
        if j.get("range"):
            j.set("range", " ".join(f"{math.degrees(float(v)):.3f}" for v in j.get("range").split()))
    for g in pelvis.iter("geom"):
        mesh = g.get("mesh", "")
        if mesh.startswith("left_shoulder") or mesh == "left_elbow_link":
            g.set("rgba", "0.95 0.50 0.15 1")
        elif mesh.startswith("right_shoulder") or mesh == "right_elbow_link":
            g.set("rgba", "0.20 0.55 0.95 1")
        elif mesh in LINK_COLORS or mesh.replace(".stl", "") in LINK_COLORS:
            g.set("rgba", LINK_COLORS[mesh])
        elif mesh and mesh != "logo_link":
            g.set("rgba", "0.16 0.17 0.20 1")
    return ET.tostring(dflt, encoding="unicode"), assets, ET.tostring(pelvis, encoding="unicode")


class H1Rig:
    """Maps the abstract pose (shoulder_x/y, elbow, hip_y, knee, in degrees) onto the Unitree H1's joints.
    Everything is analytic: legs are pitch joints; the arms convert my abduction-then-flexion shoulder
    into H1's pitch-then-roll-then-yaw shoulder."""

    def __init__(self, model_or_none=None):
        root = ET.parse(os.path.join(H1_DIR, "h1.xml")).getroot()
        self.rng = {j.get("name"): np.array([float(v) for v in j.get("range").split()])
                    for j in root.iter("joint") if j.get("range")}

    @staticmethod
    def _Rx(a):
        return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])

    @staticmethod
    def _Ry(a):
        return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])

    def _arm(self, side, pose):
        sgn = 1.0 if side == "left" else -1.0
        phi, sy, el = (math.radians(pose[f"{side[0]}_{n}"]) for n in ("shoulder_x", "shoulder_y", "elbow"))
        R = self._Rx(sgn * phi) @ self._Ry(-sy)                    # my arm frame
        u = R @ np.array([0.0, 0.0, -1.0])                         # upper-arm direction
        th_r = math.asin(float(np.clip(u[1], -1, 1)))
        th_p = math.atan2(-u[0], -u[2])
        Rh = self._Ry(th_p) @ self._Rx(th_r)                       # H1 pitch then roll
        M = Rh.T @ R                                               # remaining twist about the arm axis
        th_y = math.atan2(M[1, 0], M[0, 0])
        lo, hi = self.rng[f"{side}_shoulder_yaw"]
        for k in (-2 * math.pi, 2 * math.pi):
            if not lo <= th_y <= hi and lo <= th_y + k <= hi:
                th_y += k
                break
        return {f"{side}_shoulder_pitch": th_p, f"{side}_shoulder_roll": th_r, f"{side}_shoulder_yaw": th_y,
                f"{side}_elbow": math.pi / 2 - el}            # H1 elbow: 90 deg = straight arm

    def targets(self, pose):
        row = {}
        for side in ("left", "right"):
            h, k = pose[f"{side[0]}_hip_y"], pose[f"{side[0]}_knee"]
            row[f"{side}_hip_pitch"] = -math.radians(h)          # +pitch swings the leg backwards
            row[f"{side}_knee"] = math.radians(k)
            row[f"{side}_ankle"] = math.radians(h - k)           # keeps the sole level
            row.update(self._arm(side, pose))
        return row

    def ctrl_for(self, pose):
        row = self.targets(pose)
        return np.array([float(np.clip(row.get(n, 0.0), *self.rng[n])) for n in H1_JOINTS])


def build_xml(plan, ch):
    sh, sc = plan["shelf"], plan["screen"]
    L, D, H = sh["length"], sh["depth"], sh["height"]
    levels = sh["board_levels"]
    g = []   # scene geoms
    wood = "0.78 0.62 0.42 1"
    # boards + posts
    for z in levels:
        g.append(f'<geom type="box" size="{D/2:.3f} {L/2:.3f} 0.015" pos="0 0 {z-0.015:.3f}" rgba="{wood}"/>')
    n_posts = max(2, int(round(L / sh.get("post_spacing", 0.8))) + 1)
    for i in range(n_posts):
        y = -L / 2 + i * L / (n_posts - 1)
        for dx in (-D / 2 + 0.02, D / 2 - 0.02):
            g.append(f'<geom type="box" size="0.02 0.02 {H/2:.3f}" pos="{dx:.3f} {y:.3f} {H/2:.3f}" rgba="{wood}"/>')
    # props
    for p in plan["props"]:
        x, y, z = p["x"], p["y"], p["z"]
        r, h = p["size"]
        rgba = f'{f3(p["rgb"])} 1'
        k = p["kind"]
        if k == "plant":
            g.append(f'<geom type="cylinder" size="{r:.3f} {h*0.17:.3f}" pos="{x:.3f} {y:.3f} {z+h*0.17:.3f}" rgba="0.85 0.86 0.85 1"/>')
            g.append(f'<geom type="ellipsoid" size="{r*1.5:.3f} {r*1.5:.3f} {h*0.33:.3f}" pos="{x:.3f} {y:.3f} {z+h*0.65:.3f}" rgba="{rgba}"/>')
        elif k == "vase":
            g.append(f'<geom type="capsule" size="{r:.3f} {max(h/2-r,0.01):.3f}" pos="{x:.3f} {y:.3f} {z+h/2:.3f}" rgba="{rgba}"/>')
        elif k == "book":
            g.append(f'<geom type="box" size="{r:.3f} {r*1.3:.3f} {h/2:.3f}" pos="{x:.3f} {y:.3f} {z+h/2:.3f}" rgba="{rgba}"/>')
        elif k == "bowl":
            g.append(f'<geom type="cylinder" size="{r:.3f} {max(h/2,0.01):.3f}" pos="{x:.3f} {y:.3f} {z+h/2:.3f}" rgba="0.12 0.1 0.1 1"/>')
        else:   # candle, can
            g.append(f'<geom type="cylinder" size="{r:.3f} {h/2:.3f}" pos="{x:.3f} {y:.3f} {z+h/2:.3f}" rgba="{rgba}"/>')
    # screen + curtains
    sx = max(sc["x"], 5.0)
    sw, sht = sc["width"], sc["height"]
    g.append(f'<geom type="box" size="0.05 {sw/2:.3f} {sht/2:.3f}" pos="{sx:.3f} {sc["y"]:.3f} {sc["bottom_z"]+sht/2:.3f}" material="screen"/>')
    g.append(f'<geom type="box" size="0.1 {sw*0.28:.3f} {sht*0.04:.3f}" pos="{sx-0.1:.3f} {sc["y"]-sw*0.15:.3f} {sc["bottom_z"]+sht*0.6:.3f}" rgba="0.95 0.95 1 1"/>')
    for sy_ in (sc["y"] - sw * 0.35, sc["y"] + sw * 0.35):      # legs the screen stands on
        g.append(f'<geom type="box" size="0.04 0.04 {(sc["bottom_z"]+0.2)/2:.3f}" pos="{sx:.3f} {sy_:.3f} {(sc["bottom_z"]+0.2)/2:.3f}" rgba="0.08 0.08 0.1 1"/>')
    env = plan.get("environment")
    if env:
        g += env_geoms(env, sx)
        floor_rgb = env["room"].get("floor_rgb", (0.56, 0.55, 0.53))
    else:
        floor_rgb = (0.56, 0.55, 0.53)
        for i in range(-30, 31):
            g.append(f'<geom type="box" size="0.12 0.2 3.0" pos="{sx+0.6:.3f} {i*0.42:.3f} 3.0" rgba="0.90 0.86 0.78 1"/>')
    scene_geoms = "\n    ".join(g)

    pos0, q0 = ch.root(0.0)
    hum_default, hum_assets, hum_body = h1_xml(pos0, q0)
    acts = "\n    ".join(
        f'<position name="a_{j}" joint="{j}" kp="{GAINS[j.split("_", 1)[1] if j != "torso" else "torso"][0]}" '
        f'kv="{GAINS[j.split("_", 1)[1] if j != "torso" else "torso"][1]}" '
        f'forcerange="-{FORCE[j.split("_", 1)[1] if j != "torso" else "torso"]} {FORCE[j.split("_", 1)[1] if j != "torso" else "torso"]}"/>'
        for j in H1_JOINTS)
    return f'''
<mujoco model="argon_barrel_roll">
  <compiler meshdir="{os.path.join(H1_DIR, 'assets')}"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="52"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.5 0.5 0.53" diffuse="0.15 0.15 0.15" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.95" rgb2="0.55 0.58 0.65" width="256" height="256"/>
    <texture name="screen_tex" type="2d" builtin="gradient" rgb1="0.05 0.15 0.9" rgb2="0.02 0.03 0.25" width="256" height="256"/>
    <material name="screen" texture="screen_tex" emission="0.8" specular="0.2"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="{f3(floor_rgb)}" rgb2="{f3([c * 0.93 for c in floor_rgb])}" width="256" height="256" mark="random" markrgb="{f3([c * 0.8 for c in floor_rgb])}" random="0.08"/>
    <material name="floor" texture="floor_tex" texrepeat="12 12" texuniform="true" reflectance="0.03"/>
    {hum_assets}
  </asset>
  <default>
    {hum_default}
  </default>
  <worldbody>
    <light directional="true" pos="-3 -4 6" dir="0.4 0.5 -1" diffuse="0.42 0.42 0.4" castshadow="true"/>
    <light directional="true" pos="3 4 5" dir="0.5 0.2 -0.3" diffuse="0.15 0.15 0.18" castshadow="false"/>
    <geom type="plane" size="60 60 0.1" material="floor"/>
    <body name="focus" pos="-0.8 0 1.0"/>
    <camera name="three_quarter" pos="-5.4 -5.6 2.6" mode="targetbody" target="focus"/>
    <camera name="front" pos="-8.0 0.0 1.5" mode="targetbody" target="focus"/>
    <camera name="side" pos="-0.8 -7.5 1.5" mode="targetbody" target="focus"/>
    <body name="scene" pos="0 0 0">
    {scene_geoms}
    </body>
    <body name="mocap_root" mocap="true" pos="{f3(pos0)}" quat="{f3(q0)}"/>
    {hum_body}
  </worldbody>
  <equality>
    <weld body1="mocap_root" body2="pelvis" relpose="0 0 0 1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    {acts}
  </actuator>
</mujoco>'''


# ----------------------------------------------------------------------------- run
def sim_times(ch, seconds=VIDEO_SECONDS):
    """Sim time of each video frame: real-time except for slow motion around the flight."""
    lo, hi = T_TAKEOFF - 0.25, ch.t_b + 0.25
    def rate(t):
        up = smooth((t - lo) / 0.3)
        dn = smooth((hi - t) / 0.3)
        return 1.0 + (SLOWMO_RATE - 1.0) * min(up, dn)
    ts, out = 0.0, []
    for _ in range(int(seconds * FPS)):
        out.append(ts)
        ts += rate(ts) / FPS
    return out


def clearance_check(model, data, ch, times, rig):
    """Smallest distance between the human and any shelf/prop geom during the flight."""
    scene_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "scene")
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    hum = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0 and model.geom_bodyid[i] != scene_body
           and model.body_rootid[model.geom_bodyid[i]] == torso]
    obst = [i for i in range(model.ngeom) if model.geom_bodyid[i] == scene_body and abs(data.geom_xpos[i][0]) < 1.0
            and data.geom_xpos[i][2] < 2.5 and model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
    best = (9.0, None)
    fromto = np.zeros(6)
    for t in times:
        if not (T_TAKEOFF - 0.1 <= t <= ch.t_b + 0.1):
            continue
        step_to(model, data, ch, t, rig)
        for a in hum:
            for b in obst:
                d = mujoco.mj_geomDistance(model, data, a, b, 0.5, fromto)
                if d < best[0]:
                    best = (d, (round(t, 2), model.geom_bodyid[a], b))
    return best


def step_to(model, data, ch, t_target, rig):
    while data.time < t_target:
        pos, q = ch.root(data.time)
        data.mocap_pos[0], data.mocap_quat[0] = pos, q
        data.ctrl[:] = rig.ctrl_for(ch.joints(data.time))
        mujoco.mj_step(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="three_quarter")
    ap.add_argument("--out", default=None, help="default: renders/barrel_roll_<timestamp>.mp4")
    ap.add_argument("--check", action="store_true", help="only report clearance, don't render")
    ap.add_argument("--size", default="1280x720")
    args = ap.parse_args()
    if args.out is None:
        os.makedirs(os.path.join(HERE, "renders"), exist_ok=True)
        args.out = os.path.join(HERE, "renders", f"barrel_roll_{datetime.now():%Y%m%d_%H%M%S}.mp4")

    plan = json.load(open(os.path.join(HERE, "plan.json")))
    ch = Choreo(plan)
    print(f"apex={ch.apex:.2f} m  flight={ch.T_f:.2f} s  vx={ch.vx:.2f} m/s  run-up from x={ch.x_start:.2f}")
    xml = build_xml(plan, ch)
    open(os.path.join(HERE, "scene.xml"), "w").write(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    times = sim_times(ch)
    rig = H1Rig()

    def reset():
        mujoco.mj_resetData(model, data)
        pos, q = ch.root(0.0)
        data.qpos[:3], data.qpos[3:7] = pos, q
        for n, v in zip(H1_JOINTS, rig.ctrl_for(ch.joints(0.0))):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
        mujoco.mj_forward(model, data)

    reset()
    d, where = clearance_check(model, data, ch, times, rig)
    print(f"min human-to-shelf/prop distance during flight: {d*100:.1f} cm  {where}")
    if args.check:
        return
    reset()
    w, h = map(int, args.size.split("x"))
    renderer = mujoco.Renderer(model, h, w)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    worst = 0.0
    with imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as wr:
        for t in times:
            step_to(model, data, ch, t, rig)
            worst = max(worst, float(np.linalg.norm(data.xpos[torso] - ch.root(data.time)[0])))
            renderer.update_scene(data, camera=args.cam)
            wr.append_data(renderer.render())
    print(f"wrote {args.out}: {len(times)/FPS:.1f} s, torso tracking error max {worst*100:.1f} cm")


if __name__ == "__main__":
    main()
