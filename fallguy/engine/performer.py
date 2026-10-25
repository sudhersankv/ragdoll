"""The performer: the Unitree H1 from MuJoCo Menagerie, composed as in barrel_roll.h1_xml.

compose_h1() returns the MJCF fragments scene.build_xml() splices into the stunt scene:
- the H1 body tree with the prototype's colours (dark grey body, orange left arm, blue right arm, so the
  rotation reads on video);
- the collision proxies kept (class "collision", group 3, invisible). The sim switches them off for the
  kinematic run-up and on from take-off;
- a head IMU site + accelerometer on torso_link (the H1 has no neck: the head is part of torso_link) and a
  pelvis site;
- PD position servos on all 19 joints (prototype gains, force limits = the H1 motor limits).
H1Rig maps the prototype's abstract 10-angle pose (degrees) onto H1 joint targets and has the sagittal
leg IK the run-up uses to keep the feet planted.
"""
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT_BODY = "pelvis"
HEAD_SITE, HEAD_SITE_POS = "head_imu", (0.045, 0.0, 0.68)     # centre of the H1 helmet sphere
HEAD_SENSOR, PELVIS_SITE = "head_acc", "pelvis_site"
H1_JOINTS = [f"{side}_{j}" for side in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")] + \
            ["torso"] + \
            [f"{side}_{j}" for side in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
LEG_JOINTS = [j for j in H1_JOINTS if "hip" in j or "knee" in j or "ankle" in j] + ["torso"]
# position-servo gains (kp, kv) from barrel_roll.py; force limits are the H1 motor ctrlranges
GAINS = {"hip_yaw": (400, 25), "hip_roll": (400, 25), "hip_pitch": (500, 30), "knee": (500, 30), "ankle": (120, 8),
         "torso": (400, 25), "shoulder_pitch": (120, 8), "shoulder_roll": (120, 8), "shoulder_yaw": (40, 3), "elbow": (40, 3)}
FORCE = {"hip_yaw": 200, "hip_roll": 200, "hip_pitch": 200, "knee": 300, "ankle": 40, "torso": 200,
         "shoulder_pitch": 40, "shoulder_roll": 40, "shoulder_yaw": 18, "elbow": 18}
LINK_COLORS = {"torso_link": "0.30 0.33 0.40 1", "pelvis": "0.22 0.24 0.28 1"}
LEFT_ARM_RGBA, RIGHT_ARM_RGBA, BODY_RGBA = "0.95 0.50 0.15 1", "0.20 0.55 0.95 1", "0.16 0.17 0.20 1"

# leg geometry (h1.xml): hip-pitch joint centre in the pelvis frame, link lengths, flat-foot ankle height
HIP_PITCH_OFFSET = (0.039468, 0.0875 + 0.11536, -0.1742)
THIGH, SHIN = 0.4, 0.4
ANKLE_HEIGHT = 0.070          # ankle joint above the floor when the sole is flat (foot capsules end at -0.056 - 0.014)
TOE_X, HEEL_X = 0.154, -0.049  # sole extent along the foot, from the ankle
STAND_PELVIS_Z = ANKLE_HEIGHT + THIGH + SHIN - HIP_PITCH_OFFSET[2]   # 1.0442 m, straight legs
REACH_MAX, KNEE_MAX = 0.7985, math.radians(115.0)   # 0.7985 m = knee bent 7 deg


def _gain_key(joint):
    return "torso" if joint == "torso" else joint.split("_", 1)[1]


# ----------------------------------------------------------------------------- asset lookup
def _skill_dir():
    return Path(__file__).resolve().parents[2]


def _candidates(rel, env_var):
    out = []
    if os.environ.get(env_var):
        out.append(Path(os.path.expanduser(os.environ[env_var])))
    out.append(_skill_dir() / rel)
    out.append(_skill_dir().parents[2] / rel)
    out.append(Path.cwd() / rel)
    out.append(Path.home() / ".cache" / "fallguy" / Path(rel).relative_to("models"))
    return out


def find_h1_dir():
    """Unitree H1 assets: $FALLGUY_H1_DIR, <repo>/models/mujoco_menagerie/unitree_h1 (repo = 3 levels above the
    skill dir), ./models/mujoco_menagerie/unitree_h1, ~/.cache/fallguy/mujoco_menagerie/unitree_h1."""
    tried = _candidates("models/mujoco_menagerie/unitree_h1", "FALLGUY_H1_DIR")
    for d in tried:
        if (d / "h1.xml").is_file() and (d / "assets").is_dir():
            return str(d.resolve())
    raise FileNotFoundError("Unitree H1 model not found (need h1.xml + assets/). Tried: " + ", ".join(map(str, tried))
                            + ". Set FALLGUY_H1_DIR or run `git submodule update --init models/mujoco_menagerie`.")


def check_model(name):
    """The performer is always the Unitree H1."""
    if name in (None, "", "h1"):
        return "h1"
    raise ValueError(f"unknown performer.model {name!r}: the performer is always the Unitree H1 ('h1')")


# ----------------------------------------------------------------------------- MJCF composition
def _f3(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def compose_h1(pos, quat):
    """MJCF fragments for the H1 standing at root pose (pos, quat). Returns a dict with keys
    default, assets, body, actuators, sensors, contact, meshdir."""
    h1_dir = find_h1_dir()
    root = ET.parse(os.path.join(h1_dir, "h1.xml")).getroot()
    if root.find("compiler").get("angle") != "radian":
        raise ValueError("expected h1.xml with compiler angle='radian'")
    dflt = root.find("default/default[@class='h1']")
    coll = dflt.find("default[@class='collision']/geom")
    coll.set("contype", "1")
    coll.set("conaffinity", "1")
    coll.set("priority", "0")
    assets = "".join(ET.tostring(e, encoding="unicode") for e in root.find("asset"))
    pelvis = root.find("worldbody/body[@name='pelvis']")
    pelvis.set("pos", _f3(pos))
    pelvis.set("quat", _f3(quat))
    for g in pelvis.iter("geom"):
        mesh = g.get("mesh", "")
        if mesh.startswith("left_shoulder") or mesh == "left_elbow_link":
            g.set("rgba", LEFT_ARM_RGBA)
        elif mesh.startswith("right_shoulder") or mesh == "right_elbow_link":
            g.set("rgba", RIGHT_ARM_RGBA)
        elif mesh in LINK_COLORS:
            g.set("rgba", LINK_COLORS[mesh])
        elif mesh and mesh != "logo_link":
            g.set("rgba", BODY_RGBA)
    torso = pelvis.find(".//body[@name='torso_link']")
    ET.SubElement(torso, "site", name=HEAD_SITE, pos=_f3(HEAD_SITE_POS), size="0.01")
    ET.SubElement(pelvis, "site", name=PELVIS_SITE, pos="0 0 0", size="0.01")
    acts = "\n    ".join(
        f'<position name="a_{j}" joint="{j}" kp="{GAINS[_gain_key(j)][0]}" kv="{GAINS[_gain_key(j)][1]}" '
        f'forcerange="-{FORCE[_gain_key(j)]} {FORCE[_gain_key(j)]}"/>' for j in H1_JOINTS)
    sensors = (f'<accelerometer name="{HEAD_SENSOR}" site="{HEAD_SITE}"/>\n    '
               f'<gyro name="head_gyro" site="{HEAD_SITE}"/>\n    '
               f'<accelerometer name="pelvis_acc" site="{PELVIS_SITE}"/>\n    '
               f'<subtreecom name="com" body="{ROOT_BODY}"/>\n    '
               f'<subtreelinvel name="com_vel" body="{ROOT_BODY}"/>\n    '
               f'<subtreeangmom name="angmom" body="{ROOT_BODY}"/>')
    contact = "".join(ET.tostring(e, encoding="unicode") for e in root.find("contact"))
    return {"default": ET.tostring(dflt, encoding="unicode"), "assets": assets,
            "body": ET.tostring(pelvis, encoding="unicode"), "actuators": acts, "sensors": sensors,
            "contact": contact, "meshdir": os.path.join(h1_dir, "assets")}


class Performer:
    """Ids and handles of the H1 inside a compiled stunt model."""

    def __init__(self, model):
        import mujoco
        self.mujoco = mujoco
        B = mujoco.mjtObj
        self.root = mujoco.mj_name2id(model, B.mjOBJ_BODY, ROOT_BODY)
        if self.root < 0:
            raise ValueError(f"performer root body {ROOT_BODY!r} not in the model")
        self.bodies = [b for b in range(model.nbody) if model.body_rootid[b] == self.root]
        bodyset = set(self.bodies)
        self.geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodyset]
        self.collision_geoms = np.array([g for g in self.geoms if model.geom_group[g] == 3], int)
        self.visual_geoms = np.array([g for g in self.geoms if model.geom_group[g] != 3], int)
        self.free_qadr = int(model.jnt_qposadr[model.body_jntadr[self.root]])
        self.free_vadr = int(model.jnt_dofadr[model.body_jntadr[self.root]])
        jid = [mujoco.mj_name2id(model, B.mjOBJ_JOINT, j) for j in H1_JOINTS]
        if min(jid) < 0:
            raise ValueError("the model is missing H1 joints")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid], int)
        self.vadr = np.array([model.jnt_dofadr[j] for j in jid], int)
        self.act = np.array([mujoco.mj_name2id(model, B.mjOBJ_ACTUATOR, f"a_{j}") for j in H1_JOINTS], int)
        sid = mujoco.mj_name2id(model, B.mjOBJ_SENSOR, HEAD_SENSOR)
        self.head_adr = int(model.sensor_adr[sid])
        self.head_site = mujoco.mj_name2id(model, B.mjOBJ_SITE, HEAD_SITE)
        self.kp0 = model.actuator_gainprm[self.act, 0].copy()
        self.kv0 = -model.actuator_biasprm[self.act, 2].copy()
        self.leg_mask = np.array([j in LEG_JOINTS for j in H1_JOINTS])
        self.jrange = model.jnt_range[jid].copy()
        self.mass = float(model.body_subtreemass[self.root])

    def set_collisions(self, model, on):
        model.geom_contype[self.collision_geoms] = 1 if on else 0
        model.geom_conaffinity[self.collision_geoms] = 1 if on else 0

    def set_leg_gain_scale(self, model, s):
        """Scale leg + torso stiffness by s (damping by sqrt(s), keeping the damping ratio)."""
        kp = self.kp0 * np.where(self.leg_mask, s, 1.0)
        kv = self.kv0 * np.where(self.leg_mask, math.sqrt(max(s, 0.0)), 1.0)
        model.actuator_gainprm[self.act, 0] = kp
        model.actuator_biasprm[self.act, 1] = -kp
        model.actuator_biasprm[self.act, 2] = -kv

    def com_inertia(self, model, data):
        """(com, 3x3 inertia about the com in world axes) of the whole performer for the current kinematics."""
        b = np.array(self.bodies)
        m = model.body_mass[b]
        com = (m[:, None] * data.xipos[b]).sum(0) / m.sum()
        R = data.ximat[b].reshape(-1, 3, 3)
        I = np.einsum("nij,nj,nkj->ik", R, model.body_inertia[b], R)
        r = data.xipos[b] - com
        I += np.einsum("n,nij->ij", m, (r * r).sum(1)[:, None, None] * np.eye(3) - r[:, :, None] * r[:, None, :])
        return com, I


# ----------------------------------------------------------------------------- pose mapping
def _Rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _Ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


POSE_KEYS = ["l_shoulder_x", "l_shoulder_y", "l_elbow", "r_shoulder_x", "r_shoulder_y", "r_elbow",
             "l_hip_y", "l_knee", "r_hip_y", "r_knee"]
POSE_EXTRA = ["torso", "l_hip_x", "r_hip_x", "l_ankle", "r_ankle"]   # optional: torso yaw, hip abduction, ankle offset


class H1Rig:
    """Maps the abstract pose (degrees) onto H1 joint targets (radians), as barrel_roll.H1Rig does.

    Pose keys: {l,r}_shoulder_x (abduction), {l,r}_shoulder_y (flexion: 90 forward, 180 overhead),
    {l,r}_elbow (bend), {l,r}_hip_y (thigh flexion forward), {l,r}_knee (bend); optional torso (yaw),
    {l,r}_hip_x (abduction) and {l,r}_ankle (offset from a sole parallel to the pelvis)."""

    def __init__(self):
        root = ET.parse(os.path.join(find_h1_dir(), "h1.xml")).getroot()
        self.rng = {j.get("name"): np.array([float(v) for v in j.get("range").split()])
                    for j in root.iter("joint") if j.get("range")}
        self.lo = np.array([self.rng[j][0] for j in H1_JOINTS])
        self.hi = np.array([self.rng[j][1] for j in H1_JOINTS])

    def _arm(self, side, pose):
        sgn = 1.0 if side == "left" else -1.0
        phi, sy, el = (math.radians(pose.get(f"{side[0]}_{n}", 0.0)) for n in ("shoulder_x", "shoulder_y", "elbow"))
        R = _Rx(sgn * phi) @ _Ry(-sy)
        u = R @ np.array([0.0, 0.0, -1.0])
        th_r = math.asin(float(np.clip(u[1], -1, 1)))
        th_p = math.atan2(-u[0], -u[2])
        M = (_Ry(th_p) @ _Rx(th_r)).T @ R
        th_y = math.atan2(M[1, 0], M[0, 0])
        lo, hi = self.rng[f"{side}_shoulder_yaw"]
        for k in (-2 * math.pi, 2 * math.pi):
            if not lo <= th_y <= hi and lo <= th_y + k <= hi:
                th_y += k
                break
        return {f"{side}_shoulder_pitch": th_p, f"{side}_shoulder_roll": th_r, f"{side}_shoulder_yaw": th_y,
                f"{side}_elbow": math.pi / 2 - el}          # H1 elbow: 90 deg = straight arm

    def targets(self, pose):
        row = {"torso": math.radians(pose.get("torso", 0.0))}
        for side in ("left", "right"):
            s = side[0]
            h, k = pose.get(f"{s}_hip_y", 0.0), pose.get(f"{s}_knee", 0.0)
            row[f"{side}_hip_pitch"] = -math.radians(h)          # +pitch swings the leg backwards
            row[f"{side}_knee"] = math.radians(k)
            row[f"{side}_ankle"] = math.radians(h - k + pose.get(f"{s}_ankle", 0.0))   # sole parallel to the pelvis
            row[f"{side}_hip_roll"] = math.radians(pose.get(f"{s}_hip_x", 0.0)) * (1.0 if side == "left" else -1.0)
            row[f"{side}_hip_yaw"] = 0.0
            row.update(self._arm(side, pose))
        return row

    def vector(self, pose):
        """Abstract pose -> joint target vector in H1_JOINTS order, clipped to the joint ranges."""
        row = self.targets(pose)
        return np.clip(np.array([row[n] for n in H1_JOINTS]), self.lo, self.hi)

    def arm_vector(self, pose, out):
        """Overwrite the arm and torso entries of a joint vector from an abstract pose."""
        row = self.targets(pose)
        for i, n in enumerate(H1_JOINTS):
            if "shoulder" in n or "elbow" in n or n == "torso":
                out[i] = min(max(row[n], self.lo[i]), self.hi[i])
        return out


def leg_ik(hip_xz, ankle_xz, pelvis_pitch, foot_pitch=0.0):
    """Sagittal 2-link IK for one H1 leg (hip yaw and roll at 0).
    hip_xz, ankle_xz: world (x, z) of the hip-pitch joint and the ankle joint; pelvis_pitch: pelvis rotation
    about +Y (forward lean > 0); foot_pitch: wanted sole pitch in the world (toes down > 0).
    Returns (hip_pitch, knee, ankle) in radians and the distance the ankle target was out of reach (>= 0)."""
    dxw, dzw = ankle_xz[0] - hip_xz[0], ankle_xz[1] - hip_xz[1]
    c, s = math.cos(pelvis_pitch), math.sin(pelvis_pitch)
    x, z = c * dxw - s * dzw, s * dxw + c * dzw              # into the pelvis frame
    d0 = math.hypot(x, z)
    d_min = math.sqrt(THIGH ** 2 + SHIN ** 2 - 2 * THIGH * SHIN * math.cos(math.pi - KNEE_MAX))
    d = min(max(d0, d_min), REACH_MAX)
    knee = math.pi - math.acos((THIGH ** 2 + SHIN ** 2 - d * d) / (2 * THIGH * SHIN))
    alpha = math.acos(min(1.0, (THIGH ** 2 + d * d - SHIN ** 2) / (2 * THIGH * d)))
    phi = math.atan2(x, -z)
    hip = -(phi + alpha)
    ankle = foot_pitch - pelvis_pitch - hip - knee
    return hip, knee, ankle, max(0.0, d0 - REACH_MAX)


def hip_world_xz(pelvis_xz, pelvis_pitch):
    """World (x, z) of a hip-pitch joint for a pelvis at pelvis_xz pitched by pelvis_pitch."""
    ox, oz = HIP_PITCH_OFFSET[0], HIP_PITCH_OFFSET[2]
    c, s = math.cos(pelvis_pitch), math.sin(pelvis_pitch)
    return pelvis_xz[0] + c * ox + s * oz, pelvis_xz[1] - s * ox + c * oz


def max_pelvis_z(ankle_xz, pelvis_x, pelvis_pitch, reach=REACH_MAX - 0.002):
    """Highest pelvis z that still lets a leg reach an ankle at ankle_xz."""
    ox, oz = HIP_PITCH_OFFSET[0], HIP_PITCH_OFFSET[2]
    c, s = math.cos(pelvis_pitch), math.sin(pelvis_pitch)
    hx = pelvis_x + c * ox + s * oz
    dx = ankle_xz[0] - hx
    if abs(dx) >= reach:
        return ankle_xz[1] - (-s * ox + c * oz)
    return ankle_xz[1] + math.sqrt(reach * reach - dx * dx) - (-s * ox + c * oz)


# ----------------------------------------------------------------------------- 3D leg IK (yaw + roll + sagittal)
HIP_YAW_POS = (0.0, 0.0875, -0.1742)     # hip-yaw joint in the pelvis frame (left; y mirrored for the right)
HIP_ROLL_X = 0.039468                    # roll joint ahead of the yaw joint
HIP_PITCH_Y = 0.11536                    # pitch joint outboard of the roll joint
FOOT_Y = HIP_YAW_POS[1] + HIP_PITCH_Y    # 0.2029: a straight leg's ankle lies this far out from the pelvis centre
HIP_YAW_MAX = HIP_ROLL_MAX = 0.43


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def leg_ik3(pelvis_pos, R_pelvis, side, ankle, foot_yaw, toe_pitch=0.0):
    """Analytic IK of one H1 leg for an ankle target in the world.
    hip yaw turns the leg toward the foot's yaw (clipped to the joint range), hip roll puts the ankle in the
    leg's sagittal plane, then the 2-link sagittal IK (leg_ik) sets hip pitch and knee, and the ankle keeps the
    sole level in the world (toes down by toe_pitch). Returns ({joint: angle}, out-of-reach distance)."""
    s = 1.0 if side == "left" else -1.0
    P, R = np.asarray(pelvis_pos, float), np.asarray(R_pelvis, float)
    psi = math.atan2(R[1, 0], R[0, 0])
    q_yaw = min(max(_wrap(foot_yaw - psi), -HIP_YAW_MAX), HIP_YAW_MAX)
    R1 = R @ _rz(q_yaw)
    roll_joint = P + R @ np.array([HIP_YAW_POS[0], s * HIP_YAW_POS[1], HIP_YAW_POS[2]]) + R1 @ np.array([HIP_ROLL_X, 0, 0])
    d = R1.T @ (np.asarray(ankle, float) - roll_joint)
    rho = math.hypot(d[1], d[2])
    alpha = math.atan2(d[2], d[1])
    beta = math.acos(min(1.0, max(-1.0, s * HIP_PITCH_Y / max(rho, 1e-6))))
    if rho < 1.3 * HIP_PITCH_Y:                               # folded onto the hip: no stable roll solution
        q_roll = 0.0
    else:
        q_roll = min((_wrap(alpha + beta), _wrap(alpha - beta)), key=abs)
        q_roll = min(max(q_roll, -HIP_ROLL_MAX), HIP_ROLL_MAX)
    c, sn = math.cos(q_roll), math.sin(q_roll)
    dx, dz = d[0], -sn * d[1] + c * d[2]                     # into the rolled frame (Rx(roll)^T d)
    hip, knee, _, err = leg_ik((0.0, 0.0), (dx, min(dz, -0.05)), 0.0, 0.0)
    R0 = R1 @ np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
    tot = math.atan2(R0[2, 0], R0[2, 2]) + toe_pitch         # total sagittal pitch of the sole frame
    return ({f"{side}_hip_yaw": q_yaw, f"{side}_hip_roll": q_roll, f"{side}_hip_pitch": hip,
             f"{side}_knee": knee, f"{side}_ankle": tot - hip - knee}, err)


def hip_pitch_joint(pelvis_pos, R_pelvis, side):
    """World position of a hip-pitch joint with hip yaw and roll at zero."""
    s = 1.0 if side == "left" else -1.0
    return np.asarray(pelvis_pos, float) + np.asarray(R_pelvis, float) @ np.array([HIP_ROLL_X, s * FOOT_Y, HIP_YAW_POS[2]])
