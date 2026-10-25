"""FALLGUY renderer: scene.xml + trajectory.npz + events.json -> stills and slow-motion mp4 clips.

render_video    retimed clip (real-time run-up, slow-motion flight, real-time settle) from the hero, side or
                match camera, with an optional HUD; H.264 mp4 through the ffmpeg bundled with imageio-ffmpeg.
render_still    one frame at a given qpos (scene previews).
retime          sim time of every output frame.
framing_report  share of frames with the performer inside the camera frame (geometry only, no GL).
composite_over  the performer matted over the real reference frame, seen from the match camera.

Imports only mujoco, numpy, Pillow and imageio-ffmpeg, so it works with any scene.xml + trajectory.npz pair.
Cameras (the `camera` argument):
  hero   smoothed 3/4 tracking dolly on the performer root; the standing performer fills ~38% of the frame
         height; it stays inside the walls (zooming out instead of backing through them).
  side   the same dolly, looking perpendicular to the lane.
  match  the scene's `match` camera if it has one, else plan.camera.match {pos, lookat, fovy[, roll_deg]}.
  any other name: that camera from scene.xml.
"""
import argparse
import json
import math
import os
import subprocess
import tempfile
import time

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("hero", "side", "match")

# ----------------------------------------------------------------------------- tuning
SLOW_BEFORE, SLOW_AFTER = 0.25, 0.35   # s: slow motion covers [t_takeoff - 0.25, t_landing + 0.35]
RAMP = 0.2                             # s of sim time to ease in and out of slow motion, inside that window
SLOW_MIN, SLOW_MAX = 0.25, 1.0         # slow-motion factor range (sim seconds per video second)
PRE_ROLL = 0.5                         # s of real time shown before the run-up starts
SETTLE = 1.5                           # s of real time shown after the slow-motion window

TRACK_DT, TRACK_SIGMA = 0.01, 0.3      # s: camera keyframe spacing, Gaussian smoothing of the root path
RIGS = {  # azimuth: direction from the performer to the camera, deg from +X (-90 = the performer's right)
    "hero": {"azimuths": (-50.0, 50.0, -130.0, 130.0), "elevation": 12.0, "fovy": 40.0, "share": 0.38},
    "side": {"azimuths": (-90.0, 90.0), "elevation": 8.0, "fovy": 40.0, "share": 0.36},
}
AIM_FOLLOW_Z = 0.65                    # share of the root's rise and fall that the camera aim follows
WALL_MARGIN, PROP_MARGIN = 0.35, 0.25  # m the camera keeps from walls / ceiling and from furniture
BIG_HALF = 1.2                         # m: a static geom with two half-extents this big is a wall
OCCLUDER_MIN = 2.5                     # m: furniture blocking the view pulls the camera in, but not closer
MIN_RANGE = 0.5                        # m: the camera never gets closer to its aim point
MAX_FOVY = 75.0
RENDER_CAM = "fallguy_render"          # camera added to every loaded scene and posed per frame

G_GREEN, G_AMBER, G_FULL = 4.0, 8.0, 12.0   # head-G meter: green < 4, amber < 8, red >= 8; full scale 12 G


# ----------------------------------------------------------------------------- small helpers
def _load_json(x, what):
    """A dict, or the path of a JSON file holding one."""
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if not os.path.isfile(x):
        raise FileNotFoundError(f"{what} not found: {x}")
    with open(x) as f:
        return json.load(f)


def _even_size(size):
    """(W, H) from a tuple or 'WxH', rounded down to even numbers (yuv420p needs them)."""
    if isinstance(size, str):
        size = size.lower().split("x")
    w, h = (int(v) for v in size)
    if w < 16 or h < 16:
        raise ValueError(f"size {w}x{h} is too small")
    return w - w % 2, h - h % 2


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def _gauss(x, sigma):
    """Gaussian smoothing along axis 0 (sigma in samples), edges padded with the end values."""
    x = np.asarray(x, float)
    if sigma < 0.5 or len(x) < 3:
        return x.copy()
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(pad, k, mode="valid")
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(x.shape[1])], axis=1)


def _minfilter(x, r):
    """Running minimum over a window of 2r+1 samples."""
    if r < 1 or len(x) < 2:
        return np.asarray(x, float).copy()
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, r, mode="edge"), 2 * r + 1).min(axis=1)


def _slerp(q0, q1, a):
    n0, n1 = np.linalg.norm(q0), np.linalg.norm(q1)
    if n0 < 1e-12 or n1 < 1e-12:
        return q0 + a * (q1 - q0)
    q0, q1 = q0 / n0, q1 / n1
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def _look_quat(forward, roll_deg=0.0):
    """MuJoCo camera quaternion looking along `forward` with world-up (+Z) up, rolled by roll_deg."""
    f = np.asarray(forward, float)
    if np.linalg.norm(f) < 1e-9:
        raise ValueError("camera pos and lookat coincide")
    f = f / np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:            # looking straight up or down
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        x, y = c * x + s * y, -s * x + c * y
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, -f]).ravel())
    return q


def _world_boxes(model, data, geoms):
    """World-axis-aligned boxes (centres, half-sizes) of the given geoms, from their local geom_aabb."""
    geoms = np.asarray(geoms, int)
    if geoms.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    R = data.geom_xmat[geoms].reshape(-1, 3, 3)
    c = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", R, model.geom_aabb[geoms, :3])
    h = np.einsum("nij,nj->ni", np.abs(R), model.geom_aabb[geoms, 3:])
    return c, h


def _subtree_geoms(model, root):
    bodies = {int(root)}
    for b in range(int(root) + 1, model.nbody):      # parents always precede children
        if int(model.body_parentid[b]) in bodies:
            bodies.add(b)
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def _scene_option():
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0                  # IMU and marker sites stay invisible
    return opt


# ----------------------------------------------------------------------------- timeline
def _event_times(events):
    """(t_start, t_runup, t_takeoff, t_landing, t_end) from events.json, checked."""
    ev = _load_json(events, "events")

    def num(key, default=None):
        v = ev.get(key, default)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"events.{key} must be a number, got {v!r}") from None

    tk, te = num("t_takeoff"), num("t_end")
    if tk is None or te is None:
        raise ValueError(f"events need t_takeoff and t_end; got keys {sorted(ev)}")
    ts0 = num("t_start", 0.0)
    tr = num("t_runup", ts0)
    tl = num("t_landing")
    if tl is None or tl <= tk:            # no landing recorded: treat the second after take-off as the flight
        tl = tk + 1.0
    if not ts0 <= tk <= te:
        raise ValueError(f"events out of order: t_start {ts0}, t_takeoff {tk}, t_end {te}")
    return ts0, min(max(tr, ts0), tk), tk, min(tl, te), te


def _rate(t, a, b, s):
    """Sim seconds per video second at sim time t: s inside [a, b] (eased over RAMP at both ends), else 1."""
    t = np.asarray(t, float)
    return 1.0 + (s - 1.0) * np.minimum(_smoothstep((t - a) / RAMP), _smoothstep((b - t) / RAMP))


def _video_map(t0, t1, a, b, s):
    """(sim-time grid over [t0, t1], video time at each grid point)."""
    t = np.linspace(t0, t1, max(2001, int((t1 - t0) * 2000) + 1))
    inv = 1.0 / _rate(t, a, b, s)
    return t, np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(t))])


def _video_len(t0, t1, a, b, s):
    return float(_video_map(t0, t1, a, b, s)[1][-1])


def _timeline(events, duration, fps):
    """Frame plan: sim time and playback rate of each of round(duration * fps) frames, plus the choices made."""
    fps = float(fps)
    n = int(round(float(duration) * fps)) if fps > 0 else 0
    if n < 2:
        raise ValueError(f"need at least 2 frames, got duration={duration} s at fps={fps}")
    ts0, tr, tk, tl, te = _event_times(events)
    a, b = tk - SLOW_BEFORE, tl + SLOW_AFTER
    target = (n - 1) / fps                                   # video seconds from the first to the last frame
    t0, t1 = max(ts0, tr - PRE_ROLL), min(te, b + SETTLE)
    if t1 <= t0:
        t1 = te
    if _video_len(t0, t1, a, b, SLOW_MAX) >= target:        # too long even in real time: trim
        s = SLOW_MAX
        excess = (t1 - t0) - target
        cut = min(excess, max(0.0, (tk - 0.6) - t0))         # the start of the run-up first ...
        t0, excess = t0 + cut, excess - cut
        cut = min(excess, max(0.0, t1 - (tl + 0.6)))         # ... then the settle
        t1, excess = t1 - cut, excess - cut
        if excess > 1e-9:                                    # very short clip: centre it on the flight
            t0 = max(ts0, min(0.5 * (tk + tl) - target / 2, te - target))
            t1 = t0 + target
    elif _video_len(t0, t1, a, b, SLOW_MIN) <= target:      # too short even at the slowest: extend, then hold
        s = SLOW_MIN
        missing = target - _video_len(t0, t1, a, b, s)       # outside the slow window video time == sim time
        post = min(te - t1, missing / 2)
        pre = min(t0 - ts0, missing - post)
        post = min(te - t1, missing - pre)
        t0, t1 = t0 - pre, t1 + post
    else:                                                    # solve for the factor that fills the clip
        lo, hi = SLOW_MIN, SLOW_MAX
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if _video_len(t0, t1, a, b, mid) > target else (lo, mid)
        s = 0.5 * (lo + hi)
    grid, v = _video_map(t0, t1, a, b, s)
    vt = np.arange(n) / fps
    times = np.interp(vt, v, grid)                           # past the end: hold the last frame
    held = vt > v[-1] + 1e-6
    rates = _rate(times, a, b, s)
    rates[held] = 0.0
    return {"times": times, "rates": rates, "factor": float(round(s, 4)), "t0": float(t0), "t1": float(t1),
            "slow_window": (float(a), float(b)), "hold_frames": int(held.sum()), "frames": n, "fps": fps}


def retime(events, duration, fps):
    """Sim time of each of round(duration * fps) output frames: real time from just before the run-up to
    take-off, slow motion over [t_takeoff - 0.25, t_landing + 0.35] at the factor that fills `duration`
    (0.25-1.0), real time for the settle; held last frame if the sim is too short."""
    return _timeline(events, duration, fps)["times"]


# ----------------------------------------------------------------------------- scene, trajectory, performer
def _load_model(scene_xml, size):
    """Compile scene.xml plus a posable render camera, with an offscreen buffer of at least `size`."""
    if not scene_xml or not os.path.isfile(scene_xml):
        raise FileNotFoundError(f"scene.xml not found: {scene_xml}")
    try:
        spec = mujoco.MjSpec.from_file(os.path.abspath(scene_xml))
        spec.worldbody.add_camera(name=RENDER_CAM, pos=[0.0, 0.0, 2.0], fovy=45.0)
        if not spec.lights:                                  # a scene with no light gets a shadow-casting key
            spec.worldbody.add_light(name="fallguy_key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                     pos=[0.0, 0.0, 8.0], dir=[0.35, 0.45, -1.0], diffuse=[0.6, 0.6, 0.6],
                                     specular=[0.1, 0.1, 0.1], castshadow=True)
        model = spec.compile()
    except Exception as e:
        raise RuntimeError(f"cannot load {scene_xml}: {e}") from e
    w, h = size
    vg, vq = model.vis.global_, model.vis.quality
    vg.offwidth, vg.offheight = max(int(vg.offwidth), w), max(int(vg.offheight), h)
    vq.offsamples = max(int(vq.offsamples), 4)               # MSAA
    vq.shadowsize = max(int(vq.shadowsize), 4096)
    active = model.light_active.astype(bool)
    if active.any() and not model.light_castshadow[active].any():   # make sure something casts a shadow
        k = int(np.argmax(model.light_diffuse.sum(1) * active))
        model.light_castshadow[k] = 1
    return model


def _tune_shadows(model, data, plan, center):
    """Fit the directional shadow maps to the room around `center` (x, y). MuJoCo centres a directional
    light's shadow box on the light's pos with half-size shadowclip * extent; the extent of a venue with a
    large floor is ~36 m, which gives ~2 cm shadow texels and jagged edges. Call before making the Renderer."""
    xs, ys = [], []                                          # the plan's room and every static geom
    room = (plan or {}).get("room") or {}
    if all(k in room for k in ("x_min", "x_max", "y_min", "y_max")):
        xs, ys = [float(room["x_min"]), float(room["x_max"])], [float(room["y_min"]), float(room["y_max"])]
    mujoco.mj_kinematics(model, data)
    statics = [g for g in range(model.ngeom) if model.body_weldid[model.geom_bodyid[g]] == 0
               and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    if statics:
        c, h = _world_boxes(model, data, statics)
        lo, hi = (c - h).min(0), (c + h).max(0)
        xs, ys = xs + [lo[0], hi[0]], ys + [lo[1], hi[1]]
    if not xs:
        return
    cx, cy = float(center[0]), float(center[1])
    reach = max(6.0, max(math.hypot(x - cx, y - cy) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))) + 2.0)
    ext = float(model.stat.extent)
    if reach >= ext * model.vis.map.shadowclip:
        return
    model.vis.map.shadowclip = reach / ext
    for k in range(model.nlight):
        if (model.light_type[k] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL and model.light_castshadow[k]
                and model.light_bodyid[k] == 0 and model.light_mode[k] == mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
            d = model.light_dir[k] / max(np.linalg.norm(model.light_dir[k]), 1e-9)
            model.light_pos[k] = np.array([cx, cy, 0.0]) - d * (reach + 10.0)


def _action_center(model, data, traj, events, plan):
    """(x, y) centre of the performer's path, else of the plan's stunt lane."""
    try:
        perf = _Performer(model, events)
        p = perf.root_path(model, data, traj, np.linspace(traj.t[0], traj.t[-1], 60))
        return 0.5 * (p[:, :2].min(0) + p[:, :2].max(0))
    except ValueError:
        st = (plan or {}).get("stunt") or {}
        return np.array([0.5 * (float(st.get("start_x", 0.0)) + float(st.get("landing_x", 0.0))),
                         float(st.get("lane_y", 0.0))])


class _Trajectory:
    """trajectory.npz (t, qpos[, head_g15]) with joint-aware interpolation between physics samples."""

    def __init__(self, model, t, qpos, head_g=None, source="trajectory"):
        t = np.asarray(t, float).ravel()
        q = np.asarray(qpos, float)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[0] != t.size or t.size == 0:
            raise ValueError(f"{source}: qpos shape {q.shape} does not match t shape {t.shape}")
        if q.shape[1] != model.nq:
            raise ValueError(f"{source}: qpos has {q.shape[1]} columns but the scene has nq={model.nq}; "
                             "scene.xml and trajectory.npz must come from the same sim run")
        g = None if head_g is None else np.asarray(head_g, float).ravel()
        if g is not None and g.size != t.size:
            raise ValueError(f"{source}: head_g15 has {g.size} samples but t has {t.size}")
        if t.size == 1:                                      # a single pose: hold it
            t, q = np.r_[t, t + 1.0], np.vstack([q, q])
            g = None if g is None else np.r_[g, g]
        keep = np.r_[True, np.diff(t) > 1e-12]               # drop repeated time stamps
        t, q = t[keep], q[keep]
        g = None if g is None else g[keep]
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"{source}: t must increase")
        self.t, self.q, self.g = t, q, g
        lin, quat = [], []
        for j in range(model.njnt):
            adr, typ = int(model.jnt_qposadr[j]), model.jnt_type[j]
            if typ == mujoco.mjtJoint.mjJNT_FREE:
                lin += [adr, adr + 1, adr + 2]
                quat.append(adr + 3)
            elif typ == mujoco.mjtJoint.mjJNT_BALL:
                quat.append(adr)
            else:
                lin.append(adr)
        self.lin, self.quat = np.array(lin, int), quat

    @classmethod
    def load(cls, model, path):
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"trajectory not found: {path}")
        with np.load(path) as z:
            missing = [k for k in ("t", "qpos") if k not in z.files]
            if missing:
                raise ValueError(f"{path} lacks arrays {missing}; it has {z.files}")
            return cls(model, z["t"], z["qpos"], z["head_g15"] if "head_g15" in z.files else None, source=path)

    def qpos_at(self, ts):
        """qpos at sim time ts: positions and hinges linear, free/ball quaternions slerped."""
        t = self.t
        ts = min(max(float(ts), t[0]), t[-1])
        i = int(np.clip(np.searchsorted(t, ts, side="right") - 1, 0, len(t) - 2))
        a = (ts - t[i]) / (t[i + 1] - t[i])
        q0, q1 = self.q[i], self.q[i + 1]
        out = q0.copy()
        out[self.lin] = q0[self.lin] + a * (q1[self.lin] - q0[self.lin])
        for adr in self.quat:
            out[adr:adr + 4] = _slerp(q0[adr:adr + 4], q1[adr:adr + 4], a)
        return out

    def g_max(self, ta, tb):
        """Largest head_g15 over (ta, tb], so short impacts between frames still show; None if absent."""
        if self.g is None:
            return None
        val = float(np.interp(tb, self.t, self.g))
        i0, i1 = np.searchsorted(self.t, [min(ta, tb), tb], side="right")
        return max(val, float(self.g[i0:i1].max())) if i1 > i0 else val


class _Performer:
    """The performer: root body, rendered geoms, free-joint address."""

    def __init__(self, model, events=None):
        name = (events or {}).get("performer_root_body") or "pelvis"
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if root <= 0:
            free = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
            if not free:
                raise ValueError(f"performer root body {name!r} is not in the scene and no body has a free joint")
            root = max(free, key=lambda b: len(_subtree_geoms(model, b)))
        self.root = root
        geoms = _subtree_geoms(model, root)
        shown = [g for g in geoms if model.geom_group[g] < 3 and model.geom_rgba[g, 3] > 0]
        self.geoms = np.array(shown or geoms, int)
        j = int(model.body_jntadr[root])
        free = j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        self.free_adr = int(model.jnt_qposadr[j]) if free else None

    def root_path(self, model, data, traj, times):
        """(n, 3) root positions at the given sim times."""
        if self.free_adr is not None:
            return np.stack([np.interp(times, traj.t, traj.q[:, self.free_adr + i]) for i in range(3)], axis=1)
        out = []
        for ts in times:
            data.qpos[:] = traj.qpos_at(ts)
            mujoco.mj_kinematics(model, data)
            out.append(data.xpos[self.root].copy())
        return np.array(out)

    def extent(self, model, data):
        """World (min corner, max corner) of the performer's geom boxes for the current data."""
        c, h = _world_boxes(model, data, self.geoms)
        return (c - h).min(0), (c + h).max(0)


class _Room:
    """Where a camera may stand: in front of walls / ceiling, inside the plan's room box, outside furniture;
    and what stands between a camera and the performer."""

    RAY = {int(t) for t in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH)}   # ints: np.int32 in (enum,) is False

    def __init__(self, model, data, plan):
        self.model, self.data = model, data
        self.blockers, boxes = [], []
        for g in range(model.ngeom):
            if model.body_weldid[model.geom_bodyid[g]] != 0:             # moves: performer, loose props
                continue
            if model.geom_group[g] >= 3 or model.geom_rgba[g, 3] == 0:   # not rendered
                continue
            typ = model.geom_type[g]
            if typ == mujoco.mjtGeom.mjGEOM_PLANE:
                if data.geom_xmat[g][8] < 0.9:                           # a wall plane; the floor is free space
                    self.blockers.append(g)
            elif int(typ) in self.RAY and np.sum(model.geom_aabb[g, 3:] >= BIG_HALF) >= 2:
                self.blockers.append(g)                                  # walls, ceilings, big structures
            else:
                boxes.append(g)
        c, h = _world_boxes(model, data, boxes)
        self.box_lo, self.box_hi = c - h, c + h                          # furniture, for occlusion tests
        self.lo, self.hi = c - h - PROP_MARGIN, c + h + PROP_MARGIN      # where the camera may not stand
        room = (plan or {}).get("room") or {}
        self.box = None
        if all(k in room for k in ("x_min", "x_max", "y_min", "y_max")):
            top = float(room.get("ceiling_z") or 50.0)
            self.box = (np.array([room["x_min"] + WALL_MARGIN, room["y_min"] + WALL_MARGIN, 0.3], float),
                        np.array([room["x_max"] - WALL_MARGIN, room["y_max"] - WALL_MARGIN, top - WALL_MARGIN], float))

    def first_hit(self, centre, half, u, start=0.3):
        """Smallest s >= start at which the box (centre, half), swept along u, touches a furniture box (else
        inf): a ray from the centre against furniture boxes grown by `half` (Minkowski sum of two AABBs)."""
        if not len(self.box_lo):
            return math.inf
        v = np.where(np.abs(u) < 1e-9, 1e-9, u)
        t1, t2 = (self.box_lo - half - centre) / v, (self.box_hi + half - centre) / v
        tmin, tmax = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
        hit = (tmax >= tmin) & (tmin >= start)
        return float(tmin[hit].min()) if hit.any() else math.inf

    def hidden(self, cam, points, skip=0.2):
        """Share of `points` hidden from `cam` by furniture boxes (the last `skip` m before each point ignored)."""
        if not len(self.box_lo) or not len(points):
            return 0.0
        v = np.asarray(points, float) - cam                              # (P, 3) segments camera -> point
        L = np.linalg.norm(v, axis=1)
        v = np.where(np.abs(v) < 1e-9, 1e-9, v)
        t1 = (self.box_lo[None] - cam) / v[:, None]                      # (P, B, 3) slab entry / exit
        t2 = (self.box_hi[None] - cam) / v[:, None]
        tmin = np.minimum(t1, t2).max(axis=2)
        tmax = np.maximum(t1, t2).min(axis=2)
        end = np.maximum(1.0 - skip / np.maximum(L, 1e-9), 0.0)[:, None]
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin <= end)
        return float(hit.any(axis=1).mean())

    def limit(self, p, u, want, body=None):
        """Largest distance <= want from aim point p along unit vector u where the camera may stand.
        body: (centre, half) box of the performer; furniture in front of it pulls the camera in."""
        m, d, nrm = self.model, float(want), np.zeros(3)
        for g in self.blockers:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                s = mujoco.mj_rayMesh(m, self.data, g, p, u, nrm)
            else:
                s = mujoco.mju_rayGeom(self.data.geom_xpos[g], self.data.geom_xmat[g], m.geom_size[g], p, u,
                                       m.geom_type[g], nrm)
            if s >= 0:
                cos = abs(float(np.dot(u, nrm))) if np.linalg.norm(nrm) > 0 else 1.0
                d = min(d, s - WALL_MARGIN / max(cos, 0.3))
        if self.box is not None:                              # the plan's room (aim may be below 0.3 m: lying)
            lo, hi = self.box
            if lo[0] < p[0] < hi[0] and lo[1] < p[1] < hi[1] and p[2] < hi[2]:
                for i in range(3):
                    if u[i] > 1e-9:
                        d = min(d, (hi[i] - p[i]) / u[i])
                    elif u[i] < -1e-9:
                        d = min(d, max(0.0, (lo[i] - p[i]) / u[i]))
        centre, half = (p, np.zeros(3)) if body is None else body   # furniture in the line of sight: move in front
        occ = float((centre - p) @ u) + self.first_hit(centre, half, u) - PROP_MARGIN
        if occ >= OCCLUDER_MIN:
            d = min(d, occ)
        while d > MIN_RANGE and self.inside(p + d * u):
            d -= 0.1
        return max(d, MIN_RANGE)

    def inside(self, x):
        return bool(np.any(np.all((x >= self.lo) & (x <= self.hi), axis=1))) if len(self.lo) else False


def _match_pose(plan):
    """(pos, lookat, fovy, roll_deg) from plan.camera.match, else a view down the lane from behind the start."""
    plan = plan or {}
    cam = (plan.get("camera") or {}).get("match") or {}
    if cam.get("pos") is not None and cam.get("lookat") is not None:
        pos, look = np.asarray(cam["pos"], float).ravel(), np.asarray(cam["lookat"], float).ravel()
        if pos.size != 3 or look.size != 3:
            raise ValueError(f"plan.camera.match pos/lookat must be [x, y, z], got {cam}")
        return pos, look, float(cam.get("fovy") or 50.0), float(cam.get("roll_deg") or 0.0)
    st = plan.get("stunt") or {}
    y, x0, xt = float(st.get("lane_y", 0.0)), float(st.get("start_x", -4.0)), float(st.get("takeoff_x", -1.0))
    return np.array([x0 - 2.5, y, 1.6]), np.array([xt, y, 0.9]), 50.0, 0.0


class _Shot:
    """The camera of one shot: a scene camera, the plan's match camera, or a hero / side tracking dolly.
    apply(ts) poses the render camera for sim time ts and returns the camera id to render with."""

    def __init__(self, model, data, traj, plan, camera, events=None):
        self.model, self.camera = model, (camera or "hero")
        self.cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RENDER_CAM)
        self.scene_cam, self.azimuth = None, None
        cam_id = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.camera in RIGS:
            self._track(data, traj, _Performer(model, events), plan, RIGS[self.camera], events)
        elif self.camera == "match" and cam_id("match") < 0:
            pos, look, fovy, roll = _match_pose(plan)
            self.kt, self.kpos, self.kaim = np.array([0.0]), pos[None, :], look[None, :]
            self.kfovy, self.quat = np.array([fovy]), _look_quat(look - pos, roll)
        else:
            if cam_id(self.camera) < 0:
                names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
                raise ValueError(f"unknown camera {self.camera!r}: use one of {CAMERAS} or a scene camera "
                                 f"{[n for n in names if n and n != RENDER_CAM]}")
            self.scene_cam = cam_id(self.camera)

    def _track(self, data, traj, perf, plan, rig, events=None):
        m = self.model
        t = traj.t
        kt = np.arange(t[0], t[-1] + 0.5 * TRACK_DT, TRACK_DT) if t[-1] > t[0] else t[:1].copy()
        root = perf.root_path(m, data, traj, kt)
        sig = TRACK_SIGMA / TRACK_DT
        sm = _gauss(root, sig)
        data.qpos[:] = traj.qpos_at(t[0])
        mujoco.mj_kinematics(m, data)
        lo, hi = perf.extent(m, data)
        height = float(np.clip(hi[2] - lo[2], 0.8, 2.5))      # standing height sets the zoom, not the pose
        aim = sm.copy()
        aim[:, 2] = lo[2] + 0.5 * height + AIM_FOLLOW_Z * (sm[:, 2] - root[0, 2])
        fov0 = rig["fovy"]
        want = height / (rig["share"] * 2 * math.tan(math.radians(fov0) / 2))
        room = _Room(m, data, plan)
        el = math.radians(rig["elevation"])
        # pick the side: least wall clamping and least furniture between camera and performer, with the
        # take-off .. landing + 1.5 s part weighted 3x; the first azimuth wins unless another is clearly better
        probe = np.unique(np.linspace(0, len(kt) - 1, min(len(kt), 80)).round().astype(int))
        weight = np.ones(len(probe))
        try:
            _, _, tk, tl, _ = _event_times(events)
            weight[(kt[probe] >= tk - 0.3) & (kt[probe] <= tl + 1.5)] = 3.0
        except (ValueError, TypeError, AttributeError):
            pass
        spots, centres = [], {}                                # per keyframe: the performer's box (shrunk to
        probe_set = set(probe.tolist())                        # 70%: limbs); geom centres at the probe times
        for i in range(len(kt)):
            data.qpos[:] = traj.qpos_at(kt[i])
            mujoco.mj_kinematics(m, data)
            c, h = _world_boxes(m, data, perf.geoms)
            blo, bhi = (c - h).min(0), (c + h).max(0)
            spots.append((0.5 * (blo + bhi), 0.35 * (bhi - blo)))
            if i in probe_set:
                centres[i] = c
        best = None
        for az in rig["azimuths"]:
            u = np.array([math.cos(el) * math.cos(math.radians(az)), math.cos(el) * math.sin(math.radians(az)),
                          math.sin(el)])
            score = 0.0
            for k, i in enumerate(probe):
                d = room.limit(aim[i], u, want, spots[i])
                score += weight[k] * (min(1.0, d / want) - room.hidden(aim[i] + d * u, centres[i]))
            score /= weight.sum()
            if best is None or score > best[0] + 0.05:
                best = (score, az, u)
        self.side_score, self.azimuth, u = best
        allow = np.array([room.limit(aim[i], u, want, spots[i]) for i in range(len(kt))])
        dist = _gauss(_minfilter(np.minimum(allow, want), int(round(2 * sig))), sig)
        dist = np.maximum(np.minimum(dist, allow), 0.3)
        self.kt, self.kaim, self.kpos = kt, aim, aim + dist[:, None] * u
        self.kfovy = np.clip(np.degrees(2 * np.arctan(math.tan(math.radians(fov0) / 2) * want / dist)), fov0, MAX_FOVY)
        self.quat = _look_quat(-u)

    def apply(self, ts):
        if self.scene_cam is not None:
            return self.scene_cam
        m, c = self.model, self.cid
        m.cam_pos[c] = [np.interp(ts, self.kt, self.kpos[:, i]) for i in range(3)]
        m.cam_quat[c] = self.quat
        m.cam_fovy[c] = float(np.interp(ts, self.kt, self.kfovy))
        return c


def _project(model, data, cam, points, size):
    """Pixel coords (n, 2) and depth (n,) of world points seen by camera `cam` (needs mj_camlight)."""
    w, h = size
    R = data.cam_xmat[cam].reshape(3, 3)
    d = np.asarray(points, float) - data.cam_xpos[cam]
    depth = -(d @ R[:, 2])
    f = h / (2 * math.tan(math.radians(float(model.cam_fovy[cam])) / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = w / 2 + f * (d @ R[:, 0]) / depth
        v = h / 2 - f * (d @ R[:, 1]) / depth
    return np.stack([u, v], 1), depth


_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _performer_bbox(model, data, cam, perf, size):
    """Projected (u0, v0, u1, v1) box around the performer's geom boxes, or None if any part is behind the camera."""
    c, h = _world_boxes(model, data, perf.geoms)
    pts = (c[:, None, :] + h[:, None, :] * _SIGNS[None]).reshape(-1, 3)
    uv, depth = _project(model, data, cam, pts, size)
    if np.any(depth <= 1e-6):
        return None
    return float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())


# ----------------------------------------------------------------------------- HUD
_FONTS = {}


def _font(px, kind="regular"):
    """A TrueType font: Helvetica / DejaVu / Arial (or Menlo / DejaVu Sans Mono), else Pillow's built-in."""
    key = (px, kind)
    if key in _FONTS:
        return _FONTS[key]
    cands = {"bold": [("Helvetica.ttc", 1), ("DejaVuSans-Bold.ttf", 0), ("Arial Bold.ttf", 0), ("arialbd.ttf", 0)],
             "mono": [("Menlo.ttc", 0), ("DejaVuSansMono.ttf", 0), ("Courier New.ttf", 0), ("cour.ttf", 0)],
             "regular": [("Helvetica.ttc", 0), ("DejaVuSans.ttf", 0), ("Arial.ttf", 0), ("arial.ttf", 0)]}[kind]
    dirs = ["", "/System/Library/Fonts/", "/System/Library/Fonts/Supplemental/", "/Library/Fonts/",
            "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/", "C:/Windows/Fonts/"]
    font = None
    for name, index in cands:
        for d in dirs:
            try:
                font = ImageFont.truetype(d + name, px, index=index)
                break
            except OSError:
                continue
        if font:
            break
    if font is None:
        try:
            font = ImageFont.load_default(size=px)           # Pillow >= 10.1
        except TypeError:
            font = ImageFont.load_default()
    _FONTS[key] = font
    return font


class _Hud:
    """Small top-left overlay: title, timecode + sim time + SLOW-MO badge, head-G meter with the Gate-1
    limit tick (green < 4, amber < 8, red >= 8 G). Drawn on a crop, so it costs ~1 ms per frame."""

    def __init__(self, size, title, show_g=True):
        w_img, h_img = size
        s = self.s = h_img / 720.0
        self.f_title, self.f_text = _font(max(10, round(18 * s)), "bold"), _font(max(9, round(13 * s)))
        self.f_mono = _font(max(9, round(15 * s)), "mono")
        self.pad, self.gap, self.bp = max(4, round(10 * s)), max(3, round(8 * s)), max(2, round(6 * s))
        self.show_g = show_g
        self.x0 = self.y0 = max(4, round(16 * s))
        row = (self._w("00:00.00", self.f_mono) + self.gap + self._w("sim 00.00 s", self.f_text) + self.gap
               + self._w("SLOW-MO x0.25", self.f_text) + 2 * self.bp)
        max_inner = max(40, min(int(0.45 * w_img), w_img - 2 * self.x0) - 2 * self.pad)
        title = (title or "FALLGUY").strip()
        while len(title) > 6 and self._w(title, self.f_title) > max_inner:
            title = title[:-4].rstrip() + "..."
        self.title = title
        self.line = round(max(self._h(self.f_title), self._h(self.f_mono)) * 1.55)
        self.w = min(max_inner, max(self._w(title, self.f_title), row)) + 2 * self.pad
        self.h = 2 * self.pad + self.line * (3 if show_g else 2) - round(6 * s)

    @staticmethod
    def _w(text, font):
        return int(math.ceil(font.getlength(text)))

    @staticmethod
    def _h(font):
        box = font.getbbox("Hg0")
        return box[3] - box[1]

    def draw(self, frame, t_video, t_sim, rate, g=None, g_peak=None):
        """Draw onto the (H, W, 3) uint8 frame in place and return it."""
        x0, y0, p, s = self.x0, self.y0, self.pad, self.s
        region = frame[y0:y0 + self.h, x0:x0 + self.w]
        h, w = region.shape[:2]
        base = Image.fromarray(region).convert("RGBA")
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=round(7 * s), fill=(12, 14, 18, 176))
        d.text((p, p), self.title, font=self.f_title, fill=(255, 255, 255, 255))
        y = p + self.line
        tc = f"{int(t_video // 60):02d}:{t_video % 60:05.2f}"
        d.text((p, y), tc, font=self.f_mono, fill=(235, 236, 240, 255))
        x = p + self._w(tc, self.f_mono) + self.gap
        d.text((x, y + round(2 * s)), f"sim {t_sim:5.2f} s", font=self.f_text, fill=(160, 166, 178, 255))
        if 0.0 < rate < 0.97:
            label = f"SLOW-MO x{rate:.2f}"
            bw = self._w(label, self.f_text) + 2 * self.bp
            bx = w - p - bw
            d.rounded_rectangle([bx, y - round(2 * s), bx + bw, y + self._h(self.f_mono) + round(5 * s)],
                                radius=round(4 * s), fill=(250, 204, 21, 240))
            d.text((bx + self.bp, y + round(2 * s)), label, font=self.f_text, fill=(20, 20, 20, 255))
        if self.show_g and g is not None:
            y += self.line
            col = (34, 197, 94) if g < G_GREEN else (245, 158, 11) if g < G_AMBER else (239, 68, 68)
            d.text((p, y + round(2 * s)), "HEAD G", font=self.f_text, fill=(160, 166, 178, 255))
            vx = w - p - self._w("88.8 G", self.f_mono)
            bx0, bx1 = p + self._w("HEAD G", self.f_text) + self.gap, vx - self.gap
            bh = max(4, round(8 * s))
            by = y + (self._h(self.f_mono) - bh) // 2 + round(3 * s)
            span = lambda v: bx0 + (bx1 - bx0) * min(max(v, 0.0), G_FULL) / G_FULL
            d.rectangle([bx0, by, bx1, by + bh], fill=(255, 255, 255, 50))
            if span(g) > bx0:
                d.rectangle([bx0, by, span(g), by + bh], fill=col + (255,))
            if g_peak is not None and g_peak > g + 0.05:          # peak so far
                d.rectangle([span(g_peak) - max(1, round(s)), by, span(g_peak), by + bh], fill=(255, 255, 255, 200))
            lim = span(G_AMBER)                                    # the Gate-1 head-G limit
            d.rectangle([lim - max(1, round(s)), by - round(4 * s), lim + max(0, round(s) - 1), by + bh + round(4 * s)],
                        fill=(239, 68, 68, 255))
            d.text((vx, y), f"{g:4.1f} G", font=self.f_mono, fill=col + (255,))
        base.alpha_composite(layer)
        region[:] = np.asarray(base.convert("RGB"))
        return frame


def _default_title(params):
    pid = (params or {}).get("id")
    return f"FALLGUY . {pid}" if pid else "FALLGUY"


# ----------------------------------------------------------------------------- encoding
class _Encoder:
    """Raw RGB frames -> H.264 yuv420p mp4 (BT.709, faststart) with the ffmpeg bundled in imageio-ffmpeg."""

    def __init__(self, path, size, fps):
        w, h = size
        self.path, self.frames = path, 0
        self.err = tempfile.TemporaryFile()
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-crf", "18" if fps <= 30 else "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,"
                      "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv", path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err)

    def write(self, frame):
        try:
            self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).data)
        except (BrokenPipeError, OSError):
            self.p.wait()
            raise RuntimeError(f"ffmpeg stopped while writing {self.path}: {self._stderr()}") from None
        self.frames += 1

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (exit {rc}) writing {self.path}: {self._stderr()}")
        self.err.close()

    def abort(self):
        self.p.kill()
        self.p.wait()
        self.err.close()
        try:
            os.remove(self.path)
        except OSError:
            pass

    def _stderr(self):
        self.err.seek(0)
        return self.err.read().decode(errors="replace").strip()[-800:] or "(no message)"


# ----------------------------------------------------------------------------- compositing
class _Matte:
    """The performer matted over a still plate (the real reference frame). The alpha comes from segmentation,
    so scene geometry in front of the performer occludes it; colour is un-premultiplied (no CG fringe) and
    `ss` x supersampled renders give soft edges."""

    def __init__(self, plate, model, perf, ss=1):
        self.plate, self.ss = np.asarray(plate, np.float32), int(ss)
        self.lut = np.zeros(model.ngeom + 1, bool)            # index -1 (background) hits the False at the end
        self.lut[perf.geoms] = True

    def apply(self, rgb, seg):
        h, w, ss = self.plate.shape[0], self.plate.shape[1], self.ss
        mask = self.lut[np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)].astype(np.float32)
        col = rgb.astype(np.float32) * mask[..., None]
        if ss > 1:
            mask = mask.reshape(h, ss, w, ss).mean(axis=(1, 3))
            col = col.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
        col /= np.maximum(mask[..., None], 1e-6)
        out = self.plate * (1.0 - mask[..., None]) + col * mask[..., None]
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _plate_size(ref_image, h):
    """Composites keep the reference frame's aspect (the match fovy belongs to it): (w, h) at height h."""
    return max(16, int(round(h * ref_image.width / ref_image.height / 2)) * 2), h


# ----------------------------------------------------------------------------- public API
def render_video(inputs, out_mp4, duration, fps=30, size=(1280, 720), camera="hero", hud=False, title=""):
    """Render a simulated stunt to an mp4 of exactly round(duration * fps) frames.

    inputs = {"scene_xml": path, "trajectory": path to trajectory.npz, "events": path (or dict),
              "plan": dict, "params": dict, "metrics": dict or None}
    Optional: inputs["composite_ref"] = path of the reference frame (inputs/ref_frame.jpg) with camera="match"
    mattes the performer over that real frame; the video then takes the frame's aspect at the requested height
    (inputs["composite_ss"] = 2 for softer matte edges at ~3x the render time).
    Returns {"path", "frames", "seconds" (wall time), "slowmo_factor", "camera", "size", "fps", "duration",
             "sim_window", "hold_frames", "hud", "composite"}.
    """
    wall = time.perf_counter()
    for key in ("scene_xml", "trajectory", "events"):
        if not inputs.get(key):
            raise ValueError(f"render_video: inputs[{key!r}] is required")
    w, h = _even_size(size)
    fps = int(fps)
    ref, ss = inputs.get("composite_ref"), 1
    if ref:
        if camera != "match":
            raise ValueError(f"render_video: composite_ref needs camera='match' (the reference frame's view), got {camera!r}")
        if not os.path.isfile(ref):
            raise FileNotFoundError(f"composite_ref not found: {ref}")
        plate = Image.open(ref).convert("RGB")
        w, h = _plate_size(plate, h)
        ss = max(1, int(inputs.get("composite_ss") or 1))
    events = _load_json(inputs["events"], "events")
    plan = _load_json(inputs.get("plan"), "plan")
    params = _load_json(inputs.get("params"), "params")
    model = _load_model(inputs["scene_xml"], (w * ss, h * ss))
    data = mujoco.MjData(model)
    traj = _Trajectory.load(model, inputs["trajectory"])
    tl = _timeline(events, duration, fps)
    shot = _Shot(model, data, traj, plan, camera, events)
    _tune_shadows(model, data, plan, _action_center(model, data, traj, events, plan))
    opt = _scene_option()
    matte = _Matte(plate.resize((w, h), Image.LANCZOS), model, _Performer(model, events), ss) if ref else None
    overlay = _Hud((w, h), title or _default_title(params), show_g=traj.g is not None) if hud else None
    out_mp4 = os.path.abspath(out_mp4)
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    renderer = mujoco.Renderer(model, h * ss, w * ss)
    enc = _Encoder(out_mp4, (w, h), fps)
    g_peak, t_prev = 0.0, None
    try:
        for k, ts in enumerate(tl["times"]):
            cid = shot.apply(ts)
            data.qpos[:] = traj.qpos_at(ts)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, cid, opt)
            frame = renderer.render()
            if matte:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, cid, opt)
                seg = renderer.render()
                renderer.disable_segmentation_rendering()
                frame = matte.apply(frame, seg)
            if overlay:
                g = traj.g_max(ts if t_prev is None else t_prev, ts)
                g_peak = max(g_peak, g) if g is not None else None
                overlay.draw(frame, k / fps, ts, tl["rates"][k], g, g_peak)
            enc.write(frame)
            t_prev = ts
    except BaseException:
        enc.abort()
        raise
    finally:
        renderer.close()
    enc.close()
    if enc.frames != tl["frames"]:
        raise RuntimeError(f"wrote {enc.frames} frames to {out_mp4}, expected {tl['frames']}")
    return {"path": out_mp4, "frames": enc.frames, "seconds": round(time.perf_counter() - wall, 2),
            "slowmo_factor": tl["factor"], "camera": shot.camera, "size": [w, h], "fps": fps,
            "duration": round(enc.frames / fps, 4), "sim_window": [round(tl["t0"], 4), round(tl["t1"], 4)],
            "hold_frames": tl["hold_frames"], "hud": bool(hud), "composite": bool(ref)}


def render_still(scene_xml, qpos, plan, out_png, camera="match", size=(1280, 720)):
    """Render one frame at `qpos` (None: the model's default pose) from `camera`; returns out_png."""
    w, h = _even_size(size)
    model = _load_model(scene_xml, (w, h))
    data = mujoco.MjData(model)
    if qpos is not None:
        q = np.asarray(qpos, float).ravel()
        if q.size != model.nq:
            raise ValueError(f"render_still: qpos has {q.size} values, the scene has nq={model.nq}")
        data.qpos[:] = q
    traj = _Trajectory(model, [0.0], data.qpos.copy(), source="render_still qpos")
    plan = _load_json(plan, "plan")
    shot = _Shot(model, data, traj, plan, camera)
    _tune_shadows(model, data, plan, _action_center(model, data, traj, None, plan))
    cid = shot.apply(0.0)
    data.qpos[:] = traj.qpos_at(0.0)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, h, w) as r:
        r.update_scene(data, cid, _scene_option())
        img = r.render()
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    Image.fromarray(img).save(out_png)
    return out_png


def framing_report(inputs, camera="hero", size=(1280, 720), fps=30, t0=None, t1=None):
    """Geometry-only framing check (no GL), sampled at `fps` over [t0, t1] (default: events t_runup to
    t_landing + 1 s): share of samples whose projected performer box lies inside the frame, plus the box's
    height as a share of the frame height. inputs as for render_video."""
    w, h = _even_size(size)
    events = _load_json(inputs["events"], "events")
    model = _load_model(inputs["scene_xml"], (w, h))
    data = mujoco.MjData(model)
    traj = _Trajectory.load(model, inputs["trajectory"])
    shot = _Shot(model, data, traj, _load_json(inputs.get("plan"), "plan"), camera, events)
    perf = _Performer(model, events)
    _, tr, tk, tl, _ = _event_times(events)
    t0 = tr if t0 is None else t0
    t1 = min(tl + 1.0, float(traj.t[-1])) if t1 is None else t1
    inside, shares = 0, []
    times = np.arange(t0, t1 + 1e-9, 1.0 / fps)
    for ts in times:
        cid = shot.apply(ts)
        data.qpos[:] = traj.qpos_at(ts)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_camlight(model, data)
        box = _performer_bbox(model, data, cid, perf, (w, h))
        if box is None:
            continue
        shares.append((box[3] - box[1]) / h)
        inside += box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h
    n = len(times)
    sh = np.array(shares) if shares else np.zeros(1)
    return {"camera": shot.camera, "frames": n, "inside": int(inside),
            "framing_pct": round(100.0 * inside / max(n, 1), 1),
            "height_share": {"min": round(float(sh.min()), 3), "mean": round(float(sh.mean()), 3),
                             "max": round(float(sh.max()), 3)}}


def composite_over(ref_image_path, scene_xml, qpos, plan, out_png, performer_root="pelvis", ss=2):
    """Matte the performer (rendered from the match camera at `qpos`) over the real reference frame, at the
    frame's own size. Scene geometry in front of the performer occludes it; edges supersampled `ss` x."""
    if not os.path.isfile(ref_image_path):
        raise FileNotFoundError(f"reference frame not found: {ref_image_path}")
    plate = Image.open(ref_image_path).convert("RGB")
    w, h = _even_size(plate.size)
    ss = max(1, int(ss))
    model = _load_model(scene_xml, (w * ss, h * ss))
    data = mujoco.MjData(model)
    if qpos is not None:
        q = np.asarray(qpos, float).ravel()
        if q.size != model.nq:
            raise ValueError(f"composite_over: qpos has {q.size} values, the scene has nq={model.nq}")
        data.qpos[:] = q
    perf = _Performer(model, {"performer_root_body": performer_root})
    traj = _Trajectory(model, [0.0], data.qpos.copy(), source="composite_over qpos")
    plan = _load_json(plan, "plan")
    cid = _Shot(model, data, traj, plan, "match").apply(0.0)
    mujoco.mj_forward(model, data)
    _tune_shadows(model, data, plan, _action_center(model, data, traj, None, plan))
    mujoco.mj_forward(model, data)
    opt = _scene_option()
    with mujoco.Renderer(model, h * ss, w * ss) as r:
        r.update_scene(data, cid, opt)
        rgb = r.render()
        r.enable_segmentation_rendering()
        r.update_scene(data, cid, opt)
        seg = r.render()
    out = _Matte(plate.resize((w, h), Image.LANCZOS), model, perf, ss).apply(rgb, seg)
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    Image.fromarray(out).save(out_png)
    return out_png


def main(argv=None):
    """CLI for re-rendering a candidate dir: render.py --dir waves/wave01/c1 --plan plan.json --out c1.mp4"""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--dir", required=True, help="candidate dir with scene.xml, trajectory.npz, events.json")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--out", required=True, help=".mp4 for a video, .png for a still at the first sample")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--camera", default="hero")
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    j = lambda name: os.path.join(a.dir, name)
    inputs = {"scene_xml": j("scene.xml"), "trajectory": j("trajectory.npz"), "events": j("events.json"),
              "plan": _load_json(a.plan, "plan"), "params": _load_json(j("params.json"), "params") if os.path.isfile(j("params.json")) else {},
              "metrics": None}
    if a.out.lower().endswith(".png"):
        with np.load(inputs["trajectory"]) as z:
            q0 = z["qpos"][0]
        print(render_still(inputs["scene_xml"], q0, inputs["plan"], a.out, a.camera, _even_size(a.size)))
    else:
        print(json.dumps(render_video(inputs, a.out, a.duration, a.fps, _even_size(a.size), a.camera, a.hud, a.title)))


if __name__ == "__main__":
    main()
