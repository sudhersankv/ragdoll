"""Phase interpreter: plan.stunt (start + generic phases) -> everything the simulator needs that is not physics.

There are no stunt types. A stunt is a sequence of phases (reference/contracts.md "Choreography phases"):

  hold   kinematic  stand or hold a pose in place, feet planted
  move   kinematic  locomotion along a polyline path: corners rounded to arcs of turn_radius, heading along the
                    path tangent, hip yaw toward each foot's heading, bank lean_scale * atan(v^2 / (r g)),
                    speed profile accel_time / cruise / decel_time, every stance foot planted by leg IK
  jump   release    gather (a two-foot plant when running, a crouch when standing) and push-off; the pelvis
                    leaves with the centre-of-mass velocity [vx along the heading, vz] and an angular momentum
                    sized so rotations_deg {roll, pitch, yaw} complete over the predicted flight to the support
                    under the ballistic landing point (using the inertia of the air pose schedule)
  fall   release    lean into `pose` and leave support with an optional push [vx, vy, vz] (world frame)
  air    free       PD pose schedule over shares of the predicted flight ({"at", "pose", "tuck"})
  land   free       pose after the first touch, leg servo gains scaled by compliance (1 -> 25 %)

hold / move / the jump's gather are written on the 2 ms grid (exact foot placement); from the release on, MuJoCo
does everything. Poses are performer.H1Rig poses in degrees, or one of POSES. A running jump releases about
speed * 0.35 s past the end of the move before it (the gather hop and push-off continue along the heading).

Custom motion: subclass Choreography (override flight_pose(u), landing_targets(s, q_touch), the phase builders
_hold/_move/_jump/_fall) and hand an instance, a subclass or a callable(plan, params) -> instance to sim.run.
The simulator only uses the attributes documented on Choreography.
"""
import math

import numpy as np

from . import performer as perf
from . import scene

G = 9.81
DT = 0.002
RELEASE_REACH = 0.785    # hip-to-ankle distance at release (knee ~22 deg)
LEG_QDOT_MAX, ARM_QDOT_MAX = 0.5, 2.0    # rad/s: joint velocities handed to the servos at release are capped
TAIL_HOLD = 2.0          # s of stillness simulated after a stunt that never leaves the ground
V_STEP = 0.4             # m/s: below this, a decelerating gait stops stepping
STANCE_TRAVEL = 0.6      # m: the pelvis moves at most this far while one foot is planted
FEET_FIRST = ("stand", "crouch", "land_prep", "brace", "takeoff", "arms_up", "layout")


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def blend(p, q, w):
    keys = set(p) | set(q)
    return {k: p.get(k, 0.0) + (q.get(k, 0.0) - p.get(k, 0.0)) * w for k in keys}


def legs(hip, knee, hip_x=0.0, ankle=0.0):
    return dict(l_hip_y=hip, r_hip_y=hip, l_knee=knee, r_knee=knee, l_hip_x=hip_x, r_hip_x=hip_x, l_ankle=ankle,
                r_ankle=ankle)


def arms(fwd, out=15.0, elbow=10.0):
    return dict(l_shoulder_x=out, r_shoulder_x=out, l_shoulder_y=fwd, r_shoulder_y=fwd, l_elbow=elbow, r_elbow=elbow)


def _hermite(t0, t1, z0, z1, v0, v1, t):
    h = t1 - t0
    s = min(max((t - t0) / h, 0.0), 1.0) if h > 0 else 1.0
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * z0 + h10 * h * v0 + h01 * z1 + h11 * h * v1


def step_freq(v):
    """Steps per second (both feet counted)."""
    return min(max(1.8 + 0.3 * v, 1.8), 3.6)


def stance_frac(v):
    """Share of one leg's cycle on the ground (0.6 walking, about 0.3 when running fast)."""
    return min(max(0.62 - 0.07 * v, 0.3), 0.62)


STAND_ARMS = dict(l_shoulder_x=6, l_shoulder_y=0, l_elbow=15, r_shoulder_x=6, r_shoulder_y=0, r_elbow=15)
BACK_ARMS = dict(l_shoulder_x=10, l_shoulder_y=-45, l_elbow=25, r_shoulder_x=10, r_shoulder_y=-45, r_elbow=25)
# named body poses (H1Rig degrees); "pitch" is the pelvis pitch used when the pose is held on the ground
POSES = {
    "stand": {**legs(0, 0), **STAND_ARMS, "pitch": 0.0},
    "crouch": {**legs(60, 95), **arms(35, 20, 30), "pitch": 25.0},
    "takeoff": {**legs(5, 10), **arms(150, 15, 15), "pitch": 10.0},
    "dive": {**legs(-5, 8), **arms(170, 20, 10), "pitch": 30.0},
    "tuck": {**legs(90, 115), **arms(60, 20, 60), "pitch": 30.0},
    "pike": {**legs(95, 5), **arms(100, 15, 10), "pitch": 40.0},
    "layout": {**legs(0, 5), **arms(170, 25, 10), "pitch": 0.0},
    "land_prep": {**legs(22, 30), **arms(75, 35, 25), "pitch": 10.0},
    "brace": {**legs(45, 65), **arms(45, 25, 30), "pitch": 20.0},
    "arms_up": {**legs(0, 5), **arms(175, 15, 5), "pitch": 0.0},
}


def resolve_pose(p):
    """A named pose or an explicit H1Rig pose dict (missing keys from 'stand') -> pose dict."""
    if isinstance(p, str):
        if p not in POSES:
            raise ValueError(f"unknown pose {p!r}; named poses: {sorted(POSES)} (or an H1Rig pose dict)")
        return dict(POSES[p])
    if isinstance(p, dict):
        return {**POSES["stand"], **{k: float(v) for k, v in p.items()}}
    raise ValueError(f"pose must be a name or a dict, got {p!r}")


def _arm_part(pose):
    return {k: v for k, v in pose.items() if "shoulder" in k or "elbow" in k or k == "torso"}


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quat(yaw, pitch, roll):
    """Quaternion of Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy, cp, sp, cr, sr = (math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2),
                              math.cos(roll / 2), math.sin(roll / 2))
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr])


def _gauss1(x, sigma):
    if sigma < 0.5 or len(x) < 3:
        return np.asarray(x, float).copy()
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")


# ----------------------------------------------------------------------------- path geometry
def fillet_path(points, radius, ds=0.01):
    """Polyline with every corner rounded to an arc of `radius` (shrunk where the segments are too short).
    Returns (s, x, y, heading, curvature, radii_used) sampled every ~ds metres."""
    P = [np.asarray(points[0], float)]
    for p in points[1:]:
        p = np.asarray(p, float)
        if np.linalg.norm(p - P[-1]) > 1e-6:
            P.append(p)
    if len(P) < 2:
        raise ValueError("move path needs two distinct points")
    n = len(P)
    seg = [P[i + 1] - P[i] for i in range(n - 1)]
    L = [float(np.linalg.norm(d)) for d in seg]
    D = [d / l for d, l in zip(seg, L)]
    corners = []                                # (tangent length, signed turn, radius) per interior vertex
    for i in range(1, n - 1):
        d0, d1 = D[i - 1], D[i]
        turn = math.atan2(d0[0] * d1[1] - d0[1] * d1[0], float(d0 @ d1))
        if abs(turn) < 1e-3:
            corners.append((0.0, 0.0, 0.0))
            continue
        lin = L[i - 1] * (0.95 if i == 1 else 0.49)
        lout = L[i] * (0.95 if i == n - 2 else 0.49)
        T = min(radius * math.tan(abs(turn) / 2), lin, lout)
        corners.append((T, turn, T / math.tan(abs(turn) / 2)))
    xs, ys, hs, ks = [], [], [], []

    def line(a, b):
        l = float(np.linalg.norm(b - a))
        if l < 1e-9:
            return
        m = max(1, int(math.ceil(l / ds)))
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        for j in range(m):
            u = j / m
            xs.append(a[0] + (b[0] - a[0]) * u)
            ys.append(a[1] + (b[1] - a[1]) * u)
            hs.append(h)
            ks.append(0.0)

    cur = P[0]
    for i in range(1, n - 1):
        T, turn, r = corners[i - 1]
        a = P[i] - D[i - 1] * T
        line(cur, a)
        if T > 0:
            sgn = 1.0 if turn > 0 else -1.0
            nrm = np.array([-D[i - 1][1], D[i - 1][0]]) * sgn
            c = a + nrm * r
            h0 = math.atan2(D[i - 1][1], D[i - 1][0])
            m = max(2, int(math.ceil(r * abs(turn) / ds)))
            for j in range(m):
                hh = h0 + turn * j / m
                ang = hh - sgn * math.pi / 2         # direction from the centre to the point
                xs.append(c[0] + r * math.cos(ang))
                ys.append(c[1] + r * math.sin(ang))
                hs.append(hh)
                ks.append(sgn / r)
        cur = P[i] + D[i] * T
    line(cur, P[-1])
    xs.append(P[-1][0])
    ys.append(P[-1][1])
    hs.append(hs[-1])
    ks.append(0.0)
    x, y = np.array(xs), np.array(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return s, x, y, np.unwrap(np.array(hs)), np.array(ks), [c[2] for c in corners if c[0] > 0]


# ----------------------------------------------------------------------------- the interpreter
class Choreography:
    """Compiled choreography of one stunt. Attributes the simulator uses:
      K, t_grid                 kinematic steps 0..K on the 2 ms grid (K = release step, or the last step)
      kinematic(k)              (root pos, root quat, joint vector in performer.H1_JOINTS order) at step k
      released                  True when the stunt has a jump or fall
      t_takeoff                 release time (None when the stunt never leaves the ground)
      phase_events              [{"name", "kind", "t0", "t1"}] of the phases up to the release (air/land: sim)
      t_finish_kin              end of the last move (ground stunts), else None
      prepare(model, data, pf)  flight prediction: T_f, v_com_target, L_target, landing_surface_z
      release_joint_velocity(), flight_targets(s), landing_targets(s, q_touch), landing_gain_scale(),
      land_duration, warnings, summary(), stances {leg: [[t0, t1, x, y, z, yaw, swing_h], ...]}
    """

    def __init__(self, plan, stunt, rig=None):
        self.plan = plan
        self.stunt = st = scene.normalize_stunt(stunt)
        self.rig = rig or perf.H1Rig()
        self.warnings = []
        self.mat = scene.target_mat(plan, st)
        s0 = st["start"]
        self.fz = float(s0.get("z", 0.0))
        self.x, self.y = float(s0["pos"][0]), float(s0["pos"][1])
        self.yaw = math.radians(s0["heading_deg"])
        self.pitch = self.roll = 0.0
        self.z = self._pose_height(POSES["stand"])
        self.v = 0.0
        self.next_leg = "right"
        R = _rz(self.yaw)
        self.stances = {}
        for leg, sgn in (("left", 1.0), ("right", -1.0)):
            a = np.array([self.x, self.y, 0.0]) + R @ np.array([perf.HIP_ROLL_X, sgn * perf.FOOT_Y, 0.0])
            self.stances[leg] = [[0.0, math.inf, a[0], a[1], self.fz + perf.ANKLE_HEIGHT, self.yaw, 0.0]]
        # per-step buffers
        self._T, self._P, self._ang, self._exact, self._arm, self._ph = [], [], [], [], [], []
        self.phase_events, self.t_finish_kin = [], None
        self.released, self.release = False, None
        self._T.append(0.0)                                    # the start pose at t = 0
        self._P.append((self.x, self.y, self.z))
        self._ang.append((self.yaw, 0.0, 0.0))
        self._exact.append(False)
        self._arm.append(("keep",))
        self._ph.append(-1)
        self.air, self.land = None, None
        phases = st["phases"]
        for i, ph in enumerate(phases):
            nxt = phases[i + 1]["kind"] if i + 1 < len(phases) else None
            t0 = self._t()
            k = ph["kind"]
            if k == "hold":
                self._hold(i, ph)
            elif k == "move":
                self._move(i, ph, nxt)
                self.t_finish_kin = self._t()
            elif k in ("jump", "fall"):
                self._jump(i, ph) if k == "jump" else self._fall(i, ph)
                self.released, self.release = True, ph
            elif k == "air":
                self.air = ph
                continue
            else:
                self.land = ph
                continue
            self.phase_events.append({"name": ph["name"], "kind": k, "t0": round(t0, 4), "t1": round(self._t(), 4)})
        if not self.released:
            self._hold(len(phases), {"duration": TAIL_HOLD, "pose": None})
        self._finish()
        self.T_f = None

    # ------------------------------------------------------------------ step buffer
    def _t(self):
        return (len(self._T) - 1) * DT

    def _push(self, ph, P, yaw, pitch, roll, arm, exact=False):
        self._T.append(len(self._T) * DT)
        self._P.append(P)
        self._ang.append((yaw, pitch, roll))
        self._exact.append(exact)
        self._arm.append(arm)
        self._ph.append(ph)
        self.x, self.y, self.z = P
        self.yaw, self.pitch, self.roll = yaw, pitch, roll

    def _pose_height(self, pose):
        k = math.radians(0.5 * (pose.get("l_knee", 0.0) + pose.get("r_knee", 0.0)))
        d = min(math.sqrt(perf.THIGH ** 2 + perf.SHIN ** 2 + 2 * perf.THIGH * perf.SHIN * math.cos(k)), 0.797)
        th = math.radians(pose.get("pitch", 0.0))
        return self.fz + perf.ANKLE_HEIGHT + d - (perf.HIP_PITCH_OFFSET[2] * math.cos(th) - perf.HIP_PITCH_OFFSET[0] * math.sin(th))

    def _cut(self, leg, t):
        """End the leg's current stance(s) at t (stances starting after t are dropped)."""
        S = self.stances[leg]
        while len(S) > 1 and S[-1][0] >= t:
            S.pop()
        S[-1][1] = min(S[-1][1], max(t, S[-1][0]))

    def _foot_pos(self, leg, x, y, yaw, pitch=0.0, roll=0.0):
        """Stance ankle xy under a hip; banked (roll > 0 leans right) the feet land outboard, under the leaning
        legs, so the H1's +-25 deg hip roll is not asked to carry the whole bank."""
        sgn = 1.0 if leg == "left" else -1.0
        dx = perf.hip_world_xz((0.0, 0.0), pitch)[0]
        a = np.array([x, y, 0.0]) + _rz(yaw) @ np.array([dx, sgn * perf.FOOT_Y + 0.8 * math.tan(roll), 0.0])
        return a[0], a[1]

    # ------------------------------------------------------------------ hold
    def _hold(self, idx, ph):
        n = int(round(ph["duration"] / DT))
        if n <= 0:
            return
        pose = resolve_pose(ph["pose"]) if ph.get("pose") is not None else None
        z0, p0, r0 = self.z, self.pitch, self.roll
        z1 = self._pose_height(pose) if pose else z0
        p1 = math.radians(pose.get("pitch", 0.0)) if pose else p0
        Tb = min(0.5, 0.5 * ph["duration"])
        if idx == 0 and pose is not None:
            # opening hold from the standing start: settle into the pose over most of the hold, so a 0.5 s
            # "ready crouch" does not snap 0.24 m in 0.25 s and log a ~5 g head / 3 x BW spike at rest
            Tb = min(1.0, 0.9 * ph["duration"])
        target = _arm_part(pose) if pose else None
        for j in range(1, n + 1):
            w = smooth(j * DT / Tb) if Tb > 0 else 1.0
            arm = ("blend", target, w) if target else ("keep",)
            self._push(idx, (self.x, self.y, z0 + (z1 - z0) * w), self.yaw, p0 + (p1 - p0) * w, r0 * (1 - w), arm)
        self.v = 0.0

    # ------------------------------------------------------------------ move
    def _move(self, idx, ph, nxt):
        pts = [list(p) for p in ph["path"]]
        if math.hypot(pts[0][0] - self.x, pts[0][1] - self.y) > 0.3:
            self.warnings.append(f"{ph['name']}: path starts {math.hypot(pts[0][0] - self.x, pts[0][1] - self.y):.2f} m "
                                 f"from the performer; the path is joined from where it stands")
            pts.insert(0, [self.x, self.y])
        else:
            pts[0] = [self.x, self.y]
        s_d, x_d, y_d, h_d, k_d, radii = fillet_path(pts, ph["turn_radius"])
        for r in radii:
            if r < ph["turn_radius"] * 0.98:
                self.warnings.append(f"{ph['name']}: a corner is too close to its neighbours; turn radius "
                                     f"{ph['turn_radius']:.2f} m shrunk to {r:.2f} m")
        L = float(s_d[-1])
        v_in, v = self.v, ph["speed"]
        Ta, Td = ph["accel_time"], ph["decel_time"]
        stop = Td > 0 or nxt not in ("jump", "move")
        if Td <= 0 and stop:
            Td = 0.4
        d_a, d_d = Ta * (v_in + v) / 2, v * Td / 2
        if d_a + d_d > L:
            v_new = max(0.3, (L - Ta * v_in / 2) / (Ta / 2 + Td / 2))
            self.warnings.append(f"{ph['name']}: the path ({L:.1f} m) is too short to reach {v:.1f} m/s; "
                                 f"peak speed {v_new:.1f} m/s")
            v = v_new
            d_a, d_d = Ta * (v_in + v) / 2, v * Td / 2
            if d_a + d_d > L:
                Ta = Td = L / max(v_in / 2 + v, 1e-6)
                d_a, d_d = Ta * (v_in + v) / 2, v * Td / 2
        Tc = max(0.0, (L - d_a - d_d) / v)
        T = Ta + Tc + Td
        n = max(2, int(round(T / DT)))
        tl = np.arange(1, n + 1) * DT * (T / (n * DT))
        vel = np.where(tl < Ta, v_in + (v - v_in) * np.array([smooth(u) for u in tl / Ta]), v)
        if Td > 0:
            dec = tl > Ta + Tc
            vel = np.where(dec, v * (1 - np.array([smooth(u) for u in (tl - Ta - Tc) / Td])), vel)
        s = np.cumsum(vel) * (tl[0])
        s *= L / max(s[-1], 1e-9)
        acc = np.gradient(vel, tl)
        x = np.interp(s, s_d, x_d)
        y = np.interp(s, s_d, y_d)
        hd = np.interp(s, s_d, h_d)
        kap = np.interp(s, s_d, k_d)
        # turn in place toward the path when the start heading differs
        dh = (hd[0] - self.yaw + math.pi) % (2 * math.pi) - math.pi
        hd = hd - hd[0] + self.yaw + dh
        if abs(dh) > 0.05:
            w = np.array([smooth(u) for u in tl / min(0.6, max(Ta, 0.3))])
            hd = hd - dh * (1 - w)
        bank = _gauss1(-ph["lean_scale"] * np.arctan(vel ** 2 * kap / G), 0.15 / DT)
        lean = np.radians(np.minimum(3.0 + 3.0 * vel, 22.0)) + 0.5 * np.arctan(acc / G)
        w0 = np.array([smooth(u) for u in tl / 0.3])
        lean = _gauss1(lean, 0.1 / DT)
        lean = self.pitch + (lean - self.pitch) * w0
        bank = self.roll + (bank - self.roll) * w0
        t_abs = self._t() + tl
        tstart = self._t()
        # ---- footsteps
        moving = (vel > 0.05) & ~((acc < 0) & (vel < V_STEP))
        if stop:
            moving &= ~(tl > Ta + Tc + 0.0) | (vel > V_STEP)
        rate = np.where(moving, [step_freq(vv) for vv in vel], 0.0)
        phi0 = 0.5 if v_in < 0.3 else 0.0                       # standing start: the first foot goes at once
        phi = np.concatenate([[phi0], phi0 + np.cumsum(rate) * DT])
        tt = np.concatenate([[tstart], t_abs])
        mono = phi + np.arange(len(phi)) * 1e-12
        tof = lambda p: float(np.interp(p, mono, tt)) if p <= phi[-1] else float(tt[-1] + (p - phi[-1]) / step_freq(0.0))
        at = lambda tq, arr: float(np.interp(tq, t_abs, arr))
        s_mono = s + np.arange(n) * 1e-9
        t_of_s = lambda sq: float(np.interp(sq, s_mono, t_abs)) if sq <= s_mono[-1] else math.inf
        N = int(math.floor(phi[-1] + 1e-9))
        first = 1 if phi0 < 1 else int(phi0)
        order = [self.next_leg, "left" if self.next_leg == "right" else "right"]
        beta0 = stance_frac(max(v_in, 0.5))
        for j, leg in enumerate(order):                           # first lift-offs from the current stance
            self._cut(leg, max(tstart, min(tof(max(phi0, j + 1 - 2 * (1 - beta0))), t_of_s(0.5 * STANCE_TRAVEL))))
        end_t = float(t_abs[-1])
        for i in range(1, N + 1):
            leg = order[(i - 1) % 2]
            td = tof(i)
            beta = stance_frac(at(td, vel))
            s_td = at(td, s)
            lo = min(tof(i + 2 * beta), t_of_s(s_td + STANCE_TRAVEL))
            mid = t_of_s(0.5 * (s_td + at(min(lo, end_t), s)))
            mid = min(mid, end_t)
            fx, fy = self._foot_pos(leg, at(mid, x), at(mid, y), at(mid, hd), at(mid, lean), at(mid, bank))
            S = self.stances[leg]
            S[-1][1] = min(S[-1][1], td - 0.04)
            h = float(np.clip(0.05 + 0.035 * at(td, vel), 0.05, 0.18))
            S.append([td, lo, fx, fy, self.fz + perf.ANKLE_HEIGHT, at(mid, hd), h])
        if N:
            self.next_leg = order[N % 2]
        if stop:                                                  # close the feet beside each other at the end
            for n_, leg in enumerate(sorted(order, key=lambda l_: self.stances[l_][-1][0])):
                S = self.stances[leg]
                other = self.stances["left" if leg == "right" else "right"]
                fx, fy = self._foot_pos(leg, x[-1], y[-1], hd[-1], 0.0)
                if math.hypot(S[-1][2] - fx, S[-1][3] - fy) <= 0.06:
                    S[-1][1] = math.inf
                    continue
                lift = S[-1][1] if S[-1][0] >= tstart else S[-1][0] + 0.1
                if n_ == 1 or lift == math.inf:
                    lift = max(min(lift, end_t), other[-1][0] + 0.05)
                td_c = max(lift + 0.28, other[-1][0] + 0.05)
                S[-1][1] = lift
                S.append([td_c, math.inf, fx, fy, self.fz + perf.ANKLE_HEIGHT, hd[-1], 0.06])
                end_t = max(end_t, td_c)
        # ---- pelvis height: running height with a bob in each stance
        z_run = self._pose_height(POSES["stand"]) - (0.02 + 0.035 * np.minimum(vel, 4.0))
        # vertical oscillation once per step, lowest at mid-stance (about 1.3 G peak at 3 steps/s)
        beta_v = np.array([stance_frac(vv) for vv in vel])
        z_run = z_run - 0.015 * np.minimum(vel, 4.0) * 0.5 * (1.0 + np.cos(2 * math.pi * (phi[1:] - 1.0 - beta_v)))
        ab = np.abs(bank)                                          # banked legs reach further: run lower
        z_run = z_run - (0.75 * (1.0 / np.cos(ab) - 1.0) + 0.8 * perf.FOOT_Y * np.sin(ab))
        z = self.z + (z_run - self.z) * w0
        for j in range(n):
            self._push(idx, (float(x[j]), float(y[j]), float(z[j])), float(hd[j]), float(lean[j]), float(bank[j]),
                       ("run", float(vel[j]), float(w0[j])))
        self.v = 0.0 if stop else float(vel[-1])
        if stop and end_t > self._t():                            # let the closing step land
            self._hold(idx, {"duration": end_t - self._t() + 0.05, "pose": None})

    # ------------------------------------------------------------------ jump / fall (kinematic part)
    def _release_geometry(self, x_rel, lean, leg_angle):
        c, s = math.cos(leg_angle), math.sin(leg_angle)
        hx, hz = perf.hip_world_xz((x_rel, 0.0), lean)
        return self.fz + perf.ANKLE_HEIGHT + RELEASE_REACH * c - hz, hx - RELEASE_REACH * s

    def _jump(self, idx, ph):
        v_in = self.v
        vx = v_in if ph.get("vx") is None else float(ph["vx"])
        vz = max(float(ph["vz"]), 0.3)
        lean = math.radians(ph["lean_deg"])
        running = v_in > 0.6
        e = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        O = np.array([self.x, self.y])
        T_push = min(max(2 * 0.28 / vz, 0.12), 0.30)
        leg_angle = lean + (math.radians(min(12.0, 4.0 * vx)) if running else 0.0)
        t0 = self._t()
        if running:
            T_hop = min(max(0.55 / step_freq(v_in), 0.14), 0.24)
            x_push0 = v_in * T_hop
            x_rel = x_push0 + 0.5 * (v_in + vx) * T_push
            z_rel, x_ank = self._release_geometry(x_rel, lean, leg_angle)
            T_pre = T_hop
        else:
            feet = np.array([[S[-1][2], S[-1][3]] for S in self.stances.values()])
            x_ank = float(np.mean((feet - O) @ e))
            c, s_ = math.cos(lean), math.sin(lean)
            ox, oz = perf.HIP_PITCH_OFFSET[0], perf.HIP_PITCH_OFFSET[2]
            x_rel = x_ank + RELEASE_REACH * math.sin(leg_angle) - (c * ox + s_ * oz)
            z_rel, _ = self._release_geometry(x_rel, lean, leg_angle)
            T_pre = 0.5
            x_push0 = x_rel - 0.5 * vx * T_push
        rise = vz * T_push / 2
        z_bot = z_rel - rise
        n_pre, n_push = int(round(T_pre / DT)), int(round(T_push / DT))
        T_pre, T_push = n_pre * DT, n_push * DT
        t_push0, t_rel = t0 + T_pre, t0 + T_pre + T_push
        # feet: a two-foot plant at the release footprint
        t_feet = t_push0 - 0.04 if running else t0
        for leg in ("left", "right"):
            sgn = 1.0 if leg == "left" else -1.0
            if running:
                self._cut(leg, t0)
                S = self.stances[leg]
                S[-1][1] = min(S[-1][1], t_feet - 0.04)
                a = O + e * x_ank + np.array([-e[1], e[0]]) * sgn * perf.FOOT_Y
                S.append([t_feet, math.inf, a[0], a[1], self.fz + perf.ANKLE_HEIGHT, self.yaw, 0.12])
        z0, p0, r0 = self.z, self.pitch, self.roll
        x_start = 0.0 if running else float((np.array([self.x, self.y]) - O) @ e)
        plant = min(lean, 0.0) - math.radians(5.0)
        crouch = math.radians(max(math.degrees(lean), 0.0) + 20.0)
        takeoff_arms = _arm_part(resolve_pose(ph["pose"]))
        yaw = self.yaw
        for j in range(1, n_pre + n_push + 1):
            t = t0 + j * DT
            if t <= t_push0 + 1e-9:
                u = (t - t0) / T_pre
                xl = x_start + v_in * (t - t0) if running else _hermite(t0, t_push0, x_start, x_push0, 0.0, 0.0, t)
                z = _hermite(t0, t_push0, z0, z_bot, 0.0, 0.0, t)
                pitch = p0 + ((plant if running else crouch) - p0) * smooth(u)
                arm = ("blend2", BACK_ARMS, smooth(u), takeoff_arms, 0.0)
                exact = False
            else:
                u = (t - t_push0) / T_push
                xl = _hermite(t_push0, t_rel, x_push0, x_rel, v_in if running else 0.0, vx, t)
                z = _hermite(t_push0, t_rel, z_bot, z_rel, 0.0, vz, t)
                pitch = (plant if running else crouch) + (lean - (plant if running else crouch)) * smooth(u)
                arm = ("blend2", BACK_ARMS, 1.0, takeoff_arms, smooth(u))
                exact = True
            p = O + e * xl
            self._push(idx, (float(p[0]), float(p[1]), float(z)), yaw, pitch, r0 * (1 - smooth((t - t0) / 0.2)), arm, exact)
        self.v_com_target = np.array([vx * e[0], vx * e[1], vz])
        self.rotations = ph["rotations_deg"]

    def _fall(self, idx, ph):
        n = int(round(0.3 / DT))
        lean = math.radians(ph["lean_deg"])
        p0 = self.pitch
        target = _arm_part(resolve_pose(ph["pose"]))
        for j in range(1, n + 1):
            w = smooth(j / n)
            self._push(idx, (self.x, self.y, self.z), self.yaw, p0 + (lean - p0) * w, self.roll, ("blend", target, w))
        self.v_com_target = np.array(ph["push"], float)
        self.rotations = ph["rotations_deg"]

    # ------------------------------------------------------------------ feet, pelvis height, IK
    def _foot(self, leg, t):
        """(ankle xyz, yaw, toe pitch, planted) of one leg at time t."""
        S = self.stances[leg]
        for j, st in enumerate(S):
            if st[0] - 1e-9 <= t <= st[1] + 1e-9:
                return np.array(st[2:5]), st[5], 0.0, True
            if j + 1 < len(S) and st[1] < t < S[j + 1][0]:
                nx = S[j + 1]
                u = (t - st[1]) / (nx[0] - st[1])
                w = smooth(u)
                dyaw = (nx[5] - st[5] + math.pi) % (2 * math.pi) - math.pi
                # swing relative to the pelvis: the foot leaves from behind the hip and reaches ahead of it
                (p0, h0), (p1, h1), (pt, ht) = self._pel(st[1]), self._pel(nx[0]), self._pel(t)
                r0 = _rz(-h0) @ (np.array(st[2:5]) - p0)                 # in the pelvis heading frame
                r1 = _rz(-h1) @ (np.array(nx[2:5]) - p1)
                pos = pt + _rz(ht) @ (r0 + (r1 - r0) * w)
                pos[2] += nx[6] * math.sin(math.pi * u)
                return pos, st[5] + dyaw * w, 0.35 * min(1.0, nx[6] / 0.12) * math.sin(math.pi * u) * (1 - u), False
        st = S[-1]
        return np.array(st[2:5]), st[5], 0.0, True

    def _finish(self):
        n = len(self._T)
        self.K = n - 1
        self.t_grid = t = np.array(self._T)
        P = np.array(self._P, float)
        ang = np.array(self._ang, float)
        exact = np.array(self._exact, bool)
        self._pel = lambda tq: (np.array([np.interp(tq, t, P[:, 0]), np.interp(tq, t, P[:, 1]), 0.0]),
                                float(np.interp(tq, t, ang[:, 0])))
        feet = {leg: [self._foot(leg, ti) for ti in t] for leg in ("left", "right")}
        Rs = [_rot(*a) for a in ang]
        # pelvis height kept inside the reach of every planted foot, then smoothed (not the push-off)
        zmax = np.full(n, np.inf)
        reach = perf.REACH_MAX - 0.004
        for leg in ("left", "right"):
            for i in range(n):
                A, _, _, planted = feet[leg][i]
                if planted:
                    J = perf.hip_pitch_joint((P[i, 0], P[i, 1], 0.0), Rs[i], leg)
                    h2 = (A[0] - J[0]) ** 2 + (A[1] - J[1]) ** 2
                    zmax[i] = min(zmax[i], A[2] - J[2] + math.sqrt(max(reach * reach - h2, 0.0)))
        znom = np.where(exact, P[:, 2], _gauss1(P[:, 2], 4.0))
        dz = np.maximum(znom - zmax, 0.0)                         # reach-limited drops, anticipated and smoothed
        r = 60
        dzm = np.lib.stride_tricks.sliding_window_view(np.pad(dz, r, mode="edge"), 2 * r + 1).max(axis=1)
        dzs = _gauss1(dzm, 20.0)
        P[:, 2] = np.where(exact, P[:, 2], _gauss1(np.minimum(znom - dzs, zmax), 3.0))
        q = np.zeros((n, len(perf.H1_JOINTS)))
        J = {nme: i for i, nme in enumerate(perf.H1_JOINTS)}
        self.reach_error, self.reach_error_at = 0.0, None
        for i in range(n):
            for leg in ("left", "right"):
                A, fyaw, toe, _ = feet[leg][i]
                sol, err = perf.leg_ik3(P[i], Rs[i], leg, A, fyaw, toe)
                if not exact[i] and err > self.reach_error:
                    self.reach_error, self.reach_error_at = err, (round(float(t[i]), 3), leg)
                for k_, v_ in sol.items():
                    q[i, J[k_]] = v_
        # arms (and torso yaw), evaluated in order so each phase blends from where the last one ended
        cur = dict(STAND_ARMS, torso=0.0)
        start = dict(cur)
        prev_ph = None
        for i in range(n):
            if self._ph[i] != prev_ph:
                start, prev_ph = dict(cur), self._ph[i]
            spec = self._arm[i]
            if spec[0] == "run":
                _, vv, w0 = spec
                amp = min(1.0, vv / 2.0)
                th_l, th_r = -math.degrees(q[i, J["left_hip_pitch"]]), -math.degrees(q[i, J["right_hip_pitch"]])
                swing = 0.5 * (th_r - th_l)
                run = dict(l_shoulder_x=8, r_shoulder_x=8, l_elbow=15 + 70 * amp, r_elbow=15 + 70 * amp,
                           l_shoulder_y=float(np.clip(10 * amp + swing * amp, -50, 70)),
                           r_shoulder_y=float(np.clip(10 * amp - swing * amp, -50, 70)),
                           torso=float(10.0 * math.tanh(0.02 * (th_l - th_r) * amp)))
                cur = blend(start, run, w0)
            elif spec[0] == "blend":
                cur = blend(start, {**spec[1], "torso": spec[1].get("torso", 0.0)}, spec[2])
            elif spec[0] == "blend2":
                cur = blend(blend(start, {**spec[1], "torso": 0.0}, spec[2]), spec[3], spec[4])
            self.rig.arm_vector(cur, q[i])
        self.arm_pose_end = cur
        self.q = np.clip(q, self.rig.lo, self.rig.hi)
        self.P, self.ang = P, ang
        self.quats = np.array([_quat(*a) for a in ang])
        if self.reach_error > 0.02:
            self.warnings.append(f"leg IK out of reach by {self.reach_error * 100:.1f} cm "
                                 f"(t={self.reach_error_at[0]:.2f} s, {self.reach_error_at[1]} leg)")
        self.t_takeoff = float(t[-1]) if self.released else None

    # ------------------------------------------------------------------ used by the simulator
    def kinematic(self, k):
        k = min(max(int(k), 0), self.K)
        return self.P[k].copy(), self.quats[k].copy(), self.q[k].copy()

    def release_joint_velocity(self):
        qd = (self.q[-1] - self.q[-2]) / DT
        cap = np.array([LEG_QDOT_MAX if j in perf.LEG_JOINTS else ARM_QDOT_MAX for j in perf.H1_JOINTS])
        return np.clip(qd, -cap, cap)

    def _schedule(self):
        if self.air and self.air["poses"]:
            return [(e["at"], blend(resolve_pose(e["pose"]), POSES["tuck"], e["tuck"])) for e in self.air["poses"]]
        rel = resolve_pose(self.release.get("pose", "takeoff"))
        return [(0.0, rel), (0.3, POSES["layout"]), (0.85, POSES["land_prep"])]

    def flight_pose(self, u):
        """Abstract pose at flight share u (0 release, 1 predicted touchdown)."""
        sch = self._schedule()
        if u <= sch[0][0]:
            return sch[0][1]
        for (a0, p0), (a1, p1) in zip(sch, sch[1:]):
            if u <= a1:
                return blend(p0, p1, smooth((u - a0) / max(a1 - a0, 1e-6)))
        return sch[-1][1]

    def flight_targets(self, s):
        """Joint targets s seconds after release: ease from the release pose into flight_pose(s / T_f)."""
        T = self.T_f or 0.8
        tgt = self.rig.vector(self.flight_pose(s / T))
        w = smooth(s / 0.1)
        return (1.0 - w) * self.q[-1] + w * tgt

    def land_pose(self):
        return resolve_pose(self.land["pose"]) if self.land else POSES["crouch"]

    def landing_targets(self, s, from_q):
        tgt = self.rig.vector(self.land_pose())
        w = smooth(s / 0.15)
        return (1.0 - w) * from_q + w * tgt

    def landing_gain_scale(self):
        return 1.0 - 0.75 * float(self.land["compliance"] if self.land else 0.6)

    @property
    def land_duration(self):
        return float(self.land["duration"]) if self.land else 1.0

    def _flight_time(self, z0, z1, vz):
        disc = vz ** 2 + 2 * G * (z0 - z1)
        return max((vz + math.sqrt(disc)) / G if disc > 0 else vz / G, 0.25)

    def prepare(self, model, data, pf):
        """Flight-time estimate and take-off angular momentum (world frame), from the compiled model."""
        import mujoco
        pos, quat, qj = self.kinematic(self.K)
        data.qpos[pf.free_qadr:pf.free_qadr + 3] = pos
        data.qpos[pf.free_qadr + 3:pf.free_qadr + 7] = quat
        data.qpos[pf.qadr] = qj
        mujoco.mj_kinematics(model, data)
        com0, _ = pf.com_inertia(model, data)
        self.com0 = com0
        land = self.land["pose"] if self.land else "crouch"
        hl = 0.85 if (isinstance(land, dict) or land in FEET_FIRST) else 0.5
        v = self.v_com_target
        surf = 0.0
        for _ in range(3):
            T = self._flight_time(com0[2], surf + hl, v[2])
            surf = scene.support_top(self.plan, (com0[0] + v[0] * T, com0[1] + v[1] * T), self.mat)
        self.landing_surface_z = surf
        self.T_f = self._flight_time(com0[2], surf + hl, v[2])
        H = _rz(self.yaw)
        us = np.linspace(0.0, 1.0, 13)
        inv = []
        for u in us:
            data.qpos[pf.qadr] = self.flight_targets(u * self.T_f)
            mujoco.mj_kinematics(model, data)
            _, I = pf.com_inertia(model, data)
            inv.append(1.0 / np.diag(H.T @ I @ H))
        integral = np.trapezoid(np.array(inv), us * self.T_f, axis=0)
        r = self.rotations
        rot = np.radians([r["roll"], r["pitch"], r["yaw"]])
        self.L_target = H @ (rot / integral)
        data.qpos[pf.qadr] = qj
        mujoco.mj_kinematics(model, data)

    def summary(self):
        return {"phases": self.phase_events, "t_takeoff": None if self.t_takeoff is None else round(self.t_takeoff, 4),
                "flight_time_est": None if self.T_f is None else round(self.T_f, 3),
                "reach_error_cm": round(self.reach_error * 100, 2), "warnings": list(self.warnings)}


def from_stunt(plan, stunt, rig=None):
    return Choreography(plan, stunt, rig)
