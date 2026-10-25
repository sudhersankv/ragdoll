"""Stunt choreography library, keyed by plan.stunt.type.

A choreography is everything the simulator needs that is not physics:
- the kinematic run-up on the 2 ms grid (pelvis path from start_x to takeoff_x at approach_speed along lane_y,
  a run cycle with every stance foot planted by leg IK, a two-foot plant, the push-off), which ends in the
  release state: pelvis at takeoff_x leaning lean_deg with the legs extended;
- the take-off: centre-of-mass velocity [approach_speed, 0, takeoff_vz] and a whole-body angular momentum
  sized so rot_roll/pitch/yaw are reached at touchdown, using the inertia of the flight pose schedule over the
  ballistic flight time to the mat top (so tuck and arm reach change how fast it spins);
- PD servo targets in flight (tuck between tuck_start and tuck_end, arms at arm_reach_deg) and after
  touchdown (an absorbing landing whose leg stiffness scales with landing_compliance).
Nothing overrides the physics: apex, landing point and rotation come out of MuJoCo.

Custom stunts: subclass Choreo (or a library class) and override the hooks rotation_deg(), takeoff_arms(),
flight_pose(u), landing_pose(s) and landing_com_height(); poses are performer.H1Rig abstract poses in degrees.
"""
import math
import sys

import numpy as np

from . import performer as perf
from . import scene

G = 9.81
DT = 0.002
T_STAND = 0.8            # s standing still before the run-up starts (events t_runup)
RELEASE_REACH = 0.785    # hip-to-ankle distance at release (knee ~22 deg)
CROUCH_REACH_MIN = 0.45  # deepest crouch (knee ~110 deg)
LEG_QDOT_MAX, ARM_QDOT_MAX = 0.5, 2.0    # rad/s: joint velocities handed to the servos at release are capped
                                         # (the scripted leg extension is near-singular at release; handing it
                                         # over would make the stiff servos jerk the torso, a fake head-G spike)

STAND_ARMS = dict(l_shoulder_x=6, l_shoulder_y=0, l_elbow=15, r_shoulder_x=6, r_shoulder_y=0, r_elbow=15)
BACK_ARMS = dict(l_shoulder_x=10, l_shoulder_y=-45, l_elbow=25, r_shoulder_x=10, r_shoulder_y=-45, r_elbow=25)


def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6 * u - 15) + 10)


def smooth_int(u):
    """Integral of smooth() from 0 to u (u in [0, 1]); smooth_int(1) = 0.5."""
    u = min(max(u, 0.0), 1.0)
    return u ** 6 - 3 * u ** 5 + 2.5 * u ** 4


def window(u, a, b, ramp=0.08):
    """1 inside [a, b] with smooth ramps of width `ramp` (shrunk for short windows), else 0."""
    r = max(1e-3, min(ramp, 0.5 * (b - a)))
    return smooth((u - a) / r) * (1.0 - smooth((u - b) / r))


def blend(p, q, w):
    keys = set(p) | set(q)
    return {k: p.get(k, 0.0) + (q.get(k, 0.0) - p.get(k, 0.0)) * w for k in keys}


def legs(hip, knee, hip_x=0.0, ankle=0.0):
    return dict(l_hip_y=hip, r_hip_y=hip, l_knee=knee, r_knee=knee, l_hip_x=hip_x, r_hip_x=hip_x, l_ankle=ankle, r_ankle=ankle)


def arms(fwd, out=15.0, elbow=10.0):
    return dict(l_shoulder_x=out, r_shoulder_x=out, l_shoulder_y=fwd, r_shoulder_y=fwd, l_elbow=elbow, r_elbow=elbow)


def _hermite(t0, t1, z0, z1, v0, v1, t):
    h = t1 - t0
    s = min(max((t - t0) / h, 0.0), 1.0)
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * z0 + h10 * h * v0 + h01 * z1 + h11 * h * v1


def step_freq(v):
    """Steps per second of an accelerating run (both feet counted)."""
    return min(max(2.4 + 0.25 * v, 2.4), 3.6)


def stance_frac(v):
    """Share of one leg's cycle spent on the ground (0.5 = walking, about 0.3 when running fast)."""
    return min(max(0.55 - 0.08 * v, 0.3), 0.55)


class Choreo:
    """Base choreography (a barrel roll). See the module docstring for the hooks to override."""

    name = "barrel_roll"
    feet_first = True          # lands on the feet; False: hands / back first
    standing_takeoff = False   # True: no run-up, a standing jump (or step-off) at takeoff_x

    def __init__(self, plan, params, rig=None):
        self.plan, self.p = plan, params
        self.rig = rig or perf.H1Rig()
        self.warnings = []
        st = plan["stunt"]
        self.lane_y, self.start_x = st["lane_y"], st["start_x"]
        self.takeoff_x = float(params["takeoff_x"])
        self.v, self.vz = float(params["approach_speed"]), float(params["takeoff_vz"])
        self.lean = math.radians(params["lean_deg"])
        self.mat = scene.landing_mat(plan, params)
        self.floor_z, self.platform = 0.0, None
        self._setup()
        self._build_runup()
        self.T_f = None

    # ------------------------------------------------------------------ hooks (override for custom stunts)
    def _setup(self):
        """Type-specific set-up before the run-up is planned (e.g. stand on a platform)."""

    def rotation_deg(self):
        """(roll, pitch, yaw) to complete by touchdown, about the travel frame (X travel, Y left, Z up)."""
        p = self.p
        return p["rot_roll_deg"], p["rot_pitch_deg"], p["rot_yaw_deg"]

    def takeoff_arms(self):
        """Arm pose at the instant of release (the push-off swings the arms here from behind the body)."""
        reach = self.p["arm_reach_deg"]
        return arms(min(0.85 * reach + 10.0, 165.0), out=15.0, elbow=20.0)

    def flight_pose(self, u):
        """Abstract pose at flight fraction u (0 take-off, 1 predicted touchdown, may exceed 1)."""
        p = self.p
        a = p["tuck"] * window(u, p["tuck_start"], p["tuck_end"])
        pose = {**legs(8.0 + 82.0 * a, 10.0 + 105.0 * a), **arms(p["arm_reach_deg"] * (1.0 - 0.25 * a), 15.0, 10.0 + 25.0 * a)}
        land = smooth((u - max(p["tuck_end"], 0.55)) / max(0.15, 1.0 - max(p["tuck_end"], 0.55)))
        return blend(pose, self.landing_prep_pose(), land * (1.0 - a))

    def landing_prep_pose(self):
        """Pose reaching for the landing: legs under the body, knees soft, arms forward and out for balance."""
        return {**legs(22.0, 30.0), **arms(75.0, 35.0, 25.0)}

    def landing_pose(self, s):
        """Abstract pose s seconds after the first touch: absorb, then hold a crouch."""
        absorb = {**legs(70.0, 100.0), **arms(75.0, 30.0, 35.0)}
        hold = {**legs(45.0, 65.0), **arms(45.0, 25.0, 30.0)}
        return blend(blend(self.landing_prep_pose(), absorb, smooth(s / 0.15)), hold, smooth((s - 0.6) / 0.8))

    def landing_com_height(self):
        """Centre-of-mass height above the landing surface at the first touch (sets the flight-time estimate)."""
        return 0.92 if self.feet_first else 0.5

    # ------------------------------------------------------------------ kinematic run-up
    def _release_geometry(self, x_rel, leg_angle):
        """Pelvis z and ankle x at release: body leaning self.lean, legs extended `leg_angle` behind vertical."""
        c, s = math.cos(leg_angle), math.sin(leg_angle)
        hx, hz = perf.hip_world_xz((x_rel, 0.0), self.lean)
        z_rel = self.floor_z + perf.ANKLE_HEIGHT + RELEASE_REACH * c - hz
        x_ank = hx - RELEASE_REACH * s
        return z_rel, x_ank

    def _build_runup(self):
        v, vz, fz = self.v, max(self.vz, 0.3), self.floor_z
        standing = self.standing_takeoff or v < 0.6
        x_rel = self.takeoff_x
        leg_angle = self.lean + (0.0 if standing else math.radians(min(12.0, 4.0 * v)))
        z_rel, x_ank = self._release_geometry(x_rel, leg_angle)
        if self.platform is not None:              # the toes may not hang over the platform edge
            lo, hi = self.platform_x
            x_ank_ok = min(max(x_ank, lo + 0.1 - perf.HEEL_X), hi - perf.TOE_X)
            if abs(x_ank_ok - x_ank) > 1e-6:
                self.warnings.append(f"takeoff_x {x_rel:.2f} puts the feet off the platform; take-off moved to "
                                     f"x={x_rel + x_ank_ok - x_ank:.2f}")
                x_rel += x_ank_ok - x_ank
                x_ank = x_ank_ok
        self.takeoff_x_used = x_rel
        rise = 0.28
        T_push = min(max(2 * rise / vz, 0.12), 0.30)
        rise = vz * T_push / 2
        z_bot = z_rel - rise
        if standing:
            if abs(self.start_x - x_rel) > 1.5 and not self.standing_takeoff:
                self.warnings.append(f"approach_speed {v:.2f} m/s is a standing take-off; the run-up from "
                                     f"start_x {self.start_x:.2f} is skipped")
            T_prep = 0.5
            t_push0 = T_STAND + T_prep
            t_rel = t_push0 + T_push
            T_hop = T_acc = T_c = 0.0
        else:
            T_hop = min(max(0.55 / step_freq(v), 0.14), 0.24)
            D_run = (x_rel - self.start_x) - v * (T_push + T_hop)
            T_acc = v / 3.0
            if D_run < v * T_acc / 2:
                T_acc = 2 * D_run / v if D_run > 0 else 0.0
                if T_acc <= 0 or v / T_acc > 6.0:
                    T_acc = v / 6.0
                    D_run = v * T_acc / 2
                    new_start = x_rel - v * (T_push + T_hop) - D_run
                    self.warnings.append(f"the run-up from start_x {self.start_x:.2f} is too short to reach "
                                         f"{v:.1f} m/s; it starts at x={new_start:.2f} instead")
                T_c = 0.0
            else:
                T_c = (D_run - v * T_acc / 2) / v
            t_rel = T_STAND + T_acc + T_c + T_hop + T_push
        K = int(round(t_rel / DT))
        shift = K * DT - t_rel                      # snap release onto the 2 ms grid (stand a bit longer)
        t_run0 = T_STAND + shift
        t_rel = K * DT
        t_push0 = t_rel - T_push
        t_hop0 = t_push0 - T_hop
        t = np.arange(K + 1) * DT
        self.t_runup, self.t_takeoff, self.K = t_run0, t_rel, K
        self.t_push0, self.t_hop0 = t_push0, t_hop0
        n = K + 1

        # ---- pelvis x and speed
        x = np.empty(n)
        vel = np.empty(n)
        if standing:
            x_bot = x_rel - v * T_push / 2
            x_stand = x_ank - perf.HIP_PITCH_OFFSET[0]
            for i, ti in enumerate(t):
                if ti <= t_run0:
                    x[i], vel[i] = x_stand, 0.0
                elif ti <= t_push0:
                    x[i] = _hermite(t_run0, t_push0, x_stand, x_bot, 0.0, 0.0, ti)
                    vel[i] = 0.0
                else:
                    x[i] = _hermite(t_push0, t_rel, x_bot, x_rel, 0.0, v, ti)
                    vel[i] = v * (ti - t_push0) / T_push
            start = x_stand
        else:
            start = x_rel - v * (T_push + T_hop + T_c) - v * T_acc / 2
            for i, ti in enumerate(t):
                if ti <= t_run0:
                    x[i], vel[i] = start, 0.0
                elif ti <= t_run0 + T_acc:
                    u = (ti - t_run0) / T_acc
                    x[i], vel[i] = start + v * T_acc * smooth_int(u), v * smooth(u)
                else:
                    x[i], vel[i] = start + v * T_acc / 2 + v * (ti - t_run0 - T_acc), v
        self.start_x_used = start

        # ---- pelvis pitch: run lean, a slight lean back into the plant, then the take-off lean
        lean_run = np.radians(3.0 + 3.0 * vel)
        crouch = math.radians(max(math.degrees(self.lean), 0.0) + 20.0)
        plant = min(self.lean, 0.0) - math.radians(5.0)
        pitch = np.empty(n)
        for i, ti in enumerate(t):
            if standing:
                if ti <= t_run0:
                    pitch[i] = 0.0
                elif ti <= t_push0:
                    pitch[i] = crouch * smooth((ti - t_run0) / (t_push0 - t_run0))
                else:
                    pitch[i] = crouch + (self.lean - crouch) * smooth((ti - t_push0) / T_push)
            elif ti <= t_push0:
                w = smooth((ti - t_hop0) / (t_push0 - t_hop0)) if ti > t_hop0 else 0.0
                pitch[i] = lean_run[i] + (plant - lean_run[i]) * w
            else:
                pitch[i] = plant + (self.lean - plant) * smooth((ti - t_push0) / T_push)

        # ---- footholds: each leg is a list of stances (t0, t1, ankle_x); swings fill the gaps
        hip_dx = lambda pi: perf.hip_world_xz((0.0, 0.0), pi)[0]
        x_stand_ank = x[0] + hip_dx(pitch[0])
        st = {"left": [], "right": []}
        swing_h = {"left": [], "right": []}
        t_feet = t_push0 - 0.04 if not standing else 0.0
        if standing:
            for leg in st:
                st[leg].append((0.0, t_rel + 1.0, x_ank))
        else:
            phi = np.zeros(n)
            m = (t > t_run0) & (t <= t_hop0)
            phi[1:] = np.cumsum(np.where(m[1:], [step_freq(vv) for vv in vel[1:]], 0.0) * DT)
            beta_end = stance_frac(v)
            phi_end = float(phi[np.searchsorted(t, t_hop0)])
            N = max(1, int(round(phi_end - 2 * beta_end)))
            k = (N + 2 * beta_end) / max(phi_end, 1e-6)
            phi *= k
            mono = phi + np.arange(n) * 1e-12          # strictly increasing for interpolation
            tof = lambda ph: float(np.interp(ph, mono, t))
            xat = lambda tt: float(np.interp(tt, t, x))
            pat = lambda tt: float(np.interp(tt, t, pitch))
            vat = lambda tt: float(np.interp(tt, t, vel))
            first_lo = {}
            for i in range(N):
                leg = "right" if i % 2 == 0 else "left"
                td = tof(i + 1)
                beta = stance_frac(vat(td))
                end = t_hop0 if i == N - 1 else tof(i + 1 + 2 * beta)
                mid = 0.5 * (td + end)
                ax = xat(mid) + hip_dx(pat(mid))
                if leg not in first_lo:
                    first_lo[leg] = tof(max(0.0, i + 1 - 2 * (1 - beta)))
                st[leg].append((td, end, ax))
                swing_h[leg].append(min(0.05 + 0.035 * vat(td), 0.18))
            for leg in ("right", "left"):
                lo0 = first_lo.get(leg, t_hop0 - 0.6 / step_freq(v))
                st[leg].insert(0, (0.0, max(min(lo0, t_hop0), t_run0), x_stand_ank))
                swing_h[leg].insert(0, 0.0)
                st[leg].append((t_feet, t_rel + 1.0, x_ank))
                swing_h[leg].append(0.12)
            for leg in st:   # a stance must end before the next begins
                fixed = [st[leg][0]]
                for s0, s1, ax in st[leg][1:]:
                    p0, p1, px = fixed[-1]
                    if p1 > s0 - 0.04:
                        fixed[-1] = (p0, s0 - 0.04, px)
                    fixed.append((s0, s1, ax))
                st[leg] = fixed
        self.stances = st

        def foot(leg, ti):
            """(ankle x, ankle z, sole pitch, in_stance) of one leg at time ti."""
            S = st[leg]
            for j, (s0, s1, ax) in enumerate(S):
                if s0 <= ti <= s1:
                    return ax, fz + perf.ANKLE_HEIGHT, 0.0, True
                if j + 1 < len(S) and s1 < ti < S[j + 1][0]:
                    n0, n1, nx = S[j + 1]
                    u = (ti - s1) / (n0 - s1)
                    h = swing_h[leg][j + 1] if j + 1 < len(swing_h[leg]) else 0.1
                    spd = min(1.0, float(np.interp(ti, t, vel)) / 3.0)
                    return (ax + (nx - ax) * smooth(u), fz + perf.ANKLE_HEIGHT + h * math.sin(math.pi * u),
                            0.35 * spd * math.sin(math.pi * u) * (1.0 - u), False)
            return S[-1][2], fz + perf.ANKLE_HEIGHT, 0.0, True

        F = {leg: [foot(leg, ti) for ti in t] for leg in ("left", "right")}

        # ---- pelvis height: nominal profile, kept inside the reach of every planted foot
        z_stand = fz + perf.ANKLE_HEIGHT + 0.797 - perf.HIP_PITCH_OFFSET[2]
        z = np.empty(n)
        zn_hop0 = None                                  # running height when the gather hop starts
        for i, ti in enumerate(t):
            if standing:
                if ti <= t_run0:
                    z[i] = z_stand
                elif ti <= t_push0:
                    z[i] = _hermite(t_run0, t_push0, z_stand, z_bot, 0.0, 0.0, ti)
                else:
                    z[i] = _hermite(t_push0, t_rel, z_bot, z_rel, 0.0, vz, ti)
                continue
            if ti >= t_push0:
                z[i] = _hermite(t_push0, t_rel, z_bot, z_rel, 0.0, vz, ti)
                continue
            vv = vel[i]
            zn = z_stand - (0.02 + 0.035 * min(vv, 4.0)) * smooth((ti - t_run0) / 0.3)
            bob = 0.0
            for leg in ("left", "right"):
                for s0, s1, _ in st[leg][1:-1]:
                    if s0 <= ti <= s1 and s1 > s0:
                        bob = max(bob, math.sin(math.pi * (ti - s0) / (s1 - s0)))
            zn -= 0.012 * min(vv, 4.0) * bob
            if ti >= t_hop0:
                zn = _hermite(t_hop0, t_push0, zn if zn_hop0 is None else zn_hop0, z_bot, 0.0, 0.0, ti)
            else:
                zn_hop0 = zn
            z[i] = zn
        # reach limits of planted feet
        zmax = np.full(n, np.inf)
        for leg in ("left", "right"):
            for i in range(n):
                ax, az, _, planted = F[leg][i]
                if planted:
                    zmax[i] = min(zmax[i], perf.max_pelvis_z((ax, az), x[i], pitch[i]))
        z = np.minimum(z, zmax)
        if n > 9:
            zs = np.convolve(np.pad(z, 4, mode="edge"), np.ones(9) / 9, mode="valid")
            keep = t >= t_push0 - 0.02          # the push itself is already smooth: keep its exact end state
            z = np.where(keep, z, np.minimum(zs, zmax))

        # ---- joints: leg IK + arms
        q = np.zeros((n, len(perf.H1_JOINTS)))
        J = {nme: i for i, nme in enumerate(perf.H1_JOINTS)}
        thigh = np.zeros((n, 2))
        self.reach_error = 0.0
        for i in range(n):
            for k_leg, leg in enumerate(("left", "right")):
                ax, az, psi, _ = F[leg][i]
                hx, hz = perf.hip_world_xz((x[i], z[i]), pitch[i])
                hip, knee, ank, err = perf.leg_ik((hx, hz), (ax, az), pitch[i], psi)
                self.reach_error = max(self.reach_error, err)
                q[i, J[f"{leg}_hip_pitch"]] = hip
                q[i, J[f"{leg}_knee"]] = knee
                q[i, J[f"{leg}_ankle"]] = ank
                thigh[i, k_leg] = math.degrees(-hip)
            ti = t[i]
            if standing:
                if ti <= t_run0:
                    a = STAND_ARMS
                elif ti <= t_push0:
                    a = blend(STAND_ARMS, BACK_ARMS, smooth((ti - t_run0) / (t_push0 - t_run0)))
                else:
                    a = blend(BACK_ARMS, self.takeoff_arms(), smooth((ti - t_push0) / T_push))
                tor = 0.0
            else:
                swing = 0.5 * (thigh[i, 1] - thigh[i, 0])        # + when the right thigh is forward
                run_arms = dict(l_shoulder_x=8, r_shoulder_x=8, l_elbow=85, r_elbow=85,
                                l_shoulder_y=float(np.clip(10 + swing, -50, 70)),
                                r_shoulder_y=float(np.clip(10 - swing, -50, 70)))
                a = blend(STAND_ARMS, run_arms, smooth((ti - t_run0) / 0.35))
                if ti > t_hop0:
                    a = blend(a, BACK_ARMS, smooth((ti - t_hop0) / max(T_hop, 1e-3)))
                if ti > t_push0:
                    a = blend(BACK_ARMS, self.takeoff_arms(), smooth((ti - t_push0) / T_push))
                tor = float(np.clip(0.2 * (thigh[i, 0] - thigh[i, 1]), -10, 10)) * (1.0 - smooth((ti - t_hop0) / 0.1))
            self.rig.arm_vector({**a, "torso": tor}, q[i])
        q = np.clip(q, self.rig.lo, self.rig.hi)
        self.t_grid, self.x, self.z, self.pitch, self.q = t, x, z, pitch, q
        self.y = np.full(n, self.lane_y)
        if self.reach_error > 0.01:
            self.warnings.append(f"run-up leg IK out of reach by {self.reach_error * 100:.1f} cm")

    # ------------------------------------------------------------------ used by the simulator
    def kinematic(self, k):
        """Root position, root quaternion and joint vector at grid step k (0..K)."""
        th = self.pitch[k]
        return (np.array([self.x[k], self.y[k], self.z[k]]),
                np.array([math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]), self.q[k])

    def release_joint_velocity(self):
        """Joint velocities at release (backward difference of the run-up), capped."""
        qd = (self.q[-1] - self.q[-2]) / DT
        cap = np.array([LEG_QDOT_MAX if j in perf.LEG_JOINTS else ARM_QDOT_MAX for j in perf.H1_JOINTS])
        return np.clip(qd, -cap, cap)

    def prepare(self, model, data, performer):
        """Flight-time estimate and the take-off angular momentum, from the compiled model."""
        import mujoco
        pos, quat, qj = self.kinematic(self.K)
        q = performer
        data.qpos[q.free_qadr:q.free_qadr + 3] = pos
        data.qpos[q.free_qadr + 3:q.free_qadr + 7] = quat
        data.qpos[q.qadr] = qj
        mujoco.mj_kinematics(model, data)
        com0, _ = q.com_inertia(model, data)
        self.com0 = com0
        # flight time to the landing surface (mat top if the ballistic landing point is over the mat)
        hl = self.landing_com_height()
        surf = 0.0
        if self.mat is not None:
            T = self._flight_time(com0[2], self.mat["top"] + hl)
            xl = com0[0] + self.v * T
            if scene.inside_footprint((xl, self.lane_y), self.mat["pos"], self.mat["size"], self.mat["yaw"], margin=0.4):
                surf = self.mat["top"]
        self.landing_surface_z = surf
        self.T_f = self._flight_time(com0[2], surf + hl)
        # whole-body inertia along the flight pose schedule (root attitude as at release)
        us = np.linspace(0.0, 1.0, 13)
        inv = []
        for u in us:
            data.qpos[q.qadr] = self.flight_targets(u * self.T_f)
            mujoco.mj_kinematics(model, data)
            _, I = q.com_inertia(model, data)
            inv.append(1.0 / np.diag(I))
        inv = np.array(inv)
        integral = np.trapezoid(inv, us * self.T_f, axis=0)       # s / (kg m^2), per world axis
        rot = np.radians(np.array(self.rotation_deg(), float))
        self.L_target = rot / integral
        self.v_com_target = np.array([self.v, 0.0, self.vz])
        data.qpos[q.qadr] = qj
        mujoco.mj_kinematics(model, data)

    def _flight_time(self, z0, z1):
        disc = self.vz ** 2 + 2 * G * (z0 - z1)
        return (self.vz + math.sqrt(disc)) / G if disc > 0 else max(self.vz / G, 0.05)

    def flight_targets(self, s):
        """Joint targets s seconds after release: ease from the release pose into flight_pose(u)."""
        T = self.T_f or 0.8
        tgt = self.rig.vector(self.flight_pose(s / T))
        w = smooth(s / 0.1)
        return (1.0 - w) * self.q[-1] + w * tgt

    def landing_targets(self, s, from_q):
        """Joint targets s seconds after the first touch, eased from the pose held at touchdown."""
        tgt = self.rig.vector(self.landing_pose(s))
        w = smooth(s / 0.05)
        return (1.0 - w) * from_q + w * tgt

    def landing_gain_scale(self):
        return 1.0 - 0.75 * float(self.p["landing_compliance"])

    def summary(self):
        return {"type": self.name, "t_runup": round(self.t_runup, 3), "t_takeoff": round(self.t_takeoff, 3),
                "start_x_used": round(self.start_x_used, 3), "takeoff_x_used": round(self.takeoff_x_used, 3),
                "flight_time_est": None if self.T_f is None else round(self.T_f, 3),
                "warnings": list(self.warnings)}


# ----------------------------------------------------------------------------- the library
class BarrelRoll(Choreo):
    """Running two-foot take-off, 360 deg about the travel axis (arms high so the roll reads), feet-first."""
    name = "barrel_roll"


class FrontFlip(Choreo):
    """Running take-off, tucked forward somersault, arms reaching for the shins, feet-first landing."""
    name = "front_flip"

    def flight_pose(self, u):
        p = self.p
        a = p["tuck"] * window(u, p["tuck_start"], p["tuck_end"])
        pose = {**legs(8.0 + 82.0 * a, 10.0 + 105.0 * a), **arms(p["arm_reach_deg"], 20.0, 10.0 + 50.0 * a)}
        land = smooth((u - max(p["tuck_end"], 0.55)) / max(0.15, 1.0 - max(p["tuck_end"], 0.55)))
        return blend(pose, self.landing_prep_pose(), land * (1.0 - a))


class BackFlip(FrontFlip):
    """Standing (or slow) take-off, arms swing up, tucked backward somersault, feet-first landing."""
    name = "back_flip"
    standing_takeoff = True

    def takeoff_arms(self):
        return arms(160.0, 15.0, 15.0)


class Vault(Choreo):
    """Running take-off over a low obstacle: knees driven up (tuck), arms forward, no rotation, feet-first."""
    name = "vault"

    def takeoff_arms(self):
        return arms(max(self.p["arm_reach_deg"], 60.0), 20.0, 20.0)


class DiveRoll(Choreo):
    """Running dive: body pitches forward, arms reach overhead to meet the mat, then tuck to roll."""
    name = "dive_roll"
    feet_first = False

    def flight_pose(self, u):
        p = self.p
        a = p["tuck"] * window(u, p["tuck_start"], p["tuck_end"])
        stretch = {**legs(-5.0, 8.0), **arms(p["arm_reach_deg"], 20.0, 10.0)}
        tuck = {**legs(85.0, 110.0), **arms(110.0, 25.0, 60.0)}
        return blend(stretch, tuck, a)

    def landing_pose(self, s):
        roll = {**legs(85.0, 110.0), **arms(100.0, 25.0, 70.0)}
        rest = {**legs(40.0, 60.0), **arms(30.0, 30.0, 40.0)}
        return blend(roll, rest, smooth((s - 0.8) / 0.8))

    def landing_com_height(self):
        return 0.55


class HighFall(Choreo):
    """Stand on the stunt target / platform, step off, fall rotating back onto the mat (arms out)."""
    name = "high_fall"
    feet_first = False
    standing_takeoff = True

    def _setup(self):
        ob = scene.launch_platform(self.plan)
        if ob is None:
            self.warnings.append("high_fall: no platform under the take-off point; stepping off the floor")
            return
        self.platform = ob
        self.floor_z = scene.obstacle_top(ob)
        yaw = math.radians(ob["yaw_deg"])
        hx = ob["size"][0] if ob["kind"] == "box" else ob["size"][0]
        hy = ob["size"][1] if ob["kind"] == "box" else ob["size"][0]
        ext = abs(math.cos(yaw)) * hx + abs(math.sin(yaw)) * hy        # half extent along the lane
        self.platform_x = (ob["pos"][0] - ext, ob["pos"][0] + ext)

    def takeoff_arms(self):
        return arms(self.p["arm_reach_deg"] * 0.6, 60.0, 15.0)

    def flight_pose(self, u):
        p = self.p
        a = p["tuck"] * window(u, p["tuck_start"], p["tuck_end"])
        spread = {**legs(15.0, 20.0), **arms(p["arm_reach_deg"] * 0.6, 70.0, 15.0)}
        tuck = {**legs(70.0, 90.0), **arms(60.0, 30.0, 60.0)}
        return blend(spread, tuck, a)

    def landing_pose(self, s):
        return {**legs(30.0, 40.0), **arms(70.0, 80.0, 20.0)}

    def landing_com_height(self):
        return 0.2


LIBRARY = {"barrel_roll": BarrelRoll, "dive_roll": DiveRoll, "front_flip": FrontFlip, "back_flip": BackFlip,
           "vault": Vault, "high_fall": HighFall}


def get(stunt_type):
    """Choreography class for a stunt type; 'custom' falls back to barrel_roll with a warning."""
    if stunt_type in LIBRARY:
        return LIBRARY[stunt_type]
    if stunt_type not in (None, "", "custom"):
        raise ValueError(f"unknown stunt type {stunt_type!r}; library: {sorted(LIBRARY)} (or 'custom')")
    sys.stderr.write("warning: stunt type 'custom' has no library choreography; using barrel_roll. "
                     "Edit choreography() in the stunt script to perform the brief.\n")
    return BarrelRoll
