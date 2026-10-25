"""Biomechanics of one simulated take -> biomech.json (CONTRACT.md "Biomechanics").

Every value says how it was obtained:
  measured           free MuJoCo physics after take-off (actuator_force, contact forces, cfrc_int)
  inverse_dynamics   scripted ground phases: mj_inverse on the logged kinematics (2 ms finite differences,
                     lightly smoothed), with the floating-base residual explained by ground wrenches at the
                     stance feet (least squares on the foot Jacobians); joint torques = qfrc_inverse - J_c^T f_c.

Neck and lumbar loads are proxies on the H1 (it has no neck and a single torso link): neck = 4.5 kg human head
mass x (head-site acceleration + g) along the torso axis; lumbar = force the pelvis transmits to the torso
subtree (cfrc_int in free phases, subtree momentum rate + weight in scripted ones), along the torso axis.
Torques are on the 51 kg H1, not rescaled to a human.

python -m fallguy.biomech --take DIR   (needs scene.xml, trajectory.npz, events.json from engine.sim)
"""
import argparse
import json
import math
import os

import mujoco
import numpy as np

G0 = 9.81
HEAD_MASS = 4.5
OUT_HZ = 60
# literature limits (CONTRACT.md); "reference" = typical peak tolerated joint moment, not an injury criterion
LIMITS = {
    "hic15": {"value": 700.0, "unit": "", "source": "FMVSS 208 / NHTSA HIC15 limit"},
    "head_g_severe": {"value": 80.0, "unit": "g", "source": "FMVSS 218 / 3 ms head criterion (severe)"},
    "head_g_target": {"value": 8.0, "unit": "g", "source": "FALLGUY blueprint conservative stunt target"},
    "neck_compression": {"value": 4000.0, "unit": "N", "source": "Nij critical compression (Eppinger 1999), proxy"},
    "neck_tension": {"value": 4170.0, "unit": "N", "source": "Nij critical tension (Eppinger 1999), proxy"},
    "lumbar_compression": {"value": 3400.0, "unit": "N", "source": "NIOSH action limit L5/S1"},
    "knee_moment": {"value": 200.0, "unit": "N*m", "source": "knee moment reference (CONTRACT)"},
    "hip_moment": {"value": 300.0, "unit": "N*m", "source": "peak hip moment in sprinting/landing, reference"},
    "ankle_moment": {"value": 250.0, "unit": "N*m", "source": "peak ankle moment in sprinting/landing, reference"},
    "shoulder_moment": {"value": 100.0, "unit": "N*m", "source": "shoulder moment reference"},
    "elbow_moment": {"value": 70.0, "unit": "N*m", "source": "elbow moment reference"},
    "torso_moment": {"value": 250.0, "unit": "N*m", "source": "trunk axial/flexion moment reference"},
    "impact_grf_bw": {"value": 10.0, "unit": "x BW", "source": "impact GRF flag (> 10 x body weight)"},
    "limb_contact": {"value": 4000.0, "unit": "N", "source": "long-bone fracture order of magnitude, reference"},
}
JOINT_LIMIT_KEY = {"hip_yaw": "hip_moment", "hip_roll": "hip_moment", "hip_pitch": "hip_moment", "knee": "knee_moment",
                   "ankle": "ankle_moment", "torso": "torso_moment", "shoulder_pitch": "shoulder_moment",
                   "shoulder_roll": "shoulder_moment", "shoulder_yaw": "shoulder_moment", "elbow": "elbow_moment"}


def colour(idx):
    return "green" if idx < 0.5 else ("amber" if idx < 1.0 else "red")


def rgba(idx):
    """Stress colour ramp: green < 0.5, amber < 1, red >= 1 (smooth inside each band)."""
    if idx < 0.5:
        return (0.15, 0.75, 0.3, 1.0)
    if idx < 1.0:
        return (1.0, 0.68, 0.1, 1.0)
    return (0.92, 0.12, 0.1, 1.0)


def _hic15(g, dt):
    from .engine import metrics as M
    return float(M._hic15(np.asarray(g, float), dt))


def _gauss(x, sigma):
    if sigma <= 0.5 or len(x) < 3:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(pad[:, j], k, mode="valid")
    return out


def compute(take_dir, write=True):
    from .engine import performer as perf
    ev = json.load(open(os.path.join(take_dir, "events.json"), encoding="utf-8"))
    z = np.load(os.path.join(take_dir, "trajectory.npz"))
    t, q = z["t"], z["qpos"]
    dt = float(ev.get("dt", 0.002))
    n = len(t)
    model = mujoco.MjModel.from_xml_path(os.path.join(take_dir, "scene.xml"))
    data = mujoco.MjData(model)
    pf = perf.Performer(model)
    k_rel = int(z["k_release"]) if "k_release" in z.files and int(z["k_release"]) >= 0 else n
    k_rel = min(k_rel, n)
    nv = model.nv
    bw = pf.mass * G0
    B = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    torso, pelvis = B("torso_link"), pf.root
    feet = [B("left_ankle_link"), B("right_ankle_link")]
    jnames = perf.H1_JOINTS
    fv = pf.free_vadr

    # ------------------------------------------------ scripted phases: inverse dynamics
    tau = np.zeros((n, len(jnames)))
    grf = np.zeros((n, 3))
    foot_f = np.zeros((n, 2))
    residual = np.zeros(n)
    lumbar = np.zeros((n, 3))
    torso_z = np.zeros((n, 3))
    sub_com = np.zeros((n, 3))
    m_sub = float(model.body_subtreemass[torso])
    for i in range(n):
        data.qpos[:] = q[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        torso_z[i] = data.xmat[torso].reshape(3, 3)[:, 2]
        sub_com[i] = data.subtree_com[torso]
    if k_rel > 2:
        qv = np.zeros((k_rel, nv))
        for i in range(1, k_rel):
            mujoco.mj_differentiatePos(model, qv[i], dt, q[i - 1], q[i])
        qv[0] = qv[1]
        qv = _gauss(qv, 0.012 / dt)
        qa = np.gradient(qv, dt, axis=0)
        qa = _gauss(qa, 0.008 / dt)
        weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "root_weld")
        if weld >= 0:
            model.eq_active0[weld] = 0
            data.eq_active[weld] = 0
        pf.set_collisions(model, False)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
        for i in range(k_rel):
            data.qpos[:] = q[i]
            data.qvel[:] = qv[i]
            data.qacc[:] = qa[i]
            mujoco.mj_inverse(model, data)
            f = data.qfrc_inverse.copy()
            root = f[fv:fv + 6]
            stance = []
            for s, b in enumerate(feet):
                zf = data.xpos[b][2]
                if zf < perf.ANKLE_HEIGHT + 0.04:
                    stance.append((s, b))
            if stance:
                Js = []
                for s, b in stance:
                    mujoco.mj_jacBody(model, data, jacp, jacr, b)
                    Js.append(np.vstack([jacp, jacr]).copy())        # 6 x nv per foot
                J = np.vstack(Js)                                     # (6s) x nv
                A = J[:, fv:fv + 6].T                                 # 6 x 6s
                w, *_ = np.linalg.lstsq(A, root, rcond=None)
                gen = J.T @ w
                fj = f - gen
                residual[i] = float(np.linalg.norm(root - A @ w))
                for j, (s, b) in enumerate(stance):
                    fw = w[6 * j:6 * j + 3]
                    grf[i] += fw
                    foot_f[i, s] = float(np.linalg.norm(fw))
            else:
                fj = f
                residual[i] = float(np.linalg.norm(root[:3]))
            tau[i] = fj[pf.vadr]
        # lumbar: pelvis -> torso subtree force = m (a_subcom + g)
        a_sub = np.gradient(np.gradient(_gauss(sub_com[:k_rel], 0.01 / dt), dt, axis=0), dt, axis=0)
        lumbar[:k_rel] = m_sub * (a_sub + np.array([0.0, 0.0, G0]))
    # ------------------------------------------------ free phases: measured
    if k_rel < n:
        ft = z["free_tau"][k_rel:n]
        ok = np.isfinite(ft).all(1)
        tau[k_rel:n][ok] = ft[ok]
        g = z["free_grf"][k_rel:n]
        grf[k_rel:n] = g
        bf = z["free_body_force"][k_rel:n]
        foot_f[k_rel:n, 0], foot_f[k_rel:n, 1] = bf[:, feet[0]], bf[:, feet[1]]
        ci = z["free_cfrc_int"][k_rel:n]
        ok2 = np.isfinite(ci[:, torso, 3]).astype(bool)
        lumbar[k_rel:n][ok2] = ci[ok2, torso, 3:6]
    body_contact = np.zeros((n, model.nbody))
    if k_rel < n:
        body_contact[k_rel:n] = z["free_body_force"][k_rel:n]

    # ------------------------------------------------ head, neck
    head_pos = z["head_pos"]
    hp = _gauss(head_pos, 0.004 / dt)
    a_head = np.zeros_like(hp)
    a_head[1:-1] = (hp[2:] - 2 * hp[1:-1] + hp[:-2]) / (dt * dt)
    if k_rel < n:                                                # no finite difference across the release
        a_head[max(k_rel - 1, 0):min(k_rel + 2, n)] = a_head[max(k_rel - 2, 0)]
    f_neck = HEAD_MASS * (a_head + np.array([0.0, 0.0, G0]))
    neck_axial = np.einsum("ij,ij->i", f_neck, torso_z)          # + compression (neck pushes the head up)
    lumbar_axial = np.einsum("ij,ij->i", lumbar, torso_z)
    head_g15, head_g = z["head_g15"], z["head_g"]
    hic = _hic15(head_g, dt)
    grf_bw = np.linalg.norm(grf, axis=1) / bw

    # ------------------------------------------------ per-link stress index
    L = {k: v["value"] for k, v in LIMITS.items()}
    lim = np.array([L[JOINT_LIMIT_KEY["torso" if j == "torso" else j.split("_", 1)[1]]] for j in jnames])
    jidx = np.abs(tau) / lim[None, :]
    jcol = {j: i for i, j in enumerate(jnames)}
    links = {}

    def add(body_name, series):
        links[body_name] = np.maximum(links.get(body_name, 0.0), series)

    head_idx = np.maximum(head_g15 / L["head_g_severe"], 0.0)
    neck_idx = np.where(neck_axial >= 0, neck_axial / L["neck_compression"], -neck_axial / L["neck_tension"])
    lumb_idx = np.maximum(lumbar_axial, 0.0) / L["lumbar_compression"]
    add("torso_link", np.maximum.reduce([head_idx, neck_idx, lumb_idx, jidx[:, jcol["torso"]]]))
    add("pelvis", lumb_idx)
    for side in ("left", "right"):
        s = 0 if side == "left" else 1
        for j in ("hip_yaw", "hip_roll", "hip_pitch"):
            add(f"{side}_{j}_link", jidx[:, jcol[f"{side}_{j}"]])
        add(f"{side}_hip_pitch_link", np.maximum(jidx[:, jcol[f"{side}_hip_pitch"]], jidx[:, jcol[f"{side}_hip_roll"]]))
        add(f"{side}_knee_link", jidx[:, jcol[f"{side}_knee"]])
        add(f"{side}_ankle_link", np.maximum(jidx[:, jcol[f"{side}_ankle"]], foot_f[:, s] / (L["impact_grf_bw"] * bw)))
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"):
            add(f"{side}_{j}_link", jidx[:, jcol[f"{side}_{j}"]])
    for name in list(links):
        b = B(name)
        if b >= 0:
            links[name] = np.maximum(links[name], body_contact[:, b] / L["limb_contact"]
                                     if not name.endswith("ankle_link") else 0.0)

    # ------------------------------------------------ output (60 Hz decimation keeps the max inside each bin)
    step = max(1, int(round(1.0 / (OUT_HZ * dt))))
    bins = np.arange(0, n, step)

    def dec(x, how="max"):
        x = np.asarray(x)
        out = []
        for b0 in bins:
            seg = x[b0:b0 + step]
            out.append(seg.max(0) if how == "max" else (seg[np.argmax(np.abs(seg), axis=0), np.arange(seg.shape[1])]
                                                         if seg.ndim == 2 else seg[np.argmax(np.abs(seg))]))
        return np.array(out)

    src = np.where(np.arange(n) < k_rel, "inverse_dynamics", "measured")

    def peak(x, absval=True):
        x = np.asarray(x, float)
        i = int(np.argmax(np.abs(x) if absval else x))
        return {"value": round(float(x[i]), 2), "t": round(float(t[i]), 3), "source": str(src[i])}

    joints = {}
    for j, c in jcol.items():
        pk = peak(tau[:, c])
        key = JOINT_LIMIT_KEY["torso" if j == "torso" else j.split("_", 1)[1]]
        joints[j] = {**pk, "unit": "N*m", "limit": L[key], "index": round(abs(pk["value"]) / L[key], 3),
                     "colour": colour(abs(pk["value"]) / L[key])}
    grp = lambda key: max((joints[f"{s}_{key}"] for s in ("left", "right")), key=lambda d: abs(d["value"]))
    hip = max((joints[f"{s}_{j}"] for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw")),
              key=lambda d: abs(d["value"]))
    i_g = int(np.argmax(head_g15))
    i_grf = int(np.argmax(grf_bw))
    peaks = {
        "head_g": {"value": round(float(head_g15[i_g]), 2), "t": round(float(t[i_g]), 3), "source": str(src[i_g]),
                   "unit": "g (15 ms avg)", "limit": L["head_g_severe"], "target": L["head_g_target"],
                   "index": round(float(head_g15[i_g]) / L["head_g_severe"], 3),
                   "within_target": bool(head_g15[i_g] <= L["head_g_target"])},
        "hic15": {"value": round(hic, 1), "limit": L["hic15"], "index": round(hic / L["hic15"], 3)},
        "knee_torque": {**grp("knee")}, "hip_torque": {**hip}, "ankle_torque": {**grp("ankle")},
        "grf": {"value": round(float(np.linalg.norm(grf[i_grf])), 1), "x_bw": round(float(grf_bw[i_grf]), 2),
                "t": round(float(t[i_grf]), 3), "source": str(src[i_grf]), "unit": "N",
                "flag": bool(grf_bw[i_grf] > L["impact_grf_bw"])},
        "neck_compression": {**peak(np.maximum(neck_axial, 0), False), "unit": "N", "limit": L["neck_compression"],
                             "approximation": True},
        "neck_tension": {**peak(np.maximum(-neck_axial, 0), False), "unit": "N", "limit": L["neck_tension"],
                         "approximation": True},
        "lumbar_compression": {**peak(np.maximum(lumbar_axial, 0), False), "unit": "N", "limit": L["lumbar_compression"],
                               "approximation": True},
    }
    for k in ("neck_compression", "neck_tension", "lumbar_compression"):
        peaks[k]["index"] = round(peaks[k]["value"] / peaks[k]["limit"], 3)
    link_peak = {}
    for name, s in links.items():
        i = int(np.argmax(s))
        link_peak[name] = {"index": round(float(s[i]), 3), "t": round(float(t[i]), 3), "colour": colour(float(s[i])),
                           "source": str(src[i])}
    worst = max(link_peak.items(), key=lambda kv: kv[1]["index"])
    flags = []
    if peaks["head_g"]["value"] > L["head_g_target"]:
        flags.append(f"head {peaks['head_g']['value']} g at t={peaks['head_g']['t']} s exceeds the 8 g stunt target")
    if hic > L["hic15"]:
        flags.append(f"HIC15 {hic:.0f} > 700")
    if peaks["grf"]["flag"]:
        flags.append(f"impact GRF {peaks['grf']['x_bw']} x BW at t={peaks['grf']['t']} s")
    for name, d in link_peak.items():
        if d["index"] >= 1.0:
            flags.append(f"{name} stress index {d['index']} at t={d['t']} s ({d['source']})")
    order = sorted(links)
    out = {
        "version": 1, "dt": dt, "hz": OUT_HZ, "body_mass_kg": round(pf.mass, 2), "body_weight_n": round(bw, 1),
        "t_release": None if k_rel >= n else round(float(t[k_rel]), 4),
        "method": {"scripted": "inverse_dynamics (mj_inverse + least-squares stance-foot wrenches)",
                   "free": "measured (actuator_force, mj_contactForce, cfrc_int)",
                   "neck": "proxy: 4.5 kg head x (head-site accel + g) along the torso axis",
                   "lumbar": "proxy: pelvis->torso subtree force along the torso axis",
                   "note": "torques are on the 51 kg Unitree H1, not scaled to a human"},
        "limits": LIMITS, "peaks": peaks, "joints": joints, "links": link_peak,
        "worst_link": {"name": worst[0], **worst[1]}, "flags": flags,
        "id_residual_n_max": round(float(residual[:k_rel].max()) if k_rel > 0 else 0.0, 1),
        "series": {
            "t": [round(float(x), 4) for x in t[bins]],
            "source": [str(src[b]) for b in bins],
            "head_g": [round(float(x), 3) for x in dec(head_g15)],
            "grf_bw": [round(float(x), 3) for x in dec(grf_bw)],
            "neck_axial_n": [round(float(x), 1) for x in dec(neck_axial, "abs")],
            "lumbar_axial_n": [round(float(x), 1) for x in dec(lumbar_axial, "abs")],
            "joint_names": jnames,
            "joint_torque": [[round(float(v), 1) for v in row] for row in dec(tau, "abs")],
            "link_names": order,
            "link_index": [[round(float(v), 3) for v in row] for row in dec(np.stack([links[k] for k in order], 1))],
        },
    }
    if write:
        with open(os.path.join(take_dir, "biomech.json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def summary_text(b):
    """Compact ground-truth lines for Argon prompts."""
    p = b["peaks"]
    lines = [
        f"head: peak {p['head_g']['value']} g at t={p['head_g']['t']} s ({p['head_g']['source']}); stunt target 8 g, "
        f"severe 80 g; HIC15 {p['hic15']['value']} (limit 700)",
        f"ground reaction: peak {p['grf']['value']} N = {p['grf']['x_bw']} x body weight at t={p['grf']['t']} s "
        f"({p['grf']['source']}); flag above 10 x BW",
        f"knee torque peak {p['knee_torque']['value']} N*m at t={p['knee_torque']['t']} s ({p['knee_torque']['source']}; "
        f"ref 200); hip {p['hip_torque']['value']} N*m (ref 300); ankle {p['ankle_torque']['value']} N*m (ref 250)",
        f"neck compression proxy {p['neck_compression']['value']} N (limit 4000), tension {p['neck_tension']['value']} N "
        f"(limit 4170); lumbar compression proxy {p['lumbar_compression']['value']} N (NIOSH 3400)",
        f"worst link: {b['worst_link']['name']} stress index {b['worst_link']['index']} ({b['worst_link']['colour']}) at "
        f"t={b['worst_link']['t']} s",
    ]
    if b["flags"]:
        lines.append("flags: " + "; ".join(b["flags"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="biomechanics of a simulated take")
    ap.add_argument("--take", required=True)
    a = ap.parse_args(argv)
    b = compute(a.take)
    print(summary_text(b))


if __name__ == "__main__":
    main()
