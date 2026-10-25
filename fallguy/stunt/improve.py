"""Stage 4b: apply Argon's critique changes to make the next take, simulate it, critique it, write compare.json.

python -m fallguy.stunt improve --stunt DIR --from v1 --to v2 [--no-critique]
"""
import copy
import json
import math
import os
import re

from .common import rj, say, wj
from .simulate import run_take, take_choreography


def apply_changes(stunt, changes):
    """Returns (new stunt, applied [{name/path, old, new, reason}], rejected [...]). Tunable names map to their
    path and are clamped to the tunable range; raw paths must lead to an existing number in stunt.phases/target."""
    from ..engine import params as P
    st = copy.deepcopy(stunt)
    tun = {t["name"]: t for t in st.get("tunables") or []}
    applied, rejected = [], []
    for c in changes or []:
        name, path = (c.get("name") or "").strip(), (c.get("path") or "").strip()
        try:
            val = float(c.get("value"))
            if not math.isfinite(val):
                raise ValueError("not finite")
        except (TypeError, ValueError):
            rejected.append({**c, "why": "value is not a number"})
            continue
        lo = hi = None
        if name in tun:
            path, lo, hi = tun[name]["path"], tun[name]["min"], tun[name]["max"]
        elif not path:
            rejected.append({**c, "why": f"unknown tunable {name!r} and no path"})
            continue
        if not (path.startswith("phases[") or path.startswith("target.") or path.startswith("start.")):
            rejected.append({**c, "why": f"path {path!r} is outside phases/target/start"})
            continue
        try:
            old = P.get_path(st, path)
        except Exception as e:                                   # noqa: BLE001
            rejected.append({**c, "why": f"path {path!r} does not resolve: {e}"})
            continue
        if isinstance(old, bool) or not isinstance(old, (int, float)):
            rejected.append({**c, "why": f"{path} is not a number"})
            continue
        new = val if lo is None else min(max(val, lo), hi)
        P.set_path(st, path, round(new, 4))
        m = re.match(r"^phases\[(\d+)\]\.path\[0\]\[([01])\]$", path)
        if m and all(p.get("kind") == "hold" for p in st["phases"][:int(m.group(1))]):
            st["start"]["pos"][int(m.group(2))] = round(new, 4)        # the first move starts at the start
        applied.append({"name": name or None, "path": path, "old": old, "new": round(new, 4), "reason": c.get("reason")})
    # keep the move path chain consistent (each move starts where the previous one ended) and the zone at the end
    from .. import argon as A
    W = []
    st2 = A.normalize_stunt({**st, "target": st.get("target")}, st.get("crash_mats") or [], W)
    if st.get("crash_mats"):
        st2["crash_mats"] = st["crash_mats"]
    if (st.get("target") or {}).get("kind") == "zone":
        from ..engine import scene
        end = scene.stunt_points(scene.normalize_stunt(st2))[-1]
        st2["target"]["pos"] = [round(float(end[0]), 3), round(float(end[1]), 3)]
    return st2, applied, rejected


def _num(m, *keys):
    x = m
    for k in keys:
        x = (x or {}).get(k) if isinstance(x, dict) else None
    return x


ROWS = [("head_g_peak", ("gate1", "head_g_peak")), ("hic15", ("gate1", "hic15")),
        ("lateral_g_peak", ("gate1", "lateral_g_peak")), ("min_clearance_cm", ("gate1", "min_clearance_cm")),
        ("hazard_collisions", ("gate1", "hazard_collisions")), ("framing_pct", ("gate1", "framing_pct")),
        ("landing_error_m", ("gate1", "landing", "error_m")), ("stunt_score", ("score", "stunt_score")),
        ("knee_torque_nm", ("biomech", "knee_torque", "value")), ("hip_torque_nm", ("biomech", "hip_torque", "value")),
        ("ankle_torque_nm", ("biomech", "ankle_torque", "value")), ("grf_x_bw", ("biomech", "grf", "x_bw")),
        ("lumbar_compression_n", ("biomech", "lumbar_compression", "value")),
        ("worst_link_index", ("biomech", "worst_link", "index"))]


def compare(stunt_dir, a, b):
    sd = os.path.abspath(stunt_dir)
    out = {"from": a, "to": b, "rows": [], "critique": {}}
    ma, mb = rj(os.path.join(sd, a, "metrics.json")), rj(os.path.join(sd, b, "metrics.json"))
    for name, keys in ROWS:
        va, vb = _num(ma, *keys), _num(mb, *keys)
        d = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = round(abs(vb) - abs(va), 3) if "torque" in name else round(vb - va, 3)
        out["rows"].append({"metric": name, a: va, b: vb, "delta": d})
    out["gate1_passed"] = {a: _num(ma, "gate1", "passed"), b: _num(mb, "gate1", "passed")}
    out["failures"] = {a: _num(ma, "gate1", "failures"), b: _num(mb, "gate1", "failures")}
    out["worst_link"] = {a: _num(ma, "biomech", "worst_link"), b: _num(mb, "biomech", "worst_link")}
    for t in (a, b):
        p = os.path.join(sd, t, "critique.json")
        if os.path.isfile(p):
            c = rj(p)
            out["critique"][t] = {"overall_score": c.get("overall_score"), "approve": c.get("approve"),
                                  "brief_match": (c.get("brief_match") or {}).get("matches"),
                                  "plausibility": (c.get("motion_plausibility") or {}).get("score"),
                                  "summary": c.get("summary")}
    ch = os.path.join(sd, b, "changes.json")
    if os.path.isfile(ch):
        out["changes"] = rj(ch)
    wj(os.path.join(sd, "compare.json"), out)
    return out


def improve(stunt_dir, src="v1", dst="v2", critique_after=True, render=True):
    from .critique import critique
    sd = os.path.abspath(stunt_dir)
    cp = os.path.join(sd, src, "critique.json")
    if not os.path.isfile(cp):
        say(f"no critique for {src} yet; running it")
        critique(sd, src)
    cr = rj(cp)
    st = take_choreography(sd, src)
    new, applied, rejected = apply_changes(st, cr.get("changes"))
    for a in applied:
        say(f"{dst}: {a['name'] or a['path']} {a['old']} -> {a['new']} ({a['reason']})")
    for r in rejected:
        say(f"{dst}: rejected change {r}")
    td = os.path.join(sd, dst)
    os.makedirs(td, exist_ok=True)
    wj(os.path.join(td, "choreography.json"), new)
    wj(os.path.join(td, "changes.json"), {"from": src, "applied": applied, "rejected": rejected})
    if os.path.isfile(os.path.join(td, "params.json")):
        os.remove(os.path.join(td, "params.json"))
    run_take(sd, dst, render=render)
    if critique_after and render:
        critique(sd, dst)
    out = compare(sd, src, dst)
    S = rj(os.path.join(sd, "stunt.json"))
    S["takes"] = sorted(set((S.get("takes") or []) + [src, dst]))
    wj(os.path.join(sd, "stunt.json"), S)
    return out
