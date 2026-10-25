"""plan.json (contract schema) -> MJCF for the stunt scene, plus the legacy prototype-plan converter.

The room, curtain, couches, chairs, seated people, speakers, ceiling fixtures and screens are ported from
barrel_roll.env_geoms / build_xml. Unlike the prototype nothing is dropped for a keep-clear zone: everything
stays in the scene and all set dressing counts as a hazard. Every scene geom is named "<object>" or
"<object>__<part>", so contacts and clearances report the object id.

Contacts: the floor and furniture are hard; crash mats (and "foam" obstacles) are soft so a feet-first
landing from about 1.5-2 m onto a 0.3 m mat stays well under 8 G at the head. Scene geoms have contact
priority 1, so their material friction / softness wins over the performer's defaults.
"""
import copy
import math
import os

import numpy as np

from . import performer as perf

# ----------------------------------------------------------------------------- materials
HARD = {"solref": (0.004, 1.0), "solimp": (0.95, 0.99, 0.001, 0.5, 2)}
MATERIALS = {   # sliding friction + contact softness
    "wood": dict(friction=0.6, **HARD), "metal": dict(friction=0.45, **HARD), "glass": dict(friction=0.35, **HARD),
    "plastic": dict(friction=0.5, **HARD), "concrete": dict(friction=0.9, **HARD), "stone": dict(friction=0.8, **HARD),
    "fabric": dict(friction=0.9, solref=(0.015, 1.0), solimp=(0.9, 0.97, 0.01, 0.5, 2)),
    "foam": dict(friction=0.9, solref=(0.06, 1.0), solimp=(0.2, 0.95, 0.12, 0.5, 2)),
    "skin": dict(friction=0.8, solref=(0.01, 1.0), solimp=(0.9, 0.97, 0.005, 0.5, 2)),
}
# crash mat: soft and progressive (impedance rises with compression), well damped, grippy
MAT_CONTACT = dict(friction=0.9, solref=(0.06, 1.0), solimp=(0.2, 0.95, 0.12, 0.5, 2))
MATERIAL_OF_ROLE = {"wall": "concrete", "railing": "glass", "furniture": "wood", "prop": "plastic",
                    "structure": "concrete", "person": "skin", "stunt_target": "wood"}
DEFAULT_RGB = {"wall": (0.85, 0.85, 0.83), "railing": (0.7, 0.8, 0.85), "furniture": (0.45, 0.42, 0.40),
               "prop": (0.7, 0.7, 0.7), "structure": (0.6, 0.6, 0.62), "person": (0.2, 0.2, 0.25),
               "stunt_target": (0.55, 0.40, 0.25)}
KINDS, ROLES = ("box", "cylinder", "sphere"), tuple(MATERIAL_OF_ROLE)
PHASE_KINDS = ("hold", "move", "jump", "fall", "air", "land")
NAMED_POSES = ("stand", "crouch", "takeoff", "dive", "tuck", "pike", "layout", "land_prep", "brace", "arms_up")
VIDEO_ASPECT = 16 / 9

DEFAULT_PLAN = {
    "version": 2, "brief": "", "duration_sec": 6.0,
    "source": {"media": "", "kind": "image", "ref_frame": "", "ref_time_sec": 0.0},
    "room": {"x_min": -9.0, "x_max": 7.0, "y_min": -8.0, "y_max": 8.0, "ceiling_z": 5.0,
             "floor_rgb": [0.6, 0.6, 0.6], "wall_rgb": [0.8, 0.8, 0.78], "ceiling_rgb": [0.2, 0.2, 0.2],
             "floor_material": "concrete"},
    "camera": {"match": {"pos": [-6.0, 0.0, 1.5], "lookat": [0.0, 0.0, 0.8], "fovy": 50.0}},
    "performer": {"model": "h1"},
}


def f3(v):
    return " ".join(f"{float(x):.5g}" for x in v)


def quat_yaw(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _rot(x, y, yaw, ox, oy):
    c, s = math.cos(yaw), math.sin(yaw)
    return x + ox * c - oy * s, y + ox * s + oy * c


# ----------------------------------------------------------------------------- plan validation
class PlanError(ValueError):
    pass


def _vec(v, n, where):
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise PlanError(f"plan.json: {where} must be a list of {n} numbers, got {v!r}") from None
    if len(out) != n or not all(math.isfinite(x) for x in out):
        raise PlanError(f"plan.json: {where} must be a list of {n} finite numbers, got {v!r}")
    return out


def _fnum(v, where, default=None):
    if v is None:
        if default is None:
            raise PlanError(f"plan.json: {where} is required")
        return float(default)
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise PlanError(f"plan.json: {where} must be a number, got {v!r}") from None
    if not math.isfinite(x):
        raise PlanError(f"plan.json: {where} must be finite, got {v!r}")
    return x


def _rgb(v, default):
    try:
        out = [float(x) for x in v][:3]
        if len(out) == 3:
            return [min(1.0, max(0.0, x)) for x in out]
    except (TypeError, ValueError):
        pass
    return list(default)


def normalize_plan(plan):
    """Validated deep copy of a contract plan with defaults filled in. Raises PlanError with a specific
    message for anything the scene builder cannot use."""
    if not isinstance(plan, dict):
        raise PlanError(f"plan must be a JSON object, got {type(plan).__name__}")
    if "shelf" in plan or "choreography" in plan:
        raise PlanError("this is a prototype plan (shelf/props/choreography); convert it with scene.from_legacy_plan()")
    p = copy.deepcopy(plan)
    for k, v in DEFAULT_PLAN.items():
        if not isinstance(p.get(k), type(v)) and not (k == "duration_sec" and isinstance(p.get(k), (int, float))):
            p[k] = copy.deepcopy(v)
    room = p["room"]
    for k, v in DEFAULT_PLAN["room"].items():
        room.setdefault(k, copy.deepcopy(v))
    for k in ("x_min", "x_max", "y_min", "y_max", "ceiling_z"):
        room[k] = _fnum(room[k], f"room.{k}")
    if room["x_min"] >= room["x_max"] or room["y_min"] >= room["y_max"]:
        raise PlanError(f"plan.json: room bounds are inverted: {room}")
    room["ceiling_z"] = min(max(room["ceiling_z"], 2.5), 12.0)
    for k in ("floor_rgb", "wall_rgb", "ceiling_rgb"):
        room[k] = _rgb(room[k], DEFAULT_PLAN["room"][k])
    if room.get("floor_material") not in MATERIALS:
        room["floor_material"] = "concrete"

    obstacles, seen = [], set()
    for i, ob in enumerate(p.get("obstacles") or []):
        where = f"obstacles[{i}]"
        if not isinstance(ob, dict):
            raise PlanError(f"plan.json: {where} must be an object")
        ob = dict(ob)
        kind = ob.get("kind", "box")
        if kind not in KINDS:
            raise PlanError(f"plan.json: {where}.kind must be one of {KINDS}, got {kind!r}")
        ob["kind"] = kind
        ob["id"] = _unique(_safe_id(ob.get("id") or f"obstacle_{i + 1}"), seen)
        ob["pos"] = _vec(ob.get("pos"), 3, f"{where}.pos")
        n = {"box": 3, "cylinder": 2, "sphere": 1}[kind]
        size = ob.get("size")
        if isinstance(size, (int, float)):
            size = [size]
        size = _vec(list(size or [])[:n] if size is not None and len(list(size)) >= n else size, n, f"{where}.size ({kind})")
        if min(size) <= 0:
            raise PlanError(f"plan.json: {where}.size must be positive half-extents, got {size}")
        ob["size"] = size
        ob["yaw_deg"] = _fnum(ob.get("yaw_deg"), f"{where}.yaw_deg", 0.0)
        ob["role"] = ob.get("role") if ob.get("role") in ROLES else "furniture"
        ob["material"] = ob.get("material") if ob.get("material") in MATERIALS else MATERIAL_OF_ROLE[ob["role"]]
        ob["rgb"] = _rgb(ob.get("rgb"), DEFAULT_RGB[ob["role"]])
        ob["hazard"] = ob.get("hazard", True) is not False
        obstacles.append(ob)
    p["obstacles"] = obstacles

    sd = p.get("set_dressing") if isinstance(p.get("set_dressing"), dict) else {}
    for k in ("couches", "chairs", "people", "speakers", "ceiling_fixtures", "screens"):
        sd[k] = [x for x in (sd.get(k) or []) if isinstance(x, dict)]
    sd["curtain"] = sd.get("curtain") if isinstance(sd.get("curtain"), dict) else {}
    p["set_dressing"] = sd

    mats = []
    for i, m in enumerate(p.get("crash_mats") or []):
        where = f"crash_mats[{i}]"
        if not isinstance(m, dict):
            raise PlanError(f"plan.json: {where} must be an object")
        m = dict(m)
        m["id"] = _unique(_safe_id(m.get("id") or f"mat{i + 1}"), seen)
        m["pos"] = _vec(m.get("pos"), 3, f"{where}.pos")
        m["size"] = _vec(m.get("size"), 3, f"{where}.size")
        if min(m["size"]) <= 0:
            raise PlanError(f"plan.json: {where}.size must be positive half-extents, got {m['size']}")
        m["yaw_deg"] = _fnum(m.get("yaw_deg"), f"{where}.yaw_deg", 0.0)
        m["rgb"] = _rgb(m.get("rgb"), (0.15, 0.35, 0.85))
        mats.append(m)
    p["crash_mats"] = mats

    st = normalize_stunt(p.get("stunt"))
    p["stunt"] = st
    cam = (p.get("camera") or {}).get("match") or {}
    try:
        p["camera"]["match"] = {"pos": _vec(cam.get("pos"), 3, "camera.match.pos"),
                                "lookat": _vec(cam.get("lookat"), 3, "camera.match.lookat"),
                                "fovy": min(max(_fnum(cam.get("fovy"), "camera.match.fovy", 50.0), 10.0), 120.0),
                                "roll_deg": _fnum(cam.get("roll_deg"), "camera.match.roll_deg", 0.0)}
    except PlanError:
        pts = stunt_points(st)
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        h = math.radians(st["start"]["heading_deg"])
        p["camera"]["match"] = {"pos": [x0 - 3.0 * math.cos(h), y0 - 3.0 * math.sin(h), 1.6],
                                "lookat": [(x0 + x1) / 2, (y0 + y1) / 2, 0.9], "fovy": 55.0, "roll_deg": 0.0}
    perf.check_model((p.get("performer") or {}).get("model"))
    p["performer"] = {"model": "h1"}
    return p


def _f(v, where, default, lo=-1e6, hi=1e6):
    return float(min(max(_fnum(v, where, default), lo), hi))


def _pose(v, where):
    if isinstance(v, dict):
        return {k: _fnum(x, f"{where}.{k}") for k, x in v.items()}
    if v in NAMED_POSES:
        return v
    raise PlanError(f"plan.json: {where} must be one of {NAMED_POSES} or a pose dict, got {v!r}")


def normalize_stunt(st):
    """Validated copy of a stunt (summary, start, phases, tunables, target, default_camera, notes) with phase
    defaults filled in. Raises PlanError naming the field for anything the engine cannot run."""
    if not isinstance(st, dict):
        raise PlanError("plan.json: stunt must be an object with start, phases and target")
    if "type" in st and "phases" not in st:
        raise PlanError("plan.json: stunt.type is gone; write the stunt as phases (see reference/contracts.md)")
    st = copy.deepcopy(st)
    start = st.get("start") if isinstance(st.get("start"), dict) else {}
    st["start"] = {"pos": _vec(start.get("pos", [0.0, 0.0]), 2, "stunt.start.pos"),
                   "heading_deg": _fnum(start.get("heading_deg"), "stunt.start.heading_deg", 0.0),
                   "z": _f(start.get("z"), "stunt.start.z", 0.0, 0.0, 10.0)}
    phases = st.get("phases")
    if not isinstance(phases, list) or not phases:
        raise PlanError("plan.json: stunt.phases must be a non-empty list of phases")
    out, released = [], False
    for i, ph in enumerate(phases):
        w = f"stunt.phases[{i}]"
        if not isinstance(ph, dict) or ph.get("kind") not in PHASE_KINDS:
            raise PlanError(f"plan.json: {w}.kind must be one of {PHASE_KINDS}, got {ph!r}"[:300])
        k = ph["kind"]
        q = dict(ph)
        q["name"] = str(ph.get("name") or f"{k}{i}")
        if k == "hold":
            q["duration"] = _f(ph.get("duration"), f"{w}.duration", 0.5, 0.0, 30.0)
            q["pose"] = _pose(ph.get("pose", "stand"), f"{w}.pose")
        elif k == "move":
            path = ph.get("path")
            if not isinstance(path, list) or len(path) < 2:
                raise PlanError(f"plan.json: {w}.path must list at least 2 [x, y] points")
            q["path"] = [_vec(pt, 2, f"{w}.path[{j}]") for j, pt in enumerate(path)]
            q["speed"] = _f(ph.get("speed"), f"{w}.speed", 3.0, 0.2, 12.0)
            q["accel_time"] = _f(ph.get("accel_time"), f"{w}.accel_time", 1.0, 0.05, 10.0)
            q["decel_time"] = _f(ph.get("decel_time"), f"{w}.decel_time", 1.0, 0.0, 10.0)
            q["turn_radius"] = _f(ph.get("turn_radius"), f"{w}.turn_radius", 1.5, 0.2, 50.0)
            q["lean_scale"] = _f(ph.get("lean_scale"), f"{w}.lean_scale", 1.0, 0.0, 3.0)
            if ph.get("gait") not in (None, "walk", "run", "sprint"):
                raise PlanError(f"plan.json: {w}.gait must be walk, run or sprint")
        elif k in ("jump", "fall"):
            if k == "jump":
                q["vz"] = _f(ph.get("vz"), f"{w}.vz", 3.5, 0.0, 8.0)
                q["vx"] = None if ph.get("vx") is None else _f(ph.get("vx"), f"{w}.vx", 0.0, -3.0, 10.0)
                q["lean_deg"] = _f(ph.get("lean_deg"), f"{w}.lean_deg", 10.0, -60.0, 75.0)
                q["pose"] = _pose(ph.get("pose", "takeoff"), f"{w}.pose")
            else:
                q["push"] = _vec(ph.get("push", [0.0, 0.0, 0.0]), 3, f"{w}.push")
                q["lean_deg"] = _f(ph.get("lean_deg"), f"{w}.lean_deg", 15.0, -60.0, 75.0)
                q["pose"] = _pose(ph.get("pose", "brace"), f"{w}.pose")
            rot = ph.get("rotations_deg") if isinstance(ph.get("rotations_deg"), dict) else {}
            q["rotations_deg"] = {a: _f(rot.get(a), f"{w}.rotations_deg.{a}", 0.0, -1440.0, 1440.0)
                                  for a in ("roll", "pitch", "yaw")}
        elif k == "air":
            poses = ph.get("poses") or []
            if not isinstance(poses, list):
                raise PlanError(f"plan.json: {w}.poses must be a list")
            q["poses"] = sorted(({"at": _f(e.get("at"), f"{w}.poses[{j}].at", 0.0, 0.0, 1.5),
                                  "pose": _pose(e.get("pose", "tuck"), f"{w}.poses[{j}].pose"),
                                  "tuck": _f(e.get("tuck"), f"{w}.poses[{j}].tuck", 0.0, 0.0, 1.0)}
                                 for j, e in enumerate(poses) if isinstance(e, dict)), key=lambda e: e["at"])
        else:
            q["pose"] = _pose(ph.get("pose", "crouch"), f"{w}.pose")
            q["compliance"] = _f(ph.get("compliance"), f"{w}.compliance", 0.6, 0.0, 1.0)
            q["duration"] = _f(ph.get("duration"), f"{w}.duration", 1.0, 0.0, 10.0)
        if k in ("hold", "move") and released:
            raise PlanError(f"plan.json: {w} is a {k} after a jump/fall; after leaving the ground the performer "
                            "stays free (only air and land may follow)")
        if k in ("air", "land") and not released:
            raise PlanError(f"plan.json: {w} is {k} but no jump or fall comes before it")
        if k in ("jump", "fall"):
            if released:
                raise PlanError(f"plan.json: {w}: only one jump or fall per stunt")
            released = True
        out.append(q)
    st["phases"] = out
    tgt = st.get("target") if isinstance(st.get("target"), dict) else {"kind": "none"}
    kind = tgt.get("kind", "none")
    if kind == "zone":
        tgt = {**tgt, "pos": _vec(tgt.get("pos"), 2, "stunt.target.pos"),
               "radius": _f(tgt.get("radius"), "stunt.target.radius", 0.65, 0.05, 20.0)}
    elif kind == "mat":
        tgt = {**tgt, "id": str(tgt.get("id") or "")}
    elif kind != "none":
        raise PlanError(f"plan.json: stunt.target.kind must be mat, zone or none, got {kind!r}")
    if "offset" in tgt:
        tgt["offset"] = _vec(tgt["offset"], 2, "stunt.target.offset")
    st["target"] = tgt
    st.setdefault("summary", "")
    st.setdefault("notes", "")
    st.setdefault("tunables", [])
    st["default_camera"] = st.get("default_camera") if st.get("default_camera") in ("match_follow", "match", "hero",
                                                                                    "side") else "match_follow"
    return st


def stunt_points(st):
    """xy points that outline the stunt: the start, every move path point and the zone target."""
    pts = [tuple(st["start"]["pos"])]
    for ph in st["phases"]:
        if ph["kind"] == "move":
            pts += [tuple(pt) for pt in ph["path"]]
    if st["target"].get("kind") == "zone":
        pts.append(tuple(st["target"]["pos"]))
    return pts


def _safe_id(s):
    s = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(s).strip())
    return (s or "object").replace("__", "_")


def _unique(s, seen):
    base, k = s, 2
    while s in seen:
        s, k = f"{base}_{k}", k + 1
    seen.add(s)
    return s


# ----------------------------------------------------------------------------- legacy plan converter
def from_legacy_plan(old, brief="barrel roll over the shelf unit onto a crash mat", duration_sec=9.0):
    """Convert the prototype's plan.json (shelf, screen, props, choreography, environment) into a contract
    plan: the shelf becomes an open-shelf box obstacle (role stunt_target), each prop a small obstacle,
    the environment becomes room + set_dressing, the screen goes to set_dressing.screens, and a 0.3 m crash
    mat is added at the landing spot (the prototype had none). The run-up starts at least 3 m before
    take-off and the mat never overlaps the shelf; Argon's take-off and landing x are kept otherwise."""
    if not isinstance(old, dict) or "shelf" not in old:
        raise PlanError("from_legacy_plan expects the prototype plan with a 'shelf' entry")
    sh, env = old["shelf"], old.get("environment") or {}
    L, D, H = float(sh["length"]), float(sh["depth"]), float(sh["height"])
    obstacles = [{"id": "shelf", "kind": "box", "pos": [0.0, 0.0, H / 2], "size": [D / 2, L / 2, H / 2], "yaw_deg": 0,
                  "rgb": [0.78, 0.62, 0.42], "material": "wood", "role": "stunt_target", "hazard": True,
                  "style": "shelf", "board_levels": [float(z) for z in sh.get("board_levels") or [H]],
                  "post_spacing": float(sh.get("post_spacing", 0.8))}]
    count = {}
    for p in old.get("props") or []:
        k = p.get("kind", "prop")
        count[k] = count.get(k, 0) + 1
        x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
        r, h = (float(v) for v in p["size"])
        ob = {"id": f"{k}_{count[k]}", "pos": [x, y, z + h / 2], "yaw_deg": 0, "rgb": list(p.get("rgb", (0.7, 0.7, 0.7))),
              "role": "prop", "hazard": True, "material": "plastic"}
        if k == "plant":
            ob.update(kind="cylinder", size=[1.5 * r, h / 2], style="plant", material="fabric")
        elif k == "vase":
            ob.update(kind="cylinder", size=[r, h / 2], style="vase", material="glass")
        elif k == "book":
            ob.update(kind="box", size=[r, 1.3 * r, h / 2], material="fabric")
        elif k == "bowl":
            ob.update(kind="cylinder", size=[r, max(h / 2, 0.01)], rgb=[0.12, 0.1, 0.1], material="stone")
        else:   # candle, can
            ob.update(kind="cylinder", size=[r, h / 2])
        obstacles.append(ob)
    ch = old.get("choreography") or {}
    takeoff_x = float(ch.get("takeoff_x", -D / 2 - 0.9))
    start_x = min(float(ch.get("start_x", takeoff_x - 3.0)), takeoff_x - 3.0)
    mat_half = [0.9, 1.0, 0.15]
    landing_x = max(float(ch.get("landing_x", D / 2 + 1.5)), D / 2 + 0.05 + mat_half[0])
    sd = {k: copy.deepcopy(env.get(k) or []) for k in ("couches", "chairs", "people", "speakers", "ceiling_fixtures")}
    sd["curtain"] = copy.deepcopy(env.get("curtain") or {})
    sc = old.get("screen")
    sd["screens"] = [dict(sc)] if sc else []
    room = dict(env.get("room") or {})
    room.setdefault("floor_material", "concrete")
    return {
        "version": 1, "brief": brief, "duration_sec": float(duration_sec),
        "source": {"media": "", "kind": "image", "ref_frame": "", "ref_time_sec": 0.0, "legacy": True},
        "room": room or copy.deepcopy(DEFAULT_PLAN["room"]),
        "obstacles": obstacles,
        "set_dressing": sd,
        "crash_mats": [{"id": "mat1", "pos": [landing_x, 0.0, mat_half[2]], "size": mat_half, "rgb": [0.15, 0.35, 0.85]}],
        "camera": {"match": {"pos": [-6.0, 0.0, 1.5], "lookat": [0.0, 0.0, 0.8], "fovy": 50.0}},
        "stunt": {
            "summary": "runs at the shelf, dives over it rolling 360 deg about the travel axis, lands on the mat",
            "start": {"pos": [start_x, 0.0], "heading_deg": 0.0},
            "phases": [
                {"kind": "hold", "name": "ready", "duration": 0.8, "pose": "stand"},
                {"kind": "move", "name": "run-up", "path": [[start_x, 0.0], [takeoff_x - 1.2, 0.0]], "speed": 3.5,
                 "accel_time": 1.0, "decel_time": 0.0, "turn_radius": 1.0, "lean_scale": 1.0},
                {"kind": "jump", "name": "take-off", "vz": 4.6, "lean_deg": 0.0,
                 "rotations_deg": {"roll": 360.0, "pitch": 0.0, "yaw": 0.0}, "pose": "arms_up"},
                {"kind": "air", "name": "roll", "poses": [{"at": 0.15, "pose": "arms_up", "tuck": 0.5},
                                                          {"at": 0.7, "pose": "arms_up", "tuck": 0.5},
                                                          {"at": 0.85, "pose": "land_prep"}]},
                {"kind": "land", "name": "landing", "pose": "crouch", "compliance": 0.6, "duration": 1.0},
            ],
            "tunables": [
                {"name": "run_speed", "path": "phases[1].speed", "min": 2.0, "max": 6.0, "unit": "m/s", "doc": "run-up speed"},
                {"name": "takeoff_vz", "path": "phases[2].vz", "min": 2.5, "max": 6.0, "unit": "m/s", "doc": "vertical take-off speed"},
                {"name": "roll_deg", "path": "phases[2].rotations_deg.roll", "min": 0.0, "max": 720.0, "unit": "deg",
                 "doc": "roll completed by touchdown"},
                {"name": "runup_end_x", "path": "phases[1].path[1][0]", "min": takeoff_x - 3.0, "max": takeoff_x,
                 "unit": "m", "doc": "where the run-up ends (take-off follows about 1 m later)"},
            ],
            "target": {"kind": "mat", "id": "mat1"},
            "default_camera": "hero",
            "notes": "converted from the prototype plan"},
        "performer": {"model": "h1"},
    }


# ----------------------------------------------------------------------------- landing mat / launch platform
def target_mat(plan, stunt=None):
    """The crash mat named by stunt.target (kind mat), moved by stunt.target.offset: {"id", "index", "pos",
    "size", "yaw", "top"}; None for other targets or a plan without mats. An unknown id falls back to the mat
    nearest the end of the last move path (or the start)."""
    st = stunt if stunt is not None else plan["stunt"]
    tgt = st.get("target") or {}
    mats = plan.get("crash_mats") or []
    if tgt.get("kind") != "mat" or not mats:
        return None
    ids = [m["id"] for m in mats]
    if tgt.get("id") in ids:
        i = ids.index(tgt["id"])
    else:
        ref = stunt_points(st)[-1]
        i = min(range(len(mats)), key=lambda k: math.hypot(mats[k]["pos"][0] - ref[0], mats[k]["pos"][1] - ref[1]))
    m = mats[i]
    ox, oy = (tgt.get("offset") or [0.0, 0.0])[:2]
    pos = [m["pos"][0] + float(ox), m["pos"][1] + float(oy), m["pos"][2]]
    return {"id": m["id"], "index": i, "pos": pos, "size": list(m["size"]), "yaw": math.radians(m.get("yaw_deg", 0.0)),
            "top": pos[2] + m["size"][2]}


def landing_mat(plan, params=None):
    """Back-compat alias: target_mat of the plan's stunt."""
    return target_mat(plan)


def inside_footprint(xy, pos, size, yaw, margin=0.0):
    """Is the world point xy over the (yawed) box footprint with half-extents size[:2]?"""
    dx, dy = xy[0] - pos[0], xy[1] - pos[1]
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return abs(lx) <= size[0] + margin and abs(ly) <= size[1] + margin


def obstacle_top(ob):
    return ob["pos"][2] + (ob["size"][2] if ob["kind"] == "box" else ob["size"][1] if ob["kind"] == "cylinder" else ob["size"][0])


def _footprint(ob):
    return ob["size"] if ob["kind"] == "box" else [ob["size"][0], ob["size"][0]]


def start_support(plan, stunt=None):
    """The obstacle the performer starts on (stunt.start.z > 0.05): the obstacle under the start point whose top
    is closest to start.z; None when starting on the floor."""
    st = stunt if stunt is not None else plan["stunt"]
    z = st["start"].get("z", 0.0)
    if z <= 0.05:
        return None
    pt = st["start"]["pos"]
    best = None
    for ob in plan["obstacles"]:
        if ob["kind"] != "sphere" and inside_footprint(pt, ob["pos"], _footprint(ob), math.radians(ob["yaw_deg"]), 0.3):
            if best is None or abs(obstacle_top(ob) - z) < abs(obstacle_top(best) - z):
                best = ob
    return best


def support_top(plan, xy, mat=None, margin=0.2):
    """Height of the highest surface under the point xy: crash mats (the target mat at its moved position),
    obstacles (box / cylinder tops), else the floor (0)."""
    top = 0.0
    for i, m in enumerate(plan.get("crash_mats") or []):
        pos = mat["pos"] if mat is not None and i == mat["index"] else m["pos"]
        if inside_footprint(xy, pos, m["size"], math.radians(m.get("yaw_deg", 0.0)), margin):
            top = max(top, pos[2] + m["size"][2])
    for ob in plan.get("obstacles") or []:
        if ob["kind"] != "sphere" and inside_footprint(xy, ob["pos"], _footprint(ob), math.radians(ob["yaw_deg"]), margin):
            top = max(top, obstacle_top(ob))
    return top


# ----------------------------------------------------------------------------- geometry collection
class _Geoms:
    """Collects named scene geoms with their category (hazard / mat / floor / obstacle / launch)."""

    def __init__(self):
        self.items, self.names = [], set()

    def add(self, obj, part, typ, size, pos, quat=None, rgba=None, category="hazard", contact=None, material=None,
            fromto=None):
        name = obj if part is None else f"{obj}__{part}"
        base, k = name, 2
        while name in self.names:
            name, k = f"{base}{k}", k + 1
        self.names.add(name)
        self.items.append(dict(name=name, object=obj, type=typ, size=size, pos=pos, quat=quat, rgba=rgba,
                               category=category, contact=contact or HARD_FRICTION, material=material, fromto=fromto))

    def xml(self):
        out = []
        for g in self.items:
            a = [f'name="{g["name"]}"', f'type="{g["type"]}"']
            if g["fromto"] is not None:
                a.append(f'fromto="{f3(g["fromto"])}"')
            else:
                a.append(f'pos="{f3(g["pos"])}"')
            a.append(f'size="{f3(g["size"])}"')
            if g["quat"] is not None and abs(g["quat"][0] - 1.0) > 1e-9:
                a.append(f'quat="{f3(g["quat"])}"')
            if g["material"]:
                a.append(f'material="{g["material"]}"')
            elif g["rgba"] is not None:
                rgba = list(g["rgba"]) + ([1.0] if len(g["rgba"]) == 3 else [])
                a.append(f'rgba="{f3(rgba)}"')
            c = g["contact"]
            a.append(f'friction="{c["friction"]:g} 0.005 0.0001" solref="{f3(c["solref"])}" solimp="{f3(c["solimp"])}" '
                     f'priority="1"')
            out.append("<geom " + " ".join(a) + "/>")
        return "\n    ".join(out)


HARD_FRICTION = dict(friction=0.7, **HARD)


def _contact(material):
    return MATERIALS.get(material, HARD_FRICTION)


def _box(G, obj, part, size, pos, rgb, yaw=0.0, **kw):
    G.add(obj, part, "box", size, pos, quat_yaw(yaw), rgb, **kw)


def _room(G, plan, bounds):
    x0, x1, y0, y1 = bounds
    room = plan["room"]
    H = room["ceiling_z"]
    wall = room["wall_rgb"]
    _box(G, "wall_xmax", None, (0.1, (y1 - y0) / 2 + 0.2, H / 2), (x1 + 0.1, (y0 + y1) / 2, H / 2), wall)
    _box(G, "wall_ymin", None, ((x1 - x0) / 2 + 0.2, 0.1, H / 2), ((x0 + x1) / 2, y0 - 0.1, H / 2), wall)
    _box(G, "wall_ymax", None, ((x1 - x0) / 2 + 0.2, 0.1, H / 2), ((x0 + x1) / 2, y1 + 0.1, H / 2), wall)
    _box(G, "wall_xmin", None, (0.1, (y1 - y0) / 2 + 0.2, H / 2), (x0 - 0.1, (y0 + y1) / 2, H / 2), wall)


def _set_dressing(G, sd, bounds, H):
    """Ported from barrel_roll.env_geoms (no keep-clear zone: everything is kept)."""
    x0, x1, y0, y1 = bounds
    cu = sd.get("curtain") or {}
    if cu:
        pw = float(np.clip(cu.get("panel_width", 0.4), 0.25, 4.0))
        fd = float(np.clip(cu.get("fold_depth", 0.12), 0.05, 0.3))
        ch_ = min(float(cu.get("height", H - 0.2)), H - 0.05)
        crgb = _rgb(cu.get("rgb"), (0.9, 0.86, 0.78))
        for i in range(max(1, int((y1 - y0) / pw))):
            shade = 1.0 - 0.07 * (i % 2)
            _box(G, "curtain", f"p{i}", (0.05, pw / 2, ch_ / 2), (x1 - 0.1 - fd * (i % 2), y0 + (i + 0.5) * pw, ch_ / 2),
                 [c * shade for c in crgb], contact=_contact("fabric"))
    for n, c in enumerate(sd.get("couches") or [], 1):
        obj = f"couch_{n}"
        L, D, sh_, bh = (float(c.get(k, d)) for k, d in (("length", 2.0), ("depth", 0.9), ("seat_h", 0.42), ("back_h", 0.8)))
        x, y, yaw = float(c["x"]), float(c["y"]), math.radians(float(c.get("yaw_deg", 0)))
        rgb = _rgb(c.get("rgb"), (0.6, 0.6, 0.6))
        fab = _contact("fabric")
        _box(G, obj, "seat", (D / 2, L / 2, sh_ / 2), (x, y, sh_ / 2), rgb, yaw, contact=fab)
        bx, by = _rot(x, y, yaw, D / 2 - 0.1, 0)
        _box(G, obj, "back", (0.1, L / 2, bh / 2), (bx, by, bh / 2), [v * 0.95 for v in rgb], yaw, contact=fab)
        for k, side in enumerate((-1, 1)):
            ax, ay = _rot(x, y, yaw, 0, side * (L / 2 - 0.07))
            _box(G, obj, f"arm{k}", (D / 2, 0.07, (sh_ + 0.18) / 2), (ax, ay, (sh_ + 0.18) / 2), [v * 0.9 for v in rgb],
                 yaw, contact=fab)
    for n, c in enumerate(sd.get("chairs") or [], 1):
        obj = f"chair_{n}"
        x, y, yaw = float(c["x"]), float(c["y"]), math.radians(float(c.get("yaw_deg", 0)))
        rgb = _rgb(c.get("rgb"), (0.1, 0.1, 0.1))
        met = _contact("metal")
        _box(G, obj, "seat", (0.22, 0.22, 0.015), (x, y, 0.45), rgb, yaw, contact=met)
        bx, by = _rot(x, y, yaw, 0.21, 0)
        _box(G, obj, "back", (0.015, 0.22, 0.17), (bx, by, 0.65), rgb, yaw, contact=met)
        k = 0
        for ox in (-0.19, 0.19):
            for oy in (-0.19, 0.19):
                lx, ly = _rot(x, y, yaw, ox, oy)
                _box(G, obj, f"leg{k}", (0.015, 0.015, 0.225), (lx, ly, 0.225), rgb, yaw, contact=met)
                k += 1
    for n, p in enumerate(sd.get("people") or [], 1):
        obj = f"person_{n}"
        x, y, yaw = float(p["x"]), float(p["y"]), math.radians(float(p.get("yaw_deg", 0)))
        sz = float(p.get("seat_z", 0.45))
        W = lambda ox, oy, z: (*_rot(x, y, yaw, ox, oy), z)
        shirt = _rgb(p.get("rgb"), (0.12, 0.12, 0.15))
        skin, trousers = (0.85, 0.68, 0.55), (0.15, 0.16, 0.2)
        sk = _contact("skin")
        if p.get("pose") == "standing" or sz <= 0.05:
            G.add(obj, "torso", "capsule", (0.15,), None, rgba=shirt, contact=sk, fromto=(*W(0, 0, 1.0), *W(0, 0, 1.4)))
            G.add(obj, "head", "sphere", (0.1,), W(0, 0, 1.68), rgba=skin, contact=sk)
            for k, side in enumerate((-0.1, 0.1)):
                G.add(obj, f"leg{k}", "capsule", (0.07,), None, rgba=trousers, contact=sk,
                      fromto=(*W(0, side, 0.07), *W(0, side, 0.9)))
            continue
        G.add(obj, "torso", "capsule", (0.13,), None, rgba=shirt, contact=sk, fromto=(*W(0, 0, sz + 0.15), *W(0, 0, sz + 0.5)))
        G.add(obj, "head", "sphere", (0.1,), W(-0.02, 0, sz + 0.72), rgba=skin, contact=sk)
        for k, side in enumerate((-0.1, 0.1)):
            G.add(obj, f"thigh{k}", "capsule", (0.06,), None, rgba=trousers, contact=sk,
                  fromto=(*W(0, side, sz + 0.1), *W(-0.4, side, sz + 0.1)))
            G.add(obj, f"shin{k}", "capsule", (0.05,), None, rgba=trousers, contact=sk,
                  fromto=(*W(-0.4, side, sz + 0.1), *W(-0.4, side, 0.06)))
    for n, s in enumerate(sd.get("speakers") or [], 1):
        obj = f"speaker_{n}"
        x, y = float(s["x"]), float(s["y"])
        zb, w, h = float(s.get("z_bottom", 1.0)), float(s.get("width", 0.35)), float(s.get("height", 0.6))
        met = _contact("metal")
        G.add(obj, "pole", "cylinder", (0.02, zb / 2), (x, y, zb / 2), rgba=(0.05, 0.05, 0.05), contact=met)
        G.add(obj, "base", "cylinder", (0.22, 0.015), (x, y, 0.015), rgba=(0.05, 0.05, 0.05), contact=met)
        _box(G, obj, "box", (w * 0.4, w / 2, h / 2), (x, y, zb + h / 2), (0.06, 0.06, 0.07), contact=_contact("plastic"))
    for n, f in enumerate((sd.get("ceiling_fixtures") or [])[:12], 1):
        obj = f"fixture_{n}"
        px, py, pz = (float(v) for v in f["pos"])
        px, py = float(np.clip(px, x0 + 0.5, x1 - 0.5)), float(np.clip(py, y0 + 0.5, y1 - 0.5))
        pz = float(np.clip(pz, min(3.0, H - 0.1), H - 0.05))
        half = float(np.clip(f.get("length", 4), 1, 20)) / 2
        rad = float(np.clip(f.get("radius", 0.08), 0.03, 0.3))
        rgb = _rgb(f.get("rgb"), (0.1, 0.1, 0.1))
        axis = f.get("axis", "y")
        if f.get("kind") == "beam":
            _box(G, obj, None, (half, rad, rad) if axis == "x" else (rad, half, rad), (px, py, pz), rgb, contact=_contact("metal"))
        else:
            q = np.array([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]) if axis == "x" else \
                np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
            G.add(obj, None, "cylinder", (rad, half), (px, py, pz), q, rgb, contact=_contact("metal"))
    for n, sc in enumerate(sd.get("screens") or [], 1):
        obj = f"screen_{n}"
        sx, sy = float(sc.get("x", x1 - 0.3)), float(sc.get("y", 0.0))
        sw, sht, bz = float(sc.get("width", 3.0)), float(sc.get("height", 1.8)), float(sc.get("bottom_z", 1.0))
        yaw = math.radians(float(sc.get("yaw_deg", 0.0)))
        gl = _contact("glass")
        G.add(obj, "panel", "box", (0.05, sw / 2, sht / 2), (sx, sy, bz + sht / 2), quat_yaw(yaw), material="fg_screen",
              contact=gl)
        hx, hy = _rot(sx, sy, yaw, -0.1, -sw * 0.15)
        _box(G, obj, "glare", (0.1, sw * 0.28, sht * 0.04), (hx, hy, bz + sht * 0.6), (0.95, 0.95, 1.0), yaw, contact=gl)
        for k, oy in enumerate((-sw * 0.35, sw * 0.35)):
            lx, ly = _rot(sx, sy, yaw, 0, oy)
            _box(G, obj, f"leg{k}", (0.04, 0.04, (bz + 0.2) / 2), (lx, ly, (bz + 0.2) / 2), (0.08, 0.08, 0.1), yaw,
                 contact=_contact("metal"))


def _obstacle(G, ob, category):
    """One obstacle: a plain primitive, or a styled one (shelf / plant / vase) built from its bounding shape."""
    obj, kind, (x, y, z) = ob["id"], ob["kind"], ob["pos"]
    size, yaw, rgb = ob["size"], math.radians(ob["yaw_deg"]), ob["rgb"]
    con = MAT_CONTACT if ob["material"] == "foam" else _contact(ob["material"])
    style = ob.get("style")
    W = lambda ox, oy, oz: (*_rot(x, y, yaw, ox, oy), z + oz)
    if style == "shelf" and kind == "box":
        D, L, H = size[0] * 2, size[1] * 2, size[2] * 2
        along_y = L >= D
        long_len, short_half = (L, size[0]) if along_y else (D, size[1])
        base = z - size[2]
        for k, lvl in enumerate(ob.get("board_levels") or [base + H]):
            lvl = min(max(float(lvl), base + 0.03), base + H)
            _box(G, obj, f"board{k}", (size[0], size[1], 0.015), W(0, 0, lvl - 0.015 - z), rgb, yaw, category=category,
                 contact=con)
        n_posts = max(2, int(round(long_len / max(float(ob.get("post_spacing", 0.8)), 0.2))) + 1)
        k = 0
        for i in range(n_posts):
            a = -long_len / 2 + i * long_len / (n_posts - 1)
            a = min(max(a, -long_len / 2 + 0.02), long_len / 2 - 0.02)
            for b in (-short_half + 0.02, short_half - 0.02):
                ox, oy = (b, a) if along_y else (a, b)
                _box(G, obj, f"post{k}", (0.02, 0.02, size[2]), W(ox, oy, 0), rgb, yaw, category=category, contact=con)
                k += 1
        return
    if style == "plant" and kind == "cylinder":
        h, r = 2 * size[1], size[0] / 1.5
        base = z - size[1]
        G.add(obj, "pot", "cylinder", (r, h * 0.17), (x, y, base + h * 0.17), rgba=(0.85, 0.86, 0.85), category=category,
              contact=_contact("stone"))
        G.add(obj, "leaves", "ellipsoid", (r * 1.5, r * 1.5, h * 0.33), (x, y, base + h * 0.65), rgba=rgb, category=category,
              contact=con)
        return
    if style == "vase" and kind == "cylinder":
        r, hh = size[0], size[1]
        G.add(obj, None, "capsule", (r, max(hh - r, 0.01)), (x, y, z), rgba=rgb, category=category, contact=con)
        return
    q = quat_yaw(yaw)
    if kind == "box":
        G.add(obj, None, "box", size, (x, y, z), q, rgb, category=category, contact=con)
    elif kind == "cylinder":
        G.add(obj, None, "cylinder", size, (x, y, z), q, rgb, category=category, contact=con)
    else:
        G.add(obj, None, "sphere", size, (x, y, z), None, rgb, category=category, contact=con)


# ----------------------------------------------------------------------------- cameras
def lookat_xyaxes(pos, lookat, roll_deg=0.0):
    """MuJoCo camera frame: +x image right, +y image up, the camera looks along -z."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    if np.linalg.norm(f) < 1e-9:
        raise PlanError(f"camera pos and lookat coincide: {pos}")
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        x, y = c * x + s * y, -s * x + c * y
    return x, y


def camera_xml(name, pos, lookat, fovy, roll_deg=0.0):
    x, y = lookat_xyaxes(pos, lookat, roll_deg)
    return f'<camera name="{name}" pos="{f3(pos)}" xyaxes="{f3(x)} {f3(y)}" fovy="{fovy:.3f}"/>'


def _fits(pos, look, fovy, pts, margin=0.06):
    x, y = lookat_xyaxes(pos, look)
    f = np.cross(y, x)
    d = pts - pos
    depth = d @ f
    if np.any(depth < 0.2):
        return False
    ty = math.tan(math.radians(fovy) / 2)
    u, v = (d @ x) / depth / (ty * VIDEO_ASPECT), (d @ y) / depth / ty
    return bool(np.all(np.abs(u) <= 1 - margin) and np.all(np.abs(v) <= 1 - margin))


def fit_camera(look, direction, pts, box, fovy=45.0, max_fovy=80.0):
    """Static camera looking at `look` from along `direction` that frames all `pts`, kept inside the room
    box (x0, x1, y0, y1, z_top). Returns (pos, fovy)."""
    u = np.asarray(direction, float)
    u /= np.linalg.norm(u)
    look = np.asarray(look, float)
    x0, x1, y0, y1, zt = box
    with np.errstate(divide="ignore"):
        lim = [((x1 if u[0] > 0 else x0) - look[0]) / u[0] if abs(u[0]) > 1e-9 else np.inf,
               ((y1 if u[1] > 0 else y0) - look[1]) / u[1] if abs(u[1]) > 1e-9 else np.inf,
               (zt - look[2]) / u[2] if u[2] > 1e-9 else np.inf]
    dmax = max(0.8, min(lim))
    lo, hi = 0.8, 60.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if _fits(look + mid * u, look, fovy, pts) else (mid, hi)
    d = hi
    if d <= dmax:
        return look + d * u, fovy
    pos = look + dmax * u
    f = fovy
    while f < max_fovy and not _fits(pos, look, f, pts):
        f += 1.0
    return pos, f


def _stunt_region(plan, z_top, stunt=None):
    st = stunt if stunt is not None else plan["stunt"]
    pts = list(stunt_points(st))
    mat = target_mat(plan, st)
    if mat is not None:
        pts.append((mat["pos"][0], mat["pos"][1]))
    xy = np.array(pts, float)
    lo, hi = xy.min(0) - 0.8, xy.max(0) + 0.8
    box = np.array([[x_, y_, z_] for x_ in (lo[0], hi[0]) for y_ in (lo[1], hi[1]) for z_ in (0.0, z_top)], float)
    return box, np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.45 * z_top])


def _cameras(plan, bounds, sd_points):
    x0, x1, y0, y1 = bounds
    H = plan["room"]["ceiling_z"]
    box = (x0 + 0.35, x1 - 0.35, y0 + 0.35, y1 - 0.35, H - 0.2)
    z_top = min(3.2, H - 0.3)
    pts, look = _stunt_region(plan, z_top)
    # side: perpendicular to +X, on the side with less set dressing between camera and the stunt
    best = None
    for sgn in (-1.0, 1.0):
        between = sum(1 for p in sd_points if (p[1] - look[1]) * sgn > 0.8 and abs(p[0] - look[0]) < 6.0)
        if best is None or between < best[0]:
            best = (between, sgn)
    side_pos, side_fovy = fit_camera(look, [0.0, best[1], 0.12], pts, box)
    tq_pos, tq_fovy = fit_camera(look, [-0.62, -0.72, 0.25], pts, box)
    m = plan["camera"]["match"]
    return [camera_xml("match", m["pos"], m["lookat"], m["fovy"], m.get("roll_deg", 0.0)),
            camera_xml("side", side_pos, look, side_fovy),
            camera_xml("three_quarter", tq_pos, look, tq_fovy)]


# ----------------------------------------------------------------------------- full scene
def _bounds(plan, G):
    """Room bounds grown to contain every geom, the stunt lane and the match camera (+0.3 m)."""
    room = plan["room"]
    x0, x1, y0, y1 = room["x_min"], room["x_max"], room["y_min"], room["y_max"]
    pts = []
    for g in G.items:
        if g["fromto"] is not None:
            pts += [g["fromto"][:3], g["fromto"][3:]]
        elif g["pos"] is not None:
            r = max(g["size"]) if g["type"] != "plane" else 0.0
            pts += [(g["pos"][0] - r, g["pos"][1] - r), (g["pos"][0] + r, g["pos"][1] + r)]
    pts += [(x - 1.0 * sx, y - 1.0 * sy) for x, y in stunt_points(plan["stunt"]) for sx in (-1, 1) for sy in (-1, 1)]
    pts.append(tuple(plan["camera"]["match"]["pos"][:2]))
    for p in pts:
        x0, x1 = min(x0, p[0] - 0.3), max(x1, p[0] + 0.3)
        y0, y1 = min(y0, p[1] - 0.3), max(y1, p[1] + 0.3)
    return x0, x1, y0, y1


def build_xml(plan, stunt, root_pos, root_quat):
    """MJCF of the stunt scene for one candidate with the H1 at the given root pose. plan must come from
    normalize_plan(); stunt is the candidate's stunt (engine.params.apply; its target.offset moves the target
    mat). Returns (xml, meta); meta lists geom names by category ("hazard", "mat", "floor", "obstacle",
    "launch" = the obstacle the performer starts on), the object of every named geom and the target mat."""
    if (plan.get("world") or {}).get("dir"):
        return build_world_xml(plan, stunt, root_pos, root_quat)
    G = _Geoms()
    stunt = normalize_stunt(stunt if stunt is not None else plan["stunt"])
    plan = {**plan, "stunt": stunt}
    launch = start_support(plan, stunt)
    for ob in plan["obstacles"]:
        cat = "launch" if launch is not None and ob["id"] == launch["id"] else ("hazard" if ob["hazard"] else "obstacle")
        _obstacle(G, ob, cat)
    mat = target_mat(plan, stunt)
    for i, m in enumerate(plan["crash_mats"]):
        pos = mat["pos"] if mat and i == mat["index"] else m["pos"]
        G.add(m["id"], None, "box", m["size"], pos, quat_yaw(math.radians(m["yaw_deg"])), m["rgb"], category="mat",
              contact=MAT_CONTACT)
    n_obj = len(G.items)
    sd = plan["set_dressing"]
    sd_probe = _Geoms()
    _set_dressing(sd_probe, sd, (plan["room"]["x_min"], plan["room"]["x_max"], plan["room"]["y_min"], plan["room"]["y_max"]),
                  plan["room"]["ceiling_z"])
    bounds = _bounds(plan, _merge(G, sd_probe))
    _set_dressing(G, sd, bounds, plan["room"]["ceiling_z"])
    _room(G, plan, bounds)
    sd_points = [g["pos"] if g["pos"] is not None else g["fromto"][:3] for g in G.items[n_obj:]]
    cams = _cameras(plan, bounds, sd_points)
    h1 = perf.compose_h1(root_pos, root_quat)
    room = plan["room"]
    floor_rgb = room["floor_rgb"]
    fc = MATERIALS[room["floor_material"]]
    xml = f'''<mujoco model="fallguy_stunt">
  <compiler angle="radian" meshdir="{h1["meshdir"]}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size memory="64M"/>
  <visual>
    <global offwidth="1920" offheight="1920" fovy="45"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.5 0.5 0.53" diffuse="0.15 0.15 0.15" specular="0.1 0.1 0.1"/>
    <map znear="0.02" zfar="80"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.88 0.95" rgb2="0.55 0.58 0.65" width="256" height="256"/>
    <texture name="fg_screen_tex" type="2d" builtin="gradient" rgb1="0.05 0.15 0.9" rgb2="0.02 0.03 0.25" width="256" height="256"/>
    <material name="fg_screen" texture="fg_screen_tex" emission="0.8" specular="0.2"/>
    <texture name="fg_floor_tex" type="2d" builtin="checker" rgb1="{f3(floor_rgb)}" rgb2="{f3([c * 0.93 for c in floor_rgb])}" width="256" height="256" mark="random" markrgb="{f3([c * 0.8 for c in floor_rgb])}" random="0.08"/>
    <material name="fg_floor" texture="fg_floor_tex" texrepeat="12 12" texuniform="true" reflectance="0.03"/>
    {h1["assets"]}
  </asset>
  <default>
    {h1["default"]}
  </default>
  <worldbody>
    <light name="key" directional="true" pos="-3 -4 6" dir="0.4 0.5 -1" diffuse="0.42 0.42 0.4" castshadow="true"/>
    <light name="fill" directional="true" pos="3 4 5" dir="0.5 0.2 -0.3" diffuse="0.15 0.15 0.18" castshadow="false"/>
    <geom name="floor" type="plane" size="60 60 0.1" material="fg_floor" friction="{fc["friction"]:g} 0.005 0.0001" solref="{f3(fc["solref"])}" solimp="{f3(fc["solimp"])}" priority="1"/>
    {chr(10).join("    " + c for c in cams).strip()}
    {G.xml()}
    <body name="mocap_root" mocap="true" pos="{f3(root_pos)}" quat="{f3(root_quat)}"/>
    {h1["body"]}
  </worldbody>
  <contact>
    {h1["contact"]}
  </contact>
  <equality>
    <weld name="root_weld" body1="mocap_root" body2="pelvis" relpose="0 0 0 1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    {h1["actuators"]}
  </actuator>
  <sensor>
    {h1["sensors"]}
  </sensor>
</mujoco>
'''
    meta = {"categories": {}, "objects": {}, "landing_mat": mat, "launch": launch["id"] if launch else None,
            "bounds": bounds}
    for g in G.items:
        meta["categories"].setdefault(g["category"], []).append(g["name"])
        meta["objects"][g["name"]] = g["object"]
    meta["categories"]["floor"] = ["floor"]
    meta["objects"]["floor"] = "floor"
    return xml, meta


def _merge(a, b):
    m = _Geoms()
    m.items = a.items + b.items
    return m


# ----------------------------------------------------------------------------- world mode (stage-1 world dir)
def _world_object_index(world_json):
    """geom-name prefix -> (object id, hazard, material) from world.json objects."""
    out = {}
    for o in (world_json or {}).get("objects") or []:
        if isinstance(o, dict) and o.get("id"):
            out[str(o["id"])] = (str(o["id"]), o.get("hazard", True) is not False,
                                 o.get("material") if o.get("material") in MATERIALS else None)
    return out


def _lookup_object(name, index):
    best = None
    for oid in index:
        if name == oid or name.startswith(oid + "__") or name.startswith(oid + "_") or name.startswith(oid + "-"):
            if best is None or len(oid) > len(best):
                best = oid
    return index[best] if best else None


def world_fragments(world_dir, world_json=None):
    """Parse <world_dir>/scene.xml (visual mesh + collision primitives + floor + lights + camera "ref"; no performer)
    into MJCF fragments with absolute asset paths, and classify its geoms. Returns a dict with assets, defaults,
    worldbody (strings), categories {floor, hazard, obstacle: [geom names]}, objects {geom name: object id},
    has_floor, has_lights, cameras [names]."""
    import xml.etree.ElementTree as ET
    path = os.path.join(world_dir, "scene.xml")
    root = ET.parse(path).getroot()
    comp = root.find("compiler")
    base = {k: world_dir for k in ("meshdir", "texturedir", "assetdir")}
    if comp is not None:
        ad = comp.get("assetdir")
        for k in ("meshdir", "texturedir"):
            d = comp.get(k) or ad
            if d:
                base[k] = d if os.path.isabs(d) else os.path.join(world_dir, d)
    assets = []
    a = root.find("asset")
    for e in (list(a) if a is not None else []):
        f = e.get("file")
        if f and not os.path.isabs(f):
            d = base["meshdir"] if e.tag in ("mesh", "hfield", "skin") else base["texturedir"]
            e.set("file", os.path.abspath(os.path.join(d, f)))
        for k in ("fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
            if e.get(k) and not os.path.isabs(e.get(k)):
                e.set(k, os.path.abspath(os.path.join(base["texturedir"], e.get(k))))
        assets.append(ET.tostring(e, encoding="unicode"))
    d = root.find("default")
    defaults = "".join(ET.tostring(e, encoding="unicode") for e in (list(d) if d is not None else []))
    wb = root.find("worldbody")
    if comp is None or comp.get("angle", "degree") == "degree":         # our compiler is radian
        for e in wb.iter():
            if e.get("euler"):
                e.set("euler", f3([math.radians(float(v)) for v in e.get("euler").split()]))
            if e.get("axisangle"):
                v = [float(x) for x in e.get("axisangle").split()]
                e.set("axisangle", f3(v[:3] + [math.radians(v[3])]))
    has_mesh = any(g.get("type") == "mesh" and g.get("contype") == "0" for g in wb.iter("geom"))
    index = _world_object_index(world_json)
    cats = {"floor": [], "hazard": [], "obstacle": []}
    objects = {}
    has_floor = has_lights = False
    cams = []
    n = 0
    for g in wb.iter("geom"):
        n += 1
        name = g.get("name") or f"world_geom_{n}"
        g.set("name", name)
        typ = g.get("type", "sphere")
        visual = g.get("contype") == "0" and g.get("conaffinity") == "0"
        if visual:
            objects[name] = "environment"
            continue
        if typ == "plane" or name.lower().startswith("floor"):
            has_floor = True
            cats["floor"].append(name)
            objects[name] = "floor"
            fc = MATERIALS[(world_json or {}).get("floor", {}).get("material")
                           if (world_json or {}).get("floor", {}).get("material") in MATERIALS else "concrete"]
            g.attrib.setdefault("friction", f'{fc["friction"]:g} 0.005 0.0001')
            g.attrib.setdefault("priority", "1")
            if has_mesh:
                g.set("group", "4")              # collision floor hidden: the mesh's real floor texture shows
            continue
        hit = _lookup_object(name, index)
        oid, hz, mat = hit if hit else (name.split("__")[0], True, None)
        cats["hazard"].append(name)          # every world collision primitive is a hazard for the performer
        if has_mesh:
            g.set("group", "4")                  # the textured mesh is the look; primitives stay hidden in renders
        objects[name] = oid
        c = _contact(mat) if mat else HARD_FRICTION
        g.attrib.setdefault("friction", f'{c["friction"]:g} 0.005 0.0001')
        g.attrib.setdefault("solref", f3(c["solref"]))
        g.attrib.setdefault("solimp", f3(c["solimp"]))
        g.attrib.setdefault("priority", "1")
    for e in wb.iter("light"):
        has_lights = True
    for e in wb.iter("camera"):
        cams.append(e.get("name") or "")
    body = "".join(ET.tostring(e, encoding="unicode") for e in wb)
    has_sky = any(e.tag == "texture" and e.get("type") == "skybox" for e in (list(a) if a is not None else []))
    return {"assets": "\n    ".join(assets), "defaults": defaults, "worldbody": body, "categories": cats,
            "objects": objects, "has_floor": has_floor, "has_lights": has_lights, "cameras": cams,
            "has_mesh": has_mesh, "has_sky": has_sky}


def build_world_xml(plan, stunt, root_pos, root_quat):
    """build_xml for a plan whose plan["world"]["dir"] is a stage-1 world dir: the world's scene.xml (textured
    mesh, collision primitives as hazards, floor, lights, camera "ref") plus the H1, mocap weld, actuators and
    sensors. Crash mats from plan.crash_mats are added on top."""
    import json as _json
    wdir = plan["world"]["dir"]
    wj = {}
    try:
        with open(os.path.join(wdir, "world.json"), encoding="utf-8") as f:
            wj = _json.load(f)
    except (OSError, ValueError):
        pass
    W = world_fragments(wdir, wj)
    stunt = normalize_stunt(stunt if stunt is not None else plan["stunt"])
    plan = {**plan, "stunt": stunt}
    G = _Geoms()
    mat = target_mat(plan, stunt)
    for i, m in enumerate(plan["crash_mats"]):
        pos = mat["pos"] if mat and i == mat["index"] else m["pos"]
        G.add(m["id"], None, "box", m["size"], pos, quat_yaw(math.radians(m["yaw_deg"])), m["rgb"], category="mat",
              contact=MAT_CONTACT)
    m_ = plan["camera"]["match"]
    cams = [] if "match" in W["cameras"] else [camera_xml("match", m_["pos"], m_["lookat"], m_["fovy"], m_.get("roll_deg", 0.0))]
    h1 = perf.compose_h1(root_pos, root_quat)
    fc = MATERIALS[plan["room"]["floor_material"]]
    floor = "" if W["has_floor"] else (
        f'<geom name="floor" type="plane" size="60 60 0.1" rgba="0.5 0.5 0.5 1" friction="{fc["friction"]:g} 0.005 0.0001" '
        f'solref="{f3(fc["solref"])}" solimp="{f3(fc["solimp"])}" priority="1"/>')
    lights = "" if W["has_lights"] else (
        '<light name="key" directional="true" pos="-3 -4 6" dir="0.4 0.5 -1" diffuse="0.5 0.5 0.5" castshadow="true"/>')
    if W["has_mesh"]:            # a visual-only fill floor just under the mesh's floor (where the scan has holes)
        floor += ('<geom name="fg_floor_fill" type="plane" size="60 60 0.1" pos="0 0 -0.015" rgba="0.42 0.42 0.42 1" '
                  'contype="0" conaffinity="0" group="1"/>')
    sky = "" if W["has_sky"] else ('<texture type="skybox" builtin="gradient" rgb1="0.85 0.87 0.9" rgb2="0.6 0.62 0.66" '
                                   'width="256" height="256"/>')
    xml = f'''<mujoco model="fallguy_world_stunt">
  <compiler angle="radian" meshdir="{h1["meshdir"]}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <size memory="256M"/>
  <visual>
    <global offwidth="1920" offheight="1920" fovy="45"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.55 0.55 0.57" diffuse="0.25 0.25 0.25" specular="0.1 0.1 0.1"/>
    <map znear="0.02" zfar="120"/>
  </visual>
  <asset>
    {sky}
    {W["assets"]}
    {h1["assets"]}
  </asset>
  <default>
    {W["defaults"]}
    {h1["default"]}
  </default>
  <worldbody>
    {lights}
    {floor}
    {chr(10).join(cams)}
    {W["worldbody"]}
    {G.xml()}
    <body name="mocap_root" mocap="true" pos="{f3(root_pos)}" quat="{f3(root_quat)}"/>
    {h1["body"]}
  </worldbody>
  <contact>
    {h1["contact"]}
  </contact>
  <equality>
    <weld name="root_weld" body1="mocap_root" body2="pelvis" relpose="0 0 0 1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    {h1["actuators"]}
  </actuator>
  <sensor>
    {h1["sensors"]}
  </sensor>
</mujoco>
'''
    cats = {k: list(v) for k, v in W["categories"].items() if v}
    if not W["has_floor"]:
        cats.setdefault("floor", []).append("floor")
        W["objects"]["floor"] = "floor"
    meta = {"categories": cats, "objects": dict(W["objects"]), "landing_mat": mat, "launch": None,
            "bounds": (plan["room"]["x_min"], plan["room"]["x_max"], plan["room"]["y_min"], plan["room"]["y_max"])}
    for g in G.items:
        meta["categories"].setdefault(g["category"], []).append(g["name"])
        meta["objects"][g["name"]] = g["object"]
    return xml, meta
