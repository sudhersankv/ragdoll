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
MATERIAL_OF_ROLE = {"stunt_target": "wood", "furniture": "wood", "prop": "plastic", "structure": "concrete"}
DEFAULT_RGB = {"stunt_target": (0.55, 0.40, 0.25), "furniture": (0.45, 0.42, 0.40), "prop": (0.7, 0.7, 0.7),
               "structure": (0.6, 0.6, 0.62)}
KINDS, ROLES = ("box", "cylinder", "sphere"), ("stunt_target", "furniture", "prop", "structure")
VIDEO_ASPECT = 16 / 9

DEFAULT_PLAN = {
    "version": 1, "brief": "", "duration_sec": 9.0,
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

    st = p.get("stunt") if isinstance(p.get("stunt"), dict) else {}
    st["type"] = st.get("type") if st.get("type") in perf_types() else "custom"
    st["takeoff_x"] = _fnum(st.get("takeoff_x"), "stunt.takeoff_x", -1.0)
    st["lane_y"] = _fnum(st.get("lane_y"), "stunt.lane_y", 0.0)
    st["start_x"] = _fnum(st.get("start_x"), "stunt.start_x", st["takeoff_x"] - 3.0)
    land_default = mats[0]["pos"][0] if mats else st["takeoff_x"] + 2.6
    st["landing_x"] = _fnum(st.get("landing_x"), "stunt.landing_x", land_default)
    st.setdefault("notes", "")
    p["stunt"] = st
    cam = (p.get("camera") or {}).get("match") or {}
    try:
        p["camera"]["match"] = {"pos": _vec(cam.get("pos"), 3, "camera.match.pos"),
                                "lookat": _vec(cam.get("lookat"), 3, "camera.match.lookat"),
                                "fovy": min(max(_fnum(cam.get("fovy"), "camera.match.fovy", 50.0), 10.0), 120.0),
                                "roll_deg": _fnum(cam.get("roll_deg"), "camera.match.roll_deg", 0.0)}
    except PlanError:
        x0 = st["start_x"]
        p["camera"]["match"] = {"pos": [x0 - 2.5, st["lane_y"], 1.6], "lookat": [st["takeoff_x"], st["lane_y"], 0.9],
                                "fovy": 50.0, "roll_deg": 0.0}
    perf.check_model((p.get("performer") or {}).get("model"))
    p["performer"] = {"model": "h1"}
    return p


def perf_types():
    from .params import STUNT_TYPES
    return STUNT_TYPES


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
        "stunt": {"type": "barrel_roll", "start_x": start_x, "takeoff_x": takeoff_x, "landing_x": landing_x,
                  "lane_y": 0.0, "notes": "dive over the shelf rotating 360 deg about the travel axis, land on the mat"},
        "performer": {"model": "h1"},
    }


# ----------------------------------------------------------------------------- landing mat / launch platform
def landing_mat(plan, params=None):
    """The crash mat the stunt lands on (closest to the planned landing point), moved by mat_offset_x/y:
    {"id", "pos", "size", "yaw", "top", "index"}; None when the plan has no mat."""
    mats = plan.get("crash_mats") or []
    if not mats:
        return None
    st = plan["stunt"]
    tx, ty = st["landing_x"], st["lane_y"]
    i = min(range(len(mats)), key=lambda k: math.hypot(mats[k]["pos"][0] - tx, mats[k]["pos"][1] - ty))
    m = mats[i]
    ox = float((params or {}).get("mat_offset_x", 0.0) or 0.0)
    oy = float((params or {}).get("mat_offset_y", 0.0) or 0.0)
    pos = [m["pos"][0] + ox, m["pos"][1] + oy, m["pos"][2]]
    return {"id": m["id"], "index": i, "pos": pos, "size": list(m["size"]), "yaw": math.radians(m.get("yaw_deg", 0.0)),
            "top": pos[2] + m["size"][2]}


def inside_footprint(xy, pos, size, yaw, margin=0.0):
    """Is the world point xy over the (yawed) box footprint with half-extents size[:2]?"""
    dx, dy = xy[0] - pos[0], xy[1] - pos[1]
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return abs(lx) <= size[0] + margin and abs(ly) <= size[1] + margin


def obstacle_top(ob):
    return ob["pos"][2] + (ob["size"][2] if ob["kind"] == "box" else ob["size"][1] if ob["kind"] == "cylinder" else ob["size"][0])


def launch_platform(plan):
    """For high_fall: the obstacle the performer starts on (the highest box/cylinder under the take-off
    point, else the stunt target); None if there is none."""
    st = plan["stunt"]
    pt = (st["takeoff_x"] - 0.1, st["lane_y"])
    best = None
    for ob in plan["obstacles"]:
        if ob["kind"] == "sphere":
            continue
        size = ob["size"] if ob["kind"] == "box" else [ob["size"][0], ob["size"][0]]
        if inside_footprint(pt, ob["pos"], size, math.radians(ob["yaw_deg"]), margin=0.3):
            if best is None or obstacle_top(ob) > obstacle_top(best):
                best = ob
    if best is None:
        targets = [ob for ob in plan["obstacles"] if ob["role"] == "stunt_target" and ob["kind"] != "sphere"]
        best = max(targets, key=obstacle_top) if targets else None
    return best


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


def _stunt_region(plan, z_top):
    st = plan["stunt"]
    mat = landing_mat(plan)
    land_x = max(st["landing_x"], mat["pos"][0] if mat else st["landing_x"])
    xa, xb = min(st["start_x"], st["takeoff_x"]) - 0.6, max(land_x, st["takeoff_x"]) + 1.6
    y = st["lane_y"]
    pts = np.array([[x_, y_, z_] for x_ in (xa, xb) for y_ in (y - 0.7, y + 0.7) for z_ in (0.0, z_top)], float)
    return pts, np.array([(xa + xb) / 2, y, 0.45 * z_top])


def _cameras(plan, bounds, sd_points):
    x0, x1, y0, y1 = bounds
    H = plan["room"]["ceiling_z"]
    box = (x0 + 0.35, x1 - 0.35, y0 + 0.35, y1 - 0.35, H - 0.2)
    z_top = min(3.2, H - 0.3)
    pts, look = _stunt_region(plan, z_top)
    st = plan["stunt"]
    # side: perpendicular to the lane, on the side with less set dressing between camera and lane
    best = None
    for sgn in (-1.0, 1.0):
        between = sum(1 for p in sd_points if (p[1] - st["lane_y"]) * sgn > 0.8 and abs(p[0] - look[0]) < 6.0)
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
    st = plan["stunt"]
    pts += [(st["start_x"] - 1.0, st["lane_y"] - 1.0), (st["landing_x"] + 1.5, st["lane_y"] + 1.0),
            tuple(plan["camera"]["match"]["pos"][:2])]
    for p in pts:
        x0, x1 = min(x0, p[0] - 0.3), max(x1, p[0] + 0.3)
        y0, y1 = min(y0, p[1] - 0.3), max(y1, p[1] + 0.3)
    return x0, x1, y0, y1


def build_xml(plan, params, root_pos, root_quat):
    """MJCF of the stunt scene for one candidate (crash mat moved by its mat_offset) with the H1 at the given
    root pose. plan must come from normalize_plan(). Returns (xml, meta); meta lists geom names by category
    ("hazard", "mat", "floor", "obstacle", "launch"), the object of every named geom and the landing mat."""
    G = _Geoms()
    st = plan["stunt"]
    launch = launch_platform(plan) if st["type"] == "high_fall" else None
    for ob in plan["obstacles"]:
        cat = "launch" if launch is not None and ob["id"] == launch["id"] else ("hazard" if ob["hazard"] else "obstacle")
        _obstacle(G, ob, cat)
    mat = landing_mat(plan, params)
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
    <global offwidth="1920" offheight="1080" fovy="45"/>
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
