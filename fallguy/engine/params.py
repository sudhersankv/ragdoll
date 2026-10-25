"""Candidate parameters: the single source of truth for what the loop may change (contracts.md "Candidate params").

There are no stunt types. Argon writes the stunt as generic phases and marks the numbers the loop may change
in plan.stunt.tunables: [{"name", "path", "min", "max", "unit", "doc"}], where `path` addresses a field of
plan.stunt ("phases[1].speed", "phases[1].path[2][1]", "target.pos[0]", ...). On top of those every plan has
the generic mat_offset_x / mat_offset_y (shift the target crash mat).

  specs(plan)                {name: {min, max, unit, default, doc[, path]}}; default = the plan's current value
  clamp(params, plan)        clean copy: meta keys kept, every spec present, numeric and in range, camera valid
  apply(plan, params)        deep copy of plan.stunt with the values written into their paths (+ target mat offset)
  describe_for_prompt(plan)  compact table for LLM prompts
"""
import copy
import math
import re

CAMERAS = ["match_follow", "match", "hero", "side"]
META_KEYS = ["id", "label", "color", "camera", "why"]
GENERIC = {
    "mat_offset_x": {"min": -1.5, "max": 1.5, "unit": "m", "default": 0.0,
                     "doc": "shift the target crash mat along X (the landing target moves with it)"},
    "mat_offset_y": {"min": -1.5, "max": 1.5, "unit": "m", "default": 0.0,
                     "doc": "shift the target crash mat along Y"},
}
_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(-?\d+)\]")


class PathError(ValueError):
    pass


def parse_path(path):
    """'phases[1].path[2][1]' -> ['phases', 1, 'path', 2, 1]."""
    if not isinstance(path, str) or not path.strip():
        raise PathError(f"tunable path must be a non-empty string, got {path!r}")
    out, pos, s = [], 0, path.strip()
    while pos < len(s):
        if s[pos] == ".":
            pos += 1
            continue
        m = _TOKEN.match(s, pos)
        if not m:
            raise PathError(f"cannot parse tunable path {path!r} at {s[pos:]!r}")
        out.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
        pos = m.end()
    return out


def get_path(obj, path):
    cur = obj
    for key in parse_path(path):
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            raise PathError(f"tunable path {path!r} does not exist in the stunt") from None
    return cur


def set_path(obj, path, value):
    keys = parse_path(path)
    cur = obj
    for key in keys[:-1]:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            raise PathError(f"tunable path {path!r} does not exist in the stunt") from None
    try:
        if isinstance(keys[-1], int):
            cur[keys[-1]]                     # must exist: lists never grow
        cur[keys[-1]] = value
    except (IndexError, TypeError):
        raise PathError(f"tunable path {path!r} does not exist in the stunt") from None


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _stunt(plan):
    st = plan.get("stunt") if isinstance(plan, dict) else None
    return st if isinstance(st, dict) else {}


def specs(plan):
    """{name: {min, max, unit, default, doc, path}} from plan.stunt.tunables, plus the generic mat_offset_x/y.
    Tunables whose path is missing or not a number are skipped (spec_warnings(plan) lists them)."""
    st = _stunt(plan)
    out = {}
    for tn in st.get("tunables") or []:
        if not isinstance(tn, dict) or not isinstance(tn.get("name"), str) or tn["name"] in META_KEYS:
            continue
        try:
            cur = _num(get_path(st, tn.get("path")))
        except PathError:
            continue
        lo, hi = _num(tn.get("min")), _num(tn.get("max"))
        if cur is None or lo is None or hi is None:
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        out[tn["name"]] = {"min": lo, "max": hi, "unit": str(tn.get("unit") or "-"),
                           "default": float(min(max(cur, lo), hi)), "doc": str(tn.get("doc") or ""),
                           "path": tn["path"]}
    for k, v in GENERIC.items():
        out.setdefault(k, dict(v))
    return out


def spec_warnings(plan):
    """Human-readable problems with plan.stunt.tunables (paths that do not exist or are not numbers)."""
    st, warn = _stunt(plan), []
    for tn in st.get("tunables") or []:
        if not isinstance(tn, dict):
            warn.append(f"tunable {tn!r} is not an object")
            continue
        try:
            if _num(get_path(st, tn.get("path"))) is None:
                warn.append(f"tunable {tn.get('name')!r}: {tn.get('path')!r} is not a number")
        except PathError as e:
            warn.append(f"tunable {tn.get('name')!r}: {e}")
    return warn


def default_camera(plan):
    cam = _stunt(plan).get("default_camera")
    return cam if cam in CAMERAS else "match_follow"


def defaults(plan):
    """A complete candidate at the plan's current values."""
    return clamp({}, plan)


def clamp(params, plan):
    """Clean copy of a candidate: META_KEYS kept, every specs(plan) name present, numeric and inside its range
    (missing / non-numeric -> the default, i.e. the plan's current value), camera one of CAMERAS (else the
    plan's default_camera). Unknown keys are dropped."""
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")
    out = {k: params[k] for k in META_KEYS if params.get(k) is not None}
    for name, s in specs(plan).items():
        x = _num(params.get(name))
        out[name] = float(min(max(s["default"] if x is None else x, s["min"]), s["max"]))
    out["camera"] = out.get("camera") if out.get("camera") in CAMERAS else default_camera(plan)
    return out


def apply(plan, params):
    """Deep copy of plan.stunt with each tunable's (clamped) value written into its path; the generic mat
    offsets go to stunt.target.offset = [dx, dy] (engine.scene moves the target mat by it)."""
    st = copy.deepcopy(_stunt(plan))
    p = clamp(params or {}, plan)
    for name, s in specs(plan).items():
        if "path" in s:
            set_path(st, s["path"], p[name])
    tgt = st.get("target") if isinstance(st.get("target"), dict) else {"kind": "none"}
    tgt["offset"] = [p["mat_offset_x"], p["mat_offset_y"]]
    st["target"] = tgt
    return st


def describe_for_prompt(plan):
    """Compact table of the candidate parameters for LLM prompts."""
    lines = ["Candidate params (numbers; values outside the range are clamped):",
             "name | unit | range | current | stunt path | meaning"]
    for k, s in specs(plan).items():
        lines.append(f"{k} | {s['unit']} | {s['min']:g}..{s['max']:g} | {s['default']:g} | "
                     f"{s.get('path', '(target mat)')} | {s['doc']}")
    lines.append(f"camera | - | {', '.join(CAMERAS)} | {default_camera(plan)} | - | camera the take is judged and "
                 "rendered from (match_follow: the director's camera position, panning to follow the performer; "
                 "match: the director's view, fixed; hero: 3/4 tracking; side: tracking side-on)")
    lines.append("Also give each candidate id, label (a few words), color (#rrggbb) and why (one sentence).")
    lines.append("World frame: +X the performer's initial facing, +Y its left, +Z up. Rotations: +roll rolls to the "
                 "performer's right about its heading, +pitch is a front flip, -pitch a back flip, +yaw turns left. "
                 "Every number changes the physics; nothing is overridden.")
    return "\n".join(lines)


def json_schema(plan):
    """JSON Schema (responseJsonSchema dialect) of one candidate for this plan."""
    sp = specs(plan)
    props = {k: {"type": "number", "minimum": s["min"], "maximum": s["max"], "description": s["doc"]}
             for k, s in sp.items()}
    props.update(id={"type": "string"}, label={"type": "string"}, color={"type": "string"},
                 camera={"type": "string", "enum": list(CAMERAS)}, why={"type": "string"})
    return {"type": "object", "properties": props, "required": list(sp) + ["label", "camera", "why"]}
