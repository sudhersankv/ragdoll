"""FALLGUY gates: the single source of Gate 1 / Gate 2 thresholds, score mappings and weights.

Gate 1 (physics safety) reads only MuJoCo numbers from metrics.json. Gate 2 combines Gate 1 with Argon's
critique of the rendered take. Gate 3 is the human director. `python gates.py` prints describe().
"""
import math

THRESHOLDS = {
    "head_g_max": 8.0,             # peak head acceleration, 15 ms moving average, whole stunt (G)
    "lateral_g_max": 0.8,          # peak sideways acceleration while running (G): beyond floor grip the runner slips
    "hazard_collisions_max": 0,    # distinct (body, object) contacts deeper than 0.5 cm with a hazard
    "min_clearance_cm": 2.0,       # closest approach of the performer to any hazard over the stunt
    "landing_error_max_m": 0.65,   # landing / finish point to the stunt target (mat centre or zone centre)
    "require_inside_mat": True,    # inside the target: on the target mat, or within the zone radius
    "framing_min_pct": 85.0,       # frames (start of motion to finish + 0.5 s) with the whole performer in frame
    "gate2_min_score": 85.0,       # Gate 2: stunt score recomputed with Argon's framing
    "style_min": 70,               # Gate 2: Argon's style score
}
WEIGHTS = {"safety": 0.35, "clearance": 0.25, "landing": 0.20, "framing": 0.20}

# clearance score knots (cm -> score); 2-10 cm rises to 100, flat to 25 cm, then the over-jump decay
CLEARANCE_KNOTS = [(0.0, 0.0), (2.0, 50.0), (10.0, 100.0), (25.0, 100.0), (50.0, 50.0), (100.0, 20.0)]
LANDING_ZERO_M = 1.0


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _interp(x, knots):
    if x <= knots[0][0]:
        return knots[0][1]
    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return knots[-1][1]


def safety_score(head_g):
    """100 at or below half the head-G limit, falling linearly to 0 at twice the limit."""
    g = _num(head_g)
    if g is None:
        return 0.0
    lim = THRESHOLDS["head_g_max"]
    lo, hi = 0.5 * lim, 2.0 * lim
    return round(100.0 * min(1.0, max(0.0, (hi - g) / (hi - lo))), 1)


def lateral_score(lateral_g):
    """100 up to 80 % of the lateral-g limit, falling linearly to 0 at 1.5x the limit (missing = 100)."""
    g = _num(lateral_g)
    if g is None:
        return 100.0
    lim = THRESHOLDS["lateral_g_max"]
    lo, hi = 0.8 * lim, 1.5 * lim
    return round(100.0 * min(1.0, max(0.0, (hi - g) / (hi - lo))), 1)


def clearance_score(clearance_cm, over_target=True):
    """0 when penetrating, 50 at 2 cm rising to 100 at 10 cm, flat to 25 cm, then decaying for an
    over-jump (50 at 50 cm, 20 at 1 m and beyond). Without a hazard under the flight path
    (over_target False) a large clearance is not an over-jump and keeps 100."""
    c = _num(clearance_cm)
    if c is None:
        return 0.0
    if c >= 10.0 and not over_target:
        return 100.0
    return round(_interp(c, CLEARANCE_KNOTS), 1)


def landing_score(error_m, inside_mat=True):
    """100 at 0 m, falling linearly to 0 at 1.0 m; 0 when the landing is not on the mat."""
    e = _num(error_m)
    if e is None or inside_mat is False:
        return 0.0
    return round(100.0 * max(0.0, 1.0 - e / LANDING_ZERO_M), 1)


def framing_score(framing_pct):
    f = _num(framing_pct)
    return 0.0 if f is None else round(min(100.0, max(0.0, f)), 1)


def stunt_score(scores):
    return round(sum(WEIGHTS[k] * float(scores.get(k) or 0.0) for k in WEIGHTS), 1)


def _scores(g1, framing=None):
    land = g1.get("landing") or {}
    s = {"safety": min(safety_score(g1.get("head_g_peak")), lateral_score(g1.get("lateral_g_peak"))),
         "clearance": clearance_score(g1.get("min_clearance_cm"), g1.get("clearance_over_target", True) is not False),
         "landing": landing_score(land.get("error_m"), land.get("inside_mat")),
         "framing": framing_score(g1.get("framing_pct") if framing is None else framing)}
    s["stunt_score"] = stunt_score(s)
    return s


def evaluate_gate1(metrics):
    """Fill metrics["gate1"]["passed"], ["failures"] (short strings with numbers) and metrics["score"];
    returns metrics. A failed simulation (ok false) fails Gate 1 with its error."""
    T = THRESHOLDS
    g1 = metrics.setdefault("gate1", {})
    if not isinstance(g1, dict):
        g1 = metrics["gate1"] = {}
    fails = []
    if metrics.get("ok") is False:
        fails.append(f"simulation failed: {metrics.get('error') or 'unknown error'}")
    head = _num(g1.get("head_g_peak"))
    if head is None:
        fails.append("head G missing")
    elif head > T["head_g_max"]:
        at = _num(g1.get("head_g_peak_t"))
        fails.append(f"head G {head:.1f} > {T['head_g_max']:g} G" + (f" at t={at:.2f} s" if at is not None else ""))
    lat = _num(g1.get("lateral_g_peak"))
    if lat is not None and lat > T["lateral_g_max"]:
        at = _num(g1.get("lateral_g_peak_t"))
        fails.append(f"lateral g {lat:.2f} > {T['lateral_g_max']:g} G" + (f" at t={at:.2f} s" if at is not None else "")
                     + " (the runner would slip: slower or wider turn)")
    hc = _num(g1.get("hazard_collisions"))
    if hc is None:
        fails.append("hazard collisions missing")
    elif hc > T["hazard_collisions_max"]:
        first = [c for c in g1.get("hazard_contacts") or [] if isinstance(c, dict)][:2]
        where = ", ".join(f"{c.get('body')} x {c.get('object')} {_num(c.get('depth_cm')) or 0:.1f} cm deep at "
                          f"{_num(c.get('t')) or 0:.2f} s" for c in first)
        fails.append(f"{int(hc)} hazard collision{'s' if hc != 1 else ''}" + (f" ({where})" if where else ""))
    mc = _num(g1.get("min_clearance_cm"))
    if mc is None:
        fails.append("min clearance missing")
    elif mc < T["min_clearance_cm"]:
        at = g1.get("min_clearance_at") or {}
        where = (f" ({at.get('body')} to {at.get('object')} at {_num(at.get('t')) or 0:.2f} s)"
                 if isinstance(at, dict) and at.get("body") else "")
        fails.append(f"min clearance {mc:.1f} cm < {T['min_clearance_cm']:g} cm{where}")
    land = g1.get("landing") or {}
    err = _num(land.get("error_m"))
    if err is None:
        fails.append("landing error missing (no landing / target found)")
    elif err > T["landing_error_max_m"]:
        fails.append(f"{'finish' if land.get('target_kind') == 'zone' else 'landing'} error {err:.2f} m > "
                     f"{T['landing_error_max_m']:g} m")
    if T["require_inside_mat"] and land.get("inside_mat") is not True:
        if land.get("target_kind") == "zone":
            fails.append("finished outside the target zone")
        else:
            fails.append(f"landed on {land.get('surface') or 'nothing'}, not inside the crash mat")
    fp = _num(g1.get("framing_pct"))
    if fp is None:
        fails.append("framing missing")
    elif fp < T["framing_min_pct"]:
        fails.append(f"framing {fp:.0f}% < {T['framing_min_pct']:g}%")
    g1["passed"] = not fails
    g1["failures"] = fails
    metrics["score"] = _scores(g1)
    return metrics


def evaluate_gate2(metrics, critique):
    """Gate 2: Gate 1 passed, Argon's gate2_pass is true, style_score >= style_min and the final score
    (stunt score with framing = min(sim framing, Argon's in_frame_pct)) >= gate2_min_score.
    Returns {"passed", "reasons", "final_score"}; inputs are not modified."""
    T = THRESHOLDS
    metrics = metrics if isinstance(metrics, dict) else {}
    critique = critique if isinstance(critique, dict) else {}
    g1 = metrics.get("gate1") if isinstance(metrics.get("gate1"), dict) else {}
    reasons = []
    if g1.get("passed") is not True:
        reasons.append("Gate 1 not passed" + (": " + "; ".join(map(str, g1.get("failures") or [])) if g1.get("failures") else ""))
    if critique.get("gate2_pass") is not True:
        reasons.append("Argon's critique says gate2_pass false")
    style = _num(critique.get("style_score"))
    if style is None:
        reasons.append("Argon gave no style score")
    elif style < T["style_min"]:
        reasons.append(f"style {style:.0f} < {T['style_min']:g}")
    sim_f = _num(g1.get("framing_pct"))
    argon_f = _num((critique.get("framing") or {}).get("in_frame_pct")) if isinstance(critique.get("framing"), dict) else None
    framing = min(v for v in (sim_f, argon_f) if v is not None) if (sim_f is not None or argon_f is not None) else None
    final = _scores(g1, framing=framing if framing is not None else 0.0)["stunt_score"]
    if final < T["gate2_min_score"]:
        reasons.append(f"final score {final:.1f} < {T['gate2_min_score']:g}"
                       + (f" (framing {framing:.0f}%)" if framing is not None else ""))
    return {"passed": not reasons, "reasons": reasons, "final_score": final}


def describe():
    T, W = THRESHOLDS, WEIGHTS
    return "\n".join([
        "Gate 1 - MuJoCo physics safety (all must hold):",
        f"  head G peak (15 ms average, whole stunt) <= {T['head_g_max']:g} G",
        f"  lateral g while running (sideways acceleration in turns) <= {T['lateral_g_max']:g} G",
        f"  hazard collisions (performer-hazard contact deeper than 0.5 cm, per body/object pair) <= {T['hazard_collisions_max']}",
        f"  min clearance to any hazard over the stunt >= {T['min_clearance_cm']:g} cm",
        f"  landing / finish error to the stunt target (mat centre or zone centre) <= {T['landing_error_max_m']:g} m",
        "  inside the target (first touch on the target mat over the mat, or the finish within the zone radius)"
        if T["require_inside_mat"] else "  target not required",
        f"  framing (whole performer in frame, start of motion to finish + 0.5 s) >= {T['framing_min_pct']:g}%",
        "Gate 2 - Argon video critique: Gate 1 passed, critique gate2_pass true, "
        f"style_score >= {T['style_min']:g}, final score >= {T['gate2_min_score']:g}",
        "  final score = stunt score with framing = min(sim framing_pct, critique framing.in_frame_pct)",
        "Gate 3 - the director locks the take.",
        "Stunt score = " + " + ".join(f"{w:.2f} x {k}" for k, w in W.items()) + " (each 0-100):",
        f"  safety: min of head (100 at <= {0.5 * T['head_g_max']:g} G, 0 at {2 * T['head_g_max']:g} G) and lateral "
        f"(100 at <= {0.8 * T['lateral_g_max']:g} G, 0 at {1.5 * T['lateral_g_max']:g} G)",
        "  clearance: 0 when penetrating, 50 at 2 cm, 100 from 10 to 25 cm, then over-jump decay "
        "(50 at 50 cm, 20 at >= 1 m; no decay when no hazard lies under the flight path)",
        f"  landing: 100 at 0 m falling to 0 at {LANDING_ZERO_M:g} m; 0 if not inside the target",
        "  framing: the framing percentage",
    ])


if __name__ == "__main__":
    print(describe())
