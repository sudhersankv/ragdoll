"""Stage 4a: Argon watches a take's video.mp4 with the MuJoCo metrics and biomechanics as ground truth.

python -m fallguy.stunt critique --stunt DIR --take v1
Writes <take>/critique.json.
"""
import json
import os

from .common import argon_log, rj, say, wj
from .simulate import take_choreography

PROMPT = """You are the stunt safety supervisor and second-unit director for FALLGUY. Watch this render of a
simulated stunt take (a Unitree H1 humanoid robot, 1.8 m, 51 kg, in a 3D reconstruction of the director's location,
rendered from the director's camera position, panning to follow). The H1's links are coloured by a biomechanical
stress index: green < 0.5, amber 0.5-1, red >= 1 of the literature limit. The HUD shows the head G.

Director's brief: "{brief}"
Requested length: {duration:g} s. Take: {take}.
Video timing: {timing}

Choreography that was simulated (world coordinates, metres):
{choreo}
Tunables (name: path = current value [range] unit):
{tunables}

Ground truth from MuJoCo (trust these numbers over what you see; cite them):
{metrics}
Biomechanics ({method}):
{bio}

Judge:
1. events: timestamped (video seconds) start, turns, stops, takeoff/landing, anything notable.
2. brief match: does the take do what the brief asks, in this location, with directions as the director meant them?
3. motion plausibility: does the H1 move like a credible stunt performer (gait, lean in turns, foot sliding, stiffness,
   speed)? list issues with times and severity.
4. safety findings: cite the measured numbers (head G vs the 8 g stunt target and 80 g severe, HIC15 vs 700,
   knee/hip/ankle torque vs the references, GRF vs 10 x BW, lateral g vs 0.8, clearance to hazards, red links).
5. changes for the next take: concrete numeric changes. Prefer tunables (use the tunable name); for anything else
   give a path into the choreography (e.g. phases[1].path[2][0], phases[1].decel_time, phases[2].duration) and the
   new value. Fix safety failures first (e.g. lateral g > 0.8 g: lower the speed or widen turn_radius so
   v^2/r <= 7.8 m/s^2), then the brief match and look. 1-6 changes.
Set approve=true only if the take matches the brief, passes the safety limits and looks credible."""


def _schema():
    from .. import argon as A
    ev = A._o({"t_sec": A._n(), "event": A._e(A.EVENT_KINDS), "note": A._s()}, required=["t_sec", "event"])
    issue = A._o({"t_sec": A._n(), "issue": A._s(), "severity": A._e(A.SEVERITIES)}, required=["issue", "severity"])
    finding = A._o({"t_sec": A._n(), "finding": A._s(), "metric": A._s("e.g. head_g, knee_torque, lateral_g"),
                    "value": A._n(), "limit": A._n(), "severity": A._e(A.SEVERITIES)},
                   required=["finding", "severity"])
    change = A._o({"name": A._s("a tunable name, or empty when `path` is used"),
                   "path": A._s("path into the choreography when not a tunable, e.g. phases[1].path[2][0]"),
                   "value": A._n("the new value"), "reason": A._s("cite the number this fixes")},
                  required=["value", "reason"])
    return A._o({
        "summary": A._s("2-3 sentences"),
        "events": A._a(ev),
        "brief_match": A._o({"matches": A._b(), "notes": A._s(), "missing": A._a(A._s())}, required=["matches"]),
        "motion_plausibility": A._o({"score": A._n("0-100"), "issues": A._a(issue)}, required=["score"]),
        "safety_findings": A._a(finding),
        "framing": A._o({"notes": A._s(), "in_frame_pct": A._n()}),
        "overall_score": A._n("0-100"),
        "approve": A._b(),
        "changes": A._a(change, "concrete changes for the next take"),
    }, required=["summary", "events", "brief_match", "motion_plausibility", "safety_findings", "overall_score",
                 "approve", "changes"])


def metrics_text(m):
    g = m.get("gate1") or {}
    s = m.get("score") or {}
    lines = [f"head G peak {g.get('head_g_peak')} g at t={g.get('head_g_peak_t')} s; HIC15 {g.get('hic15')}",
             f"lateral g peak in moves {g.get('lateral_g_peak')} g at t={g.get('lateral_g_peak_t')} s (limit 0.8: slip)",
             f"hazard contacts {g.get('hazard_collisions')}: {json.dumps(g.get('hazard_contacts'))[:300]}",
             f"min clearance to hazards {g.get('min_clearance_cm')} cm at {json.dumps(g.get('min_clearance_at'))}",
             f"end point {json.dumps(g.get('landing'))}", f"framing {g.get('framing_pct')} % in frame",
             f"net rotation {json.dumps(g.get('rotation_deg'))}",
             f"gate 1 passed: {g.get('passed')}; failures: {g.get('failures')}; scores {json.dumps(s)}",
             f"sim: {json.dumps(m.get('sim'))[:400]}"]
    return "\n".join(lines)


def choreo_text(st):
    from .. import argon as A
    lines = [f"summary: {st.get('summary')}", f"start: {st.get('start')}"]
    lines += [A.phase_line(i, p) for i, p in enumerate(st.get("phases") or [])]
    lines.append(f"target: {st.get('target')}")
    return "\n".join(lines)


def tunables_text(st):
    from ..engine import params as P
    out = []
    for t in st.get("tunables") or []:
        try:
            cur = P.get_path(st, t["path"])
        except Exception:                                        # noqa: BLE001
            cur = "?"
        out.append(f"{t['name']}: {t['path']} = {cur} [{t['min']}..{t['max']}] {t.get('unit', '')}")
    return "\n".join(out) or "(none)"


def critique(stunt_dir, take, fps=8):
    from .. import argon as A
    from .. import biomech
    stunt_dir = os.path.abspath(stunt_dir)
    S = rj(os.path.join(stunt_dir, "stunt.json"))
    td = os.path.join(stunt_dir, take)
    video = os.path.join(td, "video.mp4")
    m = rj(os.path.join(td, "metrics.json"))
    b = rj(os.path.join(td, "biomech.json"))
    rep = rj(os.path.join(td, "render.json")) if os.path.isfile(os.path.join(td, "render.json")) else {}
    st = take_choreography(stunt_dir, take)
    win = rep.get("sim_window") or [0, 0]
    fac = rep.get("slowmo_factor") or 1.0
    timing = (f"the video shows sim time {win[0]:.2f}..{win[1]:.2f} s; on the ground video time = sim time - "
              f"{win[0]:.2f} s" + (f"; airborne parts play at {fac:g}x speed" if fac and fac < 0.99 else "")
              + f" (+{rep.get('hold_frames', 0)} held frames at the end). Times in the numbers below are sim time.")
    text = PROMPT.format(brief=S["brief"], duration=S["duration_sec"], take=take, timing=timing,
                         choreo=choreo_text(st), tunables=tunables_text(st), metrics=metrics_text(m),
                         method="; ".join(f"{k}: {v}" for k, v in b["method"].items()), bio=biomech.summary_text(b))
    log = argon_log(stunt_dir)
    cache = os.path.join(stunt_dir, "logs", "uploads.json")
    ans, _ = A.generate_with_video(video, fps, lambda part: [part, {"text": text}], _schema(), f"critique_{take}",
                                   log, cache)
    ans["take"] = take
    ans["video"] = video
    wj(os.path.join(td, "critique.json"), ans)
    say(f"critique {take}: score {ans.get('overall_score')}, approve {ans.get('approve')}: {ans.get('summary')}")
    return ans
