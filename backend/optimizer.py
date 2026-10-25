"""Local numeric search used inside each wave.

The recursive loop is a hybrid:
- the *proposer* (Gemini 4 Argon, or `propose_offline` without a key) picks 4
  semantically different strategies using the full attempt history;
- `refine` then runs a small cross-entropy search in MuJoCo around each
  proposal, so every ghost shown is the best local variant of its strategy.

`propose_offline` is a physics-informed heuristic proposer: it reads Gate 1
failures (collision, overshoot, head load, missed mat) and applies the
corresponding correction, plus ballistic re-aiming of the landing point.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.params import PARAM_SPACE, clamp_params

SEARCH_KEYS = [k for k in PARAM_SPACE if k != "mat_offset_x"]


def refine(
    seed: Dict[str, Any],
    evaluate: Callable[[Dict[str, Any]], Dict[str, Any]],
    iters: int = 2,
    pop: int = 8,
    n_elite: int = 3,
    sigma_scale: float = 1.0,
    rng: Optional[random.Random] = None,
    lock: Tuple[str, ...] = ("mat_offset_x",),
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Cross-entropy refinement around `seed`. Returns (best_params, best_result, all_evals)."""
    rng = rng or random.Random(0)
    keys = [k for k in PARAM_SPACE if k not in lock]
    mean = clamp_params(seed)
    sigma = {k: PARAM_SPACE[k][2] * sigma_scale for k in keys}
    best_res = evaluate(mean)
    best = mean
    evals = [best_res]
    for _ in range(iters):
        samples = []
        for _ in range(pop):
            cand = dict(mean)
            for k in keys:
                cand[k] = mean[k] + rng.gauss(0.0, sigma[k])
            cand = clamp_params(cand)
            res = evaluate(cand)
            evals.append(res)
            samples.append((res["stunt_score"] + (15 if res["gate_1_passed"] else 0), cand, res))
            if _better(res, best_res):
                best, best_res = cand, res
        samples.sort(key=lambda s: s[0], reverse=True)
        elites = [s[1] for s in samples[:n_elite]] + [best]
        for k in keys:
            vals = [e[k] for e in elites]
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            mean[k] = m
            sigma[k] = max(PARAM_SPACE[k][2] * 0.15, 0.6 * sigma[k] + 0.4 * sd)
        mean = clamp_params(mean)
    return best, best_res, evals


def _better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a["gate_1_passed"], a["stunt_score"]) > (b["gate_1_passed"], b["stunt_score"])


def rank_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(history, key=lambda h: (h["gate_1_passed"], h["stunt_score"]), reverse=True)


def _correct(params: Dict[str, Any], entry: Dict[str, Any], scene: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Apply physics-informed corrections for the Gate 1 failures of `entry`."""
    p = dict(params)
    m = entry["metrics"]
    notes: List[str] = []
    diag = entry.get("diagnostic", "")

    if "hit" in diag or "before clearing" in diag:
        p["takeoff_vz"] += 0.35
        p["tuck_time"] -= 0.04
        p["knee_tuck_deg"] += 12
        p["hip_tuck_deg"] += 12
        notes.append(f"collision/short -> vz +0.35 m/s, tuck 40 ms earlier, deeper tuck")
    elif m["min_clearance_cm"] > 40:
        p["takeoff_vz"] -= 0.3
        notes.append(f"over-jumped (+{m['min_clearance_cm']}cm) -> vz -0.3 m/s")

    # Ballistic re-aim: shift landing toward mat centre using touchdown time.
    mat = next((o for o in scene["scene_obstacles"] if o["id"] == "crash_mat"), None)
    td_t = m.get("touchdown_time_s")
    if mat and td_t:
        target_x = mat["pos"][0] + p.get("mat_offset_x", 0.0) - 0.15
        dx = target_x - m["touchdown_x"]
        if abs(dx) > 0.15:
            dv = max(-1.2, min(1.2, dx / max(0.3, td_t)))
            p["takeoff_vx"] += dv
            notes.append(f"touchdown {dx:+.2f}m from aim -> vx {dv:+.2f} m/s")

    if m["peak_head_g"] >= 8.0 or m["hic15"] >= 250 or "head struck" in diag:
        p["landing_compliance"] = max(0.72, p.get("landing_compliance", 0.5) + 0.22)
        p["pitch_rate"] *= 0.65
        p["extend_time"] -= 0.05
        notes.append(f"head load {m['peak_head_g']}g / HIC {m['hic15']:.0f} -> softer landing compliance, less rotation, earlier extension")
    return clamp_params(p), notes


def propose_offline(
    history: List[Dict[str, Any]],
    scene: Dict[str, Any],
    steer: Optional[Dict[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return 4 seeds for the next wave + human-readable mutation notes."""
    rng = rng or random.Random(len(history))
    ranked = rank_history(history)
    best = ranked[0]
    second = next((h for h in ranked[1:] if h["parameters"] != best["parameters"]), best)

    corrected, notes = _correct(best["parameters"], best, scene)
    alt, alt_notes = _correct(second["parameters"], second, scene)

    def jitter(p, scale):
        q = dict(p)
        for k in ("takeoff_vx", "takeoff_vz", "pitch_rate", "landing_compliance"):
            q[k] += rng.gauss(0, PARAM_SPACE[k][2] * scale)
        return clamp_params(q)

    seeds = [
        {**corrected, "label": "Exploit: corrected best"},
        {**jitter(corrected, 0.6), "label": "Local variant of best"},
        {**alt, "label": "Corrected runner-up"},
        {**jitter(best["parameters"], 1.5), "label": "Explore around best"},
    ]
    if steer:
        steered = []
        for s in seeds:
            q = dict(s)
            for k, dv in steer.items():
                if k in q:
                    q[k] = q[k] + dv
            steered.append({**clamp_params(q), "label": s["label"]})
        seeds = steered
        notes.append("director steer applied: " + ", ".join(f"{k} {v:+g}" for k, v in steer.items()))
    return seeds, notes + [f"(runner-up) {n}" for n in alt_notes[:1]]
