"""Stunt policy parameter space.

A candidate stunt is a compact vector of biomechanical parameters that drive
the 4-phase PD controller in `physics.py`. Every proposal (from Gemini or the
local optimizer) is clamped into these bounds before it touches MuJoCo.
"""

from __future__ import annotations

from typing import Any, Dict

# name -> (low, high, initial search sigma, unit, meaning)
PARAM_SPACE: Dict[str, tuple] = {
    "takeoff_vx": (1.5, 7.5, 0.5, "m/s", "horizontal launch speed toward the landing zone"),
    "takeoff_vz": (1.0, 6.0, 0.4, "m/s", "vertical launch speed"),
    "pitch_rate": (-3.0, 9.0, 1.0, "rad/s", "initial forward rotation (+ = head goes forward/down)"),
    "tuck_time": (0.05, 0.60, 0.06, "s", "time after takeoff when the tuck begins"),
    "extend_time": (0.25, 1.20, 0.08, "s", "time after takeoff when legs open up for landing"),
    "knee_tuck_deg": (0.0, 130.0, 12.0, "deg", "knee flexion during tuck"),
    "hip_tuck_deg": (0.0, 120.0, 12.0, "deg", "hip flexion during tuck"),
    "arm_reach_deg": (0.0, 160.0, 15.0, "deg", "shoulder forward raise"),
    "landing_compliance": (0.0, 1.0, 0.12, "0-1", "landing softness: deeper knee/hip absorb and lower stiffness"),
    "mat_offset_x": (-0.8, 0.8, 0.12, "m", "crash mat shift along the stunt axis (rigging choice)"),
}

DEFAULT_PARAMS: Dict[str, float] = {
    "takeoff_vx": 4.0,
    "takeoff_vz": 3.5,
    "pitch_rate": 0.5,
    "tuck_time": 0.15,
    "extend_time": 0.60,
    "knee_tuck_deg": 80.0,
    "hip_tuck_deg": 80.0,
    "arm_reach_deg": 70.0,
    "landing_compliance": 0.6,
    "mat_offset_x": 0.0,
}

META_KEYS = ("ghost_id", "label", "color", "origin")


def clamp_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy with every numeric param inside bounds (missing -> default)."""
    out: Dict[str, Any] = {k: raw[k] for k in META_KEYS if k in raw}
    for name, (lo, hi, *_rest) in PARAM_SPACE.items():
        try:
            val = float(raw.get(name, DEFAULT_PARAMS[name]))
        except (TypeError, ValueError):
            val = DEFAULT_PARAMS[name]
        if val != val:  # NaN
            val = DEFAULT_PARAMS[name]
        out[name] = round(min(hi, max(lo, val)), 3)
    # The landing extension must come after the tuck starts.
    if out["extend_time"] < out["tuck_time"] + 0.08:
        out["extend_time"] = round(min(PARAM_SPACE["extend_time"][1], out["tuck_time"] + 0.08), 3)
    return out


def param_space_doc() -> str:
    lines = []
    for name, (lo, hi, _s, unit, meaning) in PARAM_SPACE.items():
        lines.append(f"- {name} [{lo}, {hi}] {unit}: {meaning}")
    return "\n".join(lines)
