"""Built-in stunt scenes for testing without location footage.

Coordinates are MuJoCo world frame (Z-up, metres). X is the stunt axis
(takeoff -> landing), Y is depth, Z is height. Boxes use half-extents.
`is_hazard` boxes must never be touched; `crash_mat` is the landing target.
"""

from __future__ import annotations

from typing import Any, Dict, List

GHOST_COLORS = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]


def _seeds(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        seed = dict(row)
        seed["ghost_id"] = f"ghost_{'ABCD'[i]}"
        seed["color"] = GHOST_COLORS[i]
        seed.setdefault("origin", "preset")
        out.append(seed)
    return out


STUNT_PRESETS: Dict[str, Dict[str, Any]] = {
    "couch_vault": {
        "id": "couch_vault",
        "name": "Couch Dive-Vault",
        "description": "Launch over a couch, clear the backrest, land on the crash mat without head/neck injury.",
        "start_pos": [-1.6, 0.0, 1.05],
        "camera": {"pos": [0.4, -4.6, 1.5], "lookat": [0.5, 0.0, 0.7], "fovy": 50},
        "scene_obstacles": [
            {"id": "couch_seat", "type": "box", "pos": [0.30, 0.0, 0.22], "size": [0.45, 0.95, 0.22],
             "rgba": [0.0, 0.82, 1.0, 0.45], "is_hazard": True},
            {"id": "couch_backrest", "type": "box", "pos": [0.62, 0.0, 0.42], "size": [0.13, 0.95, 0.42],
             "rgba": [0.0, 0.65, 1.0, 0.55], "is_hazard": True},
            {"id": "crash_mat", "type": "box", "pos": [2.25, 0.0, 0.10], "size": [0.85, 0.85, 0.10],
             "rgba": [0.06, 0.92, 0.50, 0.55], "is_hazard": False},
        ],
        "wave_1_candidates": _seeds([
            {"label": "Flat low jump", "takeoff_vx": 3.2, "takeoff_vz": 2.2, "pitch_rate": 0.0,
             "tuck_time": 0.30, "extend_time": 0.60, "knee_tuck_deg": 30, "hip_tuck_deg": 30,
             "arm_reach_deg": 40, "landing_compliance": 0.3},
            {"label": "Huge ballistic launch", "takeoff_vx": 6.5, "takeoff_vz": 5.2, "pitch_rate": 4.0,
             "tuck_time": 0.15, "extend_time": 0.90, "knee_tuck_deg": 100, "hip_tuck_deg": 100,
             "arm_reach_deg": 90, "landing_compliance": 0.4},
            {"label": "Forward dive roll", "takeoff_vx": 4.4, "takeoff_vz": 3.4, "pitch_rate": 6.0,
             "tuck_time": 0.20, "extend_time": 0.70, "knee_tuck_deg": 90, "hip_tuck_deg": 95,
             "arm_reach_deg": 120, "landing_compliance": 0.5},
            {"label": "Tucked vault", "takeoff_vx": 4.6, "takeoff_vz": 3.8, "pitch_rate": 0.5,
             "tuck_time": 0.12, "extend_time": 0.55, "knee_tuck_deg": 95, "hip_tuck_deg": 90,
             "arm_reach_deg": 70, "landing_compliance": 0.7},
        ]),
    },
    "stage_drop": {
        "id": "stage_drop",
        "name": "Stage High-Fall",
        "description": "Leap off an elevated stage, clear the step hazard, absorb the drop on the crash mat.",
        "start_pos": [-1.5, 0.0, 1.75],
        "camera": {"pos": [0.2, -5.0, 1.7], "lookat": [0.3, 0.0, 0.8], "fovy": 50},
        "scene_obstacles": [
            {"id": "stage_platform", "type": "box", "pos": [-1.6, 0.0, 0.40], "size": [0.80, 1.0, 0.40],
             "rgba": [0.35, 0.55, 0.95, 0.45], "is_hazard": False},
            {"id": "stage_step", "type": "box", "pos": [-0.40, 0.0, 0.22], "size": [0.40, 0.95, 0.22],
             "rgba": [0.95, 0.35, 0.35, 0.50], "is_hazard": True},
            {"id": "crash_mat", "type": "box", "pos": [1.60, 0.0, 0.12], "size": [0.90, 0.85, 0.12],
             "rgba": [0.06, 0.92, 0.50, 0.55], "is_hazard": False},
        ],
        "wave_1_candidates": _seeds([
            {"label": "Short hop", "takeoff_vx": 1.8, "takeoff_vz": 1.5, "pitch_rate": 0.0,
             "tuck_time": 0.30, "extend_time": 0.55, "knee_tuck_deg": 30, "hip_tuck_deg": 30,
             "arm_reach_deg": 40, "landing_compliance": 0.25},
            {"label": "Stiff-legged drop", "takeoff_vx": 3.6, "takeoff_vz": 2.6, "pitch_rate": 0.0,
             "tuck_time": 0.50, "extend_time": 0.60, "knee_tuck_deg": 10, "hip_tuck_deg": 10,
             "arm_reach_deg": 20, "landing_compliance": 0.05},
            {"label": "Front flip", "takeoff_vx": 4.0, "takeoff_vz": 3.6, "pitch_rate": 8.0,
             "tuck_time": 0.10, "extend_time": 0.80, "knee_tuck_deg": 110, "hip_tuck_deg": 110,
             "arm_reach_deg": 90, "landing_compliance": 0.5},
            {"label": "Parkour drop (undershot)", "takeoff_vx": 2.3, "takeoff_vz": 1.9, "pitch_rate": 0.5,
             "tuck_time": 0.25, "extend_time": 0.55, "knee_tuck_deg": 55, "hip_tuck_deg": 55,
             "arm_reach_deg": 60, "landing_compliance": 0.5},
        ]),
    },
    "rooftop_gap": {
        "id": "rooftop_gap",
        "name": "Gap Jump Over Rail",
        "description": "Clear a safety rail between two platforms and stick the landing on the pad.",
        "start_pos": [-1.8, 0.0, 1.20],
        "camera": {"pos": [0.2, -5.0, 1.6], "lookat": [0.3, 0.0, 0.8], "fovy": 50},
        "scene_obstacles": [
            {"id": "takeoff_deck", "type": "box", "pos": [-1.9, 0.0, 0.15], "size": [0.70, 0.9, 0.15],
             "rgba": [0.3, 0.5, 0.9, 0.45], "is_hazard": False},
            {"id": "gap_rail", "type": "box", "pos": [0.20, 0.0, 0.50], "size": [0.06, 0.95, 0.50],
             "rgba": [0.95, 0.3, 0.4, 0.55], "is_hazard": True},
            {"id": "crash_mat", "type": "box", "pos": [2.20, 0.0, 0.12], "size": [0.85, 0.85, 0.12],
             "rgba": [0.06, 0.92, 0.50, 0.55], "is_hazard": False},
        ],
        "wave_1_candidates": _seeds([
            {"label": "Under-speed", "takeoff_vx": 3.0, "takeoff_vz": 2.6, "pitch_rate": 0.0,
             "tuck_time": 0.30, "extend_time": 0.60, "knee_tuck_deg": 40, "hip_tuck_deg": 40,
             "arm_reach_deg": 40, "landing_compliance": 0.4},
            {"label": "Overshoot", "takeoff_vx": 6.6, "takeoff_vz": 5.0, "pitch_rate": 3.0,
             "tuck_time": 0.20, "extend_time": 0.90, "knee_tuck_deg": 90, "hip_tuck_deg": 95,
             "arm_reach_deg": 80, "landing_compliance": 0.35},
            {"label": "Mid-arc tuck", "takeoff_vx": 4.5, "takeoff_vz": 3.6, "pitch_rate": 2.5,
             "tuck_time": 0.30, "extend_time": 0.70, "knee_tuck_deg": 70, "hip_tuck_deg": 75,
             "arm_reach_deg": 65, "landing_compliance": 0.6},
            {"label": "Early tuck", "takeoff_vx": 4.8, "takeoff_vz": 4.0, "pitch_rate": 0.5,
             "tuck_time": 0.10, "extend_time": 0.60, "knee_tuck_deg": 90, "hip_tuck_deg": 90,
             "arm_reach_deg": 75, "landing_compliance": 0.8},
        ]),
    },
}
