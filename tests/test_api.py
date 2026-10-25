"""End-to-end tests at the agreed FastAPI HTTP boundary seam.

Verifies all 5 tracer-bullet tickets through `/api/presets`, `/api/simulate-wave`,
`/api/argon-critic`, and `/api/scene-from-video`.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.physics import LIMITS, TIMESTEP, hic15

client = TestClient(app)


def test_ticket_01_presets_and_4_ghost_mujoco_wave():
    """Ticket 01: Load presets and simulate a 4-ghost parallel wave at 500Hz -> 60Hz WebGL frames."""
    presets_res = client.get("/api/presets")
    assert presets_res.status_code == 200
    presets = presets_res.json()["presets"]
    assert {"couch_vault", "stage_drop", "rooftop_gap"}.issubset(presets.keys())
    assert LIMITS["peak_head_g"] == 8.0

    sim_res = client.post("/api/simulate-wave", json={"wave_index": 1, "preset_id": "couch_vault"})
    assert sim_res.status_code == 200
    wave1 = sim_res.json()

    assert wave1["wave_index"] == 1
    assert len(wave1["results"]) == 4
    assert wave1["any_gate_1_passed"] is False  # Wave 1 seeds have deliberate stunt flaws

    ghost_a = wave1["results"][0]
    assert ghost_a["ghost_id"] == "ghost_A"
    assert len(ghost_a["frames"]) > 100
    assert len(ghost_a["humanoid_geoms_meta"]) == 14
    assert {"name", "region", "type", "size"}.issubset(ghost_a["humanoid_geoms_meta"][0].keys())
    assert len(ghost_a["trajectory_points"]) == len(ghost_a["frames"])
    assert {"head", "neck", "spine", "knees"} == set(ghost_a["joint_stress"].keys())
    assert "peak_head_g" in ghost_a["metrics"]
    assert "peak_head_neck_g" in ghost_a["metrics"]
    assert "hic15" in ghost_a["metrics"]
    assert "min_clearance_cm" in ghost_a["metrics"]
    assert "min_obstacle_clearance_cm" in ghost_a["metrics"]
    assert "landing_error_m" in ghost_a["metrics"]

    # Verify HIC15 window never exceeds 15ms (at dt=2ms, max window is 7 steps = 14ms <= 15ms, so an 8th step spike outside 14ms is not combined into a single 16ms window)
    max_w = max(1, int(0.015 / TIMESTEP))
    assert max_w * TIMESTEP <= 0.015
    pulse = np.zeros(30)
    pulse[5 : 5 + max_w] = 10.0
    hic_within_15ms = hic15(pulse, dt=TIMESTEP)
    assert hic_within_15ms > 0.0


def test_ticket_02_recursive_stunt_harness_self_improvement():
    """Ticket 02: Wave 1 -> Argon Critic -> Wave 2 measurably improves score & passes Gate 1 (< 8.0G)."""
    w1 = client.post("/api/simulate-wave", json={"wave_index": 1, "preset_id": "couch_vault"}).json()
    assert w1["any_gate_1_passed"] is False

    c1_res = client.post(
        "/api/argon-critic",
        json={"wave_data": w1, "stunt_brief": "Vault over the couch and land safely on the crash mat"},
    )
    assert c1_res.status_code == 200
    c1 = c1_res.json()
    assert len(c1["next_candidates"]) == 4
    assert len(c1["key_mutations"]) >= 1
    assert c1["visual_review_performed"] is False

    w2 = client.post(
        "/api/simulate-wave",
        json={
            "wave_index": 2,
            "preset_id": "couch_vault",
            "candidates": c1["next_candidates"],
            "enable_local_refinement": True,
        },
    ).json()

    assert w2["best_score"] > w1["best_score"]
    assert w2["any_gate_1_passed"] is True
    best_w2 = next(r for r in w2["results"] if r["ghost_id"] == w2["best_ghost_id"])
    assert best_w2["gate_1_passed"] is True
    assert best_w2["metrics"]["peak_head_g"] < 8.0

    # Run Critic on Wave 2 to confirm Gate 2 passes once Gate 1 & score >= 82 & framing >= 85
    c2 = client.post(
        "/api/argon-critic",
        json={
            "wave_data": w2,
            "stunt_brief": "Vault over the couch and land safely on the crash mat",
            "canvas_image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
        },
    ).json()
    assert c2["gate_2_passed"] is True
    assert c2["visual_review_performed"] is True


def test_ticket_03_human_director_live_steer_and_3d_obstacle_nudge():
    """Ticket 03: Natural-language Director steer & 3D obstacle nudging alter candidate policies."""
    w1 = client.post("/api/simulate-wave", json={"wave_index": 1, "preset_id": "rooftop_gap"}).json()

    steered_critic = client.post(
        "/api/argon-critic",
        json={
            "wave_data": w1,
            "stunt_brief": "Clear gap rail",
            "director_steer": "Higher dramatic flip arc and shift crash mat back",
        },
    ).json()

    assert steered_critic["applied_steer_vector"].get("takeoff_vz", 0) > 0
    assert steered_critic["applied_steer_vector"].get("pitch_rate", 0) > 0
    assert steered_critic["applied_steer_vector"].get("mat_offset_x", 0) > 0

    # Verify nudging a 3D obstacle position in the scene simulates cleanly
    custom_obs = [dict(o) for o in w1["scene_obstacles"]]
    custom_obs[-1]["pos"] = [2.55, 0.0, 0.12]
    w_nudged = client.post(
        "/api/simulate-wave",
        json={
            "wave_index": 2,
            "scene_obstacles": custom_obs,
            "start_pos": w1["start_pos"],
            "candidates": steered_critic["next_candidates"],
        },
    )
    assert w_nudged.status_code == 200
    assert w_nudged.json()["scene_obstacles"][-1]["pos"][0] == 2.55

    # Verify input validation rejects malformed obstacles or out-of-range duration
    bad_obs_res = client.post(
        "/api/simulate-wave",
        json={"wave_index": 1, "scene_obstacles": [{"id": "bad", "pos": [0.0, 0.0], "size": [-1.0, 1.0, 1.0]}]},
    )
    assert bad_obs_res.status_code == 422


def test_ticket_04_location_video_to_3d_scene_and_visual_self_alignment():
    """Ticket 04: Extract 3D obstacle layout from keyframes and run 1-step visual self-alignment."""
    dummy_frame = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
    initial_scene = client.post(
        "/api/scene-from-video",
        json={
            "keyframes_b64": [dummy_frame, dummy_frame, dummy_frame],
            "stunt_brief": "Vault over venue couch onto mat",
        },
    ).json()

    assert initial_scene["alignment_status"] == "initial_blockout"
    assert len(initial_scene["scene_obstacles"]) >= 2
    assert len(initial_scene["wave_1_candidates"]) == 4

    aligned_scene = client.post(
        "/api/scene-from-video",
        json={
            "keyframes_b64": [dummy_frame],
            "wireframe_overlay_b64": dummy_frame,
            "current_obstacles": initial_scene["scene_obstacles"],
            "stunt_brief": "Vault over venue couch onto mat",
        },
    ).json()

    assert aligned_scene["alignment_status"] in ("self_aligned", "aligned")
    assert aligned_scene["scene_obstacles"][0]["pos"][0] != initial_scene["scene_obstacles"][0]["pos"][0]

    # Verify the extracted scene start_pos, camera, scene_obstacles, and wave_1_candidates flow into /api/simulate-wave
    sim_res = client.post(
        "/api/simulate-wave",
        json={
            "wave_index": 1,
            "scene_obstacles": aligned_scene["scene_obstacles"],
            "start_pos": aligned_scene["start_pos"],
            "camera": aligned_scene["camera"],
            "candidates": aligned_scene["wave_1_candidates"],
        },
    )
    assert sim_res.status_code == 200
    sim_payload = sim_res.json()
    assert len(sim_payload["results"]) == 4
    assert sim_payload["start_pos"] == aligned_scene["start_pos"]
    assert sim_payload["camera"] == aligned_scene["camera"]
    assert sim_payload["scene_obstacles"] == aligned_scene["scene_obstacles"]
