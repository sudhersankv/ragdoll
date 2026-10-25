# FALLGUY: Autonomous Recursive Stunt & Biomechanical Safety Harness
**Triage Label:** `ready-for-agent`

---

## Problem Statement

Action film directors, stunt coordinators, and robotics engineers face a fundamental gap when planning physical stunts in real-world locations:
1. **Generative AI video tools hallucinate physics:** Existing video models produce visually dramatic clips, but actors float, clip through obstacles, and violate momentum and gravity—making them unusable as a physical safety or rigging plan.
2. **Traditional 3D stunt "techvis" is slow and manual:** Reconstructing a scouted location in 3D, rigging an articulated stunt double, and manually iterating on takeoff speeds, joint torques, and crash-pad placement takes days or weeks of specialized 3D engineering.
3. **Single-pass LLM physics coding fails:** Asking a model to blindly guess joint torques or velocities for a multi-body physics simulation in one shot almost always results in obstacle collisions or lethal cranial/cervical G-force impacts.

---

## Solution

**FALLGUY** is a web-based 3D stunt pre-visualization and biomechanical verification platform powered by **Gemini 4 Argon** and **DeepMind MuJoCo**.

A **Lead Stunt Director** selects a preset stage or uploads a phone video/photo of any real location. **Gemini 4 Argon** acts as an **Autonomous Assistant Stunt Director** operating inside a strict **3-Gate Verification Harness**:
1. **World Model Reconstruction & Visual Alignment:** Extracts 3D collision boxes and camera perspective from location footage, renders a 3D WebGL wireframe overlay over the footage, and performs a 1-step visual self-alignment pass (with manual 3D position sliders for fine tuning).
2. **Parallel Ghost Wave Simulation (MuJoCo 500Hz -> WebGL 60Hz):** Instead of a single blind guess, each iteration ("Wave") simulates **4 parallel candidate Ghost Stuntmen** using a 4-Phase Biomechanical Parameter Controller in MuJoCo, streaming 60Hz 3D bone transforms and instantaneous G-forces to an interactive Three.js WebGL viewport.
3. **3-Gate Recursive Self-Improvement Loop:**
   - **Gate 1 (Deterministic Biomechanical Physics):** Verifies cranial/cervical spine impact is $< 8.0G$ (SAE J211 15ms filter), obstacle clearance is $\ge 2.0\text{cm}$ with zero hazard collisions, and landing error from the crash pad center is $\le 0.65\text{m}$.
   - **Gate 2 (Argon Multimodal Visual Critic):** Evaluates WebGL canvas snapshots and telemetry to ensure camera framing is $\ge 85\%$ and choreography matches the Director's style prompt.
   - **Gate 3 (Human Stunt Director Live Steer & Sign-Off):** Pauses once Gates 1 & 2 pass (or after a cap of 4 auto-waves) so the Lead Director can orbit 360° in bullet-time, steer via natural language or 3D obstacle sliders, or lock the shot into a multi-scene stitched sequence.

---

## User Stories

1. As a **Stunt Director**, I want to load built-in 3D stunt presets (Venue Couch Speed-Vault, Stage Ledge High-Fall, Rooftop Gap Jump) immediately in my browser, so that I can test MuJoCo stunt simulations and WebGL playback without needing to upload footage first.
2. As a **Stunt Director**, I want to upload a scouted location video or photo from my phone, so that the system can reconstruct the real-world obstacles into a 3D MuJoCo collision scene.
3. As a **Stunt Director**, I want the Assistant Stunt Director (Gemini 4 Argon) to extract start, middle, and end keyframes from my uploaded location footage and perform a 1-step visual self-alignment check against the rendered 3D wireframe overlay, so that the virtual obstacles line up with the real environment.
4. As a **Stunt Director**, I want to manually nudge 3D obstacle and crash-mat coordinates with UI sliders, so that I can override or fine-tune the scene geometry at any time.
5. As a **Stunt Director**, I want each wave of the harness to simulate 4 candidate "Ghost Stuntmen" in parallel inside MuJoCo at 500Hz, so that I can visually compare multiple trajectory hypotheses simultaneously in the 3D WebGL viewport.
6. As a **Stunt Director**, I want to see glowing 3D parabolic trajectory ribbons for all 4 Ghost Stuntmen overlaid in the WebGL scene, so that I can immediately spot which path clears the obstacle and hits the crash pad.
7. As a **Stunt Director**, I want the 3D humanoid bones (cranium, cervical spine, lumbar spine, knees) to dynamically change color (Green / Amber / Critical Red) during WebGL playback based on instantaneous G-force load, so that I can visually identify exact moments of biomechanical injury risk.
8. As a **Stunt Director**, I want **Gate 1** of the harness to deterministically fail any trajectory where peak head/neck acceleration exceeds $8.0G$, any body part clips a hazard obstacle, clearance is $< 2.0\text{cm}$, or touchdown misses the crash mat by $> 0.65\text{m}$, so that unsafe stunts are never approved.
9. As a **Stunt Director**, I want **Gate 2** of the harness to capture a WebGL viewport snapshot along with the Gate 1 telemetry and send both to Gemini 4 Argon for multimodal critique, so that trajectories that are physically safe but poorly framed or awkward are rejected and refined.
10. As a **Stunt Director**, I want the Assistant Stunt Director to automatically mutate the 4-Phase Biomechanical Controller parameters from Wave $N$ to Wave $N+1$ (up to a maximum of 4 auto-waves), so that the Stunt Viability Score ($0\text{–}100\%$) measurably improves across waves without manual trial and error.
11. As a **Stunt Director**, I want to see a live Wave Convergence Chart showing the best Stunt Viability Score across Wave 1, Wave 2, Wave 3, and Wave 4, so that I can verify quantitative self-improvement at a glance.
12. As a **Stunt Director**, I want to inspect an Anatomical Injury Telemetry card showing separate safety ratings (`SAFE`, `WARN`, `CRITICAL`) for the Cranium, C3 Cervical Spine, Lumbar Spine, and Knees, so that I know why a candidate passed or failed Gate 1.
13. As a **Stunt Director**, I want to pause the WebGL simulation at any frame using a timeline scrubber and playback speed controls (`0.25x` Bullet-Time, `0.5x` Slow-Mo, `1.0x` Real-Time), so that I can inspect mid-air obstacle clearance in slow motion.
14. As a **Stunt Director**, I want to drag my mouse to orbit 360° around the 3D stunt in bullet-time and switch between `Camera Match View`, `Hero 3/4 View`, and `Top Tactical View`, so that I can verify spatial clearance from every angle.
15. As a **Stunt Director**, I want **Gate 3** to let me steer the Assistant Stunt Director using natural language prompts or quick-steer chips (e.g., *"Make clearance tighter"*, *"Higher dramatic flip arc"*, *"Shift crash mat back +0.35m"*), so that I can direct creative changes and watch the next wave adapt in seconds.
16. As a **Stunt Director**, I want to lock verified stunts into a 3-Shot Sequence Stitcher (`Shot 1`, `Shot 2`, `Shot 3`) and play the stitched master sequence (7.5–10 seconds total), so that I can choreograph a complete multi-beat action scene.
17. As a **Robotics / Sim-to-Real Engineer**, I want to export any verified stunt's 60Hz trajectory and control parameters as a structured JSON artifact, so that the solved trajectory can be used as a technical safety blueprint or downstream humanoid reference trajectory.

---

## Implementation Decisions

### 1. Hybrid Python MuJoCo Physics + Browser Three.js WebGL Architecture
- **Physics & Agent Backend (Python + FastAPI):** Runs native `mujoco` at `500Hz` (`timestep = 0.002s`) and orchestrates `google-genai` (`gemini-4-argon`) calls. It reads `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-4-argon` dynamically from `.env` on every request so credentials can be updated without restarting the server.
- **Interactive 3D Frontend (React + Three.js WebGL):** Receives static `humanoid_geoms_meta` (spheres, capsules, boxes) and 60Hz keyframe transforms (`xpos`, `xquat`, instantaneous `g_load`) from the backend and renders all 4 Ghost Stuntmen, glowing trajectory ribbons, and 3D wireframe obstacles over the uploaded location video/photo background plate.
- **Coordinate System Mapping:** MuJoCo uses Z-up coordinates (`[x, y, z]`), which the WebGL renderer deterministically maps to Three.js Y-up coordinates (`[x, z, -y]`).

### 2. The 4-Phase Biomechanical Parameter Controller
Instead of having the LLM output thousands of raw joint torques, the humanoid is driven by a 10-actuator PD position controller sequenced across 4 biomechanical phases (`Phase 1: Takeoff Extension`, `Phase 2: Launch Impulse`, `Phase 3: Mid-Air Vault/Tuck`, `Phase 4: Landing Absorption & Braking`). Each candidate ghost is parameterized by a compact 9-dimensional policy vector:

```json
{
  "ghost_id": "ghost_D",
  "label": "High Clearance Vault",
  "color": "#10b981",
  "takeoff_vx": 4.9,
  "takeoff_vz": 4.4,
  "pitch_impulse": 2.8,
  "tuck_time": 0.24,
  "knee_tuck_deg": 85.0,
  "hip_tuck_deg": 85.0,
  "arm_reach_deg": 75.0,
  "landing_compliance": 0.85,
  "mat_offset_x": 0.0
}
```

### 3. Quantitative Stunt Viability Score & 3-Gate State Machine
Every simulated ghost is scored from `0` to `100` using deterministic MuJoCo sensor and contact data:
- **Biomechanical G-Force (`35%` weight):** Computed from 3-axis accelerometer sensors on the head and torso filtered with a 15ms moving window (SAE J211 standard).
- **Obstacle Clearance (`25%` weight):** Minimum vertical distance between the lowest foot/body geom and any hazard obstacle bounding box during traversal.
- **Landing Accuracy (`20%` weight):** Euclidean distance in the horizontal plane between the pelvis touchdown coordinate and the center of the `crash_mat`.
- **Camera Framing (`20%` weight):** Percentage of sampled 60Hz frames in which the humanoid remains inside the active camera frustum bounds.

```text
Gate 1 Pass Condition:
  (peak_head_neck_g < 8.0) AND (hazard_collisions == 0)
  AND (min_obstacle_clearance_cm >= 2.0) AND (landing_error_m <= 0.65)

Gate 2 Pass Condition:
  Gate 1 Passed AND Argon Visual Critic confirms framing/style (stunt_score >= 85)

Auto-Loop Cap:
  Up to 4 Waves automatically; pauses at Gate 3 if passed early or if Wave 4 completes.
```

### 4. Stateless Wave-by-Wave REST API Contract
- `GET /api/presets`: Returns the 3 built-in stunt presets (`couch_vault`, `stage_drop`, `rooftop_gap`) with initial obstacle layouts and Wave 1 candidate policies.
- `POST /api/scene-from-video`: Accepts base64 location keyframes + stunt brief, calls `gemini-4-argon` to estimate 3D obstacle boxes and camera frustum, and supports a 1-step visual self-alignment refinement pass.
- `POST /api/simulate-wave`: Accepts `wave_index`, `scene_obstacles`, `start_pos`, and 4 `candidates`; runs all 4 in MuJoCo at 500Hz; returns `WaveTelemetryResult` containing 60Hz `frames`, `trajectory_points`, `metrics`, `joint_stress`, and `stunt_score`.
- `POST /api/argon-critic`: Accepts `wave_data`, `stunt_brief`, optional `director_steer`, and optional `canvas_image_b64` WebGL snapshot; calls `gemini-4-argon` (with a deterministic physics-gradient evolutionary fallback when offline or without an API key) to return `gate_2_passed`, `assistant_critique`, `key_mutations`, and `next_candidates` for Wave $N+1$.

---

## Testing Decisions

- **Single High-Level Testing Seam:** All automated verification runs against the **FastAPI HTTP boundary** (`TestClient(app)`) exercising `/api/presets`, `/api/simulate-wave`, `/api/argon-critic`, and `/api/scene-from-video`.
- **External Behavior Only:** Tests assert observable physics outcomes and contract invariants (e.g., that Wave 1 $\rightarrow$ Wave 2 $\rightarrow$ Wave 3 increases `best_score`, reduces `peak_head_neck_g` below `8.0G`, resolves `gate_1_passed == True`, and honors `director_steer` overrides) rather than internal private helper implementation details.

---

## Out of Scope

- Training reinforcement learning (PPO/SAC) policies from scratch over millions of timesteps (too slow for interactive live directing).
- External third-party video generation APIs (such as Higgsfield or Sora) that block the real-time 60 FPS WebGL loop.
- Full dense SLAM / NeRF reconstruction of arbitrary meshes (we use clean MuJoCo 3D bounding-box primitives aligned over the footage).

---

## Further Notes

- The system is designed so that **offline/mock mode** and **live `gemini-4-argon` mode** share the exact same REST response schema, ensuring the demo works reliably even under unstable hackathon venue Wi-Fi.
