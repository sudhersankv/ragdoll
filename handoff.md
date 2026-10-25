# 🤝 Agent Handoff Document (`handoff.md`) — FALLGUY (`ragdoll`)

> **Drop this file into Cursor (`@handoff.md`) to resume seamlessly with full project context.**

---

## 1. Project Context & Hackathon Goal
We are building **FALLGUY** for the **Gemini 4 Argon SF Hackathon** (Google DeepMind × Cerebral Valley, [`https://cerebralvalley.ai/e/gemini-4-argon-sf`](https://cerebralvalley.ai/e/gemini-4-argon-sf)).

**What FALLGUY Is:**
An **Autonomous 3-Gate Recursive Stunt & Biomechanical Safety Harness** powered by **Gemini 4 Argon (`gemini-4-argon`)**, **DeepMind MuJoCo (`3.17.0` at 500Hz)**, and an interactive **Three.js WebGL (60Hz)** frontend.
- Given a built-in stage or a phone video/photo of any real location (such as the hackathon venue), **Gemini 4 Argon** reconstructs the 3D collision boxes in MuJoCo and iteratively solves a human stunt across waves of **4 parallel Ghost Stuntmen**.
- Every wave is gated by a **3-Gate Verification Harness**:
  1. **Gate 1 (Deterministic MuJoCo Biomechanics):** Peak 15ms head acceleration $< 15.0g$, $\text{HIC}_{15} < 250$, zero hazard obstacle collisions, minimum signed clearance $\ge 2.0\text{cm}$ (`mj_geomDistance`), and touchdown on `crash_mat` within $\le 0.65\text{m}$ of center.
  2. **Gate 2 (Argon Multimodal Visual Critic):** Evaluates WebGL canvas snapshots + telemetry (`stunt_score >= 82%` and camera framing $\ge 85\%$).
  3. **Gate 3 (Human Stunt Director Live Steer & Sign-Off):** Natural-language steer overrides + live 3D obstacle/mat sliders + 3-Shot Sequence Stitcher (`7.5–10s` action cut).

---

## 2. Current Git & Repository State
- **GitHub Repository:** [`https://github.com/sudhersankv/ragdoll`](https://github.com/sudhersankv/ragdoll)
- **Active Branch:** `exp/mujoco-sim` (tracks `origin/exp/mujoco-sim`)
- **Open Pull Request (with Qodo Review):** [`https://github.com/sudhersankv/ragdoll/pull/1`](https://github.com/sudhersankv/ragdoll/pull/1)
- **Specification & Contracts (Do NOT duplicate; read these first):**
  - Formal Spec: [`docs/SPEC.md`](./docs/SPEC.md)
  - Shared JSON Contract (for parallel team experiments): [`CONTRACT.md`](./CONTRACT.md)
  - Tracer-Bullet Tickets (`01`–`05`): [`tickets/`](./tickets/)
- **Environment & Credentials:**
  - `.env` in repo root already contains the active `GEMINI_API_KEY` (redacted here; git-ignored) and `GEMINI_MODEL=gemini-4-argon`.
  - `backend/agent.py` reloads `.env` dynamically on every request.

---

## 3. Codebase Map & What Is Already Built (MVP Complete & Passing E2E Tests)

### Backend (`backend/` — Python 3.12 + FastAPI + MuJoCo 3.17.0)
- [`backend/params.py`](./backend/params.py): Defines the 10-parameter biomechanical stunt policy space (`takeoff_vx`, `takeoff_vz`, `pitch_rate`, `tuck_time`, `extend_time`, `knee_tuck_deg`, `hip_tuck_deg`, `arm_reach_deg`, `landing_compliance`, `mat_offset_x`) and `clamp_params()`.
- [`backend/presets.py`](./backend/presets.py): 3 built-in stunt stages (`couch_vault`, `stage_drop`, `rooftop_gap`) with deliberate Wave 1 failure modes (collision, overshoot, high-G impact).
- [`backend/physics.py`](./backend/physics.py):
  - Builds the 14-geom articulated humanoid + scene obstacle MJCF XML (`build_mjcf`).
  - Steps `mujoco.mj_step` at `500Hz` (`TIMESTEP = 0.002s`) driven by the 4-Phase Biomechanical PD Controller.
  - Computes true proper-acceleration head/spine/knee G-forces, $\text{HIC}_{15}$ (`hic15()`), exact geom-to-hazard signed clearance (`mujoco.mj_geomDistance`), touchdown surface/coordinates, camera frustum framing %, and `stunt_score` (`0–100`).
  - Exports 60Hz WebGL keyframes (`xpos`, `xquat`, instantaneous `g`) + 3D trajectory ribbons.
- [`backend/optimizer.py`](./backend/optimizer.py):
  - `refine()`: Fast Cross-Entropy Method (CEM) local optimizer in MuJoCo around each candidate seed (`iters=2, pop=8`).
  - `propose_offline()`: Physics-gradient heuristic proposer (ballistic touchdown re-aiming, compliance scaling, clearance tuning).
- [`backend/agent.py`](./backend/agent.py):
  - `critique_and_propose_next_wave()`: Hybrid **Gemini 4 Argon + CEM** recursive loop (`POST /api/argon-critic`).
  - `reconstruct_scene_from_media()`: Extracts 3D obstacle boxes & camera frustum from uploaded video/photo keyframes (`POST /api/scene-from-video`) and performs 1-step **Argon Visual Self-Alignment** over the rendered WebGL wireframe.
  - `parse_director_steer()`: Translates natural-language Director steers into parameter deltas.
- [`backend/video_renderer.py`](./backend/video_renderer.py): Renders MP4 stunt progression videos (`Wave 1 -> Wave 2`) and 4-panel PNG storyboards with HUD telemetry overlays (supports compositing over real background videos).
- [`backend/main.py`](./backend/main.py): FastAPI application exposing `/api/health`, `/api/presets`, `/api/simulate-wave`, `/api/argon-critic`, and `/api/scene-from-video`.

### Frontend (`frontend/` — React 18 + Vite + Three.js WebGL + Tailwind)
- [`frontend/src/WebGLViewport.jsx`](./frontend/src/WebGLViewport.jsx): Interactive Three.js WebGL viewport rendering translucent 3D obstacles + glowing wireframes, all 4 colored Ghost Stuntmen, 3D parabolic trajectory tubes, live cranial/spinal G-force bone highlighting (`Red >= 15g`, `Amber >= 9g`), 360° mouse orbit, and background video/image plates.
- [`frontend/src/App.jsx`](./frontend/src/App.jsx): Tactical Stunt Director Dashboard with preset switcher, scout video/photo uploader + keyframe extractor, **🎯 Argon 1-Step Self-Align** button, **🔄 Step Wave (+1)**, **⚡ Auto-Run 3-Gate Harness**, `0.25x` Bullet-Time scrubber, 3D obstacle/crash-mat position sliders, Gate 1/2/3 status cards, Wave Convergence Chart, Anatomical Injury Telemetry card, Live Director Steer bar, and 3-Shot Sequence Stitcher.

### End-to-End Tests (`tests/`)
- [`tests/test_api.py`](./tests/test_api.py): 4 end-to-end tests at the FastAPI `TestClient` boundary covering Tickets 01–04. All 4 pass (`58s` with live `gemini-4-argon` API + CEM rollouts).

---

## 4. How to Run Commands in This Workspace (Windows)

```powershell
# 1. Run the FastAPI + MuJoCo Backend (Port 8000)
d:\argon-hack\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Run the React + Three.js WebGL Frontend (Port 5173)
cd d:\argon-hack\frontend
npm run dev

# 3. Run the E2E FastAPI Boundary Test Suite
d:\argon-hack\.venv\Scripts\python.exe -m pytest d:\argon-hack\tests\test_api.py -v

# 4. Render MP4 Stunt Progression Videos & Storyboards
d:\argon-hack\.venv\Scripts\python.exe -c "from pathlib import Path; from backend.video_renderer import render_preset_progression_video; print(render_preset_progression_video('couch_vault', Path('renders/couch_vault.mp4'), Path('renders/couch_vault_storyboard.png')))"
```

---

## 5. Suggested Skills for the Next Agent Session
Local agent skills are installed in `.agents/skills/` (git-ignored so they stay local to this machine). Relevant skills to invoke or read as needed:
- `tdd` (`.agents/skills/tdd/SKILL.md`) — Red-Green-Refactor at the `tests/test_api.py` FastAPI HTTP seam.
- `code-review` (`.agents/skills/code-review/SKILL.md`) — Reviewing changes against `docs/SPEC.md` and Qodo PR feedback on PR #1.
- `improve-ui` / `baseline-ui` (`.agents/skills/improve-ui/SKILL.md`, `.agents/skills/baseline-ui/SKILL.md`) — Polishing the Stunt Director HUD and WebGL visual hierarchy.
- `diagnosing-bugs` (`.agents/skills/diagnosing-bugs/SKILL.md`) — Debugging physics or camera alignment edge cases.

---

## 6. Immediate Next Steps for the Cursor Session
1. **Test with User's Uploaded Location Video:**
   - The user is preparing to upload/share a real location video. Feed that video into `backend/video_renderer.py` (`bg_video_path=...`) and `/api/scene-from-video` to verify 3D obstacle alignment and render a composite MP4 over the real video.
2. **Review Qodo Feedback on PR #1:**
   - Check comments on [`https://github.com/sudhersankv/ragdoll/pull/1`](https://github.com/sudhersankv/ragdoll/pull/1) (`gh pr view 1 --comments`) and address any edge cases flagged by Qodo.
3. **Sync with Teammate's Parallel Simulation Track:**
   - Compare results against `CONTRACT.md` and merge the best visual/physics elements into `main`.
