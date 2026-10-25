# 02: 3-Gate Recursive Stunt Harness & Gemini 4 Argon Self-Improvement Loop

**What to build:** Connect the autonomous recursive improvement loop (`POST /api/argon-critic`) so the Assistant Stunt Director (`gemini-4-argon` loaded dynamically from `.env`, with a deterministic physics-gradient evolutionary fallback) critiques Wave $N$'s MuJoCo telemetry and WebGL canvas snapshot (Gate 2), explains the biomechanical failure modes, and spawns 4 mutated candidate policies for Wave $N+1$ (up to a cap of 4 auto-waves) until Gates 1 & 2 pass, plotting the live Stunt Viability Score convergence curve in the UI.

**Blocked by:** 01: Tracer Bullet — 4-Ghost Parallel MuJoCo Stunt Wave & 3D WebGL Playback.

**Status:** ready-for-agent

- [ ] `.env` dynamically configures `GEMINI_API_KEY` and `GEMINI_MODEL=gemini-4-argon` on every request without requiring a server restart.
- [ ] `POST /api/argon-critic` accepts `wave_data`, `stunt_brief`, and a base64 WebGL canvas snapshot (`canvas_image_b64`), returning `gate_2_passed`, `assistant_critique`, `key_mutations`, and 4 mutated `next_candidates`.
- [ ] Clicking **Step Next Wave (+1)** or **Run Recursive Stunt Harness** runs up to 4 waves automatically, stopping early at Gate 3 once both Gate 1 (`< 8.0G`, zero hazard collisions, $\ge 2\text{cm}$ clearance, $\le 0.65\text{m}$ mat error) and Gate 2 (`score >= 85%`) pass.
- [ ] The UI displays the 3-Gate status badges (`Gate 1`, `Gate 2`, `Gate 3`), the Assistant Stunt Director's reasoning & parameter diff log, and an interactive Wave Convergence Chart allowing the user to click and replay any wave (`Wave #1` -> `Wave #2` -> `Wave #3`).
- [ ] Automated HTTP boundary test verifies that running Wave 1 -> Critic -> Wave 2 -> Critic -> Wave 3 measurably increases `best_score` and achieves `gate_1_passed == True`.
