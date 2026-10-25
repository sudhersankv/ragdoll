# 01: Tracer Bullet — 4-Ghost Parallel MuJoCo Stunt Wave & 3D WebGL Playback

**What to build:** The end-to-end foundation slice: load a built-in stunt preset (`Venue Couch Speed-Vault`, `Stage Ledge High-Fall`, or `Precision Gap Jump`), step 4 parallel candidate Ghost Stuntmen through the 4-Phase Biomechanical Parameter Controller in MuJoCo at 500Hz via `POST /api/simulate-wave`, compute deterministic Gate 1 telemetry (SAE J211 15ms head/neck G-forces, obstacle clearance cm, crash-mat touchdown error m, Stunt Viability Score 0–100%), and render all 4 animated 3D Ghost Stuntmen, glowing trajectory ribbons, and bone G-force stress colors at 60Hz inside the interactive Three.js WebGL viewport with bullet-time scrubbing and 360° camera orbit.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `GET /api/presets` returns the 3 built-in stunt presets (`couch_vault`, `stage_drop`, `rooftop_gap`) with obstacle boxes, camera frustum, and Wave 1 candidate policies.
- [ ] `POST /api/simulate-wave` simulates 4 candidate policies in MuJoCo (500Hz) and returns 60Hz geom transforms (`xpos`, `xquat`), instantaneous G-loads, trajectory ribbons, anatomical `joint_stress` (`head`, `neck`, `spine`, `knees`), `gate_1_passed`, and `stunt_score`.
- [ ] The Three.js WebGL viewport renders the 3D obstacles, all 4 colored Ghost Stuntmen, and their 3D parabolic trajectory ribbons, flashing cranial/spinal bones red when instantaneous G-force exceeds `8.0G`.
- [ ] The Director can scrub any frame in bullet-time (`0.25x`, `0.5x`, `1.0x`), toggle all 4 ghosts vs. winning ghost, and orbit 360° around the 3D scene.
- [ ] Automated HTTP boundary test verifies `POST /api/simulate-wave` returns 4 scored ghosts with valid 60Hz frame arrays and Gate 1 telemetry.
