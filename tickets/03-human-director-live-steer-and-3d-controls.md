# 03: Gate 3 Human Stunt Director Live Steering & 3D Obstacle Controls

**What to build:** Enable interactive human-in-the-loop directing at Gate 3 (or between any wave): the Lead Stunt Director can steer the Assistant Stunt Director via natural-language prompts or quick-steer chips (*"Make clearance tighter"*, *"Higher dramatic flip arc"*, *"Shift crash mat back"*) AND directly adjust 3D obstacle/crash-mat positions via UI sliders, immediately re-triggering the recursive MuJoCo + Argon loop around the new constraints.

**Blocked by:** 02: 3-Gate Recursive Stunt Harness & Gemini 4 Argon Self-Improvement Loop.

**Status:** ready-for-agent

- [ ] The Director can enter a natural-language steer instruction or click quick-steer chips in the UI to trigger a steered Wave $N+1$ via `/api/argon-critic` and `/api/simulate-wave`.
- [ ] The Director can adjust 3D obstacle and crash-mat position sliders in the UI, updating the live 3D WebGL wireframe boxes and re-solving the stunt physics for the modified layout.
- [ ] Automated HTTP boundary test verifies that passing a `director_steer` instruction to `POST /api/argon-critic` produces mutated candidates reflecting the steer constraint.
