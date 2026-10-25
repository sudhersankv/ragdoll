# 05: Multi-Scene Sequence Stitcher (8–10s Action Cut) & Trajectory Export

**What to build:** Complete the end-of-loop deliverables: allow the Stunt Director to lock verified stunt beats into a 3-Shot Sequence Timeline (`Shot 1`, `Shot 2`, `Shot 3`), play back the stitched 7.5–10 second multi-beat action sequence continuously in the 3D WebGL viewport, and export the verified 60Hz trajectory + biomechanical safety certificate as a JSON artifact.

**Blocked by:** 02: 3-Gate Recursive Stunt Harness & Gemini 4 Argon Self-Improvement Loop.

**Status:** ready-for-agent

- [ ] The bottom timeline provides 3 lockable shot slots (`Shot #1`, `Shot #2`, `Shot #3`), storing each locked shot's scene obstacles, winning Ghost trajectory, wave index, and safety metrics.
- [ ] Clicking **Play Stitched Sequence** plays all locked shots back-to-back in the 3D WebGL viewport as a continuous multi-beat action sequence.
- [ ] Clicking **Export Trajectory JSON** downloads a structured JSON package containing the biomechanical safety certificate (`peak_head_neck_g`, `min_obstacle_clearance_cm`, `landing_error_m`), control policy parameters, and 60Hz 3D trajectory frames.
