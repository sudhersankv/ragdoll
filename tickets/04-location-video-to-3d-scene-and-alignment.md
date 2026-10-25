# 04: Scouted Location Video/Photo Upload, 3D Extraction & Visual Self-Alignment

**What to build:** Allow the Stunt Director to upload a real-world location video or photo from their phone, display it as the background plate of the 3D WebGL viewport, extract keyframes (`start`, `mid`, `end`), and call `POST /api/scene-from-video` so Gemini 4 Argon reconstructs the 3D obstacle bounding boxes and camera frustum—followed by a 1-step visual self-alignment check comparing the rendered WebGL wireframe against Frame 0.

**Blocked by:** 01: Tracer Bullet — 4-Ghost Parallel MuJoCo Stunt Wave & 3D WebGL Playback.

**Status:** ready-for-agent

- [ ] Uploading a video or image in the UI renders the media as a synchronized background plate behind the translucent 3D WebGL obstacles and Ghost Stuntmen.
- [ ] `POST /api/scene-from-video` accepts base64 location frame(s) + stunt brief and uses `gemini-4-argon` to output a calibrated 3D `scene_obstacles` list and `camera` frustum (`pos`, `lookat`, `fovy`).
- [ ] Supports a 1-click **Argon Visual Self-Alignment** pass that captures the composite WebGL wireframe over Frame 0 and refines obstacle positions/camera pitch so the 3D boxes lock onto the real furniture.
- [ ] Automated HTTP boundary test verifies `POST /api/scene-from-video` returns valid 3D obstacle definitions that can be simulated directly by `POST /api/simulate-wave`.
