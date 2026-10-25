# FALLGUY pipeline contract

Shared Python package `fallguy/` at the repo root, driven by two skills: `fallguy-world` (stage 1) and `fallguy-stunt` (stages 2–4). Python: `.venv/bin/python` (mujoco 3.17.0). Argon = `gemini-4-argon` via `fallguy/argon.py` (REST, key from `~/.argon_token`, thinkingLevel high, never printed). Performer: always the Unitree H1 (`models/mujoco_menagerie/unitree_h1`, composed by `fallguy/engine/performer.py`). Viewing: a local WebGL page in the director's browser. No stunt types: choreography is Argon-written generic phases (`hold, move, jump, fall, air, land`; see `CONTRACTS_OLD.md` § Choreography phases).

World frame: metres, floor z = 0, +Z up. The world builder picks the horizontal axes and records them in `world.json` (`frame` notes); camera poses are stored explicitly, so `world.json`, `scene.xml`, the stunt phases and the viewer all use those coordinates.

## Stage 1: world (`python -m fallguy.world ...`)

```bash
python -m fallguy.world build  --media PATH [--out worlds]        # reconstruct + Argon + verify (≤2 fix rounds); prints {"world_dir": ...}
python -m fallguy.world edit   --world DIR --prompt "add a crash mat at the end of the hallway"   # Argon edits, rebuild, re-render
python -m fallguy.world verify --world DIR                          # one Argon render-vs-video critique
```

`worlds/<slug>-<ts>/`:
```text
world.json        # source media, frames used, camera (intrinsics + per-frame poses if known, plus a reference pose),
                  # metric scale, objects [{id, label, role, hazard, material, collision: [{type box|cylinder|sphere, pos, size, quat}]}],
                  # floor, edits history, verify history [{round, score, issues, fixes}]
scene.xml         # MJCF: the replica's visual meshes (contype 0) + collision primitives + floor + lights + camera "ref"; NO performer
                  # (depth from the video is used only to measure layout and scale; real video pixels are NOT used as textures)
mesh/env.glb + mesh/*.obj   # game-style 3D replica: modelled assets per object (walls, curtains, railing, plant, sign, doors...) with stylised materials, same coordinates as scene.xml
frames/           # extracted video frames (+ ref.jpg)
views/            # renders of the world from the reference camera (and per-frame poses), side-by-side with the real frames
verify/round<N>.json   # Argon critique: fidelity score 0-100, per-object issues, corrections applied
```

## Stages 2–4: stunt (`python -m fallguy.stunt ...`)

```bash
python -m fallguy.stunt plan     --world DIR --brief TEXT --duration S [--out stunts]   # Argon writes the choreography (phases); prints {"stunt_dir", "questions": [...]}
python -m fallguy.stunt simulate --stunt DIR --take v1                                    # sim + biomechanics + render
python -m fallguy.stunt critique --stunt DIR --take v1                                    # Argon watches the video + reads biomechanics
python -m fallguy.stunt improve  --stunt DIR --from v1 --to v2                            # apply Argon's changes, simulate v2, critique v2
```

`stunts/<slug>-<ts>/`: `stunt.json` (brief, duration, world dir, phases, tunables, target, clarifications), `v1/`, `v2/` each with `params.json`, `scene.xml` (world + H1), `trajectory.npz`, `events.json`, `biomech.json`, `metrics.json`, `video.mp4` (server-side offscreen render from the world's reference camera, same scene the viewer shows), `critique.json`; `compare.json` (v1 vs v2 deltas).

## Biomechanics (`fallguy/biomech.py`) → `biomech.json`

Per 2 ms step (decimated to 60 Hz in the file) and peaks: head linear acceleration (G, 15 ms window) and HIC15; joint torques for every H1 joint (forward: `actuator_force`; kinematic phases: `mj_inverse`), with knee, hip, ankle torque peaks in N·m; contact/ground reaction forces (N and × body weight); internal loads from `cfrc_int` at torso/pelvis (neck and lumbar proxies, flagged as approximations on the H1); per-link stress index = load / literature limit, coloured green < 0.5, amber < 1, red ≥ 1. Limits (literature): HIC15 ≤ 700, peak head 80 g (severe) with the blueprint's 8 G shown as the conservative stunt target, neck compression 4.0 kN / tension 4.17 kN (Nij-style proxy), lumbar compression 3.4 kN (NIOSH), knee moment ≈ 200 N·m, impact GRF > 10 × BW flagged. Each value states whether it is measured (free physics) or inverse-dynamics (scripted phase).

## Viewer (`python -m fallguy.viewer --world DIR [--stunt DIR]`)

Local HTTP server + Three.js page: the reconstructed mesh (GLB), collision primitives (toggle), the reference camera view and free orbit; for stunts, the H1 (meshes) playing the trajectory with per-link stress colours, a G / torque timeline chart, Argon's critique, and v1/v2 side by side. Prints the URL and opens it.
