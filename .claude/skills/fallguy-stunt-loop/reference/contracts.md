# FALLGUY stunt loop: file and data contracts

Every script in `scripts/` agrees on these shapes. Change a shape here first, then in code.

Python: a venv with `mujoco==3.17.0 numpy imageio imageio-ffmpeg pillow` (see `scripts/requirements.txt`). Encoding uses the ffmpeg bundled with imageio-ffmpeg; no system ffmpeg.

There are **no stunt types**. Argon writes each stunt's choreography from the director's brief as a sequence of generic phases (below), and the loop tunes the numbers Argon marks as tunable. The performer is always the Unitree H1 from MuJoCo Menagerie (`models/mujoco_menagerie/unitree_h1`), composed as in `barrel_roll.py` (`h1_xml` + `H1Rig`; dark grey body, orange left arm, blue right arm).

## World frame

Metres. Floor is `z = 0`. `+X` is the performer's initial facing direction at the start of the stunt, `+Y` is the performer's left when facing `+X`, `+Z` is up. Box `size` values are **half-extents** (MuJoCo convention). `pos` is the geom centre. When the director describes a direction from the camera's point of view ("turns left"), the scene prompt converts it into this frame using the camera pose and what the footage shows.

## Run directory

`loop.py init` creates one run directory; every path below is relative to it. Names are unique per run: `<slug>` is a lowercase, hyphenated, ≤40-char slug of the brief; `<ts>` is `YYYYMMDD_HHMMSS` local time; if the directory already exists, append `_2`, `_3`, ….

```text
stunts/<slug>-<ts>/
├── run.json                          # loop state (below); the source of truth for status
├── inputs/
│   ├── media.<ext>                   # image: downscaled JPEG ≤1600 px; video: H.264 mp4 ≤720p proxy (rotation applied)
│   └── ref_frame.jpg                 # still used for camera matching
├── plan.json                         # Argon scene + choreography plan (below)
├── stunt_<slug>_<ts>.py              # the generated stunt script (from scripts/stunt_template.py)
├── preview.png                       # scene preview still from the director's camera
├── waves/wave01/
│   ├── proposal.json                 # Argon proposal for this wave
│   ├── c1/ params.json metrics.json scene.xml trajectory.npz events.json [video.mp4]
│   ├── c2/ … c4/
│   ├── critique.json                 # Argon long-video critique of the wave's best render
│   └── summary.json                  # gate results for the wave (below)
├── final_<slug>_<ts>.mp4             # Gate-3 deliverable at the requested duration
├── certificate_<slug>_<ts>.json      # safety certificate of the final candidate
├── report.md                         # convergence table + critique highlights
└── logs/argon_calls.jsonl            # one line per Argon call: label, status, latency, usage (never the key)
```

## plan.json (written by `argon.py scene`)

```json
{
  "version": 2,
  "brief": "runs towards the camera and takes a left turn",
  "duration_sec": 6.0,
  "source": {"media": "inputs/media.mp4", "kind": "image|video", "ref_frame": "inputs/ref_frame.jpg", "ref_time_sec": 1.0,
             "width": 720, "height": 1280},
  "room": {"x_min": -16, "x_max": 6, "y_min": -8, "y_max": 2, "ceiling_z": 4.5,
           "floor_rgb": [0.55,0.53,0.5], "wall_rgb": [0.9,0.9,0.88], "ceiling_rgb": [0.3,0.3,0.3], "floor_material": "concrete"},
  "obstacles": [
    {"id": "glass_railing", "kind": "box|cylinder|sphere", "pos": [-6,1.55,0.55], "size": [9,0.03,0.55], "yaw_deg": 0,
     "rgb": [0.7,0.8,0.85], "material": "wood|metal|fabric|foam|glass|plastic|concrete|stone",
     "role": "wall|railing|furniture|prop|structure|person", "hazard": true}
  ],
  "set_dressing": {},
  "crash_mats": [],
  "camera": {"match": {"pos": [3.5,0.6,1.5], "lookat": [-6,0.3,1.0], "fovy": 64}},
  "stunt": {
    "summary": "runs down the hallway toward the camera, then turns to the camera's left into the open area and stops",
    "start": {"pos": [-14.0, 0.0], "heading_deg": 0},
    "phases": [
      {"kind": "hold", "name": "ready", "duration": 0.5, "pose": "stand"},
      {"kind": "move", "name": "run", "path": [[-14.0,0.0],[-1.0,0.0],[2.5,-4.0]], "speed": 5.0, "accel_time": 1.0,
       "decel_time": 1.0, "turn_radius": 2.0, "lean_scale": 1.0},
      {"kind": "hold", "name": "stop", "duration": 1.0, "pose": "stand"}
    ],
    "tunables": [
      {"name": "run_speed", "path": "phases[1].speed", "min": 2.0, "max": 7.0, "unit": "m/s", "doc": "cruise speed"},
      {"name": "turn_radius", "path": "phases[1].turn_radius", "min": 0.8, "max": 5.0, "unit": "m", "doc": "radius of the turn"}
    ],
    "target": {"kind": "zone", "pos": [2.5, -4.0], "radius": 0.65},
    "default_camera": "match_follow",
    "notes": "the camera pans left to the open area, so the turn goes toward -Y (the camera's left)"
  },
  "performer": {"model": "h1"}
}
```

- `obstacles[].role` is descriptive; anything in or near the performer's path stays in the scene, with `hazard: true`. Nothing is dropped.
- `set_dressing` (optional, empty when the media has none) keeps the prototype's schema (`curtain`, `couches`, `chairs`, `people`, `speakers`, `ceiling_fixtures`, `screens`) for scenery away from the path.
- `crash_mats` may be empty. `camera.match` is the director's camera for the reference frame; `fovy` is the vertical angle (portrait media has a tall frame).
- `source.width/height` are the media's pixels after rotation; renders keep its aspect (portrait → 720x1280, landscape → 1280x720).

## Choreography phases (`plan.stunt.phases`)

Phases run in order from `stunt.start` (standing, feet on the floor, facing `heading_deg`; add `"z"` to start on top of an obstacle). Each phase is one generic motion primitive:

| kind | physics | fields | meaning |
|---|---|---|---|
| `hold` | kinematic | `duration`, `pose` | stand or hold a pose in place (feet planted) |
| `move` | kinematic | `path` (≥2 xy points, the first is the current position), `speed`, `accel_time`, `decel_time` (0 = arrive at speed), `turn_radius` (corner rounding), `lean_scale` (× physical bank `atan(v²/rg)`), optional `gait` `walk|run|sprint` (auto from speed) | locomotion along the path with planted feet (leg IK) |
| `jump` | release | `vz` (m/s up), optional `vx` (m/s along heading; default keeps the move speed), `lean_deg`, `rotations_deg` `{roll, pitch, yaw}` completed over the predicted flight, `pose` (take-off pose) | leave the ground: the kinematic weld is released and free MuJoCo dynamics take over |
| `fall` | release | optional `push` `[vx, vy, vz]`, `pose` | leave support without a jump (step off a ledge, topple) |
| `air` | free | `poses`: list of `{"at": 0..1 (share of the flight), "pose": name or pose dict, "tuck": 0..1}` | pose schedule while airborne (PD servos) |
| `land` | free | `pose`, `compliance` 0..1 (scales leg servo gains), `duration` (s after first contact) | absorb the landing and settle |

- Poses are body poses, not stunts: `stand, crouch, takeoff, dive, tuck, pike, layout, land_prep, brace, arms_up`, or an explicit dict in the H1Rig vocabulary (`l_shoulder_x, l_shoulder_y, l_elbow, r_shoulder_x, r_shoulder_y, r_elbow, l_hip_y, l_knee, r_hip_y, r_knee` in degrees, optional `torso`).
- After a `jump` or `fall`, the performer stays free (physics) until the end; `move`/`hold` only follow `hold`/`move` or the start.
- `tunables` name the numbers the loop may change: `path` addresses a field in `stunt` (`phases[1].speed`, `phases[1].path[2][1]`, `target.pos[0]`, …), with `min`, `max`, `unit`, `doc`. Argon writes them with the plan; the generated script and the director may add more.
- `target`: `{"kind": "mat", "id": "<crash mat id>"}` (land on that mat), `{"kind": "zone", "pos": [x, y], "radius": r}` (end there), or `{"kind": "none"}`.
- `default_camera`: `match_follow` (the director's camera position, panning to follow the performer), `match` (fixed director's view), `hero` (3/4 tracking), or `side`.

## Candidate params (`params.json`)

```json
{"id": "w1c1", "label": "fast tight turn", "color": "#ef4444", "camera": "match_follow", "why": "...",
 "run_speed": 5.4, "turn_radius": 1.6, "mat_offset_x": 0.0, "mat_offset_y": 0.0}
```

Meta keys (`id, label, color, camera, why`) plus one value per `plan.stunt.tunables[].name`, plus the generic `mat_offset_x/y` (m, −1.5…1.5, shifts the target mat). `engine.params` is the single source of truth: `specs(plan)` returns name → `{min, max, unit, default, doc}` from the plan's tunables plus the generic ones; `clamp(params, plan)` clamps and fills defaults (the plan's current values); `apply(plan, params)` returns the stunt with the values written into their paths; `describe_for_prompt(plan)` is the compact table Argon sees.

## Generated stunt script CLI (`stunt_<slug>_<ts>.py`)

```bash
python stunt_<slug>_<ts>.py --plan plan.json --params waves/wave01/c1/params.json --out-dir waves/wave01/c1 \
    [--render] [--duration 6] [--fps 30] [--size 720x1280] [--camera match_follow|match|hero|side] [--hud] \
    [--preview-still preview.png] [--check]
```

- Always simulates and writes `out-dir/{scene.xml, trajectory.npz, events.json, metrics.json}`.
- `--render` also writes `out-dir/video.mp4` at exactly `--duration` seconds. Default `--size` follows `plan.source` aspect; default `--camera` is `params.camera`, else `plan.stunt.default_camera`.
- `--preview-still PATH` renders one frame of the scene at the start pose from the director's camera and exits.
- `--check` builds and steps the model for 0.5 s, prints `OK`, exits 0.
- The script holds one hook, `choreography(plan, params)`, near the top. By default it returns the plan's phases with the params applied; edit it to change or add phases, or to return custom motion code for something the phase kinds can't express.
- Exit code 0 on success. On failure it still writes `metrics.json` with `"ok": false, "error": "..."` and exits 1.

## Simulation outputs (`engine.sim`)

- `scene.xml`: the exact MJCF simulated, with an absolute `meshdir`, so `engine.render` can rebuild the model.
- `trajectory.npz`: `t` (N,) sim seconds at every physics step (2 ms), `qpos` (N, nq) float64, `head_g15` (N,) head G with a 15 ms moving average.
- `events.json`: `{"t_start": 0.0, "t_end": 6.4, "phases": [{"name", "kind", "t0", "t1"}], "t_takeoff": 3.1 or null, "t_apex": … or null, "t_landing": … or null, "t_finish": 5.2, "performer_root_body": "pelvis", "performer_geoms": [...], "hazard_geoms": [...], "mat_geoms": [...], "floor_geoms": [...]}`. `t_takeoff/t_landing` describe the first airborne segment (null when the stunt never leaves the ground); `t_finish` is when the last `move` ends or the performer settles.

Physics: `hold` and `move` are kinematic (the pelvis follows the plan through a mocap weld, feet planted by leg IK, performer collisions off; clearance and penetration are measured geometrically). At `jump`/`fall` the weld is released, performer collisions turn on and the root gets the launch velocity; from then on everything is free MuJoCo dynamics with PD limb servos. Timestep 2 ms. The sim runs ≥2 s past the last event (≥3 s past a landing).

## metrics.json (Gate 1, written by `engine.metrics`)

```json
{
  "candidate_id": "w1c1", "ok": true, "error": null,
  "sim": {"t_takeoff": null, "t_landing": null, "t_finish": 5.2, "airborne": false, "apex_m": null, "peak_speed": 5.1},
  "gate1": {
    "head_g_peak": 2.6, "head_g_peak_t": 1.1, "hic15": 4.0, "lateral_g_peak": 0.71,
    "hazard_collisions": 0,
    "hazard_contacts": [{"t": 3.6, "body": "left_ankle_link", "object": "glass_railing", "depth_cm": 1.2}],
    "min_clearance_cm": 18.0, "min_clearance_at": {"t": 4.4, "body": "right_elbow_link", "object": "curtain_wall"},
    "landing": {"t": 5.2, "surface": "floor|crash_mat|obstacle|none", "point": [2.4, -3.8], "target": [2.5, -4.0], "error_m": 0.22, "inside_mat": true},
    "framing_pct": 97.0,
    "rotation_deg": {"roll": 0, "pitch": 0, "yaw": -88},
    "passed": true, "failures": []
  },
  "score": {"safety": 95.0, "clearance": 100.0, "landing": 78.0, "framing": 97.0, "stunt_score": 92.4}
}
```

- Head G: `|a| / 9.81` of the head site, 15 ms moving average, over the whole stunt (finite differences in kinematic phases, the same measure in free phases). HIC15 from the same signal.
- `lateral_g_peak`: peak horizontal acceleration perpendicular to the direction of travel during `move` phases, in g; must stay ≤ `gates.THRESHOLDS["lateral_g_max"]` (0.8: beyond floor grip the runner would slip).
- Hazard collision: a performer–hazard penetration > 0.5 cm (contact in free phases, geometry in kinematic ones); `hazard_contacts` lists each distinct (body, object) pair with its first time.
- Clearance: min `mj_geomDistance` between performer collision geoms and hazard geoms over the stunt; negative when penetrating.
- `landing` reports where the stunt ends against `stunt.target`: for a `mat` target, the first support contact after take-off; for a `zone`, the pelvis at `t_finish`. `inside_mat` means inside the target (on the mat, or within the zone radius). For `target.kind: none` it reports the end point with `error_m: 0`.
- Framing: % of frames from the start of motion to `t_finish + 0.5 s` in which the projected performer bounding box is inside the chosen camera's frame.
- `gate1.passed` and `score.*` are computed by `scripts/gates.py` (single source of thresholds and weights).

## Renders (`engine.render`)

Real time throughout, except slow motion inside each airborne segment (`t_takeoff − 0.25 s` to `t_landing + 0.35 s`, factor chosen to fit the duration, clamped to 0.25–1.0). The clip covers the start of motion − 0.5 s to `t_finish + 0.5 s` (or the settle after a landing), fitted to the requested duration by padding with stillness or trimming the start. `match_follow` stays at `camera.match.pos` with its fovy and pans/tilts smoothly to keep the performer centred.

## Argon CLI (`scripts/argon.py`)

```bash
python argon.py ping
python argon.py scene    --media inputs/media.mp4 --brief "…" --duration 6 --out plan.json [--ref-frame inputs/ref_frame.jpg]
python argon.py propose  --plan plan.json --history history.json --out waves/wave02/proposal.json [--steer "…"] [--n 4]
python argon.py critique --video waves/wave02/c3/video.mp4 --plan plan.json --params …/params.json --metrics …/metrics.json --out waves/wave02/critique.json [--fps 5]
```

Common to all: key from `~/.argon_token` (override path with `FALLGUY_ARGON_TOKEN_FILE`), never printed, redacted from errors. Model `gemini-4-argon` (`ARGON_MODEL`), `thinkingLevel` `high` (`ARGON_THINKING`). REST over stdlib `urllib`. Strict JSON through `responseJsonSchema`. Each call is appended to `<run>/logs/argon_calls.jsonl` (`--log` overrides). `FALLGUY_ARGON_MOCK=1` returns deterministic canned outputs for tests. Exit 0 on success; non-zero with a one-line redacted error on failure.

- `scene` describes only what the director's media shows (any location), and writes the whole plan including `stunt` (summary, start, phases, tunables, target, default camera, notes) from the brief.
- `propose` returns n candidates over `engine.params.specs(plan)`.
- `critique` watches the render (long-video summarization) and judges it against the brief and the plan's choreography summary, with the MuJoCo metrics as ground truth for clearance, contact and G.
- Director notes from `history.json` appear in both `propose` and `critique` prompts.

`history.json` (built by `loop.py`): `{"brief": "...", "director_notes": [...], "waves": [{"wave": 1, "candidates": [{"params": {...}, "metrics": {...}}], "best_id": "w1c2", "critique": {...}, "steer": null}]}`.

`proposal.json`: `{"rationale": "...", "candidates": [{...params..., "why": "..."} × n]}`. The params are already clamped by `engine.params.clamp`.

`critique.json` (Gate 2):

```json
{
  "summary": "...",
  "events": [{"t_sec": 3.1, "event": "start|takeoff|apex|landing|turn|stop|settled|other", "note": "..."}],
  "brief_match": {"matches": true, "missing": []},
  "unnatural_motion": [{"t_sec": 3.4, "issue": "...", "severity": "low|medium|high"}],
  "framing": {"in_frame_pct": 95, "subject_size": "small|medium|large", "notes": "..."},
  "style_score": 78,
  "gate2_pass": true,
  "proposed_changes": [{"param": "run_speed", "change": "increase|decrease|set", "value": 5.6, "reason": "..."}]
}
```

`proposed_changes[].param` must be a name from `engine.params.specs(plan)`.

## Loop CLI (`scripts/loop.py`)

```bash
python loop.py init     --media PATH --brief TEXT --duration S [--runs-dir stunts]
python loop.py wave     --run-dir DIR [--steer TEXT] [--candidates 4]
python loop.py finalize --run-dir DIR [--candidate ID]
python loop.py status   --run-dir DIR
python loop.py gates
```

- `init` prints one JSON line `{"run_dir": ..., "script": ..., "plan": ..., "preview": ...}`.
- `wave` runs exactly one wave and prints one JSON line (the `summary.json` content): `{"wave": 2, "best_id": "w2c3", "gate1_passed": true, "gate2_passed": false, "stunt_score": 81.2, "failures": [...], "critique_summary": "...", "next": "wave|checkpoint|finalize"}`.
  - `next` is `finalize` when Gate 2 passed; `checkpoint` after every `checkpoint_every`-th wave (5, 10, 15, …) that did not pass; otherwise `wave`. There is no hard wave cap: at a checkpoint the director decides whether to continue and with what feedback.
  - `--steer TEXT` is director feedback. It is appended to `run.json` `director_notes` (persistent: every later `propose` and `critique` prompt includes all notes, newest first) and recorded on that wave's summary as `steer`.
- `finalize` renders `final_<slug>_<ts>.mp4` (requested duration, 60 fps, HUD) from the best candidate, writes the certificate and `report.md`, prints `{"final_video": ..., "report": ..., "certificate": ...}`.
- Wave renders and `finalize` pass `--size` from `plan.source` aspect and leave `--camera` to the candidate.

`run.json`: `{"brief", "duration_sec", "media", "script", "plan", "status": "planned|running|checkpoint|passed|finalized", "waves": [summary...], "best": {"wave", "id", "stunt_score"}, "checkpoint_every": 5, "director_notes": [{"after_wave": 5, "text": "..."}]}`.
