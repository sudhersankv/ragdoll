---
name: fallguy-stunt-loop
description: FALLGUY recursive stunt loop. Turns a site photo or video, a stunt brief and a target length into a MuJoCo-simulated stunt video that Gemini 4 Argon critiques, iterating waves until the safety and critique gates pass. Use when the user wants to simulate, previs or safety-test a stunt at a real location.
---

# FALLGUY stunt loop

The 3-gate recursive stunt harness from `fallguy_blueprint.md`, run from the terminal:
- **Argon** (`gemini-4-argon`) builds the scene from the director's own site media and watches every rendered take.
- **MuJoCo** judges safety.
- **The performer** is always the Unitree H1 from MuJoCo Menagerie, built exactly as in `barrel_roll.py`: its `h1_xml` composition and `H1Rig` pose mapping, a dark grey body, an orange left arm and a blue right arm. Call it the H1 when you talk to the director.
- **You** run the waves and report to the director.

`$SKILL` below is this skill's base directory (shown when the skill loads). Run every command from the repo root; runs land in `./stunts/`. File shapes and the run-directory layout are in [reference/contracts.md](reference/contracts.md). Read it before you edit a generated script or a `plan.json`.

The Argon key is in `~/.argon_token`, and only the scripts read it. Keep it out of every command line, message and file you write.

Every question to the director is plain text with no suggested answers. Ask, then wait for their reply.

## 1. Set up

`bash $SKILL/scripts/setup_env.sh` prints `FALLGUY_PYTHON=…` and `FALLGUY_H1_DIR=…`. Export both, and run every script below with `$FALLGUY_PYTHON`.

Done when both lines are printed and the script exits 0. If it fails, show the director its message and stop.

## 2. Ask the director for three inputs

Ask in one message, in this order:
1. The path to the site video or image.
2. The stunt: what happens in the scene.
3. The length of the final video, in seconds.

Skip any question `$ARGUMENTS` already answers.

Done when the media path exists and is an image or a video, the stunt is a concrete action, and the length is a number from 3 to 30. Ask again for anything that fails.

## 3. Plan the scene with Argon

```bash
$FALLGUY_PYTHON $SKILL/scripts/loop.py init --media "<path>" --brief "<stunt>" --duration <seconds>
```

This takes 1–2 minutes and prints `run_dir`, `script`, `plan` and `preview`; keep `run_dir`.
- Argon watches the director's media and writes `plan.json`: room, obstacles, crash mat, camera and stunt layout. Everything in the scene comes from that media.
- The loop generates the stunt script `stunt_<slug>_<timestamp>.py`, unique to this run, and renders `preview.png`.

Open `preview.png` with Read and compare it with the media. Check that the stunt area is there at a believable size, the camera sees the lane, and a crash mat sits where the performer lands (when the stunt has a landing). If the plan is off, fix it either way:
- edit `plan.json` and re-render with the script's `--preview-still preview.png`
- re-run `init` (Argon's plans vary from run to run)

Done when the preview matches the media. Tell the director the run dir and where the preview is.

## 4. Check the choreography

Argon writes the stunt itself: `plan.json` → `stunt` holds its summary, the ordered phases (`hold`, `move`, `jump`, `fall`, `air`, `land`) and the tunables the loop will adjust. Check that the phases perform the brief in order and that the tunables cover the numbers that decide success, such as speeds, heights, rotations, turn radius and the target. Fix gaps in `plan.json`. For motion the phase kinds can't express, edit the `choreography()` hook near the top of the generated script. The phase kinds and poses are listed in contracts.md. Then run:

```bash
$FALLGUY_PYTHON <run_dir>/<script> --plan <run_dir>/plan.json --params <any params.json> --out-dir /tmp/fallguy-check --check
```

Done when `--check` prints `OK`.

## 5. Run waves

```bash
$FALLGUY_PYTHON $SKILL/scripts/loop.py wave --run-dir <run_dir>
```

Give the Bash call a 10-minute timeout; a wave usually takes about 2 minutes. Each wave runs:
1. Argon proposes 4 candidate takes.
2. MuJoCo simulates all 4 for Gate 1: head G, hazard collisions, clearance, landing or finish, framing.
3. The best take is rendered at the requested length.
4. Argon watches that render (long-video summarization) and critiques it against the brief for Gate 2.

After every wave, give the director two lines: the best score and what failed, then the gist of Argon's critique.

Act on the printed `next`:
- **`wave`:** run the next wave.
- **`checkpoint`:** 5 more waves are done and the gates still fail. Ask the director whether to continue and, if so, what feedback the next waves should follow. If they stop, go to step 6. If they continue, run the next wave with `--steer "<feedback>"`, or plainly when they gave none. The feedback becomes a standing director note that Argon sees in every later proposal and critique.
- **`finalize`:** Gate 2 passed; go to step 6.

On errors:
- **Exit 2:** every candidate crashed. Read the first error, fix the generated script, and re-run the wave.
- **Exit 3:** Argon failed twice. Show the error line, and re-run the wave once if it looks transient.

For exact thresholds and score weights, run `loop.py gates`; for the convergence table, run `loop.py status --run-dir <run_dir>`.

## 6. Gate 3: the director decides

```bash
$FALLGUY_PYTHON $SKILL/scripts/loop.py finalize --run-dir <run_dir>
```

`open` the final video, and show the convergence table from `report.md`. Then ask the director what they want to do with this take. If they give feedback, run `loop.py wave --run-dir <run_dir> --steer "<feedback>"`, go back to step 5, and finalize again.

Done when the director locks a take. Give them the paths to the final video, `report.md`, the certificate and the generated script.
