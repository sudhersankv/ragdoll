---
name: fallguy-stunt
description: FALLGUY stages 2–4. Choreographs a stunt in an approved FALLGUY world, simulates the Unitree H1 performing it in MuJoCo with traced biomechanics (G, HIC, joint torques, impact and spinal loads), has Gemini 4 Argon critique the video and numbers, and runs one improvement take. Use when the user wants to choreograph, simulate or test a stunt in a world built by fallguy-world.
---

# FALLGUY stunt

Argon (`gemini-4-argon`) acts as the Assistant Stunt Director from `fallguy_blueprint.md`:
- It writes the choreography from the director's brief.
- It watches the rendered take and critiques it, using the video plus the biomechanics.

The performer is always the Unitree H1. Ground moves follow the choreography, and MuJoCo inverse dynamics computes their forces. From take-off on it's free physics. Safety is judged with literature injury limits. The blueprint's 8 G is shown as the conservative stunt target.

Run everything from the repo root with `.venv/bin/python`. Shapes and folders are in `fallguy/CONTRACT.md`. The Argon key is in `~/.argon_token`, and only the scripts read it. Ask the director plain-text questions with no suggested answers, then wait for their reply.

## 1. Get the world and the stunt

You need an approved world directory from `fallguy-world`. Ask the director for:
- the world, if more than one exists under `worlds/`
- the stunt they want to test
- the final video length in seconds

Done when the world exists, the stunt is a concrete action, and the length is a number from 3 to 30.

## 2. Plan the choreography

```bash
.venv/bin/python -m fallguy.stunt plan --world <world_dir> --brief "<stunt>" --duration <seconds>
```

This prints `stunt_dir` and any questions Argon has about the brief, such as which side "left" is or where the stunt ends. Ask the director those questions, and re-run `plan` with the brief clarified. Read `<stunt_dir>/stunt.json`, then summarise the phases back to the director in one or two sentences.

Done when the director agrees the choreography is what they meant.

## 3. Simulate take v1

```bash
.venv/bin/python -m fallguy.stunt simulate --stunt <stunt_dir> --take v1
```

This writes the video, `biomech.json` and `metrics.json`. Open the viewer for the director:

```bash
.venv/bin/python -m fallguy.viewer --world <world_dir> --stunt <stunt_dir>
```

Tell them the headline numbers: peak head G, HIC15, peak knee torque, peak impact force, clearance, and any limits exceeded.

## 4. Critique and improve once

```bash
.venv/bin/python -m fallguy.stunt critique --stunt <stunt_dir> --take v1
.venv/bin/python -m fallguy.stunt improve  --stunt <stunt_dir> --from v1 --to v2
```

Argon watches v1 and reads its numbers, then proposes changes. `improve` applies them, simulates v2 and critiques it.

Reload the viewer, where v1 and v2 appear side by side. Give the director Argon's critique of v1, what changed, and the v1 → v2 deltas from `compare.json`. Then ask what they want next.

Done when the director has seen v1, v2 and both critiques.
