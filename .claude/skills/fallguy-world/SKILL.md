---
name: fallguy-world
description: FALLGUY stage 1. Turns a video of a real place into a MuJoCo world model (reconstructed textured mesh plus collision geometry) that Gemini 4 Argon checks against the video, then lets the director refine it by prompt. Use when the user wants to build or edit a simulation world from a video of a location.
---

# FALLGUY world model

Builds the world that stunts run in:
- the place, reconstructed from the director's video as a textured mesh
- MuJoCo collision geometry aligned to that mesh
- a camera matching the director's shot

Argon (`gemini-4-argon`) understands the video, and then checks renders of the world against it. The director sees the world in a WebGL page in their browser.

Run everything from the repo root with `.venv/bin/python`. Shapes and folders are in `fallguy/CONTRACT.md`. The Argon key is in `~/.argon_token`, and only the scripts read it. Ask the director plain-text questions with no suggested answers, then wait for their reply.

## 1. Get the video

Ask for the path to the video of the place. Done when the path exists and is a video.

## 2. Build and check the world

```bash
.venv/bin/python -m fallguy.world build --media "<path>"
```

This takes a few minutes and prints `world_dir`. It works in three passes:
1. It reconstructs the mesh from the video's frames.
2. Argon maps the objects (walls, floor, railings, furniture, props) to collision shapes and estimates the camera.
3. Argon compares renders with the real frames and fixes what's off, for up to 2 rounds.

Open the side-by-side images in `<world_dir>/views/` with Read, and read Argon's last score and issues in `<world_dir>/verify/`.

Done when the build exits 0 and the side-by-sides show the same layout as the video.

## 3. Show the director

```bash
.venv/bin/python -m fallguy.viewer --world <world_dir>
```

This opens the world in the director's browser. Tell them Argon's fidelity score and the main issues it reported, then ask whether to change or add anything.

## 4. Apply edits

For each request:

```bash
.venv/bin/python -m fallguy.world edit --world <world_dir> --prompt "<request>"
```

Then reload the viewer page and ask again. Run `verify` after big changes.

Done when the director approves the world. Give them `world_dir` and point them to the `fallguy-stunt` skill to choreograph a stunt in it.
