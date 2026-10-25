"""Director edits: Argon turns a plain-text prompt into world.json changes; rebuild scene.xml and re-render."""
import json
import os
from datetime import datetime

from fallguy import argon as A
from fallguy.world import build as B
from fallguy.world import verify as V

EDIT_PROMPT = """You edit a VIDEO-GAME-STYLE 3D replica of a real place for stunt previs. World frame (metres): floor
z = 0, +z up, phone at x = 0, y = 0 looking along +x (down the hallway), +y = camera LEFT, -y = camera RIGHT.
Image 1 = real reference frame | replica (side-by-side, plus the end of the pan), image 2 = top-down view of the
replica with collision primitives, image 3 = the reference frame with the world floor grid (1 m).

Current objects (center [x, y], base_z, facing_deg, size [width, depth, height]; floor/ceiling/atrium_void size =
[extent x, extent y, thickness]):
{objects}

Director's request: "{prompt}"

Return only the changes needed (add / replace / remove) with full objects in the same asset convention. A crash mat is
category "crash_mat", look "foam", hazard false, collide true, on the floor (base_z 0), typically size
[1.5, 2.0, 0.3] unless told otherwise, bright blue or red; keep it clear of walls and the railing.
"""


def edit(wdir, prompt):
    world = B.read_json(os.path.join(wdir, "world.json"))
    n = len(world.get("edits") or []) + 1
    last = sorted((world.get("views") or {"r0": None}).keys())[-1]
    views = (world.get("views") or {}).get(last) or B.render_views(world, wdir, last)
    parts = []
    for i, p in enumerate([views["side_by_side"], views["top"], "views/argon_grid_ref.png"], 1):
        part, _ = A.image_part(os.path.join(wdir, p))
        parts += [{"text": f"Image {i}:"}, part]
    parts.append({"text": EDIT_PROMPT.format(objects=json.dumps(B.compact_objects(world)), prompt=prompt)})
    ans, _ = A.generate(parts, B.edit_schema(), f"world_edit_{n}", B.log_path(wdir))
    applied = V.apply_changes(world, ans.get("changes") or [])
    B.save_world(world, wdir)
    tag = f"e{n}"
    world.setdefault("views", {})[tag] = B.render_views(world, wdir, tag)
    world.setdefault("edits", []).append({"n": n, "prompt": prompt, "summary": ans.get("summary"), "applied": applied,
                                          "changes": ans.get("changes"), "views": tag,
                                          "ts": datetime.now().isoformat(timespec="seconds")})
    B.write_json(os.path.join(wdir, "world.json"), world)
    B.say(f"edit {n}: {ans.get('summary')} ({', '.join(applied) or 'no changes'})")
    return world["edits"][-1]
