"""Argon render-vs-video critique of a world; optionally applies the corrections and re-renders."""
import json
import os

from fallguy import argon as A
from fallguy.world import build as B

VERIFY_PROMPT = """You are checking a VIDEO-GAME-STYLE 3D replica of a real place against the real video (stylised
low-poly assets with flat colours are intended; do NOT ask for photo textures or realism). World frame (metres):
floor z = 0, +z up, phone at x = 0, y = 0, height {cam_h} m, looking along +x at the reference frame; +y = camera
LEFT, -y = camera RIGHT; the video then pans left to yaw {end_yaw} deg.

Images: (1) side-by-side: real reference frame | replica from the same camera | real end-of-pan frame | replica
from the end-of-pan camera; (2) oblique overview of the replica (ceiling hidden); (3) top-down view of the replica
with the collision primitives (blue); (4) the reference frame with our world floor grid (yellow = constant x,
cyan = constant y, 1 m) to read positions.

Current objects (center [x, y], base_z, facing_deg, size [width, depth, height]):
{objects}

Score 0-100 the LAYOUT FIDELITY: are the right objects there, in the right places, with the right sizes, shapes and
colours, compared with the video, from both cameras? Then list the concrete problems (most important first, at most
12): misplaced or mis-sized objects (replace with the full corrected object, same id), missing objects (add), and
wrong / duplicate objects (remove). Keep the asset conventions (category, look, rgb, accent, center, base_z,
facing_deg, size, hazard, collide, parts=[] unless category other).
"""


def verify(wdir, rnd=1, apply=True):
    world = B.read_json(os.path.join(wdir, "world.json"))
    tag = f"r{rnd - 1}"
    views = (world.get("views") or {}).get(tag) or B.render_views(world, wdir, tag)
    log = B.log_path(wdir)
    parts = []
    for i, p in enumerate([views["side_by_side"], views["oblique"], views["top"], "views/argon_grid_ref.png"], 1):
        part, _ = A.image_part(os.path.join(wdir, p), max_side=2200 if i == 1 else 1600)
        parts += [{"text": f"Image {i}:"}, part]
    parts.append({"text": VERIFY_PROMPT.format(objects=json.dumps(B.compact_objects(world)),
                                               cam_h=world["camera"]["ref"]["pos"][2],
                                               end_yaw=(world["camera"].get("end") or {}).get("yaw_deg", 0))})
    ans, _ = A.generate(parts, B.critique_schema(), f"world_verify_r{rnd}", log)
    score = float(ans.get("score") or 0)
    applied = []
    if apply:
        applied = apply_changes(world, ans.get("issues") or [])
        B.save_world(world, wdir)
        world.setdefault("views", {})[f"r{rnd}"] = B.render_views(world, wdir, f"r{rnd}")
    rec = {"round": rnd, "score": score, "summary": ans.get("summary"), "issues": ans.get("issues"),
           "fixes": applied, "renders_judged": views}
    B.write_json(os.path.join(wdir, "verify", f"round{rnd}.json"), rec)
    world.setdefault("verify", []).append({k: rec[k] for k in ("round", "score", "summary", "fixes")}
                                          | {"issues": len(rec["issues"] or [])})
    B.write_json(os.path.join(wdir, "world.json"), world)
    B.say(f"verify round {rnd}: score {score:.0f}, {len(ans.get('issues') or [])} issues, applied {len(applied)}")
    return rec


def apply_changes(world, changes):
    """add / replace / remove objects in place. Returns a list of 'action id' strings."""
    objs = {o["id"]: o for o in world["objects"]}
    done = []
    for c in changes:
        act = c.get("action")
        oid = (c.get("object_id") or (c.get("object") or {}).get("id") or "").strip()
        if act == "remove":
            if oid in objs:
                del objs[oid]
                done.append(f"remove {oid}")
            continue
        o = c.get("object")
        if not o or not o.get("size"):
            continue
        new = B.to_object({**o, "id": oid or o.get("id")})
        if act == "replace" and new["id"] not in objs:
            act = "add"
        objs[new["id"]] = new
        done.append(f"{act} {new['id']}")
    world["objects"] = list(objs.values())
    return done
