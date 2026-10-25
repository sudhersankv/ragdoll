"""Ask Gemini 4 Argon to describe the room around the shelf (floor, walls, curtains, seating, people,
speakers, ceiling) and merge it into plan.json under "environment".

Reads the API key from ~/.argon_token (never printed).
"""
import json
import os
import sys
import urllib.error
import urllib.request

from argon_plan import IMAGE, MODEL, THINKING_LEVEL, load_image_b64

PLAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")

PROMPT = """You are the scene designer for a MuJoCo physics simulation. The photo shows a hackathon venue:
a long open wooden shelf unit in the foreground, a big LED screen on the back wall, cream floor-to-ceiling
curtains, grey sofas and black metal chairs with people sitting behind the shelf, speakers on stands,
a polished concrete floor and a dark industrial ceiling with pipes and cables.

World frame (metres): +X points from the camera toward the screen (the back wall), +Y is to the camera's left,
+Z is up, floor is z=0. The shelf unit stands at x=0,y=0 with its long axis along Y. The screen is on the back wall
around x=5.5.  Build a believable room around the shelf by describing the surroundings only (NOT the shelf or its props).

KEEP-CLEAR ZONE: nothing may stand inside x in [-6, 3.2] and y in [-3.5, 3.5] except the shelf: a person will run
and leap through that area. Put seating, people and speakers outside it (x > 3.4, or |y| > 3.8).

Return ONLY JSON:
{
 "room": {"x_min": m (<= -9), "x_max": m (>= 6.5, the back wall surface), "y_min": m, "y_max": m, "ceiling_z": m,
          "floor_rgb": [r,g,b 0-1], "wall_rgb": [..], "ceiling_rgb": [..]},
 "curtain": {"rgb": [..], "panel_width": m, "fold_depth": m, "height": m},
 "couches": [ {"x": m, "y": m, "yaw_deg": deg (0 = backrest toward +X), "length": m, "depth": m, "seat_h": m,
               "back_h": m, "rgb": [..]} ... up to 8 ],
 "chairs": [ {"x": m, "y": m, "yaw_deg": deg, "rgb": [..]} ... up to 24 ],
 "people": [ {"x": m, "y": m, "yaw_deg": deg, "seat_z": m (height of the seat they sit on), "rgb": [shirt colour]} ... up to 8 ],
 "speakers": [ {"x": m, "y": m, "z_bottom": m, "width": m, "height": m, "stand_height": m} ... up to 4 ],
 "ceiling_fixtures": [ {"kind": "pipe|beam", "axis": "x|y", "pos": [x,y,z], "length": m, "radius": m, "rgb": [..]} ... up to 12 ]
}
"""


def main():
    key = open(os.path.expanduser("~/.argon_token")).read().strip()
    body = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": load_image_b64(IMAGE)}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4,
                             "thinkingConfig": {"thinkingLevel": THINKING_LEVEL}},
    }
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=400) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500].replace(key, '<KEY>')}")
    text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
    env = json.loads(text)
    plan = json.load(open(PLAN))
    plan["environment"] = env
    json.dump(plan, open(PLAN, "w"), indent=2)
    print("usage:", resp.get("usageMetadata", {}).get("thoughtsTokenCount"), "thought tokens")
    print({k: (len(v) if isinstance(v, list) else v) for k, v in env.items() if k != "curtain"})


if __name__ == "__main__":
    main()
