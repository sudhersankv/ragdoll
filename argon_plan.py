"""Ask Gemini 4 Argon to turn the photo into a scene layout + barrel-roll choreography.

Reads the API key from ~/.argon_token (never printed). Writes plan.json.
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request

from PIL import Image

MODEL = "gemini-4-argon"
IMAGE = os.environ.get("SCENE_IMAGE", os.path.expanduser("~/Documents/IMG_2872.HEIC.png"))
THINKING_LEVEL = os.environ.get("ARGON_THINKING", "high")
OUT = os.path.join(os.path.dirname(__file__), "plan.json")

JOINTS = ["l_shoulder_x", "l_shoulder_y", "l_elbow", "r_shoulder_x", "r_shoulder_y", "r_elbow",
          "l_hip_y", "l_knee", "r_hip_y", "r_knee"]

PROMPT = f"""You are the scene designer for a MuJoCo physics simulation.
The photo shows the front of a long open wooden shelf unit (the "table") in a hackathon venue,
with a big screen on the back wall. A human will do a BARREL ROLL: a dive-roll leaping over the
shelf unit, rotating 360 degrees about their direction of travel.

World frame (metres): +X points from the camera toward the screen, +Y is to the camera's left,
+Z is up. The floor is z=0. The shelf unit is centred at x=0, y=0 and its long axis runs along Y.
Estimate real-world sizes (a person is 1.75 m tall; shelf boards are ~3 cm thick).

Return ONLY JSON with this shape:
{{
 "shelf": {{"length": m, "depth": m, "height": m (top of the highest board), "board_levels": [z of each board top],
            "post_spacing": m}},
 "screen": {{"x": m, "y": m, "width": m, "height": m, "bottom_z": m}},
 "props": [ {{"kind": "plant|candle|vase|book|bowl|can", "x": m, "y": m, "z": m (base height),
              "size": [radius, height], "rgb": [r,g,b 0-1]}} ... up to 30, placed on shelf boards or floor ],
 "choreography": {{
   "start_x": m (behind the camera side, negative), "takeoff_x": m, "landing_x": m,
   "apex_z": m (height of the torso centre at the top of the flight; must clear shelf height by >= 0.3 m),
   "poses": {{ "stand": POSE, "crouch": POSE, "takeoff": POSE, "flight": POSE, "land": POSE }}
 }}
}}
where POSE is an object mapping each of these joint names to an angle in DEGREES:
{JOINTS}
Joint conventions: shoulder_y raises the arm forward(+)/back(-), shoulder_x abducts the arm out to the side(+),
elbow bends 0..140, hip_y flexes the thigh forward(+) up to 120 / back(-) to -30, knee bends 0..140.
'flight' is the body held in a streamlined horizontal dive pose (arms extended along the travel direction).
"""


def load_image_b64(path, max_side=1600):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


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
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode()[:500].replace(key, '<KEY>')}")
    text = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
    plan = json.loads(text)
    with open(OUT, "w") as f:
        json.dump(plan, f, indent=2)
    print("usage:", resp.get("usageMetadata"))
    print(f"wrote {OUT}: {len(plan.get('props', []))} props, shelf={plan.get('shelf')}")


if __name__ == "__main__":
    main()
