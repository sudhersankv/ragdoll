"""Fake FALLGUY stunt script for the loop.py tests: obeys the stunt-script CLI without MuJoCo.

loop.py init copies it like scripts/stunt_template.py (tests set FALLGUY_STUNT_TEMPLATE) and fills the
placeholders below. Gate 1 is scored by the real scripts/gates.py. Every call appends its argv to
<out-dir>/fake_calls.jsonl so tests can check the flags loop.py passed.

Knobs (environment):
  FAKE_STUNT_PASS_FROM_WAVE  first wave whose candidates pass Gate 1 (default 1; 0 = never)
  FAKE_STUNT_CRASH           "all" or comma-separated candidate numbers whose simulation crashes
  FAKE_STUNT_SLEEP           "<candidate number>:<seconds>" makes that candidate's simulation hang
"""
SCRIPTS_DIR = "__FALLGUY_SCRIPTS_DIR__"
BRIEF = "__FALLGUY_BRIEF__"
STUNT_TYPE = '__FALLGUY_STUNT_TYPE__'
GENERATED_AT = """__FALLGUY_GENERATED_AT__"""

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, SCRIPTS_DIR)

SCENE_XML = """<mujoco model="fake"><worldbody><geom type="plane" size="5 5 0.1"/>
<body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""


def fake_metrics(cid, wave, idx, passing):
    """Plausible metrics.json content; passing takes clear every Gate 1 limit by a wide margin."""
    if passing:
        g1 = {"head_g_peak": 3.0 + 0.5 * idx, "head_g_peak_t": 4.05, "hic15": 40.0 + 5 * idx, "hazard_collisions": 0,
              "hazard_contacts": [], "min_clearance_cm": 14.0 - idx,
              "min_clearance_at": {"t": 3.5, "body": "left_ankle_link", "object": "table"},
              "landing": {"t": 4.0, "surface": "crash_mat", "point": [1.6 + 0.02 * idx, 0.0], "target": [1.6, 0.0],
                          "error_m": 0.02 * idx, "inside_mat": True},
              "framing_pct": 99.0 - idx, "rotation_deg": {"roll": 356, "pitch": 6, "yaw": 2}}
    else:
        g1 = {"head_g_peak": 10.5 + idx, "head_g_peak_t": 4.02, "hic15": 380.0, "hazard_collisions": 1,
              "hazard_contacts": [{"t": 3.6, "body": "left_ankle_link", "object": "table", "depth_cm": 1.5}],
              "min_clearance_cm": -1.5, "min_clearance_at": {"t": 3.6, "body": "left_ankle_link", "object": "table"},
              "landing": {"t": 4.1, "surface": "floor", "point": [2.6, 0.4], "target": [1.6, 0.0], "error_m": 0.9 + 0.01 * idx,
                          "inside_mat": False},
              "framing_pct": 80.0, "rotation_deg": {"roll": 250, "pitch": 30, "yaw": 10}}
    return {"candidate_id": cid, "ok": True, "error": None,
            "sim": {"t_takeoff": 3.1, "t_landing": 4.0, "flight_time": 0.9, "apex_m": 1.9, "takeoff_velocity": [3.5, 3.4]},
            "gate1": g1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--camera", default="hero", choices=["hero", "match", "side"])
    ap.add_argument("--hud", action="store_true")
    ap.add_argument("--preview-still")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "fake_calls.jsonl"), "a") as f:
        f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "brief": BRIEF, "stunt_type": STUNT_TYPE}) + "\n")
    plan = json.load(open(a.plan))
    params = json.load(open(a.params))
    if a.check:
        print("OK")
        return 0
    if a.preview_still:
        from PIL import Image
        Image.new("RGB", (320, 180), (90, 110, 140)).save(a.preview_still)
        return 0

    cid = str(params.get("id", "w0c0"))
    m = re.fullmatch(r"w(\d+)c(\d+)", cid)
    wave, idx = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    crash = os.environ.get("FAKE_STUNT_CRASH", "")
    sleep = os.environ.get("FAKE_STUNT_SLEEP", "")
    if sleep and sleep.split(":")[0] == str(idx):
        time.sleep(float(sleep.split(":")[1]))
    mpath = os.path.join(a.out_dir, "metrics.json")
    if crash == "all" or str(idx) in crash.split(","):
        json.dump({"candidate_id": cid, "ok": False, "error": f"fake crash: model blew up at t=3.2 s ({cid})"}, open(mpath, "w"))
        print(f"fake crash in {cid}", file=sys.stderr)
        return 1

    import numpy as np
    n = 50
    t = np.arange(n) * 0.002
    qpos = np.zeros((n, 7))
    qpos[:, 2], qpos[:, 3] = 1.0, 1.0
    np.savez(os.path.join(a.out_dir, "trajectory.npz"), t=t, qpos=qpos, head_g15=np.zeros(n))
    with open(os.path.join(a.out_dir, "scene.xml"), "w") as f:
        f.write(SCENE_XML)
    json.dump({"t_start": 0.0, "t_runup": 0.8, "t_takeoff": 3.1, "t_apex": 3.55, "t_landing": 4.0, "t_end": 7.0,
               "performer_root_body": "pelvis", "performer_geoms": [1], "hazard_geoms": [], "mat_geoms": [],
               "floor_geoms": [0]}, open(os.path.join(a.out_dir, "events.json"), "w"))
    pass_from = int(os.environ.get("FAKE_STUNT_PASS_FROM_WAVE", "1"))
    metrics = fake_metrics(cid, wave, idx, passing=pass_from > 0 and wave >= pass_from)
    import gates
    metrics = gates.evaluate_gate1(metrics) or metrics
    json.dump(metrics, open(mpath, "w"), indent=2)

    if a.render:
        import imageio.v2 as imageio
        w, h = (int(v) for v in a.size.split("x"))
        frames = int(round(a.duration * a.fps))
        with imageio.get_writer(os.path.join(a.out_dir, "video.mp4"), fps=a.fps, codec="libx264", quality=5,
                                macro_block_size=1) as wr:
            for i in range(frames):
                img = np.zeros((h, w, 3), np.uint8)
                img[..., 0] = 255 * i // max(1, frames - 1)
                img[..., 1] = 120 if a.hud else 40
                wr.append_data(img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
