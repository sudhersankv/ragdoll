"""Stage 2: Argon writes the choreography (generic phases in the world's coordinates) from the brief.

python -m fallguy.stunt plan --world DIR --brief TEXT --duration S [--out stunts] [--answers TEXT]
Prints {"stunt_dir", "questions": [...]}. Questions come back with the assumption Argon used, so the pipeline
can go on; re-run with --answers "..." to override.
"""
import json
import math
import os

import numpy as np

from .common import argon_log, engine_plan, new_dir, rj, say, wj, world_info

PHASES_DOC = """Choreography phases (generic motion primitives; no stunt types). Phases run in order from `start`
(standing, feet on the floor, facing heading_deg measured from +X toward +Y):
- hold  (kinematic): duration, pose. Stand or hold a pose in place, feet planted.
- move  (kinematic): path (>= 2 [x, y] points, the first is the current position), speed (m/s; walk 1.2-1.8,
  jog 2.5-3.5, run 4-6), accel_time, decel_time (0 = arrive at speed), turn_radius (corner rounding, m),
  lean_scale (x physical bank, usually 1), optional gait walk|run|sprint. Locomotion with planted feet.
- jump  (release): vz (m/s up), optional vx, lean_deg, rotations_deg {roll, pitch, yaw}, pose. Leaves the ground;
  free MuJoCo physics from here on.
- fall  (release): optional push [vx, vy, vz], pose. Leave support without a jump.
- air   (free): poses [{at 0..1, pose, tuck 0..1}] during the flight.
- land  (free): pose, compliance 0..1, duration (s after first contact).
After a jump or fall only air / land may follow. Poses: stand, crouch, takeoff, dive, tuck, pike, layout,
land_prep, brace, arms_up.
Lateral acceleration in turns v^2/r must stay <= 0.8 g (7.8 m/s^2) or the runner slips: e.g. 4 m/s needs r >= 2.1 m.
Keep >= 0.4 m between the performer's path and every collision object (the H1 is about 0.6 m wide, 1.8 m tall)."""

PROMPT = """You are the stunt coordinator for FALLGUY. The performer is always a Unitree H1 humanoid robot (1.8 m,
51 kg). A director filmed a location (the reference frame is attached, followed by a render of our reconstructed
3D world from the same camera, and a top-down map of the world's collision objects with the axes and the camera).
Write the choreography for this brief in the WORLD'S coordinates.

Brief: "{brief}"
Requested video length: {duration:g} s (the render covers the motion from 0.5 s before it starts to 0.5 s after it
ends; aim for {motion:.1f} s of motion including a short ready hold at the start and a stop at the end).
{answers}
World frame (metres, floor z = 0, +Z up): {frame_notes}
Reference camera (the video is rendered from here, panning to follow the performer): pos {cam_pos}, looking at
{cam_look}, vertical fov {fovy:g} deg. Its forward direction on the floor is {cam_fwd}; the camera's LEFT on the floor is
{cam_left} and its RIGHT is {cam_right}.
Walkable floor bounds (collision AABB of the world): x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}.
Collision objects (world AABBs, all hazards to avoid):
{objects}

{phases_doc}

Resolve every direction word in the brief explicitly (for example "left" can mean the camera's left or the
performer's own left, which are opposite when the performer runs toward the camera). Use what the footage shows
(for example where the camera pans to, where there is open floor) to pick the most sensible reading, state it in
`notes`, and if the brief is genuinely ambiguous add a clarification question with the options and the reading you
assumed. Choose a start and path that stay on open floor, inside the bounds, clear of objects, and visible from the
reference camera. Mark 2-6 tunables (move speeds, turn radii, path points) with sensible ranges, and set a zone
target at the end point."""


def _schema():
    from .. import argon as A
    st = A.stunt_schema()
    q = A._o({"question": A._s("the question for the director"), "options": A._a(A._s(), "possible answers"),
              "assumed": A._s("the answer this choreography assumes")}, required=["question", "assumed"])
    return A._o({"stunt": st, "questions": A._a(q, "clarification questions; empty if the brief is clear"),
                 "crash_mats": A._a(A._o({"id": A._s(), "pos": A._v3("centre; z = half thickness"),
                                          "size": A._v3("half-extents")}),
                                    "only if the stunt lands from a jump or fall; else empty")},
                required=["stunt", "questions"])


def draw_map(W, out_png, stunt=None, size=900):
    """Top-down map: collision AABBs labelled, axes with metre ticks, the ref camera and its view direction,
    and (optionally) the choreography path."""
    from PIL import Image, ImageDraw, ImageFont
    b = W["bounds"]
    cam = W["camera"]
    xs = [b["x_min"], b["x_max"], cam["pos"][0]]
    ys = [b["y_min"], b["y_max"], cam["pos"][1]]
    x0, x1, y0, y1 = min(xs) - 1, max(xs) + 1, min(ys) - 1, max(ys) + 1
    span = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1, y0, y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
    s = (size - 80) / span
    im = Image.new("RGB", (size, size), (250, 250, 248))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except OSError:
        f = ImageFont.load_default()
    # screen: +X to the right, +Y up
    P = lambda x, y: (40 + (x - x0) * s, size - 40 - (y - y0) * s)
    for gx in range(int(math.floor(x0)), int(math.ceil(x1)) + 1):
        d.line([P(gx, y0), P(gx, y1)], fill=(225, 225, 225) if gx % 5 else (190, 190, 190))
        if gx % 2 == 0:
            d.text((P(gx, y0)[0] - 6, size - 34), str(gx), fill=(90, 90, 90), font=f)
    for gy in range(int(math.floor(y0)), int(math.ceil(y1)) + 1):
        d.line([P(x0, gy), P(x1, gy)], fill=(225, 225, 225) if gy % 5 else (190, 190, 190))
        if gy % 2 == 0:
            d.text((8, P(x0, gy)[1] - 7), str(gy), fill=(90, 90, 90), font=f)
    d.text((size - 70, size - 20), "+X  ->", fill=(0, 0, 0), font=f)
    d.text((6, 6), "+Y (up)", fill=(0, 0, 0), font=f)
    for o in W["objects"]:
        lo, hi = o["lo"], o["hi"]
        a, c = P(lo[0], hi[1]), P(hi[0], lo[1])
        d.rectangle([a, c], outline=(200, 60, 40), fill=(245, 205, 195))
        d.text((a[0] + 2, a[1] + 1), o["geom"].split("__")[0][:22], fill=(120, 30, 20), font=f)
    cp = np.array(cam["pos"][:2])
    fw = np.array(cam["lookat"][:2]) - cp
    fw /= max(np.linalg.norm(fw), 1e-9)
    half = math.radians(cam["fovy"] * (W["width"] / W["height"]) / 2)
    for sgn in (-1, 1):
        c_, s_ = math.cos(sgn * half), math.sin(sgn * half)
        r = np.array([fw[0] * c_ - fw[1] * s_, fw[0] * s_ + fw[1] * c_]) * 6
        d.line([P(*cp), P(*(cp + r))], fill=(40, 90, 200), width=2)
    d.ellipse([P(*cp)[0] - 6, P(*cp)[1] - 6, P(*cp)[0] + 6, P(*cp)[1] + 6], fill=(40, 90, 200))
    d.text((P(*cp)[0] + 8, P(*cp)[1] - 6), "camera", fill=(40, 90, 200), font=f)
    if stunt:
        from ..engine import scene
        try:
            pts = scene.stunt_points(scene.normalize_stunt(stunt))
            d.line([P(*p) for p in pts], fill=(20, 150, 60), width=3)
            d.text(P(*pts[0]), "start", fill=(20, 150, 60), font=f)
        except Exception:                                        # noqa: BLE001
            pass
    im.save(out_png)
    return out_png


def world_view(W, out_png):
    """The world's own view from its ref camera (views/*.png from stage 1 if any, else a MuJoCo render)."""
    import glob
    hits = sorted(glob.glob(os.path.join(W["dir"], "views", "*ref*.png")) + glob.glob(os.path.join(W["dir"], "views", "*.png")))
    if hits:
        return hits[0]
    import mujoco
    from PIL import Image
    m = mujoco.MjModel.from_xml_path(os.path.join(W["dir"], "scene.xml"))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, 720), max(m.vis.global_.offheight, 1280)
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    h = 960
    w = int(round(h * W["width"] / W["height"] / 2)) * 2
    with mujoco.Renderer(m, h, w) as r:
        r.update_scene(dd, "ref" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "ref") >= 0 else 0)
        Image.fromarray(r.render()).save(out_png)
    return out_png


def _cam_dirs(cam):
    fw = np.array(cam["lookat"][:2], float) - np.array(cam["pos"][:2], float)
    fw /= max(np.linalg.norm(fw), 1e-9)
    left = np.array([-fw[1], fw[0]])
    fmt = lambda v: f"[{v[0]:+.2f}, {v[1]:+.2f}]"
    return fmt(fw), fmt(left), fmt(-left)


def plan(world_dir, brief, duration, out_root="stunts", answers=None):
    from .. import argon as A
    W = world_info(world_dir)
    sd = new_dir(out_root, brief)
    log = argon_log(sd)
    mp = draw_map(W, os.path.join(sd, "map.png"))
    wv = world_view(W, os.path.join(sd, "world_view.png"))
    b = W["bounds"]
    objs = "\n".join(f"- {o['geom']}: x {o['lo'][0]:.2f}..{o['hi'][0]:.2f}, y {o['lo'][1]:.2f}..{o['hi'][1]:.2f}, "
                     f"z {o['lo'][2]:.2f}..{o['hi'][2]:.2f}" for o in W["objects"][:60])
    wjson = W["json"]
    labels = {str(o.get("id")): f"{o.get('label', '')} ({o.get('role', '')})" for o in wjson.get("objects") or []
              if isinstance(o, dict)}
    if labels:
        objs += "\nObject labels: " + "; ".join(f"{k} = {v}" for k, v in labels.items())
    fn = wjson.get("frame")
    frame_notes = (fn.get("notes") if isinstance(fn, dict) else fn) or "see the map"
    cf, cl, cr = _cam_dirs(W["camera"])
    text = PROMPT.format(brief=brief, duration=duration, motion=max(duration - 1.0, 1.0),
                         answers=(f"Director's answers to earlier questions: {answers}\n" if answers else ""),
                         frame_notes=frame_notes, cam_pos=W["camera"]["pos"], cam_look=W["camera"]["lookat"],
                         fovy=W["camera"]["fovy"], cam_fwd=cf, cam_left=cl, cam_right=cr, x0=b["x_min"],
                         x1=b["x_max"], y0=b["y_min"], y1=b["y_max"], objects=objs, phases_doc=PHASES_DOC)
    parts = [{"text": text}]
    for label, p in (("Reference frame from the director's video:", W["ref_frame"]),
                     ("Our reconstructed world rendered from the same camera:", wv),
                     ("Top-down map of the world (+X right, +Y up, metres; red = collision objects, blue = camera):", mp)):
        if p and os.path.isfile(p):
            parts.append({"text": label})
            parts.append(A.image_part(p)[0])
    ans, _ = A.generate(parts, _schema(), "stunt_plan", log)
    warn = []
    mats = []
    for i, m in enumerate(ans.get("crash_mats") or []):
        try:
            mats.append({"id": A._slug(m.get("id"), f"mat{i + 1}"), "pos": [float(v) for v in m["pos"]][:3],
                         "size": [abs(float(v)) for v in m["size"]][:3], "yaw_deg": 0.0, "rgb": [0.15, 0.35, 0.85]})
        except Exception:                                        # noqa: BLE001
            warn.append(f"crash mat {i} unusable; dropped")
    stunt = A.normalize_stunt(ans.get("stunt") or {}, mats, warn)
    if mats:
        stunt["crash_mats"] = mats
    warn = [w for w in warn if "heading_deg" not in w]
    questions = [q for q in ans.get("questions") or [] if isinstance(q, dict) and q.get("question")]
    S = {"version": 1, "brief": brief, "duration_sec": float(duration), "world_dir": W["dir"], "stunt": stunt,
         "questions": questions, "clarifications": answers, "plan_warnings": warn,
         "argon": {"model": A.model_name(), "thinking": A.thinking_level()}, "takes": []}
    wj(os.path.join(sd, "stunt.json"), S)
    draw_map(W, os.path.join(sd, "map.png"), stunt)
    from ..engine import scene
    scene.normalize_plan(engine_plan(W, stunt, brief, duration))       # fail early on anything unusable
    say(f"plan: {stunt.get('summary')}")
    for w in warn:
        say(f"plan warning: {w}")
    return {"stunt_dir": sd, "summary": stunt.get("summary"), "notes": stunt.get("notes"), "questions": questions}
