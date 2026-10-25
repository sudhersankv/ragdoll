"""Stage 1: build a GAME-STYLE 3D replica of the place in a video.

frames + depth (measurement only) -> Argon object list (category, pose, size, colours, look, hazard) -> procedural
low-poly assets (world/assets.py) -> mesh/env.glb + mesh/*.obj + scene.xml (visual meshes + collision primitives +
lights + camera "ref") -> MuJoCo renders next to the real frames -> Argon verify/fix rounds (<= 2, stop at >= 80).
Real video pixels are never used as textures.
"""
import json
import os
import re
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from fallguy import argon as A
from fallguy.world import assets as AS
from fallguy.world import reconstruct as R

RENDER_SIZE = (540, 960)
PHONE_HEIGHT_PRIOR = 1.5          # m; the measurement grid is pre-scaled so the phone sits at this height


def say(msg):
    R.say(msg)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def log_path(wdir):
    return os.path.join(wdir, "logs", "argon_calls.jsonl")


# ----------------------------------------------------------------------------- Argon schemas
def part_schema():
    return A._o({"type": A._e(["box", "cylinder", "sphere"]),
                 "center": A._v3("[x, y, z] in the object's local frame: x = towards its front, y = its left, "
                                 "z = height above the object's base"),
                 "dims": A._v3("[width, depth, height] in metres"), "rgb": A._rgb_s("colour"),
                 "material": A._e(AS.LOOKS)})


def object_schema(with_parts=True):
    props = {
        "id": A._s("short snake_case id, unique"), "label": A._s("what it is, a few words"),
        "category": A._e(AS.CATEGORIES, "asset category; 'other' only if nothing fits (then give parts)"),
        "look": A._e(AS.LOOKS, "surface look of the main body"),
        "rgb": A._rgb_s("main colour, a clean stylised colour matching the real object"),
        "accent": A._rgb_s("secondary colour (frame, trim, pot, screen image, rail, pants...)"),
        "center": A._xy("[x, y] centre of the footprint in world metres (read off the grids)"),
        "base_z": A._n("height of its bottom above the floor in metres (0 = on the floor; ceiling items high up; "
                       "the atrium's lower floor is negative)"),
        "facing_deg": A._n("direction its FRONT faces in the floor plane: 0 = +x (away from the camera), "
                           "180 = -x (towards the camera), 90 = +y (camera left), -90 = -y (camera right). For a "
                           "wall, curtain or railing: the direction its visible face points (its length runs "
                           "perpendicular to that)."),
        "size": A._v3("[width (side to side seen from its front; for walls/curtains/railings = their LENGTH), "
                      "depth (front to back; thickness), height] in metres; for floor / ceiling / atrium_void: "
                      "[extent along x, extent along y, thickness]"),
        "hazard": A._b("could hurt a performer (hard edge, glass, a drop, screen, metal stand)"),
        "collide": A._b("solid for the performer (false for decals, floor patches, lights, far-away scenery)"),
    }
    if with_parts:
        props["parts"] = A._a(part_schema(), "ONLY for category 'other': 1-8 primitives building its shape; [] otherwise")
    return A._o(props)


def layout_schema():
    return A._o({
        "scale_checks": A._a(A._o({"what": A._s("something of known size: door height, railing height, hallway "
                                                 "width, a person"),
                                   "real_m": A._n("your real-world estimate"),
                                   "grid_m": A._n("the same size read off our grids")}), "2-5 size checks"),
        "camera_height_m": A._n("your estimate of the phone's real height above the floor"),
        "floor": A._o({"look": A._e(AS.LOOKS), "rgb": A._rgb_s(), "friction": A._n("0.3-1.2")}),
        "objects": A._a(object_schema(), "EVERY object of the place, 25-70 of them"),
        "notes": A._s("one or two sentences on what you could not place well")})


def change_schema():
    return A._o({"action": A._e(["replace", "add", "remove"]),
                 "object_id": A._s("id of an existing object (replace/remove) or a new id (add)"),
                 "problem": A._s("what was wrong / what changes, concretely"),
                 "severity": A._e(A.SEVERITIES),
                 "object": object_schema(with_parts=False)})


def critique_schema():
    return A._o({
        "score": A._n("layout fidelity 0-100 of the replica vs the real place (positions, sizes, colours, coverage)"),
        "summary": A._s("one sentence"),
        "issues": A._a(change_schema(), "at most 10 real layout problems, most important first; for replace/add "
                                        "give the FULL corrected object; for remove any minimal object")})


def edit_schema():
    return A._o({"changes": A._a(change_schema(), "the edits to apply"),
                 "summary": A._s("one sentence of what changed")})


# ----------------------------------------------------------------------------- objects
ROLE_OF = {"floor": "floor_feature", "floor_patch": "floor_feature", "floor_line": "floor_feature",
           "ceiling": "structure", "wall": "wall", "curtain": "wall", "glass_railing": "railing", "railing": "railing",
           "doorway": "door", "door": "door", "window": "structure", "arched_window": "structure", "arch": "structure",
           "column": "structure", "beam": "structure", "pipe": "structure", "light_fixture": "prop",
           "led_screen": "screen", "tv_screen": "screen", "speaker_stand": "prop", "truss": "structure",
           "potted_plant": "plant", "sign_stand": "prop", "wall_sign": "prop", "exit_sign": "prop",
           "bench": "furniture", "chair": "furniture", "table": "furniture", "counter": "furniture", "bin": "prop",
           "box": "prop", "person": "person", "stairs": "structure", "crash_mat": "crash_mat",
           "atrium_void": "floor_feature", "other": "prop"}


def clean_id(s):
    return re.sub(r"[^a-z0-9_]+", "_", str(s or "obj").lower()).strip("_") or "obj"


def to_object(o):
    keep = ("id", "label", "category", "look", "rgb", "accent", "center", "base_z", "facing_deg", "size", "hazard",
            "collide", "parts", "look2")
    n = {k: o.get(k) for k in keep if k in o}
    n["id"] = clean_id(o.get("id") or o.get("label"))
    n["hazard"] = bool(o.get("hazard"))
    if n.get("category") != "other":
        n["parts"] = []
    AS.normalize(n)
    n["role"] = ROLE_OF.get(n["category"], "prop")
    return n


def unique_ids(objs):
    seen = {}
    for o in objs:
        base = o["id"]
        if base in seen:
            seen[base] += 1
            o["id"] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
    return objs


def compact_objects(world):
    """Objects as Argon sees them (the same schema it writes)."""
    keys = ("id", "label", "category", "look", "rgb", "accent", "center", "base_z", "facing_deg", "size", "hazard",
            "collide")
    return [{k: o.get(k) for k in keys} for o in world["objects"]]


def ensure_floor(objs):
    """A floor slab under everything if Argon gave none."""
    if any(o["category"] == "floor" for o in objs):
        return objs
    xs = [o["center"][0] for o in objs] or [0]
    ys = [o["center"][1] for o in objs] or [0]
    f = to_object({"id": "floor_main", "label": "floor", "category": "floor", "look": "concrete",
                   "rgb": [0.48, 0.46, 0.44], "center": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2],
                   "size": [max(ys) - min(ys) + 4, max(xs) - min(xs) + 4, 0.1], "facing_deg": 0})
    return [f] + objs


# ----------------------------------------------------------------------------- assembly: meshes, GLB, scene.xml
def _write_obj(path, m):
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    with open(path, "w") as f:
        f.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V))
        f.write("".join(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in N))
        f.write("".join(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n" for a, b, c in m.faces))


def _mat_key(look, rgb):
    return f"{look}_{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"


SMOOTH = {"curtain"}


def assemble(world, wdir):
    """Build every object's asset; write mesh/<id>__v<k>.obj and mesh/env.glb; fill o["collision"]. Returns the
    visual geoms [(name, obj file, material key)] and the material table."""
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    mdir = os.path.join(wdir, "mesh")
    os.makedirs(mdir, exist_ok=True)
    for f in os.listdir(mdir):
        if f.endswith((".obj", ".glb", ".mtl")):
            os.remove(os.path.join(mdir, f))
    scene = trimesh.Scene()
    visuals, mats = [], {}
    tris = 0
    for o in world["objects"]:
        parts, prims = AS.build_object(o)
        o["collision"] = prims
        groups = {}
        for look, rgb, m in parts:
            groups.setdefault(_mat_key(look, rgb), (look, rgb, []))[2].append(m)
        for k, (key, (look, rgb, ms)) in enumerate(groups.items()):
            m = trimesh.util.concatenate(ms)
            if o["category"] not in SMOOTH or look != o["look"]:
                m.unmerge_vertices()               # faceted, crisp low-poly shading
            name = f"{o['id']}__v{k}"
            _write_obj(os.path.join(mdir, f"{name}.obj"), m)
            visuals.append((name, f"{name}.obj", key))
            mats[key] = (look, rgb)
            rough, metal, alpha, emis = AS.LOOK_PROPS[look]
            col = [int(255 * c) for c in rgb] + [int(255 * alpha)]
            pm = PBRMaterial(name=key, baseColorFactor=col, metallicFactor=metal, roughnessFactor=rough,
                             alphaMode="BLEND" if alpha < 1 else "OPAQUE", doubleSided=True,
                             emissiveFactor=[c * emis for c in rgb] if emis else None)
            g = m.copy()
            g.visual = TextureVisuals(material=pm)
            scene.add_geometry(g, node_name=name, geom_name=name)
            tris += len(m.faces)
    # a wide ground fill below everything (no sky through gaps between floor slabs / under the atrium)
    zmin = min([o["base_z"] for o in world["objects"]] + [0.0]) - 0.6
    frgb = world["floor"]["rgb"]
    g = trimesh.creation.box(extents=[120, 120, 0.1])
    g.apply_translation([0, 0, zmin - 0.05])
    _write_obj(os.path.join(mdir, "ground_fill.obj"), g)
    key = _mat_key("concrete", AS.mix(frgb, 0.8))
    visuals.append(("ground_fill", "ground_fill.obj", key))
    mats[key] = ("concrete", AS.mix(frgb, 0.8))
    gg = g.copy()
    gg.visual = TextureVisuals(material=PBRMaterial(name=key, baseColorFactor=[int(255 * c) for c in AS.mix(frgb, 0.8)] + [255],
                                                    metallicFactor=0.0, roughnessFactor=0.95, doubleSided=True))
    scene.add_geometry(gg, node_name="ground_fill", geom_name="ground_fill")
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)   # z-up world -> glTF y-up
    scene.apply_transform(T)
    scene.export(os.path.join(mdir, "env.glb"), include_normals=True)
    world["mesh"] = {"glb": "mesh/env.glb", "objs": "mesh/<object id>__v<k>.obj (one per object and material)",
                     "style": "game-style procedural low-poly assets with flat stylised materials; no photo textures",
                     "glb_axes": "glTF Y-up: glb (X, Y, Z) = world (x, z, -y); rotate +90 deg about X for z-up",
                     "obj_axes": "world frame (z up), same as scene.xml", "triangles": int(tris),
                     "visual_geoms": len(visuals)}
    return visuals, mats


def _bounds(world):
    lo, hi = np.array([1e9, 1e9, 0.0]), np.array([-1e9, -1e9, 2.5])
    for o in world["objects"]:
        for p in o.get("collision") or []:
            d = np.asarray(p["dims"]) / 2
            r = max(d[0], d[1])
            c = np.asarray(p["pos"])
            lo = np.minimum(lo, c - [r, r, d[2]])
            hi = np.maximum(hi, c + [r, r, d[2]])
    if lo[0] > hi[0]:
        lo, hi = np.array([-2, -3, 0.0]), np.array([12, 3, 3.0])
    return lo, hi


def scene_xml(world, visuals, mats):
    cam = world["camera"]["ref"]
    fl = world["floor"]
    lo, hi = _bounds(world)
    ceils = [o["base_z"] for o in world["objects"] if o["category"] == "ceiling"]
    ceil = min(6.0, max(2.6, min(ceils) if ceils else hi[2]))
    xs = [f'<mujoco model="world_{world["slug"]}">',
          '  <compiler angle="degree" meshdir="mesh"/>',
          '  <option timestep="0.002" gravity="0 0 -9.81"/>',
          '  <visual><headlight ambient="0.42 0.42 0.44" diffuse="0.32 0.32 0.32" specular="0.06 0.06 0.06"/>'
          '<map znear="0.02" zfar="200" haze="0.15"/><quality shadowsize="8192" offsamples="8"/>'
          '<global offwidth="1920" offheight="1920"/></visual>',
          '  <asset>',
          '    <texture type="skybox" builtin="gradient" rgb1="0.86 0.9 0.96" rgb2="0.55 0.6 0.68" width="512" height="512"/>']
    for key, (look, rgb) in sorted(mats.items()):
        rough, metal, alpha, emis = AS.LOOK_PROPS[look]
        spec = 0.6 if look in ("glass", "metal", "screen", "ceramic") else 0.15
        shin = round(1 - rough, 2)
        refl = 0.15 if look == "glass" else 0.05 if look in ("metal", "ceramic") else 0.0
        xs.append(f'    <material name="{key}" rgba="{rgb[0]} {rgb[1]} {rgb[2]} {alpha}" specular="{spec}" '
                  f'shininess="{shin}" reflectance="{refl}" emission="{round(emis * 0.9, 2)}"/>')
    for name, f, _ in visuals:
        xs.append(f'    <mesh name="{name}" file="{f}" inertia="shell"/>')
    xs += ['  </asset>', '  <worldbody>']
    # lights under the ceiling along the place (a roofed interior: a directional sun would be blocked by the ceiling)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    lz = ceil - 0.3
    fw = np.asarray(cam.get("forward") or [1, 0, 0], float)
    fw2 = fw[:2] / max(1e-6, np.linalg.norm(fw[:2]))
    kx, ky = cam["pos"][0] + 3.5 * fw2[0], cam["pos"][1] + 3.5 * fw2[1]      # over the action, ~3.5 m ahead
    # the shadow-casting key sits below every high object (pipes, beams, pendants) so they don't shade the floor
    highs = [o["base_z"] for o in world["objects"] if o["base_z"] > 2.3 and o["category"] not in ("atrium_void",)]
    kz = max(2.4, min([lz] + highs) - 0.15)
    xs.append(f'    <light name="key" pos="{kx:.2f} {ky:.2f} {kz:.2f}" dir="0.12 -0.08 -1" diffuse="0.5 0.48 0.45" '
              f'ambient="0.3 0.3 0.31" specular="0.2 0.2 0.2" castshadow="true" cutoff="88" exponent="1"/>')
    # soft directional fills from both sides (spot fills without shadows black out back faces in MuJoCo's renderer)
    for i, (dx, dy) in enumerate([(0.3, -0.5), (-0.4, 0.6), (0.6, 0.2)]):
        xs.append(f'    <light name="fill{i}" pos="{cx:.2f} {cy:.2f} {lz:.2f}" dir="{dx} {dy} -1" directional="true" '
                  f'diffuse="0.16 0.16 0.16" specular="0.04 0.04 0.04" castshadow="false"/>')
    xs.append(f'    <geom name="floor" type="plane" size="60 60 0.1" rgba="{fl["rgb"][0]} {fl["rgb"][1]} {fl["rgb"][2]} 1" '
              f'friction="{fl.get("friction", 0.8)} 0.005 0.0001" group="0"/>')
    xs.append('    <camera name="ref" pos="{} {} {}" xyaxes="{}" fovy="{}"/>'.format(
        *cam["pos"], " ".join(f"{v:.5f}" for v in cam["xyaxes"]), round(cam["fovy_deg"], 3)))
    end = world["camera"].get("end")
    if end:
        xs.append('    <camera name="end" pos="{} {} {}" xyaxes="{}" fovy="{}"/>'.format(
            *end["pos"], " ".join(f"{v:.5f}" for v in end["xyaxes"]), round(cam["fovy_deg"], 3)))
    for name, _, key in visuals:
        xs.append(f'    <geom name="{name}" type="mesh" mesh="{name}" material="{key}" contype="0" conaffinity="0" group="1"/>')
    for o in world["objects"]:
        fr = "0.9 0.01 0.001" if o["material"] in ("foam", "fabric") else "1.0 0.005 0.0001"
        solref = ' solref="0.06 1"' if o["material"] == "foam" else ""
        for i, p in enumerate(o.get("collision") or []):
            size = " ".join(str(v) for v in p["size"] if v) or "0.1"
            xs.append(f'    <geom name="{o["id"]}__{i}" type="{p["type"]}" pos="{" ".join(map(str, p["pos"]))}" '
                      f'size="{size}" quat="{" ".join(map(str, p["quat"]))}" rgba="0.3 0.75 1 0.35" '
                      f'friction="{fr}"{solref} group="2"/>')
    xs += ['  </worldbody>', '</mujoco>', '']
    return "\n".join(xs)


def save_world(world, wdir):
    """objects -> assets, GLB, OBJs, scene.xml (compiled to check), world.json."""
    world["objects"] = unique_ids(ensure_floor(world["objects"]))
    visuals, mats = assemble(world, wdir)
    path = os.path.join(wdir, "scene.xml")
    with open(path, "w") as f:
        f.write(scene_xml(world, visuals, mats))
    import mujoco
    mujoco.MjModel.from_xml_path(path)          # fail loudly if the MJCF is broken
    write_json(os.path.join(wdir, "world.json"), world)
    return path


# ----------------------------------------------------------------------------- renders
def _label(im, text):
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    dr.text((6, 5), text, fill=(255, 255, 255))
    return im


def render_views(world, wdir, tag="r0"):
    """MuJoCo renders of the replica from the reference + end-of-pan cameras, an oblique and a top view (ceiling
    hidden), and a side-by-side with the real frames."""
    import mujoco
    vdir = os.path.join(wdir, "views")
    os.makedirs(vdir, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(os.path.join(wdir, "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w, h = RENDER_SIZE
    lo, hi = _bounds(world)
    high = {o["id"] for o in world["objects"]
            if o["category"] in ("ceiling", "light_fixture", "pipe", "beam") or o["base_z"] > 2.4}
    out = {}
    with mujoco.Renderer(m, h, w) as r:
        def shot(groups, cam="ref", hide_ceiling=False):
            opt = mujoco.MjvOption()
            for g in range(6):
                opt.geomgroup[g] = 1 if g in groups else 0
            r.update_scene(d, camera=cam, scene_option=opt)
            if hide_ceiling:
                for gi in range(r.scene.ngeom):
                    g = r.scene.geoms[gi]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g.objid) or ""
                        if nm.split("__")[0] in high:
                            g.pos[:] = [0.0, 0.0, -1e4]          # drop it out of view (alpha 0 still draws)
            return Image.fromarray(r.render())
        ims = {"ref_mesh": shot({1}), "ref_prims": shot({1, 2})}
        if world["camera"].get("end"):
            ims["end_mesh"] = shot({1}, cam="end")
    with mujoco.Renderer(m, w, h) as r:            # landscape overviews
        free = mujoco.MjvCamera()
        free.lookat[:] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0.3]
        span = max(hi[0] - lo[0], hi[1] - lo[1], 6)
        free.distance, free.azimuth, free.elevation = span * 0.95, 12.0, -52.0
        ims["oblique"] = shot({1}, cam=free, hide_ceiling=True)
        free.azimuth, free.elevation, free.distance = 90.0, -89.9, span * 0.95
        ims["top"] = shot({1, 2}, cam=free, hide_ceiling=True)
    for name, im in ims.items():
        p = os.path.join(vdir, f"{tag}_{name}.png")
        im.save(p)
        out[name] = os.path.relpath(p, wdir)
    real = Image.open(os.path.join(wdir, world["frames"]["ref"])).convert("RGB").resize((w, h))
    cols = [(real, "real frame (ref, t=%.1fs)" % world["frames"]["ref_time"]), (ims["ref_mesh"], "replica, ref camera")]
    if "end_mesh" in ims:
        endp = os.path.join(wdir, world["camera"]["end"]["file"])
        cols += [(Image.open(endp).convert("RGB").resize((w, h)), "real frame (end of pan)"),
                 (ims["end_mesh"], "replica, end-of-pan camera")]
    sbs = Image.new("RGB", (w * len(cols), h))
    for i, (im, t) in enumerate(cols):
        sbs.paste(_label(im.copy(), t), (i * w, 0))
    p = os.path.join(vdir, f"{tag}_side_by_side.png")
    sbs.save(p)
    out["side_by_side"] = os.path.relpath(p, wdir)
    say(f"renders -> {vdir} ({tag}_*.png)")
    return out


# ----------------------------------------------------------------------------- Argon layout
LAYOUT_PROMPT = """You are the level designer for FALLGUY, a stunt-previs tool. Build a VIDEO-GAME-STYLE 3D REPLICA of the
place in this video: every object becomes a clean modelled asset from our library (no photo textures), placed and
sized to match the video so a render from the phone's camera looks like the real shot.

WORLD FRAME (metres): floor z = 0, +z up. The phone stands at x = 0, y = 0 (on the floor below it) at height
{cam_h} m, looking along +x at the reference frame (t = {ref_time} s). +y = camera LEFT, -y = camera RIGHT.
The camera holds still, then pans LEFT (towards +y) by about {pan} deg by the end of the video.
Images: 2 = reference frame with the world floor grid projected on it (yellow = constant x every 1 m, cyan =
constant y every 1 m, red = y 0, orange = x <= 0); 3 = a mid-pan frame and 4 = the end-of-pan frame with the same
grid; 5 = top-down plan view of points measured from monocular depth (grey floor, orange 0.1-1.2 m high, blue
1.2-2.5 m, purple above 2.5 m; noisy, especially far away and on curtains/glass). Read positions off the grids.
Measurements: {meas}

Library categories (choose the best fit; sizes are [width, depth, height]):
{cats}
- floor, ceiling, atrium_void are AREAS: size = [extent along x, extent along y, thickness], facing_deg 0.
- Walls, curtains, railings: give LENGTH as width, thickness as depth, facing_deg = the direction their visible face
  points (a wall on the camera's left side of a hallway along x faces -90; one across the far end faces 180).
- curtain = floor-to-ceiling pleated drape (we model the folds); glass_railing = glass panels with a metal handrail
  (rgb = glass tint, accent = metal); arch = a big arched opening in a wall (rgb = wall, accent = stone/brick trim);
  arched_window / window = framed glazing (rgb = glass tint, accent = frame); led_screen = video wall (rgb = frame,
  accent = the bright screen colour); truss = black box-truss tower/beam; speaker_stand = tripod + speaker;
  potted_plant (rgb = leaves, accent = pot); sign_stand (rgb = board, accent = text); doorway = open frame
  (accent = frame colour); exit_sign (rgb = glowing green/red); floor_patch = irregular lighter/darker patches of a
  patchy floor (decals, 1-3 m); floor_line = thin floor stripe/tape; atrium_void = the lower floor of an atrium /
  drop seen beyond a railing (base_z negative); ceiling = ceiling slab (accent = exposed beams); light_fixture =
  pendant/panel light; pipe = duct/pipe along its width; person = people in the shot (rgb = shirt, accent = pants).
- Cover the WHOLE place the video shows, including the area revealed by the pan: floor slab(s) for the hallway AND
  the open area, ceiling, every wall/curtain segment, the railing, the arch(es), the atrium beyond the railing (a
  lower floor + the far arched windows/walls you can see, collide=false), doorways with exit signs at the far end,
  plants, the sign, people, LED screen, truss, speaker stands, lights, and 3-8 floor patches that capture the
  patchy concrete. 25-70 objects. Far scenery outside the reachable area: collide = false.
- Colours: clean stylised versions of the real colours (sample them by eye), not grey defaults.
- Size checks: compare real sizes (door ~2.05 m, railing ~1.05 m, people ~1.7 m) with the same thing on our grid.
"""


def argon_layout(wdir, media, meas, info):
    log = log_path(wdir)
    prompt = LAYOUT_PROMPT.format(cam_h=info["cam_h"], ref_time=info["ref_time"], pan=round(info["pan_deg"]),
                                  meas=json.dumps(meas), cats=", ".join(AS.CATEGORIES))
    imgs = [("Image 2: reference frame + world grid", info["grid_ref"]),
            ("Image 3: mid-pan frame + world grid", info["grid_mid"]),
            ("Image 4: end-of-pan frame + world grid", info["grid_end"]),
            ("Image 5: plan view of measured points", info["plan"])]
    ps = [(t, A.image_part(p)[0]) for t, p in imgs]

    def parts(vp):
        out = [{"text": "Video 1: the site."}, vp]
        for t, p in ps:
            out += [{"text": t}, p]
        return out + [{"text": prompt}]
    ans, _ = A.generate_with_video(media, 4.0, parts, layout_schema(), "world_layout", log, A.files_cache_path(log))
    write_json(os.path.join(wdir, "logs", "argon_layout.json"), ans)
    return ans


# ----------------------------------------------------------------------------- build
def slugify(path):
    return re.sub(r"[^a-z0-9]+", "-", os.path.splitext(os.path.basename(path))[0].lower()).strip("-") or "world"


def measure_stage(media, wdir, long_fov):
    """Frames + depth measurement + grids/plan for Argon. Returns (rec, meas, info, scale)."""
    rec = R.reconstruct(media, wdir, long_fov)
    Pw, depth, panP = rec.pop("_Pw"), rec.pop("_depth"), rec.pop("_pan_P")
    cam_raw = rec["camera_height_raw_m"]
    s = PHONE_HEIGHT_PRIOR / cam_raw
    Pw, panP = Pw * s, panP * s
    cam_h = PHONE_HEIGHT_PRIOR
    for pf in rec["per_frame"]:
        pf["pos"] = [0.0, 0.0, cam_h]
    Rm = np.array(rec["R_world_from_cam"])
    meas = R.measure(Pw, depth)
    vdir = os.path.join(wdir, "views")
    plan = os.path.join(vdir, "argon_plan_view.png")
    R.plan_view(Pw, depth, plan, extra=panP)
    K = rec["K"]
    info = {"cam_h": cam_h, "ref_time": rec["frames"]["ref_time"], "plan": plan,
            "pan_deg": rec["per_frame"][-1]["yaw_deg"] - rec["per_frame"][rec["ref_index"]]["yaw_deg"]}
    info["grid_ref"] = os.path.join(vdir, "argon_grid_ref.png")
    R.grid_overlay(os.path.join(wdir, rec["frames"]["ref"]), Rm, cam_h, K, info["grid_ref"], title="reference frame")
    n = len(rec["per_frame"])
    for key, idx in (("grid_mid", (rec["ref_index"] + n - 1) // 2 + 2), ("grid_end", n - 1)):
        pf = rec["per_frame"][min(idx, n - 1)]
        info[key] = os.path.join(vdir, f"argon_{key}.png")
        R.grid_overlay(os.path.join(wdir, pf["file"]), np.array(pf["R_world_from_cam"]), cam_h, K, info[key],
                       title=f"t={pf['t']}s, camera yaw {pf['yaw_deg']} deg")
    np.save(os.path.join(wdir, "mesh", "measure_points.npy"),
            np.concatenate([Pw.reshape(-1, 3)[::7], panP[::5]]).astype(np.float32))
    return rec, meas, info, s


def build(media, out="worlds", rounds=2, target=80.0, long_fov=R.DEFAULT_LONG_FOV, layout_json=None):
    t0 = time.time()
    slug = slugify(media)
    wdir = os.path.abspath(os.path.join(out, f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    for sub in ("frames", "mesh", "views", "verify", "logs"):
        os.makedirs(os.path.join(wdir, sub), exist_ok=True)
    say(f"building {wdir}")
    rec, meas, info, s = measure_stage(media, wdir, long_fov)
    say(f"measurement done in {rec['seconds']} s; grid pre-scaled x{s:.3f} (phone at {PHONE_HEIGHT_PRIOR} m); {meas}")
    if layout_json:
        ans = read_json(layout_json)
        write_json(os.path.join(wdir, "logs", "argon_layout.json"), ans)
        say(f"layout from {layout_json} (no Argon call)")
    else:
        ans = argon_layout(wdir, media, meas, info)
    objects = unique_ids([to_object(o) for o in ans.get("objects") or []])
    fx, fy, cx, cy = rec["K"]
    W, _ = rec["depth_size"]
    sx = rec["frames"]["width"] / W
    cam_h = info["cam_h"]
    intr = {"model": "pinhole", "width": rec["frames"]["width"], "height": rec["frames"]["height"],
            "fx": round(fx * sx, 2), "fy": round(fy * sx, 2), "cx": round(cx * sx, 2), "cy": round(cy * sx, 2),
            "fovy_deg": rec["fovy_deg"], "long_fov_deg": long_fov, "source": "iPhone 24 mm-eq. video prior"}
    ref_pose = R.camera_pose(np.array(rec["R_world_from_cam"]), cam_h)
    per_frame = [{k: pf[k] for k in ("file", "t", "pos", "xyaxes", "forward", "pitch_deg", "yaw_deg")}
                 for pf in rec["per_frame"]]
    floor = ans.get("floor") or {}
    world = {
        "version": 2, "slug": slug, "created": datetime.now().isoformat(timespec="seconds"),
        "style": "game-style replica: procedural low-poly assets, stylised materials, no photo textures",
        "source_media": os.path.abspath(media), "frames": rec["frames"],
        "frame": {"units": "m", "up": "+z", "floor_z": 0.0, "origin": "floor point below the phone",
                  "x": "camera forward at the reference frame, projected on the floor (down the hallway)",
                  "y": "camera left (the pan turns towards +y)", "z": "up"},
        "camera": {"intrinsics": intr,
                   "ref": {**ref_pose, "fovy_deg": rec["fovy_deg"], "time_s": rec["frames"]["ref_time"]},
                   "end": {**per_frame[-1], "fovy_deg": rec["fovy_deg"]}, "per_frame": per_frame,
                   "per_frame_note": "rotation-only pan tracking (ORB homographies); position assumed fixed"},
        "scale": {"method": f"phone height prior {PHONE_HEIGHT_PRIOR} m (grid pre-scaled x{s:.3f} from monocular depth)",
                  "factor": round(s, 4), "checks": ans.get("scale_checks"),
                  "argon_camera_height_m": ans.get("camera_height_m")},
        "measurement": {"method": rec["method"], "values": meas, "floor_inlier_frac": rec["floor_inlier_frac"],
                        "pan_frames": rec["pan_frames"],
                        "note": "depth is used only to measure layout; never as geometry or texture"},
        "floor": {"z": 0.0, "material": "concrete" if floor.get("look") not in ("wood", "stone") else floor["look"],
                  "look": floor.get("look", "concrete"), "rgb": floor.get("rgb") or [0.5, 0.48, 0.45],
                  "friction": float(floor.get("friction") or 0.8)},
        "objects": objects, "argon_notes": ans.get("notes"), "edits": [], "verify": [],
    }
    save_world(world, wdir)
    world["views"] = {"r0": render_views(world, wdir, "r0")}
    write_json(os.path.join(wdir, "world.json"), world)
    say(f"world r0: {len(world['objects'])} objects, {sum(len(o['collision']) for o in world['objects'])} primitives, "
        f"{world['mesh']['triangles']} triangles")
    from fallguy.world import verify as V
    for rnd in range(1, rounds + 1):
        try:
            res = V.verify(wdir, rnd, apply=True)
        except A.ArgonError as ex:          # keep the world; the director can run `verify` again
            say(f"verify round {rnd} failed: {A.redact(str(ex))}; keeping the current world")
            break
        if res["score"] >= target:
            say(f"verify round {rnd}: score {res['score']} >= {target}; stopping")
            break
    world = read_json(os.path.join(wdir, "world.json"))
    world["build_seconds"] = round(time.time() - t0, 1)
    write_json(os.path.join(wdir, "world.json"), world)
    return wdir, world
