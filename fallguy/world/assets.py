"""Procedural, game-style asset library for FALLGUY worlds.

Every object in world.json is turned into a clean low-poly model with stylised flat materials (no photo textures):
`build_object(obj) -> {"parts": [(look_key, trimesh.Trimesh)], "collision": [prim, ...]}` in WORLD coordinates.

Object convention (what Argon returns, metres, world frame z-up):
  center [x, y]   footprint centre on the floor plane
  base_z          height of the object's bottom (0 = on the floor; negative = below the floor, e.g. an atrium)
  facing_deg      the direction the object's FRONT faces in the x-y plane (0 = +x, 90 = +y, 180 = -x, -90 = -y)
  size [w, d, h]  width (side to side, seen from the front), depth (front to back), height
  rgb, accent     main and secondary colour; material = look (paint, fabric, glass, metal, ...)
Local builder frame: +x = front direction (depth axis), +y = the object's left (width axis), z up from base_z.
Unknown categories use `parts` (primitives in that local frame) or a rounded block.
"""
import math

import numpy as np
import trimesh

CATEGORIES = ["floor", "floor_patch", "floor_line", "ceiling", "wall", "curtain", "glass_railing", "railing",
              "doorway", "door", "window", "arched_window", "arch", "column", "beam", "pipe", "light_fixture",
              "led_screen", "tv_screen", "speaker_stand", "truss", "potted_plant", "sign_stand", "wall_sign",
              "exit_sign", "bench", "chair", "table", "counter", "bin", "box", "person", "stairs", "crash_mat",
              "atrium_void", "other"]
LOOKS = ["paint", "concrete", "fabric", "glass", "metal", "wood", "plastic", "stone", "ceramic", "foliage", "screen",
         "foam", "skin", "emissive"]
PHYSICS = {"paint": "concrete", "concrete": "concrete", "fabric": "fabric", "glass": "glass", "metal": "metal",
           "wood": "wood", "plastic": "plastic", "stone": "stone", "ceramic": "stone", "foliage": "fabric",
           "screen": "glass", "foam": "foam", "skin": "skin", "emissive": "plastic"}
# render properties per look: roughness, metallic, alpha, emission (0-1), mujoco specular / shininess / reflectance
LOOK_PROPS = {
    "paint": (0.85, 0.0, 1.0, 0.0), "concrete": (0.9, 0.0, 1.0, 0.0), "fabric": (0.95, 0.0, 1.0, 0.0),
    "glass": (0.05, 0.0, 0.28, 0.0), "metal": (0.35, 0.8, 1.0, 0.0), "wood": (0.7, 0.0, 1.0, 0.0),
    "plastic": (0.5, 0.0, 1.0, 0.0), "stone": (0.8, 0.0, 1.0, 0.0), "ceramic": (0.4, 0.0, 1.0, 0.0),
    "foliage": (0.8, 0.0, 1.0, 0.0), "screen": (0.2, 0.0, 1.0, 0.85), "foam": (0.95, 0.0, 1.0, 0.0),
    "skin": (0.7, 0.0, 1.0, 0.0), "emissive": (0.5, 0.0, 1.0, 1.0)}
NO_COLLIDE = {"floor", "floor_patch", "floor_line", "light_fixture", "exit_sign", "wall_sign", "atrium_void"}


# ----------------------------------------------------------------------------- tiny geometry kit (local frame)
def box(c, e):
    m = trimesh.creation.box(extents=np.maximum(np.asarray(e, float), 1e-3))
    m.apply_translation(c)
    return m


def cyl(p0, p1, r, sections=14):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if np.linalg.norm(p1 - p0) < 1e-4:
        p1 = p0 + [0, 0, 1e-3]
    return trimesh.creation.cylinder(radius=max(r, 1e-3), segment=[p0, p1], sections=sections)


def frustum(c, r0, r1, h, sections=16):
    """Tapered cylinder standing on c (bottom radius r0, top r1)."""
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], 1)
    V = np.concatenate([np.c_[ring * r0, np.zeros(sections)], np.c_[ring * r1, np.full(sections, h)],
                        [[0, 0, 0], [0, 0, h]]])
    F = []
    for i in range(sections):
        j = (i + 1) % sections
        F += [[i, j, sections + j], [i, sections + j, sections + i],
              [2 * sections, j, i], [2 * sections + 1, sections + i, sections + j]]
    m = trimesh.Trimesh(V, F, process=False)
    m.apply_translation(c)
    return m


def blob(c, r, sub=1, scale=(1, 1, 1), seed=0, jitter=0.18):
    """Faceted low-poly blob (flat shaded), e.g. foliage."""
    m = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    rng = np.random.default_rng(seed)
    m.vertices *= (1 + jitter * (rng.random((len(m.vertices), 1)) - 0.5))
    m.vertices *= np.asarray(scale, float)
    m.apply_translation(c)
    return m


def strip(A, B, w0, w1, axes=(0, 1, 2)):
    """Closed solid between two matching 2-D polylines A, B (N x 2), extruded from w0 to w1 along the third axis.
    axes maps (u, v, w) -> world axis indices."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = len(A)
    V = np.zeros((4 * n, 3))
    for k, (P, w) in enumerate([(A, w0), (B, w0), (A, w1), (B, w1)]):
        V[k * n:(k + 1) * n, axes[0]] = P[:, 0]
        V[k * n:(k + 1) * n, axes[1]] = P[:, 1]
        V[k * n:(k + 1) * n, axes[2]] = w
    a0, b0, a1, b1 = 0, n, 2 * n, 3 * n
    F = []
    for i in range(n - 1):
        j = i + 1
        F += [[a0 + i, b0 + i, b0 + j], [a0 + i, b0 + j, a0 + j]]          # cap w0
        F += [[a1 + i, b1 + j, b1 + i], [a1 + i, a1 + j, b1 + j]]          # cap w1
        F += [[a0 + i, a0 + j, a1 + j], [a0 + i, a1 + j, a1 + i]]          # side A
        F += [[b0 + i, b1 + j, b0 + j], [b0 + i, b1 + i, b1 + j]]          # side B
    for i in (0, n - 1):
        F += [[a0 + i, a1 + i, b1 + i], [a0 + i, b1 + i, b0 + i]]
    m = trimesh.Trimesh(V, F, process=True)
    trimesh.repair.fix_normals(m)
    return m


def polyslab(pts, z0, z1):
    """Vertical prism over a star-shaped 2-D polygon (fan triangulated from its centroid)."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y) < 0:      # make it counter-clockwise
        pts = pts[::-1]
    n = len(pts)
    c = pts.mean(0)
    V = np.concatenate([np.c_[pts, np.full(n, z0)], np.c_[pts, np.full(n, z1)], [[*c, z0], [*c, z1]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[2 * n, j, i], [2 * n + 1, n + i, n + j], [i, j, n + j], [i, n + j, n + i]]
    return trimesh.Trimesh(V, F, process=True)


def arc_pts(cx, cz, r, a0, a1, n=24):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cz + r * np.sin(a)], 1)


def arch_profile(w, h, n=24):
    """Inner outline of an arched opening (x across, z up): straight sides then a semicircle (or flatter arc)."""
    r = w / 2
    spring = max(0.0, h - r)
    if h < r:                                     # segmental arch
        r = (r * r + h * h) / (2 * h)
        a = math.asin(min(1.0, (w / 2) / r))
        return arc_pts(0, h - r, r, math.pi / 2 + a, math.pi / 2 - a, n), 0.0
    return np.concatenate([[[-w / 2, 0]], arc_pts(0, spring, r, math.pi, 0, n), [[w / 2, 0]]]), spring


def mix(c, k):
    c = np.asarray(c, float)
    return np.clip(c * k if k < 1 else c + (1 - c) * (k - 1), 0, 1).round(3).tolist()


# ----------------------------------------------------------------------------- category builders
# each returns (parts [(look, rgb, mesh)], collision [(type, center_local, full_dims_local)])
AREA = {"floor", "ceiling", "atrium_void"}       # size = [extent along x, extent along y, thickness] at facing 0


def b_floor(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, -0.05], [W, D, 0.1]))], []


def b_floor_patch(W, D, H, o):
    rng = np.random.default_rng(abs(hash(o["id"])) % 2 ** 32)
    a = np.linspace(0, 2 * np.pi, 22, endpoint=False)
    r = 0.5 * (1 + 0.28 * (rng.random(22) - 0.5)) * (1 + 0.15 * np.sin(3 * a + rng.random() * 6))
    pts = np.stack([np.cos(a) * r * D, np.sin(a) * r * W], 1)
    return [(o["look"], o["rgb"], polyslab(pts, 0.0, 0.004 + 0.002 * (rng.random())))], []


def b_floor_line(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, 0.003], [max(D, 0.03), W, 0.006]))], []


def b_ceiling(W, D, H, o):
    t = max(0.08, min(H, 0.4))
    W, D = D, W                                    # area convention: W = extent along local x
    parts = [(o["look"], o["rgb"], box([0, 0, t / 2], [D, W, t]))]
    # exposed beams under the slab every ~2 m along the longer axis
    acc = o["accent"]
    if W >= D:
        for y in np.arange(-W / 2 + 1.0, W / 2, 2.0):
            parts.append(("metal", acc, box([0, y, -0.12], [D, 0.12, 0.24])))
    else:
        for x in np.arange(-D / 2 + 1.0, D / 2, 2.0):
            parts.append(("metal", acc, box([x, 0, -0.12], [0.12, W, 0.24])))
    return parts, [("box", [0, 0, t / 2], [D, W, t])]


def b_wall(W, D, H, o):
    D = max(D, 0.08)
    parts = [(o["look"], o["rgb"], box([0, 0, H / 2], [D, W, H]))]
    parts.append(("paint", mix(o["accent"], 0.9), box([0, 0, 0.05], [D + 0.024, W, 0.1])))      # skirting
    return parts, [("box", [0, 0, H / 2], [D, W, H])]


def b_curtain(W, D, H, o):
    n = max(8, int(W / 0.03))
    y = np.linspace(-W / 2, W / 2, n)
    lam = 0.22
    amp = max(0.03, min(D / 2 - 0.01, 0.07))
    x = amp * np.sin(2 * np.pi * (y + W / 2) / lam)
    t = 0.012
    A = np.stack([x - t / 2, y], 1)
    B = np.stack([x + t / 2, y], 1)
    hem = 0.015
    parts = [(o["look"], o["rgb"], strip(A, B, hem, H - 0.06, axes=(0, 1, 2)))]
    parts.append(("metal", o["accent"], box([0, 0, H - 0.03], [0.05, W + 0.04, 0.05])))           # track
    return parts, [("box", [0, 0, H / 2], [max(D, 2 * amp + 0.02), W, H])]


def b_glass_railing(W, D, H, o):
    H = H or 1.05
    acc = o["accent"]
    n = max(1, int(math.ceil(W / 1.4)))
    pw = W / n
    parts = [("metal", mix(acc, 0.7), box([0, 0, 0.06], [0.08, W, 0.12]))]                         # base shoe
    for i in range(n):
        yc = -W / 2 + (i + 0.5) * pw
        parts.append(("glass", o["rgb"], box([0, yc, (0.12 + H - 0.04) / 2], [0.016, pw - 0.025, H - 0.16])))
    parts.append(("metal", acc, cyl([0, -W / 2, H], [0, W / 2, H], 0.03)))                          # handrail
    return parts, [("box", [0, 0, H / 2], [max(D, 0.08), W, H])]


def b_railing(W, D, H, o):
    H = H or 1.0
    acc = o["rgb"]
    parts = [("metal", acc, cyl([0, -W / 2, H], [0, W / 2, H], 0.025)),
             ("metal", acc, cyl([0, -W / 2, H * 0.5], [0, W / 2, H * 0.5], 0.015))]
    for y in np.linspace(-W / 2, W / 2, max(2, int(W / 1.2) + 1)):
        parts.append(("metal", acc, cyl([0, y, 0], [0, y, H], 0.022)))
    return parts, [("box", [0, 0, H / 2], [max(D, 0.06), W, H])]


def _frame(W, H, D, fw, look, rgb, top=True):
    parts = [(look, rgb, box([0, -W / 2 + fw / 2, H / 2], [D, fw, H])),
             (look, rgb, box([0, W / 2 - fw / 2, H / 2], [D, fw, H]))]
    if top:
        parts.append((look, rgb, box([0, 0, H - fw / 2], [D, W, fw])))
    return parts


def b_doorway(W, D, H, o):
    D = max(D, 0.12)
    parts = _frame(W, H, D + 0.03, 0.08, "paint", o["accent"])
    parts.append(("paint", [0.12, 0.12, 0.14], box([-D / 2 - 0.02, 0, (H - 0.08) / 2], [0.02, W - 0.16, H - 0.08])))
    col = [("box", [0, -W / 2 + 0.04, H / 2], [D, 0.08, H]), ("box", [0, W / 2 - 0.04, H / 2], [D, 0.08, H]),
           ("box", [0, 0, H - 0.04], [D, W, 0.08])]
    return parts, col


def b_door(W, D, H, o):
    parts = _frame(W, H, max(D, 0.1) + 0.03, 0.07, "paint", o["accent"])
    parts.append((o["look"], o["rgb"], box([0, 0, (H - 0.07) / 2], [0.05, W - 0.14, H - 0.07])))
    parts.append(("metal", [0.75, 0.75, 0.78], box([0.05, W / 2 - 0.2, 1.0], [0.04, 0.14, 0.03])))
    return parts, [("box", [0, 0, H / 2], [max(D, 0.1), W, H])]


def b_window(W, D, H, o, arched=False):
    fw = 0.07
    D = max(D, 0.06)
    acc = o["accent"]
    parts = []
    if arched:
        prof, spring = arch_profile(W - 2 * fw, H - fw)
        outer, _ = arch_profile(W, H)
        inner = prof + [0, fw]
        n = min(len(inner), len(outer))
        parts.append(("metal", acc, strip(outer[:n] * [1, 1], inner[:n], -D / 2, D / 2, axes=(1, 2, 0))))
        parts.append(("metal", acc, box([0, 0, fw / 2], [D, W, fw])))
        parts.append(("glass", o["rgb"], polyslab_xz(inner, -0.006, 0.006)))
        for y in np.linspace(-W / 2, W / 2, max(3, int(W / 0.6)))[1:-1]:
            top = np.interp(y, inner[:, 0], inner[:, 1]) if spring else H - fw
            parts.append(("metal", acc, box([0, y, top / 2], [D * 0.6, 0.03, top])))
        for z in np.arange(0.6, (spring or H) - 0.1, 0.6):
            parts.append(("metal", acc, box([0, 0, z], [D * 0.6, W - 2 * fw, 0.03])))
    else:
        parts += _frame(W, H, D, fw, "metal", acc)
        parts.append(("metal", acc, box([0, 0, fw / 2], [D, W, fw])))
        parts.append(("glass", o["rgb"], box([0, 0, H / 2], [0.012, W - 2 * fw, H - 2 * fw])))
        for y in np.linspace(-W / 2, W / 2, max(2, int(W / 0.9) + 1))[1:-1]:
            parts.append(("metal", acc, box([0, y, H / 2], [D * 0.6, 0.03, H])))
    return parts, [("box", [0, 0, H / 2], [D, W, H])]


def polyslab_xz(prof, y0, y1):
    """A flat panel in the x-z plane (local y-z for the builder: u = width, v = height) from an outline."""
    pts = np.asarray(prof, float)
    c = pts.mean(0)
    n = len(pts)
    V = []
    for d in (y0, y1):
        V += [[d, p[0], p[1]] for p in pts] + [[d, c[0], c[1]]]
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[n, i, j], [2 * n + 1, n + 1 + j, n + 1 + i], [i, n + 1 + i, n + 1 + j], [i, n + 1 + j, j]]
    m = trimesh.Trimesh(np.array(V), F, process=True)
    trimesh.repair.fix_normals(m)
    return m


def b_arch(W, D, H, o):
    """An arched opening in a wall: two piers and an arch band, with an accent (stone) trim on the inner edge."""
    D = max(D, 0.2)
    pier = max(0.25, min(0.8, W * 0.12))
    w = W - 2 * pier
    r = w / 2
    spring = H - pier - r
    if spring < 0:                                  # segmental arch, centre below the springing line
        h = H - pier
        r = (r * r + h * h) / (2 * h)
        cz = h - r
        a0 = math.asin(min(1.0, (w / 2) / r))
        ang = np.linspace(math.pi / 2 + a0, math.pi / 2 - a0, 28)
        spring = 0.0
    else:
        cz = spring
        ang = np.linspace(math.pi, 0, 28)
    inner = np.stack([r * np.cos(ang), cz + r * np.sin(ang)], 1)
    outer = []
    for a in ang:
        dx, dz = math.cos(a), math.sin(a)
        t = min((W / 2) / max(abs(dx), 1e-6), (H - cz) / max(dz, 1e-6))
        outer.append([dx * t, cz + dz * t])
    outer = np.array(outer)
    outer[0] = [-W / 2, inner[0, 1]]
    outer[-1] = [W / 2, inner[-1, 1]]
    parts = [(o["look"], o["rgb"], strip(outer, inner, -D / 2, D / 2, axes=(1, 2, 0))),
             (o["look"], o["rgb"], box([0, -W / 2 + pier / 2, inner[0, 1] / 2], [D, pier, max(inner[0, 1], 0.01)])),
             (o["look"], o["rgb"], box([0, W / 2 - pier / 2, inner[0, 1] / 2], [D, pier, max(inner[0, 1], 0.01)]))]
    tw = 0.14
    t_in = inner
    t_out = np.stack([(r + tw) * np.cos(ang), cz + (r + tw) * np.sin(ang)], 1) if spring > 0 else inner + [0, tw]
    for sgn in (-1, 1):
        parts.append((o["look2"], o["accent"], strip(t_out, t_in, sgn * D / 2, sgn * (D / 2 + 0.03), axes=(1, 2, 0))))
        if spring > 0:
            for side in (-1, 1):
                parts.append((o["look2"], o["accent"], box([sgn * (D / 2 + 0.015), side * (r + tw / 2), spring / 2],
                                                           [0.03, tw, spring])))
    col = [("box", [0, -W / 2 + pier / 2, H / 2], [D, pier, H]), ("box", [0, W / 2 - pier / 2, H / 2], [D, pier, H]),
           ("box", [0, 0, H - pier / 2], [D, W, pier])]
    return parts, col


def b_column(W, D, H, o):
    if "round" in o["label"].lower() or o["look"] in ("metal",):
        return ([(o["look"], o["rgb"], cyl([0, 0, 0], [0, 0, H], max(W, D) / 2, 20))],
                [("cylinder", [0, 0, H / 2], [max(W, D), max(W, D), H])])
    return [(o["look"], o["rgb"], box([0, 0, H / 2], [D, W, H]))], [("box", [0, 0, H / 2], [D, W, H])]


def b_beam(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, H / 2], [D, W, H]))], [("box", [0, 0, H / 2], [D, W, H])]


def b_pipe(W, D, H, o):
    r = max(0.02, min(D, H) / 2)
    parts = [(o["look"], o["rgb"], cyl([0, -W / 2, r], [0, W / 2, r], r, 16))]
    for y in np.arange(-W / 2 + 0.5, W / 2, 1.5):
        parts.append(("metal", mix(o["rgb"], 0.7), cyl([0, y, r - 0.01], [0, y + 0.06, r - 0.01], r * 1.15, 16)))
    return parts, [("box", [0, 0, r], [2 * r, W, 2 * r])]


def b_light_fixture(W, D, H, o):
    H = max(H, 0.1)
    parts = [("metal", [0.15, 0.15, 0.16], cyl([0, 0, H * 0.3], [0, 0, H], 0.006, 6))]
    if W > 2 * max(D, 0.3):               # linear / panel light
        parts.append(("emissive", o["rgb"], box([0, 0, H * 0.3], [max(D, 0.12), W, 0.05])))
    else:
        parts.append(("emissive", o["rgb"], blob([0, 0, H * 0.3], max(W, D) / 2, 1, (1, 1, 0.6), jitter=0)))
    return parts, []


def b_screen(W, D, H, o, legs=False):
    D = max(D, 0.08)
    z0 = 0.0 if H < 1.5 or not legs else 0.0
    parts = [("metal", [0.05, 0.05, 0.06], box([0, 0, H / 2], [D, W, H]))]
    bez = 0.04
    parts.append(("screen", o["accent"], box([D / 2 + 0.004, 0, H / 2], [0.008, W - 2 * bez, H - 2 * bez])))
    # a soft gradient band so the panel reads as a lit image, not a flat card
    parts.append(("screen", mix(o["accent"], 1.35), box([D / 2 + 0.009, 0, H * 0.62],
                                                         [0.004, (W - 2 * bez) * 0.7, (H - 2 * bez) * 0.22])))
    return parts, [("box", [0, 0, H / 2 + z0], [D, W, H])]


def b_speaker_stand(W, D, H, o):
    r = max(0.35, min(W, D) / 2)
    blk = o["rgb"]
    hub = 0.75
    parts = []
    for k in range(3):
        a = 2 * np.pi * k / 3 + 0.3
        parts.append(("metal", blk, cyl([r * math.cos(a), r * math.sin(a), 0.02], [0, 0, hub], 0.014, 8)))
    sh = min(0.6, H * 0.32)
    top = H - sh
    parts.append(("metal", blk, cyl([0, 0, 0.3], [0, 0, top], 0.022, 10)))
    sw, sd = 0.36, 0.32
    parts.append(("plastic", mix(blk, 1.15), box([0, 0, top + sh / 2], [sd, sw, sh])))
    parts.append(("fabric", mix(blk, 0.6), box([sd / 2 + 0.003, 0, top + sh / 2], [0.006, sw - 0.05, sh - 0.06])))
    parts.append(("metal", o["accent"], cyl([sd / 2 + 0.008, 0, top + sh * 0.35], [sd / 2 + 0.012, 0, top + sh * 0.35],
                                            sw * 0.32, 18)))
    return parts, [("cylinder", [0, 0, 0.4], [2 * r * 0.8, 2 * r * 0.8, 0.8]),
                   ("box", [0, 0, (0.8 + H) / 2], [sd, sw, H - 0.8])]


def b_truss(W, D, H, o):
    dims = np.array([D, W, H], float)
    ax = int(np.argmax(dims))
    L = dims[ax]
    s = max(0.25, min(0.45, sorted(dims)[1]))     # square section side
    rc = 0.022
    blk = o["rgb"]
    other = [i for i in range(3) if i != ax]
    corners = [(-s / 2, -s / 2), (s / 2, -s / 2), (s / 2, s / 2), (-s / 2, s / 2)]
    parts = []

    def P(t, u, v):
        p = np.zeros(3)
        p[ax] = t
        p[other[0]], p[other[1]] = u, v
        if ax != 2:
            p[2] += H / 2
        return p
    t0, t1 = (0, L) if ax == 2 else (-L / 2, L / 2)
    for (u, v) in corners:
        parts.append(("metal", blk, cyl(P(t0, u, v), P(t1, u, v), rc, 8)))
    step = s
    ts = np.arange(t0, t1 - 1e-6, step)
    for k in range(4):
        (u0, v0), (u1, v1) = corners[k], corners[(k + 1) % 4]
        for i, t in enumerate(ts):
            tn = min(t + step, t1)
            a, b = ((u0, v0), (u1, v1)) if i % 2 == 0 else ((u1, v1), (u0, v0))
            parts.append(("metal", blk, cyl(P(t, *a), P(tn, *b), rc * 0.55, 6)))
    e = np.zeros(3)
    e[ax] = L
    e[other[0]] = e[other[1]] = s
    c = np.array([0, 0, H / 2 if ax != 2 else L / 2])
    if ax == 2:
        parts.append(("metal", blk, box([0, 0, 0.01], [s + 0.15, s + 0.15, 0.02])))
    return parts, [("box", c.tolist(), [e[0], e[1], e[2]])]


def b_potted_plant(W, D, H, o):
    r = max(0.12, min(W, D) / 2 * 0.65)
    ph = min(0.55, max(0.25, H * 0.3))
    parts = [(o["look2"], o["accent"], frustum([0, 0, 0], r * 0.78, r, ph, 18)),
             ("concrete", [0.22, 0.17, 0.12], frustum([0, 0, ph - 0.03], r * 0.95, r * 0.95, 0.03, 18))]
    parts.append(("wood", [0.36, 0.27, 0.18], cyl([0, 0, ph], [0, 0, H * 0.75], 0.025, 6)))
    rng = np.random.default_rng(abs(hash(o["id"])) % 2 ** 32)
    fr = max(W, D) / 2
    green = np.asarray(o["rgb"], float)
    for k in range(16):
        f = k / 15
        z = ph + (H - ph) * (0.3 + 0.62 * f) + 0.05 * rng.random()
        a = rng.random() * 2 * np.pi
        spread = fr * (0.35 + 0.65 * math.sin(math.pi * min(1, 0.25 + f)))
        d = spread * (0.4 + 0.6 * rng.random())
        rr = fr * (0.22 + 0.12 * rng.random())
        col = mix(green, 0.7 + 0.6 * rng.random())
        parts.append(("foliage", col, blob([d * math.cos(a), d * math.sin(a), min(z, H - rr * 0.5)], rr, 1,
                                           (1.35, 0.8, 0.55), seed=k, jitter=0.3)))
        parts[-1][2].apply_transform(trimesh.transformations.rotation_matrix(a, [0, 0, 1],
                                                                             [d * math.cos(a), d * math.sin(a), z]))
    return parts, [("cylinder", [0, 0, H / 2], [2 * r, 2 * r, H])]


def b_sign_stand(W, D, H, o):
    bw = max(0.3, min(W, 0.6))
    parts = [("metal", [0.82, 0.83, 0.85], box([0, 0, 0.01], [max(D, 0.35), bw, 0.02]))]
    bh = min(0.7, H * 0.45)
    bwid = max(0.3, W * 0.9)
    parts.append(("metal", [0.7, 0.71, 0.73], cyl([0, 0, 0.02], [0, 0, H - bh * 0.6], 0.02, 10)))
    board = box([0, 0, 0], [0.025, bwid, bh])
    board.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-12), [0, 1, 0]))
    board.apply_translation([0, 0, H - bh / 2])
    parts.append(("plastic", o["rgb"], board))
    for k, (zz, ww) in enumerate([(0.25, 0.55), (0.05, 0.75), (-0.08, 0.6)]):
        bar = box([0, 0, 0], [0.004, bwid * ww, bh * (0.1 if k == 0 else 0.04)])
        bar.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-12), [0, 1, 0]))
        bar.apply_translation([0.016 + bh * zz * math.sin(math.radians(12)), 0, H - bh / 2 + bh * zz])
        parts.append(("paint", o["accent"], bar))
    return parts, [("box", [0, 0, 0.05], [max(D, 0.35), bw, 0.1]), ("box", [0, 0, H / 2], [0.1, 0.1, H]),
                   ("box", [0, 0, H - bh / 2], [0.12, bwid, bh])]


def b_wall_sign(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, H / 2], [max(D, 0.02), W, H])),
            ("paint", o["accent"], box([max(D, 0.02) / 2 + 0.002, 0, H / 2], [0.004, W * 0.7, H * 0.25]))], []


def b_exit_sign(W, D, H, o):
    W, H = W or 0.35, H or 0.18
    return [("plastic", [0.9, 0.9, 0.9], box([0, 0, H / 2], [max(D, 0.06), W, H])),
            ("emissive", o["rgb"], box([max(D, 0.06) / 2 + 0.003, 0, H / 2], [0.006, W * 0.85, H * 0.7]))], []


def b_bench(W, D, H, o):
    H = H or 0.45
    parts = [(o["look"], o["rgb"], box([0, 0, H - 0.03], [D, W, 0.06]))]
    for y in (-W / 2 + 0.1, W / 2 - 0.1):
        parts.append(("metal", o["accent"], box([0, y, (H - 0.06) / 2], [D * 0.85, 0.05, H - 0.06])))
    return parts, [("box", [0, 0, H / 2], [D, W, H])]


def b_table(W, D, H, o):
    parts = [(o["look"], o["rgb"], box([0, 0, H - 0.025], [D, W, 0.05]))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(("metal", o["accent"], cyl([sx * (D / 2 - 0.06), sy * (W / 2 - 0.06), 0],
                                                    [sx * (D / 2 - 0.06), sy * (W / 2 - 0.06), H - 0.05], 0.02, 8)))
    return parts, [("box", [0, 0, H - 0.025], [D, W, 0.05]), ("box", [0, 0, (H - 0.05) / 2], [D - 0.1, W - 0.1, H - 0.05])]


def b_chair(W, D, H, o):
    sh = min(0.47, H * 0.5)
    parts = [(o["look"], o["rgb"], box([0, 0, sh], [D, W, 0.05])),
             (o["look"], o["rgb"], box([-D / 2 + 0.03, 0, (sh + H) / 2], [0.05, W, H - sh]))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(("metal", o["accent"], cyl([sx * (D / 2 - 0.04), sy * (W / 2 - 0.04), 0],
                                                    [sx * (D / 2 - 0.04), sy * (W / 2 - 0.04), sh], 0.015, 6)))
    return parts, [("box", [0, 0, H / 2], [D, W, H])]


def b_counter(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, (H - 0.04) / 2], [D - 0.04, W - 0.04, H - 0.04])),
            (o["look2"], o["accent"], box([0, 0, H - 0.02], [D, W, 0.04]))], [("box", [0, 0, H / 2], [D, W, H])]


def b_bin(W, D, H, o):
    r = max(W, D) / 2
    return [(o["look"], o["rgb"], frustum([0, 0, 0], r * 0.85, r, H, 16)),
            ("plastic", o["accent"], frustum([0, 0, H], r * 1.02, r * 0.9, 0.04, 16))], \
        [("cylinder", [0, 0, H / 2], [2 * r, 2 * r, H])]


def b_box(W, D, H, o):
    b = 0.02
    return [(o["look"], o["rgb"], box([0, 0, H / 2], [D, W, H])),
            (o["look"], mix(o["rgb"], 0.85), box([0, 0, H - b / 2], [D + 0.01, W + 0.01, b]))], \
        [("box", [0, 0, H / 2], [D, W, H])]


def b_crash_mat(W, D, H, o):
    H = H or 0.3
    parts = [("foam", o["rgb"], box([0, 0, H / 2], [D, W, H]))]
    for y in np.linspace(-W / 2, W / 2, 4)[1:-1]:
        parts.append(("fabric", mix(o["rgb"], 0.75), box([0, y, H / 2], [D + 0.006, 0.03, H + 0.006])))
    return parts, [("box", [0, 0, H / 2], [D, W, H])]


def b_person(W, D, H, o):
    H = H or 1.75
    k = H / 1.75
    shirt, pants = o["rgb"], o["accent"]
    skin = [0.78, 0.6, 0.48]
    parts = []
    for s in (-1, 1):
        parts.append(("fabric", pants, cyl([0, s * 0.1 * k, 0.05 * k], [0, s * 0.1 * k, 0.86 * k], 0.07 * k, 8)))
        parts.append(("plastic", [0.1, 0.1, 0.1], box([0.04 * k, s * 0.1 * k, 0.04 * k], [0.24 * k, 0.1 * k, 0.08 * k])))
        parts.append(("fabric", shirt, cyl([0, s * 0.24 * k, 1.42 * k], [0.03 * k, s * 0.27 * k, 0.9 * k], 0.045 * k, 8)))
        parts.append(("skin", skin, blob([0.03 * k, s * 0.27 * k, 0.86 * k], 0.045 * k, 1, jitter=0)))
    parts.append(("fabric", shirt, box([0, 0, 1.17 * k], [0.24 * k, 0.42 * k, 0.6 * k])))
    parts.append(("skin", skin, cyl([0, 0, 1.46 * k], [0, 0, 1.54 * k], 0.05 * k, 8)))
    parts.append(("skin", skin, blob([0, 0, 1.64 * k], 0.11 * k, 2, (1, 0.9, 1.1), jitter=0)))
    parts.append(("fabric", mix(pants, 0.5), blob([-0.015 * k, 0, 1.69 * k], 0.105 * k, 2, (1, 0.92, 0.75), jitter=0)))
    return parts, [("cylinder", [0, 0, H / 2], [0.5 * k, 0.5 * k, H])]


def b_stairs(W, D, H, o):
    n = max(2, int(round(H / 0.17)))
    parts = []
    for i in range(n):
        h = H * (i + 1) / n
        parts.append((o["look"], o["rgb"], box([-D / 2 + D * (i + 0.5) / n, 0, h / 2], [D / n, W, h])))
    return parts, [("box", [-D / 2 + D * (i + 0.5) / n, 0, H * (i + 1) / n / 2], [D / n, W, H * (i + 1) / n])
                   for i in range(n)]


def b_atrium_void(W, D, H, o):
    return [(o["look"], o["rgb"], box([0, 0, -0.1], [W, D, 0.2]))], []


def b_other(W, D, H, o):
    parts, col = [], []
    for p in o.get("parts") or []:
        c = np.asarray(p.get("center") or [0, 0, 0.5], float)
        d = np.maximum(np.abs(np.asarray(p.get("dims") or [0.3, 0.3, 0.3], float)), 0.01)
        rgb = p.get("rgb") or o["rgb"]
        look = p.get("material") if p.get("material") in LOOKS else o["look"]
        t = p.get("type", "box")
        if t == "cylinder":
            parts.append((look, rgb, cyl(c - [0, 0, d[2] / 2], c + [0, 0, d[2] / 2], max(d[0], d[1]) / 2, 16)))
            col.append(("cylinder", c.tolist(), [max(d[0], d[1])] * 2 + [d[2]]))
        elif t == "sphere":
            parts.append((look, rgb, blob(c, max(d) / 2, 2, d / max(d), jitter=0)))
            col.append(("sphere", c.tolist(), [max(d)] * 3))
        else:
            # local parts use the object convention: dims [w, d, h] -> builder frame (x = depth)
            parts.append((look, rgb, box(c, [d[1], d[0], d[2]])))
            col.append(("box", c.tolist(), [d[1], d[0], d[2]]))
    if not parts:
        parts = [(o["look"], o["rgb"], box([0, 0, H / 2], [D, W, H])),
                 (o["look"], mix(o["rgb"], 0.8), box([0, 0, 0.03], [D + 0.02, W + 0.02, 0.06]))]
        col = [("box", [0, 0, H / 2], [D, W, H])]
    return parts, col


BUILDERS = {"floor": b_floor, "floor_patch": b_floor_patch, "floor_line": b_floor_line, "ceiling": b_ceiling,
            "wall": b_wall, "curtain": b_curtain, "glass_railing": b_glass_railing, "railing": b_railing,
            "doorway": b_doorway, "door": b_door, "window": b_window,
            "arched_window": lambda W, D, H, o: b_window(W, D, H, o, arched=True), "arch": b_arch,
            "column": b_column, "beam": b_beam, "pipe": b_pipe, "light_fixture": b_light_fixture,
            "led_screen": lambda W, D, H, o: b_screen(W, D, H, o, legs=True), "tv_screen": b_screen,
            "speaker_stand": b_speaker_stand, "truss": b_truss, "potted_plant": b_potted_plant,
            "sign_stand": b_sign_stand, "wall_sign": b_wall_sign, "exit_sign": b_exit_sign, "bench": b_bench,
            "chair": b_chair, "table": b_table, "counter": b_counter, "bin": b_bin, "box": b_box,
            "person": b_person, "stairs": b_stairs, "crash_mat": b_crash_mat, "atrium_void": b_atrium_void,
            "other": b_other}
DEFAULT_ACCENT = {"wall": [0.55, 0.55, 0.55], "curtain": [0.6, 0.6, 0.62], "glass_railing": [0.75, 0.76, 0.78],
                  "doorway": [0.25, 0.25, 0.27], "door": [0.3, 0.3, 0.32], "arched_window": [0.25, 0.27, 0.3],
                  "window": [0.25, 0.27, 0.3], "arch": [0.78, 0.68, 0.5], "led_screen": [0.15, 0.35, 0.95],
                  "tv_screen": [0.15, 0.35, 0.95], "speaker_stand": [0.3, 0.3, 0.32], "potted_plant": [0.25, 0.26, 0.28],
                  "sign_stand": [0.85, 0.85, 0.85], "ceiling": [0.55, 0.56, 0.58], "person": [0.2, 0.22, 0.3]}


# ----------------------------------------------------------------------------- object -> world meshes + prims
def yaw_matrix(deg):
    a = math.radians(deg)
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return T


def quat_yaw(deg):
    h = math.radians(deg) / 2
    return [round(math.cos(h), 6), 0.0, 0.0, round(math.sin(h), 6)]


def normalize(o):
    """Fill defaults on an object dict (in place) and return it."""
    o["category"] = o.get("category") if o.get("category") in BUILDERS else "other"
    o["look"] = o.get("look") or o.get("material_look") or "paint"
    if o["look"] not in LOOKS:
        o["look"] = "paint"
    o["rgb"] = [round(float(v), 3) for v in (o.get("rgb") or [0.7, 0.7, 0.7])[:3]]
    acc = o.get("accent")
    o["accent"] = [round(float(v), 3) for v in (acc if acc and len(acc) >= 3 else
                                                  DEFAULT_ACCENT.get(o["category"], mix(o["rgb"], 0.6)))[:3]]
    o["look2"] = o.get("look2") if o.get("look2") in LOOKS else ("stone" if o["category"] == "arch" else
                                                                  "ceramic" if o["category"] == "potted_plant" else "paint")
    s = [abs(float(v)) for v in (o.get("size") or [1, 1, 1])[:3]]
    o["size"] = [round(max(0.01, v), 3) for v in s]
    o["center"] = [round(float(v), 3) for v in (o.get("center") or [0, 0])[:2]]
    o["base_z"] = round(float(o.get("base_z") or 0.0), 3)
    o["facing_deg"] = round(float(o.get("facing_deg") or 0.0), 1)
    if o["category"] in ("floor", "floor_patch", "floor_line") and -0.6 <= o["base_z"] < 0.05:
        o["base_z"] = 0.0                 # the walkable floor's top is z = 0 (slabs hang below it)
    o["label"] = str(o.get("label") or o["category"])
    if "collide" not in o or o["collide"] is None:
        o["collide"] = o["category"] not in NO_COLLIDE
    if o["category"] in NO_COLLIDE:
        o["collide"] = False
    o["material"] = PHYSICS.get(o["look"], "concrete")
    if o["category"] == "crash_mat":
        o["material"], o["look"] = "foam", "foam"
    return o


def build_object(o, scale=1.0):
    """World-space parts [(look, rgb, mesh)] and collision prims [{type, pos, size, quat, dims, yaw_deg}]."""
    W, D, H = [v * scale for v in o["size"]]
    parts, col = BUILDERS[o["category"]](W, D, H, o)
    T = yaw_matrix(o["facing_deg"])
    T[:3, 3] = [o["center"][0] * scale, o["center"][1] * scale, o["base_z"] * scale]
    out = []
    for look, rgb, m in parts:
        m = m.copy()
        m.apply_transform(T)
        out.append((look, [round(float(v), 3) for v in rgb], m))
    prims = []
    if o.get("collide"):
        for typ, c, dims in col:
            p = T @ np.r_[np.asarray(c, float), 1.0]
            d = [max(0.02, float(v)) for v in dims]
            if typ == "cylinder":
                size = [max(d[0], d[1]) / 2, d[2] / 2, 0]
            elif typ == "sphere":
                size = [max(d) / 2, 0, 0]
            else:
                size = [d[0] / 2, d[1] / 2, d[2] / 2]
            prims.append({"type": typ, "pos": np.round(p[:3], 3).tolist(), "size": [round(v, 3) for v in size],
                          "quat": quat_yaw(o["facing_deg"]), "yaw_deg": o["facing_deg"],
                          "dims": [round(v, 3) for v in d]})
    return out, prims
