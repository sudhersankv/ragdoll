"""Stage 1a: frames + MEASUREMENT of the place (never used as a texture or visual mesh).

Depth Anything V2 Metric-Indoor (transformers, MPS) on a few frames from the still part of the shot, median-fused;
back-projected with the phone's intrinsics; RANSAC floor plane -> world frame (floor z = 0, +Z up, +X = camera
forward along the floor, +Y = camera left, origin under the camera). The pan after ~2.5 s is tracked as a pure
rotation (ORB + homography), so depth from the pan frames lands in the same world frame. The point cloud only feeds
the plan view / grid overlays / measurements that Argon reads to place the modelled assets.
"""
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEPTH_MODEL = os.environ.get("FALLGUY_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf")
# iPhone main camera (24 mm eq.) 4K video with stabilisation crop: ~68 deg across the long (portrait vertical) side.
DEFAULT_LONG_FOV = 68.0
REF_TIME = 1.0                     # s; the camera holds still for the first ~2.5 s
STILL_TIMES = [0.3, 0.7, 1.0, 1.4, 1.8]
MAX_DEPTH = 30.0
TEX_SIZE = (1080, 1920)            # portrait texture (w, h)
DEPTH_SIZE = (540, 960)            # depth / geometry resolution (w, h)
GRID = 4                           # px between mesh vertices at DEPTH_SIZE


def say(msg):
    print(f"world: {msg}", flush=True, file=__import__("sys").stderr)


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")


def extract_frames(media, frames_dir, fps=4):
    """All frames at `fps` (720 px wide portrait, rotation applied by ffmpeg) + ref.jpg at REF_TIME + still frames."""
    os.makedirs(frames_dir, exist_ok=True)
    ff = ffmpeg()
    _run([ff, "-y", "-loglevel", "error", "-i", media, "-vf", f"fps={fps},scale=720:-2", "-q:v", "3",
          os.path.join(frames_dir, "f_%03d.jpg")])
    _run([ff, "-y", "-loglevel", "error", "-ss", str(REF_TIME), "-i", media, "-frames:v", "1",
          "-vf", f"scale={TEX_SIZE[0]}:{TEX_SIZE[1]}", "-q:v", "2", os.path.join(frames_dir, "ref.jpg")])
    stills = []
    for i, t in enumerate(STILL_TIMES):
        p = os.path.join(frames_dir, f"still_{i}.jpg")
        _run([ff, "-y", "-loglevel", "error", "-ss", str(t), "-i", media, "-frames:v", "1",
              "-vf", f"scale={TEX_SIZE[0]}:{TEX_SIZE[1]}", "-q:v", "2", p])
        stills.append(p)
    # the last frame (end of the pan) for the critic
    _run([ff, "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", media, "-frames:v", "1",
          "-vf", f"scale={TEX_SIZE[0]}:{TEX_SIZE[1]}", "-q:v", "2", os.path.join(frames_dir, "end.jpg")])
    allf = sorted(f for f in os.listdir(frames_dir) if f.startswith("f_"))
    w, h = Image.open(os.path.join(frames_dir, "ref.jpg")).size
    say(f"extracted {len(allf)} frames at {fps} fps + ref.jpg ({w}x{h}) at t={REF_TIME}s")
    return {"fps": fps, "files": allf, "ref": "frames/ref.jpg", "ref_time": REF_TIME, "end": "frames/end.jpg",
            "stills": [os.path.relpath(p, os.path.dirname(frames_dir)) for p in stills], "width": w, "height": h}


def intrinsics(w, h, long_fov=DEFAULT_LONG_FOV):
    """Pinhole K for a w x h image whose long side spans long_fov degrees. Returns (fx, fy, cx, cy, fovy_deg)."""
    f = (max(w, h) / 2) / math.tan(math.radians(long_fov) / 2)
    fovy = math.degrees(2 * math.atan(h / 2 / f))
    return f, f, w / 2, h / 2, fovy


_DEPTH = {}


def depth_maps(paths, fuse=True):
    """Metric depth (m) per image at DEPTH_SIZE; median over images (fuse) or the list. MPS when available."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    say(f"depth: {DEPTH_MODEL} on {dev} for {len(paths)} still frame(s)")
    if "model" not in _DEPTH:
        _DEPTH["proc"] = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        _DEPTH["model"] = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval()
    proc, model = _DEPTH["proc"], _DEPTH["model"]
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize(DEPTH_SIZE, Image.BICUBIC)
        inp = proc(images=im, return_tensors="pt").to(dev)
        with torch.no_grad():
            d = model(**inp).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None], size=(DEPTH_SIZE[1], DEPTH_SIZE[0]), mode="bicubic",
                                            align_corners=False)[0, 0]
        out.append(d.float().cpu().numpy())
    return np.median(np.stack(out), axis=0) if fuse else out


def backproject(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - cx) / fx * depth
    y = (v - cy) / fy * depth
    return np.stack([x, y, depth], -1)          # OpenCV camera frame: x right, y down, z forward


def fit_floor(P, iters=600, thr=0.03, seed=0):
    """RANSAC plane on the lower 40 % of the image whose normal points roughly 'up' (-y camera)."""
    h = P.shape[0]
    pts = P[int(h * 0.6):].reshape(-1, 3)
    pts = pts[(pts[:, 2] > 0.3) & (pts[:, 2] < 12)]
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        if n[1] > 0:
            n = -n
        if -n[1] < 0.7:                       # must be roughly horizontal
            continue
        d = -n @ a
        cnt = np.count_nonzero(np.abs(pts @ n + d) < thr * max(1.0, 1.0))
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thr]
    # least-squares refine
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    n2 = vt[2] if vt[2][1] < 0 else -vt[2]
    d2 = -n2 @ c
    return n2, d2, len(inl) / len(pts)


def world_frame(n, d):
    """R (3x3, rows = world axes in camera coords) and camera height. World: z up (floor normal toward the camera),
    x = camera forward projected on the floor, y = left; origin = foot of the camera on the floor."""
    z = n / np.linalg.norm(n)
    h = d                                      # signed distance of the camera centre (origin) from the plane
    if h < 0:
        z, h = -z, -h
    fwd = np.array([0.0, 0.0, 1.0])
    x = fwd - (fwd @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z]), float(h)


def to_world(Pc, R, cam_h):
    Pw = Pc @ R.T
    Pw[..., 2] += cam_h
    return Pw


def camera_pose(R, cam_h):
    """MuJoCo camera: pos and xyaxes (camera x = right, y = up, looks along -z)."""
    right = R @ np.array([1.0, 0, 0])
    up = R @ np.array([0, -1.0, 0])
    fwd = R @ np.array([0, 0, 1.0])
    return {"pos": [0.0, 0.0, round(cam_h, 4)], "xyaxes": [*np.round(right, 5).tolist(), *np.round(up, 5).tolist()],
            "forward": np.round(fwd, 5).tolist(),
            "pitch_deg": round(math.degrees(math.asin(np.clip(fwd[2], -1, 1))), 2)}


def grid_mesh(Pw, depth, step=GRID, max_depth=MAX_DEPTH, jump=0.2):
    """Vertices on a pixel grid; two triangles per cell unless a depth discontinuity or a far point cuts it."""
    h, w = depth.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    V = Pw[np.ix_(ys, xs)].reshape(-1, 3)
    D = depth[np.ix_(ys, xs)]
    uv = np.stack(np.meshgrid((xs + 0.5) / w, 1 - (ys + 0.5) / h), -1).reshape(-1, 2)
    gh, gw = len(ys), len(xs)
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    Df = D.reshape(-1)

    def ok(tri):
        dd = Df[tri]
        return (dd.max(1) < max_depth) & (dd.max(1) / np.maximum(dd.min(1), 1e-3) < 1 + jump * step / 4 + 0.02)
    t1 = np.stack([a.ravel(), c.ravel(), b.ravel()], 1)
    t2 = np.stack([b.ravel(), c.ravel(), e.ravel()], 1)
    F = np.concatenate([t1[ok(t1)], t2[ok(t2)]])
    return V, F, uv


def write_obj(path, V, F, uv, tex_name):
    base = os.path.splitext(path)[0]
    mtl = base + ".mtl"
    with open(mtl, "w") as f:
        f.write(f"newmtl env\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {tex_name}\n")
    with open(path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl)}\nusemtl env\n")
        f.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in V))
        f.write("".join(f"vt {u:.5f} {v:.5f}\n" for u, v in uv))
        f.write("".join(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n" for a, b, c in F))


def write_glb(path, V, F, uv, tex_path):
    import trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    img = Image.open(tex_path).convert("RGB")
    mat = PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
    m = trimesh.Trimesh(vertices=V, faces=F, visual=TextureVisuals(uv=uv, material=mat), process=False)
    # glTF is Y-up: rotate our Z-up world so viewers show it upright (x stays x, z->y, y->-z)
    T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    m.apply_transform(T)
    m.export(path)


def measure(Pw, depth):
    """Key extents of the reconstruction for Argon (metres, world frame)."""
    P = Pw.reshape(-1, 3)
    d = depth.reshape(-1)
    P = P[d < 20]
    floor = P[np.abs(P[:, 2]) < 0.05]
    mid = P[(P[:, 2] > 0.3) & (P[:, 2] < 1.8) & (P[:, 0] > 1.0) & (P[:, 0] < 10)]
    left = mid[mid[:, 1] > 0.3]
    right = mid[mid[:, 1] < -0.3]
    q = lambda a, p: round(float(np.percentile(a, p)), 2) if len(a) else None
    return {
        "floor_x_range": [q(floor[:, 0], 2), q(floor[:, 0], 98)],
        "floor_y_range": [q(floor[:, 1], 2), q(floor[:, 1], 98)],
        "left_side_y_median": q(left[:, 1], 50), "right_side_y_median": q(right[:, 1], 50),
        "points_x_max": q(P[:, 0], 98), "points_z_max": q(P[:, 2], 98),
    }


def plan_view(Pw, depth, out, x_range=(-6, 20), y_range=(-8, 10), ppm=34, extra=None):
    """Top-down image of the measured points with a labelled 1 m grid (x up the image, +y to the left).
    `extra` = more world points (the pan frames)."""
    W = int((y_range[1] - y_range[0]) * ppm)
    H = int((x_range[1] - x_range[0]) * ppm)
    img = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(img)
    for x in range(int(x_range[0]), int(x_range[1]) + 1):
        py = H - (x - x_range[0]) * ppm
        dr.line([(0, py), (W, py)], fill=(215, 215, 215))
        dr.text((2, py - 11), f"x={x}", fill=(120, 120, 120))
    for y in range(int(y_range[0]), int(y_range[1]) + 1):
        px = (y_range[1] - y) * ppm
        dr.line([(px, 0), (px, H)], fill=(215, 215, 215))
        dr.text((px + 2, H - 12), f"y={y}", fill=(120, 120, 120))
    P = Pw.reshape(-1, 3)[depth.reshape(-1) < MAX_DEPTH]
    P = P[::3]
    if extra is not None and len(extra):
        P = np.concatenate([P, extra[::2]])
    px = ((y_range[1] - P[:, 1]) * ppm).astype(int)
    py = (H - (P[:, 0] - x_range[0]) * ppm).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    arr = np.asarray(img).copy()
    z = P[:, 2]
    col = np.where(z[:, None] < 0.08, [190, 190, 190],
                   np.where(z[:, None] < 1.2, [220, 120, 30], np.where(z[:, None] < 2.5, [40, 110, 220], [150, 60, 170])))
    order = np.argsort(z[ok])
    arr[py[ok][order], px[ok][order]] = col[ok][order]
    img = Image.fromarray(arr.astype(np.uint8))
    dr = ImageDraw.Draw(img)
    cx, cy = (y_range[1] - 0) * ppm, H - (0 - x_range[0]) * ppm
    dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(220, 0, 0))
    dr.text((cx + 8, cy - 6), "camera", fill=(220, 0, 0))
    dr.text((4, 4), "PLAN VIEW (top-down). grey=floor, orange=0.1-1.2 m high, blue=1.2-2.5 m, purple=>2.5 m. "
            "+x up the image (away from camera), +y to the LEFT", fill=(0, 0, 0))
    img.save(out)
    return {"x_range": list(x_range), "y_range": list(y_range), "px_per_m": ppm}


def grid_overlay(ref_path, R, cam_h, K, out, size=DEPTH_SIZE, title=None):
    """The reference frame with the world floor grid (1 m) projected onto it, labelled in world x/y."""
    fx, fy, cx, cy = K
    im = Image.open(ref_path).convert("RGB").resize(size)
    dr = ImageDraw.Draw(im)

    def proj(p):
        pc = R.T @ (np.asarray(p, float) - np.array([0, 0, cam_h]))
        if pc[2] < 0.2:
            return None
        return (fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy)
    for x in range(1, 21):
        seg = [proj((x, y, 0)) for y in np.linspace(-6, 6, 49)]
        seg = [s for s in seg if s]
        if len(seg) > 1:
            dr.line(seg, fill=(255, 220, 0), width=1)
        if x <= 12:
            p = proj((x, 0, 0))
            if p:
                dr.text((p[0] + 2, p[1] - 10), f"x={x}", fill=(255, 255, 0))
    for y in range(-6, 7):
        seg = [proj((x, y, 0)) for x in np.linspace(-6, 20, 105)]
        seg = [s for s in seg if s]
        if len(seg) > 1:
            dr.line(seg, fill=(0, 230, 255) if y else (255, 60, 60), width=1)
        p = proj((2.0, y, 0))
        if p and 0 < p[0] < size[0]:
            dr.text((p[0] + 2, p[1] + 2), f"y={y}", fill=(0, 255, 255))
    for x in range(-6, 1):        # the area behind / beside the camera (seen during the pan)
        seg = [proj((x, y, 0)) for y in np.linspace(-6, 10, 65)]
        seg = [s for s in seg if s]
        if len(seg) > 1:
            dr.line(seg, fill=(255, 140, 0), width=1)
        for y in (2, 4, 6):
            p = proj((x, y, 0))
            if p and 0 < p[0] < size[0] and 0 < p[1] < size[1]:
                dr.text((p[0] + 2, p[1] - 10), f"x={x},y={y}", fill=(255, 170, 60))
    if title:
        dr.rectangle([0, 0, size[0], 16], fill=(0, 0, 0))
        dr.text((4, 3), title, fill=(255, 255, 255))
    im.save(out)


PAN_TIMES = [2.75, 3.25, 3.75, 4.25, 4.75]     # s; frames from the pan whose depth is added to the measurements


def track_rotations(frames_dir, files, fps, long_fov, ref_index):
    """Per-frame camera rotation relative to the reference frame, assuming a rotation-only pan (handheld phone).
    Returns {file: R_i_ref (3x3, maps reference-camera coords to frame-i camera coords)}."""
    import cv2
    ims = [cv2.imread(os.path.join(frames_dir, f), cv2.IMREAD_GRAYSCALE) for f in files]
    h, w = ims[0].shape
    fx, fy, cx, cy, _ = intrinsics(w, h, long_fov)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    orb = cv2.ORB_create(4000)
    feats = [orb.detectAndCompute(im, None) for im in ims]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def rel(a, b):
        ka, da = feats[a]
        kb, db = feats[b]
        if da is None or db is None:
            return np.eye(3)
        m = sorted(bf.match(da, db), key=lambda x: x.distance)[:800]
        if len(m) < 12:
            return np.eye(3)
        pa = np.float32([ka[x.queryIdx].pt for x in m])
        pb = np.float32([kb[x.trainIdx].pt for x in m])
        H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
        if H is None:
            return np.eye(3)
        Rr = Ki @ H @ K
        U, _, Vt = np.linalg.svd(Rr)
        Rr = U @ Vt
        if np.linalg.det(Rr) < 0:
            Rr = -Rr
        return Rr
    out = {files[ref_index]: np.eye(3)}
    cur = np.eye(3)
    for i in range(ref_index + 1, len(files)):
        cur = rel(i - 1, i) @ cur
        out[files[i]] = cur.copy()
    cur = np.eye(3)
    for i in range(ref_index - 1, -1, -1):
        cur = rel(i + 1, i) @ cur
        out[files[i]] = cur.copy()
    return out


def yaw_of(Rw):
    """Heading (deg) of the camera's forward axis in the world x-y plane (0 = +x, +90 = +y / left)."""
    f = Rw @ np.array([0, 0, 1.0])
    return math.degrees(math.atan2(f[1], f[0]))


def reconstruct(media, wdir, long_fov=DEFAULT_LONG_FOV):
    """Frames, depth, floor, world frame, pan rotations, measurement point cloud. Returns the reconstruction record."""
    import time
    t0 = time.time()
    frames = extract_frames(media, os.path.join(wdir, "frames"))
    stills = [os.path.join(wdir, s) for s in frames["stills"]]
    w, h = DEPTH_SIZE
    fx, fy, cx, cy, fovy = intrinsics(w, h, long_fov)
    depth = depth_maps(stills)
    np.save(os.path.join(wdir, "mesh", "depth_ref.npy") if os.path.isdir(os.path.join(wdir, "mesh")) else
            os.path.join(wdir, "depth_ref.npy"), depth.astype(np.float32))
    Pc = backproject(depth, fx, fy, cx, cy)
    n, d, inlier_frac = fit_floor(Pc)
    R, cam_h = world_frame(n, d)
    say(f"floor plane: camera height {cam_h:.2f} m (raw model scale), inliers {inlier_frac:.0%}")
    Pw = to_world(Pc, R, cam_h)
    # ---- the pan: rotation-only tracking, depth of a few pan frames into the same world frame (measurement only)
    fdir = os.path.join(wdir, "frames")
    files = frames["files"]
    ref_i = min(len(files) - 1, int(round(REF_TIME * frames["fps"])))
    rots = track_rotations(fdir, files, frames["fps"], long_fov, ref_i)
    per_frame = []
    for k, f in enumerate(files):
        Rw = R @ rots[f].T
        per_frame.append({"file": f"frames/{f}", "t": round(k / frames["fps"], 3),
                          **camera_pose(Rw, cam_h), "yaw_deg": round(yaw_of(Rw), 1), "R_world_from_cam": Rw.tolist()})
    pan_sel = sorted({min(len(files) - 1, int(round(t * frames["fps"]))) for t in PAN_TIMES})
    pan_pts, pan_used = [], []
    if pan_sel:
        dps = depth_maps([os.path.join(fdir, files[i]) for i in pan_sel], fuse=False)
        for i, dp in zip(pan_sel, dps):
            Rw = np.array(per_frame[i]["R_world_from_cam"])
            Pci = backproject(dp, fx, fy, cx, cy)
            Pwi = to_world(Pci, R @ rots[files[i]].T, cam_h)
            fl = Pwi[int(h * 0.75):].reshape(-1, 3)
            fl = fl[np.abs(fl[:, 2]) < 0.6]
            # per-frame metric depth drifts: rescale about the camera so this frame's floor sits at z = 0
            if len(fl) > 200:
                zf = float(np.median(fl[:, 2]))
                k = cam_h / max(0.3, cam_h - zf)
                C = np.array([0, 0, cam_h])
                Pwi = (Pwi - C) * k + C
            m = dp.reshape(-1) < 20
            pan_pts.append(Pwi.reshape(-1, 3)[m][::2])
            pan_used.append({"file": f"frames/{files[i]}", "t": per_frame[i]["t"], "yaw_deg": per_frame[i]["yaw_deg"]})
    pan_P = np.concatenate(pan_pts) if pan_pts else np.zeros((0, 3))
    say(f"pan: camera yaw from {per_frame[0]['yaw_deg']:.0f} to {per_frame[-1]['yaw_deg']:.0f} deg; "
        f"{len(pan_P)} measurement points from {len(pan_used)} pan frames")
    rec = {"method": f"monocular metric depth ({DEPTH_MODEL}), median of {len(stills)} still frames + "
                     f"{len(pan_used)} pan frames (rotation-only tracking), RANSAC floor; used ONLY to measure layout",
           "per_frame": per_frame, "pan_frames": pan_used, "ref_index": ref_i,
           "frames": frames, "depth_size": [w, h], "long_fov_deg": long_fov,
           "K": [fx, fy, cx, cy], "fovy_deg": round(fovy, 3), "R_world_from_cam": R.tolist(),
           "camera_height_raw_m": round(cam_h, 4), "floor_inlier_frac": round(inlier_frac, 3),
           "seconds": None}
    rec["_Pw"], rec["_depth"], rec["_pan_P"] = Pw, depth, pan_P
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def export_mesh(rec, wdir, scale=1.0):
    """Scale the reconstruction about the world origin and write mesh/env.obj, env.mtl, env.png, env.glb."""
    mdir = os.path.join(wdir, "mesh")
    os.makedirs(mdir, exist_ok=True)
    Pw = rec["_Pw"] * scale
    depth = rec["_depth"] * scale
    V, F, uv = grid_mesh(Pw, depth)
    tex = os.path.join(mdir, "env.png")
    Image.open(os.path.join(wdir, rec["frames"]["ref"])).convert("RGB").save(tex)
    write_obj(os.path.join(mdir, "env.obj"), V, F, uv, "env.png")
    write_glb(os.path.join(mdir, "env.glb"), V, F, uv, tex)
    say(f"mesh: {len(V)} vertices, {len(F)} triangles -> mesh/env.obj, mesh/env.glb")
    lo, hi = V[F.ravel()].min(0), V[F.ravel()].max(0)
    return {"obj": "mesh/env.obj", "glb": "mesh/env.glb", "texture": "mesh/env.png",
            "obj_axes": "world frame (z up), same as scene.xml",
            "glb_axes": "glTF Y-up: glb (X, Y, Z) = world (x, z, -y); in three.js a Z-up scene gets world coords back "
                        "by rotating the loaded GLB +90 deg about X",
            "vertices": int(len(V)),
            "triangles": int(len(F)), "bbox": [np.round(lo, 2).tolist(), np.round(hi, 2).tolist()]}
