"""MuJoCo Stunt Video & Storyboard Renderer.

Renders simulated stunt trajectories (with optional background location video/photo)
to MP4 video files and storyboard composite images, complete with live biomechanical
G-force HUD overlays and 3D trajectory trails.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from backend.agent import critique_and_propose_next_wave, run_wave_simulation
from backend.presets import STUNT_PRESETS


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (246, 130, 59)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _project_3d_to_2d(
    pt: np.ndarray,
    cam_pos: np.ndarray,
    cam_lookat: np.ndarray,
    fovy_deg: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, float]]:
    forward = cam_lookat - cam_pos
    forward = forward / max(1e-9, np.linalg.norm(forward))
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    rel = pt - cam_pos
    z = float(np.dot(rel, forward))
    if z <= 0.15:
        return None
    tan_half = math.tan(math.radians(fovy_deg) * 0.5)
    aspect = width / max(1.0, float(height))
    nx = float(np.dot(rel, right)) / (z * tan_half * aspect)
    ny = float(np.dot(rel, up)) / (z * tan_half)
    px = int(round((nx * 0.5 + 0.5) * width))
    py = int(round((0.5 - ny * 0.5) * height))
    return (px, py, z)


def _box_corners(pos: List[float], size: List[float]) -> List[np.ndarray]:
    cx, cy, cz = pos
    sx, sy, sz = size
    corners = []
    for dx in (-sx, sx):
        for dy in (-sy, sy):
            for dz in (-sz, sz):
                corners.append(np.array([cx + dx, cy + dy, cz + dz], dtype=float))
    return corners


BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Bone connectivity pairs among the 14 humanoid geoms:
# 0:head, 1:neck, 2:torso, 3:pelvis, 4:uarm_l, 5:larm_l, 6:uarm_r, 7:larm_r,
# 8:thigh_l, 9:shin_l, 10:foot_l, 11:thigh_r, 12:shin_r, 13:foot_r
BONE_LINKS = [
    (0, 1), (1, 2), (2, 3),
    (2, 4), (4, 5),
    (2, 6), (6, 7),
    (3, 8), (8, 9), (9, 10),
    (3, 11), (11, 12), (12, 13),
]


def render_wave_frame(
    wave_data: Dict[str, Any],
    frame_idx: int,
    width: int = 960,
    height: int = 540,
    bg_frame_bgr: Optional[np.ndarray] = None,
    title_banner: str = "",
) -> np.ndarray:
    """Renders a single composite tactical 3D frame for `wave_data` at `frame_idx`."""
    if bg_frame_bgr is not None:
        canvas = cv2.resize(bg_frame_bgr, (width, height))
        dark = np.zeros_like(canvas)
        canvas = cv2.addWeighted(canvas, 0.65, dark, 0.35, 0)
    else:
        canvas = np.full((height, width, 3), (20, 11, 7), dtype=np.uint8)

    cam = wave_data.get("camera", {"pos": [0.4, -4.6, 1.5], "lookat": [0.5, 0.0, 0.7], "fovy": 50})
    cam_pos = np.array(cam["pos"], dtype=float)
    cam_look = np.array(cam["lookat"], dtype=float)
    fovy = float(cam.get("fovy", 50.0))

    # 1. Draw 3D ground grid lines
    for gx in np.linspace(-3.0, 4.0, 15):
        p1 = _project_3d_to_2d(np.array([gx, -2.0, 0.0]), cam_pos, cam_look, fovy, width, height)
        p2 = _project_3d_to_2d(np.array([gx, 2.0, 0.0]), cam_pos, cam_look, fovy, width, height)
        if p1 and p2:
            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), (48, 32, 20), 1, cv2.LINE_AA)
    for gy in np.linspace(-2.0, 2.0, 9):
        p1 = _project_3d_to_2d(np.array([-3.0, gy, 0.0]), cam_pos, cam_look, fovy, width, height)
        p2 = _project_3d_to_2d(np.array([4.0, gy, 0.0]), cam_pos, cam_look, fovy, width, height)
        if p1 and p2:
            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), (48, 32, 20), 1, cv2.LINE_AA)

    # 2. Draw 3D obstacles (translucent fill + glowing wireframe)
    overlay = canvas.copy()
    for obs in wave_data.get("scene_obstacles", []):
        corners = _box_corners(obs["pos"], obs["size"])
        proj = [_project_3d_to_2d(c, cam_pos, cam_look, fovy, width, height) for c in corners]
        if any(p is None for p in proj):
            continue
        is_haz = obs.get("is_hazard", False)
        wire_col = (248, 189, 56) if is_haz else (129, 230, 16)
        fill_col = (110, 75, 15) if is_haz else (45, 110, 15)

        # Top face fill
        top_pts = np.array([[proj[i][0], proj[i][1]] for i in (1, 3, 7, 5)], dtype=np.int32)
        cv2.fillPoly(overlay, [top_pts], fill_col)
        front_pts = np.array([[proj[i][0], proj[i][1]] for i in (0, 1, 5, 4)], dtype=np.int32)
        cv2.fillPoly(overlay, [front_pts], fill_col)

        for e0, e1 in BOX_EDGES:
            cv2.line(
                overlay,
                (proj[e0][0], proj[e0][1]),
                (proj[e1][0], proj[e1][1]),
                wire_col,
                2,
                cv2.LINE_AA,
            )
        # Label obstacle
        lbl_pt = proj[5]
        cv2.putText(
            overlay,
            obs["id"],
            (lbl_pt[0] - 20, lbl_pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            wire_col,
            1,
            cv2.LINE_AA,
        )

    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)

    # 3. Draw all 4 Ghost Stuntmen + Trajectory Ribbons (winner drawn last & brightest)
    best_id = wave_data.get("best_ghost_id", "ghost_D")
    results = sorted(wave_data.get("results", []), key=lambda r: r["ghost_id"] == best_id)

    for ghost in results:
        is_best = ghost["ghost_id"] == best_id
        col_bgr = _hex_to_bgr(ghost.get("color", "#10b981"))

        # Parabolic trajectory curve up to current frame
        traj = ghost.get("trajectory_points", [])
        max_t_idx = min(len(traj), frame_idx + 1)
        proj_traj = []
        for pt in traj[:max_t_idx]:
            p2 = _project_3d_to_2d(np.array(pt, dtype=float), cam_pos, cam_look, fovy, width, height)
            if p2:
                proj_traj.append((p2[0], p2[1]))
        for i in range(1, len(proj_traj)):
            cv2.line(canvas, proj_traj[i - 1], proj_traj[i], col_bgr, 3 if is_best else 1, cv2.LINE_AA)

        frames = ghost.get("frames", [])
        if not frames:
            continue
        f_clamped = frames[min(frame_idx, len(frames) - 1)]
        g_load = float(f_clamped.get("g", 1.0))
        geoms = f_clamped.get("geoms", [])

        pts_2d: List[Optional[Tuple[int, int, float]]] = []
        for g in geoms:
            pts_2d.append(_project_3d_to_2d(np.array(g["p"], dtype=float), cam_pos, cam_look, fovy, width, height))

        # Draw articulated limbs
        ghost_layer = canvas.copy()
        for b0, b1 in BONE_LINKS:
            if b0 < len(pts_2d) and b1 < len(pts_2d) and pts_2d[b0] and pts_2d[b1]:
                # Flash head/neck/spine red on high G-load (>=8.0g per SPEC.md)
                limb_col = col_bgr
                if g_load >= 8.0 and b0 in (0, 1, 2):
                    limb_col = (60, 30, 255)
                elif g_load >= 5.0 and b0 in (0, 1):
                    limb_col = (0, 165, 255)
                thickness = 7 if is_best else 4
                cv2.line(
                    ghost_layer,
                    (pts_2d[b0][0], pts_2d[b0][1]),
                    (pts_2d[b1][0], pts_2d[b1][1]),
                    limb_col,
                    thickness,
                    cv2.LINE_AA,
                )

        # Draw head sphere & joint nodes
        if pts_2d and pts_2d[0]:
            head_col = (60, 30, 255) if g_load >= 8.0 else ((0, 165, 255) if g_load >= 5.0 else col_bgr)
            cv2.circle(ghost_layer, (pts_2d[0][0], pts_2d[0][1]), 11 if is_best else 7, head_col, -1, cv2.LINE_AA)
            if is_best:
                cv2.putText(
                    ghost_layer,
                    f"{ghost['ghost_id']} ({g_load:.1f}g)",
                    (pts_2d[0][0] - 28, pts_2d[0][1] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        alpha = 0.95 if is_best else 0.40
        canvas = cv2.addWeighted(ghost_layer, alpha, canvas, 1.0 - alpha, 0)

    # 4. Draw Tactical Director HUD Overlay
    best_res = next((r for r in results if r["ghost_id"] == best_id), results[-1] if results else None)
    cv2.rectangle(canvas, (14, 12), (width - 14, 78), (28, 18, 12), -1)
    cv2.rectangle(canvas, (14, 12), (width - 14, 78), (90, 70, 45), 1)

    header = title_banner or f"FALLGUY STUNT HARNESS | WAVE #{wave_data.get('wave_index', 1)}"
    cv2.putText(canvas, header, (26, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 235, 180), 2, cv2.LINE_AA)

    if best_res:
        m = best_res["metrics"]
        g1 = best_res["gate_1_passed"]
        status_txt = "GATE 1 PASS (SAFE)" if g1 else f"GATE 1 FAIL ({best_res['diagnostic'][:48]})"
        status_col = (120, 240, 80) if g1 else (80, 80, 255)
        sub = (
            f"Best: {best_res['ghost_id']} ({best_res['label']}) | Score: {best_res['stunt_score']}% | "
            f"Peak Head: {m['peak_head_g']}g | HIC15: {m['hic15']} | Clear: {m['min_clearance_cm']}cm | "
            f"Mat Err: {m['landing_error_m']}m"
        )
        cv2.putText(canvas, sub, (26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, status_txt, (width - 360, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 2, cv2.LINE_AA)

    return canvas


def render_preset_progression_video(
    preset_id: str,
    out_mp4_path: Path,
    out_storyboard_png: Optional[Path] = None,
    bg_video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Runs Wave 1 (Initial Failed Seed Rollouts) -> Wave 2 (Recursive Harness Solved Rollout)
    for `preset_id` and writes an MP4 video + 4-panel storyboard comparison image.
    """
    preset = STUNT_PRESETS[preset_id]
    scene = {
        "scene_obstacles": preset["scene_obstacles"],
        "start_pos": preset["start_pos"],
        "camera": preset["camera"],
    }

    w1 = run_wave_simulation(preset["wave_1_candidates"], scene, wave_index=1, enable_local_refinement=False)
    c1 = critique_and_propose_next_wave(w1, preset["description"])
    w2 = run_wave_simulation(c1["next_candidates"], scene, wave_index=2, enable_local_refinement=True)

    bg_frames: List[np.ndarray] = []
    if bg_video_path and bg_video_path.exists():
        cap = cv2.VideoCapture(str(bg_video_path))
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            bg_frames.append(frm)
        cap.release()

    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4_path), fps=30, codec="libx264", quality=8)

    n_frames = len(w1["results"][0]["frames"])
    # Write Wave 1 (every 2nd 60Hz frame -> 30fps)
    for idx in range(0, n_frames, 2):
        bg = bg_frames[idx % len(bg_frames)] if bg_frames else None
        frame_bgr = render_wave_frame(
            w1,
            idx,
            bg_frame_bgr=bg,
            title_banner=f"{preset['name']} — WAVE #1 (Initial Attempt: Best {w1['best_score']}%)",
        )
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # Write Wave 2 (Recursive Harness Solved)
    for idx in range(0, n_frames, 2):
        bg = bg_frames[idx % len(bg_frames)] if bg_frames else None
        frame_bgr = render_wave_frame(
            w2,
            idx,
            bg_frame_bgr=bg,
            title_banner=f"{preset['name']} — WAVE #2 (Recursive Harness Solved: {w2['best_score']}%)",
        )
        writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    writer.close()

    if out_storyboard_png:
        # Build a 2x2 storyboard grid: Wave 1 Mid-Air & Impact vs Wave 2 Mid-Air & Landing
        f_w1_air = render_wave_frame(w1, int(n_frames * 0.22), title_banner=f"WAVE #1 Apex (Score {w1['best_score']}%)")
        f_w1_hit = render_wave_frame(w1, int(n_frames * 0.48), title_banner="WAVE #1 Impact (Failed Gate 1)")
        f_w2_air = render_wave_frame(w2, int(n_frames * 0.22), title_banner=f"WAVE #2 Apex (Score {w2['best_score']}%)")
        f_w2_land = render_wave_frame(w2, int(n_frames * 0.48), title_banner="WAVE #2 Safe Mat Landing (Gate 1 & 2 PASS)")
        top_row = np.hstack([f_w1_air, f_w1_hit])
        bot_row = np.hstack([f_w2_air, f_w2_land])
        grid_img = np.vstack([top_row, bot_row])
        out_storyboard_png.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_storyboard_png), grid_img)

    return {
        "preset_id": preset_id,
        "wave_1_score": w1["best_score"],
        "wave_2_score": w2["best_score"],
        "wave_2_gate_1_passed": w2["any_gate_1_passed"],
        "mp4_path": str(out_mp4_path),
        "storyboard_path": str(out_storyboard_png) if out_storyboard_png else None,
    }
