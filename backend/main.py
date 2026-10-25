"""RAGDOLL FastAPI Server — Stateless Wave-by-Wave REST Boundary & Native Video World Architect."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.agent import (
    critique_and_propose_next_wave,
    get_api_key,
    get_model_name,
    reconstruct_scene_from_media,
    run_wave_simulation,
)
from backend.presets import STUNT_PRESETS

app = FastAPI(title="RAGDOLL Autonomous Video-to-MuJoCo Simulation & Recursive Stunt Harness", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_obstacles(obstacles: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if obstacles is None:
        return None
    if not isinstance(obstacles, list) or len(obstacles) == 0:
        raise ValueError("scene_obstacles must be a non-empty list of obstacle dicts")
    seen_ids = set()
    for obs in obstacles:
        if not isinstance(obs, dict) or not obs.get("id"):
            raise ValueError("Each obstacle must include a non-empty string 'id'")
        oid = str(obs["id"]).strip()
        if oid in seen_ids:
            raise ValueError(f"Duplicate obstacle id '{oid}'")
        seen_ids.add(oid)
        pos = obs.get("pos")
        size = obs.get("size")
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            raise ValueError(f"Obstacle '{oid}' pos must have 3 coordinates [x, y, z]")
        if not isinstance(size, (list, tuple)) or len(size) != 3:
            raise ValueError(f"Obstacle '{oid}' size must have 3 half-extents [sx, sy, sz]")
        for v in list(pos) + list(size):
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"Obstacle '{oid}' contains non-finite coordinates")
        if any(float(s) <= 0 for s in size):
            raise ValueError(f"Obstacle '{oid}' size dimensions must be positive (> 0)")
    return obstacles


class SimulateWaveRequest(BaseModel):
    wave_index: int = Field(default=1, ge=1, le=20)
    preset_id: Optional[str] = "couch_vault"
    scene_obstacles: Optional[List[Dict[str, Any]]] = None
    start_pos: Optional[List[float]] = None
    camera: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    duration_sec: float = Field(default=2.4, ge=0.5, le=6.0)
    enable_local_refinement: bool = True

    @field_validator("scene_obstacles")
    @classmethod
    def check_obstacles(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        return _validate_obstacles(v)

    @field_validator("start_pos")
    @classmethod
    def check_start_pos(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        if len(v) != 3 or any(not math.isfinite(float(x)) for x in v):
            raise ValueError("start_pos must be a 3-element finite [x, y, z] list")
        return [float(x) for x in v]


class ArgonCriticRequest(BaseModel):
    wave_data: Dict[str, Any]
    stunt_brief: str = "Launch over obstacle, clear hazard, and land safely on crash mat"
    history: Optional[List[Dict[str, Any]]] = None
    director_steer: Optional[str] = None
    canvas_image_b64: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None

    @field_validator("wave_data")
    @classmethod
    def check_wave_data(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict) or not v.get("results"):
            raise ValueError("wave_data must contain a non-empty 'results' list")
        return v


class SceneFromVideoRequest(BaseModel):
    video_b64: Optional[str] = None
    video_mime: Optional[str] = "video/mp4"
    video_name: Optional[str] = None
    env_prompt: Optional[str] = None
    media_processing: str = "STATIC"
    image_b64: Optional[str] = None
    keyframes_b64: List[str] = Field(default_factory=list)
    wireframe_overlay_b64: Optional[str] = None
    current_obstacles: Optional[List[Dict[str, Any]]] = None
    current_start_pos: Optional[List[float]] = None
    current_camera: Optional[Dict[str, Any]] = None
    stunt_brief: str = "Vault obstacle and land on crash mat"
    api_key: Optional[str] = None
    model_name: Optional[str] = None

    @field_validator("current_obstacles")
    @classmethod
    def check_current_obstacles(cls, v: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        return _validate_obstacles(v)


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "platform": "RAGDOLL",
        "engine": "MuJoCo 3.17.0 (500Hz)",
        "configured_model": get_model_name(),
        "api_key_configured": bool(get_api_key()),
    }


@app.get("/api/presets")
def api_presets() -> Dict[str, Any]:
    return {"presets": STUNT_PRESETS}


@app.post("/api/simulate-wave")
def api_simulate_wave(req: SimulateWaveRequest) -> Dict[str, Any]:
    try:
        preset = STUNT_PRESETS.get(req.preset_id or "couch_vault", STUNT_PRESETS["couch_vault"])
        scene = {
            "scene_obstacles": req.scene_obstacles if req.scene_obstacles is not None else preset["scene_obstacles"],
            "start_pos": req.start_pos if req.start_pos is not None else preset["start_pos"],
            "camera": req.camera if req.camera is not None else preset["camera"],
        }
        candidates = req.candidates if req.candidates is not None else preset["wave_1_candidates"]
        return run_wave_simulation(
            candidates=candidates,
            scene=scene,
            wave_index=req.wave_index,
            duration_sec=req.duration_sec,
            enable_local_refinement=req.enable_local_refinement,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/argon-critic")
def api_argon_critic(req: ArgonCriticRequest) -> Dict[str, Any]:
    try:
        return critique_and_propose_next_wave(
            wave_data=req.wave_data,
            stunt_brief=req.stunt_brief,
            history=req.history,
            director_steer=req.director_steer,
            canvas_image_b64=req.canvas_image_b64,
            api_key=req.api_key,
            model_name=req.model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/scene-from-video")
async def api_scene_from_video(request: Request) -> Dict[str, Any]:
    try:
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("video")
            if upload is None or not hasattr(upload, "read"):
                raise ValueError("Attach the location video as the 'video' form field")
            video_bytes = await upload.read()
            if not video_bytes:
                raise ValueError("The uploaded video was empty")
            raw_obstacles = form.get("current_obstacles")
            current_obstacles = json.loads(raw_obstacles) if raw_obstacles else None
            return reconstruct_scene_from_media(
                stunt_brief=str(form.get("stunt_brief") or "Replicate this location. Do not add a person."),
                video_bytes=video_bytes,
                video_mime=getattr(upload, "content_type", None) or "video/mp4",
                video_name=getattr(upload, "filename", None) or "location-video",
                env_prompt=str(form.get("env_prompt") or "") or None,
                current_obstacles=current_obstacles,
                media_processing="STATIC",
            )
        req = SceneFromVideoRequest.model_validate(await request.json())
        frames = list(req.keyframes_b64)
        if req.image_b64 and req.image_b64 not in frames:
            frames.insert(0, req.image_b64)
        return reconstruct_scene_from_media(
            keyframes_b64=frames,
            stunt_brief=req.stunt_brief,
            video_b64=req.video_b64,
            video_mime=req.video_mime,
            video_name=req.video_name,
            env_prompt=req.env_prompt,
            media_processing=req.media_processing,
            wireframe_overlay_b64=req.wireframe_overlay_b64,
            current_obstacles=req.current_obstacles,
            current_start_pos=req.current_start_pos,
            current_camera=req.current_camera,
            api_key=req.api_key,
            model_name=req.model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
