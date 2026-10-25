"""FALLGUY FastAPI Server — Stateless Wave-by-Wave REST Boundary."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agent import (
    critique_and_propose_next_wave,
    get_api_key,
    get_model_name,
    reconstruct_scene_from_media,
    run_wave_simulation,
)
from backend.presets import STUNT_PRESETS

app = FastAPI(title="FALLGUY Autonomous Recursive Stunt Harness", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulateWaveRequest(BaseModel):
    wave_index: int = 1
    preset_id: Optional[str] = "couch_vault"
    scene_obstacles: Optional[List[Dict[str, Any]]] = None
    start_pos: Optional[List[float]] = None
    camera: Optional[Dict[str, Any]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    duration_sec: float = 2.4
    enable_local_refinement: bool = True


class ArgonCriticRequest(BaseModel):
    wave_data: Dict[str, Any]
    stunt_brief: str = "Launch over obstacle, clear hazard, and land safely on crash mat"
    history: Optional[List[Dict[str, Any]]] = None
    director_steer: Optional[str] = None
    canvas_image_b64: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None


class SceneFromVideoRequest(BaseModel):
    image_b64: Optional[str] = None
    keyframes_b64: List[str] = Field(default_factory=list)
    wireframe_overlay_b64: Optional[str] = None
    current_obstacles: Optional[List[Dict[str, Any]]] = None
    stunt_brief: str = "Vault obstacle and land on crash mat"
    api_key: Optional[str] = None
    model_name: Optional[str] = None


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": "MuJoCo 3.17.0 (500Hz)",
        "configured_model": get_model_name(),
        "api_key_configured": bool(get_api_key()),
    }


@app.get("/api/presets")
def api_presets() -> Dict[str, Any]:
    return {"presets": STUNT_PRESETS}


@app.post("/api/simulate-wave")
def api_simulate_wave(req: SimulateWaveRequest) -> Dict[str, Any]:
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


@app.post("/api/argon-critic")
def api_argon_critic(req: ArgonCriticRequest) -> Dict[str, Any]:
    return critique_and_propose_next_wave(
        wave_data=req.wave_data,
        stunt_brief=req.stunt_brief,
        history=req.history,
        director_steer=req.director_steer,
        canvas_image_b64=req.canvas_image_b64,
        api_key=req.api_key,
        model_name=req.model_name,
    )


@app.post("/api/scene-from-video")
def api_scene_from_video(req: SceneFromVideoRequest) -> Dict[str, Any]:
    frames = list(req.keyframes_b64)
    if req.image_b64 and req.image_b64 not in frames:
        frames.insert(0, req.image_b64)
    return reconstruct_scene_from_media(
        keyframes_b64=frames,
        stunt_brief=req.stunt_brief,
        wireframe_overlay_b64=req.wireframe_overlay_b64,
        current_obstacles=req.current_obstacles,
        api_key=req.api_key,
        model_name=req.model_name,
    )
