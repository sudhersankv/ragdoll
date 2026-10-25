"""Tests for scripts/stunt_template.py, the per-run stunt script that `loop.py init` generates.

The script is generated with loop.generate_script (the real placeholder substitution) and run as a subprocess on
a small hand-written phases plan: a short hallway run toward the camera that turns to the camera's left (-Y).
Run: .venv/bin/python -m pytest .claude/skills/fallguy-stunt-loop/tests/test_template.py -q
"""
import inspect
import json
import os
import subprocess
import sys

import pytest

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(SKILL, "scripts")
sys.path.insert(0, SCRIPTS)
import loop  # noqa: E402

PY = sys.executable
BRIEF = 'runs "towards" the camera\\ and it\'s a left turn'

PLAN = {
    "version": 2, "brief": "runs towards the camera and takes a left turn", "duration_sec": 3.0,
    "source": {"media": "inputs/media.mp4", "kind": "video", "width": 720, "height": 1280},
    "room": {"x_min": -8, "x_max": 4, "y_min": -5, "y_max": 1.6, "ceiling_z": 4.0,
             "floor_rgb": [0.55, 0.53, 0.5], "wall_rgb": [0.9, 0.9, 0.88], "ceiling_rgb": [0.3, 0.3, 0.3],
             "floor_material": "concrete"},
    "obstacles": [{"id": "glass_railing", "kind": "box", "pos": [-3, 1.5, 0.55], "size": [5, 0.03, 0.55],
                   "yaw_deg": 0, "rgb": [0.7, 0.8, 0.85], "material": "glass", "role": "railing", "hazard": True}],
    "set_dressing": {}, "crash_mats": [],
    "camera": {"match": {"pos": [3.5, 0.3, 1.5], "lookat": [-4, 0, 1.0], "fovy": 64}},
    "stunt": {
        "summary": "runs toward the camera, then turns to the camera's left and stops",
        "start": {"pos": [-6.0, 0.0], "heading_deg": 0},
        "phases": [
            {"kind": "hold", "name": "ready", "duration": 0.3, "pose": "stand"},
            {"kind": "move", "name": "run", "path": [[-6.0, 0.0], [-1.0, 0.0], [1.0, -2.5]], "speed": 3.0,
             "accel_time": 0.8, "decel_time": 0.8, "turn_radius": 1.5, "lean_scale": 1.0},
            {"kind": "hold", "name": "stop", "duration": 0.5, "pose": "stand"},
        ],
        "tunables": [
            {"name": "run_speed", "path": "phases[1].speed", "min": 1.5, "max": 6.0, "unit": "m/s", "doc": "cruise"},
            {"name": "turn_radius", "path": "phases[1].turn_radius", "min": 0.8, "max": 4.0, "unit": "m",
             "doc": "turn radius"},
        ],
        "target": {"kind": "zone", "pos": [1.0, -2.5], "radius": 0.65},
        "default_camera": "match_follow",
    },
    "performer": {"model": "h1"},
}


def generate(dest, brief):
    """loop.generate_script, with or without the retired stunt_type argument (loop.py is rebuilt in parallel)."""
    n = len(inspect.signature(loop.generate_script).parameters)
    loop.generate_script(dest, brief, *(["generic"] if n >= 3 else []))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tmpl")
    (tmp / "plan.json").write_text(json.dumps(PLAN))
    (tmp / "params.json").write_text(json.dumps({"id": "t1", "label": "test", "run_speed": 3.5}))
    script = tmp / "stunt_test.py"
    generate(str(script), BRIEF)
    return tmp, script


def call(tmp, script, *extra, out="c1", params="params.json", timeout=600):
    cmd = [PY, str(script), "--plan", "plan.json", "--params", params, "--out-dir", out, *extra]
    return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout)


def test_placeholders_filled(run):
    _, script = run
    src = script.read_text()
    assert "__FALLGUY_" not in src
    ns = {}
    exec(compile(src.split("if SCRIPTS_DIR.startswith", 1)[0], str(script), "exec"), ns)
    assert ns["BRIEF"] == BRIEF and ns["SCRIPTS_DIR"] == loop.SCRIPTS_DIR and ns["GENERATED_AT"]
    assert "def choreography(plan, params):" in src
    for gone in ("STUNT_TYPE", "stunt.type", "run_turn", "BarrelRoll"):
        assert gone not in src, gone


def test_check(run):
    tmp, script = run
    r = call(tmp, script, "--check", out="chk")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "OK"


def test_sim_metrics_and_render(run):
    import imageio.v3 as iio
    tmp, script = run
    r = call(tmp, script, "--render", "--duration", "2", "--fps", "10", "--size", "180x320")
    assert r.returncode == 0, r.stderr
    c1 = tmp / "c1"
    for name in ("scene.xml", "trajectory.npz", "events.json", "metrics.json", "video.mp4"):
        assert (c1 / name).is_file(), name
    m = json.loads((c1 / "metrics.json").read_text())
    assert m["ok"] is True and m["candidate_id"] == "t1"
    assert isinstance(m["gate1"]["passed"], bool) and "stunt_score" in m["score"]
    frames = list(iio.imiter(c1 / "video.mp4"))
    assert len(frames) == 20 and frames[0].shape[:2] == (320, 180)


def test_preview_still(run):
    tmp, script = run
    r = call(tmp, script, "--preview-still", "preview.png", "--size", "180x320", out="pv")
    assert r.returncode == 0, r.stderr
    assert (tmp / "preview.png").stat().st_size > 1000


def test_failure_writes_metrics(run):
    tmp, script = run
    r = call(tmp, script, out="bad", params="missing.json")
    assert r.returncode == 1
    m = json.loads((tmp / "bad" / "metrics.json").read_text())
    assert m["ok"] is False and "missing.json" in m["error"]
