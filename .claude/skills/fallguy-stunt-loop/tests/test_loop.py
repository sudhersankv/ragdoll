"""Tests for scripts/loop.py, scripts/report.py and scripts/setup_env.sh.

Argon runs in mock mode (FALLGUY_ARGON_MOCK=1, a fake key file) and the generated stunt script is
tests/fixtures/fake_stunt_template.py, so the tests need no network, no real key and no MuJoCo run.
Gate 1 / Gate 2 scoring uses the real scripts/gates.py and the real scripts/argon.py mock.

    python -m pytest tests/test_loop.py -q

FALLGUY_TEST_SCRIPTS_DIR points the tests at another scripts dir (a dev copy with stubs).
"""
import datetime as dt
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
SCRIPTS = os.environ.get("FALLGUY_TEST_SCRIPTS_DIR") or os.path.join(SKILL, "scripts")
LOOP = os.path.join(SCRIPTS, "loop.py")
FAKE_TEMPLATE = os.path.join(HERE, "fixtures", "fake_stunt_template.py")
FAKE_KEY = "AIzaTESTKEY00000000000000000000000000042"     # fake; must never appear in any output
BRIEF = "Barrel roll over the table onto a crash mat"
SLUG = "barrel-roll-over-the-table-onto-a-crash"
TRICKY_BRIEF = 'Barrel roll over the "table" \\ it\'s fast\n\tthen land — café \U00010348 {x}'


# ----------------------------------------------------------------------------- helpers
def make_env(tmp_path, **extra):
    token = tmp_path / "argon_token"
    if not token.exists():
        token.write_text(FAKE_KEY + "\n")
        token.chmod(0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_STUNT_", "FAKE_ARGON_", "FALLGUY_"))}
    env.update(FALLGUY_ARGON_MOCK="1", FALLGUY_STUNT_TEMPLATE=FAKE_TEMPLATE, FALLGUY_ARGON_TOKEN_FILE=str(token),
               FALLGUY_ARGON_RETRY_DELAY="0", FALLGUY_ARGON_BACKOFF="0", FAKE_STUNT_PASS_FROM_WAVE="2")
    env.update({k: str(v) for k, v in extra.items()})
    return env


def loop(tmp_path, *args, ok=True, env=None, timeout=300):
    """Run loop.py in tmp_path; returns (exit code, the one stdout JSON object, stderr)."""
    p = subprocess.run([sys.executable, LOOP, *args], cwd=tmp_path, env=env or make_env(tmp_path),
                       capture_output=True, text=True, timeout=timeout)
    lines = [line for line in p.stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"expected one JSON line on stdout, got {p.stdout!r}\nstderr:\n{p.stderr}"
    out = json.loads(lines[0])
    if ok:
        assert p.returncode == 0, f"exit {p.returncode}: {out}\nstderr:\n{p.stderr}"
    assert FAKE_KEY not in p.stdout + p.stderr
    return p.returncode, out, p.stderr


def jload(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return json.load(f)


def calls(path):
    """argv lists the fake stunt script recorded in one out-dir."""
    with open(os.path.join(path, "fake_calls.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def video_meta(path):
    import imageio_ffmpeg
    gen = imageio_ffmpeg.read_frames(path)
    try:
        return next(gen)
    finally:
        gen.close()


def init_run(tmp_path, media, *extra, env=None, brief=BRIEF, duration="2"):
    _, out, _ = loop(tmp_path, "init", "--media", media, "--brief", brief, "--duration", duration, *extra, env=env)
    return out


def import_scripts_module(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", os.path.join(SCRIPTS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def image(tmp_path):
    from PIL import Image
    path = tmp_path / "venue.png"
    Image.new("RGB", (2400, 1200), (120, 90, 60)).save(path)
    return str(path)


def make_video(path, w, h, frames=15, fps=15):
    import imageio.v2 as imageio
    import numpy as np
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1, quality=5) as wr:
        for i in range(frames):
            wr.append_data(np.full((h, w, 3), (i * 15 % 256, 80, 160), np.uint8))
    return str(path)


# ----------------------------------------------------------------------------- init
def test_init_image(tmp_path, image):
    out = init_run(tmp_path, image, brief=TRICKY_BRIEF)
    run_dir = out["run_dir"]
    assert os.path.dirname(run_dir) == str(tmp_path / "stunts")            # default --runs-dir ./stunts
    m = re.fullmatch(r"([a-z0-9-]+)-(\d{8}_\d{6})", os.path.basename(run_dir))
    assert m and len(m.group(1)) <= 40, run_dir
    slug, tag = m.groups()
    assert os.path.basename(out["script"]) == f"stunt_{slug}_{tag}.py"
    assert os.path.dirname(out["script"]) == run_dir

    from PIL import Image
    with Image.open(os.path.join(run_dir, "inputs", "media.jpg")) as im:
        assert im.format == "JPEG" and max(im.size) == 1600
    assert os.path.isfile(os.path.join(run_dir, "inputs", "ref_frame.jpg"))
    assert os.path.getsize(out["preview"]) > 0

    plan = jload(out["plan"])
    assert plan["source"]["kind"] == "image" and plan["source"]["media"] == "inputs/media.jpg"
    assert plan["source"]["ref_frame"] == "inputs/ref_frame.jpg"
    assert plan["performer"]["model"] == "h1" and plan["duration_sec"] == 2.0

    run = jload(run_dir, "run.json")
    assert run["status"] == "planned" and run["checkpoint_every"] == 5 and run["waves"] == []
    assert run["director_notes"] == [] and run["best"] is None and run["plan"] == "plan.json"
    assert run["media_original"] == image and run["media"] == "inputs/media.jpg"
    assert run["script"] == os.path.basename(out["script"]) and run["brief"] == TRICKY_BRIEF.strip()

    # the brief survives into every string-literal style of the template, byte for byte
    src = open(out["script"], encoding="utf-8").read()
    assert "__FALLGUY_" not in src
    ns = {}
    exec(compile(src.split("import argparse")[0], out["script"], "exec"), ns)
    assert ns["BRIEF"] == TRICKY_BRIEF.strip()
    assert ns["SCRIPTS_DIR"] == SCRIPTS and ns["STUNT_TYPE"] == plan["stunt"]["type"]
    dt.datetime.fromisoformat(ns["GENERATED_AT"])
    assert os.access(out["script"], os.X_OK)

    # init ran the script with --check and --preview-still, from the run dir
    argvs = [c["argv"] for c in calls(os.path.join(run_dir, "check"))]
    assert any("--check" in a for a in argvs) and any("--preview-still" in a for a in argvs)
    assert all(c["cwd"] == run_dir for c in calls(os.path.join(run_dir, "check")))


@pytest.mark.parametrize("w,h,box", [(1920, 1080, (1280, 720)), (1080, 1920, (720, 1280))])
def test_init_video_makes_h264_proxy_and_ref_frame(tmp_path, w, h, box):
    src = make_video(tmp_path / "site.mp4", w, h)
    out = init_run(tmp_path, src)
    run_dir = out["run_dir"]
    proxy = os.path.join(run_dir, "inputs", "media.mp4")
    meta = video_meta(proxy)
    assert meta["codec"] == "h264" and tuple(meta["size"]) == box and meta["fps"] <= 30
    from PIL import Image
    with Image.open(os.path.join(run_dir, "inputs", "ref_frame.jpg")) as im:
        assert im.format == "JPEG" and im.size == box
    plan = jload(run_dir, "plan.json")
    assert plan["source"]["kind"] == "video" and plan["source"]["media"] == "inputs/media.mp4"
    assert 0.0 <= plan["source"]["ref_time_sec"] <= meta["duration"]
    run = jload(run_dir, "run.json")
    assert run["media_kind"] == "video" and run["media_original"] == src


@pytest.mark.skipif(not shutil.which("sips"), reason="needs macOS sips")
def test_init_heic_photo(tmp_path, image):
    heic = str(tmp_path / "venue.heic")
    subprocess.run(["sips", "-s", "format", "heic", image, "--out", heic], check=True, capture_output=True)
    out = init_run(tmp_path, heic)
    from PIL import Image
    with Image.open(os.path.join(out["run_dir"], "inputs", "media.jpg")) as im:
        assert im.format == "JPEG" and max(im.size) == 1600
    assert not os.path.exists(os.path.join(out["run_dir"], "inputs", "_heic_source.jpg"))


def test_unique_names_on_collision(tmp_path, image):
    # every run dir name init could pick in the next seconds is already taken -> it must append _2
    runs = tmp_path / "stunts"
    now = dt.datetime.now()
    for s in range(-1, 8):
        (runs / f"{SLUG}-{now + dt.timedelta(seconds=s):%Y%m%d_%H%M%S}").mkdir(parents=True)
    out = init_run(tmp_path, image)
    base = os.path.basename(out["run_dir"])
    assert re.fullmatch(rf"{SLUG}-\d{{8}}_\d{{6}}_2", base), base
    tag = base[len(SLUG) + 1:]
    assert os.path.basename(out["script"]) == f"stunt_{SLUG}_{tag}.py"
    assert jload(out["run_dir"], "run.json")["tag"] == tag

    loop_mod = import_scripts_module("loop")
    made = [loop_mod.make_run_dir(str(tmp_path / "other"), "s", "20261025_120000") for _ in range(3)]
    assert [t for _, t in made] == ["20261025_120000", "20261025_120000_2", "20261025_120000_3"]
    assert all(os.path.isdir(p) for p, _ in made)


def test_slugify():
    slugify = import_scripts_module("loop").slugify
    assert slugify(BRIEF) == SLUG
    assert slugify("Barrel roll over the wooden shelf unit and land") == "barrel-roll-over-the-wooden-shelf-unit"
    assert slugify("Café — DIVE!!  roll") == "cafe-dive-roll"
    assert slugify("!!!") == "stunt" and len(slugify("x" * 99)) == 40


def test_py_escape_round_trip():
    py_escape = import_scripts_module("loop").py_escape
    for s in (TRICKY_BRIEF, "", "\\", "'''", '"""', "a\rb\x00c\x7f", "\udcff"):
        for q in ('"%s"', "'%s'", '"""%s"""', "'''%s'''"):
            assert eval(q % py_escape(s)) == s, (s, q)


# ----------------------------------------------------------------------------- waves
def test_wave_fails_gate1_then_passes_then_finalize(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]

    _, s1, err = loop(tmp_path, "wave", "--run-dir", run_dir)
    assert s1["wave"] == 1 and re.fullmatch(r"w1c[1-4]", s1["best_id"])
    assert s1["gate1_passed"] is False and s1["gate2_passed"] is False and s1["next"] == "wave"
    assert s1["failures"] and isinstance(s1["critique_summary"], str) and isinstance(s1["stunt_score"], float)
    assert s1 == jload(run_dir, "waves", "wave01", "summary.json")
    assert "[wave 1]" in err
    w1 = os.path.join(run_dir, "waves", "wave01")
    for f in ("history.json", "proposal.json", "critique.json"):
        assert os.path.isfile(os.path.join(w1, f)), f
    for i in range(1, 5):
        p = jload(w1, f"c{i}", "params.json")
        m = jload(w1, f"c{i}", "metrics.json")
        assert p["id"] == f"w1c{i}" and p["camera"] in ("hero", "match", "side") and re.fullmatch(r"#[0-9a-fA-F]{6}", p["color"])
        assert m["ok"] and m["gate1"]["passed"] is False and "stunt_score" in m["score"]
    run = jload(run_dir, "run.json")
    assert run["status"] == "running" and len(run["waves"]) == 1 and run["best"]["id"] == s1["best_id"]

    # the best take was rendered at the run's duration, 30 fps, its own camera, no HUD
    bdir = os.path.join(run_dir, "waves", "wave01", f"c{s1['best_id'][-1]}")
    render = [c["argv"] for c in calls(bdir) if "--render" in c["argv"]]
    assert len(render) == 1
    a = render[0]
    cam = jload(bdir, "params.json")["camera"]
    assert a[a.index("--duration") + 1] == "2" and a[a.index("--fps") + 1] == "30" and a[a.index("--camera") + 1] == cam
    assert "--hud" not in a
    meta = video_meta(os.path.join(bdir, "video.mp4"))
    assert abs(meta["duration"] - 2.0) < 0.15 and meta["fps"] == 30

    _, s2, _ = loop(tmp_path, "wave", "--run-dir", run_dir)
    assert s2["wave"] == 2 and s2["gate1_passed"] is True and s2["gate2_passed"] is True
    assert s2["next"] == "finalize" and s2["failures"] == []
    run = jload(run_dir, "run.json")
    assert run["status"] == "passed" and run["best"]["id"] == s2["best_id"] and run["best"]["wave"] == 2
    hist = jload(run_dir, "waves", "wave02", "history.json")
    assert [w["wave"] for w in hist["waves"]] == [1] and len(hist["waves"][0]["candidates"]) == 4
    assert hist["waves"][0]["best_id"] == s1["best_id"] and hist["waves"][0]["critique"]
    assert hist["brief"] == BRIEF

    _, fin, _ = loop(tmp_path, "finalize", "--run-dir", run_dir)
    tag = jload(run_dir, "run.json")["tag"]
    assert fin["final_video"] == os.path.join(run_dir, f"final_{SLUG}_{tag}.mp4")
    assert fin["certificate"] == os.path.join(run_dir, f"certificate_{SLUG}_{tag}.json")
    assert fin["report"] == os.path.join(run_dir, "report.md") and fin["passed"] is True
    meta = video_meta(fin["final_video"])
    assert meta["fps"] == 60 and abs(meta["duration"] - 2.0) < 0.15
    fa = [c["argv"] for c in calls(os.path.join(run_dir, "final"))][-1]
    assert "--hud" in fa and fa[fa.index("--fps") + 1] == "60" and fa[fa.index("--duration") + 1] == "2"
    assert fa[fa.index("--params") + 1] == f"waves/wave02/c{s2['best_id'][-1]}/params.json"

    cert = jload(fin["certificate"])
    assert cert["verdict"] == "PASS" and cert["passed"] is True and cert["brief"] == BRIEF
    assert cert["candidate"]["id"] == s2["best_id"] and cert["params"]["id"] == s2["best_id"]
    assert cert["gate1"]["passed"] and cert["gate1"]["metrics"]["hazard_collisions"] == 0
    assert cert["gate2"]["passed"] and cert["gate2"]["critique_summary"] == s2["critique_summary"]
    assert cert["run"]["run_dir"] == run_dir and cert["final_video"] == os.path.basename(fin["final_video"])
    dt.datetime.fromisoformat(cert["generated_at"])

    report = open(fin["report"], encoding="utf-8").read()
    assert "## Convergence" in report and "Verdict: PASS" in report
    assert re.search(rf"\| 1 \| {s1['best_id']} \| .*\| FAIL \| FAIL \|", report)
    assert re.search(rf"\| 2 \| {s2['best_id']} \| .*\| PASS \| PASS \|", report)
    assert "## Director notes" in report and "| t (s) | Event | Note |" in report
    assert os.path.basename(fin["final_video"]) in report and os.path.basename(fin["certificate"]) in report
    run = jload(run_dir, "run.json")
    assert run["status"] == "finalized" and run["final"]["candidate"] == s2["best_id"]

    _, st, err = loop(tmp_path, "status", "--run-dir", run_dir)
    assert st["status"] == "finalized" and st["next"] == "gate3" and st["waves_done"] == 2
    assert [w["best_id"] for w in st["waves"]] == [s1["best_id"], s2["best_id"]]
    assert "wave  best" in err and "PASS" in err and "FAIL" in err


def test_checkpoint_every_fifth_wave_without_a_pass(tmp_path, image):
    env = make_env(tmp_path, FAKE_STUNT_PASS_FROM_WAVE=0)        # Gate 1 never passes
    run_dir = init_run(tmp_path, image, env=env)["run_dir"]
    nexts = [loop(tmp_path, "wave", "--run-dir", run_dir, env=env)[1]["next"] for _ in range(5)]
    assert nexts == ["wave", "wave", "wave", "wave", "checkpoint"]
    run = jload(run_dir, "run.json")
    assert run["status"] == "checkpoint" and len(run["waves"]) == 5

    # the director says "yes, more waves" with feedback: no hard cap, the loop goes on
    _, s6, _ = loop(tmp_path, "wave", "--run-dir", run_dir, "--steer", "move the mat 0.5 m out", env=env)
    assert s6["wave"] == 6 and s6["next"] == "wave" and jload(run_dir, "run.json")["status"] == "running"
    run = jload(run_dir, "run.json")                      # checkpoint_every is read from run.json
    run["checkpoint_every"] = 7
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run, f)
    assert loop(tmp_path, "wave", "--run-dir", run_dir, env=env)[1]["next"] == "checkpoint"

    # "No, finalize the best take": the certificate says FAIL with the Gate 1 failures
    _, fin, _ = loop(tmp_path, "finalize", "--run-dir", run_dir, env=env)
    cert = jload(fin["certificate"])
    assert fin["passed"] is False and cert["verdict"] == "FAIL" and cert["gate1"]["failures"]
    assert cert["candidate"]["wave"] in (6, 7)            # the note given before wave 6 restarted the competition
    assert "Verdict: FAIL" in open(fin["report"], encoding="utf-8").read()


def test_steer_becomes_a_standing_director_note(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    steer = "land softer, keep the 'roll' tight"
    _, s1, err = loop(tmp_path, "wave", "--run-dir", run_dir, "--steer", steer)
    assert s1["steer"] == steer and steer in err
    note1 = {"after_wave": 0, "text": steer}
    run = jload(run_dir, "run.json")
    assert run["director_notes"] == [note1] and run["waves"][0]["steer"] == steer
    h1 = jload(run_dir, "waves", "wave01", "history.json")
    assert h1["director_notes"] == [note1] and h1["current_steer"] == steer and h1["waves"] == []

    _, s2, _ = loop(tmp_path, "wave", "--run-dir", run_dir)
    assert s2["steer"] is None
    h2 = jload(run_dir, "waves", "wave02", "history.json")
    assert h2["director_notes"] == [note1] and h2["current_steer"] is None
    assert h2["waves"][0]["steer"] == steer and h2["waves"][0]["wave"] == 1

    # a steer given to a wave that crashed still applies when that wave is re-run without --steer
    steer2 = "use the side camera"
    rc, _, _ = loop(tmp_path, "wave", "--run-dir", run_dir, "--steer", steer2, ok=False,
                    env=make_env(tmp_path, FAKE_STUNT_CRASH="all"))
    assert rc == 2
    _, s3, _ = loop(tmp_path, "wave", "--run-dir", run_dir)
    assert s3["wave"] == 3 and s3["steer"] == steer2
    notes = [note1, {"after_wave": 2, "text": steer2}]
    assert jload(run_dir, "run.json")["director_notes"] == notes
    h3 = jload(run_dir, "waves", "wave03", "history.json")
    assert h3["director_notes"] == notes and h3["current_steer"] == steer2
    assert [w["steer"] for w in h3["waves"]] == [steer, None]
    run = jload(run_dir, "run.json")
    assert run["best"]["since_wave"] == 3 and run["best"]["wave"] == 3     # a note restarts the competition
    report = open(os.path.join(run_dir, "report.md"), encoding="utf-8").read()
    assert steer in report and steer2 in report and "## Director notes" in report


def test_steer_reaches_argon_propose(tmp_path, image, monkeypatch, capsys):
    run_dir = init_run(tmp_path, image)["run_dir"]
    for k, v in make_env(tmp_path).items():
        if k.startswith(("FALLGUY_", "FAKE_")):
            monkeypatch.setenv(k, v)
    loop_mod = import_scripts_module("loop")
    seen, real = [], loop_mod.call_argon

    def spy(rd, sub, args, out_rel, what):
        seen.append((sub, list(args)))
        return real(rd, sub, args, out_rel, what)

    monkeypatch.setattr(loop_mod, "call_argon", spy)
    assert loop_mod.main(["wave", "--run-dir", run_dir, "--steer", "go higher"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["steer"] == "go higher"
    (sub, args), = [x for x in seen if x[0] == "propose"]
    assert args[args.index("--steer") + 1] == "go higher" and args[args.index("--n") + 1] == "4"
    assert args[args.index("--history") + 1] == "waves/wave01/history.json"
    (sub, args), = [x for x in seen if x[0] == "critique"]
    assert args[args.index("--out") + 1] == "waves/wave01/critique.json" and args[args.index("--fps") + 1] == "5"
    assert loop_mod.main(["wave", "--run-dir", run_dir]) == 0              # no steer -> no --steer flag
    capsys.readouterr()
    assert "--steer" not in [x for x in seen if x[0] == "propose"][-1][1]


def test_all_candidates_crash_exits_2_and_rerun_recovers(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    rc, out, err = loop(tmp_path, "wave", "--run-dir", run_dir, ok=False, env=make_env(tmp_path, FAKE_STUNT_CRASH="all"))
    assert rc == 2 and out["exit_code"] == 2
    script = jload(run_dir, "run.json")["script"]
    assert "all 4 candidates failed" in out["error"] and "fake crash" in out["error"] and script in out["error"]
    assert out["script"] == os.path.join(run_dir, script)
    assert jload(run_dir, "run.json")["waves"] == []
    assert jload(run_dir, "waves", "wave01", "c1", "metrics.json")["ok"] is False

    _, s1, err = loop(tmp_path, "wave", "--run-dir", run_dir)
    assert s1["wave"] == 1 and "unfinished earlier attempt" in err
    assert any(d.startswith("wave01_stale_") for d in os.listdir(os.path.join(run_dir, "waves")))


def test_one_crash_and_one_timeout_do_not_stop_the_wave(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    env = make_env(tmp_path, FAKE_STUNT_CRASH="3", FAKE_STUNT_SLEEP="2:60", FALLGUY_SIM_TIMEOUT=3)
    _, s1, _ = loop(tmp_path, "wave", "--run-dir", run_dir, env=env, timeout=60)
    rows = {c["id"]: c for c in s1["candidates"]}
    assert rows["w1c2"]["ok"] is False and "timed out after 3 s" in rows["w1c2"]["error"]
    assert rows["w1c3"]["ok"] is False and "fake crash" in rows["w1c3"]["error"]
    assert rows["w1c1"]["ok"] and rows["w1c4"]["ok"] and s1["best_id"] in ("w1c1", "w1c4")
    assert jload(run_dir, "waves", "wave01", "c2", "metrics.json")["ok"] is False


def test_argon_failure_exits_3_after_one_retry(tmp_path, image):
    # real argon.py without mock and without a key file: it must fail, twice, with a one-line error
    env = make_env(tmp_path, FALLGUY_ARGON_MOCK=0, FALLGUY_ARGON_TOKEN_FILE=str(tmp_path / "no_such_token"))
    rc, out, err = loop(tmp_path, "init", "--media", image, "--brief", BRIEF, "--duration", "2", ok=False, env=env)
    assert rc == 3 and out["exit_code"] == 3 and out["error"].startswith("Argon scene failed twice")
    assert "attempt 1/2" in err and "attempt 2/2" in err and "\n" not in out["error"]
    # the error names the half-made run dir; its logs explain the failure and it has no run.json
    assert os.path.isfile(os.path.join(out["run_dir"], "logs", "argon_scene.log"))
    assert not os.path.exists(os.path.join(out["run_dir"], "run.json"))


def test_redact_scrubs_the_key(tmp_path, monkeypatch):
    token = tmp_path / "tok"
    token.write_text("my-secret-key-123456\n")
    monkeypatch.setenv("FALLGUY_ARGON_TOKEN_FILE", str(token))
    loop_mod = import_scripts_module("loop")
    text = ("HTTP 400 my-secret-key-123456 https://x/y?key=abc123&alt=json "
            f"{FAKE_KEY} x-goog-api-key: zzz999")
    red = loop_mod.redact(text)
    assert "my-secret-key-123456" not in red and "abc123" not in red and FAKE_KEY not in red and "zzz999" not in red
    assert red.count("<KEY>") == 4


# ----------------------------------------------------------------------------- status, gates, errors
def test_status_of_fresh_run_and_gates(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    _, st, err = loop(tmp_path, "status", "--run-dir", run_dir)
    assert st["status"] == "planned" and st["next"] == "wave" and st["waves"] == [] and st["checkpoint_every"] == 5
    assert "(no waves yet)" in err

    _, g, err = loop(tmp_path, "gates")
    th = g["thresholds"]
    assert th["head_g_max"] == 8.0 and th["hazard_collisions_max"] == 0 and th["min_clearance_cm"] == 2.0
    assert th["landing_error_max_m"] == 0.65 and th["framing_min_pct"] == 85.0 and th["gate2_min_score"] == 85.0
    assert g["weights"] == {"safety": 0.35, "clearance": 0.25, "landing": 0.20, "framing": 0.20}
    assert g["description"].strip() and g["description"].strip() in err


@pytest.mark.parametrize("args,needle", [
    (["status", "--run-dir", "nope"], "run dir not found"),
    (["wave", "--run-dir", "."], "has no run.json"),
    (["init", "--media", "missing.jpg", "--brief", "x", "--duration", "5"], "media not found"),
    (["init", "--brief", "x", "--duration", "5"], "--media"),
    (["init", "--media", "notes.txt", "--brief", "x", "--duration", "5"], "unsupported media type"),
    (["init", "--media", "notes.txt", "--brief", "  ", "--duration", "5"], "--brief is empty"),
    (["init", "--media", "notes.txt", "--brief", "x", "--duration", "500"], "out of range"),
])
def test_bad_input_exits_1_with_json(tmp_path, args, needle):
    (tmp_path / "notes.txt").write_text("not media")
    rc, out, err = loop(tmp_path, *args, ok=False)
    assert rc == 1 and out["exit_code"] == 1 and needle in out["error"] and needle in err


def test_finalize_needs_a_wave_and_a_valid_candidate(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    rc, out, _ = loop(tmp_path, "finalize", "--run-dir", run_dir, ok=False)
    assert rc == 1 and "no waves yet" in out["error"]
    loop(tmp_path, "wave", "--run-dir", run_dir)
    rc, out, _ = loop(tmp_path, "finalize", "--run-dir", run_dir, "--candidate", "bogus", ok=False)
    assert rc == 1 and "bad candidate id" in out["error"]


def test_finalize_other_candidate_gets_its_own_critique(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    best = loop(tmp_path, "wave", "--run-dir", run_dir)[1]["best_id"]
    other = next(f"w1c{i}" for i in range(1, 5) if f"w1c{i}" != best)
    _, fin, err = loop(tmp_path, "finalize", "--run-dir", run_dir, "--candidate", other)
    cert = jload(fin["certificate"])
    assert cert["candidate"]["id"] == other and cert["gate2"]["critique"] == "final/critique.json"
    assert os.path.isfile(os.path.join(run_dir, "final", "critique.json")) and "Argon watches the final video" in err


def test_concurrent_command_is_refused(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with open(os.path.join(run_dir, ".loop.lock"), "w") as f:
            json.dump({"pid": holder.pid, "command": "wave", "since": "now"}, f)
        rc, out, _ = loop(tmp_path, "wave", "--run-dir", run_dir, ok=False)
        assert rc == 1 and "is working on this run" in out["error"]
        _, st, _ = loop(tmp_path, "status", "--run-dir", run_dir)
        assert st["busy"]["pid"] == holder.pid
    finally:
        holder.kill()
        holder.wait()
    _, s1, _ = loop(tmp_path, "wave", "--run-dir", run_dir)          # a dead holder's lock is taken over
    assert s1["wave"] == 1 and not os.path.exists(os.path.join(run_dir, ".loop.lock"))


def test_report_cli_rewrites_report(tmp_path, image):
    run_dir = init_run(tmp_path, image)["run_dir"]
    loop(tmp_path, "wave", "--run-dir", run_dir)
    os.remove(os.path.join(run_dir, "report.md"))
    p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "report.py"), "--run-dir", run_dir],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and json.loads(p.stdout)["report"] == os.path.join(run_dir, "report.md")
    text = open(os.path.join(run_dir, "report.md"), encoding="utf-8").read()
    assert "Not finalized yet" in text and "## Convergence" in text


# ----------------------------------------------------------------------------- setup_env.sh
def _setup(tmp_path, **env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("FALLGUY_")}
    env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(["bash", os.path.join(SKILL, "scripts", "setup_env.sh")], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)


def _h1_dir():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(SKILL)))
    for d in (os.environ.get("FALLGUY_H1_DIR"), os.path.join(repo, "models", "mujoco_menagerie", "unitree_h1")):
        if d and os.path.isfile(os.path.join(d, "h1.xml")):
            return d
    pytest.skip("Unitree H1 model not available")


def test_setup_env_prints_python_and_h1(tmp_path):
    token = tmp_path / "tok"
    token.write_text(FAKE_KEY + "\n")
    token.chmod(0o644)
    p = _setup(tmp_path, FALLGUY_PYTHON=sys.executable, FALLGUY_H1_DIR=_h1_dir(), FALLGUY_ARGON_TOKEN_FILE=token)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [f"FALLGUY_PYTHON={sys.executable}", f"FALLGUY_H1_DIR={_h1_dir()}"]
    assert "wider than 600" in p.stderr and FAKE_KEY not in p.stdout + p.stderr


@pytest.mark.parametrize("extra,needle", [
    ({"FALLGUY_H1_DIR": "/nonexistent/h1"}, "has no h1.xml"),
    ({"FALLGUY_ARGON_TOKEN_FILE": "/nonexistent/token"}, "no Argon API key"),
    ({"FALLGUY_PYTHON": "/nonexistent/python"}, "is not an executable"),
])
def test_setup_env_fails_loudly(tmp_path, extra, needle):
    token = tmp_path / "tok"
    token.write_text("k\n")
    token.chmod(0o600)
    env = {"FALLGUY_PYTHON": sys.executable, "FALLGUY_H1_DIR": _h1_dir(), "FALLGUY_ARGON_TOKEN_FILE": token}
    env.update(extra)
    p = _setup(tmp_path, **env)
    assert p.returncode == 1 and needle in p.stderr and p.stdout == ""
