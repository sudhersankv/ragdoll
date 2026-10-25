"""Tests for scripts/argon.py, the Argon client.

No network: HTTP behaviour (request shape, retries, timeouts, Files API upload, redaction) runs against a local fake
Gemini server, and the CLI runs in FALLGUY_ARGON_MOCK=1 mode. FALLGUY_ARGON_LIVE=1 adds one real ping.
Run: .venv/bin/python -m pytest .claude/skills/fallguy-stunt-loop/tests/test_argon.py -q
"""
import base64
import copy
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(SKILL, "scripts")
FIXTURES = os.path.join(SCRIPTS, "fixtures", "argon_mock")
sys.path.insert(0, SCRIPTS)
import argon  # noqa: E402

P = argon.engine_params()
PY = sys.executable
FAKE_KEY = "AIzaFAKEtestKEY0123456789abcdefGHIJ"
CONTRACT_PLAN_KEYS = {"version", "brief", "duration_sec", "source", "room", "obstacles", "set_dressing", "crash_mats",
                      "camera", "stunt", "performer"}
CONTRACT_CRITIQUE_KEYS = {"summary", "events", "brief_match", "rotation_observed_deg", "unnatural_motion", "framing",
                          "style_score", "gate2_pass", "proposed_changes"}


def load_fixture(name):
    with open(os.path.join(FIXTURES, f"{name}.json")) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    key_file = tmp_path / "argon_token"
    key_file.write_text(FAKE_KEY + "\n")
    for k in ("FALLGUY_ARGON_MOCK", "FALLGUY_ARGON_MOCK_DIR", "ARGON_MODEL", "ARGON_THINKING", "FALLGUY_ARGON_BASE_URL",
              "FALLGUY_ARGON_INLINE_MAX_MB", "FALLGUY_ARGON_RAW_DIR", "FALLGUY_ARGON_TIMEOUT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FALLGUY_ARGON_TOKEN_FILE", str(key_file))
    monkeypatch.setenv("FALLGUY_ARGON_BACKOFF", "0")
    monkeypatch.setenv("FALLGUY_ARGON_POLL_S", "0")
    monkeypatch.setenv("FALLGUY_ARGON_QUIET", "1")
    argon._KEY = None
    yield
    argon._KEY = None


# ----------------------------------------------------------------------------- fake Gemini server
def gen_response(answer, thought=True, finish="STOP"):
    parts = ([{"text": "**Thinking** {not json", "thought": True}] if thought else [])
    parts.append({"text": answer if isinstance(answer, str) else json.dumps(answer), "thoughtSignature": "c2ln" * 80})
    return {"candidates": [{"content": {"parts": parts, "role": "model"}, "finishReason": finish, "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 40,
                              "thoughtsTokenCount": 25},
            "modelVersion": "gemini-4-argon", "responseId": "r1"}


class FakeGemini:
    """Routes: generate, upload_start, upload_finalize, file_get. Each route pops queued replies, then a default."""

    def __init__(self):
        self.requests = []
        self.queue = {"generate": [], "upload_start": [], "upload_finalize": [], "file_get": []}
        self.answer = {"pong": True}
        self.delay = 0.0
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _route(self):
                p = self.path
                if ":generateContent" in p:
                    return "generate"
                if p.startswith("/upload/v1beta/files"):
                    return "upload_start"
                if p.startswith("/upload/session/"):
                    return "upload_finalize"
                if p.startswith("/v1beta/files/"):
                    return "file_get"
                return None

            def _do(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                route = self._route()
                fake.requests.append({"route": route, "method": self.command, "path": self.path,
                                      "headers": {k.lower(): v for k, v in self.headers.items()}, "raw": raw})
                if fake.delay:
                    time.sleep(fake.delay)
                status, body, headers = fake.reply(route)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _do

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def push(self, route, status=200, body=None, headers=None):
        self.queue[route].append((status, body if body is not None else {}, headers or {}))

    def reply(self, route):
        if route in self.queue and self.queue[route]:
            return self.queue[route].pop(0)
        if route == "generate":
            return 200, gen_response(self.answer), {}
        if route == "upload_start":
            return 200, {}, {"X-Goog-Upload-URL": f"{self.url}/upload/session/s1?upload_id=u1"}
        if route == "upload_finalize":
            return 200, {"file": {"name": "files/abc", "uri": f"{self.url}/v1beta/files/abc", "state": "PROCESSING"}}, {}
        if route == "file_get":
            return 200, {"name": "files/abc", "uri": f"{self.url}/v1beta/files/abc", "state": "ACTIVE",
                         "expirationTime": "2026-10-27T00:00:00Z"}, {}
        return 404, {"error": {"code": 404, "message": "no route", "status": "NOT_FOUND"}}, {}

    def routes(self):
        return [r["route"] for r in self.requests]

    def bodies(self, route):
        return [json.loads(r["raw"]) for r in self.requests if r["route"] == route]


@pytest.fixture
def fake(monkeypatch):
    f = FakeGemini()
    monkeypatch.setenv("FALLGUY_ARGON_BASE_URL", f.url)
    yield f
    f.server.shutdown()
    f.server.server_close()


def read_log(path):
    with open(path) as f:
        text = f.read()
    assert FAKE_KEY not in text
    return [json.loads(line) for line in text.splitlines()]


SCHEMA = {"type": "object", "properties": {"pong": {"type": "boolean"}}, "required": ["pong"]}


# ----------------------------------------------------------------------------- key, redaction, answer parsing
def test_key_loading_and_redaction(tmp_path, monkeypatch):
    assert argon.load_key() == FAKE_KEY
    assert argon.redact(f"oops {FAKE_KEY} leaked") == "oops <KEY> leaked"
    assert "<KEY>" in argon.redact("https://x/y?key=abc123&alt=json") and "abc123" not in argon.redact("?key=abc123")
    named = tmp_path / "named"
    named.write_text(f"# comment\nGEMINI_API_KEY={FAKE_KEY}\n")
    monkeypatch.setenv("FALLGUY_ARGON_TOKEN_FILE", str(named))
    argon._KEY = None
    assert argon.load_key() == FAKE_KEY
    empty = tmp_path / "empty"
    empty.write_text("\n")
    monkeypatch.setenv("FALLGUY_ARGON_TOKEN_FILE", str(empty))
    argon._KEY = None
    with pytest.raises(argon.ArgonError, match="is empty"):
        argon.load_key()
    monkeypatch.setenv("FALLGUY_ARGON_TOKEN_FILE", str(tmp_path / "missing"))
    argon._KEY = None
    with pytest.raises(argon.ArgonError, match="does not exist"):
        argon.load_key()


def test_answer_comes_from_non_thought_parts():
    resp = gen_response({"pong": True})
    assert argon.parse_answer(resp, "t") == {"pong": True}
    fenced = gen_response('```json\n{"pong": true}\n```', thought=False)
    assert argon.parse_answer(fenced, "t") == {"pong": True}


def test_parse_answer_errors_are_specific():
    with pytest.raises(argon.ArgonError, match="blockReason=SAFETY"):
        argon.parse_answer({"promptFeedback": {"blockReason": "SAFETY"}}, "scene")
    with pytest.raises(argon.ArgonError, match="declined"):
        argon.parse_answer(gen_response({"a": 1}, finish="SAFETY"), "scene")
    with pytest.raises(argon.ArgonError) as e:
        argon.parse_answer(gen_response("not json", finish="MAX_TOKENS"), "scene")
    assert e.value.retry and "MAX_TOKENS" in str(e.value)
    with pytest.raises(argon.ArgonError) as e:
        argon.parse_answer(gen_response("[1, 2]"), "scene")
    assert e.value.retry and "expected an object" in str(e.value)


# ----------------------------------------------------------------------------- HTTP behaviour
def test_generate_request_shape_and_log(fake, tmp_path):
    log = str(tmp_path / "logs" / "argon_calls.jsonl")
    ans, resp = argon.generate([{"text": "hi"}], SCHEMA, "ping", log)
    assert ans == {"pong": True} and resp["modelVersion"] == "gemini-4-argon"
    (req,) = fake.requests
    assert req["path"] == "/v1beta/models/gemini-4-argon:generateContent"
    assert req["headers"]["x-goog-api-key"] == FAKE_KEY
    assert FAKE_KEY not in req["path"]
    body = json.loads(req["raw"])
    gc = body["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": "high"}
    assert gc["responseMimeType"] == "application/json" and gc["responseJsonSchema"] == SCHEMA
    assert body["contents"][0]["parts"] == [{"text": "hi"}]
    (rec,) = read_log(log)
    assert rec["label"] == "ping" and rec["status"] == 200 and rec["usage"]["thoughtsTokenCount"] == 25
    assert rec["model"] == "gemini-4-argon" and rec["thinking"] == "high" and "latency_s" in rec


def test_model_and_thinking_overrides(fake, monkeypatch):
    monkeypatch.setenv("ARGON_MODEL", "gemini-test-model")
    monkeypatch.setenv("ARGON_THINKING", "LOW")
    argon.generate([{"text": "hi"}], SCHEMA, "ping", None)
    req = fake.requests[0]
    assert req["path"] == "/v1beta/models/gemini-test-model:generateContent"
    assert json.loads(req["raw"])["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"


def test_retries_on_503_and_429_then_succeeds(fake, tmp_path):
    log = str(tmp_path / "calls.jsonl")
    fake.push("generate", 503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
    fake.push("generate", 429, {"error": {"code": 429, "message": "slow down", "status": "RESOURCE_EXHAUSTED",
                                          "details": [{"retryDelay": "0s"}]}}, {"Retry-After": "0"})
    ans, _ = argon.generate([{"text": "hi"}], SCHEMA, "ping", log)
    assert ans == {"pong": True}
    assert fake.routes() == ["generate"] * 3
    assert [r["status"] for r in read_log(log)] == [503, 429, 200]


def test_gives_up_after_three_tries(fake):
    for _ in range(3):
        fake.push("generate", 500, {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}})
    with pytest.raises(argon.ArgonError, match="HTTP 500 after 3 tries") as e:
        argon.generate([{"text": "hi"}], SCHEMA, "ping", None)
    assert e.value.status == 500 and len(fake.requests) == 3


def test_400_is_not_retried_and_the_key_is_redacted(fake, tmp_path):
    log = str(tmp_path / "calls.jsonl")
    fake.push("generate", 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                          "message": f"API key {FAKE_KEY} cannot use thinking level MINIMAL"}})
    with pytest.raises(argon.ArgonError) as e:
        argon.generate([{"text": "hi"}], SCHEMA, "ping", log)
    assert len(fake.requests) == 1 and e.value.status == 400
    assert FAKE_KEY not in str(e.value) and "<KEY>" in str(e.value) and "INVALID_ARGUMENT" in str(e.value)
    assert read_log(log)[0]["status"] == 400


def test_schema_is_sent_lean_and_a_bare_400_retries_with_a_minimal_one(fake):
    schema = {"type": "object", "required": ["pong", "maximum"], "additionalProperties": False,
              "properties": {"pong": {"type": "boolean", "description": "always true"},
                             "maximum": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                         "minItems": 3, "maxItems": 3}}}
    fake.push("generate", 400, {"error": {"code": 400, "message": "Request contains an invalid argument.",
                                          "status": "INVALID_ARGUMENT"}})
    ans, _ = argon.generate([{"text": "hi"}], schema, "ping", None)
    assert ans == {"pong": True}
    first, second = (b["generationConfig"]["responseJsonSchema"] for b in fake.bodies("generate"))
    assert first["properties"]["maximum"] == {"type": "array", "items": {"type": "number"}}     # property name kept
    assert first["properties"]["pong"] == {"type": "boolean", "description": "always true"}
    assert second["properties"]["pong"] == {"type": "boolean"} and second["required"] == ["pong", "maximum"]
    fake.push("generate", 400, {"error": {"code": 400, "message": "Request contains an invalid argument.",
                                          "status": "INVALID_ARGUMENT"}})
    fake.push("generate", 400, {"error": {"code": 400, "message": "Request contains an invalid argument.",
                                          "status": "INVALID_ARGUMENT"}})
    with pytest.raises(argon.ArgonError, match="HTTP 400"):
        argon.generate([{"text": "hi"}], schema, "ping", None)
    assert len(fake.requests) == 4


def test_unusable_answer_is_asked_once_more(fake):
    fake.push("generate", 200, gen_response("{truncated", finish="MAX_TOKENS"))
    ans, _ = argon.generate([{"text": "hi"}], SCHEMA, "ping", None)
    assert ans == {"pong": True} and len(fake.requests) == 2


def test_unreachable_server_fails_after_three_tries(monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                      # nothing listens there now
    monkeypatch.setenv("FALLGUY_ARGON_BASE_URL", f"http://127.0.0.1:{port}")
    with pytest.raises(argon.ArgonError, match="cannot reach Argon after 3 tries"):
        argon.generate([{"text": "hi"}], SCHEMA, "ping", None)


def test_timeout_is_reported_not_retried(fake, monkeypatch):
    fake.delay = 1.5
    monkeypatch.setenv("FALLGUY_ARGON_TIMEOUT", "0.3")
    with pytest.raises(argon.ArgonError, match="no answer from Argon within 0 s|within 0 s"):
        argon.generate([{"text": "hi"}], SCHEMA, "ping", None)
    assert len(fake.requests) == 1


# ----------------------------------------------------------------------------- media
@pytest.fixture(scope="session")
def tiny_mp4(tmp_path_factory):
    import imageio.v2 as imageio
    import numpy as np
    path = str(tmp_path_factory.mktemp("vid") / "tiny.mp4")
    with imageio.get_writer(path, fps=10, codec="libx264", macro_block_size=1) as w:
        for i in range(20):
            w.append_data(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    return path


def test_media_kind(tmp_path, tiny_mp4):
    from PIL import Image
    assert argon.media_kind(tiny_mp4) == "video"
    png = tmp_path / "shot.HEIC.png"
    Image.new("RGB", (8, 8)).save(png)
    assert argon.media_kind(str(png)) == "image"
    odd = tmp_path / "frame.data"
    Image.new("RGB", (8, 8)).save(odd, "PNG")
    assert argon.media_kind(str(odd)) == "image"
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(argon.ArgonError, match="not an image or a supported video"):
        argon.media_kind(str(txt))
    with pytest.raises(argon.ArgonError, match="media not found"):
        argon.media_kind(str(tmp_path / "nope.jpg"))


def test_image_part_downscales_and_reads_exif_fov(tmp_path):
    from PIL import Image
    src = tmp_path / "phone.jpg"
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 24          # FocalLengthIn35mmFilm: iPhone main camera
    Image.new("RGB", (3200, 2400), (90, 120, 150)).save(src, exif=exif.tobytes())
    part, info = argon.image_part(str(src))
    assert part["inlineData"]["mimeType"] == "image/jpeg"
    im = Image.open(io.BytesIO(base64.b64decode(part["inlineData"]["data"])))
    assert im.format == "JPEG" and max(im.size) == 1600 and info["width"] == 1600 and info["height"] == 1200
    assert info["fovy_exif"] == pytest.approx(56.8, abs=0.2)
    plain = tmp_path / "plain.png"
    Image.new("RGB", (800, 600)).save(plain)
    assert argon.image_part(str(plain))[1]["fovy_exif"] is None


@pytest.mark.skipif(not shutil.which("sips"), reason="needs macOS sips")
def test_heic_goes_through_sips(tmp_path):
    from PIL import Image
    jpg, heic = tmp_path / "a.jpg", tmp_path / "a.heic"
    Image.new("RGB", (2000, 1500), (200, 30, 30)).save(jpg)
    subprocess.run(["sips", "-s", "format", "heic", str(jpg), "--out", str(heic)], check=True, capture_output=True)
    part, info = argon.image_part(str(heic))
    assert (info["width"], info["height"]) == (1600, 1200)
    assert Image.open(io.BytesIO(base64.b64decode(part["inlineData"]["data"]))).format == "JPEG"


def test_small_video_goes_inline(tiny_mp4):
    part, how = argon.video_part(tiny_mp4, 5.0, None, None)
    assert how == "inline" and part["videoMetadata"] == {"fps": 5.0}
    with open(tiny_mp4, "rb") as f:
        assert base64.b64decode(part["inlineData"]["data"]) == f.read()
    assert part["inlineData"]["mimeType"] == "video/mp4"
    dur, fps = argon.video_info(tiny_mp4)
    assert dur == pytest.approx(2.0, abs=0.15) and fps == pytest.approx(10.0)


def test_large_video_uploads_polls_and_is_cached(fake, tmp_path, tiny_mp4, monkeypatch):
    monkeypatch.setenv("FALLGUY_ARGON_INLINE_MAX_MB", "0.0001")      # ~100 bytes: force the Files API
    log = str(tmp_path / "logs" / "argon_calls.jsonl")
    cache = argon.files_cache_path(log)
    part, how = argon.video_part(tiny_mp4, 5.0, log, cache)
    assert how == "file" and part["fileData"]["fileUri"].endswith("/v1beta/files/abc")
    assert part["videoMetadata"] == {"fps": 5.0}
    assert fake.routes() == ["upload_start", "upload_finalize", "file_get"]
    start, fin = fake.requests[0], fake.requests[1]
    size = os.path.getsize(tiny_mp4)
    assert start["headers"]["x-goog-upload-protocol"] == "resumable"
    assert start["headers"]["x-goog-upload-command"] == "start"
    assert start["headers"]["x-goog-upload-header-content-length"] == str(size)
    assert start["headers"]["x-goog-upload-header-content-type"] == "video/mp4"
    assert json.loads(start["raw"])["file"]["display_name"] == "tiny.mp4"
    assert fin["headers"]["x-goog-upload-command"] == "upload, finalize" and fin["headers"]["x-goog-upload-offset"] == "0"
    with open(tiny_mp4, "rb") as f:
        assert fin["raw"] == f.read()
    with open(cache) as f:
        entry = next(iter(json.load(f).values()))
    assert entry["name"] == "files/abc" and entry["size"] == size
    labels = [r["label"] for r in read_log(log)]
    assert labels == ["upload_start", "upload_finalize", "file_poll"]
    assert all("upload_id" not in r["url"] for r in read_log(log))
    # second use within 47 h: no new upload
    part2, how2 = argon.video_part(tiny_mp4, 5.0, log, cache)
    assert how2 == "file" and part2["fileData"] == part["fileData"] and len(fake.requests) == 3
    # stale entry (48 h old): uploads again
    with open(cache) as f:
        c = json.load(f)
    for v in c.values():
        v["uploaded_at"] -= 48 * 3600
    with open(cache, "w") as f:
        json.dump(c, f)
    argon.video_part(tiny_mp4, 5.0, log, cache)
    assert fake.routes()[3:] == ["upload_start", "upload_finalize", "file_get"]


def test_inline_video_refused_as_too_large_falls_back_to_upload(fake, tmp_path, tiny_mp4):
    fake.push("generate", 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                          "message": "Request payload size exceeds the limit: 20971520 bytes."}})
    ans, _ = argon.generate_with_video(tiny_mp4, 5.0, lambda vp: [vp, {"text": "judge"}], SCHEMA, "critique", None,
                                       str(tmp_path / "files.json"))
    assert ans == {"pong": True}
    assert fake.routes() == ["generate", "upload_start", "upload_finalize", "file_get", "generate"]
    first, last = fake.bodies("generate")
    assert "inlineData" in first["contents"][0]["parts"][0]
    assert last["contents"][0]["parts"][0]["fileData"]["fileUri"].endswith("/files/abc")
    assert last["contents"][0]["parts"][1] == {"text": "judge"}


# ----------------------------------------------------------------------------- plan normalisation
def validate_plan(plan):
    """Contract checks every plan.json must pass (reference/contracts.md)."""
    assert CONTRACT_PLAN_KEYS <= set(plan)
    assert plan["version"] == 1 and isinstance(plan["duration_sec"], float)
    assert plan["source"]["kind"] in ("image", "video") and isinstance(plan["source"]["ref_time_sec"], float)
    room = plan["room"]
    for k in ("x_min", "x_max", "y_min", "y_max", "ceiling_z"):
        assert isinstance(room[k], float)
    assert room["x_min"] < room["x_max"] and room["y_min"] < room["y_max"]
    for k in ("floor_rgb", "wall_rgb", "ceiling_rgb"):
        assert len(room[k]) == 3 and all(0 <= c <= 1 for c in room[k])
    ids = [o["id"] for o in plan["obstacles"]]
    assert len(ids) == len(set(ids)) and ids
    for o in plan["obstacles"]:
        assert set(o) == {"id", "kind", "pos", "size", "yaw_deg", "rgb", "material", "role", "hazard"}
        assert o["kind"] in argon.KINDS and o["material"] in argon.MATERIALS and o["role"] in argon.ROLES
        assert isinstance(o["hazard"], bool) and len(o["pos"]) == 3 and len(o["size"]) == 3
        need = {"box": 3, "cylinder": 2, "sphere": 1}[o["kind"]]
        assert all(s > 0 for s in o["size"][:need]), o
        b = argon.aabb(o)
        assert b[4] >= -0.011, o                                         # nothing below the floor
        assert room["x_min"] <= b[0] and b[1] <= room["x_max"] and room["y_min"] <= b[2] and b[3] <= room["y_max"]
    tgt = [o for o in plan["obstacles"] if o["role"] == "stunt_target"]
    assert tgt and all(o["hazard"] for o in tgt)
    st = plan["stunt"]
    assert st["type"] in P.STUNT_TYPES
    assert st["start_x"] < st["takeoff_x"] < st["landing_x"]
    t = argon._union([argon.aabb(o) for o in tgt])
    if st["type"] != "high_fall":
        assert st["takeoff_x"] < t[0] and st["landing_x"] > t[1]
    assert plan["crash_mats"]
    on_mat = False
    for m in plan["crash_mats"]:
        assert set(m) == {"id", "pos", "size", "rgb"} and all(s > 0 for s in m["size"])
        assert m["pos"][2] == pytest.approx(m["size"][2])               # resting on the floor
        on_mat |= abs(st["landing_x"] - m["pos"][0]) <= m["size"][0] and abs(st["lane_y"] - m["pos"][1]) <= m["size"][1]
    assert on_mat, "the landing point must be on a mat"
    cam = plan["camera"]["match"]
    assert len(cam["pos"]) == 3 and len(cam["lookat"]) == 3 and 10 <= cam["fovy"] <= 120
    assert room["x_min"] <= cam["pos"][0] <= room["x_max"] and room["y_min"] <= cam["pos"][1] <= room["y_max"]
    sd = plan["set_dressing"]
    assert set(sd) == {"curtain", "couches", "chairs", "people", "speakers", "ceiling_fixtures", "screens"}
    for p in sd["people"]:
        assert p["posture"] in ("seated", "standing")
    assert plan["performer"]["model"] in ("h1", "humanoid")


def norm(ans, **kw):
    args = dict(brief="barrel roll over the table", duration=9, media="inputs/media.jpg", kind="image",
                ref_frame="inputs/ref_frame.jpg", stunt_types=P.STUNT_TYPES)
    args.update(kw)
    return argon.normalize_plan(copy.deepcopy(ans), **args)


def test_mock_scene_fixture_is_schema_valid_and_needs_no_fixes():
    ans = load_fixture("scene")
    assert argon.schema_errors(ans, argon.scene_schema(P.STUNT_TYPES)) == []
    plan, warns = norm(ans)
    assert warns == []
    validate_plan(plan)
    assert plan["source"] == {"media": "inputs/media.jpg", "kind": "image", "ref_frame": "inputs/ref_frame.jpg",
                              "ref_time_sec": 0.0}
    assert plan["brief"] == "barrel roll over the table" and plan["duration_sec"] == 9.0
    assert plan["stunt"]["type"] == "barrel_roll" and plan["performer"] == {"model": "h1"}
    assert len(plan["argon"]["stunt_target"]) == 11


def test_normalize_repairs_a_bad_answer():
    ans = {
        "notes": "messy",
        "stunt": {"type": "Barrel Roll", "start_x": -1.5, "takeoff_x": 0.1, "landing_x": 0.2, "lane_y": 5.0},
        "obstacles": [
            {"id": "Table Top", "kind": "box", "pos": [0, 0, 0.74], "size": [0.4, -0.8, 0.02], "rgb": [140, 100, 60],
             "material": "oak", "role": "furniture", "hazard": False},
            {"id": "table top", "kind": "capsule", "pos": [0, 0.5, -0.3], "size": [0.05, 0.36, 0], "role": "prop"},
            {"id": "vase", "kind": "cylinder", "pos": [0.0, -0.3, 0.85], "size": [0.06, 0.1, 0], "hazard": "false"},
            {"id": "far_cabinet", "kind": "box", "pos": [0, 6.0, 0.5], "size": [0.3, 0.3, 0.5], "hazard": False},
            "junk",
        ],
        "crash_mats": [],
        "set_dressing": {"curtain": {"present": False}, "chairs": [{"x": 3, "y": 3}, {"y": 2}, {"x": 90, "y": 0}],
                         "people": [{"x": 3.5, "y": -2, "posture": "standing", "seat_z": 0.9}]},
        "room": {"x_min": 2, "x_max": -2, "y_min": -1, "y_max": 1, "ceiling_z": 0.5, "floor_rgb": [0.5, 0.5, 0.5]},
        "camera": {"match": {"pos": [-6, 0, 1.5], "lookat": [-6, 0, 1.5], "fovy": 500}},
        "source": {"ref_time_sec": 0},
    }
    plan, warns = norm(ans)
    validate_plan(plan)
    text = "\n".join(warns)
    ob = {o["id"]: o for o in plan["obstacles"]}
    assert set(ob) == {"table_top", "table_top_2", "vase", "far_cabinet"}
    assert ob["table_top"]["size"][1] == 0.8 and ob["table_top"]["rgb"] == [0.549, 0.392, 0.235]
    assert ob["table_top"]["material"] == "wood" and ob["table_top_2"]["kind"] == "box"
    assert ob["table_top_2"]["pos"][2] == ob["table_top_2"]["size"][2]          # raised onto the floor
    assert plan["stunt"]["type"] == "barrel_roll"                                # "Barrel Roll" understood
    assert ob["table_top"]["role"] == "stunt_target" and ob["table_top"]["hazard"] is True
    assert ob["vase"]["hazard"] is True and "vase lies in the stunt corridor" in text
    assert ob["far_cabinet"]["hazard"] is False                                  # 6 m off the lane
    st = plan["stunt"]
    assert st["lane_y"] == 0.8 and st["takeoff_x"] < -0.4 and st["landing_x"] > 0.4 and st["start_x"] <= st["takeoff_x"] - 1.5
    assert plan["crash_mats"][0]["id"] == "mat1" and "no crash mat" in text
    assert plan["camera"]["match"]["fovy"] == 120 and plan["camera"]["match"]["lookat"] != [-6, 0, 1.5]
    assert plan["room"]["x_min"] <= -6.3 and plan["room"]["ceiling_z"] >= 2.2
    assert plan["set_dressing"]["curtain"] == {}
    assert [(c["x"], c["y"]) for c in plan["set_dressing"]["chairs"]] == [(3.0, 3.0)]
    assert "set_dressing.chairs[1] dropped" in text and "outside the room" in text
    assert plan["set_dressing"]["people"][0]["posture"] == "standing"
    assert "obstacle 4: not an object" in text


def test_normalize_requires_obstacles():
    with pytest.raises(argon.ArgonError, match="no obstacles"):
        norm({"obstacles": []})


def test_mat_overlapping_the_target_is_moved_beyond_it():
    ans = load_fixture("scene")
    ans["crash_mats"] = [{"id": "mat1", "pos": [0.3, 0.0, 0.15], "size": [0.9, 1.0, 0.15], "rgb": [0, 0, 1]}]
    ans["stunt"]["landing_x"] = 0.6
    plan, warns = norm(ans)
    validate_plan(plan)
    m = plan["crash_mats"][0]
    assert m["pos"][0] - m["size"][0] >= 0.17 and any("overlapped the stunt target" in w for w in warns)


def test_video_source_ref_time():
    ans = load_fixture("scene")
    ans["source"]["ref_time_sec"] = 99.0
    plan, _ = norm(ans, kind="video", media="inputs/media.mp4", ref_frame=None, video_sec=12.0)
    assert plan["source"] == {"media": "inputs/media.mp4", "kind": "video", "ref_frame": None, "ref_time_sec": 12.0}
    plan, _ = norm(ans, kind="video", media="inputs/media.mp4", ref_frame=None, video_sec=12.0, ref_time=4.5)
    assert plan["source"]["ref_time_sec"] == 4.5
    plan, _ = norm(ans, ref_frame=None)
    assert plan["source"]["ref_frame"] == "inputs/media.jpg" and plan["source"]["ref_time_sec"] == 0.0
    plan, _ = norm(ans, fovy_exif=56.8)
    assert plan["camera"]["match"]["fovy"] == 56.8


def test_scene_prompt_carries_the_requirements():
    p = argon.scene_prompt("barrel roll over the table", 9, "image", P.STUNT_TYPES,
                           {"width": 1600, "height": 1200, "fovy_exif": 56.8})
    for s in ('"barrel roll over the table"', "9 s", "+X is the stunt's travel direction", "1.75 m", "0.75 m",
              "0.45 m", "3 cm", "HALF-extents", "Never omit", "hazard", "crash mat", "far side of the stunt target",
              "camera.match", "VERTICAL", "56.8 degrees", "1600x1200"):
        assert s in p, s
    for t in P.STUNT_TYPES:
        assert t in p
    v = argon.scene_prompt("vault the bench", 6, "video", P.STUNT_TYPES, {"video_sec": 14.0, "fps": 2.0})
    assert "clearest frame of the stunt area" in v and "14.0 s long" in v and "2 fps" in v


# ----------------------------------------------------------------------------- propose
def history_with_two_waves():
    def cand(cid, passed, score, extra=None):
        p = P.clamp(dict(P.defaults(), id=cid, label=f"take {cid}", why="x"))
        p.update(extra or {})
        g1 = {"head_g_peak": 6.2, "head_g_peak_t": 4.1, "hic15": 110.0, "hazard_collisions": 0 if passed else 1,
              "hazard_contacts": [] if passed else [{"t": 3.6, "body": "left_ankle_link", "object": "plant_3_leaves",
                                                     "depth_cm": 1.4}],
              "min_clearance_cm": 4.0 if passed else -1.4,
              "min_clearance_at": {"t": 3.6, "body": "left_ankle_link", "object": "plant_3_leaves"},
              "landing": {"t": 4.0, "surface": "crash_mat", "point": [1.5, 0.1], "target": [1.6, 0.0],
                          "error_m": 0.14, "inside_mat": True},
              "framing_pct": 95.0, "rotation_deg": {"roll": 350, "pitch": 10, "yaw": 2}, "passed": passed,
              "failures": [] if passed else ["1 hazard collision (left_ankle_link x plant_3_leaves 1.4 cm deep at 3.60 s)",
                                             "min clearance -1.4 cm < 2 cm"]}
        m = {"candidate_id": cid, "ok": True, "error": None, "gate1": g1,
             "sim": {"t_takeoff": 3.1, "t_landing": 4.0, "flight_time": 0.9, "apex_m": 1.9, "takeoff_velocity": [3.5, 3.4]},
             "score": {"safety": 90.0, "clearance": 60.0, "landing": 86.0, "framing": 95.0, "stunt_score": score}}
        return {"params": p, "metrics": m}
    crit = {"summary": "Clean roll but the arms are rigid.", "gate2_pass": False, "style_score": 64,
            "gate2_reasons": ["style 64 < 70"], "unnatural_motion": [{"t_sec": 4.2, "issue": "rigid arms", "severity": "medium"}],
            "proposed_changes": [{"param": "arm_reach_deg", "change": "decrease", "value": 120, "reason": "rigid arms"}],
            "framing": {"in_frame_pct": 95, "subject_size": "medium"}, "rotation_observed_deg": 350}
    return {"brief": "barrel roll over the table", "wave": 3, "director_notes": ["keep it low over the plants"],
            "current_steer": "softer landing",
            "waves": [{"wave": 1, "best_id": "w1c2", "candidates": [cand("w1c1", False, 40.0), cand("w1c2", False, 55.0)],
                       "critique": None, "steer": None},
                      {"wave": 2, "best_id": "w2c3", "candidates": [cand("w2c1", False, 70.0), cand("w2c3", True, 82.4)],
                       "critique": crit, "steer": "keep it low over the plants"}]}


def test_best_so_far_prefers_gate1_pass():
    best, wave = argon.best_so_far(history_with_two_waves())
    assert best["params"]["id"] == "w2c3" and wave == 2
    assert argon.best_so_far({"waves": []}) == (None, None)


def test_propose_schema_covers_every_param():
    s = argon.propose_schema(P, 4)
    assert s["properties"]["candidates"]["minItems"] == 4 == s["properties"]["candidates"]["maxItems"]
    props = s["properties"]["candidates"]["items"]["properties"]["params"]["properties"]
    for name, spec in P.PARAM_SPECS.items():
        assert props[name]["minimum"] == spec["min"] and props[name]["maximum"] == spec["max"]
    assert props["camera"]["enum"] == list(getattr(P, "CAMERAS", argon.CAMERAS))
    fx = load_fixture("propose")
    for c in fx["candidates"]:
        assert set(c["params"]) <= set(props)
        for k, v in c["params"].items():
            sch = props[k]
            assert (v in sch["enum"]) if "enum" in sch else sch["minimum"] <= v <= sch["maximum"], (k, v)


def test_finish_proposal_clamps_ids_colours_and_fills():
    plan = norm(load_fixture("scene"))[0]
    ans = {"rationale": " try ", "candidates": [
        {"label": "too fast", "why": "speed", "params": {**P.defaults(), "approach_speed": 99.0, "camera": "drone"}},
        {"label": "flat", "why": "flat params", "takeoff_vz": 3.9, "camera": "side"},
        {"label": "dup", "why": "same", "params": {**P.defaults(), "approach_speed": 99.0, "camera": "drone"}}]}
    prop = argon.finish_proposal(ans, P, 2, 4, history_with_two_waves(), plan=plan)
    cands = prop["candidates"]
    assert [c["id"] for c in cands] == ["w2c1", "w2c2", "w2c3", "w2c4"]
    assert [c["color"] for c in cands] == ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]
    assert cands[0]["approach_speed"] == P.PARAM_SPECS["approach_speed"]["max"] and cands[0]["camera"] == "hero"
    assert cands[1]["takeoff_vz"] == 3.9 and cands[1]["camera"] == "side"
    assert cands[1]["takeoff_x"] == plan["stunt"]["takeoff_x"]               # defaults come from the plan
    for c in cands:
        assert c["why"] and c["label"]
        for name, spec in P.PARAM_SPECS.items():
            assert spec["min"] <= c[name] <= spec["max"]
        assert {k: c[k] for k in P.PARAM_SPECS} == {k: P.clamp(c, plan=plan)[k] for k in P.PARAM_SPECS}
    w = "\n".join(prop["warnings"])
    assert "w2c1 approach_speed 99.0" in w and "returned 3 of 4" in w and "w2c3 repeats another candidate" in w
    assert cands[3]["why"].startswith("filler") and prop["wave"] == 2 and prop["rationale"] == "try"


def test_propose_prompt_wave1_and_later_waves():
    plan = norm(load_fixture("scene"))[0]
    first = argon.propose_prompt(plan, {"brief": plan["brief"], "waves": []}, P, 1, 4, None)
    assert "No take has run yet" in first and "Spread the 4 candidates" in first
    assert P.describe_for_prompt() in first and "Gate 1" in first and '"barrel roll over the table"' in first
    later = argon.propose_prompt(plan, history_with_two_waves(), P, 3, 4, "move the mat back 0.3 m")
    for s in ("best take so far is w2c3", "exploratory", "-1.4 cm", "plant_3_leaves", "rigid arms",
              '"keep it low over the plants"', '"softer landing"', '"move the mat back 0.3 m"', "WAVE 3",
              "w1c2", "Gate 1 FAILED"):
        assert s in later, s


def test_wave_number_prefers_the_out_dir(tmp_path):
    assert argon.wave_number(str(tmp_path / "waves" / "wave03" / "proposal.json"), {"waves": [{}, {}]}) == 3
    assert argon.wave_number(str(tmp_path / "p.json"), {"wave": 5, "waves": []}) == 5
    assert argon.wave_number(str(tmp_path / "p.json"), {"waves": [{}]}) == 2


def test_mock_propose_later_wave_nudges_the_best():
    plan = norm(load_fixture("scene"))[0]
    ans = argon.mock_propose(plan, history_with_two_waves(), P, 4, None)
    prop = argon.finish_proposal(ans, P, 3, 4, history_with_two_waves(), plan=plan)
    assert len(prop["candidates"]) == 4 and "w2c3" in prop["rationale"]
    best = argon.best_so_far(history_with_two_waves())[0]["params"]
    assert prop["candidates"][0]["takeoff_vz"] == pytest.approx(best["takeoff_vz"] + 0.15)


# ----------------------------------------------------------------------------- critique
def passing_metrics(inside=True):
    return {"candidate_id": "w1c1", "ok": True, "error": None,
            "sim": {"t_takeoff": 3.1, "t_landing": 4.0, "flight_time": 0.9, "apex_m": 1.92, "takeoff_velocity": [3.5, 3.4]},
            "gate1": {"head_g_peak": 6.1, "head_g_peak_t": 4.05, "hic15": 120.0, "hazard_collisions": 0,
                      "hazard_contacts": [], "min_clearance_cm": 4.2,
                      "min_clearance_at": {"t": 3.5, "body": "left_ankle_link", "object": "plant_3_leaves"},
                      "landing": {"t": 4.0, "surface": "crash_mat", "point": [1.5, 0.1], "target": [1.6, 0.0],
                                  "error_m": 0.14, "inside_mat": inside},
                      "framing_pct": 97.0, "rotation_deg": {"roll": 352, "pitch": 14, "yaw": 3},
                      "passed": True, "failures": []},
            "score": {"safety": 100.0, "clearance": 76.0, "landing": 86.0, "framing": 97.0, "stunt_score": 89.6}}


def test_critique_schema_names_only_param_specs():
    s = argon.critique_schema(P, 9.0)
    item = s["properties"]["proposed_changes"]["items"]
    assert item["properties"]["param"]["enum"] == list(P.PARAM_SPECS)
    assert s["properties"]["events"]["items"]["properties"]["t_sec"]["maximum"] == 9.5
    assert argon.schema_errors(load_fixture("critique"), s) == []


def test_finish_critique_applies_the_fixed_rubric():
    plan = norm(load_fixture("scene"))[0]
    params = P.clamp(dict(P.defaults(), id="w1c1", label="nominal"), plan=plan)
    th = argon.thresholds()
    crit = argon.finish_critique(load_fixture("critique"), P, plan, params, passing_metrics(), th, video="v.mp4",
                                 fps=5.0, video_sec=9.0)
    assert CONTRACT_CRITIQUE_KEYS <= set(crit)
    assert crit["gate2_pass"] is False and crit["rubric"]["no_high_severity_motion"] is False
    assert crit["gate2_reasons"] == ["high-severity unnatural motion: Both feet slide about 30 cm on the mat after "
                                     "touchdown while the knees are bent."]
    assert [e["t_sec"] for e in crit["events"]] == sorted(e["t_sec"] for e in crit["events"])
    assert crit["candidate_id"] == "w1c1" and crit["rotation_axis"] == "roll" and crit["rotation_required_deg"] == 360
    ans = load_fixture("critique")
    ans["unnatural_motion"] = [u for u in ans["unnatural_motion"] if u["severity"] != "high"]
    ok = argon.finish_critique(ans, P, plan, params, passing_metrics(), th)
    assert ok["gate2_pass"] is True and ok["gate2_reasons"] == [] and all(ok["rubric"].values())
    bad = copy.deepcopy(ans)
    bad["rotation_observed_deg"] = 250
    bad["framing"]["subject_size"] = "small"
    bad["style_score"] = 69
    bad["brief_match"] = {"matches": False, "missing": ["the roll"]}
    bad["proposed_changes"] = [{"param": "gravity", "change": "set", "value": 1, "reason": "x"},
                               {"param": "takeoff_vz", "change": "increase", "value": 99, "reason": "higher"}]
    r = argon.finish_critique(bad, P, plan, params, passing_metrics(inside=False), th)
    assert r["gate2_pass"] is False
    assert {k for k, v in r["rubric"].items() if not v} == {"brief_satisfied", "rotation_ok", "lands_on_mat",
                                                            "framing_ok", "style_ok"}
    assert r["proposed_changes"] == [{"param": "takeoff_vz", "change": "increase",
                                      "value": P.PARAM_SPECS["takeoff_vz"]["max"], "reason": "higher"}]
    assert any("unknown parameter 'gravity'" in w for w in r["warnings"])
    assert any("MuJoCo measured inside_mat=False" in w for w in r["warnings"])


def test_required_rotation():
    assert argon.required_rotation({"stunt": {"type": "barrel_roll"}}, {}) == ("roll", 360.0)
    assert argon.required_rotation({"stunt": {"type": "back_flip"}}, {}) == ("pitch", 360.0)
    assert argon.required_rotation({"stunt": {"type": "dive_roll"}}, {"rot_pitch_deg": 150}) == ("pitch", 150.0)
    assert argon.required_rotation({"stunt": {"type": "vault"}}, {"rot_roll_deg": 0}) == ("none", 0.0)


def test_critique_prompt_gives_ground_truth_and_rubric():
    plan = norm(load_fixture("scene"))[0]
    params = P.clamp(dict(P.defaults(), id="w1c1", label="nominal"), plan=plan)
    th = argon.thresholds()
    events = {"t_start": 0.0, "t_runup": 0.8, "t_takeoff": 3.1, "t_apex": 3.55, "t_landing": 4.0, "t_end": 7.0}
    pb = argon.playback_info(events, 9.0, 30.0)
    assert "slow motion" in pb and "take-off" in pb and "PLAYBACK" in pb
    p = argon.critique_prompt(plan, params, passing_metrics(), P, th, ("roll", 360.0), "9.0 s long", pb, ['"lower"'])
    for s in ("min clearance 4.2 cm", "head G 6.1", "do not re-estimate", "every 0.5 s", "within 45 degrees",
              f"in_frame_pct >= {th['framing_min_pct']:g}", f"style_score >= {th['style_min']:g}", "360 degrees while airborne: roll about the travel direction",
              "Name only these parameters: approach_speed",
              P.describe_for_prompt(), "DIRECTOR NOTES", "intentional edit", "w1c1"):
        assert s in p, s
    assert argon.playback_info(None, 9.0, 30.0).startswith("Playback speed unknown")


def test_mock_critique_follows_gate1():
    plan = norm(load_fixture("scene"))[0]
    params = P.clamp(dict(P.defaults(), id="w1c1"), plan=plan)
    th = argon.thresholds()
    good = argon.finish_critique(argon.mock_critique(plan, params, passing_metrics(), None), P, plan, params,
                                 passing_metrics(), th)
    assert good["gate2_pass"] is True and good["rotation_observed_deg"] == 352
    failing = passing_metrics()
    failing["gate1"].update(passed=False, failures=["min clearance 1.0 cm < 2 cm"])
    bad = argon.finish_critique(argon.mock_critique(plan, params, failing, None), P, plan, params, failing, th)
    assert bad["gate2_pass"] is False


# ----------------------------------------------------------------------------- schemas, logs
def test_schemas_use_only_the_supported_subset():
    allowed = {"type", "properties", "required", "additionalProperties", "items", "enum", "description", "minimum",
               "maximum", "minItems", "maxItems"}

    def walk(s):
        assert set(s) <= allowed, set(s) - allowed
        for v in s.get("properties", {}).values():
            walk(v)
        if "items" in s:
            walk(s["items"])
    for s in (argon.scene_schema(P.STUNT_TYPES), argon.propose_schema(P, 4), argon.critique_schema(P, 9.0)):
        walk(s)
        json.dumps(s)
        sent = json.dumps(argon.lean(s))          # what goes on the wire: the subset proven against gemini-4-argon
        for k in ("minimum", "maximum", "minItems", "maxItems"):
            assert f'"{k}"' not in sent


def test_log_and_run_dir_resolution(tmp_path):
    run = tmp_path / "stunts" / "roll-20261025_120000"
    (run / "waves" / "wave02").mkdir(parents=True)
    assert argon.find_run_dir(str(run / "plan.json")) == str(run)               # scene: plan.json not written yet
    assert argon.find_run_dir(str(run / "waves" / "wave02" / "proposal.json")) is None
    (run / "plan.json").write_text("{}")
    assert argon.find_run_dir(str(run / "waves" / "wave02" / "proposal.json")) == str(run)

    class A:
        log, out = None, str(run / "waves" / "wave02" / "critique.json")
    assert argon.resolve_log(A) == str(run / "logs" / "argon_calls.jsonl")
    A.log = "custom.jsonl"
    assert argon.resolve_log(A) == os.path.abspath("custom.jsonl")
    assert argon.files_cache_path(str(run / "logs" / "argon_calls.jsonl")) == str(run / "logs" / "argon_files.json")


def test_raw_dump_elides_payloads(fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FALLGUY_ARGON_RAW_DIR", str(tmp_path / "raw"))
    argon.generate([{"inlineData": {"mimeType": "image/jpeg", "data": "A" * 5000}}, {"text": "hi"}], SCHEMA, "scene", None)
    files = sorted(os.listdir(tmp_path / "raw"))
    assert len(files) == 2 and files[0].endswith(".request.json") and files[1].endswith(".response.json")
    req = json.loads((tmp_path / "raw" / files[0]).read_text())
    assert req["contents"][0]["parts"][0]["inlineData"]["data"] == "<5000 chars>"
    resp = json.loads((tmp_path / "raw" / files[1]).read_text())
    assert resp["candidates"][0]["content"]["parts"][1]["thoughtSignature"] == "<320 chars>"


# ----------------------------------------------------------------------------- CLI (mock mode, subprocess)
def run_cli(args, cwd, extra_env=None):
    env = dict(os.environ)
    env["FALLGUY_ARGON_MOCK"] = "1"
    env.update(extra_env or {})
    return subprocess.run([PY, os.path.join(SCRIPTS, "argon.py"), *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)


@pytest.fixture
def run_dir(tmp_path):
    from PIL import Image
    run = tmp_path / "stunts" / "barrel-roll-20261025_130000"
    (run / "inputs").mkdir(parents=True)
    Image.new("RGB", (1600, 1200), (128, 120, 110)).save(run / "inputs" / "media.jpg")
    shutil.copyfile(run / "inputs" / "media.jpg", run / "inputs" / "ref_frame.jpg")
    return run


def test_cli_mock_round_trip(run_dir, tiny_mp4):
    r = run_cli(["scene", "--media", "inputs/media.jpg", "--brief", "barrel roll over the table", "--duration", "9",
                 "--out", "plan.json", "--ref-frame", "inputs/ref_frame.jpg"], run_dir)
    assert r.returncode == 0, r.stderr
    line = json.loads(r.stdout.strip().splitlines()[-1])
    assert line["obstacles"] == 45 and line["mock"] is True and len(r.stdout.strip().splitlines()) == 1
    plan = json.loads((run_dir / "plan.json").read_text())
    validate_plan(plan)
    assert plan["source"]["media"] == "inputs/media.jpg" and plan["source"]["ref_frame"] == "inputs/ref_frame.jpg"

    w = run_dir / "waves" / "wave01"
    w.mkdir(parents=True)
    (w / "history.json").write_text(json.dumps({"brief": plan["brief"], "wave": 1, "waves": []}))
    r = run_cli(["propose", "--plan", "plan.json", "--history", "waves/wave01/history.json",
                 "--out", "waves/wave01/proposal.json", "--n", "4"], run_dir)
    assert r.returncode == 0, r.stderr
    prop = json.loads((w / "proposal.json").read_text())
    assert set(prop) >= {"rationale", "candidates"}
    assert [c["id"] for c in prop["candidates"]] == ["w1c1", "w1c2", "w1c3", "w1c4"]
    assert [c["color"] for c in prop["candidates"]] == argon.COLORS
    for c in prop["candidates"]:
        assert c == {**P.clamp(c, plan=plan), "why": c["why"], "label": c["label"], "id": c["id"], "color": c["color"]}

    c1 = w / "c1"
    c1.mkdir()
    (c1 / "params.json").write_text(json.dumps(prop["candidates"][0]))
    (c1 / "metrics.json").write_text(json.dumps(passing_metrics()))
    (c1 / "events.json").write_text(json.dumps({"t_start": 0, "t_runup": 0.8, "t_takeoff": 3.1, "t_apex": 3.5,
                                                "t_landing": 4.0, "t_end": 7.0}))
    shutil.copyfile(tiny_mp4, c1 / "video.mp4")
    r = run_cli(["critique", "--video", "waves/wave01/c1/video.mp4", "--plan", "plan.json", "--params",
                 "waves/wave01/c1/params.json", "--metrics", "waves/wave01/c1/metrics.json",
                 "--out", "waves/wave01/critique.json", "--fps", "5"], run_dir)
    assert r.returncode == 0, r.stderr
    crit = json.loads((w / "critique.json").read_text())
    assert CONTRACT_CRITIQUE_KEYS <= set(crit) and crit["gate2_pass"] is True
    for c in crit["proposed_changes"]:
        assert c["param"] in P.PARAM_SPECS

    logs = read_log(run_dir / "logs" / "argon_calls.jsonl")
    assert [x["label"] for x in logs] == ["scene", "propose", "critique"] and all(x["mock"] for x in logs)
    for path in run_dir.rglob("*.json"):
        assert FAKE_KEY not in path.read_text()


def test_cli_errors_are_one_redacted_line(run_dir):
    r = run_cli(["scene", "--media", "inputs/missing.jpg", "--brief", "x", "--duration", "9", "--out", "plan.json"],
                run_dir)
    assert r.returncode == 1 and r.stderr.strip().splitlines()[-1].startswith("argon.py scene: error: media not found")
    r = run_cli(["scene", "--media", "inputs/media.jpg", "--brief", "x", "--duration", "-1", "--out", "plan.json"], run_dir)
    assert r.returncode == 2
    r = run_cli(["fly"], run_dir)
    assert r.returncode == 2
    r = run_cli(["propose", "--plan", "nope.json", "--history", "h.json", "--out", "p.json"], run_dir)
    assert r.returncode == 1 and "plan not found: nope.json" in r.stderr
    r = run_cli(["critique", "--video", "inputs/media.jpg", "--plan", "plan.json", "--params", "p.json",
                 "--metrics", "m.json", "--out", "c.json", "--fps", "30"], run_dir)
    assert r.returncode == 1 and "--fps 30.0" in r.stderr


def test_cli_live_path_against_fake_server(run_dir, fake):
    """The real (non-mock) scene path end to end: request built from the photo, answer normalised and logged."""
    fake.answer = load_fixture("scene")
    env = {"FALLGUY_ARGON_MOCK": "0", "FALLGUY_ARGON_BASE_URL": fake.url}
    r = run_cli(["scene", "--media", "inputs/media.jpg", "--brief", "barrel roll over the table", "--duration", "9",
                 "--out", "plan.json", "--ref-frame", "inputs/ref_frame.jpg"], run_dir, env)
    assert r.returncode == 0, r.stderr
    validate_plan(json.loads((run_dir / "plan.json").read_text()))
    (body,) = fake.bodies("generate")
    parts = body["contents"][0]["parts"]
    assert len(parts) == 2 and parts[0]["inlineData"]["mimeType"] == "image/jpeg"     # identical ref frame sent once
    assert "barrel roll over the table" in parts[1]["text"]
    sch = body["generationConfig"]["responseJsonSchema"]
    assert sch["properties"]["stunt"]["properties"]["type"]["enum"] == P.STUNT_TYPES
    (rec,) = read_log(run_dir / "logs" / "argon_calls.jsonl")
    assert rec["status"] == 200 and rec["label"] == "scene"
    assert FAKE_KEY not in r.stderr + r.stdout


LIVE = os.path.join(FIXTURES, "live")


def live_fixture(name):
    path = os.path.join(LIVE, f"{name}.json")
    if not os.path.isfile(path):
        pytest.skip(f"no live fixture {path}")
    with open(path) as f:
        return json.load(f)


def test_live_scene_answer_normalises_to_a_valid_plan():
    """The real gemini-4-argon answer for the venue photo (barrel roll over the table, 9 s)."""
    plan, warns = norm(live_fixture("scene"))
    validate_plan(plan)
    assert plan["stunt"]["type"] == "barrel_roll"
    assert sum(o["role"] == "stunt_target" for o in plan["obstacles"]) >= 1 and len(plan["obstacles"]) >= 5
    for o in live_fixture("scene.output")["obstacles"]:
        assert o in plan["obstacles"]                       # the saved output is what argon.py wrote


def test_live_critique_answer_has_the_contract_shape():
    ans = live_fixture("critique")
    out = live_fixture("critique.output")
    assert CONTRACT_CRITIQUE_KEYS <= set(out) and len(out["events"]) >= 6
    assert all(c["param"] in P.PARAM_SPECS for c in out["proposed_changes"])
    assert out["gate2_pass"] is all(out["rubric"].values())
    assert set(ans) >= {"events", "summary", "brief_match", "rotation_observed_deg", "framing", "style_score"}


def test_live_fixtures_replay_through_the_cli(run_dir):
    live_fixture("scene")
    r = run_cli(["scene", "--media", "inputs/media.jpg", "--brief", "barrel roll over the table", "--duration", "9",
                 "--out", "plan.json", "--ref-frame", "inputs/ref_frame.jpg"], run_dir, {"FALLGUY_ARGON_MOCK_DIR": LIVE})
    assert r.returncode == 0, r.stderr
    validate_plan(json.loads((run_dir / "plan.json").read_text()))


@pytest.mark.skipif(os.environ.get("FALLGUY_ARGON_LIVE") != "1", reason="set FALLGUY_ARGON_LIVE=1 to call the real API")
def test_live_ping(tmp_path, monkeypatch):
    monkeypatch.delenv("FALLGUY_ARGON_TOKEN_FILE", raising=False)
    monkeypatch.delenv("FALLGUY_ARGON_BASE_URL", raising=False)
    argon._KEY = None
    r = subprocess.run([PY, os.path.join(SCRIPTS, "argon.py"), "ping", "--log", str(tmp_path / "calls.jsonl")],
                       capture_output=True, text=True, timeout=300,
                       env={k: v for k, v in os.environ.items() if not k.startswith("FALLGUY_ARGON_MOCK")})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["ok"] is True
