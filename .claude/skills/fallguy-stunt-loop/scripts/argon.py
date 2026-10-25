#!/usr/bin/env python3
"""FALLGUY Argon client: gemini-4-argon plans the scene, proposes candidate takes and critiques renders.

    python argon.py ping
    python argon.py scene    --media inputs/media.jpg --brief "..." --duration 9 --out plan.json
                             [--ref-frame inputs/ref_frame.jpg] [--ref-time S]
    python argon.py propose  --plan plan.json --history history.json --out waves/wave02/proposal.json [--steer "..."] [--n 4]
    python argon.py critique --video waves/wave02/c3/video.mp4 --plan plan.json --params .../params.json
                             --metrics .../metrics.json --out waves/wave02/critique.json [--fps 5] [--events E] [--steer "..."]

Every command also takes --log PATH (default <run>/logs/argon_calls.jsonl, where <run> is the nearest directory at
or above --out that holds plan.json). File shapes: reference/contracts.md. Success prints one JSON line and exits 0;
failure prints one redacted error line on stderr and exits 1 (2 for bad arguments). Progress goes to stderr.

Key: ~/.argon_token (FALLGUY_ARGON_TOKEN_FILE overrides). It is read here only, sent only in the x-goog-api-key
header, and redacted from every message, log line and file.
Env: ARGON_MODEL (gemini-4-argon), ARGON_THINKING (high), FALLGUY_ARGON_MOCK=1 (canned answers from
fixtures/argon_mock, no network; FALLGUY_ARGON_MOCK_DIR picks another fixture dir), FALLGUY_ARGON_BASE_URL,
FALLGUY_ARGON_TIMEOUT (600 s), FALLGUY_ARGON_BACKOFF (5 s retry base), FALLGUY_ARGON_INLINE_MAX_MB (15),
FALLGUY_ARGON_RAW_DIR (save every request, base64 elided, and response), FALLGUY_ARGON_QUIET=1, FALLGUY_ARGON_DEBUG=1.
"""
import argparse
import base64
import copy
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)          # engine.params, engine.render and gates live next to this file
SKILL_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(SKILL_DIR)))
FIXTURES = os.path.join(HERE, "fixtures", "argon_mock")

DEFAULT_MODEL = "gemini-4-argon"
DEFAULT_BASE = "https://generativelanguage.googleapis.com"
TRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}
IMAGE_MAX_SIDE = 1600
FILE_REUSE_S = 47 * 3600            # Files API uploads expire after 48 h
CORRIDOR_HALF_WIDTH = 1.2           # m either side of the lane: what the performer can reach
COLORS = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

KINDS = ["box", "cylinder", "sphere"]
MATERIALS = ["wood", "metal", "fabric", "foam", "glass", "plastic", "concrete", "stone"]
FLOOR_MATERIALS = ["concrete", "wood", "carpet", "tile", "stone", "rubber", "grass", "asphalt", "other"]
ROLES = ["stunt_target", "furniture", "prop", "structure"]
EVENT_KINDS = ["run_start", "takeoff", "apex", "landing", "settled", "turn_start", "turn_end", "finish", "other"]
GROUND_TYPES = ("run_turn",)          # ground stunts: kinematic run, turn, swerve or stop; no flight, no mat
TURN_SIDES = ["left", "right", "none"]
SEVERITIES = ["low", "medium", "high"]
SUBJECT_SIZES = ["small", "medium", "large"]
CHANGES = ["increase", "decrease", "set"]
CAMERAS = ["match_follow", "match", "hero", "side"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"}
VIDEO_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mov", ".webm": "video/webm",
              ".mpeg": "video/mpeg", ".mpg": "video/mpg", ".avi": "video/avi", ".wmv": "video/wmv",
              ".3gp": "video/3gpp", ".flv": "video/x-flv"}
TYPE_ROTATION = {"barrel_roll": ("roll", 360.0), "front_flip": ("pitch", 360.0), "back_flip": ("pitch", 360.0)}
CAMERA_TEXT = {"hero": "a 3/4 tracking dolly on the performer", "side": "a side-on tracking dolly",
               "match": "the director's camera position and angle, static",
               "match_follow": "the director's camera position, panning to follow the performer"}
AXIS_TEXT = {"roll": "roll about the travel direction (+X), as in a barrel roll",
             "pitch": "pitch head over heels (about +Y), as in a flip",
             "yaw": "yaw about the vertical (a spin)", "none": "no rotation is required"}
# Contract defaults; scripts/gates.py is the source of truth when it imports.
THRESHOLD_DEFAULTS = {"head_g_max": 8.0, "hazard_collisions_max": 0, "min_clearance_cm": 2.0,
                      "landing_error_max_m": 0.65, "require_inside_mat": True, "framing_min_pct": 85.0,
                      "gate2_min_score": 85.0, "style_min": 70}
WEIGHT_DEFAULTS = {"safety": 0.35, "clearance": 0.25, "landing": 0.20, "framing": 0.20}


class ArgonError(Exception):
    """A failed Argon call or an unusable answer. The message is redacted; `retry` marks answers worth asking again."""

    def __init__(self, msg, status=None, retry=False):
        super().__init__(redact(msg))
        self.status, self.retry = status, retry


# ----------------------------------------------------------------------------- config, key, logging
def _env_float(name, default):
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        raise ArgonError(f"{name}={v!r} is not a number")


def model_name():
    return os.environ.get("ARGON_MODEL", "").strip() or DEFAULT_MODEL


def thinking_level():
    return os.environ.get("ARGON_THINKING", "").strip().lower() or "high"


def base_url():
    return (os.environ.get("FALLGUY_ARGON_BASE_URL", "").strip() or DEFAULT_BASE).rstrip("/")


def is_mock():
    return os.environ.get("FALLGUY_ARGON_MOCK", "").strip().lower() not in ("", "0", "false", "no")


_KEY = None


def load_key():
    """The API key from FALLGUY_ARGON_TOKEN_FILE or ~/.argon_token (then ./.argon_token, <repo>/.argon_token)."""
    global _KEY
    if _KEY:
        return _KEY
    env = os.environ.get("FALLGUY_ARGON_TOKEN_FILE", "").strip()
    paths = [env] if env else ["~/.argon_token", os.path.join(os.getcwd(), ".argon_token"),
                               os.path.join(REPO, ".argon_token")]
    for p in paths:
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            with open(p) as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if not lines:
                raise ArgonError(f"the Argon key file {p} is empty")
            m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*['\"]?([^'\"\s]+)['\"]?$", lines[0])   # NAME=value files
            _KEY = m.group(1) if m else lines[0]
            return _KEY
    if env:
        raise ArgonError(f"FALLGUY_ARGON_TOKEN_FILE points to {env}, which does not exist")
    raise ArgonError("no Argon key: put it in ~/.argon_token (or point FALLGUY_ARGON_TOKEN_FILE at a key file)")


def redact(text):
    s = text if isinstance(text, str) else str(text)
    if _KEY:
        s = s.replace(_KEY, "<KEY>")
    return re.sub(r"(?i)([?&](?:key|api_key)=)[^&\s\"']+", r"\1<KEY>", s)


def say(msg):
    """Progress line on stderr (stdout carries only the final JSON line)."""
    if not os.environ.get("FALLGUY_ARGON_QUIET"):
        print(f"argon: {redact(msg)}", file=sys.stderr, flush=True)


def log_call(log, rec):
    """Append one call record (never the key) to the run's argon_calls.jsonl."""
    if not log:
        return
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), **rec}
    with open(log, "a") as f:
        f.write(redact(json.dumps(rec)) + "\n")


def find_run_dir(out):
    """The run directory for an output path: the nearest directory at or above it that holds plan.json."""
    d = os.path.dirname(os.path.abspath(out))
    if os.path.basename(out) == "plan.json":
        return d
    for _ in range(3):              # out, waves/waveNN/out, waves/out
        if os.path.isfile(os.path.join(d, "plan.json")):
            return d
        d = os.path.dirname(d)
    return None


def resolve_log(args):
    if getattr(args, "log", None):
        return os.path.abspath(args.log)
    out = getattr(args, "out", None)
    if not out:
        return None
    return os.path.join(find_run_dir(out) or os.path.dirname(os.path.abspath(out)), "logs", "argon_calls.jsonl")


def files_cache_path(log):
    if log:
        return os.path.join(os.path.dirname(log), "argon_files.json")
    return os.path.join(os.path.expanduser("~"), ".cache", "fallguy", "argon_files.json")


def read_json(path, what):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ArgonError(f"{what} not found: {path}")
    except json.JSONDecodeError as e:
        raise ArgonError(f"{what} {path} is not valid JSON: {e}")


def write_json(path, obj):
    """Atomic write; the text is redacted as a last line of defence."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(redact(json.dumps(obj, indent=2)) + "\n")
    os.replace(tmp, path)


def elide(obj):
    """Copy with base64 payloads and thought signatures shortened, safe and small enough to save."""
    o = copy.deepcopy(obj)

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("data", "thoughtSignature") and isinstance(v, str) and len(v) > 200:
                    x[k] = f"<{len(v)} chars>"
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(o)
    return o


def save_raw(label, body, resp):
    """With FALLGUY_ARGON_RAW_DIR set: the request (before sending) and the response, base64 elided."""
    d = os.environ.get("FALLGUY_ARGON_RAW_DIR", "").strip()
    if not d:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for kind, obj in (("request", body), ("response", resp)):
        if obj is not None:
            write_json(os.path.join(d, f"{label}_{stamp}.{kind}.json"), elide(obj))


# ----------------------------------------------------------------------------- HTTP
def _http(method, url, body=None, raw=None, headers=None, timeout=600.0):
    """One request. Returns (status, lower-cased headers, text, latency_s, bytes_sent); HTTP errors do not raise."""
    h = {"x-goog-api-key": load_key()}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, rh, txt = r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        status, rh, txt = e.code, e.headers, e.read()
    rh = {k.lower(): v for k, v in (rh.items() if rh else [])}
    return status, rh, txt.decode("utf-8", "replace"), time.time() - t0, len(data or b"")


def _error_text(parsed, txt):
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        e = parsed["error"]
        return f"{e.get('status', '')}: {e.get('message', '')}".strip(": ")
    return " ".join((txt or "").split())


def _backoff(attempt, rh, parsed):
    base = _env_float("FALLGUY_ARGON_BACKOFF", 5.0)
    if base <= 0:
        return 0.0
    delay = base * 3 ** (attempt - 1)
    hints = [rh.get("retry-after")]
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        hints += [d.get("retryDelay") for d in parsed["error"].get("details") or [] if isinstance(d, dict)]
    for hint in hints:
        try:
            delay = max(delay, float(str(hint).rstrip("s")))
        except (TypeError, ValueError):
            pass
    return min(delay, 90.0)


def call(method, url, label, log, body=None, raw=None, headers=None):
    """HTTP call with 3 tries on 429/500/502/503/504 and dropped connections. Returns (json, headers)."""
    timeout = _env_float("FALLGUY_ARGON_TIMEOUT", 600.0)
    for attempt in range(1, TRIES + 1):
        rec = {"label": label, "method": method, "url": redact(url.split("?")[0]), "attempt": attempt}
        if ":generateContent" in url:
            rec.update(model=model_name(), thinking=thinking_level())
        try:
            status, rh, txt, dt, nbytes = _http(method, url, body, raw, headers, timeout)
        except Exception as e:      # no HTTP status: timeout, refused or dropped connection
            timed_out = isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError)
            log_call(log, {**rec, "status": -1, "error": f"{type(e).__name__}: {e}"[:300]})
            if timed_out:
                raise ArgonError(f"{label}: no answer from Argon within {timeout:.0f} s (FALLGUY_ARGON_TIMEOUT)")
            if attempt < TRIES:
                d = _backoff(attempt, {}, None)
                say(f"{label}: {type(e).__name__}: {e}; retrying in {d:.0f} s ({attempt}/{TRIES})")
                time.sleep(d)
                continue
            raise ArgonError(f"{label}: cannot reach Argon after {TRIES} tries: {type(e).__name__}: {e}")
        try:
            parsed = json.loads(txt) if txt.strip() else {}
        except json.JSONDecodeError:
            parsed = None
        rec.update(status=status, latency_s=round(dt, 2), request_bytes=nbytes)
        if isinstance(parsed, dict):
            if "usageMetadata" in parsed:
                rec["usage"] = parsed["usageMetadata"]
            cand = (parsed.get("candidates") or [{}])[0]
            if isinstance(cand, dict) and cand.get("finishReason"):
                rec["finish_reason"] = cand["finishReason"]
            if parsed.get("modelVersion"):
                rec["model_version"] = parsed["modelVersion"]
        if status != 200:
            rec["error"] = _error_text(parsed, txt)[:500]
        log_call(log, rec)
        if status == 200:
            if not isinstance(parsed, dict):
                raise ArgonError(f"{label}: HTTP 200 but the body is not a JSON object: {txt[:200]!r}")
            return parsed, rh
        if status in RETRY_STATUS and attempt < TRIES:
            d = _backoff(attempt, rh, parsed)
            say(f"{label}: HTTP {status}, retrying in {d:.0f} s ({attempt}/{TRIES})")
            time.sleep(d)
            continue
        tail = f" after {TRIES} tries" if status in RETRY_STATUS else ""
        raise ArgonError(f"{label}: HTTP {status}{tail}: {_error_text(parsed, txt)[:400]}", status=status)


def answer_text(resp):
    """The answer: text of the non-thought parts of the first candidate."""
    cand = (resp.get("candidates") or [{}])[0] or {}
    parts = (cand.get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))


def parse_answer(resp, label):
    cands = resp.get("candidates") or []
    if not cands:
        block = (resp.get("promptFeedback") or {}).get("blockReason")
        raise ArgonError(f"{label}: Argon returned no answer (promptFeedback.blockReason={block})")
    finish = cands[0].get("finishReason")
    if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"):
        raise ArgonError(f"{label}: Argon declined to answer (finishReason={finish}); rephrase the brief")
    text = answer_text(resp).strip()
    if not text:
        raise ArgonError(f"{label}: empty answer (finishReason={finish})", retry=True)
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise ArgonError(f"{label}: the answer is not valid JSON ({e.msg} at char {e.pos}, finishReason={finish}): "
                         f"{text[:120]!r}", retry=True)
    if not isinstance(obj, dict):
        raise ArgonError(f"{label}: the answer is a JSON {type(obj).__name__}, expected an object", retry=True)
    return obj


def usage_line(resp):
    u = resp.get("usageMetadata") or {}
    return (f"{u.get('promptTokenCount', '?')} prompt, {u.get('thoughtsTokenCount', 0)} thinking, "
            f"{u.get('candidatesTokenCount', '?')} answer tokens")


COMPLEX_KEYS = ("minimum", "maximum", "minItems", "maxItems")


def lean(schema, drop=COMPLEX_KEYS):
    """Copy of a schema without the keywords that make Gemini's constrained decoding too complex. A bare
    "400 Request contains an invalid argument" is how the API rejects such a schema; values are range-checked
    after the answer arrives instead (normalize_plan, engine.params.clamp, finish_critique)."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "properties":
            out[k] = {pk: lean(pv, drop) for pk, pv in v.items()}
        elif k == "items":
            out[k] = lean(v, drop)
        else:
            out[k] = v
    return out


def generate(parts, schema, label, log):
    """generateContent with strict JSON (responseJsonSchema, lean) and explicit thinking. Returns (answer, response).
    A bare 400 INVALID_ARGUMENT is retried once with a minimal schema (no descriptions either)."""
    url = f"{base_url()}/v1beta/models/{model_name()}:generateContent"
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": lean(schema),
                                 "thinkingConfig": {"thinkingLevel": thinking_level()}}}
    n, simplified = 0, False
    while True:
        n += 1
        say(f"{label}: asking {model_name()} (thinking {thinking_level()}); this can take a few minutes")
        t0 = time.time()
        save_raw(label, body, None)
        try:
            resp, _ = call("POST", url, label, log, body=body)
        except ArgonError as e:
            if e.status == 400 and not simplified and "request contains an invalid argument" in str(e).lower():
                simplified = True
                body["generationConfig"]["responseJsonSchema"] = lean(schema, COMPLEX_KEYS + ("description",))
                say(f"{label}: HTTP 400 without details (schema too complex?); retrying once with a minimal schema")
                continue
            raise
        save_raw(label, None, resp)
        try:
            ans = parse_answer(resp, label)
        except ArgonError as e:
            if not e.retry or n >= 2:
                raise
            say(f"{e}; asking once more")
            continue
        say(f"{label}: answered in {time.time() - t0:.0f} s ({usage_line(resp)})")
        return ans, resp


def mock_answer(name, log):
    """Canned answer fixtures/argon_mock/<name>.json (or $FALLGUY_ARGON_MOCK_DIR); no network."""
    path = os.path.join(os.environ.get("FALLGUY_ARGON_MOCK_DIR", "").strip() or FIXTURES, f"{name}.json")
    ans = read_json(path, "mock fixture")
    log_call(log, {"label": name, "status": 200, "latency_s": 0.0, "mock": True, "fixture": path})
    say(f"{name}: mock answer from {path}")
    return ans


# ----------------------------------------------------------------------------- media
def media_kind(path):
    if not os.path.isfile(path):
        raise ArgonError(f"media not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_MIME:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return "image"
    except Exception:
        raise ArgonError(f"{path}: not an image or a supported video ({' '.join(sorted(VIDEO_MIME))})")


def _open_image(path):
    """Pillow image; .heic/.heif go through pillow-heif when installed, else macOS sips."""
    from PIL import Image
    try:
        im = Image.open(path)
        im.load()
        return im
    except Exception as e:
        if os.path.splitext(path)[1].lower() not in (".heic", ".heif"):
            raise ArgonError(f"cannot read image {path}: {e}")
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        im = Image.open(path)
        im.load()
        return im
    except Exception:
        pass
    sips = shutil.which("sips")
    if not sips:
        raise ArgonError(f"{path}: reading HEIC needs macOS sips or `pip install pillow-heif`")
    tmp = tempfile.mkdtemp(prefix="fallguy_heic_")
    try:
        out = os.path.join(tmp, "converted.jpg")
        r = subprocess.run([sips, "-s", "format", "jpeg", path, "--out", out], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(out):
            raise ArgonError(f"sips could not convert {path} to JPEG: {(r.stderr or r.stdout).strip()[:200]}")
        im = Image.open(out)
        im.load()
        return im
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def exif_fovy(im, size):
    """Vertical field of view (deg) from EXIF FocalLengthIn35mmFilm for an image of `size`, or None."""
    try:
        f35 = float(im.getexif().get_ifd(0x8769).get(0xA405) or 0)
    except Exception:
        return None
    if f35 <= 0:
        return None
    w, h = size
    f_px = f35 / math.hypot(36.0, 24.0) * math.hypot(w, h)
    return round(math.degrees(2 * math.atan(h / 2 / f_px)), 2)


def image_part(path, max_side=IMAGE_MAX_SIDE):
    """Inline JPEG part (EXIF-oriented, longest side <= max_side) and {width, height, fovy_exif}."""
    from PIL import ImageOps
    src = _open_image(path)
    im = ImageOps.exif_transpose(src).convert("RGB")
    fovy = exif_fovy(src, im.size)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    part = {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}}
    return part, {"width": im.width, "height": im.height, "fovy_exif": fovy}


def video_info(path):
    """(duration_s, fps) from the container header via imageio-ffmpeg; (None, None) when unknown."""
    try:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.read_frames(path)
        meta = next(gen)
        gen.close()
        return (float(meta.get("duration") or 0) or None), (float(meta.get("fps") or 0) or None)
    except Exception:
        return None, None


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_video(path, mime, log, cache_path, force=False):
    """Files API resumable upload, polled until ACTIVE. URIs are cached per sha1 and reused within 47 h."""
    sha = _sha1(path)
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        try:
            cache = read_json(cache_path, "upload cache")
        except ArgonError:
            cache = {}
    hit = cache.get(sha)
    if hit and not force and time.time() - float(hit.get("uploaded_at", 0)) < FILE_REUSE_S:
        say(f"reusing upload {hit.get('name')} for {os.path.basename(path)}")
        return hit
    size = os.path.getsize(path)
    say(f"uploading {os.path.basename(path)} ({size / 1e6:.1f} MB) through the Files API")
    _, rh = call("POST", f"{base_url()}/upload/v1beta/files", "upload_start", log,
                 body={"file": {"display_name": os.path.basename(path)[:100]}},
                 headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                          "X-Goog-Upload-Header-Content-Length": str(size),
                          "X-Goog-Upload-Header-Content-Type": mime})
    up = rh.get("x-goog-upload-url")
    if not up:
        raise ArgonError("upload_start: the reply has no x-goog-upload-url header")
    with open(path, "rb") as f:
        raw = f.read()
    resp, _ = call("POST", up, "upload_finalize", log, raw=raw,
                   headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    f = resp.get("file") or {}
    if not f.get("name") or not f.get("uri"):
        raise ArgonError(f"upload_finalize: unexpected reply {json.dumps(resp)[:200]}")
    t0, poll = time.time(), _env_float("FALLGUY_ARGON_POLL_S", 2.0)
    while f.get("state") == "PROCESSING":
        if time.time() - t0 > 300:
            raise ArgonError(f"uploaded video {f['name']} is still PROCESSING after 300 s")
        time.sleep(poll)
        f, _ = call("GET", f"{base_url()}/v1beta/{f['name']}", "file_poll", log)
    if f.get("state") not in (None, "ACTIVE"):
        raise ArgonError(f"uploaded video {f.get('name')} is {f.get('state')}: {f.get('error')}")
    entry = {"name": f["name"], "uri": f["uri"], "mime": mime, "size": size, "path": os.path.abspath(path),
             "uploaded_at": time.time(), "expiration_time": f.get("expirationTime")}
    if cache_path:
        cache[sha] = entry
        write_json(cache_path, cache)
    say(f"uploaded {f['name']} (ACTIVE after {time.time() - t0:.0f} s)")
    return entry


def video_part(path, fps, log, cache_path, force_upload=False):
    """Inline base64 when <= FALLGUY_ARGON_INLINE_MAX_MB (15 MB), else a Files API upload. Returns (part, how)."""
    mime = VIDEO_MIME.get(os.path.splitext(path)[1].lower())
    if not mime:
        raise ArgonError(f"{path}: unsupported video type (use {' '.join(sorted(VIDEO_MIME))})")
    size = os.path.getsize(path)
    meta = {"fps": fps}
    if size <= _env_float("FALLGUY_ARGON_INLINE_MAX_MB", 15.0) * 1024 * 1024 and not force_upload:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        say(f"sending {os.path.basename(path)} ({size / 1e6:.1f} MB) inline, sampled at {fps:g} fps")
        return {"inlineData": {"mimeType": mime, "data": data}, "videoMetadata": meta}, "inline"
    f = upload_video(path, mime, log, cache_path, force=force_upload)
    return {"fileData": {"mimeType": mime, "fileUri": f["uri"]}, "videoMetadata": meta}, "file"


def generate_with_video(path, fps, make_parts, schema, label, log, cache_path):
    """generate() with a video part; falls back to a fresh upload if the inline body or a cached URI is refused."""
    part, how = video_part(path, fps, log, cache_path)
    try:
        return generate(make_parts(part), schema, label, log)
    except ArgonError as e:
        msg = str(e).lower()
        if how == "inline" and e.status in (400, 413) and re.search(r"size|exceed|too large|payload|limit", msg):
            say("the inline video was refused as too large; uploading it through the Files API instead")
        elif how == "file" and e.status in (400, 403, 404) and "file" in msg:
            say("the cached upload is no longer usable; uploading the video again")
        else:
            raise
    part, how = video_part(path, fps, log, cache_path, force_upload=True)
    return generate(make_parts(part), schema, label, log)


# ----------------------------------------------------------------------------- cross-module helpers
def engine_params():
    """The real scripts/engine/params.py: PARAM_SPECS, STUNT_TYPES, defaults(), clamp(), describe_for_prompt()."""
    try:
        from engine import params
    except Exception as e:
        raise ArgonError(f"cannot import engine.params from {os.path.join(HERE, 'engine')}: {type(e).__name__}: {e}")
    for name in ("PARAM_SPECS", "STUNT_TYPES", "clamp", "describe_for_prompt"):
        if not hasattr(params, name):
            raise ArgonError(f"engine.params has no {name} (see reference/contracts.md)")
    return params


def _gates():
    try:
        import gates
        return gates
    except Exception as e:
        say(f"warning: cannot import gates.py ({type(e).__name__}: {e}); using the contract's default thresholds")
        return None


def thresholds():
    th = dict(THRESHOLD_DEFAULTS)
    g = _gates()
    if g is not None and isinstance(getattr(g, "THRESHOLDS", None), dict):
        th.update(g.THRESHOLDS)
    return th


def gates_text():
    g = _gates()
    if g is not None and callable(getattr(g, "describe", None)):
        try:
            return str(g.describe()).strip()
        except Exception as e:
            say(f"warning: gates.describe() failed ({type(e).__name__}: {e})")
    th = dict(THRESHOLD_DEFAULTS, **(getattr(g, "THRESHOLDS", None) or {}))
    w = dict(WEIGHT_DEFAULTS, **(getattr(g, "WEIGHTS", None) or {}))
    return (f"Gate 1 (MuJoCo, every take): head G peak (15 ms average) <= {th['head_g_max']}; hazard collisions <= "
            f"{th['hazard_collisions_max']}; min clearance >= {th['min_clearance_cm']} cm; landing error <= "
            f"{th['landing_error_max_m']} m and inside the mat; framing >= {th['framing_min_pct']} %.\n"
            f"Gate 2 (critique of the best render): Gate 1 passed; brief satisfied; required rotation within 45 deg; "
            f"lands on the mat; no high-severity unnatural motion; in frame >= {th['framing_min_pct']} % at a readable "
            f"size; style >= {th['style_min']}; final stunt score >= {th['gate2_min_score']}, where stunt score = "
            f"{w['safety']} safety + {w['clearance']} clearance + {w['landing']} landing + {w['framing']} framing.")


def _choices(name, spec):
    for k in ("choices", "options", "enum", "values"):
        if isinstance(spec.get(k), (list, tuple)) and spec[k]:
            return [str(c) for c in spec[k]]
    if isinstance(spec.get("default"), str):
        return list(CAMERAS) if name == "camera" else [spec["default"]]
    return None


def _is_numeric(spec):
    return all(isinstance(spec.get(k), (int, float)) and not isinstance(spec.get(k), bool) for k in ("min", "max"))


def stunt_type_of(plan):
    return str(((plan or {}).get("stunt") or {}).get("type") or "custom")


def is_ground(plan_or_type):
    t = plan_or_type if isinstance(plan_or_type, str) else stunt_type_of(plan_or_type)
    return t in GROUND_TYPES


def relevant_names(P, stype):
    """The PARAM_SPECS names that apply to this stunt type: engine.params.relevant(stunt_type) when it exists,
    else every parameter."""
    rel = getattr(P, "relevant", None)
    if callable(rel):
        try:
            names = [n for n in rel(stype) if n in P.PARAM_SPECS]
            if names:
                return names
        except Exception as e:
            say(f"warning: engine.params.relevant({stype!r}) failed ({type(e).__name__}: {e}); showing every parameter")
    return list(P.PARAM_SPECS)


def camera_choices(P):
    spec = P.PARAM_SPECS.get("camera") if isinstance(P.PARAM_SPECS.get("camera"), dict) else {}
    cams = _choices("camera", spec) if spec else None
    return [str(c) for c in (cams or getattr(P, "CAMERAS", None) or CAMERAS)]


def default_camera(stype, P=None):
    """match_follow for ground stunts, hero for aerial ones (or the nearest camera engine.params knows)."""
    cams = camera_choices(P) if P is not None else list(CAMERAS)
    prefs = ("match_follow", "match", "hero") if stype in GROUND_TYPES else ("hero", "side", "match")
    return next((c for c in prefs if c in cams), cams[0] if cams else "hero")


def param_table(P, plan):
    """Prompt table of the parameters that apply to this plan's stunt type, with its defaults, and the camera."""
    stype = stunt_type_of(plan)
    names = relevant_names(P, stype)
    try:
        d = _defaults(P, plan)
    except Exception:
        d = {}
    lines = ["name | unit | range | start value | meaning"]
    for k in names:
        s = P.PARAM_SPECS[k]
        if _is_numeric(s):
            lines.append(f"{k} | {s.get('unit', '')} | {s['min']:g}..{s['max']:g} | "
                         f"{_num(d.get(k, s.get('default')))} | {s.get('doc', '')}")
        elif _choices(k, s) and k != "camera":
            lines.append(f"{k} | - | {', '.join(_choices(k, s))} | {d.get(k, s.get('default'))} | {s.get('doc', '')}")
    cams = camera_choices(P)
    lines.append(f"camera | - | {', '.join(cams)} | {default_camera(stype, P)} | the camera the take is rendered and "
                 "judged from: " + "; ".join(f"{c}: {CAMERA_TEXT.get(c, c)}" for c in cams))
    if not is_ground(stype):
        lines.append("Rotations: +roll rolls to the performer's right about the travel axis X, +pitch is a front flip, "
                     "-pitch a back flip, +yaw turns left.")
    lines.append("Every number changes the simulation; nothing is overridden.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- small coercions
def _f(v, default=0.0):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def _vec(v, default):
    v = list(v) if isinstance(v, (list, tuple)) else []
    return [_f(v[i], float(default[i])) if i < len(v) else float(default[i]) for i in range(len(default))]


def _rgb(v, default=(0.6, 0.6, 0.6)):
    c = _vec(v, default)
    if max(c) > 1.0:                # answered on a 0-255 scale
        c = [x / 255.0 for x in c]
    return [round(_clip(x, 0.0, 1.0), 3) for x in c]


def _enum(v, allowed, default):
    v = str(v).strip().lower() if v is not None else ""
    return v if v in allowed else default


def _slug(s, fallback):
    s = re.sub(r"[^a-z0-9]+", "_", str(s or "").lower()).strip("_")
    return s[:48] or fallback


def _r(x, nd=3):
    return round(float(x), nd)


def _num(v, nd=3):
    """Compact number for prompts."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{round(float(v), nd):g}"


def aabb(o):
    """(x0, x1, y0, y1, z0, z1) of an obstacle or mat in contract form (pos centre, MuJoCo half sizes, yaw)."""
    x, y, z = (_f(v) for v in (list(o.get("pos") or []) + [0, 0, 0])[:3])
    s = [abs(_f(v)) for v in (list(o.get("size") or []) + [0, 0, 0])[:3]]
    kind = o.get("kind", "box")
    if kind == "cylinder":
        ex = ey = s[0]
        hz = s[1]
    elif kind == "sphere":
        ex = ey = hz = s[0]
    else:
        yaw = math.radians(_f(o.get("yaw_deg")))
        c, si = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey, hz = s[0] * c + s[1] * si, s[0] * si + s[1] * c, s[2]
    return x - ex, x + ex, y - ey, y + ey, z - hz, z + hz


def _union(boxes):
    return (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes),
            max(b[3] for b in boxes), min(b[4] for b in boxes), max(b[5] for b in boxes))


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


# ----------------------------------------------------------------------------- JSON schemas (Gemini subset)
def _o(props, required=None, desc=None):
    s = {"type": "object", "properties": props, "required": list(props) if required is None else required,
         "additionalProperties": False}
    if desc:
        s["description"] = desc
    return s


def _a(items, desc=None, **kw):
    s = {"type": "array", "items": items, **kw}
    if desc:
        s["description"] = desc
    return s


def _n(desc=None, **kw):
    return {"type": "number", **({"description": desc} if desc else {}), **kw}


def _s(desc=None):
    return {"type": "string", **({"description": desc} if desc else {})}


def _b(desc=None):
    return {"type": "boolean", **({"description": desc} if desc else {})}


def _e(values, desc=None):
    return {"type": "string", "enum": list(values), **({"description": desc} if desc else {})}


def _v3(desc):
    return {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": desc}


def _rgb_s(desc="colour [r, g, b], each 0-1, sampled from the media"):
    return {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 3, "maxItems": 3,
            "description": desc}


def scene_schema(stunt_types):
    obstacle = _o({
        "id": _s("unique snake_case id naming the thing, e.g. left_wall, railing_glass_1, plant_1_pot, table_top"),
        "kind": _e(KINDS),
        "pos": _v3("centre [x, y, z], metres"),
        "size": _v3("box: half-extents [hx, hy, hz] before yaw; cylinder (upright): [radius, half_height, 0]; "
                    "sphere: [radius, 0, 0]"),
        "yaw_deg": _n("rotation about +Z, degrees"),
        "rgb": _rgb_s(),
        "material": _e(MATERIALS, "stone for ceramic or pottery, fabric for foliage and upholstery"),
        "role": _e(ROLES),
        "hazard": _b("true for everything in the stunt corridor: the performer must not touch it"),
    })
    couch = _o({"x": _n(), "y": _n(), "yaw_deg": _n("0 = backrest toward +X"), "length": _n(), "depth": _n(),
                "seat_h": _n(), "back_h": _n(), "rgb": _rgb_s()})
    chair = _o({"x": _n(), "y": _n(), "yaw_deg": _n("0 = backrest toward +X"), "rgb": _rgb_s()})
    person = _o({"x": _n(), "y": _n(), "yaw_deg": _n("0 = facing -X"), "posture": _e(["seated", "standing"]),
                 "seat_z": _n("seat height for a seated person; about 0.9 (hip height) for a standing one"),
                 "rgb": _rgb_s("shirt colour")})
    speaker = _o({"x": _n(), "y": _n(), "z_bottom": _n(), "width": _n(), "height": _n(), "stand_height": _n()})
    fixture = _o({"kind": _e(["pipe", "beam"]), "axis": _e(["x", "y"]), "pos": _v3("centre [x, y, z]"),
                  "length": _n(), "radius": _n(), "rgb": _rgb_s()})
    screen = _o({"x": _n(), "y": _n(), "width": _n(), "height": _n(), "bottom_z": _n()})
    curtain = _o({"present": _b("false unless the media shows a curtain"), "rgb": _rgb_s(), "panel_width": _n(),
                  "fold_depth": _n(), "height": _n()})
    mat = _o({"id": _s(), "pos": _v3("centre; z = half thickness (on the floor)"), "size": _v3("half-extents"),
              "rgb": _rgb_s("mat colour, e.g. [0.15, 0.35, 0.85]")})
    return _o({
        "notes": _s("what the media shows, the layout and sizes, the scale cues you used, how the camera moves, "
                    "uncertainties"),
        "stunt": _o({"type": _e(stunt_types), "start_x": _n(), "takeoff_x": _n(), "landing_x": _n(),
                     "lane_y": _n(), "turn_x": _n("run_turn: x where the turn begins; aerial: = landing_x"),
                     "finish": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                                "description": "run_turn: [x, y] where the performer ends; aerial: [landing_x, lane_y]"},
                     "notes": _s("the choreography; for a turn: its side in the camera's view and in world y, and why")}),
        "obstacles": _a(obstacle, "0 to 60 obstacles", maxItems=60),
        "crash_mats": _a(mat, "aerial stunts: 1 to 3 crash mats; ground stunts: none", maxItems=3),
        "set_dressing": _o({"curtain": curtain, "couches": _a(couch, maxItems=8), "chairs": _a(chair, maxItems=24),
                            "people": _a(person, maxItems=12), "speakers": _a(speaker, maxItems=6),
                            "ceiling_fixtures": _a(fixture, maxItems=12), "screens": _a(screen, maxItems=4)}),
        "room": _o({"x_min": _n(), "x_max": _n(), "y_min": _n(), "y_max": _n(), "ceiling_z": _n(),
                    "floor_rgb": _rgb_s(), "wall_rgb": _rgb_s(), "ceiling_rgb": _rgb_s(),
                    "floor_material": _e(FLOOR_MATERIALS)}),
        "camera": _o({"match": _o({"pos": _v3("camera position"), "lookat": _v3("a point on the optical axis"),
                                   "fovy": _n("vertical field of view in degrees, 10-120", minimum=10, maximum=120)})}),
        "source": _o({"ref_time_sec": _n("video: time of the reference frame in seconds (in a still part of the "
                                         "clip); photo: 0", minimum=0)}),
    })


def param_schema(P, plan=None):
    """The PARAM_SPECS fields that apply to the plan's stunt type, with their ranges, plus the camera choice."""
    stype = stunt_type_of(plan)
    props = {}
    for name in relevant_names(P, stype):
        spec = P.PARAM_SPECS[name]
        ch = _choices(name, spec)
        if name == "camera":
            continue
        if _is_numeric(spec):
            props[name] = _n(f"{spec.get('doc', '')} [{spec.get('unit', '')}], range {spec['min']:g}..{spec['max']:g}",
                             minimum=spec["min"], maximum=spec["max"])
        elif ch:
            props[name] = _e(ch, spec.get("doc"))
    props["camera"] = _e(camera_choices(P), "camera the take is rendered and judged from; default "
                         + default_camera(stype, P))
    return _o(props)


def propose_schema(P, n, plan=None):
    cand = _o({"label": _s("2-4 words"),
               "why": _s("1-2 sentences citing the measured numbers this take fixes or explores"),
               "params": param_schema(P, plan)})
    return _o({"rationale": _s("2-4 sentences: what the measurements say and the plan for this wave"),
               "candidates": _a(cand, f"exactly {n} candidates", minItems=n, maxItems=n)})


def critique_schema(P, video_sec=None, plan=None):
    t = _n("seconds from the start of the video", minimum=0, **({"maximum": round(video_sec + 0.5, 2)} if video_sec else {}))
    names = [k for k in relevant_names(P, stunt_type_of(plan)) if k != "camera"]
    change = {"param": _e(names), "change": _e(CHANGES),
              "value": _n("new value inside the parameter's range"), "reason": _s()}
    if not all(_is_numeric(P.PARAM_SPECS[k]) for k in names):
        change["choice"] = _s("only for a non-numeric parameter: the chosen option; otherwise an empty string")
    if is_ground(plan):
        return _o({
            "events": _a(_o({"t_sec": t, "event": _e(EVENT_KINDS), "note": _s()}),
                         "dense log, in time order, at least every 0.5 s from the start of the run to the finish",
                         minItems=6),
            "summary": _s(),
            "gait": _o({"plausible": _b("reads as a believable run: cadence, stride, arm swing, lean"),
                        "foot_skating": _b("a planted foot visibly slides on the floor"),
                        "notes": _s()}),
            "turn": _o({"screen_side": _e(TURN_SIDES, "which edge of the frame the performer heads toward after "
                                                     "the turn, as the viewer sees it"),
                        "performer_side": _e(TURN_SIDES, "the turn from the performer's own point of view"),
                        "start_sec": _n("video time the turn begins", minimum=0),
                        "end_sec": _n("video time the turn ends", minimum=0),
                        "timing_ok": _b("the turn happens where and when the brief and the footage put it"),
                        "notes": _s()}),
            "timing": _o({"playback_speed": _n("1.0 = real time"), "believable": _b(), "notes": _s()}),
            "brief_match": _o({"matches": _b(), "missing": _a(_s())}),
            "reaches_finish": _b("the performer ends at the intended spot"),
            "unnatural_motion": _a(_o({"t_sec": t, "issue": _s(), "severity": _e(SEVERITIES)})),
            "framing": _o({"in_frame_pct": _n("0-100", minimum=0, maximum=100), "subject_size": _e(SUBJECT_SIZES),
                           "notes": _s()}),
            "style_score": {"type": "integer", "description": "0-100", "minimum": 0, "maximum": 100},
            "gate2_pass": _b(),
            "proposed_changes": _a(_o(change), "at most 5 changes", maxItems=6),
        })
    return _o({
        "events": _a(_o({"t_sec": t, "event": _e(EVENT_KINDS), "note": _s()}),
                     "dense log, in time order, at least every 0.5 s around the flight", minItems=6),
        "summary": _s(),
        "timing": _o({"airborne_video_sec": _n(), "flight_playback_speed": _n("1.0 = real time"),
                      "believable": _b(), "notes": _s()}),
        "brief_match": _o({"matches": _b(), "missing": _a(_s())}),
        "rotation_axis": _e(["roll", "pitch", "yaw", "none"]),
        "rotation_observed_deg": _n("degrees about the required axis, take-off to landing", minimum=0, maximum=1440),
        "lands_on_mat": _b(),
        "unnatural_motion": _a(_o({"t_sec": t, "issue": _s(), "severity": _e(SEVERITIES)})),
        "framing": _o({"in_frame_pct": _n("0-100", minimum=0, maximum=100), "subject_size": _e(SUBJECT_SIZES),
                       "notes": _s()}),
        "style_score": {"type": "integer", "description": "0-100", "minimum": 0, "maximum": 100},
        "gate2_pass": _b(),
        "proposed_changes": _a(_o(change), "at most 5 changes", maxItems=6),
    })


def schema_errors(obj, schema, path="$"):
    """Errors of `obj` against the JSON-Schema subset used above (for fixtures and answers)."""
    errs = []
    t = schema.get("type")
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool}
    if t in kinds and not isinstance(obj, kinds[t]):
        return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if t in ("number", "integer"):
        if isinstance(obj, bool) or not isinstance(obj, (int, float)) or (t == "integer" and not float(obj).is_integer()):
            return [f"{path}: expected {t}, got {obj!r}"]
        if "minimum" in schema and obj < schema["minimum"] or "maximum" in schema and obj > schema["maximum"]:
            errs.append(f"{path}: {obj} outside [{schema.get('minimum')}, {schema.get('maximum')}]")
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        props = schema.get("properties", {})
        errs += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in obj]
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in obj if k not in props]
        for k, v in obj.items():
            if k in props:
                errs += schema_errors(v, props[k], f"{path}.{k}")
    if t == "array":
        if len(obj) < schema.get("minItems", 0) or len(obj) > schema.get("maxItems", 1 << 30):
            errs.append(f"{path}: {len(obj)} items, want {schema.get('minItems', 0)}..{schema.get('maxItems', 'any')}")
        for i, v in enumerate(obj):
            errs += schema_errors(v, schema.get("items", {}), f"{path}[{i}]")
    return errs


# ----------------------------------------------------------------------------- prompt helpers
def scene_summary(plan):
    """Compact text of a contract plan.json for propose/critique prompts."""
    st = plan.get("stunt") or {}
    obs = [o for o in plan.get("obstacles") or [] if isinstance(o, dict)]
    lines = []
    tgt = [o for o in obs if o.get("role") == "stunt_target"]
    if tgt:
        b = _union([aabb(o) for o in tgt])
        ids = ", ".join(o.get("id", "?") for o in tgt[:6]) + (", ..." if len(tgt) > 6 else "")
        lines.append(f"Stunt target ({ids}): x {b[0]:.2f}..{b[1]:.2f}, y {b[2]:.2f}..{b[3]:.2f}, top {b[5]:.2f} m.")
    haz = [o for o in obs if o.get("hazard")]
    lane = _f(st.get("lane_y"))
    path = (min(_f(st.get("takeoff_x")), _f(st.get("landing_x"))), max(_f(st.get("takeoff_x")), _f(st.get("landing_x"))),
            lane - CORRIDOR_HALF_WIDTH, lane + CORRIDOR_HALF_WIDTH)
    in_path = [(aabb(o), o) for o in haz if _overlap(aabb(o), path)]
    text = f"Hazards: {len(haz)} obstacles, {len(in_path)} under the flight path"
    if in_path:
        b, o = max(in_path, key=lambda bo: bo[0][5])
        text += f"; tallest there: {o.get('id')} (top {b[5]:.2f} m at x {o['pos'][0]:.2f}, y {o['pos'][1]:.2f})"
    lines.append(text + ".")
    for m in plan.get("crash_mats") or []:
        b = aabb(m)
        lines.append(f"Crash mat {m.get('id')}: centre ({m['pos'][0]:.2f}, {m['pos'][1]:.2f}), "
                     f"{b[1] - b[0]:.2f} x {b[3] - b[2]:.2f} m, top {b[5]:.2f} m.")
    lines.append(f"Layout: run-up from x {_f(st.get('start_x')):.2f}, take-off x {_f(st.get('takeoff_x')):.2f}, "
                 f"landing x {_f(st.get('landing_x')):.2f}, lane y {lane:.2f}; type {st.get('type')}: {st.get('notes', '')}")
    cam = (plan.get("camera") or {}).get("match") or {}
    if cam:
        lines.append(f"Match camera at {[_r(v, 2) for v in cam.get('pos', [])]} looking at "
                     f"{[_r(v, 2) for v in cam.get('lookat', [])]}, fovy {cam.get('fovy')} deg.")
    room = plan.get("room") or {}
    if room:
        lines.append(f"Room x {room.get('x_min')}..{room.get('x_max')}, y {room.get('y_min')}..{room.get('y_max')}, "
                     f"ceiling {room.get('ceiling_z')} m, {room.get('floor_material', 'concrete')} floor.")
    return "\n".join(lines)


def params_line(p, P):
    return ", ".join(f"{k} {_num(p[k])}" for k in P.PARAM_SPECS if k in p)


def metrics_line(m):
    """One compact line of a contract metrics.json."""
    if not m:
        return "no measurements"
    if m.get("ok") is False:
        return f"simulation failed: {m.get('error')}"
    g, s, sc = m.get("gate1") or {}, m.get("sim") or {}, m.get("score") or {}
    bits = []
    if "head_g_peak" in g:
        bits.append(f"head G {_num(g['head_g_peak'], 2)} at {_num(g.get('head_g_peak_t', 0), 2)} s"
                    + (f", HIC15 {_num(g['hic15'], 1)}" if "hic15" in g else ""))
    if "hazard_collisions" in g:
        hc = f"hazard collisions {g['hazard_collisions']}"
        for c in (g.get("hazard_contacts") or [])[:3]:
            hc += f" [{c.get('body')}/{c.get('object')} at {_num(c.get('t', 0), 2)} s, {_num(c.get('depth_cm', 0), 1)} cm deep]"
        bits.append(hc)
    if "min_clearance_cm" in g:
        at = g.get("min_clearance_at") or {}
        bits.append(f"min clearance {_num(g['min_clearance_cm'], 1)} cm"
                    + (f" ({at.get('body')}/{at.get('object')} at {_num(at.get('t', 0), 2)} s)" if at else ""))
    ld = g.get("landing") or {}
    if ld:
        bits.append(f"landing on {ld.get('surface')} at {[_r(v, 2) for v in ld.get('point') or []]} "
                    f"(target {[_r(v, 2) for v in ld.get('target') or []]}), error {_num(ld.get('error_m', 0), 2)} m, "
                    f"{'inside' if ld.get('inside_mat') else 'NOT inside'} the mat")
    if "framing_pct" in g:
        bits.append(f"framing {_num(g['framing_pct'], 1)} %")
    rot = g.get("rotation_deg") or {}
    if rot:
        bits.append("rotation " + ", ".join(f"{k} {_num(v, 0)} deg" for k, v in rot.items()))
    if s:
        bits.append(f"flight {_num(s.get('flight_time', 0), 2)} s, apex {_num(s.get('apex_m', 0), 2)} m, "
                    f"take-off velocity {[_r(v, 2) for v in s.get('takeoff_velocity') or []]} m/s")
    if sc:
        bits.append("score " + ", ".join(f"{k} {_num(v, 1)}" for k, v in sc.items()))
    verdict = g.get("passed")
    if verdict is not None:
        fails = "; ".join(str(x) for x in g.get("failures") or [])
        bits.append("Gate 1 PASSED" if verdict else f"Gate 1 FAILED: {fails}")
    return "; ".join(bits)


def critique_line(cr):
    if not cr:
        return "no critique"
    um = "; ".join(f"{_num(u.get('t_sec', 0), 1)} s {u.get('severity')}: {u.get('issue')}"
                   for u in (cr.get("unnatural_motion") or [])[:4])
    pc = "; ".join(f"{c.get('param')} {c.get('change')} {_num(c.get('value'))} ({c.get('reason')})"
                   for c in (cr.get("proposed_changes") or [])[:5])
    reasons = "; ".join(cr.get("gate2_reasons") or [])
    return (f"Gate 2 {'PASSED' if cr.get('gate2_pass') else 'FAILED'}{': ' + reasons if reasons else ''}; style "
            f"{cr.get('style_score')}; rotation seen {cr.get('rotation_observed_deg')} deg; framing "
            f"{(cr.get('framing') or {}).get('in_frame_pct')} % ({(cr.get('framing') or {}).get('subject_size')}); "
            f"{cr.get('summary', '')}" + (f" Unnatural motion: {um}." if um else "") + (f" Proposed: {pc}." if pc else ""))


def _score(c):
    m = c.get("metrics") or {}
    return (bool((m.get("gate1") or {}).get("passed")), _f((m.get("score") or {}).get("stunt_score"), -1.0))


def best_so_far(history):
    """(candidate {params, metrics}, wave number) with the best (Gate 1 passed, stunt_score), or (None, None)."""
    best, best_wave = None, None
    for w in history.get("waves") or []:
        for c in w.get("candidates") or []:
            if best is None or _score(c) > _score(best):
                best, best_wave = c, w.get("wave")
    return best, best_wave


def history_summary(history, P, full_waves=2):
    waves = history.get("waves") or []
    if not waves:
        return "No take has run yet."
    out = []
    for w in waves[:-full_waves]:
        cands = w.get("candidates") or []
        b = next((c for c in cands if (c.get("params") or {}).get("id") == w.get("best_id")), None)
        out.append(f"wave {w.get('wave')}: best {w.get('best_id')} -> {metrics_line((b or {}).get('metrics'))}"
                   + (f"; critique: Gate 2 {'PASSED' if (w.get('critique') or {}).get('gate2_pass') else 'FAILED'}, "
                      f"{(w.get('critique') or {}).get('summary', '')}" if w.get("critique") else ""))
    for w in waves[-full_waves:]:
        out.append(f"wave {w.get('wave')} (best {w.get('best_id')})"
                   + (f", director steer: \"{w['steer']}\"" if w.get("steer") else "") + ":")
        for c in w.get("candidates") or []:
            p = c.get("params") or {}
            out.append(f"- {p.get('id')} \"{p.get('label', '')}\": {params_line(p, P)}")
            out.append(f"  -> {metrics_line(c.get('metrics'))}")
        if w.get("critique"):
            out.append(f"  Argon critique of {w.get('best_id')}: {critique_line(w['critique'])}")
    return "\n".join(out)


def director_notes(history, steer=None):
    """Standing director notes, oldest first: history.director_notes (loop.py) or each wave's steer, then the
    current steer (--steer or history.current_steer)."""
    notes = [str(x) for x in history.get("director_notes") or [] if x]
    if not notes:
        notes = [str(w["steer"]) for w in history.get("waves") or [] if w.get("steer")]
    for s in (history.get("current_steer"), steer):
        if s and (not notes or notes[-1] != s):
            notes.append(str(s))
    return [f'"{n}"' for n in notes]


def required_rotation(plan, params):
    """(axis, degrees) the brief needs: from the stunt type, else the candidate's largest rot_*_deg."""
    st = (plan.get("stunt") or {}).get("type")
    if st in TYPE_ROTATION:
        return TYPE_ROTATION[st]
    rots = {a: abs(_f(params.get(f"rot_{a}_deg"))) for a in ("roll", "pitch", "yaw")}
    axis = max(rots, key=rots.get)
    return (axis, rots[axis]) if rots[axis] > 0 else ("none", 0.0)


# ----------------------------------------------------------------------------- scene
SCENE_PROMPT = """You are the set builder for FALLGUY, a stunt pre-visualisation harness. From the director's \
{media_word} you rebuild the location as a MuJoCo scene in which a stunt double (a Unitree H1 humanoid, 1.79 m tall, \
51.4 kg) performs the director's brief. Your plan is the only geometry the physics engine sees: an object you leave \
out cannot be hit, so a take that passes in simulation could hurt a real performer.

DIRECTOR'S BRIEF: "{brief}"
FINAL VIDEO LENGTH: {duration:g} s
{media_lines}

WORLD FRAME (metres): the floor is z = 0. +X is the stunt's travel direction (run-up -> take-off -> landing), +Y is \
the performer's left when travelling +X, +Z is up. Put the origin on the floor under the centre of the stunt target. \
Pick the travel direction the brief implies (over a table or shelf: across its short side). Express everything, the \
camera included, in this frame.
Geoms use MuJoCo conventions: pos is the centre; box size = [half_x, half_y, half_z] (HALF-extents, before yaw); \
cylinder size = [radius, half_height, 0] (upright); sphere size = [radius, 0, 0]; yaw_deg turns about +Z.

SCALE PRIORS (use them to fix the metric scale): an adult is about 1.75 m tall (eyes at 1.6 m); dining and work \
tables are about 0.75 m high; sofa seats about 0.45 m and sofa backs about 0.8 m; chair seats about 0.45 m; shelf \
boards about 3 cm thick; doors about 2.0 m high; floor tiles often 0.6 m.{exif_line}

FILL THE FIELDS IN ORDER:
1. notes: what you see, which object is the stunt target and why (the brief may name it loosely: a "table" can be a \
desk, a counter or a low shelf unit), the scale cues you used and anything uncertain.
2. stunt: type, the closest of {types} ("custom" only if none fits). start_x = where the run-up starts (2-5 m before \
take-off for a running stunt); takeoff_x = take-off point before the target's near face; landing_x = landing point \
beyond the target's far face, on the crash mat; lane_y = the line the performer travels along (the clearest line \
across the target). notes: one sentence of choreography.
3. obstacles: the stunt target and EVERY object in or near the stunt corridor (x from start_x - 0.5 to landing_x + \
1.5, y within lane_y +- {corridor:g} m, floor to 3 m): furniture, everything standing on it (plants, pots, lamps, \
candles, books, bottles, bowls), posts, stands, steps, low walls, cables at height. Never omit an object because it is \
small or partly hidden. Split open furniture into its parts (each shelf board, post or leg, the table top) so the gaps \
are real; a prop on a surface sits on it (pos z = surface top + half height). role: stunt_target for every part of the \
target; furniture, prop or structure for the rest. hazard: true for everything in the corridor; false only well \
outside it. Unique snake_case ids, realistic rgb from the media, a material. At most 60 obstacles; background clutter \
belongs in set_dressing.
4. crash_mats: at least one crash mat where the performer lands, on the far side of the stunt target, flat on the \
floor (pos z = half thickness), big enough to land on (half-size about [0.9-1.2, 1.0-1.5, 0.1-0.2]), not overlapping \
any obstacle. The landing point (landing_x, lane_y) must lie on a mat.
5. set_dressing: the rest of the location so the render looks like it: couches, chairs, people (seated or standing), \
speakers on stands, screens, ceiling pipes or beams, and a curtain along the back wall if there is one (present = \
false otherwise). Keep everything where it really is.
6. room: walls and ceiling enclosing everything, the camera and the whole run-up (x_min <= start_x - 1, x_max >= \
landing_x + 2); colours sampled from the media; the floor material.
7. camera.match: the camera that took the {ref_word}: pos, lookat (a point on its optical axis) and fovy (VERTICAL \
field of view in degrees). Reason from perspective: where the floor meets the target, converging lines, the target's \
apparent size against its metric size, and the horizon (the camera's eye height).
8. source.ref_time_sec: {ref_time_text}"""


def scene_prompt(brief, duration, kind, stunt_types, info):
    if kind == "image":
        media_word, ref_word, ref_time_text = "photo", "photo", "0 (a photo)."
        media_lines = f"MEDIA: one photo of the location ({info.get('width')}x{info.get('height')} px)."
        if info.get("ref_image"):
            media_lines += " Image 2 is the reference frame for camera matching."
            ref_word = "reference frame (Image 2)"
    else:
        dur = info.get("video_sec")
        media_word = "video"
        media_lines = (f"MEDIA: a video of the location" + (f", {dur:.1f} s long" if dur else "")
                       + f", sampled for you at {info['fps']:g} fps (timestamps are seconds from its start).")
        if info.get("ref_image"):
            at = info.get("ref_time")
            media_lines += (" The image after the video is the reference frame for camera matching, a still from this "
                            "video" + (f" at {at:g} s." if at is not None else "."))
            ref_word = "reference still (the image)"
            ref_time_text = (f"{at:g} (given)." if at is not None else
                             "the time in the video where the reference still was taken (best match).")
        else:
            ref_word = "frame at source.ref_time_sec"
            ref_time_text = ("the time of the clearest frame of the stunt area: the reference frame for camera "
                             "matching; camera.match must be for that frame.")
    fovy = info.get("fovy_exif")
    exif_line = (f"\nThe reference image's EXIF focal length gives a vertical field of view of {fovy:.1f} degrees: use "
                 f"it for camera.match.fovy." if fovy else "")
    return SCENE_PROMPT.format(media_word=media_word, brief=brief, duration=duration, media_lines=media_lines,
                               exif_line=exif_line, types=", ".join(stunt_types), corridor=CORRIDOR_HALF_WIDTH,
                               ref_word=ref_word, ref_time_text=ref_time_text)


def _need(d, k):
    v = _f(d.get(k), None)
    if v is None:
        raise ValueError(f"missing {k}")
    return v


def _pos(d, k, default):
    v = _f(d.get(k), default)
    return v if v > 0 else default


SD_NORMALIZERS = {
    "couches": lambda c: {"x": _need(c, "x"), "y": _need(c, "y"), "yaw_deg": _f(c.get("yaw_deg")),
                          "length": _pos(c, "length", 2.0), "depth": _pos(c, "depth", 0.9),
                          "seat_h": _pos(c, "seat_h", 0.45), "back_h": _pos(c, "back_h", 0.8),
                          "rgb": _rgb(c.get("rgb"), (0.75, 0.74, 0.72))},
    "chairs": lambda c: {"x": _need(c, "x"), "y": _need(c, "y"), "yaw_deg": _f(c.get("yaw_deg")),
                         "rgb": _rgb(c.get("rgb"), (0.15, 0.15, 0.16))},
    "people": lambda p: {"x": _need(p, "x"), "y": _need(p, "y"), "yaw_deg": _f(p.get("yaw_deg")),
                         "posture": _enum(p.get("posture"), ("seated", "standing"), "seated"),
                         "seat_z": max(0.0, _f(p.get("seat_z"), 0.45)), "rgb": _rgb(p.get("rgb"), (0.2, 0.2, 0.22))},
    "speakers": lambda s: {"x": _need(s, "x"), "y": _need(s, "y"), "z_bottom": max(0.0, _f(s.get("z_bottom"), 1.0)),
                           "width": _pos(s, "width", 0.35), "height": _pos(s, "height", 0.6),
                           "stand_height": max(0.0, _f(s.get("stand_height"), _f(s.get("z_bottom"), 1.0)))},
    "ceiling_fixtures": lambda f: {"kind": _enum(f.get("kind"), ("pipe", "beam"), "pipe"),
                                   "axis": _enum(f.get("axis"), ("x", "y"), "y"),
                                   "pos": [_need({"v": v}, "v") for v in (list(f.get("pos") or []) + [None] * 3)[:3]],
                                   "length": _pos(f, "length", 4.0), "radius": _pos(f, "radius", 0.06),
                                   "rgb": _rgb(f.get("rgb"), (0.7, 0.7, 0.72))},
    "screens": lambda s: {"x": _need(s, "x"), "y": _need(s, "y"), "width": _pos(s, "width", 3.0),
                          "height": _pos(s, "height", 1.8), "bottom_z": max(0.0, _f(s.get("bottom_z"), 1.0))},
}


def _obstacle(o, i, W):
    kind = _enum(o.get("kind"), KINDS, "box")
    if kind != o.get("kind"):
        W.append(f"obstacle {i}: kind {o.get('kind')!r} -> box")
    pos = _vec(o.get("pos"), (0.0, 0.0, 0.0))
    raw = _vec(o.get("size"), (0.1, 0.1, 0.1))
    need = {"box": 3, "cylinder": 2, "sphere": 1}[kind]
    size = [abs(v) for v in raw[:need]]
    if any(v < 0.005 for v in size):
        W.append(f"obstacle {o.get('id', i)}: size {raw[:need]} has a non-positive part; set to >= 0.005 m")
        size = [max(v, 0.005) for v in size]
    size += [0.0] * (3 - need)
    oid = _slug(o.get("id"), f"obstacle_{i + 1}")
    half_h = size[2] if kind == "box" else size[1] if kind == "cylinder" else size[0]
    if pos[2] - half_h < -0.01:
        W.append(f"obstacle {oid}: bottom at z {pos[2] - half_h:.2f} is below the floor; raised onto it")
        pos[2] = half_h
    role = _enum(o.get("role"), ROLES, "prop")
    return {"id": oid, "kind": kind, "pos": [_r(v) for v in pos], "size": [_r(v, 4) for v in size],
            "yaw_deg": _r(((_f(o.get("yaw_deg")) + 180.0) % 360.0) - 180.0, 1), "rgb": _rgb(o.get("rgb")),
            "material": _enum(o.get("material"), MATERIALS, "wood" if role in ("stunt_target", "furniture") else "plastic"),
            "role": role, "hazard": bool(o.get("hazard", True)) if not isinstance(o.get("hazard"), str)
            else o["hazard"].lower() == "true"}


def normalize_plan(ans, *, brief, duration, media, kind, ref_frame=None, ref_time=None, video_sec=None,
                   performer="h1", stunt_types=None, fovy_exif=None, mock=False):
    """Validate and repair Argon's scene answer into a contract plan.json. Returns (plan, warnings)."""
    W = []
    stunt_types = list(stunt_types or ["barrel_roll", "dive_roll", "front_flip", "back_flip", "vault", "high_fall", "custom"])

    # obstacles and the stunt target
    obstacles, seen = [], set()
    for i, o in enumerate(ans.get("obstacles") or []):
        if not isinstance(o, dict):
            W.append(f"obstacle {i}: not an object, dropped")
            continue
        ob = _obstacle(o, i, W)
        base, k = ob["id"], 2
        while ob["id"] in seen:
            ob["id"], k = f"{base}_{k}", k + 1
        seen.add(ob["id"])
        obstacles.append(ob)
    if not obstacles:
        raise ArgonError("scene: Argon's plan has no obstacles, so there is no stunt target; run scene again")
    st_in = ans.get("stunt") or {}
    lane = _f(st_in.get("lane_y"), 0.0)
    if not any(o["role"] == "stunt_target" for o in obstacles):     # the prompt puts the origin under the target
        pick = min(obstacles, key=lambda o: (o["role"] not in ("furniture", "structure"), math.hypot(*o["pos"][:2])))
        pick["role"] = "stunt_target"
        W.append(f"no obstacle had role stunt_target; chose {pick['id']} (the furniture nearest the origin)")
    tgt_ids = [o["id"] for o in obstacles if o["role"] == "stunt_target"]
    for o in obstacles:
        if o["role"] == "stunt_target" and not o["hazard"]:
            o["hazard"] = True
    t = _union([aabb(o) for o in obstacles if o["role"] == "stunt_target"])

    # stunt layout, consistent with the target
    raw_type = re.sub(r"[\s-]+", "_", str(st_in.get("type") or "").strip().lower())
    stype = raw_type if raw_type in stunt_types else "custom"
    if stype != raw_type:
        W.append(f"stunt.type {st_in.get('type')!r} is not one of {stunt_types}; using custom")
    take = _f(st_in.get("takeoff_x"), t[0] - 0.8)
    land = _f(st_in.get("landing_x"), t[1] + 1.2)
    start = _f(st_in.get("start_x"), take - 3.0)
    if stype == "high_fall":
        if land < t[1] + 0.3:
            W.append(f"landing_x {land:.2f} is not beyond the target's far face (x {t[1]:.2f}); moved to {t[1] + 1.0:.2f}")
            land = t[1] + 1.0
    else:
        if take > t[0] - 0.3:
            W.append(f"takeoff_x {take:.2f} is not before the target's near face (x {t[0]:.2f}); moved to {t[0] - 0.8:.2f}")
            take = t[0] - 0.8
        if land < t[1] + 0.4:
            W.append(f"landing_x {land:.2f} is not beyond the target's far face (x {t[1]:.2f}); moved to {t[1] + 1.2:.2f}")
            land = t[1] + 1.2
    if land <= take:
        land = take + 1.5
    if start > take - 1.5:
        W.append(f"start_x {start:.2f} leaves less than 1.5 m of run-up; moved to {take - 3.0:.2f}")
        start = take - 3.0
    if not t[2] - 0.3 <= lane <= t[3] + 0.3:
        new = _clip(lane, t[2], t[3])
        W.append(f"lane_y {lane:.2f} misses the target (y {t[2]:.2f}..{t[3]:.2f}); moved to {new:.2f}")
        lane = new
    stunt = {"type": stype, "start_x": _r(start), "takeoff_x": _r(take), "landing_x": _r(land), "lane_y": _r(lane),
             "notes": str(st_in.get("notes") or "").strip()}

    # hazards: everything in the stunt corridor
    corridor = (start - 0.5, land + 1.5, lane - CORRIDOR_HALF_WIDTH, lane + CORRIDOR_HALF_WIDTH)
    for o in obstacles:
        b = aabb(o)
        if not o["hazard"] and _overlap(b, corridor) and b[4] < 3.0:
            o["hazard"] = True
            W.append(f"{o['id']} lies in the stunt corridor; marked hazard")

    # crash mats: on the floor, beyond the target, under the landing point
    mats, seen = [], set()
    for i, m in enumerate(ans.get("crash_mats") or []):
        if not isinstance(m, dict):
            continue
        size = [abs(v) for v in _vec(m.get("size"), (0.9, 1.0, 0.15))]
        if min(size) < 0.02:
            W.append(f"crash mat {m.get('id', i)}: size {size} too small; using at least [0.5, 0.5, 0.05]")
            size = [max(size[0], 0.5), max(size[1], 0.5), max(size[2], 0.05)]
        pos = _vec(m.get("pos"), (land, lane, size[2]))
        if abs(pos[2] - size[2]) > 0.02:
            W.append(f"crash mat {m.get('id', i)}: not resting on the floor (z {pos[2]:.2f}); set z = {size[2]:.2f}")
            pos[2] = size[2]
        mid = _slug(m.get("id"), f"mat{i + 1}")
        while mid in seen:
            mid += "_b"
        seen.add(mid)
        mats.append({"id": mid, "pos": pos, "size": size, "rgb": _rgb(m.get("rgb"), (0.15, 0.35, 0.85))})
    if not mats:
        mats.append({"id": "mat1", "pos": [land + 0.2, lane, 0.15], "size": [0.9, 1.0, 0.15], "rgb": [0.15, 0.35, 0.85]})
        W.append(f"no crash mat in the answer; added mat1 at x {land + 0.2:.2f}")

    def on(m, x, y):
        return abs(x - m["pos"][0]) <= m["size"][0] and abs(y - m["pos"][1]) <= m["size"][1]

    if not any(on(m, land, lane) for m in mats):
        m = min(mats, key=lambda m: math.hypot(m["pos"][0] - land, m["pos"][1] - lane))
        W.append(f"landing point ({land:.2f}, {lane:.2f}) is on no mat; centred {m['id']} on it")
        m["pos"][0], m["pos"][1] = land, lane
    if stype != "high_fall":
        for m in mats:
            mb = aabb(dict(m, kind="box"))
            if _overlap(mb, t):
                carried = on(m, land, lane)
                m["pos"][0] = t[1] + 0.05 + m["size"][0]
                W.append(f"crash mat {m['id']} overlapped the stunt target; moved to x {m['pos'][0]:.2f}")
                if carried and not on(m, land, lane):
                    land = m["pos"][0]
                    stunt["landing_x"] = _r(land)
                    W.append(f"landing_x moved onto {m['id']} at {land:.2f}")
    for m in mats:
        m["pos"] = [_r(v) for v in m["pos"]]
        m["size"] = [_r(v) for v in m["size"]]
        for o in obstacles:
            if o["role"] != "stunt_target" and _overlap(aabb(dict(m, kind="box")), aabb(o)) and aabb(o)[4] < 2 * m["size"][2]:
                W.append(f"crash mat {m['id']} overlaps {o['id']}")

    # camera
    cam_in = (ans.get("camera") or {}).get("match") or {}
    tc = [(t[0] + t[1]) / 2, (t[2] + t[3]) / 2, (t[4] + t[5]) / 2]
    cpos = _vec(cam_in.get("pos"), (start - 2.0, lane - 4.0, 1.6))
    look = _vec(cam_in.get("lookat"), tc)
    fovy = _clip(_f(cam_in.get("fovy"), 50.0), 10.0, 120.0)
    if fovy_exif:
        fovy = float(fovy_exif)
    if math.dist(cpos, look) < 0.1:
        look = tc
        W.append("camera.match lookat equals its position; aimed at the stunt target")
    fwd, to_t = [a - b for a, b in zip(look, cpos)], [a - b for a, b in zip(tc, cpos)]
    cosang = sum(a * b for a, b in zip(fwd, to_t)) / max(math.hypot(*fwd) * math.hypot(*to_t), 1e-9)
    if math.degrees(math.acos(_clip(cosang, -1.0, 1.0))) > fovy * 0.75:
        W.append("camera.match does not seem to see the stunt target; check preview.png")
    camera = {"match": {"pos": [_r(v) for v in cpos], "lookat": [_r(v) for v in look], "fovy": _r(fovy, 2)}}

    # room: encloses the corridor, every obstacle, the mats and the camera
    rin = ans.get("room") or {}
    room = {"x_min": _f(rin.get("x_min"), -9.0), "x_max": _f(rin.get("x_max"), 7.0),
            "y_min": _f(rin.get("y_min"), -8.0), "y_max": _f(rin.get("y_max"), 8.0),
            "ceiling_z": _clip(_f(rin.get("ceiling_z"), 4.0), 2.2, 30.0)}
    for lo, hi in (("x_min", "x_max"), ("y_min", "y_max")):
        if room[lo] > room[hi]:
            room[lo], room[hi] = room[hi], room[lo]
            W.append(f"room {lo}/{hi} were swapped")
    boxes = [aabb(o) for o in obstacles if max(abs(v) for v in o["pos"][:2]) < 50] + \
            [aabb(dict(m, kind="box")) for m in mats]
    need = [min([start - 1.0, cpos[0] - 0.3] + [b[0] - 0.2 for b in boxes]),
            max([land + 1.5, cpos[0] + 0.3] + [b[1] + 0.2 for b in boxes]),
            min([lane - 2.0, cpos[1] - 0.3] + [b[2] - 0.2 for b in boxes]),
            max([lane + 2.0, cpos[1] + 0.3] + [b[3] + 0.2 for b in boxes])]
    for key, v, grow in (("x_min", need[0], min), ("x_max", need[1], max), ("y_min", need[2], min), ("y_max", need[3], max)):
        new = grow(room[key], v)
        if abs(new - room[key]) > 0.5:
            W.append(f"room {key} {room[key]:.2f} -> {new:.2f} to enclose the stunt, obstacles and camera")
        room[key] = new
    top = max([b[5] for b in boxes] + [cpos[2]])
    if room["ceiling_z"] < top + 0.3:
        W.append(f"ceiling_z {room['ceiling_z']:.2f} -> {top + 0.3:.2f} (above the tallest object and the camera)")
        room["ceiling_z"] = top + 0.3
    room = {k: _r(v, 2) for k, v in room.items()}
    room.update(floor_rgb=_rgb(rin.get("floor_rgb"), (0.6, 0.6, 0.6)), wall_rgb=_rgb(rin.get("wall_rgb"), (0.8, 0.8, 0.78)),
                ceiling_rgb=_rgb(rin.get("ceiling_rgb"), (0.2, 0.2, 0.2)),
                floor_material=_enum(rin.get("floor_material"), FLOOR_MATERIALS, "concrete"))

    # set dressing (prototype argon_env.py schema + screens); items far outside the room are dropped
    sd_in = ans.get("set_dressing") or {}
    cu = sd_in.get("curtain") if isinstance(sd_in.get("curtain"), dict) else {}
    present = cu.get("present", bool(cu))
    curtain = {} if not present or str(present).lower() == "false" else {
        "rgb": _rgb(cu.get("rgb"), (0.9, 0.86, 0.78)), "panel_width": _r(_clip(_f(cu.get("panel_width"), 0.4), 0.2, 1.5)),
        "fold_depth": _r(_clip(_f(cu.get("fold_depth"), 0.1), 0.02, 0.4)),
        "height": _r(_clip(_f(cu.get("height"), room["ceiling_z"] - 0.2), 0.5, room["ceiling_z"]))}
    sd, in_corridor = {"curtain": curtain}, 0
    for key, norm in SD_NORMALIZERS.items():
        sd[key] = []
        for i, it in enumerate(sd_in.get(key) or []):
            try:
                item = norm(it if isinstance(it, dict) else {})
            except ValueError as e:
                W.append(f"set_dressing.{key}[{i}] dropped: {e}")
                continue
            x, y = (item["pos"][0], item["pos"][1]) if key == "ceiling_fixtures" else (item["x"], item["y"])
            if not (room["x_min"] - 1 <= x <= room["x_max"] + 1 and room["y_min"] - 1 <= y <= room["y_max"] + 1):
                W.append(f"set_dressing.{key}[{i}] at ({x:.1f}, {y:.1f}) is outside the room; dropped")
                continue
            x, y = _clip(x, room["x_min"], room["x_max"]), _clip(y, room["y_min"], room["y_max"])
            if key == "ceiling_fixtures":
                item["pos"] = [_r(x), _r(y), _r(_clip(item["pos"][2], 0.5, room["ceiling_z"] - item["radius"]))]
            else:
                item["x"], item["y"] = _r(x), _r(y)
            in_corridor += int(corridor[0] <= x <= corridor[1] and corridor[2] <= y <= corridor[3] and key != "ceiling_fixtures")
            sd[key].append({k: (_r(v) if isinstance(v, float) else v) for k, v in item.items()})
    if in_corridor:
        W.append(f"{in_corridor} set-dressing item(s) stand in the stunt corridor; they stay and count as hazards")

    rt = 0.0
    if kind == "video":
        rt = ref_time if ref_time is not None else _f((ans.get("source") or {}).get("ref_time_sec"), 0.0)
        rt = _clip(rt, 0.0, video_sec) if video_sec else max(rt, 0.0)
    plan = {
        "version": 1, "brief": brief, "duration_sec": float(duration),
        "source": {"media": media, "kind": kind, "ref_frame": ref_frame or (media if kind == "image" else None),
                   "ref_time_sec": _r(rt, 2)},
        "room": room, "obstacles": obstacles, "set_dressing": sd, "crash_mats": mats, "camera": camera,
        "stunt": stunt, "performer": {"model": performer},
        "argon": {"model": model_name(), "thinking": thinking_level(), "mock": bool(mock),
                  "notes": str(ans.get("notes") or ""), "stunt_target": tgt_ids, "warnings": W},
    }
    return plan, W


def cmd_scene(args):
    P = engine_params()
    media = args.media
    kind = media_kind(media)
    if args.ref_frame and not os.path.isfile(args.ref_frame):
        raise ArgonError(f"reference frame not found: {args.ref_frame}")
    log = resolve_log(args)
    info, parts, video_sec = {}, [], None
    if kind == "video":
        video_sec, _ = video_info(media)
        info.update(video_sec=video_sec, fps=2.0 if video_sec and video_sec <= 30 else 1.0, ref_time=args.ref_time)
    if is_mock():
        ans = mock_answer("scene", log)
    else:
        if kind == "image":
            part, im = image_part(media)
            parts.append(part)
            info.update(im)
            if args.ref_frame and _sha1(args.ref_frame) != _sha1(media):
                rpart, rim = image_part(args.ref_frame)
                parts += [{"text": "Image 2 (reference frame):"}, rpart]
                info.update(ref_image=True, fovy_exif=rim["fovy_exif"] or info.get("fovy_exif"))
        if args.ref_frame and kind == "video":
            rpart, rim = image_part(args.ref_frame)
            info.update(ref_image=True, fovy_exif=rim["fovy_exif"])
        prompt = scene_prompt(args.brief, args.duration, kind, P.STUNT_TYPES, info)
        schema = scene_schema(list(P.STUNT_TYPES))
        if kind == "image":
            ans, _ = generate(parts + [{"text": prompt}], schema, "scene", log)
        else:
            extra = [{"text": "Reference still:"}, rpart] if info.get("ref_image") else []
            ans, _ = generate_with_video(media, info["fps"], lambda vp: [vp] + extra + [{"text": prompt}], schema,
                                         "scene", log, files_cache_path(log))
    for e in schema_errors(ans, scene_schema(list(P.STUNT_TYPES)))[:8]:
        say(f"scene answer deviates from the schema: {e}")
    plan, warns = normalize_plan(ans, brief=args.brief, duration=args.duration, media=media, kind=kind,
                                 ref_frame=args.ref_frame, ref_time=args.ref_time, video_sec=video_sec,
                                 performer=args.performer, stunt_types=P.STUNT_TYPES,
                                 fovy_exif=info.get("fovy_exif"), mock=is_mock())
    for w in warns:
        say(f"plan fix: {w}")
    write_json(args.out, plan)
    print(json.dumps({"out": os.path.abspath(args.out), "kind": kind, "obstacles": len(plan["obstacles"]),
                      "hazards": sum(o["hazard"] for o in plan["obstacles"]),
                      "stunt_target": plan["argon"]["stunt_target"], "crash_mats": len(plan["crash_mats"]),
                      "stunt": plan["stunt"], "warnings": len(warns), "mock": is_mock()}))
    return 0


# ----------------------------------------------------------------------------- propose
PROPOSE_PROMPT = """You are the assistant stunt coordinator in FALLGUY, a recursive stunt pre-visualisation loop. \
Each wave you propose {n} candidate takes. A MuJoCo simulator performs every take with a Unitree H1 humanoid \
(1.79 m, 51.4 kg) as the stunt double: the run-up is scripted until take-off, then everything is free physics (the body \
leaves the ground with the take-off velocity, turns with the angular momentum it was given, and contacts are real). The \
simulator measures Gate 1 for every take; you then watch the best render for Gate 2. Converge on a take that passes both.

DIRECTOR'S BRIEF: "{brief}"
STUNT: {stype}. {snotes}
{notes_block}
SCENE (metres; +X = travel direction, +Y = performer's left, +Z up, floor z = 0):
{scene}

PARAMETERS (every candidate sets all of them; ranges are hard limits):
{table}

PHYSICS TO PLAN WITH (g = 9.81 m/s^2): flight time ~ 2 * takeoff_vz / g when take-off and landing heights match; \
distance covered in the air ~ approach_speed * flight time; the centre of mass rises ~ takeoff_vz^2 / (2 g) above \
its take-off height (about 1 m). Calibrate with the measured flight time, apex and landing point of earlier takes.

GATES:
{gates}

HISTORY:
{history}

YOUR TASK FOR WAVE {wave}:
{task}

Write the rationale first (2-4 sentences: what the numbers say and your plan), then the {n} candidates. Each gets a \
label of 2-4 words, a why of 1-2 sentences that cites the numbers it acts on (e.g. "clearance 1.2 cm < 2 cm at \
left_ankle_link/plant_3 -> takeoff_vz 3.4 -> 3.6"), and a value for every parameter. No two candidates may be equal."""


def propose_prompt(plan, history, P, wave, n, steer):
    st = plan.get("stunt") or {}
    notes = director_notes(history, steer)
    notes_block = ("DIRECTOR NOTES (standing; they override your own preferences, every candidate must respect them):\n"
                   + "\n".join(f"- {x}" for x in notes) + "\n") if notes else ""
    best, best_wave = best_so_far(history)
    if best is None:
        task = (f"No take has run yet. Spread the {n} candidates across the plausible space around the plan (take-off "
                f"near x {_num(st.get('takeoff_x'))}, landing on the mat near x {_num(st.get('landing_x'))}): e.g. a "
                "conservative take (lower and slower, softer landing), a nominal one, a higher and faster one, and one "
                "with different rotation timing or tuck. Every candidate must be a complete, plausible attempt at the "
                "brief with the full required rotation.")
    else:
        bp, bm = best.get("params") or {}, best.get("metrics") or {}
        explore = ("Keep exactly one exploratory candidate in a clearly different region (another take-off point, speed "
                   "or rotation timing) in case the best is a local optimum. " if n > 1 else "")
        task = (f"The best take so far is {bp.get('id')} (wave {best_wave}): {metrics_line(bm)}.\n"
                f"Exploit it: make {n - 1 if n > 1 else 1} refinement(s) of it that fix its exact Gate-1 failures and "
                "the critique's issues, changing one to three parameters each by amounts you justify from the "
                f"measurements (cite them). {explore}Do not repeat a take that already ran.")
    return PROPOSE_PROMPT.format(n=n, brief=plan.get("brief") or history.get("brief", ""), stype=st.get("type"),
                                 snotes=st.get("notes", ""), notes_block=notes_block, scene=scene_summary(plan),
                                 table=P.describe_for_prompt(), gates=gates_text(),
                                 history=history_summary(history, P), wave=wave, task=task)


def wave_number(out, history):
    """The wave being proposed: from the --out directory (waves/waveNN/), else history.wave, else waves + 1."""
    m = re.fullmatch(r"wave0*(\d+)", os.path.basename(os.path.dirname(os.path.abspath(out))))
    hist = history.get("wave") if isinstance(history.get("wave"), int) else len(history.get("waves") or []) + 1
    if m:
        if int(m.group(1)) != hist:
            say(f"note: --out says wave {int(m.group(1))} but history.json says wave {hist}; using {int(m.group(1))}")
        return int(m.group(1))
    return hist


def _clamp(P, params, plan):
    try:
        return P.clamp(params, plan=plan)
    except TypeError:               # an engine.params without the plan argument
        return P.clamp(params)


def _defaults(P, plan):
    try:
        return P.defaults(plan=plan)
    except TypeError:
        return P.defaults()


def mock_propose(plan, history, P, n, log):
    """Deterministic proposal: fixture takes for wave 1, small nudges of the best take afterwards."""
    fx = mock_answer("propose", log)
    cands = copy.deepcopy(fx.get("candidates") or [])
    take = (plan.get("stunt") or {}).get("takeoff_x")
    for c in cands:
        if take is not None and "takeoff_x" in P.PARAM_SPECS:
            c.setdefault("params", {})["takeoff_x"] = take
    best, _ = best_so_far(history)
    rationale = fx.get("rationale", "mock proposal")
    if best is not None:
        bp = best.get("params") or {}
        nudges = [("takeoff_vz", 0.15), ("approach_speed", 0.25), ("tuck", 0.1), ("landing_compliance", 0.1)]
        ref = [{"label": f"refine {k}", "why": f"mock: {bp.get('id')} with {k} {d:+g}",
                "params": {**bp, k: _f(bp.get(k)) + d}} for k, d in nudges if k in P.PARAM_SPECS and k in bp]
        cands = ref[:max(n - 1, 1)] + cands[-1:]
        rationale = f"mock: exploit {bp.get('id')} with small nudges, keep one exploratory take"
    while cands and len(cands) < n:
        cands.append(copy.deepcopy(cands[len(cands) % max(len(cands), 1)]))
    return {"rationale": rationale, "candidates": cands[:n]}


def finish_proposal(ans, P, wave, n, history, plan=None):
    """Flatten, clamp with engine.params.clamp, set ids w<wave>c<k> and colours; fill to n if Argon gave fewer."""
    W, out = [], []
    for k, c in enumerate([c for c in ans.get("candidates") or [] if isinstance(c, dict)][:n], 1):
        params = dict(c.get("params") or {})
        for name in list(P.PARAM_SPECS) + ["camera"]:
            if name in c and name not in params:
                params[name] = c[name]
        cid = f"w{wave}c{k}"
        params.update(id=cid, label=str(c.get("label") or f"take {k}").strip()[:40], color=COLORS[(k - 1) % len(COLORS)],
                      why=str(c.get("why") or "").strip())
        cl = _clamp(P, params, plan)
        for key in ("id", "label", "color", "why"):
            cl[key] = params[key]
        for name, spec in P.PARAM_SPECS.items():
            if _is_numeric(spec) and name in params and abs(_f(params[name], 0.0) - _f(cl.get(name), 0.0)) > 1e-6:
                W.append(f"{cid} {name} {params[name]} -> {cl.get(name)} (clamped)")
        if "camera" in params and params["camera"] != cl.get("camera"):
            W.append(f"{cid} camera {params['camera']!r} -> {cl.get('camera')!r}")
        out.append(cl)
    if len(out) < n:
        W.append(f"Argon returned {len(out)} of {n} candidates; filled the rest with small variations")
        best, _ = best_so_far(history)
        base = (best or {}).get("params") or (out[0] if out else _defaults(P, plan))
        rng = random.Random(wave * 1000 + len(out))
        for k in range(len(out) + 1, n + 1):
            p = dict(base)
            for name, spec in P.PARAM_SPECS.items():
                if _is_numeric(spec) and name in p:
                    p[name] = _f(p[name]) + rng.gauss(0, 0.05) * (spec["max"] - spec["min"])
            cid = f"w{wave}c{k}"
            p.update(id=cid, label=f"variation {k}", color=COLORS[(k - 1) % len(COLORS)],
                     why=f"filler: small random variation of {base.get('id', 'the default take')}")
            cl = _clamp(P, p, plan)
            for key in ("id", "label", "color", "why"):
                cl[key] = p[key]
            out.append(cl)
    seen = set()
    for c in out:
        sig = json.dumps({k: c.get(k) for k in list(P.PARAM_SPECS) + ["camera"]}, sort_keys=True)
        if sig in seen:
            W.append(f"{c['id']} repeats another candidate")
        seen.add(sig)
    return {"rationale": str(ans.get("rationale") or "").strip(), "wave": wave, "candidates": out, "warnings": W}


def cmd_propose(args):
    P = engine_params()
    if args.n < 1 or args.n > 12:
        raise ArgonError(f"--n {args.n}: use 1..12 candidates")
    plan = read_json(args.plan, "plan")
    if os.path.isfile(args.history):
        history = read_json(args.history, "history")
    else:
        say(f"note: {args.history} does not exist; treating this as wave 1 with no history")
        history = {"brief": plan.get("brief", ""), "waves": []}
    wave = wave_number(args.out, history)
    log = resolve_log(args)
    if is_mock():
        ans = mock_propose(plan, history, P, args.n, log)
    else:
        ans, _ = generate([{"text": propose_prompt(plan, history, P, wave, args.n, args.steer)}],
                          propose_schema(P, args.n), "propose", log)
    proposal = finish_proposal(ans, P, wave, args.n, history, plan=plan)
    for w in proposal["warnings"]:
        say(f"proposal fix: {w}")
    write_json(args.out, proposal)
    print(json.dumps({"out": os.path.abspath(args.out), "wave": wave, "ids": [c["id"] for c in proposal["candidates"]],
                      "labels": [c["label"] for c in proposal["candidates"]], "warnings": len(proposal["warnings"]),
                      "mock": is_mock()}))
    return 0


# ----------------------------------------------------------------------------- critique
CRITIQUE_PROMPT = """You are the Gate 2 critic in FALLGUY, a stunt pre-visualisation loop. The video is the render \
of one simulated take: a Unitree H1 humanoid robot (1.79 m) stands in for the stunt double in a MuJoCo rebuild of \
the location. Watch it from start to end and log what happens over time before you judge it.

DIRECTOR'S BRIEF: "{brief}"
{notes_block}REQUIRED ROTATION: {req_text}
VIDEO: {video_line}
{playback}

SCENE (metres; +X = travel direction, +Y = performer's left, +Z up):
{scene}

CANDIDATE TAKE {cid} "{label}": {params}

MUJOCO MEASUREMENTS, GROUND TRUTH (times are simulation seconds). The simulator measures clearance, contacts and \
G-forces exactly and a video cannot, so do not re-estimate them; treat them as facts:
{metrics}

JUDGE ONLY WHAT YOU SEE: timing and pacing, the rotation actually performed, whether the take matches the brief, \
unnatural motion, framing and style.

FILL THE FIELDS IN ORDER:
1. events: a dense timestamped log of what is on screen: every key moment (run_start, takeoff, apex, landing, \
settled) and at least one entry every 0.5 s from 0.5 s before take-off until the performer has settled after landing \
(use "other" in between). Each note says what the body is doing: pose, rotation so far, position against the target \
and the mat.
2. summary: 2-4 sentences: what the take shows and its main problem, if any.
3. timing: airborne time in video seconds, the apparent playback speed of the flight (1.0 = real time) after \
allowing for the slow-motion segments listed above, whether the timing is physically believable, notes.
4. brief_match: does the take do what the brief asks? List what is missing.
5. rotation_axis and rotation_observed_deg: the total rotation about the required axis between take-off and landing \
(360 = one full turn). Count the turns from your event log.
6. lands_on_mat: does the performer come down on the crash mat?
7. unnatural_motion: each problem with its time and severity. high = breaks believability (floating or hovering, \
rotation that speeds up, stops or reverses in the air, limbs passing through objects or the floor, popping or \
teleporting, feet sliding while planted); medium = clearly robotic or stiff; low = minor. Slow motion listed above is \
an intentional edit, not a problem in itself.
8. framing: in_frame_pct = % of the video from the run start to 1 s after landing in which the whole performer is \
inside the frame; subject_size: small (under 15 % of the frame height, hard to read), medium (15-40 %), large (over 40 %).
9. style_score 0-100: how well it reads as the brief's stunt, athletic and cinematic. 90+ ready to show; 70-89 good \
with minor issues; 50-69 recognisable but flawed; under 50 fails the brief.
10. gate2_pass, by this fixed rubric: true only if ALL hold: the brief is satisfied; the required rotation is observed \
within 45 degrees; the performer lands on the mat; no high-severity unnatural motion; in_frame_pct >= {framing_min:g} \
and subject_size medium or large; style_score >= {style_min:g}.
11. proposed_changes: up to 5 parameter changes for the next wave that fix what you saw. Name only these \
parameters: {names}. Give a value inside the parameter's range and cite the time and issue in the reason. (The \
camera is not changed here; put framing advice in framing.notes.) The parameters:
{table}"""


def playback_info(events, video_sec, video_fps):
    """Lines describing slow-motion segments and the video times of the simulator's events, via engine.render.retime."""
    if not events or not video_sec or not video_fps:
        return "Playback speed unknown: parts of the video may be in slow motion around the flight."
    try:
        import numpy as np
        from engine.render import retime
        ts = np.asarray(retime(events, float(video_sec), int(round(video_fps))), dtype=float)
    except Exception as e:
        say(f"note: no playback map ({type(e).__name__}: {e})")
        return "Playback speed unknown: parts of the video may be in slow motion around the flight."
    if ts.ndim != 1 or len(ts) < 3:
        return "Playback speed unknown."
    vt = np.arange(len(ts)) / float(round(video_fps))
    rate = np.gradient(ts) * float(round(video_fps))
    segs = []
    for i, r in enumerate(rate):
        kind = "hold" if r < 0.02 else "real" if abs(r - 1.0) <= 0.08 else ("slow" if r < 1.0 else "fast")
        if segs and segs[-1]["kind"] == kind:
            segs[-1]["end"], segs[-1]["rates"] = vt[i], segs[-1]["rates"] + [r]
        else:
            segs.append({"kind": kind, "start": vt[i], "end": vt[i], "rates": [r]})
    parts = []
    for s in segs:
        span = f"{s['start']:.1f}-{s['end'] + 1.0 / video_fps:.1f} s"
        if s["kind"] == "real":
            parts.append(f"{span} real time")
        elif s["kind"] == "hold":
            parts.append(f"{span} held still frame")
        elif s["kind"] == "slow":
            parts.append(f"{span} slow motion (down to {max(min(s['rates']), 0):.2f}x real time)")
        else:
            parts.append(f"{span} sped up (up to {max(s['rates']):.2f}x)")
    lines = ["PLAYBACK: " + "; ".join(parts) + "."]
    ev = []
    for key, name in (("t_runup", "run-up starts"), ("t_takeoff", "take-off"), ("t_apex", "apex"),
                      ("t_landing", "landing"), ("t_end", "end of motion")):
        t = events.get(key)
        if isinstance(t, (int, float)) and ts[0] <= t <= ts[-1]:
            ev.append(f"{name} {float(np.interp(t, ts, vt)):.2f} s")
    if ev:
        lines.append("SIMULATOR EVENTS IN VIDEO TIME: " + ", ".join(ev) + ".")
    return "\n".join(lines)


def critique_prompt(plan, params, metrics, P, th, req, video_line, playback, notes):
    axis, deg = req
    req_text = (f"about {deg:g} degrees while airborne: {AXIS_TEXT[axis]}." if axis != "none"
                else AXIS_TEXT["none"] + ".")
    notes_block = ("DIRECTOR NOTES (standing): " + " | ".join(notes) + "\n") if notes else ""
    return CRITIQUE_PROMPT.format(brief=plan.get("brief", ""), notes_block=notes_block, req_text=req_text,
                                  video_line=video_line, playback=playback, scene=scene_summary(plan),
                                  cid=params.get("id", "?"), label=params.get("label", ""),
                                  params=params_line(params, P), metrics=metrics_line(metrics),
                                  framing_min=float(th["framing_min_pct"]), style_min=float(th["style_min"]),
                                  names=", ".join(P.PARAM_SPECS), table=P.describe_for_prompt())


def finish_critique(ans, P, plan, params, metrics, th, video=None, fps=None, video_sec=None):
    """Contract critique.json from Argon's answer; gate2_pass is recomputed from the fixed rubric."""
    W = []
    events = []
    for e in ans.get("events") or []:
        if isinstance(e, dict) and _f(e.get("t_sec"), None) is not None:
            events.append({"t_sec": _r(max(_f(e["t_sec"]), 0.0), 2), "event": _enum(e.get("event"), EVENT_KINDS, "other"),
                           "note": str(e.get("note") or "").strip()})
    events.sort(key=lambda e: e["t_sec"])
    if len(events) < 6:
        W.append(f"only {len(events)} events in the log")
    bm = ans.get("brief_match") or {}
    brief_match = {"matches": bm.get("matches") is True, "missing": [str(x) for x in bm.get("missing") or []]}
    axis, req_deg = required_rotation(plan, params)
    seen_deg = abs(_f(ans.get("rotation_observed_deg"), 0.0))
    um = [{"t_sec": _r(max(_f(u.get("t_sec")), 0.0), 2), "issue": str(u.get("issue") or "").strip(),
           "severity": _enum(u.get("severity"), SEVERITIES, "medium")}
          for u in ans.get("unnatural_motion") or [] if isinstance(u, dict)]
    fr = ans.get("framing") or {}
    framing = {"in_frame_pct": _r(_clip(_f(fr.get("in_frame_pct"), 0.0), 0.0, 100.0), 1),
               "subject_size": _enum(fr.get("subject_size"), SUBJECT_SIZES, "small"),
               "notes": str(fr.get("notes") or "").strip()}
    style = int(round(_clip(_f(ans.get("style_score"), 0.0), 0.0, 100.0)))
    changes = []
    for c in ans.get("proposed_changes") or []:
        if not isinstance(c, dict):
            continue
        name = c.get("param")
        if name not in P.PARAM_SPECS:
            W.append(f"dropped a proposed change to unknown parameter {name!r}")
            continue
        spec = P.PARAM_SPECS[name]
        if _is_numeric(spec):
            v = _f(c.get("value"), None)
            if v is None:
                W.append(f"dropped the proposed change to {name}: no numeric value")
                continue
            value = _r(_clip(v, spec["min"], spec["max"]), 4)
        else:
            ch = _choices(name, spec) or []
            value = str(c.get("choice") or c.get("value") or "").strip().lower()
            if value not in ch:
                W.append(f"dropped the proposed change to {name}: {value!r} is not one of {ch}")
                continue
        changes.append({"param": name, "change": _enum(c.get("change"), CHANGES, "set"), "value": value,
                        "reason": str(c.get("reason") or "").strip()})
    landing = ((metrics or {}).get("gate1") or {}).get("landing") or {}
    sim_on_mat = landing.get("inside_mat") if isinstance(landing.get("inside_mat"), bool) else None
    seen_on_mat = ans.get("lands_on_mat") is True
    if sim_on_mat is not None and sim_on_mat != seen_on_mat:
        W.append(f"Argon saw lands_on_mat={seen_on_mat} but MuJoCo measured inside_mat={sim_on_mat}; the rubric uses MuJoCo")
    rubric = {
        "brief_satisfied": brief_match["matches"],
        "rotation_ok": abs(seen_deg - req_deg) <= 45.0,
        "lands_on_mat": sim_on_mat if sim_on_mat is not None else seen_on_mat,
        "no_high_severity_motion": not any(u["severity"] == "high" for u in um),
        "framing_ok": framing["in_frame_pct"] >= float(th["framing_min_pct"]) and framing["subject_size"] != "small",
        "style_ok": style >= float(th["style_min"]),
    }
    why = {"brief_satisfied": "brief not satisfied" + (f" (missing: {', '.join(brief_match['missing'])})" if brief_match["missing"] else ""),
           "rotation_ok": f"rotation {seen_deg:g} deg vs required {req_deg:g} deg ({axis})",
           "lands_on_mat": "does not land on the mat",
           "no_high_severity_motion": "high-severity unnatural motion: " + "; ".join(u["issue"] for u in um if u["severity"] == "high"),
           "framing_ok": f"framing {framing['in_frame_pct']:g} % / subject {framing['subject_size']} "
                         f"(need >= {float(th['framing_min_pct']):g} % and medium or large)",
           "style_ok": f"style {style} < {float(th['style_min']):g}"}
    t_in = ans.get("timing") or {}
    return {
        "summary": str(ans.get("summary") or "").strip(),
        "events": events,
        "brief_match": brief_match,
        "rotation_observed_deg": _r(seen_deg, 1),
        "unnatural_motion": um,
        "framing": framing,
        "style_score": style,
        "gate2_pass": all(rubric.values()),
        "proposed_changes": changes,
        "candidate_id": params.get("id"),
        "rotation_axis": axis,
        "rotation_required_deg": req_deg,
        "lands_on_mat_seen": seen_on_mat,
        "timing": {"airborne_video_sec": _r(_f(t_in.get("airborne_video_sec")), 2),
                   "flight_playback_speed": _r(_f(t_in.get("flight_playback_speed"), 1.0), 2),
                   "believable": t_in.get("believable") is True, "notes": str(t_in.get("notes") or "").strip()},
        "rubric": rubric,
        "gate2_reasons": [why[k] for k, ok in rubric.items() if not ok],
        "argon_gate2_pass": ans.get("gate2_pass") is True,
        "video": {"path": video, "duration_sec": _r(video_sec, 2) if video_sec else None, "sampled_fps": fps},
        "warnings": W,
    }


def mock_critique(plan, params, metrics, log):
    """Canned critique whose verdict follows the MuJoCo metrics, so mock loops can converge."""
    ans = mock_answer("critique", log)
    g1 = (metrics or {}).get("gate1") or {}
    passed = g1.get("passed") is True
    axis, req = required_rotation(plan, params)
    rot = g1.get("rotation_deg") or {}
    ans["rotation_axis"] = axis
    ans["rotation_observed_deg"] = abs(_f(rot.get(axis), req)) if axis in rot else req
    ans["lands_on_mat"] = (g1.get("landing") or {}).get("inside_mat") is True
    fr = ans.setdefault("framing", {})
    fr["in_frame_pct"] = _f(g1.get("framing_pct"), fr.get("in_frame_pct", 90.0))
    fr.setdefault("subject_size", "medium")
    ans["brief_match"] = {"matches": True, "missing": []}
    if passed:
        ans["unnatural_motion"] = [u for u in ans.get("unnatural_motion") or [] if u.get("severity") != "high"]
        ans["style_score"] = max(int(_f(ans.get("style_score"), 0)), 80)
    else:
        ans["style_score"] = min(int(_f(ans.get("style_score"), 55)), 60)
    ans["gate2_pass"] = passed
    return ans


def cmd_critique(args):
    P = engine_params()
    if not 0 < args.fps <= 24:
        raise ArgonError(f"--fps {args.fps}: use a sampling rate in (0, 24]")
    if not os.path.isfile(args.video):
        raise ArgonError(f"video not found: {args.video}")
    plan = read_json(args.plan, "plan")
    params = read_json(args.params, "params")
    metrics = read_json(args.metrics, "metrics")
    ev_path = args.events or next((p for p in (os.path.join(os.path.dirname(os.path.abspath(args.metrics)), "events.json"),
                                               os.path.join(os.path.dirname(os.path.abspath(args.video)), "events.json"))
                                   if os.path.isfile(p)), None)
    events = read_json(ev_path, "events") if ev_path else None
    th = thresholds()
    log = resolve_log(args)
    video_sec, video_fps = video_info(args.video)
    if is_mock():
        ans = mock_critique(plan, params, metrics, log)
    else:
        video_line = ((f"{video_sec:.1f} s long, " if video_sec else "") + (f"{video_fps:g} fps, " if video_fps else "")
                      + f"sampled for you at {args.fps:g} fps. Timestamps are seconds from the start of the video.")
        cam = params.get("camera") or "hero"
        video_line += f" Rendered from the {cam} camera ({CAMERA_TEXT.get(cam, 'a scene camera')})."
        hist_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "history.json")
        hist = read_json(hist_path, "history") if os.path.isfile(hist_path) else {}
        notes = director_notes(hist if isinstance(hist, dict) else {}, args.steer)
        prompt = critique_prompt(plan, params, metrics, P, th, required_rotation(plan, params), video_line,
                                 playback_info(events, video_sec, video_fps), notes)
        ans, _ = generate_with_video(args.video, args.fps, lambda vp: [vp, {"text": prompt}], critique_schema(P, video_sec),
                                     "critique", log, files_cache_path(log))
    crit = finish_critique(ans, P, plan, params, metrics, th, video=args.video, fps=args.fps, video_sec=video_sec)
    for w in crit["warnings"]:
        say(f"critique note: {w}")
    write_json(args.out, crit)
    print(json.dumps({"out": os.path.abspath(args.out), "candidate_id": crit["candidate_id"],
                      "gate2_pass": crit["gate2_pass"], "style_score": crit["style_score"],
                      "reasons": crit["gate2_reasons"], "events": len(crit["events"]), "mock": is_mock()}))
    return 0


# ----------------------------------------------------------------------------- ping + CLI
def cmd_ping(args):
    log = resolve_log(args)
    t0 = time.time()
    if is_mock():
        ans, resp = mock_answer("ping", log), {"modelVersion": "mock"}
    else:
        schema = _o({"pong": _b("always true")})
        ans, resp = generate([{"text": 'FALLGUY health check. Reply with the JSON object {"pong": true}.'}],
                             schema, "ping", log)
    ok = ans.get("pong") is True
    print(json.dumps({"ok": ok, "model": model_name(), "thinking": thinking_level(),
                      "model_version": resp.get("modelVersion"), "latency_s": round(time.time() - t0, 1),
                      "mock": is_mock()}))
    if not ok:
        raise ArgonError(f"ping: unexpected answer {json.dumps(ans)[:120]}")
    return 0


def _parser():
    ap = argparse.ArgumentParser(prog="argon.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ping", help="check the key, the model and the thinking level")
    p.add_argument("--log", help="call log (JSONL); default: none for ping")
    s = sub.add_parser("scene", help="build plan.json from site media + brief")
    s.add_argument("--media", required=True, help="site photo or video")
    s.add_argument("--brief", required=True)
    s.add_argument("--duration", required=True, type=float, help="final video length, seconds")
    s.add_argument("--out", required=True, help="plan.json to write")
    s.add_argument("--ref-frame", help="still used for camera matching (recorded as source.ref_frame)")
    s.add_argument("--ref-time", type=float, help="video only: time of --ref-frame in the video, seconds")
    s.add_argument("--performer", default="h1", choices=["h1", "humanoid"])
    s.add_argument("--log")
    q = sub.add_parser("propose", help="propose the next wave of candidate takes")
    q.add_argument("--plan", required=True)
    q.add_argument("--history", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--steer", help="director's note for this and later waves")
    q.add_argument("--n", type=int, default=4)
    q.add_argument("--log")
    c = sub.add_parser("critique", help="watch a render and judge it (Gate 2)")
    c.add_argument("--video", required=True)
    c.add_argument("--plan", required=True)
    c.add_argument("--params", required=True)
    c.add_argument("--metrics", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--fps", type=float, default=5.0, help="sampling rate Argon watches at (default 5)")
    c.add_argument("--events", help="events.json (default: next to --metrics or --video)")
    c.add_argument("--steer", help="standing director notes")
    c.add_argument("--log")
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    if getattr(args, "duration", 1.0) <= 0:
        print("argon.py scene: error: --duration must be > 0", file=sys.stderr)
        return 2
    try:
        return {"ping": cmd_ping, "scene": cmd_scene, "propose": cmd_propose, "critique": cmd_critique}[args.cmd](args)
    except ArgonError as e:
        print(f"argon.py {args.cmd}: error: {' '.join(str(e).split())}", file=sys.stderr)
        return 1
    except Exception as e:      # a bug: still one specific, redacted line (full trace with FALLGUY_ARGON_DEBUG=1)
        if os.environ.get("FALLGUY_ARGON_DEBUG"):
            print(redact(traceback.format_exc()), file=sys.stderr)
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(f"argon.py {args.cmd}: error: {type(e).__name__}: {' '.join(redact(str(e)).split())} "
              f"({os.path.basename(tb.filename)}:{tb.lineno} in {tb.name})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
