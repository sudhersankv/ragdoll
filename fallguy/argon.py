#!/usr/bin/env python3
"""FALLGUY Argon client: gemini-4-argon plans the scene and the choreography, proposes candidate takes and
critiques renders. There are no stunt types: plan.stunt is generic phases + tunables (reference/contracts.md).

    python argon.py ping
    python argon.py scene    --media inputs/media.mp4 --brief "..." --duration 6 --out plan.json
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
CORRIDOR_HALF_WIDTH = 1.2           # m either side of the path: what the performer can reach
COLORS = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981"]

KINDS = ["box", "cylinder", "sphere"]
MATERIALS = ["wood", "metal", "fabric", "foam", "glass", "plastic", "concrete", "stone"]
FLOOR_MATERIALS = ["concrete", "wood", "carpet", "tile", "stone", "rubber", "grass", "asphalt", "other"]
ROLES = ["wall", "railing", "furniture", "prop", "structure", "person"]
EVENT_KINDS = ["start", "takeoff", "apex", "landing", "turn", "stop", "settled", "other"]
# choreography vocabulary (reference/contracts.md, "Choreography phases"): no stunt types, only generic phases
PHASE_KINDS = ["hold", "move", "jump", "fall", "air", "land"]
POSES = ["stand", "crouch", "takeoff", "dive", "tuck", "pike", "layout", "land_prep", "brace", "arms_up"]
RIG_KEYS = ["l_shoulder_x", "l_shoulder_y", "l_elbow", "r_shoulder_x", "r_shoulder_y", "r_elbow", "l_hip_y", "l_knee",
            "r_hip_y", "r_knee", "torso"]
GAITS = ["walk", "run", "sprint"]
TARGET_KINDS = ["mat", "zone", "none"]
GENERIC_PARAMS = ["mat_offset_x", "mat_offset_y"]
META_KEYS = ["id", "label", "color", "camera", "why"]
SEVERITIES = ["low", "medium", "high"]
SUBJECT_SIZES = ["small", "medium", "large"]
CHANGES = ["increase", "decrease", "set"]
CAMERAS = ["match_follow", "match", "hero", "side"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"}
VIDEO_MIME = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mov", ".webm": "video/webm",
              ".mpeg": "video/mpeg", ".mpg": "video/mpg", ".avi": "video/avi", ".wmv": "video/wmv",
              ".3gp": "video/3gpp", ".flv": "video/x-flv"}
CAMERA_TEXT = {"hero": "a 3/4 tracking dolly on the performer", "side": "a side-on tracking dolly",
               "match": "the director's camera position and angle, static",
               "match_follow": "the director's camera position, panning to follow the performer"}
# Contract defaults; scripts/gates.py is the source of truth when it imports.
THRESHOLD_DEFAULTS = {"head_g_max": 8.0, "lateral_g_max": 0.8, "hazard_collisions_max": 0, "min_clearance_cm": 2.0,
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
    """The real scripts/engine/params.py: specs(plan), clamp(params, plan), apply(plan, params),
    describe_for_prompt(plan), CAMERAS (reference/contracts.md)."""
    try:
        from engine import params
    except Exception as e:
        raise ArgonError(f"cannot import engine.params from {os.path.join(HERE, 'engine')}: {type(e).__name__}: {e}")
    for name in ("specs", "clamp", "apply", "describe_for_prompt"):
        if not callable(getattr(params, name, None)):
            raise ArgonError(f"engine.params has no {name}() (see reference/contracts.md)")
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
    return (f"Gate 1 (MuJoCo, every take): head G peak (15 ms average) <= {th['head_g_max']}; lateral G in moves <= "
            f"{th['lateral_g_max']}; hazard collisions <= {th['hazard_collisions_max']}; min clearance >= "
            f"{th['min_clearance_cm']} cm; end point within {th['landing_error_max_m']} m of the target (on the mat or "
            f"in the zone); framing >= {th['framing_min_pct']} %.\nGate 2 (critique of the best render): Gate 1 "
            f"passed; the brief's moves happen in order and direction; no high-severity unnatural motion; style >= "
            f"{th['style_min']}; final stunt score >= {th['gate2_min_score']}, where stunt score = {w['safety']} safety "
            f"+ {w['clearance']} clearance + {w['landing']} landing + {w['framing']} framing.")


def _is_numeric(spec):
    return isinstance(spec, dict) and all(isinstance(spec.get(k), (int, float)) and not isinstance(spec.get(k), bool)
                                          for k in ("min", "max"))


def specs_of(P, plan):
    """engine.params.specs(plan), numeric entries only: name -> {min, max, unit, default, doc}."""
    try:
        sp = P.specs(plan)
    except Exception as e:
        raise ArgonError(f"engine.params.specs(plan) failed: {type(e).__name__}: {e}")
    return {k: v for k, v in (sp or {}).items() if _is_numeric(v)}


def camera_choices(P):
    return [str(c) for c in (getattr(P, "CAMERAS", None) or CAMERAS)]


def default_camera(plan, P=None):
    cams = camera_choices(P) if P is not None else list(CAMERAS)
    cam = ((plan or {}).get("stunt") or {}).get("default_camera")
    return cam if cam in cams else ("match_follow" if "match_follow" in cams else cams[0])


def _clamp(P, params, plan):
    try:
        return dict(P.clamp(params, plan))
    except Exception as e:
        raise ArgonError(f"engine.params.clamp failed: {type(e).__name__}: {e}")


def _defaults(P, plan):
    """The plan's own values: clamp of an empty candidate."""
    return _clamp(P, {"camera": default_camera(plan, P)}, plan)


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


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _seg_box_dist(p, q, b):
    """Horizontal distance from segment p-q to the box footprint b (0 when they touch), sampled."""
    n = max(2, int(math.dist(p, q) / 0.1) + 1)
    best = float("inf")
    for i in range(n):
        u = i / (n - 1)
        x, y = p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u
        dx = max(b[0] - x, 0.0, x - b[1])
        dy = max(b[2] - y, 0.0, y - b[3])
        best = min(best, math.hypot(dx, dy))
    return best


# ----------------------------------------------------------------------------- stunt paths ("phases[1].path[2][1]")
_PATH_TOK = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(-?\d+)\]")


def path_steps(path):
    """Steps of a tunable path into the stunt dict, or None when it does not parse. A leading "stunt." is allowed."""
    s = re.sub(r"^stunt\.", "", str(path or "").strip())
    steps, pos = [], 0
    while pos < len(s):
        m = _PATH_TOK.match(s, pos)
        if not m or m.end() == pos:
            return None
        steps.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
        pos = m.end()
    return steps or None


def resolve(obj, path):
    """The value at a tunable path in the stunt dict; KeyError when the path does not lead anywhere."""
    steps = path_steps(path)
    if steps is None:
        raise KeyError(path)
    for st in steps:
        if isinstance(st, int):
            if not isinstance(obj, list) or not -len(obj) <= st < len(obj):
                raise KeyError(path)
        elif not isinstance(obj, dict) or st not in obj:
            raise KeyError(path)
        obj = obj[st]
    return obj


def path_text(steps):
    return "".join(f"[{s}]" if isinstance(s, int) else (f".{s}" if i else s) for i, s in enumerate(steps))


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


def _xy(desc="[x, y] in metres"):
    return {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": desc}


def _rgb_s(desc="colour [r, g, b], each 0-1, sampled from the media"):
    return {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 3, "maxItems": 3,
            "description": desc}


def phase_schema():
    air_pose = _o({"at": _n("0..1, share of the flight"), "pose": _e(POSES), "tuck": _n("0..1")})
    return _o({
        "kind": _e(PHASE_KINDS),
        "name": _s("short unique snake_case name, e.g. ready, run, turn, takeoff"),
        "duration": _n("hold: seconds held; land: seconds after first contact"),
        "pose": _e(POSES, "hold, jump, fall, land: the body pose"),
        "path": _a(_xy(), "move: >= 2 [x, y] points; the first is where the performer is when the move starts",
                   minItems=2, maxItems=12),
        "speed": _n("move: cruise speed m/s (walk 1.2-1.8, jog 2.5-3.5, run 4-6)"),
        "accel_time": _n("move: s from the speed at the start of the move to cruise"),
        "decel_time": _n("move: s to stop at the end of the path; 0 = arrive at speed"),
        "turn_radius": _n("move: corner rounding radius, m"),
        "lean_scale": _n("move: bank into turns as a multiple of the physical bank atan(v^2 / (r g)), usually 1"),
        "gait": _e(GAITS, "move: optional; auto from speed"),
        "vz": _n("jump: vertical take-off velocity m/s"),
        "vx": _n("jump: optional speed along the heading m/s (default keeps the move speed)"),
        "lean_deg": _n("jump: body lean at take-off, deg"),
        "rotations_deg": _o({"roll": _n(), "pitch": _n("+ front flip"), "yaw": _n("+ turns left")},
                            desc="jump: rotation completed over the flight, deg"),
        "push": _v3("fall: [vx, vy, vz] push off, m/s"),
        "poses": _a(air_pose, "air: pose schedule over the flight", maxItems=8),
        "compliance": _n("land: 0..1, leg softness"),
    }, required=["kind", "name"])


def stunt_schema():
    tunable = _o({"name": _s("unique snake_case, e.g. run_speed"),
                  "path": _s("path into stunt, e.g. phases[1].speed, phases[1].path[2][1], phases[3].vz"),
                  "min": _n(), "max": _n(), "unit": _s(), "doc": _s("what it changes")})
    return _o({
        "notes": _s("how the brief maps onto this location: where the performer starts and goes, how the camera's "
                    "left/right were resolved into +Y/-Y and from what evidence in the footage"),
        "summary": _s("one sentence: what the performer does, in this location"),
        "start": _o({"pos": _xy("start position [x, y]"), "heading_deg": _n("0: +X is the initial facing"),
                     "z": _n("0 on the floor; the top of an obstacle when starting on one")},
                    required=["pos", "heading_deg"]),
        "phases": _a(phase_schema(), "ordered phases", minItems=1, maxItems=12),
        "tunables": _a(tunable, "2 to 8 numbers the loop may change", maxItems=10),
        "target": _o({"kind": _e(TARGET_KINDS), "id": _s("mat: the crash mat id"), "pos": _xy("zone: centre"),
                      "radius": _n("zone: radius m")}, required=["kind"]),
        "default_camera": _e(CAMERAS),
    })


def scene_schema():
    obstacle = _o({
        "id": _s("unique snake_case id naming the thing, e.g. left_wall, glass_railing, door_frame_left, bench_1"),
        "kind": _e(KINDS),
        "pos": _v3("centre [x, y, z], metres"),
        "size": _v3("box: half-extents [hx, hy, hz] before yaw; cylinder (upright): [radius, half_height, 0]; "
                    "sphere: [radius, 0, 0]"),
        "yaw_deg": _n("rotation about +Z, degrees"),
        "rgb": _rgb_s(),
        "material": _e(MATERIALS, "stone for ceramic or pottery, fabric for foliage, curtains and upholstery"),
        "role": _e(ROLES),
        "hazard": _b("true for everything within reach of the performer's path"),
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
    curtain = _o({"present": _b("true only if the media shows a curtain along a room wall"), "rgb": _rgb_s(),
                  "panel_width": _n(), "fold_depth": _n(), "height": _n()}, required=["present"])
    mat = _o({"id": _s(), "pos": _v3("centre; z = half thickness (on the floor)"), "size": _v3("half-extents"),
              "rgb": _rgb_s("mat colour, e.g. [0.15, 0.35, 0.85]")})
    return _o({
        "notes": _s("what the media shows (layout, sizes, the scale cues you used), how the camera moves over time, "
                    "uncertainties"),
        "source": _o({"ref_time_sec": _n("video: time of the reference frame in a still part of the clip; photo: 0",
                                         minimum=0)}),
        "camera": _o({"match": _o({"pos": _v3("camera position"), "lookat": _v3("a point on the optical axis"),
                                   "fovy": _n("VERTICAL field of view in degrees", minimum=10, maximum=120)})}),
        "room": _o({"x_min": _n(), "x_max": _n(), "y_min": _n(), "y_max": _n(), "ceiling_z": _n(),
                    "floor_rgb": _rgb_s(), "wall_rgb": _rgb_s(), "ceiling_rgb": _rgb_s(),
                    "floor_material": _e(FLOOR_MATERIALS)}),
        "obstacles": _a(obstacle, "0 to 60 obstacles", maxItems=60),
        "set_dressing": _o({"curtain": curtain, "couches": _a(couch, maxItems=8), "chairs": _a(chair, maxItems=24),
                            "people": _a(person, maxItems=12), "speakers": _a(speaker, maxItems=6),
                            "ceiling_fixtures": _a(fixture, maxItems=12), "screens": _a(screen, maxItems=4)},
                           required=[], desc="optional scenery away from the path, only what the media shows"),
        "crash_mats": _a(mat, "only when the stunt lands from a jump or fall; otherwise empty", maxItems=3),
        "stunt": stunt_schema(),
    })


def param_schema(P, plan):
    """One number per engine.params.specs(plan) entry (range in the description) plus the camera choice."""
    props = {}
    for name, spec in specs_of(P, plan).items():
        props[name] = _n(f"{spec.get('doc', '')} [{spec.get('unit', '')}], range {spec['min']:g}..{spec['max']:g}",
                         minimum=spec["min"], maximum=spec["max"])
    props["camera"] = _e(camera_choices(P), "camera the take is rendered and judged from; default "
                         + default_camera(plan, P))
    return _o(props)


def propose_schema(P, n, plan):
    cand = _o({"label": _s("2-4 words"),
               "why": _s("1-2 sentences citing the measured numbers this take fixes or explores"),
               "params": param_schema(P, plan)})
    return _o({"rationale": _s("2-4 sentences: what the measurements say and the plan for this wave"),
               "candidates": _a(cand, f"exactly {n} candidates", minItems=n, maxItems=n)})


def phase_names(plan):
    return [str(p.get("name") or f"phase_{i}") for i, p in enumerate(((plan or {}).get("stunt") or {}).get("phases") or [])
            if isinstance(p, dict)]


def required_rotation(plan):
    """(axis, degrees) the choreography asks for in its jump phases (largest axis), or ("none", 0)."""
    tot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    for p in ((plan or {}).get("stunt") or {}).get("phases") or []:
        if isinstance(p, dict) and p.get("kind") == "jump":
            for a in tot:
                tot[a] += abs(_f((p.get("rotations_deg") or {}).get(a)))
    axis = max(tot, key=tot.get)
    return (axis, tot[axis]) if tot[axis] >= 45 else ("none", 0.0)


def critique_schema(P, plan, video_sec=None):
    t = _n("seconds from the start of the video", minimum=0, **({"maximum": round(video_sec + 0.5, 2)} if video_sec else {}))
    names = list(specs_of(P, plan)) or ["mat_offset_x"]
    change = {"param": _e(names), "change": _e(CHANGES), "value": _n("new value inside the parameter's range"),
              "reason": _s()}
    props = {
        "events": _a(_o({"t_sec": t, "event": _e(EVENT_KINDS), "note": _s()}),
                     "dense log in time order, at least one entry every 0.5 s while the performer moves", minItems=6),
        "summary": _s(),
        "phases_seen": _a(_o({"phase": _e(phase_names(plan) or ["stunt"]), "performed": _b(),
                              "start_sec": _n("video time it starts; -1 if not seen"), "notes": _s()}),
                          "one entry per planned phase, in the plan's order"),
        "order_ok": _b("the phases happen in the planned order"),
        "direction": _o({"screen_side": _e(["left", "right", "toward", "away", "none"],
                                           "where the performer heads at the end, as the viewer sees it"),
                         "matches_brief": _b("the direction matches the brief and the footage"), "notes": _s()}),
        "motion": _o({"plausible": _b("reads as a believable human body: cadence, stride, arm swing, balance"),
                      "foot_skating": _b("a planted foot visibly slides on the floor"), "notes": _s()}),
        "brief_match": _o({"matches": _b(), "missing": _a(_s())}),
        "reaches_target": _b("the performer ends on the mat or in the target area"),
    }
    if required_rotation(plan)[0] != "none":
        props["rotation_observed_deg"] = _n("degrees about the required axis, take-off to landing", minimum=0,
                                            maximum=1440)
    props.update({
        "unnatural_motion": _a(_o({"t_sec": t, "issue": _s(), "severity": _e(SEVERITIES)})),
        "framing": _o({"in_frame_pct": _n("0-100", minimum=0, maximum=100), "subject_size": _e(SUBJECT_SIZES),
                       "notes": _s()}),
        "style_score": {"type": "integer", "description": "0-100", "minimum": 0, "maximum": 100},
        "gate2_pass": _b(),
        "proposed_changes": _a(_o(change), "at most 5 changes", maxItems=6),
    })
    return _o(props)


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
def phase_line(i, p):
    k = p.get("kind")
    bits = [f"[{i}] {k} \"{p.get('name')}\""]
    if k == "hold":
        bits.append(f"{_num(p.get('duration'))} s, pose {p.get('pose')}")
    elif k == "move":
        bits.append("path " + " -> ".join(f"({_num(x, 2)}, {_num(y, 2)})" for x, y in p.get("path") or []))
        bits.append(f"speed {_num(p.get('speed'))} m/s, accel {_num(p.get('accel_time'))} s, decel "
                    f"{_num(p.get('decel_time'))} s, turn radius {_num(p.get('turn_radius'))} m, lean x{_num(p.get('lean_scale'))}")
    elif k == "jump":
        rot = p.get("rotations_deg") or {}
        bits.append(f"vz {_num(p.get('vz'))} m/s" + (f", vx {_num(p.get('vx'))} m/s" if p.get("vx") is not None else "")
                    + f", lean {_num(p.get('lean_deg', 0))} deg, rotations roll {_num(rot.get('roll', 0))} pitch "
                    f"{_num(rot.get('pitch', 0))} yaw {_num(rot.get('yaw', 0))} deg, pose {p.get('pose')}")
    elif k == "fall":
        bits.append(f"push {p.get('push')}, pose {p.get('pose')}")
    elif k == "air":
        bits.append("poses " + ", ".join(f"{_num(q.get('at'), 2)}: {q.get('pose')} (tuck {_num(q.get('tuck', 0), 2)})"
                                         for q in p.get("poses") or []))
    elif k == "land":
        bits.append(f"pose {p.get('pose')}, compliance {_num(p.get('compliance'))}, {_num(p.get('duration'))} s")
    return ": ".join([bits[0], ", ".join(bits[1:])]) if len(bits) > 1 else bits[0]


def choreography_text(plan):
    st = plan.get("stunt") or {}
    start = st.get("start") or {}
    lines = [f"Summary: {st.get('summary', '')}",
             f"Start: ({_num(_vec(start.get('pos'), (0, 0))[0], 2)}, {_num(_vec(start.get('pos'), (0, 0))[1], 2)}), "
             f"heading {_num(start.get('heading_deg', 0))} deg" + (f", z {_num(start['z'])} m" if start.get("z") else "")]
    lines += [phase_line(i, p) for i, p in enumerate(st.get("phases") or []) if isinstance(p, dict)]
    tg = st.get("target") or {}
    if tg.get("kind") == "mat":
        lines.append(f"Target: crash mat {tg.get('id')}")
    elif tg.get("kind") == "zone":
        lines.append(f"Target: end inside the zone centred ({_num(_vec(tg.get('pos'), (0, 0))[0], 2)}, "
                     f"{_num(_vec(tg.get('pos'), (0, 0))[1], 2)}), radius {_num(tg.get('radius'))} m")
    else:
        lines.append("Target: none")
    if st.get("notes"):
        lines.append(f"Notes: {st['notes']}")
    return "\n".join(lines)


def path_points(stunt):
    """The planned ground track: start, every move path point, and a rough landing point after a jump."""
    start = _vec((stunt.get("start") or {}).get("pos"), (0.0, 0.0))
    pts, cur, speed, heading = [start], start, 0.0, math.radians(_f((stunt.get("start") or {}).get("heading_deg")))
    for p in stunt.get("phases") or []:
        k = p.get("kind")
        if k == "move":
            path = [_vec(q, (0.0, 0.0)) for q in p.get("path") or []]
            pts += path[1:]
            if len(path) >= 2:
                cur = path[-1]
                heading = math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0])
            speed = 0.0 if _f(p.get("decel_time")) > 0 else _f(p.get("speed"))
        elif k == "jump":
            vx = _f(p.get("vx"), speed) if p.get("vx") is not None else speed
            t_air = 2 * max(_f(p.get("vz")), 0.0) / 9.81 + 0.1
            land = [cur[0] + vx * t_air * math.cos(heading), cur[1] + vx * t_air * math.sin(heading)]
            pts.append(land)
            cur = land
        elif k == "fall":
            push = _vec(p.get("push"), (0.0, 0.0, 0.0))
            land = [cur[0] + push[0] * 0.6 + 0.5 * math.cos(heading), cur[1] + push[1] * 0.6 + 0.5 * math.sin(heading)]
            pts.append(land)
            cur = land
    return pts


def near_path(o, pts, reach=CORRIDOR_HALF_WIDTH):
    b = aabb(o)
    if b[4] > 3.0:
        return False
    if len(pts) == 1:
        return _seg_box_dist(pts[0], pts[0], b) <= reach
    return any(_seg_box_dist(p, q, b) <= reach for p, q in zip(pts, pts[1:]))


def scene_summary(plan):
    """Compact text of a contract plan.json for propose/critique prompts."""
    st = plan.get("stunt") or {}
    obs = [o for o in plan.get("obstacles") or [] if isinstance(o, dict)]
    pts = path_points(st) if st else []
    lines = []
    haz = [o for o in obs if o.get("hazard")]
    close = [o for o in haz if pts and near_path(o, pts)]
    text = f"Hazards: {len(haz)} obstacles, {len(close)} within {CORRIDOR_HALF_WIDTH:g} m of the path"
    if close:
        text += ": " + "; ".join(f"{o.get('id')} ({o.get('role')}, centre ({_num(o['pos'][0], 2)}, {_num(o['pos'][1], 2)}), "
                                 f"top {_num(aabb(o)[5], 2)} m)" for o in close[:10])
    lines.append(text + ".")
    for m in plan.get("crash_mats") or []:
        b = aabb(m)
        lines.append(f"Crash mat {m.get('id')}: centre ({m['pos'][0]:.2f}, {m['pos'][1]:.2f}), "
                     f"{b[1] - b[0]:.2f} x {b[3] - b[2]:.2f} m, top {b[5]:.2f} m.")
    cam = (plan.get("camera") or {}).get("match") or {}
    if cam:
        lines.append(f"Director's camera at {[_r(v, 2) for v in cam.get('pos', [])]} looking at "
                     f"{[_r(v, 2) for v in cam.get('lookat', [])]}, vertical fov {cam.get('fovy')} deg.")
    room = plan.get("room") or {}
    if room:
        lines.append(f"Room x {room.get('x_min')}..{room.get('x_max')}, y {room.get('y_min')}..{room.get('y_max')}, "
                     f"ceiling {room.get('ceiling_z')} m, {room.get('floor_material', 'concrete')} floor.")
    return "\n".join(lines)


def params_line(p, names):
    return ", ".join(f"{k} {_num(p[k])}" for k in names if k in p) + (f", camera {p['camera']}" if p.get("camera") else "")


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
    if "lateral_g_peak" in g:
        bits.append(f"lateral G {_num(g['lateral_g_peak'], 2)}")
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
        bits.append(f"ends on {ld.get('surface')} at {[_r(v, 2) for v in ld.get('point') or []]} "
                    f"(target {[_r(v, 2) for v in ld.get('target') or []]}), error {_num(ld.get('error_m', 0), 2)} m, "
                    f"{'inside' if ld.get('inside_mat') else 'NOT inside'} the target")
    if "framing_pct" in g:
        bits.append(f"framing {_num(g['framing_pct'], 1)} %")
    rot = g.get("rotation_deg") or {}
    if rot:
        bits.append("rotation " + ", ".join(f"{k} {_num(v, 0)} deg" for k, v in rot.items()))
    if s:
        sim = []
        if s.get("airborne"):
            sim.append(f"airborne {_num(s.get('t_takeoff'), 2)}-{_num(s.get('t_landing'), 2)} s, apex {_num(s.get('apex_m'), 2)} m")
        if s.get("peak_speed") is not None:
            sim.append(f"peak speed {_num(s.get('peak_speed'), 2)} m/s")
        if s.get("t_finish") is not None:
            sim.append(f"finish {_num(s.get('t_finish'), 2)} s")
        if sim:
            bits.append(", ".join(sim))
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
            f"{cr.get('style_score')}; framing {(cr.get('framing') or {}).get('in_frame_pct')} % "
            f"({(cr.get('framing') or {}).get('subject_size')}); {cr.get('summary', '')}"
            + (f" Unnatural motion: {um}." if um else "") + (f" Proposed: {pc}." if pc else ""))


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


def history_summary(history, names, full_waves=2):
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
            out.append(f"- {p.get('id')} \"{p.get('label', '')}\": {params_line(p, names)}")
            out.append(f"  -> {metrics_line(c.get('metrics'))}")
        if w.get("critique"):
            out.append(f"  Argon critique of {w.get('best_id')}: {critique_line(w['critique'])}")
    return "\n".join(out)


def director_notes(history, steer=None):
    """Standing director notes, newest first: history.director_notes (loop.py: {after_wave, text} or strings), else
    each wave's steer, plus the current steer (--steer or history.current_steer)."""
    notes = []
    for x in history.get("director_notes") or []:
        t = x.get("text") if isinstance(x, dict) else x
        if t and str(t) not in notes:
            notes.append(str(t))
    if not notes:
        notes = [str(w["steer"]) for w in history.get("waves") or [] if w.get("steer")]
    for s in (history.get("current_steer"), steer):
        if s and str(s) not in notes:
            notes.append(str(s))
    return [f'"{n}"' for n in reversed(notes)]


# ----------------------------------------------------------------------------- scene
SCENE_PROMPT = """You are the set builder and stunt choreographer for FALLGUY, a stunt pre-visualisation harness. \
From the director's {media_word} you rebuild the location as a MuJoCo scene, and you write the choreography that a \
stunt double (a Unitree H1 humanoid robot, 1.79 m tall, 51.4 kg) performs there for the director's brief. Your plan \
is the only geometry the physics engine sees: an object you leave out cannot be hit, so a take that passes in \
simulation could hurt a real performer. Build only what the media shows. Assume nothing about the kind of place: \
name and size what you actually see (walls, railings, doors, columns, steps, furniture, props, people).

DIRECTOR'S BRIEF: "{brief}"
FINAL VIDEO LENGTH: {duration:g} s
{media_lines}

WORLD FRAME (metres): the floor is z = 0. +X is the performer's initial facing direction at the start of the stunt, \
+Y is the performer's left when facing +X, +Z is up. Put the origin on the floor at a convenient point of the stunt \
area. Express everything, the camera included, in this frame. Geoms use MuJoCo conventions: pos is the centre; box \
size = [half_x, half_y, half_z] (HALF-extents, before yaw); cylinder size = [radius, half_height, 0] (upright); \
sphere size = [radius, 0, 0]; yaw_deg turns about +Z.

LEFT AND RIGHT: unless the brief says "the performer's left/right", directions in the brief are as the camera sees \
them. Convert them with the camera pose: the camera's left is the direction that runs to the left edge of its image. \
A performer running toward the camera faces it, so a turn to the camera's left is a turn to the performer's right \
(toward -Y); a performer running away from the camera turns to the camera's left toward +Y. Use what the footage \
reveals to place the turn: where the camera pans or walks, which side is open floor and which is wall or railing. \
State in stunt.notes how you resolved it and from what evidence.

SCALE PRIORS (fix the metric scale with them and give honest sizes): an adult is about 1.75 m tall (eyes at 1.6 m); \
a door is about 2.0-2.1 m high and 0.9 m wide; railings and balustrades are about 1.0-1.1 m high; ceilings are \
2.5-4 m in rooms and corridors, more in halls; tables are about 0.75 m high; chair seats about 0.45 m; stair risers \
about 0.17 m; floor tiles often 0.6 m.{exif_line}

FILL THE FIELDS IN ORDER:
1. notes: what you see, the layout and its sizes, the scale cues you used, how the camera moves over time, and what \
is uncertain.
2. source.ref_time_sec: {ref_time_text}
3. camera.match: the camera that took the {ref_word}: pos, lookat (a point on its optical axis) and fovy, the \
VERTICAL field of view in degrees. The media is {orient}: {fov_hint} Reason from perspective: converging floor and \
wall lines, the vanishing point, known sizes against their apparent size, and the horizon (the camera's eye height; \
handheld is about 1.4-1.6 m).
4. room: the box enclosing the whole space the stunt and the camera use (walls at its sides, ceiling on top); \
colours sampled from the media; the floor material. Walls, railings or partitions inside it are obstacles.
5. obstacles: every wall, railing, door frame, column, piece of furniture, prop, plant, sign or person within \
{corridor:g} m of the performer's path (floor to 3 m) and every boundary along the path (a wall or railing beside \
the path is a long thin box). Never omit an object because it is small or partly hidden; split furniture into its \
parts when the gaps matter; a prop on a surface sits on it. role: wall, railing, furniture, prop, structure or \
person. hazard: true for everything within reach of the path; false only for things well away from it. Unique \
snake_case ids, realistic rgb from the media, a material. At most 60.
6. set_dressing: optional scenery well away from the path that makes the render look like the place, in the given \
shapes, only for things the media shows; leave it empty otherwise.
7. crash_mats: only when the stunt comes down from a jump or a fall: flat on the floor (pos z = half thickness), \
big enough to land on (half-size about [0.9-1.2, 1.0-1.5, 0.1-0.2]), clear of every obstacle. None otherwise.
8. stunt: the choreography of the brief at this location, as ordered phases.
   - notes first: where the performer starts, where they go and why, and how left/right were resolved.
   - summary: one sentence.
   - start: pos [x, y] on the floor, heading_deg 0 (the frame's +X is the initial facing); z only when starting on \
top of an obstacle.
   - phases, each one of: hold {{duration, pose}}; move {{path (>= 2 [x, y] points, the first is where the \
performer is), speed, accel_time, decel_time (0 = arrive at speed), turn_radius (corner rounding), lean_scale (x the \
physical bank, usually 1)}}; jump {{vz, optional vx, lean_deg, rotations_deg {{roll, pitch, yaw}} completed over the \
flight, pose}}; fall {{push [vx, vy, vz], pose}}; air {{poses: [{{at 0..1 of the flight, pose, tuck 0..1}}]}}; land \
{{pose, compliance 0..1, duration}}. Poses: {poses}. hold and move come only before any jump or fall; air and land \
only after one; after a jump or fall the body is free physics to the end. Give every phase a unique name.
   - Physics: walk 1.2-1.8 m/s, jog 2.5-3.5, run 4-6. A turn at speed v needs turn_radius >= v^2 / (0.8 x 9.81) or the \
feet slip (4 m/s: 2.0 m; 5 m/s: 3.2 m); slow down for a tight corner with a separate move. Flight time = 2 vz / g, \
apex rise = vz^2 / (2 g). The motion should fill about {motion:g} s of the {duration:g} s video.
   - tunables: 2 to 8 numbers the loop may change, the ones that decide success (speeds, turn radii, the path points \
where a turn starts or the stunt ends, jump velocities, rotations, landing compliance). path addresses a number in \
stunt, e.g. phases[1].speed, phases[1].path[2][1], phases[3].vz. min and max are physically sensible and contain the \
current value. Unique snake_case names with a unit and a doc.
   - target: {{"kind": "mat", "id": crash mat id}} when the stunt lands on a mat; {{"kind": "zone", "pos": [x, y], \
"radius": 0.5-1.0}} when it ends at a place (the end of the last move); {{"kind": "none"}} otherwise.
   - default_camera: match_follow (the director's camera position, panning to follow; best for runs and when the \
footage pans), match (fixed director's view), hero (3/4 tracking) or side."""


def scene_prompt(brief, duration, kind, info):
    w, h = info.get("width"), info.get("height")
    portrait = bool(w and h and h > w)
    if kind == "image":
        media_word, ref_word, ref_time_text = "photo", "photo", "0 (a photo)."
        media_lines = f"MEDIA: one photo of the location ({w}x{h} px)."
        if info.get("ref_image"):
            media_lines += " Image 2 is the reference frame for camera matching."
            ref_word = "reference frame (Image 2)"
    else:
        dur = info.get("video_sec")
        media_word = "video"
        media_lines = ("MEDIA: a video of the location" + (f", {dur:.2f} s long" if dur else "")
                       + (f", {w}x{h} px" if w and h else "")
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
            ref_time_text = ("a time in a still part of the clip (before or between camera moves) that shows the "
                             "stunt area clearly: the reference frame for camera matching; camera.match is for that "
                             "frame.")
    orient = (f"portrait ({w}x{h}, a tall frame)" if portrait else f"landscape ({w}x{h})") if w and h else "of unknown size"
    fov_hint = ("for a phone's main camera the long side of the frame spans about 65-75 degrees and the short side "
                "about 40-48 degrees (an ultra-wide lens about 100-110 and 75); fovy is the angle across the "
                + ("long, vertical side." if portrait else "short, vertical side.") if w and h else
                "fovy is the angle across the vertical side of the frame.")
    fovy = info.get("fovy_exif")
    exif_line = (f"\nThe reference image's EXIF focal length gives a vertical field of view of {fovy:.1f} degrees: use "
                 f"it for camera.match.fovy." if fovy else "")
    return SCENE_PROMPT.format(media_word=media_word, brief=brief, duration=duration, media_lines=media_lines,
                               exif_line=exif_line, corridor=CORRIDOR_HALF_WIDTH, ref_word=ref_word,
                               ref_time_text=ref_time_text, orient=orient, fov_hint=fov_hint,
                               poses=", ".join(POSES), motion=max(duration - 1.0, 1.0))


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
                          "rgb": _rgb(c.get("rgb"), (0.6, 0.6, 0.6))},
    "chairs": lambda c: {"x": _need(c, "x"), "y": _need(c, "y"), "yaw_deg": _f(c.get("yaw_deg")),
                         "rgb": _rgb(c.get("rgb"), (0.3, 0.3, 0.3))},
    "people": lambda p: {"x": _need(p, "x"), "y": _need(p, "y"), "yaw_deg": _f(p.get("yaw_deg")),
                         "posture": _enum(p.get("posture"), ("seated", "standing"), "standing"),
                         "seat_z": max(0.0, _f(p.get("seat_z"), 0.9)), "rgb": _rgb(p.get("rgb"), (0.3, 0.3, 0.3))},
    "speakers": lambda s: {"x": _need(s, "x"), "y": _need(s, "y"), "z_bottom": max(0.0, _f(s.get("z_bottom"), 1.0)),
                           "width": _pos(s, "width", 0.35), "height": _pos(s, "height", 0.6),
                           "stand_height": max(0.0, _f(s.get("stand_height"), _f(s.get("z_bottom"), 1.0)))},
    "ceiling_fixtures": lambda f: {"kind": _enum(f.get("kind"), ("pipe", "beam"), "pipe"),
                                   "axis": _enum(f.get("axis"), ("x", "y"), "x"),
                                   "pos": [_need({"v": v}, "v") for v in (list(f.get("pos") or []) + [None] * 3)[:3]],
                                   "length": _pos(f, "length", 4.0), "radius": _pos(f, "radius", 0.06),
                                   "rgb": _rgb(f.get("rgb"), (0.6, 0.6, 0.6))},
    "screens": lambda s: {"x": _need(s, "x"), "y": _need(s, "y"), "width": _pos(s, "width", 2.0),
                          "height": _pos(s, "height", 1.2), "bottom_z": max(0.0, _f(s.get("bottom_z"), 1.0))},
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
    hz = o.get("hazard", True)
    return {"id": oid, "kind": kind, "pos": [_r(v) for v in pos], "size": [_r(v, 4) for v in size],
            "yaw_deg": _r(((_f(o.get("yaw_deg")) + 180.0) % 360.0) - 180.0, 1), "rgb": _rgb(o.get("rgb")),
            "material": _enum(o.get("material"), MATERIALS, "plastic"), "role": role,
            "hazard": hz.strip().lower() == "true" if isinstance(hz, str) else bool(hz)}


def _pose(v, default, W, where):
    """A pose name from POSES or an explicit H1Rig dict; anything else becomes `default`."""
    if isinstance(v, dict):
        d = {k: _r(_f(x), 1) for k, x in v.items() if k in RIG_KEYS and _f(x, None) is not None}
        if d:
            return d
    elif isinstance(v, str) and v.strip().lower() in POSES:
        return v.strip().lower()
    if v not in (None, ""):
        W.append(f"{where}: pose {v!r} is not one of {POSES}; using {default}")
    return default


def _pos_num(v, default, lo, hi):
    x = _f(v, default)
    return _r(_clip(x, lo, hi))


def normalize_phases(raw, start, W):
    """Contract phases: known kinds, filled fields, unique names, legal order, move paths starting where the
    performer is. Returns (phases, index_map) where index_map maps Argon's phase index to the kept one (never
    empty: a short hold when nothing usable came back)."""
    out, names, cur, released, index_map = [], set(), list(start), False, {}
    for i, p in enumerate(raw or []):
        if not isinstance(p, dict):
            W.append(f"phase {i}: not an object, dropped")
            continue
        k = _enum(p.get("kind"), PHASE_KINDS, "")
        if not k:
            W.append(f"phase {i}: kind {p.get('kind')!r} is not one of {PHASE_KINDS}; dropped")
            continue
        if k in ("hold", "move") and released:
            W.append(f"phase {i} ({k}) comes after a jump or fall, where the body is free; dropped")
            continue
        if k in ("jump", "fall") and released:
            W.append(f"phase {i} ({k}): the performer is already free after an earlier jump or fall; dropped")
            continue
        if k in ("air", "land") and not released:
            W.append(f"phase {i} ({k}) needs an earlier jump or fall; dropped")
            continue
        name = _slug(p.get("name"), f"{k}_{len(out) + 1}")
        base, n = name, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.add(name)
        q = {"kind": k, "name": name}
        where = f"phase {name}"
        if k == "hold":
            q.update(duration=_pos_num(p.get("duration"), 0.5, 0.0, 10.0), pose=_pose(p.get("pose"), "stand", W, where))
        elif k == "move":
            pts = [_vec(v, (0.0, 0.0)) for v in p.get("path") or [] if isinstance(v, (list, tuple)) and len(v) >= 2]
            if pts and math.dist(pts[0], cur) > 0.05:
                if math.dist(pts[0], cur) > 0.5:
                    W.append(f"{where}: path starts at ({pts[0][0]:.2f}, {pts[0][1]:.2f}), not where the performer is "
                             f"({cur[0]:.2f}, {cur[1]:.2f}); the current position was put first")
                    pts = [list(cur)] + pts
                else:
                    pts[0] = list(cur)
            elif not pts:
                pts = [list(cur)]
            if len(pts) < 2:
                W.append(f"{where}: a move needs >= 2 path points; dropped")
                names.discard(name)
                continue
            q.update(path=[[_r(x), _r(y)] for x, y in pts], speed=_pos_num(p.get("speed"), 3.0, 0.2, 10.0),
                     accel_time=_pos_num(p.get("accel_time"), 0.8, 0.0, 5.0),
                     decel_time=_pos_num(p.get("decel_time"), 0.0, 0.0, 5.0),
                     turn_radius=_pos_num(p.get("turn_radius"), 1.0, 0.0, 20.0),
                     lean_scale=_pos_num(p.get("lean_scale"), 1.0, 0.0, 2.0))
            if p.get("gait") in GAITS:
                q["gait"] = p["gait"]
            cur = list(q["path"][-1])
        elif k == "jump":
            rot = p.get("rotations_deg") if isinstance(p.get("rotations_deg"), dict) else {}
            q.update(vz=_pos_num(p.get("vz"), 3.0, 0.0, 8.0), lean_deg=_pos_num(p.get("lean_deg"), 0.0, -60.0, 90.0),
                     rotations_deg={a: _r(_clip(_f(rot.get(a)), -1440.0, 1440.0), 1) for a in ("roll", "pitch", "yaw")},
                     pose=_pose(p.get("pose"), "takeoff", W, where))
            if p.get("vx") is not None and _f(p.get("vx"), None) is not None:
                q["vx"] = _pos_num(p.get("vx"), 0.0, -3.0, 10.0)
            released = True
        elif k == "fall":
            q.update(push=[_r(v) for v in _vec(p.get("push"), (0.0, 0.0, 0.0))], pose=_pose(p.get("pose"), "brace", W, where))
            released = True
        elif k == "air":
            poses = []
            for a in p.get("poses") or []:
                if isinstance(a, dict):
                    poses.append({"at": _r(_clip(_f(a.get("at")), 0.0, 1.0), 2),
                                  "pose": _pose(a.get("pose"), "tuck", W, where),
                                  "tuck": _r(_clip(_f(a.get("tuck")), 0.0, 1.0), 2)})
            q["poses"] = sorted(poses, key=lambda a: a["at"]) or [{"at": 0.0, "pose": "layout", "tuck": 0.0}]
        elif k == "land":
            q.update(pose=_pose(p.get("pose"), "land_prep", W, where),
                     compliance=_pos_num(p.get("compliance"), 0.6, 0.0, 1.0),
                     duration=_pos_num(p.get("duration"), 1.0, 0.2, 5.0))
        index_map[i] = len(out)
        out.append(q)
    if not out:
        W.append("no usable phases; the stunt is a 1 s hold")
        out = [{"kind": "hold", "name": "hold", "duration": 1.0, "pose": "stand"}]
    return out, index_map


def auto_tunables(stunt):
    """Sensible tunables when Argon gave none that resolve: move speeds and turn radii, jump vz, land compliance."""
    out = []
    for i, p in enumerate(stunt["phases"]):
        n = p["name"]
        if p["kind"] == "move":
            out.append({"name": f"{n}_speed", "path": f"phases[{i}].speed", "min": 0.5, "max": 7.0, "unit": "m/s",
                        "doc": f"cruise speed of {n}"})
            if len(p["path"]) > 2:
                out.append({"name": f"{n}_turn_radius", "path": f"phases[{i}].turn_radius", "min": 0.3, "max": 6.0,
                            "unit": "m", "doc": f"corner rounding of {n}"})
        elif p["kind"] == "jump":
            out.append({"name": f"{n}_vz", "path": f"phases[{i}].vz", "min": 0.5, "max": 6.5, "unit": "m/s",
                        "doc": "vertical take-off velocity"})
        elif p["kind"] == "land":
            out.append({"name": f"{n}_compliance", "path": f"phases[{i}].compliance", "min": 0.0, "max": 1.0,
                        "unit": "0-1", "doc": "leg softness on landing"})
    return out[:8]


def normalize_tunables(raw, stunt, W, index_map=None):
    """Tunables whose path resolves to a number in the stunt, unique names, min < max around the current value.
    `index_map` renumbers phases[i] after normalize_phases dropped some."""
    out, names = [], set(GENERIC_PARAMS) | set(META_KEYS)
    for i, t in enumerate(raw or []):
        if not isinstance(t, dict):
            continue
        steps = path_steps(t.get("path"))
        if steps and index_map is not None and len(steps) > 1 and steps[0] == "phases" and isinstance(steps[1], int):
            steps = steps[:1] + [index_map.get(steps[1], -10 ** 6)] + steps[2:]
        try:
            cur = resolve(stunt, path_text(steps)) if steps else None
        except KeyError:
            cur = None
        if steps is None or isinstance(cur, bool) or not isinstance(cur, (int, float)):
            W.append(f"tunable {t.get('name')!r}: path {t.get('path')!r} does not lead to a number in stunt; dropped")
            continue
        name = _slug(t.get("name"), f"tunable_{i + 1}")
        if name in names:
            W.append(f"tunable {name}: duplicate name; dropped")
            continue
        path = path_text(steps)
        if any(o["path"] == path for o in out):
            W.append(f"tunable {name}: {path} is already tunable; dropped")
            continue
        lo, hi = _f(t.get("min"), cur), _f(t.get("max"), cur)
        if lo > hi:
            lo, hi = hi, lo
        if not lo <= cur <= hi:
            W.append(f"tunable {name}: range {lo:g}..{hi:g} did not contain the current value {cur:g}; widened")
            lo, hi = min(lo, cur), max(hi, cur)
        if hi - lo < 1e-6:
            span = max(abs(cur) * 0.3, 0.2)
            lo, hi = cur - span, cur + span
            W.append(f"tunable {name}: empty range; using {lo:g}..{hi:g}")
        names.add(name)
        out.append({"name": name, "path": path, "min": _r(lo, 4), "max": _r(hi, 4), "unit": str(t.get("unit") or ""),
                    "doc": str(t.get("doc") or "").strip()})
    if not out:
        out = auto_tunables(stunt)
        W.append(f"no usable tunables; using {[t['name'] for t in out]}")
    return out[:10]


def normalize_stunt(st_in, mats, W):
    st_in = st_in if isinstance(st_in, dict) else {}
    s_in = st_in.get("start") if isinstance(st_in.get("start"), dict) else {}
    start = {"pos": [_r(v) for v in _vec(s_in.get("pos"), (0.0, 0.0))], "heading_deg": _r(_f(s_in.get("heading_deg")), 1)}
    if abs(start["heading_deg"]) > 1:
        W.append(f"stunt.start.heading_deg {start['heading_deg']:g}: the frame's +X is the initial facing; kept as given")
    if _f(s_in.get("z")) > 0.02:
        start["z"] = _r(_f(s_in.get("z")))
    phases, index_map = normalize_phases(st_in.get("phases"), start["pos"], W)
    stunt = {"summary": str(st_in.get("summary") or "").strip(), "start": start, "phases": phases}
    released = any(p["kind"] in ("jump", "fall") for p in phases)
    # target
    t_in = st_in.get("target") if isinstance(st_in.get("target"), dict) else {}
    kind = _enum(t_in.get("kind"), TARGET_KINDS, "")
    end = path_points(stunt)[-1]
    if kind == "mat":
        ids = [m["id"] for m in mats]
        mid = _slug(t_in.get("id"), "")
        if mid not in ids and ids:
            pick = min(mats, key=lambda m: math.dist(m["pos"][:2], end))["id"]
            W.append(f"target mat {t_in.get('id')!r} does not exist; using {pick}, the mat nearest the landing")
            mid = pick
        target = {"kind": "mat", "id": mid} if ids else None
        if not ids:
            W.append("target is a crash mat but the plan has none; the target became a zone at the end point")
            kind = "zone"
    if kind == "zone":
        pos = _vec(t_in.get("pos"), end) if t_in.get("pos") is not None else list(end)
        target = {"kind": "zone", "pos": [_r(pos[0]), _r(pos[1])], "radius": _r(_clip(_f(t_in.get("radius"), 0.65), 0.2, 3.0))}
        if math.dist(pos, end) > 1.5:
            W.append(f"target zone ({pos[0]:.2f}, {pos[1]:.2f}) is {math.dist(pos, end):.1f} m from where the choreography "
                     f"ends ({end[0]:.2f}, {end[1]:.2f})")
    elif kind not in ("mat",):
        if kind != "none":
            W.append(f"target kind {t_in.get('kind')!r} unknown; "
                     + ("using the nearest mat" if released and mats else "using none"))
        target = ({"kind": "mat", "id": min(mats, key=lambda m: math.dist(m["pos"][:2], end))["id"]}
                  if released and mats and kind != "none" else {"kind": "none"})
    stunt["target"] = target
    stunt["tunables"] = normalize_tunables(st_in.get("tunables"), stunt, W, index_map)
    cam = st_in.get("default_camera")
    stunt["default_camera"] = cam if cam in CAMERAS else "match_follow"
    if cam not in CAMERAS:
        W.append(f"default_camera {cam!r} unknown; using match_follow")
    stunt["notes"] = str(st_in.get("notes") or "").strip()
    return stunt


def normalize_plan(ans, *, brief, duration, media, kind, ref_frame=None, ref_time=None, video_sec=None,
                   width=None, height=None, fovy_exif=None, mock=False):
    """Validate and repair Argon's scene answer into a contract plan.json (version 2). Returns (plan, warnings).
    Every default here is generic: nothing from an earlier plan or a fixture is filled in."""
    W = []
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

    # crash mats: on the floor
    mats, seen = [], set()
    for i, m in enumerate(ans.get("crash_mats") or []):
        if not isinstance(m, dict):
            continue
        size = [abs(v) for v in _vec(m.get("size"), (1.0, 1.2, 0.15))]
        if min(size) < 0.02:
            W.append(f"crash mat {m.get('id', i)}: size {size} too small; using at least [0.5, 0.5, 0.05]")
            size = [max(size[0], 0.5), max(size[1], 0.5), max(size[2], 0.05)]
        pos = _vec(m.get("pos"), (0.0, 0.0, size[2]))
        if abs(pos[2] - size[2]) > 0.02:
            W.append(f"crash mat {m.get('id', i)}: not resting on the floor (z {pos[2]:.2f}); set z = {size[2]:.2f}")
            pos[2] = size[2]
        mid = _slug(m.get("id"), f"mat{i + 1}")
        while mid in seen:
            mid += "_b"
        seen.add(mid)
        mats.append({"id": mid, "pos": [_r(v) for v in pos], "size": [_r(v) for v in size],
                     "rgb": _rgb(m.get("rgb"), (0.15, 0.35, 0.85))})

    stunt = normalize_stunt(ans.get("stunt"), mats, W)
    released = any(p["kind"] in ("jump", "fall") for p in stunt["phases"])
    if mats and not released and stunt["target"]["kind"] != "mat":
        W.append(f"{len(mats)} crash mat(s) dropped: the stunt never leaves the ground")
        mats = []
    for m in mats:
        for o in obstacles:
            if _overlap(aabb(dict(m, kind="box")), aabb(o)) and aabb(o)[4] < 2 * m["size"][2]:
                W.append(f"crash mat {m['id']} overlaps {o['id']}")

    # hazards: everything within reach of the path
    pts = path_points(stunt)
    for o in obstacles:
        if not o["hazard"] and near_path(o, pts):
            o["hazard"] = True
            W.append(f"{o['id']} lies within {CORRIDOR_HALF_WIDTH:g} m of the path; marked hazard")

    # camera
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mid = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 1.0]
    cam_in = (ans.get("camera") or {}).get("match") or {}
    cpos = _vec(cam_in.get("pos"), (max(xs) + 3.0, mid[1], 1.5))
    look = _vec(cam_in.get("lookat"), mid)
    fovy = _clip(_f(cam_in.get("fovy"), 60.0 if (height or 0) > (width or 0) else 45.0), 10.0, 120.0)
    if fovy_exif:
        fovy = float(fovy_exif)
    if math.dist(cpos, look) < 0.1:
        look = mid
        W.append("camera.match lookat equals its position; aimed at the middle of the path")
    fwd, to_m = [a - b for a, b in zip(look, cpos)], [a - b for a, b in zip(mid, cpos)]
    cosang = sum(a * b for a, b in zip(fwd, to_m)) / max(math.hypot(*fwd) * math.hypot(*to_m), 1e-9)
    if math.degrees(math.acos(_clip(cosang, -1.0, 1.0))) > fovy * 0.75:
        W.append("camera.match does not seem to see the middle of the path; check preview.png")
    camera = {"match": {"pos": [_r(v) for v in cpos], "lookat": [_r(v) for v in look], "fovy": _r(fovy, 2)}}

    # room: encloses the path, every obstacle, the mats and the camera
    rin = ans.get("room") or {}
    room = {"x_min": _f(rin.get("x_min"), min(xs) - 2), "x_max": _f(rin.get("x_max"), max(xs) + 2),
            "y_min": _f(rin.get("y_min"), min(ys) - 2), "y_max": _f(rin.get("y_max"), max(ys) + 2),
            "ceiling_z": _clip(_f(rin.get("ceiling_z"), 3.0), 2.2, 30.0)}
    for lo, hi in (("x_min", "x_max"), ("y_min", "y_max")):
        if room[lo] > room[hi]:
            room[lo], room[hi] = room[hi], room[lo]
            W.append(f"room {lo}/{hi} were swapped")
    boxes = [aabb(o) for o in obstacles if max(abs(v) for v in o["pos"][:2]) < 60] + \
            [aabb(dict(m, kind="box")) for m in mats]
    need = [min([min(xs) - 0.8, cpos[0] - 0.3] + [b[0] for b in boxes]),
            max([max(xs) + 0.8, cpos[0] + 0.3] + [b[1] for b in boxes]),
            min([min(ys) - 0.8, cpos[1] - 0.3] + [b[2] for b in boxes]),
            max([max(ys) + 0.8, cpos[1] + 0.3] + [b[3] for b in boxes])]
    for key, v, grow in (("x_min", need[0], min), ("x_max", need[1], max), ("y_min", need[2], min), ("y_max", need[3], max)):
        new = grow(room[key], v)
        if abs(new - room[key]) > 0.5:
            W.append(f"room {key} {room[key]:.2f} -> {new:.2f} to enclose the path, obstacles and camera")
        room[key] = new
    top = max([b[5] for b in boxes] + [cpos[2], 2.2])
    if room["ceiling_z"] < top + 0.2:
        W.append(f"ceiling_z {room['ceiling_z']:.2f} -> {top + 0.2:.2f} (above the tallest object and the camera)")
        room["ceiling_z"] = top + 0.2
    room = {k: _r(v, 2) for k, v in room.items()}
    room.update(floor_rgb=_rgb(rin.get("floor_rgb"), (0.55, 0.55, 0.55)), wall_rgb=_rgb(rin.get("wall_rgb"), (0.8, 0.8, 0.8)),
                ceiling_rgb=_rgb(rin.get("ceiling_rgb"), (0.7, 0.7, 0.7)),
                floor_material=_enum(rin.get("floor_material"), FLOOR_MATERIALS, "concrete"))

    # set dressing (optional, only what the media shows); items outside the room are dropped
    sd_in = ans.get("set_dressing") or {}
    cu = sd_in.get("curtain") if isinstance(sd_in.get("curtain"), dict) else {}
    present = cu.get("present", False)
    curtain = {} if not present or str(present).lower() == "false" else {
        "rgb": _rgb(cu.get("rgb"), (0.8, 0.8, 0.8)), "panel_width": _r(_clip(_f(cu.get("panel_width"), 0.4), 0.2, 1.5)),
        "fold_depth": _r(_clip(_f(cu.get("fold_depth"), 0.1), 0.02, 0.4)),
        "height": _r(_clip(_f(cu.get("height"), room["ceiling_z"] - 0.2), 0.5, room["ceiling_z"]))}
    sd, in_path = {"curtain": curtain}, 0
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
                in_path += int(near_path({"pos": [x, y, 0.5], "size": [0.3, 0.3, 0.5]}, pts, 0.6))
            sd[key].append({k: (_r(v) if isinstance(v, float) else v) for k, v in item.items()})
    if in_path:
        W.append(f"{in_path} set-dressing item(s) stand on the path; they stay and count as hazards")

    rt = 0.0
    if kind == "video":
        rt = ref_time if ref_time is not None else _f((ans.get("source") or {}).get("ref_time_sec"), 0.0)
        rt = _clip(rt, 0.0, max(video_sec - 0.05, 0.0)) if video_sec else max(rt, 0.0)
    source = {"media": media, "kind": kind, "ref_frame": ref_frame or (media if kind == "image" else None),
              "ref_time_sec": _r(rt, 2)}
    if width and height:
        source.update(width=int(width), height=int(height))
    plan = {
        "version": 2, "brief": brief, "duration_sec": float(duration), "source": source,
        "room": room, "obstacles": obstacles, "set_dressing": sd, "crash_mats": mats, "camera": camera,
        "stunt": stunt, "performer": {"model": "h1"},
        "argon": {"model": model_name(), "thinking": thinking_level(), "mock": bool(mock),
                  "notes": str(ans.get("notes") or ""), "warnings": W},
    }
    return plan, W


def media_size(path, kind):
    """(width, height) of the media after rotation, or (None, None)."""
    try:
        if kind == "image":
            from PIL import ImageOps
            im = ImageOps.exif_transpose(_open_image(path))
            return im.width, im.height
        import imageio_ffmpeg
        gen = imageio_ffmpeg.read_frames(path)
        meta = next(gen)
        gen.close()
        w, h = (list(meta.get("size") or (0, 0)) + [0, 0])[:2]     # imageio-ffmpeg reports the rotated size
        return (int(w), int(h)) if w and h else (None, None)
    except Exception:
        return None, None


def cmd_scene(args):
    media = args.media
    kind = media_kind(media)
    if args.ref_frame and not os.path.isfile(args.ref_frame):
        raise ArgonError(f"reference frame not found: {args.ref_frame}")
    log = resolve_log(args)
    width, height = media_size(media, kind)
    info, parts, video_sec = {"width": width, "height": height}, [], None
    if kind == "video":
        video_sec, _ = video_info(media)
        info.update(video_sec=video_sec, fps=4.0 if video_sec and video_sec <= 15 else 2.0 if video_sec and video_sec <= 60
                    else 1.0, ref_time=args.ref_time)
    schema = scene_schema()
    if is_mock():
        ans = mock_answer("scene", log)
    else:
        if kind == "image":
            part, im = image_part(media)
            parts.append(part)
            info["fovy_exif"] = im["fovy_exif"]
            if args.ref_frame and _sha1(args.ref_frame) != _sha1(media):
                rpart, rim = image_part(args.ref_frame)
                parts += [{"text": "Image 2 (reference frame):"}, rpart]
                info.update(ref_image=True, fovy_exif=rim["fovy_exif"] or info.get("fovy_exif"))
        if args.ref_frame and kind == "video":
            rpart, rim = image_part(args.ref_frame)
            info.update(ref_image=True, fovy_exif=rim["fovy_exif"])
        prompt = scene_prompt(args.brief, args.duration, kind, info)
        if kind == "image":
            ans, _ = generate(parts + [{"text": prompt}], schema, "scene", log)
        else:
            extra = [{"text": "Reference still:"}, rpart] if info.get("ref_image") else []
            ans, _ = generate_with_video(media, info["fps"], lambda vp: [vp] + extra + [{"text": prompt}], schema,
                                         "scene", log, files_cache_path(log))
    for e in schema_errors(ans, schema)[:8]:
        say(f"scene answer deviates from the schema: {e}")
    plan, warns = normalize_plan(ans, brief=args.brief, duration=args.duration, media=media, kind=kind,
                                 ref_frame=args.ref_frame, ref_time=args.ref_time, video_sec=video_sec,
                                 width=width, height=height, fovy_exif=info.get("fovy_exif"), mock=is_mock())
    for w in warns:
        say(f"plan fix: {w}")
    write_json(args.out, plan)
    st = plan["stunt"]
    print(json.dumps({"out": os.path.abspath(args.out), "kind": kind, "obstacles": len(plan["obstacles"]),
                      "hazards": sum(o["hazard"] for o in plan["obstacles"]), "crash_mats": len(plan["crash_mats"]),
                      "phases": [f"{p['kind']}:{p['name']}" for p in st["phases"]],
                      "tunables": [t["name"] for t in st["tunables"]], "target": st["target"],
                      "summary": st["summary"], "warnings": len(warns), "mock": is_mock()}))
    return 0


# ----------------------------------------------------------------------------- propose
PROPOSE_PROMPT = """You are the assistant stunt coordinator in FALLGUY, a recursive stunt pre-visualisation loop. \
Each wave you propose {n} candidate takes. A MuJoCo simulator performs every take with a Unitree H1 humanoid \
(1.79 m, 51.4 kg) as the stunt double, following the choreography below with your numbers: hold and move phases are \
scripted (the pelvis follows the path, feet planted by leg IK); from a jump or fall on, everything is free physics. \
The simulator measures Gate 1 for every take; then the best take is rendered and judged on video for Gate 2. \
Converge on a take that passes both.

DIRECTOR'S BRIEF: "{brief}"
{notes_block}
CHOREOGRAPHY (world frame, metres: +X = the performer's initial facing, +Y = their left, +Z up, floor z = 0):
{choreo}

SCENE:
{scene}

PARAMETERS (every candidate sets all of them; ranges are hard limits; the current value is the plan's):
{table}
camera: one of {cams}; default {cam_default}. {cam_text}

PHYSICS TO PLAN WITH (g = 9.81 m/s^2): in a turn the lateral acceleration is v^2 / r and must stay <= \
{lat_g:g} g, so a turn of radius r allows v <= sqrt({lat_g:g} g r) (r 1 m: 2.8 m/s; 2 m: 4.0 m/s; 3 m: 4.9 m/s); a \
wider turn swings toward the outside, so mind the clearance there. Flight time ~ 2 vz / g, apex rise ~ vz^2 / (2 g), \
distance in the air ~ horizontal speed x flight time. Calibrate with the measured numbers of earlier takes.

GATES:
{gates}

HISTORY:
{history}

YOUR TASK FOR WAVE {wave}:
{task}

Write the rationale first (2-4 sentences: what the numbers say and your plan), then the {n} candidates. Each gets a \
label of 2-4 words, a why of 1-2 sentences that cites the numbers it acts on (e.g. "lateral G 0.93 > 0.8 at the \
turn -> turn_radius 1.6 -> 2.3"), a value for every parameter and a camera. No two candidates may be equal."""


def propose_prompt(plan, history, P, wave, n, steer):
    sp = specs_of(P, plan)
    th = thresholds()
    notes = director_notes(history, steer)
    notes_block = ("DIRECTOR NOTES (standing, newest first; they override your own preferences, every candidate must "
                   "respect them):\n" + "\n".join(f"- {x}" for x in notes) + "\n") if notes else ""
    best, best_wave = best_so_far(history)
    if best is None:
        task = (f"No take has run yet. Make {n} diverse, complete attempts at the brief that spread over the parameter "
                "space around the plan's values: e.g. the plan as written, a safer one (slower, wider, softer), a "
                "bolder one (faster, tighter, higher), and one that changes a different number from the others. Keep "
                "every candidate physically plausible (respect the lateral-G limit in turns).")
    else:
        bp, bm = best.get("params") or {}, best.get("metrics") or {}
        explore = ("Keep exactly one exploratory candidate in a clearly different region in case the best is a local "
                   "optimum. " if n > 1 else "")
        last = (history.get("waves") or [{}])[-1].get("critique") or {}
        crit = f"\nThe last critique: {critique_line(last)}" if last else ""
        task = (f"The best take so far is {bp.get('id')} (wave {best_wave}): {metrics_line(bm)}.{crit}\n"
                f"Exploit it: make {n - 1 if n > 1 else 1} refinement(s) of it that fix its exact Gate-1 failures and "
                "the critique's issues, changing one to three parameters each by amounts you justify from the "
                f"measurements (cite them). {explore}Do not repeat a take that already ran.")
    cams = camera_choices(P)
    return PROPOSE_PROMPT.format(
        n=n, brief=plan.get("brief") or history.get("brief", ""), notes_block=notes_block,
        choreo=choreography_text(plan), scene=scene_summary(plan), table=P.describe_for_prompt(plan),
        cams=", ".join(cams), cam_default=default_camera(plan, P),
        cam_text="; ".join(f"{c}: {CAMERA_TEXT.get(c, c)}" for c in cams), lat_g=float(th["lateral_g_max"]),
        gates=gates_text(), history=history_summary(history, list(sp)), wave=wave, task=task)


def wave_number(out, history):
    """The wave being proposed: from the --out directory (waves/waveNN/), else history.wave, else waves + 1."""
    m = re.fullmatch(r"wave0*(\d+)", os.path.basename(os.path.dirname(os.path.abspath(out))))
    hist = history.get("wave") if isinstance(history.get("wave"), int) else len(history.get("waves") or []) + 1
    if m:
        if int(m.group(1)) != hist:
            say(f"note: --out says wave {int(m.group(1))} but history.json says wave {hist}; using {int(m.group(1))}")
        return int(m.group(1))
    return hist


def mock_propose(plan, history, P, n, log):
    """Deterministic proposal over specs(plan): fixture blends for wave 1, small nudges of the best take afterwards.
    A fixture candidate's `blend` b in -1..1 puts every parameter at default + b x (half its range toward that side)."""
    fx = mock_answer("propose", log)
    sp = specs_of(P, plan)
    cam = default_camera(plan, P)
    best, _ = best_so_far(history)
    if best is None:
        cands = []
        for c in fx.get("candidates") or []:
            b = _clip(_f(c.get("blend")), -1.0, 1.0)
            params = {k: s["default"] + b * ((s["max"] - s["default"]) if b > 0 else (s["default"] - s["min"])) * 0.5
                      for k, s in sp.items() if not k.startswith("mat_offset")}
            params["camera"] = c.get("camera") or cam
            cands.append({"label": c.get("label"), "why": c.get("why"), "params": params})
        rationale = fx.get("rationale", "mock proposal")
    else:
        bp = best.get("params") or {}
        tun = [k for k in sp if not k.startswith("mat_offset")] or list(sp)
        cands = []
        for i in range(max(n - 1, 1)):
            k = tun[i % len(tun)]
            s = sp[k]
            d = 0.1 * (s["max"] - s["min"]) * (1 if i % 2 == 0 else -1)
            cands.append({"label": f"refine {k}", "why": f"mock: {bp.get('id')} with {k} {d:+.3g}",
                          "params": {**{q: bp[q] for q in sp if q in bp}, k: _f(bp.get(k), s["default"]) + d,
                                     "camera": bp.get("camera") or cam}})
        fxc = (fx.get("candidates") or [{}])[-1]
        cands.append({"label": "explore", "why": "mock: exploratory take", "params": {
            k: s["min"] + 0.3 * (s["max"] - s["min"]) for k, s in sp.items() if not k.startswith("mat_offset")}})
        cands[-1]["params"]["camera"] = fxc.get("camera") or cam
        rationale = f"mock: exploit {bp.get('id')} with small nudges, keep one exploratory take"
    return {"rationale": rationale, "candidates": cands[:n]}


def finish_proposal(ans, P, wave, n, history, plan):
    """Flatten, clamp with engine.params.clamp(params, plan), set ids w<wave>c<k> and colours; fill to n."""
    W, out = [], []
    sp = specs_of(P, plan)
    cams = camera_choices(P)
    for k, c in enumerate([c for c in ans.get("candidates") or [] if isinstance(c, dict)][:n], 1):
        params = dict(c.get("params") or {})
        for name in list(sp) + ["camera"]:
            if name in c and name not in params:
                params[name] = c[name]
        if params.get("camera") not in cams:
            params["camera"] = default_camera(plan, P)
        cid = f"w{wave}c{k}"
        params.update(id=cid, label=str(c.get("label") or f"take {k}").strip()[:40], color=COLORS[(k - 1) % len(COLORS)],
                      why=str(c.get("why") or "").strip())
        cl = _clamp(P, params, plan)
        for key in ("id", "label", "color", "why", "camera"):
            cl[key] = params[key]
        for name in sp:
            if name in params and abs(_f(params[name], 0.0) - _f(cl.get(name), 0.0)) > 1e-6:
                W.append(f"{cid} {name} {params[name]} -> {cl.get(name)} (clamped)")
        out.append(cl)
    if len(out) < n:
        W.append(f"Argon returned {len(out)} of {n} candidates; filled the rest with small variations")
        best, _ = best_so_far(history)
        base = (best or {}).get("params") or (out[0] if out else _defaults(P, plan))
        rng = random.Random(wave * 1000 + len(out))
        for k in range(len(out) + 1, n + 1):
            p = dict(base)
            for name, spec in sp.items():
                p[name] = _f(p.get(name), spec["default"]) + rng.gauss(0, 0.05) * (spec["max"] - spec["min"])
            cid = f"w{wave}c{k}"
            p.update(id=cid, label=f"variation {k}", color=COLORS[(k - 1) % len(COLORS)],
                     why=f"filler: small random variation of {base.get('id', 'the plan')}",
                     camera=base.get("camera") if base.get("camera") in cams else default_camera(plan, P))
            cl = _clamp(P, p, plan)
            for key in ("id", "label", "color", "why", "camera"):
                cl[key] = p[key]
            out.append(cl)
    seen = set()
    for c in out:
        sig = json.dumps({k: c.get(k) for k in list(sp) + ["camera"]}, sort_keys=True)
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
                          propose_schema(P, args.n, plan), "propose", log)
    proposal = finish_proposal(ans, P, wave, args.n, history, plan)
    for w in proposal["warnings"]:
        say(f"proposal fix: {w}")
    write_json(args.out, proposal)
    print(json.dumps({"out": os.path.abspath(args.out), "wave": wave, "ids": [c["id"] for c in proposal["candidates"]],
                      "labels": [c["label"] for c in proposal["candidates"]], "warnings": len(proposal["warnings"]),
                      "mock": is_mock()}))
    return 0


# ----------------------------------------------------------------------------- critique
CRITIQUE_PROMPT = """You are the Gate 2 critic in FALLGUY, a stunt pre-visualisation loop. The video is the render \
of one simulated take: a Unitree H1 humanoid robot (1.79 m; dark grey body, orange left arm, blue right arm) stands \
in for the stunt double in a MuJoCo rebuild of the location. Watch it from start to end and log what happens over \
time before you judge it.

DIRECTOR'S BRIEF: "{brief}"
{notes_block}PLANNED CHOREOGRAPHY (world frame: +X = the performer's initial facing, +Y = their left):
{choreo}
{rot_line}VIDEO: {video_line}
{playback}

SCENE:
{scene}

CANDIDATE TAKE {cid} "{label}": {params}

MUJOCO MEASUREMENTS, GROUND TRUTH (times are simulation seconds). The simulator measures clearance, contacts and \
G-forces exactly and a video cannot, so do not re-estimate them; treat them as facts:
{metrics}

JUDGE WHAT YOU SEE: whether the planned moves happen in order and in the brief's direction, how believable the body \
motion is, framing and style.

FILL THE FIELDS IN ORDER:
1. events: a dense timestamped log of what is on screen, at least one entry every 0.5 s while the performer moves \
(start, turn, takeoff, apex, landing, stop, settled, other in between). Each note says what the body is doing and \
where it is in the scene.
2. summary: 2-4 sentences: what the take shows and its main problem, if any.
3. phases_seen: one entry per planned phase, in the plan's order: performed or not, when it starts (video seconds; -1 \
if not seen), notes. order_ok: they happen in the planned order.
4. direction: where the performer heads at the end as the viewer sees it, and whether that matches the brief.
5. motion: does it read as a believable human body (cadence, stride length, arm swing, balance, lean into turns)? \
foot_skating: does a planted foot visibly slide on the floor?
6. brief_match: does the take do what the brief asks? List what is missing.
7. reaches_target: does the performer end on the mat or in the target area?{rot_field}
8. unnatural_motion: each problem with its time and severity. high = breaks believability (floating or hovering, \
limbs passing through objects or the floor, popping or teleporting, feet sliding a long way while planted, moving \
the wrong way); medium = clearly robotic or stiff; low = minor. Slow motion listed above is an intentional edit.
9. framing: in_frame_pct = % of the video from the start of motion to 1 s after it ends in which the whole performer \
is inside the frame; subject_size: small (under 15 % of the frame height, hard to read), medium (15-40 %), large \
(over 40 %).
10. style_score 0-100: how well it reads as the brief, athletic and cinematic. 90+ ready to show; 70-89 good with \
minor issues; 50-69 recognisable but flawed; under 50 fails the brief.
11. gate2_pass, by this fixed rubric: true only if ALL hold: the brief is satisfied; every planned phase is performed \
in order; the direction matches the brief; the performer reaches the target; no high-severity unnatural motion; \
in_frame_pct >= {framing_min:g} and subject_size medium or large; style_score >= {style_min:g}.
12. proposed_changes: up to 5 parameter changes for the next wave that fix what you saw. Name only these \
parameters: {names}. Give a value inside the range and cite the time and issue in the reason. (The camera is not \
changed here; put framing advice in framing.notes.) The parameters:
{table}"""


def playback_info(events, video_sec, video_fps):
    """Lines describing slow-motion segments and the video times of the simulator's events, via engine.render.retime."""
    if not events or not video_sec or not video_fps:
        return "Playback speed unknown: parts of the video may be in slow motion around a flight."
    try:
        import numpy as np
        from engine.render import retime
        ts = np.asarray(retime(events, float(video_sec), int(round(video_fps))), dtype=float)
    except Exception as e:
        say(f"note: no playback map ({type(e).__name__}: {e})")
        return "Playback speed unknown: parts of the video may be in slow motion around a flight."
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
    for key, name in (("t_takeoff", "take-off"), ("t_apex", "apex"), ("t_landing", "landing"), ("t_finish", "finish")):
        t = events.get(key)
        if isinstance(t, (int, float)) and ts[0] <= t <= ts[-1]:
            ev.append(f"{name} {float(np.interp(t, ts, vt)):.2f} s")
    for ph in events.get("phases") or []:
        t = ph.get("t0") if isinstance(ph, dict) else None
        if isinstance(t, (int, float)) and ts[0] <= t <= ts[-1]:
            ev.append(f"phase {ph.get('name')} starts {float(np.interp(t, ts, vt)):.2f} s")
    if ev:
        lines.append("SIMULATOR EVENTS IN VIDEO TIME: " + ", ".join(ev) + ".")
    return "\n".join(lines)


def critique_prompt(plan, params, metrics, P, th, video_line, playback, notes):
    sp = specs_of(P, plan)
    axis, deg = required_rotation(plan)
    rot_line = (f"REQUIRED ROTATION: about {deg:g} degrees of {axis} while airborne.\n" if axis != "none" else "")
    rot_field = (" rotation_observed_deg: the total rotation about that axis between take-off and landing (360 = "
                 "one full turn); count it from your log." if axis != "none" else "")
    notes_block = ("DIRECTOR NOTES (standing, newest first): " + " | ".join(notes) + "\n") if notes else ""
    return CRITIQUE_PROMPT.format(brief=plan.get("brief", ""), notes_block=notes_block, choreo=choreography_text(plan),
                                  rot_line=rot_line, rot_field=rot_field, video_line=video_line, playback=playback,
                                  scene=scene_summary(plan), cid=params.get("id", "?"), label=params.get("label", ""),
                                  params=params_line(params, list(sp)), metrics=metrics_line(metrics),
                                  framing_min=float(th["framing_min_pct"]), style_min=float(th["style_min"]),
                                  names=", ".join(sp), table=P.describe_for_prompt(plan))


def finish_critique(ans, P, plan, params, metrics, th, video=None, fps=None, video_sec=None):
    """Contract critique.json from Argon's answer; gate2_pass is recomputed from the fixed rubric, with the MuJoCo
    metrics as ground truth for reaching the target."""
    W = []
    sp = specs_of(P, plan)
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
    planned = phase_names(plan)
    seen = {}
    for ph in ans.get("phases_seen") or []:
        if isinstance(ph, dict) and ph.get("phase") in planned:
            seen[ph["phase"]] = {"phase": ph["phase"], "performed": ph.get("performed") is True,
                                 "start_sec": _r(_f(ph.get("start_sec"), -1.0), 2), "notes": str(ph.get("notes") or "")}
    phases_seen = [seen.get(n, {"phase": n, "performed": False, "start_sec": -1.0, "notes": "not reported"})
                   for n in planned]
    unreported = [n for n in planned if n not in seen]
    if unreported and seen:
        W.append(f"the critique did not report phases {unreported}")
    # a hold is easy to miss on video: only the moving phases must be seen
    moving = {p.get("name") for p in ((plan.get("stunt") or {}).get("phases") or []) if p.get("kind") != "hold"}
    phases_ok = all(p["performed"] for p in phases_seen if p["phase"] in moving) if seen else brief_match["matches"]
    d_in = ans.get("direction") or {}
    direction = {"screen_side": _enum(d_in.get("screen_side"), ("left", "right", "toward", "away", "none"), "none"),
                 "matches_brief": d_in.get("matches_brief") is not False, "notes": str(d_in.get("notes") or "")}
    mo = ans.get("motion") or {}
    motion = {"plausible": mo.get("plausible") is not False, "foot_skating": mo.get("foot_skating") is True,
              "notes": str(mo.get("notes") or "")}
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
        if name not in sp:
            W.append(f"dropped a proposed change to unknown parameter {name!r}")
            continue
        v = _f(c.get("value"), None)
        if v is None:
            W.append(f"dropped the proposed change to {name}: no numeric value")
            continue
        changes.append({"param": name, "change": _enum(c.get("change"), CHANGES, "set"),
                        "value": _r(_clip(v, sp[name]["min"], sp[name]["max"]), 4),
                        "reason": str(c.get("reason") or "").strip()})
    landing = ((metrics or {}).get("gate1") or {}).get("landing") or {}
    sim_in = landing.get("inside_mat") if isinstance(landing.get("inside_mat"), bool) else None
    seen_in = ans.get("reaches_target") is not False
    if sim_in is not None and sim_in != seen_in:
        W.append(f"Argon saw reaches_target={seen_in} but MuJoCo measured inside={sim_in}; the rubric uses MuJoCo")
    axis, req_deg = required_rotation(plan)
    seen_deg = abs(_f(ans.get("rotation_observed_deg"), 0.0))
    rubric = {
        "brief_satisfied": brief_match["matches"],
        "phases_in_order": bool(phases_ok) and ans.get("order_ok") is not False,
        "direction_ok": direction["matches_brief"],
        "reaches_target": sim_in if sim_in is not None else seen_in,
        "no_high_severity_motion": not any(u["severity"] == "high" for u in um),
        "framing_ok": framing["in_frame_pct"] >= float(th["framing_min_pct"]) and framing["subject_size"] != "small",
        "style_ok": style >= float(th["style_min"]),
    }
    if axis != "none":
        rubric["rotation_ok"] = abs(seen_deg - req_deg) <= 45.0
    missing = [p["phase"] for p in phases_seen if not p["performed"] and p["phase"] in moving]
    why = {"brief_satisfied": "brief not satisfied" + (f" (missing: {', '.join(brief_match['missing'])})" if brief_match["missing"] else ""),
           "phases_in_order": "planned phases not all performed in order" + (f" (not seen: {', '.join(missing)})" if missing else ""),
           "direction_ok": f"direction does not match the brief ({direction['screen_side']}: {direction['notes']})".strip(),
           "reaches_target": "does not reach the target",
           "no_high_severity_motion": "high-severity unnatural motion: " + "; ".join(u["issue"] for u in um if u["severity"] == "high"),
           "framing_ok": f"framing {framing['in_frame_pct']:g} % / subject {framing['subject_size']} "
                         f"(need >= {float(th['framing_min_pct']):g} % and medium or large)",
           "style_ok": f"style {style} < {float(th['style_min']):g}",
           "rotation_ok": f"rotation {seen_deg:g} deg vs required {req_deg:g} deg ({axis})"}
    return {
        "summary": str(ans.get("summary") or "").strip(),
        "events": events,
        "brief_match": brief_match,
        "phases_seen": phases_seen,
        "order_ok": ans.get("order_ok") is not False,
        "direction": direction,
        "motion": motion,
        "unnatural_motion": um,
        "framing": framing,
        "style_score": style,
        "gate2_pass": all(rubric.values()),
        "proposed_changes": changes,
        "candidate_id": params.get("id"),
        "reaches_target_seen": seen_in,
        **({"rotation_axis": axis, "rotation_required_deg": req_deg, "rotation_observed_deg": _r(seen_deg, 1)}
           if axis != "none" else {}),
        "rubric": rubric,
        "gate2_reasons": [why[k] for k, ok in rubric.items() if not ok],
        "argon_gate2_pass": ans.get("gate2_pass") is True,
        "video": {"path": video, "duration_sec": _r(video_sec, 2) if video_sec else None, "sampled_fps": fps},
        "warnings": W,
    }


def mock_critique(plan, params, metrics, log):
    """Canned critique whose verdict follows the MuJoCo metrics, so mock loops can converge."""
    ans = copy.deepcopy(mock_answer("critique", log))
    g1 = (metrics or {}).get("gate1") or {}
    passed = g1.get("passed") is True
    ans["phases_seen"] = [{"phase": n, "performed": True, "start_sec": -1, "notes": "mock"} for n in phase_names(plan)]
    ans["order_ok"] = True
    ans["direction"] = {"screen_side": "none", "matches_brief": True, "notes": "mock"}
    ans["reaches_target"] = (g1.get("landing") or {}).get("inside_mat") is not False
    axis, req = required_rotation(plan)
    if axis != "none":
        ans["rotation_observed_deg"] = abs(_f((g1.get("rotation_deg") or {}).get(axis), req))
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
        cam = params.get("camera") or default_camera(plan, P)
        video_line += f" Rendered from the {cam} camera ({CAMERA_TEXT.get(cam, 'a scene camera')})."
        hist_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "history.json")
        hist = read_json(hist_path, "history") if os.path.isfile(hist_path) else {}
        notes = director_notes(hist if isinstance(hist, dict) else {}, args.steer)
        prompt = critique_prompt(plan, params, metrics, P, th, video_line, playback_info(events, video_sec, video_fps),
                                 notes)
        ans, _ = generate_with_video(args.video, args.fps, lambda vp: [vp, {"text": prompt}],
                                     critique_schema(P, plan, video_sec), "critique", log, files_cache_path(log))
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
    s = sub.add_parser("scene", help="build plan.json (scene + choreography) from site media + brief")
    s.add_argument("--media", required=True, help="site photo or video")
    s.add_argument("--brief", required=True)
    s.add_argument("--duration", required=True, type=float, help="final video length, seconds")
    s.add_argument("--out", required=True, help="plan.json to write")
    s.add_argument("--ref-frame", help="still used for camera matching (recorded as source.ref_frame)")
    s.add_argument("--ref-time", type=float, help="video only: time of --ref-frame in the video, seconds")
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
