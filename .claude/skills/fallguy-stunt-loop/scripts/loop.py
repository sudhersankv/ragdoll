#!/usr/bin/env python3
"""FALLGUY stunt loop orchestrator: init, wave, finalize, status, gates.

  python loop.py init     --media PATH --brief TEXT --duration S [--runs-dir stunts]
  python loop.py wave     --run-dir DIR [--steer TEXT] [--candidates 4]
  python loop.py finalize --run-dir DIR [--candidate ID]
  python loop.py status   --run-dir DIR
  python loop.py gates

Every command streams short progress lines to stderr and prints exactly one JSON line to stdout
(on failure: {"error": ..., "exit_code": ...}). Run state lives in <run_dir>/run.json; all file
shapes are in ../reference/contracts.md. argon.py and the generated stunt script run as
subprocesses of this Python (sys.executable), with the run directory as working directory.
A wave says next = finalize once Gate 2 passed, checkpoint after every checkpoint_every-th wave
(5, 10, ...) that did not, else wave; there is no hard wave cap. --steer adds a standing director note.

Exit codes: 0 ok, 1 bad input or run state, 2 the generated stunt script failed,
3 an Argon call failed twice.

Environment: FALLGUY_STUNT_TEMPLATE (template copied by init, default scripts/stunt_template.py),
FALLGUY_SIM_TIMEOUT (s per candidate simulation, 180), FALLGUY_RENDER_TIMEOUT (s per render, 900),
FALLGUY_ARGON_CALL_TIMEOUT (s per argon.py run: scene 600, propose and critique 300),
FALLGUY_ARGON_RETRY_DELAY (s before loop.py retries a failed Argon call, 5).
"""
import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import unicodedata

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPTS_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

ARGON_PY = os.path.join(SCRIPTS_DIR, "argon.py")
DEFAULT_TEMPLATE = os.path.join(SCRIPTS_DIR, "stunt_template.py")
CHECKPOINT_EVERY = 5
MAX_PARALLEL = 4
CAMERAS = ("hero", "match", "side")
PALETTE = ("#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6", "#ec4899", "#84cc16")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".3gp", ".mpg", ".mpeg")
PLACEHOLDERS = ("__FALLGUY_SCRIPTS_DIR__", "__FALLGUY_BRIEF__", "__FALLGUY_STUNT_TYPE__", "__FALLGUY_GENERATED_AT__")
ARGON_TIMEOUTS = {"scene": 600.0, "propose": 300.0, "critique": 300.0}
# fit inside 1280x720 (landscape) or 720x1280 (portrait), even sizes, never upscale
PROXY_VF = ("scale=w='if(gte(iw,ih),min(1280,iw),min(720,iw))':h='if(gte(iw,ih),min(720,ih),min(1280,ih))'"
            ":force_original_aspect_ratio=decrease:force_divisible_by=2")

_REAL_STDOUT = sys.stdout


class LoopError(Exception):
    """A failure with a specific message and exit code (1 input/state, 2 stunt script, 3 Argon)."""

    def __init__(self, msg, code=1, **extra):
        super().__init__(msg)
        self.code = code
        self.extra = extra


# ----------------------------------------------------------------------------- small helpers
def log(msg=""):
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()


def emit(obj):
    """The command's single JSON line on the real stdout."""
    _REAL_STDOUT.write(json.dumps(obj, default=_json_default, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def env_float(name, default):
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        raise LoopError(f"{name}={v!r} is not a number") from None


def num(v):
    """float(v), or None for missing / non-numeric values."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_default(o):
    if hasattr(o, "tolist"):          # numpy arrays and scalars
        return o.tolist()
    return str(o)


def read_json(path, what="file"):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise LoopError(f"{what} not found: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise LoopError(f"{what} is not valid JSON: {path} ({e})") from None


def try_read_json(path):
    try:
        return read_json(path)
    except LoopError:
        return None


def write_json(path, obj):
    """Atomic JSON write (temp file + rename), so readers never see half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def last_line(lines):
    for line in reversed(list(lines)):
        if line.strip():
            return line.strip()[:500]
    return ""


def wave_rel(n):
    return f"waves/wave{int(n):02d}"


# ----------------------------------------------------------------------------- secrets
_KEY = None
_KEY_PATTERNS = (
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "<KEY>"),
    (re.compile(r"([?&]key=)[^&\s\"']+"), r"\1<KEY>"),
    (re.compile(r"(x-goog-api-key[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]+", re.I), r"\1<KEY>"),
)


def _argon_key():
    """The Argon key, read only so it can be scrubbed from text; never printed or written."""
    global _KEY
    if _KEY is None:
        path = os.environ.get("FALLGUY_ARGON_TOKEN_FILE") or os.path.expanduser("~/.argon_token")
        try:
            with open(path, encoding="utf-8") as f:
                _KEY = f.read().strip()
        except (OSError, UnicodeDecodeError):
            _KEY = ""
    return _KEY


def redact(text):
    text = str(text)
    key = _argon_key()
    if len(key) >= 8:
        text = text.replace(key, "<KEY>")
    for pat, rep in _KEY_PATTERNS:
        text = pat.sub(rep, text)
    return text


# ----------------------------------------------------------------------------- subprocesses
_ACTIVE = set()
_ACTIVE_LOCK = threading.Lock()


def _kill(p):
    """Terminate a child's whole process group (the stunt script may run ffmpeg)."""
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _on_signal(signum, frame):
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    raise KeyboardInterrupt(f"signal {signum}")


def run_child(cmd, cwd, timeout, log_path=None, echo=None, heartbeat=None, label="child"):
    """Run cmd in its own process group; stdout+stderr go (redacted) to log_path and, with a prefix
    `echo`, to our stderr. Returns (returncode, last_lines, seconds, timed_out)."""
    t0 = time.time()
    tail = collections.deque(maxlen=80)
    logf = open(log_path, "a", encoding="utf-8") if log_path else None
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    try:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
    except OSError as e:
        if logf:
            logf.close()
        raise LoopError(f"could not start {' '.join(cmd[:2])}: {e}") from None
    with _ACTIVE_LOCK:
        _ACTIVE.add(p)

    def pump():
        for raw in p.stdout:
            line = redact(raw.rstrip("\r\n"))
            tail.append(line)
            try:
                if logf:
                    logf.write(line + "\n")
                    logf.flush()
            except ValueError:             # log closed: a grandchild kept the pipe open past the join timeout
                pass
            if echo is not None and line.strip() and not line.lstrip().startswith("{"):   # JSON results stay in the log
                log(f"{echo}{line[:200]}")

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    timed_out = False
    deadline = t0 + timeout
    try:
        while True:
            left = deadline - time.time()
            if left <= 0:
                timed_out = True
                _kill(p)
                break
            try:
                p.wait(timeout=min(left, heartbeat or left))
                break
            except subprocess.TimeoutExpired:
                if heartbeat and time.time() < deadline:
                    log(f"  {label}: still running ({time.time() - t0:.0f} s)")
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.discard(p)
        th.join(timeout=10)
        if logf:
            logf.close()
    return p.returncode, list(tail), time.time() - t0, timed_out


def call_argon(run_dir, sub, args, out_rel, what):
    """Run `argon.py <sub> ...` in the run dir, retrying once. Returns the parsed --out JSON;
    raises exit code 3 with the redacted error after the second failure."""
    if not os.path.isfile(ARGON_PY):
        raise LoopError(f"argon.py is missing next to loop.py ({ARGON_PY}); the skill install is incomplete", code=3)
    cmd = [sys.executable, ARGON_PY, sub, *args, "--log", os.path.join("logs", "argon_calls.jsonl")]
    out_path = os.path.join(run_dir, out_rel)
    timeout = env_float("FALLGUY_ARGON_CALL_TIMEOUT", ARGON_TIMEOUTS.get(sub, 300.0))
    err = ""
    for attempt in (1, 2):
        if os.path.exists(out_path):
            os.remove(out_path)
        log(f"{what}: Argon {sub} (attempt {attempt}/2, high thinking: usually 30-90 s) ...")
        rc, tail, secs, timed_out = run_child(cmd, run_dir, timeout, echo="  | ", heartbeat=30,
                                              log_path=os.path.join(run_dir, "logs", f"argon_{sub}.log"),
                                              label=f"argon {sub}")
        if timed_out:
            err = f"no answer within {timeout:.0f} s"
        elif rc != 0:
            err = last_line(tail) or f"argon.py exited with code {rc}"
        elif not os.path.isfile(out_path):
            err = f"argon.py exited 0 but wrote no {out_rel}"
        else:
            try:
                data = read_json(out_path, f"Argon {sub} output")
                log(f"{what}: Argon {sub} answered in {secs:.0f} s")
                return data
            except LoopError as e:
                err = str(e)
        log(f"{what}: Argon {sub} failed (attempt {attempt}/2): {redact(err)}")
        if attempt == 1:
            time.sleep(env_float("FALLGUY_ARGON_RETRY_DELAY", 5.0))
    raise LoopError(f"Argon {sub} failed twice: {redact(err)} (details: {run_dir}/logs/argon_{sub}.log)", code=3)


# ----------------------------------------------------------------------------- run directory
def slugify(text, max_len=40):
    """Lowercase, hyphenated, <= max_len chars, cut at a word boundary when possible."""
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        cut = s[:max_len]
        if s[max_len] != "-" and "-" in cut:
            cut = cut[:cut.rfind("-")]
        s = cut.strip("-")
    return s or "stunt"


def make_run_dir(runs_dir, slug, ts):
    """Create <runs_dir>/<slug>-<ts>, or <slug>-<ts>_2, _3 ... when taken. Returns (path, tag); the tag
    (ts plus any suffix) also names the run's script, final video and certificate, so they stay unique."""
    os.makedirs(runs_dir, exist_ok=True)
    for n in range(1, 1000):
        tag = ts if n == 1 else f"{ts}_{n}"
        path = os.path.join(runs_dir, f"{slug}-{tag}")
        try:
            os.mkdir(path)
            return path, tag
        except FileExistsError:
            continue
    raise LoopError(f"could not find a free run directory name for {slug}-{ts} in {runs_dir}")


def resolve_run_dir(path):
    run_dir = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(run_dir):
        raise LoopError(f"run dir not found: {run_dir} (use the run_dir printed by `loop.py init`)")
    return run_dir


def load_run(run_dir):
    path = os.path.join(run_dir, "run.json")
    if not os.path.isfile(path):
        raise LoopError(f"{run_dir} has no run.json: `loop.py init` did not finish there; run init again")
    run = read_json(path, "run.json")
    for k, v in (("waves", []), ("director_notes", []), ("checkpoint_every", CHECKPOINT_EVERY), ("best", None),
                 ("final", None)):
        run.setdefault(k, v)
    if not isinstance(run["checkpoint_every"], int) or run["checkpoint_every"] < 1:
        raise LoopError(f"run.json checkpoint_every must be a positive integer, not {run['checkpoint_every']!r}")
    return run


def save_run(run_dir, run):
    run["updated_at"] = now_iso()
    write_json(os.path.join(run_dir, "run.json"), run)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError, OverflowError):
        return True
    return True


def read_lock(run_dir):
    """The live lock holder of a run ({pid, command, since}), or None."""
    info = try_read_json(os.path.join(run_dir, ".loop.lock"))
    if isinstance(info, dict) and info.get("pid") and _pid_alive(info["pid"]):
        return info
    return None


class RunLock:
    """One wave/finalize at a time per run directory (a stale lock from a dead process is taken over)."""

    def __init__(self, run_dir, command):
        self.path = os.path.join(run_dir, ".loop.lock")
        self.command = command

    def __enter__(self):
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = try_read_json(self.path)
                if isinstance(info, dict) and info.get("pid") and _pid_alive(info["pid"]):
                    raise LoopError(f"another `loop.py {info.get('command')}` (pid {info['pid']}, since "
                                    f"{info.get('since')}) is working on this run; wait for it to finish") from None
                try:
                    young = time.time() - os.path.getmtime(self.path) < 10
                except OSError:
                    young = False
                if not isinstance(info, dict) and young:     # created a moment ago, holder still writing it
                    raise LoopError("another loop.py command is starting on this run; wait for it to finish")
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass
                continue
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "command": self.command, "since": now_iso()}, f)
            return self
        raise LoopError(f"could not take the run lock {self.path}")

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        return False


# ----------------------------------------------------------------------------- media
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # missing package or binary
        raise LoopError(f"the ffmpeg bundled with imageio-ffmpeg is not available ({e}); run scripts/setup_env.sh") from None


def probe_video(path):
    """{duration, fps, size} of a video from its header (bundled ffmpeg)."""
    try:
        import imageio_ffmpeg
        gen = imageio_ffmpeg.read_frames(path)
        try:
            meta = next(gen)
        finally:
            gen.close()
    except Exception as e:
        raise LoopError(f"could not read the video {path}: {e}") from None
    return {"duration": float(meta.get("duration") or 0.0), "fps": float(meta.get("fps") or 0.0),
            "size": list(meta.get("size") or (0, 0))}


def _heic_to_jpeg(src, dst):
    sips = shutil.which("sips")
    if sips:
        r = subprocess.run([sips, "-s", "format", "jpeg", src, "--out", dst], capture_output=True, text=True,
                           errors="replace", timeout=120)
        if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return
        raise LoopError(f"macOS sips could not convert {src} to JPEG: {last_line((r.stderr or r.stdout).splitlines())}")
    try:
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        with Image.open(src) as im:
            im.convert("RGB").save(dst, "JPEG", quality=95)
        return
    except ImportError:
        pass
    raise LoopError(f"{src} is HEIC/HEIF: converting it needs macOS `sips` or the pillow-heif package; "
                    "convert it to JPEG and pass that instead")


def _ingest_image(src, ext, inputs_dir):
    from PIL import Image, ImageOps
    path, tmp = src, None
    if ext in (".heic", ".heif"):
        tmp = os.path.join(inputs_dir, "_heic_source.jpg")
        _heic_to_jpeg(src, tmp)
        path = tmp
    out = os.path.join(inputs_dir, "media.jpg")
    try:
        with Image.open(path) as im0:
            im = ImageOps.exif_transpose(im0)
            if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                im = Image.new("RGB", rgba.size, (255, 255, 255))
                im.paste(rgba, mask=rgba.getchannel("A"))
            else:
                im = im.convert("RGB")
            im.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            im.save(out, "JPEG", quality=90)
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        raise LoopError(f"could not read the image {src}: {e}") from None
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    shutil.copyfile(out, os.path.join(inputs_dir, "ref_frame.jpg"))   # for a photo, the ref frame is the photo


def _ingest_video(src, inputs_dir):
    out = os.path.join(inputs_dir, "media.mp4")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src, "-map", "0:v:0",
           "-vf", PROXY_VF, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-fpsmax", "30", "-an", "-movflags", "+faststart", out]
    log(f"[init] making a <=720p H.264 proxy of {os.path.basename(src)} ...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        raise LoopError(f"ffmpeg needed more than 30 min for the proxy of {src}") from None
    if r.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) == 0:
        raise LoopError(f"ffmpeg could not convert the video {src}: "
                        f"{last_line((r.stderr or '').splitlines()) or f'exit code {r.returncode}'}")
    meta = probe_video(out)
    if meta["duration"] <= 0:
        raise LoopError(f"the video {src} has no frames")
    log(f"[init] proxy {meta['size'][0]}x{meta['size'][1]}, {meta['duration']:.1f} s")


def ingest_media(src, inputs_dir):
    """Copy the director's media into inputs/: a photo becomes a JPEG <= 1600 px (and the ref frame),
    a video an H.264 mp4 proxy <= 720p. Returns (kind, media path relative to the run dir)."""
    ext = os.path.splitext(src)[1].lower()
    if ext in IMAGE_EXTS:
        _ingest_image(src, ext, inputs_dir)
        return "image", "inputs/media.jpg"
    if ext in VIDEO_EXTS:
        _ingest_video(src, inputs_dir)
        return "video", "inputs/media.mp4"
    raise LoopError(f"unsupported media type {ext or '(no extension)'} for {src}; use a photo "
                    f"({' '.join(IMAGE_EXTS)}) or a video ({' '.join(VIDEO_EXTS)})")


def extract_frame(video, t, out):
    """Save the frame at t seconds (clamped into the video) as a JPEG; returns the time used."""
    dur = probe_video(video)["duration"]
    t = max(0.0, min(num(t) or 0.0, max(0.0, dur - 0.05)))
    err = ""
    for tt in (t, 0.0):
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", f"{tt:.3f}", "-i", video,
               "-frames:v", "1", "-q:v", "2", "-update", "1", out]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=120)
        if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 0:
            return tt
        err = last_line((r.stderr or "").splitlines()) or f"exit code {r.returncode}"
    raise LoopError(f"could not extract the reference frame at {t:.2f} s from {video}: {err}")


# ----------------------------------------------------------------------------- generated stunt script
def py_escape(s):
    """Escape text for the inside of any Python string literal ('...', "...", or triple-quoted)."""
    out = []
    for ch in str(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch in "\"'":
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 32 or o == 127:
            out.append(f"\\x{o:02x}")
        elif 0xD800 <= o <= 0xDFFF:       # lone surrogates (undecodable argv bytes) cannot be written as UTF-8
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    return "".join(out)


def generate_script(dest, brief, stunt_type):
    """Copy the stunt template to dest with the four placeholders filled in; it must compile."""
    template = os.environ.get("FALLGUY_STUNT_TEMPLATE") or DEFAULT_TEMPLATE
    if not os.path.isfile(template):
        raise LoopError(f"stunt template not found: {template} (scripts/stunt_template.py is part of the skill)", code=2)
    with open(template, encoding="utf-8") as f:
        src = f.read()
    values = {"__FALLGUY_SCRIPTS_DIR__": SCRIPTS_DIR, "__FALLGUY_BRIEF__": brief,
              "__FALLGUY_STUNT_TYPE__": stunt_type, "__FALLGUY_GENERATED_AT__": now_iso()}
    for k, v in values.items():
        if k not in src:
            log(f"[init] warning: the template has no {k} placeholder")
        src = src.replace(k, py_escape(v))
    left = sorted(set(re.findall(r"__FALLGUY_[A-Z0-9_]+__", src)))
    if left:
        log(f"[init] warning: placeholders loop.py does not know are left in the script: {', '.join(left)}")
    try:
        compile(src, dest, "exec")
    except SyntaxError as e:
        raise LoopError(f"the generated stunt script does not compile ({e.msg}, line {e.lineno}); "
                        f"template: {template}", code=2) from None
    with open(dest, "w", encoding="utf-8") as f:
        f.write(src)
    os.chmod(dest, 0o755)


def _eparams():
    try:
        from engine import params as eparams
    except ImportError as e:
        raise LoopError(f"cannot import engine.params from {SCRIPTS_DIR} ({e}); the skill install is incomplete") from None
    return eparams


def clamp_params(params, plan):
    """engine.params.clamp with the plan (stunt-type defaults, plan take-off x); plain clamp for an
    engine.params without the plan argument."""
    clamp = _eparams().clamp
    try:
        return dict(clamp(params, plan=plan))
    except TypeError:
        return dict(clamp(params))


def _gates():
    try:
        import gates
    except ImportError as e:
        raise LoopError(f"cannot import gates from {SCRIPTS_DIR} ({e}); the skill install is incomplete") from None
    return gates


def ensure_gate1(metrics):
    """Fill gate1.passed / failures / score with gates.evaluate_gate1 when the script did not."""
    if metrics.get("ok") and ("passed" not in (metrics.get("gate1") or {}) or "stunt_score" not in (metrics.get("score") or {})):
        metrics = _gates().evaluate_gate1(metrics) or metrics
    return metrics


def evaluate_gate2(metrics, critique):
    try:
        res = _gates().evaluate_gate2(metrics, critique)
    except LoopError:
        raise
    except Exception as e:
        log(f"warning: gates.evaluate_gate2 could not read this critique ({type(e).__name__}: {e}); Gate 2 fails")
        return {"passed": False, "reasons": [f"critique could not be evaluated: {e}"], "final_score": None}
    return {"passed": bool(res.get("passed")), "reasons": [str(r) for r in res.get("reasons") or []],
            "final_score": num(res.get("final_score"))}


def script_cmd(run, params_rel, out_rel, *extra):
    return [sys.executable, run["script"], "--plan", run.get("plan", "plan.json"), "--params", params_rel,
            "--out-dir", out_rel, *extra]


def render(run_dir, run, params_rel, out_rel, cid, fps, camera, hud):
    """Render one candidate with the generated script at the run's duration; returns the video path
    relative to the run dir. Exit code 2 on failure."""
    timeout = env_float("FALLGUY_RENDER_TIMEOUT", 900.0)
    d = float(run["duration_sec"])
    extra = ["--render", "--duration", f"{d:g}", "--fps", str(fps), "--camera", camera] + (["--hud"] if hud else [])
    video_rel = f"{out_rel}/video.mp4"
    video = os.path.join(run_dir, video_rel)
    if os.path.exists(video):          # never mistake an older render for this one
        os.remove(video)
    log(f"  rendering {cid}: {d:g} s at {fps} fps, {camera} camera{', HUD' if hud else ''} ...")
    rc, tail, secs, timed_out = run_child(script_cmd(run, params_rel, out_rel, *extra), run_dir, timeout,
                                          log_path=os.path.join(run_dir, out_rel, "render.log"), heartbeat=30,
                                          label=f"render {cid}")
    if timed_out or rc != 0 or not os.path.isfile(video) or os.path.getsize(video) == 0:
        why = f"timed out after {timeout:.0f} s" if timed_out else (f"exit code {rc}" if rc else "no video.mp4 written")
        m = try_read_json(os.path.join(run_dir, out_rel, "metrics.json")) or {}
        detail = m.get("error") if isinstance(m, dict) and m.get("error") else last_line(tail)
        raise LoopError(f"rendering {cid} failed ({why}): {detail}. Script: {os.path.join(run_dir, run['script'])}; "
                        f"log: {os.path.join(run_dir, out_rel, 'render.log')}", code=2)
    log(f"  rendered {video_rel} in {secs:.0f} s")
    return video_rel


# ----------------------------------------------------------------------------- init
def cmd_init(args):
    media = os.path.abspath(os.path.expanduser(args.media))
    if not os.path.isfile(media):
        raise LoopError(f"media not found: {media}")
    brief = args.brief.strip()
    if not brief:
        raise LoopError('--brief is empty: describe the stunt, e.g. "barrel roll over the table onto a crash mat"')
    duration = float(args.duration)
    if not 1.0 <= duration <= 120.0:
        raise LoopError(f"--duration {duration:g} s is out of range (1-120 s; 3-30 s is typical)")
    if not 3.0 <= duration <= 30.0:
        log(f"[init] warning: {duration:g} s is outside the usual 3-30 s")

    slug = slugify(brief)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir, tag = make_run_dir(os.path.abspath(os.path.expanduser(args.runs_dir)), slug, ts)
    try:
        return _init_in(run_dir, slug, tag, media, brief, duration)
    except LoopError as e:              # say where the half-made run is (its logs/ explain the failure)
        e.extra.setdefault("run_dir", run_dir)
        raise


def _init_in(run_dir, slug, tag, media, brief, duration):
    """init once the run dir exists: media copy, Argon scene plan, ref frame, stunt script, check, preview."""
    log(f"[init] run dir {run_dir}")
    inputs = os.path.join(run_dir, "inputs")
    os.makedirs(inputs)
    os.makedirs(os.path.join(run_dir, "logs"))
    kind, media_rel = ingest_media(media, inputs)
    log(f"[init] {kind} stored as {media_rel}")

    # Argon builds the scene plan from the media (long-video understanding for videos)
    argv = ["--media", media_rel, "--brief", brief, "--duration", f"{duration:g}", "--out", "plan.json"]
    if kind == "image":
        argv += ["--ref-frame", "inputs/ref_frame.jpg"]
    plan = call_argon(run_dir, "scene", argv, "plan.json", "[init] scene plan")
    if not isinstance(plan, dict):
        raise LoopError("Argon's plan.json is not a JSON object", code=3)
    source = plan.get("source") if isinstance(plan.get("source"), dict) else {}
    if kind == "video":
        t = extract_frame(os.path.join(run_dir, media_rel), source.get("ref_time_sec"),
                          os.path.join(inputs, "ref_frame.jpg"))
        log(f"[init] reference frame at {t:.2f} s -> inputs/ref_frame.jpg")
        source["ref_time_sec"] = round(t, 3)
    else:
        source.setdefault("ref_time_sec", 0.0)
    source.update(media=media_rel, kind=kind, ref_frame="inputs/ref_frame.jpg")
    plan["source"] = source
    plan.setdefault("brief", brief)
    plan["duration_sec"] = duration
    performer = plan.get("performer") if isinstance(plan.get("performer"), dict) else {}
    if performer.get("model", "h1") != "h1":
        log(f"[init] warning: the plan asked for performer {performer.get('model')!r}; the performer is always the H1")
    plan["performer"] = dict(performer, model="h1")
    eparams = _eparams()
    stunt = plan.get("stunt") if isinstance(plan.get("stunt"), dict) else {}
    stunt_type = str(stunt.get("type") or "custom")
    if stunt_type not in getattr(eparams, "STUNT_TYPES", [stunt_type]):
        log(f"[init] warning: Argon's stunt type {stunt_type!r} is unknown; using 'custom'")
        stunt_type = "custom"
    stunt["type"] = stunt_type
    plan["stunt"] = stunt
    write_json(os.path.join(run_dir, "plan.json"), plan)
    log(f"[init] plan: {stunt_type}, {len(plan.get('obstacles') or [])} obstacles, "
        f"{len(plan.get('crash_mats') or [])} crash mats, performer H1")

    # the run's own stunt script, generated from the template
    script = f"stunt_{slug}_{tag}.py"
    generate_script(os.path.join(run_dir, script), brief, stunt_type)
    log(f"[init] generated {script}")
    run = {
        "version": 1, "brief": brief, "duration_sec": duration,
        "media": media_rel, "media_kind": kind, "media_original": media, "ref_frame": "inputs/ref_frame.jpg",
        "plan": "plan.json", "script": script, "preview": None, "slug": slug, "tag": tag,
        "performer": "h1", "stunt_type": stunt_type, "created_at": now_iso(),
        "status": "planned", "waves": [], "best": None, "checkpoint_every": CHECKPOINT_EVERY, "director_notes": [],
        "final": None,
    }
    save_run(run_dir, run)

    # smoke-test the script, then render the preview still from the match camera
    check_dir = os.path.join(run_dir, "check")
    os.makedirs(check_dir, exist_ok=True)
    write_json(os.path.join(check_dir, "params.json"),
               clamp_params({"id": "check", "label": "default params", "camera": "match"}, plan))
    script_abs = os.path.join(run_dir, script)
    for flag, extra, log_name in (("--check", ["--check"], "check.log"),
                                  ("--preview-still", ["--preview-still", "preview.png"], "preview.log")):
        log(f"[init] {script} {flag} ...")
        rc, tail, secs, timed_out = run_child(script_cmd(run, "check/params.json", "check", *extra), run_dir,
                                              env_float("FALLGUY_SIM_TIMEOUT", 180.0), heartbeat=30, label=flag,
                                              log_path=os.path.join(check_dir, log_name))
        bad = "timed out" if timed_out else (f"exit code {rc}" if rc else None)
        if flag == "--preview-still" and not bad and not os.path.isfile(os.path.join(run_dir, "preview.png")):
            bad = "no preview.png written"
        if bad:
            raise LoopError(f"the generated stunt script failed {flag} ({bad}): {last_line(tail)}. Script: {script_abs}; "
                            f"log: {os.path.join(check_dir, log_name)}. Fix plan.json or the script and re-run "
                            f"`{script} --plan plan.json --params check/params.json --out-dir check {' '.join(extra)}` "
                            f"in the run dir", code=2, run_dir=run_dir, script=script_abs)
        if flag == "--check" and not any(line.strip().startswith("OK") for line in tail):
            log("[init] warning: --check exited 0 but did not print OK")
    run["preview"] = "preview.png"
    save_run(run_dir, run)
    log(f"[init] done: open {os.path.join(run_dir, 'preview.png')} and compare it with the media")
    j = lambda rel: os.path.join(run_dir, rel)
    return {"run_dir": run_dir, "script": script_abs, "plan": j("plan.json"), "preview": j("preview.png"),
            "ref_frame": j("inputs/ref_frame.jpg"), "media": j(media_rel), "stunt_type": stunt_type,
            "status": "planned"}


# ----------------------------------------------------------------------------- wave
def steer_for_wave(run, wave_no):
    """The director note given for this wave (after_wave = wave - 1), newest first, or None."""
    notes = [n for n in run.get("director_notes") or [] if isinstance(n, dict) and n.get("after_wave") == wave_no - 1]
    return notes[-1].get("text") if notes else None


def candidate_dirs(wave_dir):
    if not os.path.isdir(wave_dir):
        return []
    names = [d for d in os.listdir(wave_dir) if re.fullmatch(r"c\d+", d) and os.path.isdir(os.path.join(wave_dir, d))]
    return sorted(names, key=lambda d: int(d[1:]))


def build_history(run_dir, run, wave_no, steer):
    """history.json for argon.py propose (and critique, which reads it from the wave dir): every earlier
    wave's candidates (params + metrics), best id, critique and steer, plus the standing director notes."""
    waves = []
    for s in run["waves"]:
        wdir = os.path.join(run_dir, wave_rel(s["wave"]))
        cands = []
        for d in candidate_dirs(wdir):
            params = try_read_json(os.path.join(wdir, d, "params.json"))
            if params is None:
                continue
            metrics = try_read_json(os.path.join(wdir, d, "metrics.json")) or {"ok": False, "error": "no metrics.json"}
            cands.append({"params": params, "metrics": metrics})
        waves.append({"wave": s["wave"], "candidates": cands, "best_id": s.get("best_id"),
                      "critique": try_read_json(os.path.join(wdir, "critique.json")), "steer": s.get("steer"),
                      "gate1_passed": s.get("gate1_passed"), "gate2_passed": s.get("gate2_passed"),
                      "stunt_score": s.get("stunt_score"), "final_score": s.get("final_score"),
                      "failures": s.get("failures") or []})
    return {"brief": run["brief"], "duration_sec": run["duration_sec"], "wave": wave_no,
            "checkpoint_every": run["checkpoint_every"], "director_notes": list(run.get("director_notes") or []),
            "current_steer": steer, "waves": waves}


def normalize_candidates(proposal, wave_no, n, plan):
    cands = proposal.get("candidates") if isinstance(proposal, dict) else None
    if not isinstance(cands, list) or not cands or not all(isinstance(c, dict) for c in cands):
        raise LoopError("Argon's proposal has no usable candidates list", code=3)
    if len(cands) < n:
        log(f"[wave {wave_no}] warning: Argon proposed {len(cands)} candidates, not {n}")
    out = []
    for i, c in enumerate(cands[:n], 1):
        p = clamp_params(dict(c), plan)
        for k in ("label", "color", "camera", "why"):
            if k in c and k not in p:
                p[k] = c[k]
        p["id"] = f"w{wave_no}c{i}"
        p["label"] = str(p.get("label") or f"candidate {i}")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(p.get("color") or "")):
            p["color"] = PALETTE[(i - 1) % len(PALETTE)]
        if p.get("camera") not in CAMERAS:
            p["camera"] = "hero"
        out.append(p)
    return out


def _g1(metrics):
    return (metrics or {}).get("gate1") or {}


def candidate_row(r):
    g1 = _g1(r["metrics"])
    land = g1.get("landing") or {}
    row = {"id": r["id"], "label": r["params"].get("label"), "camera": r["params"].get("camera"), "ok": r["ok"]}
    if not r["ok"]:
        row["error"] = r["error"]
        return row
    row.update(gate1_passed=r["gate1_passed"], stunt_score=r["stunt_score"], head_g_peak=num(g1.get("head_g_peak")),
               hazard_collisions=g1.get("hazard_collisions"), min_clearance_cm=num(g1.get("min_clearance_cm")),
               landing_error_m=num(land.get("error_m")), inside_mat=land.get("inside_mat"),
               framing_pct=num(g1.get("framing_pct")), failures=list(g1.get("failures") or []))
    return row


def describe_result(r):
    label = f'{r["id"]} "{r["params"].get("label")}"'
    if not r["ok"]:
        return f"{label}: simulation FAILED after {r['secs']:.0f} s: {r['error']}"
    row = candidate_row(r)
    f = lambda v, nd: "?" if v is None else f"{v:.{nd}f}"
    s = (f"{label}: gate 1 {'PASS' if r['gate1_passed'] else 'FAIL'}, score {f(row['stunt_score'], 1)}, "
         f"head {f(row['head_g_peak'], 1)} G, clearance {f(row['min_clearance_cm'], 1)} cm, "
         f"landing {f(row['landing_error_m'], 2)} m{'' if row['inside_mat'] else ' (off the mat)'}, "
         f"framing {f(row['framing_pct'], 0)}% ({r['secs']:.0f} s)")
    if row["failures"]:
        s += " - " + "; ".join(str(x) for x in row["failures"])
    return s


def collect_result(run_dir, index, crel, params, rc, tail, secs, timed_out, timeout):
    """Read a candidate's metrics.json after its simulation; a failure is written back into metrics.json
    as {"ok": false, "error": ...} so the candidate directory explains itself."""
    mpath = os.path.join(run_dir, crel, "metrics.json")
    metrics = try_read_json(mpath)
    if not isinstance(metrics, dict):
        metrics = None
    err = None
    if timed_out:
        err = f"simulation timed out after {timeout:.0f} s"
    elif metrics is None:
        err = f"exit code {rc} and no metrics.json; last output: {last_line(tail) or '(none)'}"
    elif not metrics.get("ok"):
        err = str(metrics.get("error") or f"metrics.json says ok=false (exit code {rc})")
    elif rc != 0:
        err = f"exit code {rc} after writing metrics.json; last output: {last_line(tail) or '(none)'}"
    if err:
        metrics = dict(metrics or {}, candidate_id=params["id"], ok=False, error=err)
        write_json(mpath, metrics)
    else:
        before = json.dumps(metrics, sort_keys=True, default=_json_default)
        metrics = ensure_gate1(metrics)
        if json.dumps(metrics, sort_keys=True, default=_json_default) != before:
            write_json(mpath, metrics)
    return {"index": index, "id": params["id"], "dir": crel, "params": params, "metrics": metrics,
            "ok": err is None, "error": err, "gate1_passed": err is None and bool(_g1(metrics).get("passed")),
            "stunt_score": num((metrics.get("score") or {}).get("stunt_score")) if err is None else None,
            "secs": round(secs, 1), "log": f"{crel}/sim.log"}


def simulate(run_dir, run, wave_no, cands):
    """Run the stunt script (sim + metrics only) for every candidate, up to 4 at a time."""
    timeout = env_float("FALLGUY_SIM_TIMEOUT", 180.0)
    wrel = wave_rel(wave_no)

    def one(i, p):
        crel = f"{wrel}/c{i}"
        rc, tail, secs, timed_out = run_child(script_cmd(run, f"{crel}/params.json", crel), run_dir, timeout,
                                              log_path=os.path.join(run_dir, crel, "sim.log"), label=p["id"])
        return collect_result(run_dir, i, crel, p, rc, tail, secs, timed_out, timeout)

    log(f"[wave {wave_no}] simulating {len(cands)} candidates in MuJoCo ({min(MAX_PARALLEL, len(cands))} at a time, "
        f"{timeout:.0f} s limit each) ...")
    results, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(cands))) as ex:
        pending = {ex.submit(one, i, p) for i, p in enumerate(cands, 1)}
        while pending:
            done, pending = cf.wait(pending, timeout=30, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                r = fut.result()
                results.append(r)
                log(f"[wave {wave_no}] {describe_result(r)}")
            if not done:
                log(f"[wave {wave_no}] simulating: {len(results)}/{len(cands)} done ({time.time() - t0:.0f} s)")
    return sorted(results, key=lambda r: r["index"])


def pick_best(run):
    """Best wave-best so far: Gate 2 passers, then Gate 1 passers, then stunt score. A director note
    redefines the target, so only waves after the latest note compete."""
    after = [n.get("after_wave") for n in run.get("director_notes") or [] if isinstance(n, dict)]
    after = [int(a) for a in after if isinstance(a, int)]
    since = max(after) + 1 if after else 1
    pool = [w for w in run["waves"] if w["wave"] >= since] or run["waves"]
    score = lambda w: num(w.get("stunt_score")) if num(w.get("stunt_score")) is not None else -1e9
    w = max(pool, key=lambda w: (bool(w.get("gate2_passed")), bool(w.get("gate1_passed")), score(w)))
    return {"wave": w["wave"], "id": w["best_id"], "stunt_score": w.get("stunt_score"),
            "final_score": w.get("final_score"), "gate1_passed": bool(w.get("gate1_passed")),
            "gate2_passed": bool(w.get("gate2_passed")), "since_wave": since}


def cmd_wave(args):
    run_dir = resolve_run_dir(args.run_dir)
    if not 1 <= args.candidates <= 8:
        raise LoopError(f"--candidates {args.candidates} is out of range (1-8)")
    with RunLock(run_dir, "wave"):
        run = load_run(run_dir)
        for rel in (run["script"], run.get("plan", "plan.json")):
            if not os.path.isfile(os.path.join(run_dir, rel)):
                raise LoopError(f"{os.path.join(run_dir, rel)} is missing; re-run `loop.py init`")
        plan = read_json(os.path.join(run_dir, run.get("plan", "plan.json")), "plan.json")
        n = len(run["waves"]) + 1
        wrel = wave_rel(n)
        wdir = os.path.join(run_dir, wrel)
        if os.path.exists(wdir):          # leftovers of a crashed attempt at this wave
            stale = f"{wdir}_stale_{dt.datetime.now():%H%M%S}"
            os.replace(wdir, stale)
            log(f"[wave {n}] moved the unfinished earlier attempt to {os.path.relpath(stale, run_dir)}")
        os.makedirs(wdir)

        # a steer becomes a standing director note from this wave on (after_wave = n - 1)
        steer = (args.steer or "").strip() or None
        if steer:
            if not any(isinstance(x, dict) and x.get("after_wave") == n - 1 and x.get("text") == steer
                       for x in run["director_notes"]):
                run["director_notes"].append({"after_wave": n - 1, "text": steer})
        else:                              # a steer given to an earlier, crashed attempt at this wave still counts
            steer = steer_for_wave(run, n)
        if run.get("final"):
            run["final"]["stale"] = True
        run["status"] = "running"
        save_run(run_dir, run)
        t0 = time.time()

        # 1. Argon proposes candidates from the full history
        write_json(os.path.join(wdir, "history.json"), build_history(run_dir, run, n, steer))
        argv = ["--plan", run.get("plan", "plan.json"), "--history", f"{wrel}/history.json",
                "--out", f"{wrel}/proposal.json", "--n", str(args.candidates)]
        if steer:
            argv += ["--steer", steer]
            log(f'[wave {n}] director steer: "{steer}"')
        if len(run["director_notes"]) > (1 if steer else 0):
            log(f"[wave {n}] standing director notes: {len(run['director_notes'])}")
        proposal = call_argon(run_dir, "propose", argv, f"{wrel}/proposal.json", f"[wave {n}] proposal")
        cands = normalize_candidates(proposal, n, args.candidates, plan)
        for i, p in enumerate(cands, 1):
            os.makedirs(os.path.join(wdir, f"c{i}"), exist_ok=True)
            write_json(os.path.join(wdir, f"c{i}", "params.json"), p)
            log(f'[wave {n}] {p["id"]} "{p["label"]}" ({p["camera"]} camera): {p.get("why") or ""}'.rstrip(": "))

        # 2. MuJoCo: Gate 1 for every candidate
        results = simulate(run_dir, run, n, cands)
        ok = [r for r in results if r["ok"]]
        if not ok:
            first = results[0]
            raise LoopError(f"all {len(results)} candidates failed to simulate. First error ({first['id']}): "
                            f"{first['error']}. Fix the stunt script {os.path.join(run_dir, run['script'])} "
                            f"(log: {os.path.join(run_dir, first['log'])}) and re-run the wave", code=2,
                            script=os.path.join(run_dir, run["script"]), log=os.path.join(run_dir, first["log"]))
        best = max(ok, key=lambda r: (r["gate1_passed"], r["stunt_score"] if r["stunt_score"] is not None else -1e9))
        log(f"[wave {n}] best: {best['id']} (gate 1 {'PASS' if best['gate1_passed'] else 'FAIL'}, "
            f"score {best['stunt_score']})")

        # 3. render the best take at the requested length and let Argon watch it (Gate 2)
        crel = best["dir"]
        camera = best["params"].get("camera") or "hero"
        video_rel = render(run_dir, run, f"{crel}/params.json", crel, best["id"], 30, camera, False)
        metrics = try_read_json(os.path.join(run_dir, crel, "metrics.json"))
        if not isinstance(metrics, dict) or not metrics.get("ok"):
            raise LoopError(f"the render run of {best['id']} wrote no valid metrics.json: "
                            f"{(metrics or {}).get('error') if isinstance(metrics, dict) else 'missing'}. "
                            f"Script: {os.path.join(run_dir, run['script'])}", code=2)
        metrics = ensure_gate1(metrics)
        s_after = num((metrics.get("score") or {}).get("stunt_score"))
        if s_after is not None and best["stunt_score"] is not None and abs(s_after - best["stunt_score"]) > 0.5:
            log(f"[wave {n}] warning: the render re-simulation of {best['id']} scored {s_after:.1f}, "
                f"the first simulation {best['stunt_score']:.1f}; using the rendered run")
        critique = call_argon(run_dir, "critique",
                              ["--video", video_rel, "--plan", run.get("plan", "plan.json"),
                               "--params", f"{crel}/params.json", "--metrics", f"{crel}/metrics.json",
                               "--out", f"{wrel}/critique.json", "--fps", "5"],
                              f"{wrel}/critique.json", f"[wave {n}] critique")
        if not isinstance(critique, dict):
            raise LoopError("Argon's critique.json is not a JSON object", code=3)
        g2 = evaluate_gate2(metrics, critique)

        g1 = _g1(metrics)
        land = g1.get("landing") or {}
        g1_failures = [str(x) for x in g1.get("failures") or []]
        g2_only = [r for r in g2["reasons"] if r not in g1_failures and not r.lower().startswith("gate 1")]
        failures = g1_failures + (g2_only if not g2["passed"] else [])
        summary = {
            "wave": n, "best_id": best["id"], "gate1_passed": bool(g1.get("passed")), "gate2_passed": g2["passed"],
            "stunt_score": num((metrics.get("score") or {}).get("stunt_score")), "failures": failures,
            "critique_summary": str(critique.get("summary") or ""), "next": "wave",
            "final_score": g2["final_score"], "style_score": num(critique.get("style_score")),
            "gate2_reasons": g2["reasons"], "steer": steer,
            "best": {"label": best["params"].get("label"), "camera": camera,
                     "head_g_peak": num(g1.get("head_g_peak")), "hic15": num(g1.get("hic15")),
                     "hazard_collisions": g1.get("hazard_collisions"),
                     "min_clearance_cm": num(g1.get("min_clearance_cm")), "landing_error_m": num(land.get("error_m")),
                     "inside_mat": land.get("inside_mat"), "framing_pct": num(g1.get("framing_pct")),
                     "argon_in_frame_pct": num((critique.get("framing") or {}).get("in_frame_pct")),
                     "rotation_observed_deg": num(critique.get("rotation_observed_deg")),
                     "video": video_rel, "params": f"{crel}/params.json", "metrics": f"{crel}/metrics.json",
                     "critique": f"{wrel}/critique.json"},
            "candidates": [candidate_row(r) for r in results],
            "rationale": str(proposal.get("rationale") or ""),
            "elapsed_s": None,
        }
        run["waves"].append(summary)
        run["best"] = pick_best(run)
        if run["best"]["gate2_passed"]:            # a Gate-2 take exists (since the latest director note)
            run["status"], summary["next"] = "passed", "finalize"
        elif n % run["checkpoint_every"] == 0:      # 5, 10, 15, ... waves without a pass: the director decides
            run["status"] = summary["next"] = "checkpoint"
        else:
            run["status"], summary["next"] = "running", "wave"
        summary["elapsed_s"] = round(time.time() - t0, 1)
        write_json(os.path.join(wdir, "summary.json"), summary)
        save_run(run_dir, run)
        _refresh_report(run_dir, run)
        log(f"[wave {n}] gate 1 {'PASS' if summary['gate1_passed'] else 'FAIL'}, gate 2 "
            f"{'PASS' if summary['gate2_passed'] else 'FAIL'}, stunt score {summary['stunt_score']}, "
            f"final score {summary['final_score']} -> next: {summary['next']} ({summary['elapsed_s']:.0f} s)")
        if summary["next"] == "checkpoint":
            log(f"[wave {n}] checkpoint: {run['checkpoint_every']} more waves without a Gate 2 pass; "
                "ask the director whether to continue and with what feedback")
        return summary


def _refresh_report(run_dir, run):
    try:
        import report
        report.write_report(run_dir, run)
    except Exception as e:  # the report is a convenience; never fail a wave over it
        log(f"warning: could not refresh report.md ({type(e).__name__}: {e})")


# ----------------------------------------------------------------------------- finalize
def cmd_finalize(args):
    run_dir = resolve_run_dir(args.run_dir)
    with RunLock(run_dir, "finalize"):
        run = load_run(run_dir)
        if not run["waves"]:
            raise LoopError(f"no waves yet; run `loop.py wave --run-dir {run_dir}` first")
        cid = (args.candidate or (run.get("best") or {}).get("id") or "").strip()
        m = re.fullmatch(r"w(\d+)c(\d+)", cid)
        if not m:
            raise LoopError(f"bad candidate id {cid!r}; ids look like w2c3 (see `loop.py status`)")
        wn, ci = int(m.group(1)), int(m.group(2))
        crel = f"{wave_rel(wn)}/c{ci}"
        params = read_json(os.path.join(run_dir, crel, "params.json"), f"params of {cid}")
        wave_metrics = read_json(os.path.join(run_dir, crel, "metrics.json"), f"metrics of {cid}")
        if not wave_metrics.get("ok"):
            raise LoopError(f"{cid} failed to simulate ({wave_metrics.get('error')}); pick another with --candidate")
        camera = params.get("camera") if params.get("camera") in CAMERAS else "hero"
        log(f"[finalize] {cid} \"{params.get('label')}\" -> final render")

        fdir = os.path.join(run_dir, "final")
        if os.path.exists(fdir):
            shutil.rmtree(fdir)
        os.makedirs(fdir)
        shutil.copyfile(os.path.join(run_dir, crel, "params.json"), os.path.join(fdir, "params.json"))
        video_rel = render(run_dir, run, f"{crel}/params.json", "final", cid, 60, camera, True)
        final_name = f"final_{run['slug']}_{run['tag']}.mp4"
        os.replace(os.path.join(run_dir, video_rel), os.path.join(run_dir, final_name))
        metrics_rel = "final/metrics.json"
        metrics = try_read_json(os.path.join(run_dir, metrics_rel))
        if not isinstance(metrics, dict) or not metrics.get("ok"):
            log("[finalize] warning: the final render wrote no valid metrics.json; certifying the wave's metrics")
            metrics, metrics_rel = wave_metrics, f"{crel}/metrics.json"
        metrics = ensure_gate1(metrics)
        a = num((metrics.get("score") or {}).get("stunt_score"))
        b = num((wave_metrics.get("score") or {}).get("stunt_score"))
        if a is not None and b is not None and abs(a - b) > 0.5:
            log(f"[finalize] warning: the final simulation scored {a:.1f}, the wave's {b:.1f}; certifying the final run")

        # Gate 2 verdict: the wave's critique of this take, or Argon watches the final video now
        wsum = next((w for w in run["waves"] if w["wave"] == wn), None)
        crit_rel = f"{wave_rel(wn)}/critique.json"
        critique = try_read_json(os.path.join(run_dir, crit_rel)) if wsum and wsum.get("best_id") == cid else None
        if not isinstance(critique, dict):
            log(f"[finalize] {cid} was not the critiqued take of wave {wn}; Argon watches the final video")
            crit_rel = "final/critique.json"     # argon.py reads the director notes from history.json beside --out
            write_json(os.path.join(fdir, "history.json"), build_history(run_dir, run, len(run["waves"]) + 1, None))
            critique = call_argon(run_dir, "critique",
                                  ["--video", final_name, "--plan", run.get("plan", "plan.json"),
                                   "--params", f"{crel}/params.json", "--metrics", metrics_rel,
                                   "--out", crit_rel, "--fps", "5"], crit_rel, "[finalize] critique")
            if not isinstance(critique, dict):
                raise LoopError("Argon's critique.json is not a JSON object", code=3)
        g2 = evaluate_gate2(metrics, critique)

        import report
        cert_name = f"certificate_{run['slug']}_{run['tag']}.json"
        cert = report.build_certificate(run_dir, run, cid, wn, params, metrics, critique, g2, final_name, crit_rel)
        write_json(os.path.join(run_dir, cert_name), cert)
        run["final"] = {"candidate": cid, "wave": wn, "video": final_name, "certificate": cert_name,
                        "report": "report.md", "critique": crit_rel, "passed": cert["passed"],
                        "verdict": cert["verdict"], "finalized_at": now_iso(), "stale": False}
        run["status"] = "finalized"
        save_run(run_dir, run)
        report.write_report(run_dir, run)
        log(f"[finalize] {cert['verdict']}: {final_name}, {cert_name}, report.md")
        j = lambda rel: os.path.join(run_dir, rel)
        return {"final_video": j(final_name), "report": j("report.md"), "certificate": j(cert_name),
                "passed": cert["passed"], "verdict": cert["verdict"], "candidate": cid, "script": j(run["script"])}


# ----------------------------------------------------------------------------- status / gates
def cmd_status(args):
    run_dir = resolve_run_dir(args.run_dir)
    run = load_run(run_dir)
    busy = read_lock(run_dir)
    import report
    log(report.status_table(run, run_dir, busy))
    rows = [{k: w.get(k) for k in ("wave", "best_id", "stunt_score", "final_score", "gate1_passed", "gate2_passed",
                                   "style_score", "steer", "failures")} for w in run["waves"]]
    return {"run_dir": run_dir, "brief": run.get("brief"), "status": run.get("status"), "waves_done": len(run["waves"]),
            "checkpoint_every": run.get("checkpoint_every"), "next": report.NEXT_FOR_STATUS.get(run.get("status")),
            "best": run.get("best"), "director_notes": run.get("director_notes"), "final": run.get("final"),
            "busy": busy, "script": os.path.join(run_dir, run.get("script", "")), "waves": rows}


def cmd_gates(args):
    gates = _gates()
    text = gates.describe()
    log(text)
    return {"thresholds": gates.THRESHOLDS, "weights": gates.WEIGHTS, "description": text}


# ----------------------------------------------------------------------------- CLI
class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        log(f"{self.prog}: error: {message}")
        emit({"error": message, "exit_code": 1})
        sys.exit(1)


def build_parser():
    ap = _Parser(prog="loop.py", description="FALLGUY recursive stunt loop (see reference/contracts.md).")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="{init,wave,finalize,status,gates}")
    p = sub.add_parser("init", help="copy the media, let Argon plan the scene, generate the stunt script, preview")
    p.add_argument("--media", required=True, help="site photo or video")
    p.add_argument("--brief", required=True, help="the stunt, in the director's words")
    p.add_argument("--duration", required=True, type=float, help="final video length in seconds")
    p.add_argument("--runs-dir", default="stunts", help="where run directories go (default ./stunts)")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("wave", help="propose, simulate, render the best, critique, gate")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--steer", default=None, help="director feedback; a standing note for this and every later wave")
    p.add_argument("--candidates", type=int, default=4)
    p.set_defaults(func=cmd_wave)
    p = sub.add_parser("finalize", help="final render (60 fps, HUD), certificate and report.md")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--candidate", default=None, help="candidate id such as w2c3 (default: the best so far)")
    p.set_defaults(func=cmd_finalize)
    p = sub.add_parser("status", help="convergence table of a run")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("gates", help="Gate 1 / Gate 2 thresholds and score weights")
    p.set_defaults(func=cmd_gates)
    return ap


def main(argv=None):
    global _REAL_STDOUT
    _REAL_STDOUT = sys.stdout
    args = build_parser().parse_args(argv)
    sys.stdout = sys.stderr            # stray prints from imported modules must not break the JSON line
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    if threading.current_thread() is threading.main_thread():
        for s in old:
            signal.signal(s, _on_signal)
    try:
        emit(args.func(args))
        return 0
    except LoopError as e:
        msg = redact(str(e))
        log(f"loop.py {args.cmd}: error: {msg}")
        emit({"error": msg, "exit_code": e.code, **{k: redact(v) for k, v in e.extra.items()}})
        return e.code
    except KeyboardInterrupt:
        log(f"loop.py {args.cmd}: interrupted; child processes were stopped")
        emit({"error": "interrupted", "exit_code": 130})
        return 130
    except Exception as e:  # a bug: show the traceback, still print one JSON line
        log(redact(traceback.format_exc()))
        emit({"error": redact(f"internal error: {type(e).__name__}: {e}"), "exit_code": 1})
        return 1
    finally:
        sys.stdout = _REAL_STDOUT
        if threading.current_thread() is threading.main_thread():
            for s, h in old.items():
                signal.signal(s, h)


if __name__ == "__main__":
    sys.exit(main())
