"""Background jobs: run the fallguy CLIs as subprocesses (cwd = repo root), keep a log tail, parse the final JSON line.

State lives in memory and in jobs/<id>.json so the UI can re-attach after a page reload (or a server restart:
jobs that were running when the server died are reported as failed with a note)."""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO / "jobs"
PY = sys.executable


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _last_json(lines):
    for ln in reversed(lines):
        s = ln.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                continue
    return None


class Job:
    def __init__(self, kind, steps, meta=None, job_id=None):
        self.id = job_id or uuid.uuid4().hex[:12]
        self.kind = kind
        self.steps = steps            # [(label, argv)]
        self.meta = dict(meta or {})  # world_id, stunt_id, take ...
        self.status = "queued"
        self.step = None
        self.step_index = 0
        self.log = deque(maxlen=400)
        self.result = None
        self.results = []
        self.error = None
        self.started = None
        self.ended = None
        self.proc = None
        self._last_save = 0.0

    def to_dict(self, full=False):
        d = {"job_id": self.id, "kind": self.kind, "status": self.status, "step": self.step,
             "step_index": self.step_index, "steps": [s[0] for s in self.steps],
             "log_tail": list(self.log)[-40:], "result": self.result, "results": self.results, "error": self.error,
             "started": self.started, "ended": self.ended, **{k: v for k, v in self.meta.items() if k != "_hooks"}}
        if full:
            d["log"] = list(self.log)
            d["argv"] = [s[1] for s in self.steps]
        return d

    def save(self, force=False):
        now = time.time()
        if not force and now - self._last_save < 1.0:
            return
        self._last_save = now
        try:
            JOBS_DIR.mkdir(exist_ok=True)
            tmp = JOBS_DIR / f"{self.id}.json.tmp"
            tmp.write_text(json.dumps(self.to_dict(full=True), default=str))
            os.replace(tmp, JOBS_DIR / f"{self.id}.json")
        except Exception as e:  # noqa: BLE001
            print(f"[app] job save failed: {e}", flush=True)


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        if not JOBS_DIR.is_dir():
            return
        for p in JOBS_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            j = Job(d.get("kind"), [(lbl, argv) for lbl, argv in zip(d.get("steps") or [], d.get("argv") or [])],
                    job_id=d.get("job_id"))
            for k in ("status", "step", "step_index", "result", "results", "error", "started", "ended"):
                setattr(j, k, d.get(k))
            j.results = j.results or []
            j.log.extend(d.get("log") or d.get("log_tail") or [])
            j.meta = {k: v for k, v in d.items() if k in ("world_id", "stunt_id", "take", "from", "to", "media",
                                                         "parent_stunt_id", "world_dir", "stunt_dir")}
            if j.status in ("running", "queued"):
                j.status, j.error, j.ended = "failed", "server restarted while the job was running", j.ended or _now()
                j.save(force=True)
            self.jobs[j.id] = j

    def get(self, job_id):
        return self.jobs.get(job_id)

    def list(self, **filt):
        kind = filt.pop("kind", None)
        out = [j for j in self.jobs.values() if (kind is None or j.kind == kind)
               and all(j.meta.get(k) == v for k, v in filt.items())]
        return sorted(out, key=lambda j: j.started or "", reverse=True)

    def running(self, **filt):
        return [j for j in self.list(**filt) if j.status in ("queued", "running")]

    def start(self, kind, steps, meta=None, on_line=None, on_step_done=None, on_done=None):
        """steps: [(label, argv)] run in sequence; stop at the first failing step.
        on_line(job, line), on_step_done(job, label, result), on_done(job) are called from the worker thread."""
        j = Job(kind, steps, meta)
        with self.lock:
            self.jobs[j.id] = j
        j.save(force=True)
        threading.Thread(target=self._run, args=(j, on_line, on_step_done, on_done), daemon=True).start()
        return j

    def _run(self, j, on_line, on_step_done, on_done):
        j.status, j.started = "running", _now()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            for i, (label, argv) in enumerate(j.steps):
                j.step_index, j.step = i, label
                j.log.append(f"$ {label}")
                j.save(force=True)
                lines = []
                p = subprocess.Popen(argv, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, env=env)
                j.proc = p
                for ln in p.stdout:
                    ln = ln.rstrip("\n")
                    lines.append(ln)
                    j.log.append(ln[:2000])
                    if ln.strip() and not ln.strip().startswith("{"):
                        j.step = f"{label}: {ln.strip()[:160]}"
                    if on_line:
                        try:
                            on_line(j, ln)
                        except Exception as e:  # noqa: BLE001
                            print(f"[app] on_line: {e}", flush=True)
                    j.save()
                rc = p.wait()
                res = _last_json(lines)
                j.results.append({"step": label, "rc": rc, "result": res})
                if res is not None:
                    j.result = res
                if rc != 0:
                    j.status, j.error = "failed", f"{label} exited with code {rc}"
                    break
                if on_step_done:
                    try:
                        on_step_done(j, label, res)
                    except Exception as e:  # noqa: BLE001
                        j.log.append(f"[app] post-step hook failed: {e}")
            else:
                j.status = "done"
            j.step = j.step if j.status == "failed" else "done"
        except Exception as e:  # noqa: BLE001
            j.status, j.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            j.ended = _now()
            j.proc = None
            if on_done:
                try:
                    on_done(j)
                except Exception as e:  # noqa: BLE001
                    j.log.append(f"[app] on_done hook failed: {e}")
            j.save(force=True)

    def kill_all(self):
        for j in self.jobs.values():
            if j.proc and j.proc.poll() is None:
                j.proc.terminate()
