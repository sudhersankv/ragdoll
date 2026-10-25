"""FALLGUY web app backend (FastAPI): upload a location video -> world replica -> stunt takes -> metrics/critique loop.

Runs the existing CLIs (`python -m fallguy.world ...`, `python -m fallguy.stunt ...`) as background jobs and reads their
output folders (worlds/, stunts/). The 3D data endpoints reuse fallguy/viewer/server.py (Viewer). The frontend is
served from fallguy/app/static/ at /; the stock viewer page is also mounted per world/stunt at /viewer/<world>/<stunt>/.
"""
import json
import mimetypes
import os
import re
import time
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..viewer import server as VS
from .jobs import PY, REPO, JobManager

WORLDS = REPO / "worlds"
STUNTS = REPO / "stunts"
UPLOADS = REPO / "uploads"
STATIC = Path(__file__).resolve().parent / "static"
VIEWER_STATIC = VS.STATIC
TAKE_RE = re.compile(r"v\d+")
ID_RE = re.compile(r"[A-Za-z0-9._-]+")
mimetypes.add_type("model/gltf-binary", ".glb")

app = FastAPI(title="FALLGUY")
JOBS = JobManager()
for d in (WORLDS, STUNTS, UPLOADS):
    d.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- helpers
def rj(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def wj(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def _iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _take_key(name):
    return int(name[1:]) if TAKE_RE.fullmatch(name) else 10 ** 6


def world_dir(wid):
    if not ID_RE.fullmatch(wid or "") or not (WORLDS / wid).is_dir():
        raise HTTPException(404, f"no world {wid!r}")
    return WORLDS / wid


def stunt_dir(sid):
    if not ID_RE.fullmatch(sid or "") or not (STUNTS / sid / "stunt.json").is_file():
        raise HTTPException(404, f"no stunt {sid!r}")
    return STUNTS / sid


def check_take(take):
    if not TAKE_RE.fullmatch(take or ""):
        raise HTTPException(400, f"bad take {take!r} (want v1, v2, ...)")
    return take


def wurl(wid, rel):
    return f"/files/worlds/{wid}/{rel}"


def surl(sid, rel):
    return f"/files/stunts/{sid}/{rel}"


def _rewrite(obj, wid, sid):
    """viewer URLs (/files/world/, /files/stunt/) -> app URLs."""
    if isinstance(obj, str):
        if obj.startswith("/files/world/"):
            return wurl(wid, obj[len("/files/world/"):])
        if obj.startswith("/files/stunt/") and sid:
            return surl(sid, obj[len("/files/stunt/"):])
        return obj
    if isinstance(obj, dict):
        return {k: _rewrite(v, wid, sid) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rewrite(v, wid, sid) for v in obj]
    return obj


def _num(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ----------------------------------------------------------------------------- viewer instances (3D data)
_viewers = {}


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return 0


def viewer(wid, sid=None):
    wd = world_dir(wid) if wid else None
    sd = stunt_dir(sid) if sid else None
    if wd is None and sd is not None:
        wd = _stunt_world_dir(sd)
    fp = (_mtime(wd / "scene.xml") if wd else 0, _mtime(wd / "world.json") if wd else 0,
          tuple((t.name, _mtime(t / "scene.xml")) for t in sorted(sd.iterdir()) if t.is_dir()) if sd else ())
    key = (str(wd), str(sd))
    ent = _viewers.get(key)
    if ent is None or ent[0] != fp:
        ent = (fp, VS.Viewer(wd, sd))
        _viewers[key] = ent
    return ent[1]


def _stunt_world_dir(sd):
    st = rj(sd / "stunt.json", {}) or {}
    w = st.get("world_dir") or st.get("world")
    if isinstance(w, str):
        p = Path(w)
        p = p if p.is_absolute() else REPO / p
        if p.is_dir():
            return p.resolve()
    return None


def _stunt_world_id(sd):
    w = _stunt_world_dir(sd)
    return w.name if w is not None and w.parent.resolve() == WORLDS.resolve() else (w.name if w else None)


# ----------------------------------------------------------------------------- worlds
def _world_views(wd, wid):
    vd = wd / "views"
    return [wurl(wid, f"views/{p.name}") for p in sorted(vd.glob("*.png"))] if vd.is_dir() else []


def _verify_rounds(wd):
    out = []
    vd = wd / "verify"
    if vd.is_dir():
        for p in sorted(vd.glob("round*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
            r = rj(p)
            if isinstance(r, dict):
                out.append(r)
    return out


def _fidelity(wd, wj_):
    rounds = _verify_rounds(wd)
    if rounds and isinstance(rounds[-1].get("score"), (int, float)):
        return rounds[-1]["score"]
    v = (wj_ or {}).get("verify") or []
    if isinstance(v, list) and v and isinstance(v[-1], dict):
        return v[-1].get("score")
    return None


def _thumb(wd, wid):
    vd = wd / "views"
    if vd.is_dir():
        refs = sorted(vd.glob("r*_ref_mesh.png"), key=lambda p: (len(p.name), p.name))
        if refs:
            return wurl(wid, f"views/{refs[-1].name}")
        for name in ("argon_grid_ref.png",):
            if (vd / name).is_file():
                return wurl(wid, f"views/{name}")
    if (wd / "frames" / "ref.jpg").is_file():
        return wurl(wid, "frames/ref.jpg")
    return None


def _newest_mtime(d, depth=2):
    best = _mtime(d)
    try:
        for p in d.iterdir():
            best = max(best, _mtime(p))
            if depth > 1 and p.is_dir():
                best = max(best, _newest_mtime(p, depth - 1))
    except OSError:
        pass
    return best


def world_status(wd):
    wid = wd.name
    if JOBS.running(world_id=wid, kind="world_build"):
        return "building"
    if (wd / "world.json").is_file() and (wd / "scene.xml").is_file():
        return "ready"
    past = JOBS.list(world_id=wid, kind="world_build")
    if past and past[0].status == "failed":
        return "failed"
    # a build started outside the app (CLI): still working if its files changed recently
    if time.time() - _newest_mtime(wd) < 20 * 60:
        return "building"
    return "failed"


def world_summary(wd):
    wid = wd.name
    w = rj(wd / "world.json", {}) or {}
    src = w.get("source_media")
    media = (src.get("path") or src.get("file")) if isinstance(src, dict) else src
    return {"id": wid, "name": w.get("slug") or re.sub(r"-\d{8}-\d{6}$", "", wid), "media": os.path.basename(media) if media else None,
            "created": w.get("created") or _iso(_mtime(wd)), "status": world_status(wd),
            "fidelity_score": _fidelity(wd, w), "thumb_url": _thumb(wd, wid),
            "objects": len(w.get("objects") or []),
            "jobs": [j.id for j in JOBS.running(world_id=wid)]}


@app.get("/api/worlds")
def list_worlds():
    ws = [world_summary(d) for d in WORLDS.iterdir() if d.is_dir() and not d.name.startswith(".")]
    return sorted(ws, key=lambda w: w["created"] or "", reverse=True)


@app.post("/api/worlds")
async def upload_world(video: UploadFile = File(...)):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(video.filename or "video.mov")) or "video.mov"
    dst = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    with open(dst, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    before = {p.name for p in WORLDS.iterdir() if p.is_dir()}

    def on_line(job, line):
        if job.meta.get("world_id"):
            return
        new = [p for p in WORLDS.iterdir() if p.is_dir() and p.name not in before]
        if new:
            job.meta["world_id"] = max(new, key=_mtime).name

    def on_done(job):
        wd = (job.result or {}).get("world_dir") if isinstance(job.result, dict) else None
        if wd:
            job.meta["world_id"] = Path(wd).name

    job = JOBS.start("world_build", [("build world", [PY, "-m", "fallguy.world", "build", "--media", str(dst),
                                                      "--out", "worlds"])],
                     meta={"media": dst.name}, on_line=on_line, on_done=on_done)
    return {"job_id": job.id, "world_id": job.meta.get("world_id"), "upload": f"/files/uploads/{dst.name}"}


@app.get("/api/worlds/{wid}")
def get_world(wid: str):
    wd = world_dir(wid)
    w = rj(wd / "world.json", {}) or {}
    sbs = []
    views = w.get("views") if isinstance(w.get("views"), dict) else {}
    for k in sorted(views):
        v = views[k]
        if isinstance(v, dict):
            sbs.append({"key": k, **{kk: wurl(wid, vv) if isinstance(vv, str) and vv.endswith((".png", ".jpg")) else vv
                                      for kk, vv in v.items()}})
    stunts = [stunt_summary(sd) for sd in _stunt_dirs() if _stunt_world_id(sd) == wid]
    return {"id": wid, "summary": world_summary(wd), "world": w, "views": _world_views(wd, wid), "side_by_side": sbs,
            "verify": _verify_rounds(wd), "stunts": stunts,
            "jobs": [j.to_dict() for j in JOBS.list(world_id=wid)][:10]}


@app.get("/api/worlds/{wid}/scene")
def world_scene(wid: str):
    v = viewer(wid)
    info = v.info()
    out = _rewrite(info, wid, None)
    try:
        geoms = v.world()[0]
    except Exception as e:  # noqa: BLE001
        geoms = {"geoms": [], "meshes": {}, "error": f"{type(e).__name__}: {e}"}
    out.update({"id": wid, "geoms": geoms, "geoms_bin_url": f"/api/worlds/{wid}/scene.bin",
                "viewer_url": f"/viewer/{wid}/-/"})
    return out


@app.get("/api/worlds/{wid}/scene.bin")
def world_scene_bin(wid: str):
    return Response(viewer(wid).world()[1], media_type="application/octet-stream")


@app.post("/api/worlds/{wid}/edit")
def edit_world(wid: str, body: dict = Body(...)):
    wd = world_dir(wid)
    prompt = (body or {}).get("prompt")
    if not prompt:
        raise HTTPException(400, "prompt required")
    job = JOBS.start("world_edit", [("edit world", [PY, "-m", "fallguy.world", "edit", "--world", str(wd),
                                                    "--prompt", str(prompt)])],
                     meta={"world_id": wid, "prompt": prompt})
    return {"job_id": job.id}


@app.post("/api/worlds/{wid}/verify")
def verify_world(wid: str, body: dict = Body(default={})):
    wd = world_dir(wid)
    argv = [PY, "-m", "fallguy.world", "verify", "--world", str(wd)] + (["--apply"] if (body or {}).get("apply") else [])
    job = JOBS.start("world_verify", [("verify world", argv)], meta={"world_id": wid})
    return {"job_id": job.id}


# ----------------------------------------------------------------------------- jobs
@app.get("/api/jobs")
def list_jobs(world: str = None, stunt: str = None, active: bool = False):
    filt = {k: v for k, v in (("world_id", world), ("stunt_id", stunt)) if v}
    js = JOBS.running(**filt) if active else JOBS.list(**filt)
    return [j.to_dict() for j in js[:50]]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j.to_dict()


# ----------------------------------------------------------------------------- stunts
def _stunt_dirs():
    return sorted((d for d in STUNTS.iterdir() if (d / "stunt.json").is_file()), key=_mtime, reverse=True) \
        if STUNTS.is_dir() else []


def _takes(sd):
    return sorted((d.name for d in sd.iterdir() if d.is_dir() and TAKE_RE.fullmatch(d.name)), key=_take_key)


def gate2(metrics, critique):
    if not isinstance(critique, dict):
        return None
    from .. import gates as G
    c = dict(critique)
    c.setdefault("gate2_pass", bool(critique.get("approve")))
    c.setdefault("style_score", critique.get("overall_score"))
    try:
        return G.evaluate_gate2(metrics or {}, c)
    except Exception as e:  # noqa: BLE001
        return {"passed": None, "reasons": [f"gate2 eval failed: {e}"], "final_score": None}


def loop_row(sd, take):
    td = sd / take
    m = rj(td / "metrics.json", {}) or {}
    c = rj(td / "critique.json")
    g1 = m.get("gate1") or {}
    bio = m.get("biomech") or {}
    knee = _num(bio, "knee_torque", "value")
    g2 = gate2(m, c)
    return {"take": take, "at": _iso(time.time()),
            "stunt_score": _num(m, "score", "stunt_score"),
            "head_g_peak": g1.get("head_g_peak", _num(bio, "head_g", "value")),
            "hic15": g1.get("hic15", _num(bio, "hic15", "value")),
            "knee_torque_peak_nm": abs(knee) if isinstance(knee, (int, float)) else None,
            "hip_torque_peak_nm": _num(bio, "hip_torque", "value"),
            "ankle_torque_peak_nm": _num(bio, "ankle_torque", "value"),
            "peak_grf_bw": _num(bio, "grf", "x_bw"),
            "neck_compression_n": _num(bio, "neck_compression", "value"),
            "lumbar_compression_n": _num(bio, "lumbar_compression", "value"),
            "lateral_g_peak": g1.get("lateral_g_peak"),
            "min_clearance_cm": g1.get("min_clearance_cm"),
            "landing_error_m": _num(g1, "landing", "error_m"),
            "hazard_collisions": g1.get("hazard_collisions"),
            "worst_link": _num(bio, "worst_link", "name"), "worst_link_index": _num(bio, "worst_link", "index"),
            "gate1_passed": g1.get("passed"), "gate1_failures": g1.get("failures"),
            "gate2_passed": (g2 or {}).get("passed"), "gate2_reasons": (g2 or {}).get("reasons"),
            "final_score": (g2 or {}).get("final_score"),
            "argon_style_score": (c or {}).get("style_score", (c or {}).get("overall_score")),
            "argon_approve": (c or {}).get("approve"),
            "critique_summary": (c or {}).get("summary")}


def append_loop(sd, take, event):
    if not (sd / take / "metrics.json").is_file():
        return None
    rows = rj(sd / "loop.json", []) or []
    row = {**loop_row(sd, take), "event": event, "iteration": len(rows) + 1}
    rows.append(row)
    wj(sd / "loop.json", rows)
    return row


def loop_rows(sd):
    rows = rj(sd / "loop.json")
    if isinstance(rows, list) and rows:
        return rows
    return [{**loop_row(sd, t), "event": "existing", "iteration": i + 1}
            for i, t in enumerate(t for t in _takes(sd) if (sd / t / "metrics.json").is_file())]


def stunt_summary(sd):
    st = rj(sd / "stunt.json", {}) or {}
    takes = _takes(sd)
    best = None
    for t in takes:
        m = rj(sd / t / "metrics.json")
        if not m:
            continue
        b = m.get("biomech") or {}
        knee = _num(b, "knee_torque", "value")
        row = {"take": t, "head_g": _num(m, "gate1", "head_g_peak"), "hic15": _num(m, "gate1", "hic15"),
               "knee_nm": abs(knee) if isinstance(knee, (int, float)) else None,
               "stunt_score": _num(m, "score", "stunt_score"), "gate1_passed": _num(m, "gate1", "passed")}
        if best is None or (row["stunt_score"] or -1) >= (best["stunt_score"] or -1):
            best = row
    return {"id": sd.name, "brief": st.get("brief"), "duration": st.get("duration_sec", st.get("duration")),
            "created": _iso(_mtime(sd / "stunt.json")), "world_id": _stunt_world_id(sd),
            "summary": _num(st, "stunt", "summary"), "questions": st.get("questions") or [],
            "clarifications": st.get("clarifications"), "takes": takes, "best": best,
            "jobs": [j.id for j in JOBS.running(stunt_id=sd.name)]}


@app.get("/api/stunts")
def list_stunts(world: str = None):
    out = [stunt_summary(sd) for sd in _stunt_dirs()]
    return [s for s in out if not world or s["world_id"] == world]


def _plan_job(wd, brief, duration, answers=None, parent=None):
    argv = [PY, "-m", "fallguy.stunt", "plan", "--world", str(wd), "--brief", brief, "--duration", str(duration),
            "--out", "stunts"]
    if answers:
        argv += ["--answers", answers]

    def on_done(job):
        sdir = (job.result or {}).get("stunt_dir") if isinstance(job.result, dict) else None
        if sdir:
            job.meta["stunt_id"] = Path(sdir).name
            job.result["stunt_id"] = Path(sdir).name
            if parent:
                try:
                    p = STUNTS / Path(sdir).name / "stunt.json"
                    st = rj(p, {})
                    st["parent_stunt"] = parent
                    wj(p, st)
                except Exception:
                    pass

    meta = {"world_id": wd.name, "brief": brief}
    if parent:
        meta["parent_stunt_id"] = parent
    return JOBS.start("stunt_plan", [("plan stunt", argv)], meta=meta, on_done=on_done)


def _answers_text(ans, questions=None):
    if ans is None:
        return ""
    if isinstance(ans, str):
        return ans
    if isinstance(ans, dict):
        return "; ".join(f"{k}: {v}" for k, v in ans.items() if v not in (None, ""))
    if isinstance(ans, list):
        parts = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                parts.append(f"{a.get('question', f'Q{i + 1}')}: {a.get('answer', '')}")
            elif a not in (None, ""):
                q = (questions[i].get("question") if questions and i < len(questions) and isinstance(questions[i], dict)
                     else f"Q{i + 1}")
                parts.append(f"{q}: {a}")
        return "; ".join(parts)
    return str(ans)


@app.post("/api/stunts")
def create_stunt(body: dict = Body(...)):
    wid, brief = body.get("world_id"), (body.get("brief") or "").strip()
    if not wid or not brief:
        raise HTTPException(400, "world_id and brief required")
    wd = world_dir(wid)
    dur = float(body.get("duration") or 6.0)
    job = _plan_job(wd, brief, dur, _answers_text(body["answers"]) if body.get("answers") else None)
    return {"job_id": job.id}


@app.post("/api/stunts/{sid}/answer")
def answer_stunt(sid: str, body: dict = Body(...)):
    sd = stunt_dir(sid)
    st = rj(sd / "stunt.json", {}) or {}
    wd = _stunt_world_dir(sd)
    if wd is None:
        raise HTTPException(400, "stunt has no world_dir")
    ans = _answers_text(body.get("answers"), st.get("questions"))
    if not ans:
        raise HTTPException(400, "answers required")
    prev = st.get("clarifications")
    if prev:
        ans = f"{prev}; {ans}"
    job = _plan_job(wd, body.get("brief") or st.get("brief"), float(body.get("duration") or st.get("duration_sec") or 6),
                    ans, parent=sid)
    return {"job_id": job.id}


@app.post("/api/stunts/{sid}/run")
def run_take(sid: str, body: dict = Body(default={})):
    sd = stunt_dir(sid)
    take = check_take((body or {}).get("take") or "v1")
    steps = [(f"simulate {take}", [PY, "-m", "fallguy.stunt", "simulate", "--stunt", str(sd), "--take", take])]
    if not (body or {}).get("no_critique"):
        steps.append((f"critique {take}", [PY, "-m", "fallguy.stunt", "critique", "--stunt", str(sd), "--take", take]))

    def on_done(job):
        if job.results and job.results[0]["rc"] == 0:
            row = append_loop(sd, take, "run")
            job.meta["loop_row"] = row

    job = JOBS.start("stunt_run", steps, meta={"stunt_id": sid, "world_id": _stunt_world_id(sd), "take": take},
                     on_done=on_done)
    return {"job_id": job.id}


@app.post("/api/stunts/{sid}/improve")
def improve_take(sid: str, body: dict = Body(default={})):
    sd = stunt_dir(sid)
    takes = _takes(sd)
    src = check_take((body or {}).get("from") or (takes[-1] if takes else "v1"))
    if not (sd / src / "critique.json").is_file():
        raise HTTPException(400, f"{src} has no critique yet; run it first")
    dst = check_take((body or {}).get("to") or f"v{max([_take_key(t) for t in takes] + [1]) + 1}")
    argv = [PY, "-m", "fallguy.stunt", "improve", "--stunt", str(sd), "--from", src, "--to", dst]

    def on_done(job):
        if (sd / dst / "metrics.json").is_file():
            job.meta["loop_row"] = append_loop(sd, dst, f"improve {src}->{dst}")

    job = JOBS.start("stunt_improve", [(f"improve {src} -> {dst}", argv)],
                     meta={"stunt_id": sid, "world_id": _stunt_world_id(sd), "from": src, "to": dst, "take": dst},
                     on_done=on_done)
    return {"job_id": job.id}


def _biomech_summary(td):
    b = rj(td / "biomech.json")
    if not isinstance(b, dict):
        return None
    return {k: b.get(k) for k in ("peaks", "worst_link", "flags", "links", "limits", "method", "body_mass_kg",
                                  "body_weight_n", "t_release")}


@app.get("/api/stunts/{sid}")
def get_stunt(sid: str):
    sd = stunt_dir(sid)
    takes = {}
    for t in _takes(sd):
        td = sd / t
        m, c = rj(td / "metrics.json"), rj(td / "critique.json")
        stills = sorted(p.name for p in td.glob("*.png")) + sorted(p.name for p in td.glob("*.jpg"))
        takes[t] = {"metrics": m, "biomech_summary": _biomech_summary(td), "critique": c,
                    "gate2": gate2(m, c) if c else None,
                    "params": rj(td / "params.json"), "changes": rj(td / "changes.json"),
                    "video_url": surl(sid, f"{t}/video.mp4") if (td / "video.mp4").is_file() else None,
                    "stills": [surl(sid, f"{t}/{s}") for s in stills],
                    "has_trajectory": (td / "trajectory.npz").is_file()}
    extra = {k: surl(sid, f) for k, f in (("map_url", "map.png"), ("world_view_url", "world_view.png"))
             if (sd / f).is_file()}
    return {"id": sid, "summary": stunt_summary(sd), "stunt": rj(sd / "stunt.json"), "takes": takes,
            "compare": rj(sd / "compare.json"), "loop": loop_rows(sd), **extra,
            "viewer_url": f"/viewer/{_stunt_world_id(sd) or '-'}/{sid}/",
            "jobs": [j.to_dict() for j in JOBS.list(stunt_id=sid)][:10]}


@app.get("/api/stunts/{sid}/loop")
def get_loop(sid: str):
    return loop_rows(stunt_dir(sid))


@app.get("/api/stunts/{sid}/trajectory")
def stunt_trajectory(sid: str, take: str = "v1"):
    sd = stunt_dir(sid)
    check_take(take)
    v = viewer(None, sid)
    try:
        tr = dict(v.traj(take))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    bio = v.biomech(take)
    raw = rj(sd / take / "biomech.json", {}) or {}
    tr["stress"] = bio.get("stress") or {}
    tr["stress_series"] = bio.get("stress_series") or {}
    tr["stress_t"] = (bio.get("series") or {}).get("t")
    tr["stress_colour"] = {k: d.get("colour") for k, d in (raw.get("links") or {}).items() if isinstance(d, dict)}
    tr["biomech_series"] = bio.get("series")
    return tr


@app.get("/api/stunts/{sid}/performer")
def stunt_performer(sid: str, take: str = "v1"):
    stunt_dir(sid)
    return viewer(None, sid).performer(check_take(take))[0]


@app.get("/api/stunts/{sid}/performer.bin")
def stunt_performer_bin(sid: str, take: str = "v1"):
    stunt_dir(sid)
    return Response(viewer(None, sid).performer(check_take(take))[1], media_type="application/octet-stream")


@app.get("/api/stunts/{sid}/biomech")
def stunt_biomech(sid: str, take: str = "v1"):
    stunt_dir(sid)
    return viewer(None, sid).biomech(check_take(take))


# ----------------------------------------------------------------------------- stock viewer page, per world/stunt
@app.get("/viewer/{wid}/{sid}/", response_class=HTMLResponse)
def viewer_page(wid: str, sid: str):
    html = (VIEWER_STATIC / "index.html").read_text()
    return html.replace('"/static/app.js', '"app.js')


@app.get("/viewer/{wid}/{sid}/app.js")
def viewer_js(wid: str, sid: str):
    base = f"/viewer/{wid}/{sid}/api/"
    js = (VIEWER_STATIC / "app.js").read_text().replace("'/api/", f"'{base}").replace("`/api/", f"`{base}")
    return Response(js, media_type="text/javascript")


@app.get("/viewer/{wid}/{sid}/api/{name}")
def viewer_api(wid: str, sid: str, name: str, take: str = "v1"):
    w = None if wid == "-" else wid
    s = None if sid == "-" else sid
    if s:
        stunt_dir(s)
    v = viewer(w, s)
    wid_eff = w or (v.world_dir.name if v.world_dir else "-")
    try:
        if name == "info":
            return _rewrite(v.info(), wid_eff, s)
        if name == "world":
            return v.world()[0]
        if name == "world.bin":
            return Response(v.world()[1], media_type="application/octet-stream")
        if name == "performer":
            return v.performer(check_take(take))[0]
        if name == "performer.bin":
            return Response(v.performer(check_take(take))[1], media_type="application/octet-stream")
        if name == "traj":
            return v.traj(check_take(take))
        if name == "biomech":
            return v.biomech(check_take(take))
        if name == "critique":
            return v.critique(check_take(take))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    raise HTTPException(404, f"unknown viewer api {name}")


# ----------------------------------------------------------------------------- static files
@app.exception_handler(Exception)
async def _err(request, exc):  # noqa: ARG001
    import traceback
    traceback.print_exc()
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.mount("/files/worlds", StaticFiles(directory=str(WORLDS), check_dir=False), name="worlds")
app.mount("/files/stunts", StaticFiles(directory=str(STUNTS), check_dir=False), name="stunts")
app.mount("/files/uploads", StaticFiles(directory=str(UPLOADS), check_dir=False), name="uploads")


@app.get("/static/{rel:path}")
def static_file(rel: str):
    """app static first, then the viewer's static dir (vendored three.js at /static/vendor/...)."""
    for base in (STATIC, VIEWER_STATIC):
        p = (base / rel).resolve()
        if str(p).startswith(str(base.resolve())) and p.is_file():
            return FileResponse(p)
    raise HTTPException(404, f"not found: {rel}")


@app.get("/")
def index():
    p = STATIC / "index.html"
    if p.is_file():
        return FileResponse(p, headers={"Cache-Control": "no-store"})
    return HTMLResponse("<h1>FALLGUY</h1><p>frontend not built yet (fallguy/app/static/index.html). "
                        "API at <a href=/api/worlds>/api/worlds</a>.</p>")


app.mount("/", StaticFiles(directory=str(STATIC), check_dir=False), name="app_static")
