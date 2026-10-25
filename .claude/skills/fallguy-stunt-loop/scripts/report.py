#!/usr/bin/env python3
"""report.md, the safety certificate and the status table of a FALLGUY run.

loop.py finalize calls build_certificate() and write_report(); every wave refreshes report.md.
`python report.py --run-dir DIR` rewrites report.md from run.json and the files in the run dir
and prints one JSON line {"report": path}.
"""
import argparse
import datetime as dt
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

NEXT_FOR_STATUS = {"planned": "wave", "running": "wave", "checkpoint": "checkpoint", "passed": "finalize",
                   "finalized": "gate3"}


# ----------------------------------------------------------------------------- formatting
def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _f(v, nd=1, dash="-"):
    v = _num(v)
    return dash if v is None else f"{v:.{nd}f}"


def _pf(v):
    return "-" if v is None else ("PASS" if v else "FAIL")


def _cell(text):
    """Text safe for one Markdown table cell."""
    return str(text if text is not None else "-").replace("|", "\\|").replace("\n", " ").strip() or "-"


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _wave_dir(n):
    return f"waves/wave{int(n):02d}"


def _notes(run):
    return [n.get("text") for n in run.get("director_notes") or [] if isinstance(n, dict) and n.get("text")]


# ----------------------------------------------------------------------------- certificate
def build_certificate(run_dir, run, cid, wave, params, metrics, critique, gate2, final_video, critique_rel):
    """The safety certificate of the finalized take: Gate-1 numbers from MuJoCo, Gate-2 verdict from
    Argon's critique, overall PASS only when both gates pass. Gate 3 stays with the director."""
    g1 = dict(metrics.get("gate1") or {})
    g1_passed = bool(metrics.get("ok")) and bool(g1.get("passed"))
    g2_passed = bool(gate2.get("passed"))
    passed = g1_passed and g2_passed
    thresholds = weights = None
    try:
        import gates
        thresholds, weights = dict(gates.THRESHOLDS), dict(gates.WEIGHTS)
    except Exception:  # the certificate still carries the verdicts
        pass
    critique = critique or {}
    return {
        "certificate": "FALLGUY stunt safety certificate",
        "version": 1,
        "verdict": "PASS" if passed else "FAIL",
        "passed": passed,
        "generated_at": _now(),
        "brief": run.get("brief"),
        "duration_sec": run.get("duration_sec"),
        "run": {"run_dir": run_dir, "slug": run.get("slug"), "tag": run.get("tag"), "plan": run.get("plan"),
                "script": run.get("script"), "media": run.get("media"), "media_original": run.get("media_original"),
                "performer": run.get("performer"), "stunt_type": run.get("stunt_type"),
                "waves": len(run.get("waves") or []), "director_notes": _notes(run)},
        "candidate": {"id": cid, "wave": wave, "label": params.get("label"), "camera": params.get("camera"),
                      "why": params.get("why")},
        "params": params,
        "gate1": {"passed": g1_passed, "failures": list(g1.get("failures") or []),
                  "metrics": {k: v for k, v in g1.items() if k not in ("passed", "failures")},
                  "thresholds": thresholds, "source": "MuJoCo simulation of the final take (ground truth)"},
        "score": metrics.get("score"),
        "weights": weights,
        "sim": metrics.get("sim"),
        "gate2": {"passed": g2_passed, "final_score": gate2.get("final_score"), "reasons": list(gate2.get("reasons") or []),
                  "argon_gate2_pass": critique.get("gate2_pass"), "style_score": critique.get("style_score"),
                  "critique_summary": critique.get("summary"), "brief_match": critique.get("brief_match"),
                  "rotation_observed_deg": critique.get("rotation_observed_deg"), "framing": critique.get("framing"),
                  "unnatural_motion": critique.get("unnatural_motion") or [], "critique": critique_rel,
                  "model": os.environ.get("ARGON_MODEL", "gemini-4-argon")},
        "gate3": {"status": "pending", "decided_by": "the human stunt director, after watching the final video"},
        "final_video": final_video,
    }


# ----------------------------------------------------------------------------- status table (text)
def status_table(run, run_dir=None, busy=None):
    """Compact fixed-width convergence table for `loop.py status`."""
    status = run.get("status")
    out = [f"run     {run_dir or ''}",
           f"brief   {run.get('brief')}",
           f"status  {status}   waves {len(run.get('waves') or [])} (checkpoint every {run.get('checkpoint_every')})   "
           f"next {NEXT_FOR_STATUS.get(status, '-')}   script {run.get('script')}"]
    if busy:
        out.append(f"busy    `loop.py {busy.get('command')}` is running (pid {busy.get('pid')}, since {busy.get('since')})")
    out.append("")
    out.append(f"{'wave':>4}  {'best':<7}{'score':>6}{'final':>7}{'headG':>7}{'clr cm':>8}{'land m':>8}"
               f"{'frame%':>8}{'style':>7}  {'gate1':<6}{'gate2':<6}steer")
    for w in run.get("waves") or []:
        b = w.get("best") or {}
        out.append(f"{w.get('wave'):>4}  {str(w.get('best_id')):<7}{_f(w.get('stunt_score')):>6}{_f(w.get('final_score')):>7}"
                   f"{_f(b.get('head_g_peak')):>7}{_f(b.get('min_clearance_cm')):>8}{_f(b.get('landing_error_m'), 2):>8}"
                   f"{_f(b.get('framing_pct'), 0):>8}{_f(w.get('style_score'), 0):>7}  "
                   f"{_pf(w.get('gate1_passed')):<6}{_pf(w.get('gate2_passed')):<6}{w.get('steer') or '-'}")
    if not run.get("waves"):
        out.append("   (no waves yet)")
    best = run.get("best")
    if best:
        out.append("")
        out.append(f"best    {best.get('id')} (wave {best.get('wave')}): stunt score {_f(best.get('stunt_score'))}, "
                   f"final score {_f(best.get('final_score'))}, gate 1 {_pf(best.get('gate1_passed'))}, "
                   f"gate 2 {_pf(best.get('gate2_passed'))}")
    final = run.get("final")
    if final:
        out.append(f"final   {final.get('video')} ({final.get('verdict')}, {final.get('candidate')})"
                   + ("  [stale: a wave ran after finalize]" if final.get("stale") else ""))
    for i, n in enumerate(n for n in run.get("director_notes") or [] if isinstance(n, dict)):
        out.append(f"{'notes' if i == 0 else '':<8}from wave {int(n.get('after_wave') or 0) + 1}: \"{n.get('text')}\"")
    return "\n".join(out)


# ----------------------------------------------------------------------------- report.md
def _check_rows(g1, th):
    """(check, value, limit, result) rows for the Gate 1 table."""
    th = th or {}
    land = g1.get("landing") or {}
    rows = []

    def row(name, value, limit, ok):
        rows.append((name, value, limit, "-" if ok is None else ("PASS" if ok else "FAIL")))

    hg, lim = _num(g1.get("head_g_peak")), _num(th.get("head_g_max"))
    row("Peak head G (15 ms)", f"{_f(hg)} G at {_f(g1.get('head_g_peak_t'), 2)} s",
        f"<= {_f(lim)} G" if lim is not None else "-", None if hg is None or lim is None else hg <= lim)
    row("HIC15", _f(g1.get("hic15"), 0), "-", None)
    hc, lim = _num(g1.get("hazard_collisions")), _num(th.get("hazard_collisions_max"))
    contacts = ", ".join(f"{c.get('body')} x {c.get('object')} at {_f(c.get('t'), 2)} s"
                         for c in (g1.get("hazard_contacts") or [])[:4] if isinstance(c, dict))
    row("Hazard collisions", _f(hc, 0) + (f" ({contacts})" if contacts else ""), f"<= {_f(lim, 0)}" if lim is not None else "-",
        None if hc is None or lim is None else hc <= lim)
    mc, lim = _num(g1.get("min_clearance_cm")), _num(th.get("min_clearance_cm"))
    at = g1.get("min_clearance_at") or {}
    where = f" ({at.get('body')} to {at.get('object')} at {_f(at.get('t'), 2)} s)" if isinstance(at, dict) and at.get("body") else ""
    row("Minimum clearance", f"{_f(mc)} cm{where}", f">= {_f(lim)} cm" if lim is not None else "-",
        None if mc is None or lim is None else mc >= lim)
    le, lim = _num(land.get("error_m")), _num(th.get("landing_error_max_m"))
    row("Landing error", f"{_f(le, 2)} m on {land.get('surface', '-')}", f"<= {_f(lim, 2)} m" if lim is not None else "-",
        None if le is None or lim is None else le <= lim)
    inside = land.get("inside_mat")
    row("Lands inside the mat", "-" if inside is None else ("yes" if inside else "no"),
        "required" if th.get("require_inside_mat") else "-", None if inside is None or not th.get("require_inside_mat") else bool(inside))
    fp, lim = _num(g1.get("framing_pct")), _num(th.get("framing_min_pct"))
    row("Framing (sim)", f"{_f(fp, 0)} %", f">= {_f(lim, 0)} %" if lim is not None else "-",
        None if fp is None or lim is None else fp >= lim)
    rot = g1.get("rotation_deg") or {}
    if rot:
        row("Rotation (take-off to landing)", ", ".join(f"{k} {_f(v, 0)} deg" for k, v in rot.items()), "-", None)
    return rows


def _critique_section(critique, title):
    out = [f"## {title}", ""]
    if not isinstance(critique, dict):
        return out + ["No Argon critique yet.", ""]
    if critique.get("summary"):
        out += [str(critique["summary"]).strip(), ""]
    bm = critique.get("brief_match") or {}
    fr = critique.get("framing") or {}
    out += [f"- Brief match: {'yes' if bm.get('matches') else 'no'}"
            + (f" (missing: {', '.join(map(str, bm.get('missing') or []))})" if bm.get("missing") else ""),
            f"- Style score: {_f(critique.get('style_score'), 0)} / 100; Argon's own Gate 2 call: "
            f"{_pf(critique.get('gate2_pass'))}",
            f"- Rotation Argon observed: {_f(critique.get('rotation_observed_deg'), 0)} deg",
            f"- Framing Argon observed: {_f(fr.get('in_frame_pct'), 0)} % in frame, subject {fr.get('subject_size', '-')}"
            + (f"; {fr.get('notes')}" if fr.get("notes") else ""), ""]
    events = sorted((e for e in critique.get("events") or [] if isinstance(e, dict)), key=lambda e: _num(e.get("t_sec")) or 0.0)
    out += ["| t (s) | Event | Note |", "|---:|---|---|"]
    out += [f"| {_f(e.get('t_sec'), 2)} | {_cell(e.get('event'))} | {_cell(e.get('note'))} |" for e in events]
    if not events:
        out.append("| - | - | no events reported |")
    out.append("")
    um = [u for u in critique.get("unnatural_motion") or [] if isinstance(u, dict)]
    if um:
        out += ["Unnatural motion Argon flagged:", ""]
        out += [f"- {_f(u.get('t_sec'), 2)} s, {u.get('severity', '-')}: {u.get('issue')}" for u in um]
        out.append("")
    pc = [c for c in critique.get("proposed_changes") or [] if isinstance(c, dict)]
    if pc:
        out += ["Changes Argon proposed for the next wave:", ""]
        out += [f"- `{c.get('param')}` {c.get('change')}" + (f" to {c.get('value')}" if c.get("value") is not None else "")
                + f": {c.get('reason')}" for c in pc]
        out.append("")
    return out


def render_report(run_dir, run):
    """report.md: run facts, convergence table, all candidates, steers, final take, critique timeline, artifacts."""
    waves = run.get("waves") or []
    final = run.get("final") or None
    cert = _read(os.path.join(run_dir, final["certificate"])) if final and final.get("certificate") else None
    best = run.get("best") or {}
    out = ["# FALLGUY stunt report", "", f"> {run.get('brief')}", ""]
    out += ["| | |", "|---|---|",
            f"| Run | `{run_dir}` |",
            f"| Status | **{run.get('status')}** after {len(waves)} wave(s) (director checkpoint every {run.get('checkpoint_every')} waves) |",
            f"| Final length | {_f(run.get('duration_sec'), 1)} s |",
            f"| Site media | `{_cell(run.get('media_original'))}` ({run.get('media_kind')}) |",
            f"| Stunt / performer | {run.get('stunt_type')} / Unitree H1 |",
            f"| Stunt script | `{run.get('script')}` |",
            f"| Report written | {_now()} |", ""]
    if cert:
        out += [f"**Verdict: {cert.get('verdict')}** for take {cert['candidate']['id']} "
                f"\"{_cell(cert['candidate'].get('label'))}\": Gate 1 {_pf(cert['gate1']['passed'])}, "
                f"Gate 2 {_pf(cert['gate2']['passed'])}, final score {_f(cert['gate2'].get('final_score'))}. "
                f"Gate 3 is the director's call." + (" (Stale: a wave ran after this finalize.)" if final.get("stale") else ""), ""]
    elif best:
        out += [f"Not finalized yet. Best so far: {best.get('id')} (wave {best.get('wave')}), stunt score "
                f"{_f(best.get('stunt_score'))}, Gate 1 {_pf(best.get('gate1_passed'))}, Gate 2 {_pf(best.get('gate2_passed'))}.", ""]

    out += ["## Convergence", "",
            "| Wave | Best | Stunt score | Final score | Head G | Clearance (cm) | Landing error (m) | Framing (%) | Style | Gate 1 | Gate 2 | Steer |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---|"]
    for w in waves:
        b = w.get("best") or {}
        out.append(f"| {w.get('wave')} | {w.get('best_id')} | {_f(w.get('stunt_score'))} | {_f(w.get('final_score'))} | "
                   f"{_f(b.get('head_g_peak'))} | {_f(b.get('min_clearance_cm'))} | {_f(b.get('landing_error_m'), 2)} | "
                   f"{_f(b.get('framing_pct'), 0)} | {_f(w.get('style_score'), 0)} | {_pf(w.get('gate1_passed'))} | "
                   f"{_pf(w.get('gate2_passed'))} | {_cell(w.get('steer'))} |")
    if not waves:
        out.append("| - | no waves yet | | | | | | | | | | |")
    out.append("")
    if waves:
        out += ["<details><summary>Every candidate</summary>", "",
                "| Wave | Id | Label | Camera | Sim | Stunt score | Head G | Clearance (cm) | Landing error (m) | Framing (%) | Gate 1 | Failures |",
                "|---:|---|---|---|:-:|---:|---:|---:|---:|---:|:-:|---|"]
        for w in waves:
            for c in w.get("candidates") or []:
                if not c.get("ok"):
                    out.append(f"| {w.get('wave')} | {c.get('id')} | {_cell(c.get('label'))} | {c.get('camera')} | FAIL | | | | | | | "
                               f"{_cell(c.get('error'))} |")
                    continue
                out.append(f"| {w.get('wave')} | {c.get('id')} | {_cell(c.get('label'))} | {c.get('camera')} | ok | "
                           f"{_f(c.get('stunt_score'))} | {_f(c.get('head_g_peak'))} | {_f(c.get('min_clearance_cm'))} | "
                           f"{_f(c.get('landing_error_m'), 2)} | {_f(c.get('framing_pct'), 0)} | {_pf(c.get('gate1_passed'))} | "
                           f"{_cell('; '.join(map(str, c.get('failures') or [])) or '-')} |")
        out += ["", "</details>", ""]

    notes = [n for n in run.get("director_notes") or [] if isinstance(n, dict)]
    out += ["## Director notes", "", "Standing feedback from the director; Argon sees every note in later proposals and critiques.", ""]
    out += [f"{i}. From wave {int(n.get('after_wave') or 0) + 1}: \"{_cell(n.get('text'))}\"" for i, n in enumerate(notes, 1)] or ["None."]
    out.append("")

    critique, crit_title = None, "Argon critique timeline"
    if cert:
        c = cert["candidate"]
        out += [f"## Final take: {c['id']} \"{_cell(c.get('label'))}\"", ""]
        if c.get("why"):
            out += [f"Why Argon proposed it: {c['why']}", ""]
        out += ["| Check | Value | Limit | Result |", "|---|---|---|:-:|"]
        out += [f"| {n} | {_cell(v)} | {_cell(lim)} | {r} |" for n, v, lim, r in
                _check_rows(cert["gate1"].get("metrics") or {}, cert["gate1"].get("thresholds"))]
        out.append("")
        if cert["gate1"].get("failures"):
            out += ["Gate 1 failures: " + "; ".join(map(str, cert["gate1"]["failures"])), ""]
        sc = cert.get("score") or {}
        if sc:
            out += ["Score: " + ", ".join(f"{k} {_f(v)}" for k, v in sc.items()), ""]
        g2 = cert["gate2"]
        out += [f"Gate 2: {_pf(g2.get('passed'))}, final score {_f(g2.get('final_score'))}, style {_f(g2.get('style_score'), 0)}"
                + (". Reasons: " + "; ".join(map(str, g2.get("reasons"))) if g2.get("reasons") else "."), ""]
        params = cert.get("params") or {}
        skip = {"id", "label", "color", "why"}
        out += ["<details><summary>Parameters</summary>", "", "| Param | Value |", "|---|---|"]
        out += [f"| {k} | {_cell(v)} |" for k, v in params.items() if k not in skip]
        out += ["", "</details>", ""]
        critique = _read(os.path.join(run_dir, g2.get("critique") or ""))
        crit_title = f"Argon critique timeline ({c['id']})"
    elif best.get("wave"):
        critique = _read(os.path.join(run_dir, _wave_dir(best["wave"]), "critique.json"))
        crit_title = f"Argon critique timeline (best so far, {best.get('id')})"
    out += _critique_section(critique, crit_title)

    out += ["## Artifacts", ""]
    arts = []
    if final:
        arts += [("Final video", final.get("video")), ("Certificate", final.get("certificate"))]
    arts += [("Stunt script", run.get("script")), ("Scene plan", run.get("plan")), ("Preview still", run.get("preview")),
             ("Reference frame", run.get("ref_frame")), ("Site media copy", run.get("media")), ("Run state", "run.json"),
             ("Argon call log", "logs/argon_calls.jsonl")]
    for w in waves:
        b = w.get("best") or {}
        arts.append((f"Wave {w.get('wave')} best render", b.get("video")))
    for name, rel in arts:
        if rel and os.path.exists(os.path.join(run_dir, rel)):
            out.append(f"- {name}: [`{rel}`]({rel})")
    if run.get("preview") and os.path.exists(os.path.join(run_dir, run["preview"])):
        out += ["", f"![Preview from the match camera]({run['preview']})"]
    out.append("")
    return "\n".join(out)


def write_report(run_dir, run):
    path = os.path.join(run_dir, "report.md")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(render_report(run_dir, run))
    os.replace(tmp, path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rewrite report.md of a FALLGUY run from run.json.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    run = _read(os.path.join(run_dir, "run.json"))
    if not isinstance(run, dict):
        sys.stderr.write(f"report.py: error: no readable run.json in {run_dir}\n")
        print(json.dumps({"error": f"no readable run.json in {run_dir}", "exit_code": 1}))
        return 1
    path = write_report(run_dir, run)
    sys.stderr.write(f"wrote {path}\n")
    print(json.dumps({"report": path}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
