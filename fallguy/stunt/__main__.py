"""python -m fallguy.stunt {plan, simulate, critique, improve, run} (CONTRACT.md stages 2-4)."""
import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fallguy.stunt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="Argon writes the choreography for the brief in the world")
    p.add_argument("--world", required=True)
    p.add_argument("--brief", required=True)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--out", default="stunts")
    p.add_argument("--answers", default=None, help="director's answers to the plan's questions")
    s = sub.add_parser("simulate", help="simulate + biomechanics + render one take")
    s.add_argument("--stunt", required=True)
    s.add_argument("--take", default="v1")
    s.add_argument("--no-render", action="store_true")
    c = sub.add_parser("critique", help="Argon watches the take's video with the numbers")
    c.add_argument("--stunt", required=True)
    c.add_argument("--take", default="v1")
    i = sub.add_parser("improve", help="apply the critique, simulate and critique the next take, compare")
    i.add_argument("--stunt", required=True)
    i.add_argument("--from", dest="src", default="v1")
    i.add_argument("--to", dest="dst", default="v2")
    i.add_argument("--no-critique", action="store_true")
    r = sub.add_parser("run", help="plan -> v1 -> critique -> v2 -> critique, in one go")
    r.add_argument("--world", required=True)
    r.add_argument("--brief", required=True)
    r.add_argument("--duration", type=float, default=6.0)
    r.add_argument("--out", default="stunts")
    r.add_argument("--answers", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        from .plan import plan
        out = plan(a.world, a.brief, a.duration, a.out, a.answers)
    elif a.cmd == "simulate":
        from .simulate import run_take
        out = run_take(a.stunt, a.take, render=not a.no_render)
    elif a.cmd == "critique":
        from .critique import critique
        out = critique(a.stunt, a.take)
        out = {k: out.get(k) for k in ("take", "overall_score", "approve", "summary", "changes")}
    elif a.cmd == "improve":
        from .improve import improve
        out = improve(a.stunt, a.src, a.dst, critique_after=not a.no_critique)
    else:
        from .critique import critique
        from .improve import improve
        from .plan import plan
        from .simulate import run_take
        pl = plan(a.world, a.brief, a.duration, a.out, a.answers)
        sd = pl["stunt_dir"]
        run_take(sd, "v1")
        critique(sd, "v1")
        cmp_ = improve(sd, "v1", "v2")
        out = {"stunt_dir": sd, "questions": pl["questions"], "compare": os.path.join(sd, "compare.json"),
               "critique": cmp_.get("critique")}
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
