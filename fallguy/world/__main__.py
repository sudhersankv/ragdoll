"""python -m fallguy.world {build,verify,edit} ... (see fallguy/CONTRACT.md, Stage 1)."""
import argparse
import json
import os
import sys

from fallguy import argon as A


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fallguy.world")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="reconstruct + Argon layout + verify (<= 2 fix rounds)")
    b.add_argument("--media", required=True)
    b.add_argument("--out", default="worlds")
    b.add_argument("--rounds", type=int, default=2)
    b.add_argument("--target", type=float, default=80.0)
    b.add_argument("--fov", type=float, default=None, help="degrees across the long image side (default iPhone prior)")
    b.add_argument("--layout", default=None, help="reuse an Argon layout JSON (logs/argon_layout.json) instead of asking")
    v = sub.add_parser("verify", help="one Argon render-vs-video critique")
    v.add_argument("--world", required=True)
    v.add_argument("--apply", action="store_true", help="also apply the corrections and re-render")
    e = sub.add_parser("edit", help="director prompt -> world changes, rebuild, re-render")
    e.add_argument("--world", required=True)
    e.add_argument("--prompt", required=True)
    r = sub.add_parser("render", help="re-render views from world.json (no Argon)")
    r.add_argument("--world", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            from fallguy.world import build as B, reconstruct as R
            wdir, world = B.build(a.media, a.out, a.rounds, a.target, a.fov or R.DEFAULT_LONG_FOV, layout_json=a.layout)
            vs = world.get("verify") or []
            last = sorted(world["views"])[-1]
            print(json.dumps({"world_dir": wdir, "objects": len(world["objects"]), "triangles": world["mesh"]["triangles"],
                              "scores": [x["score"] for x in vs],
                              "side_by_side": os.path.join(wdir, world["views"][last]["side_by_side"]),
                              "seconds": world.get("build_seconds")}))
        elif a.cmd == "verify":
            from fallguy.world import verify as V
            n = len(json.load(open(os.path.join(a.world, "world.json"))).get("verify") or []) + 1
            rec = V.verify(os.path.abspath(a.world), n, apply=a.apply)
            print(json.dumps({"world_dir": os.path.abspath(a.world), "round": n, "score": rec["score"],
                              "issues": len(rec["issues"] or []), "fixes": rec["fixes"]}))
        elif a.cmd == "edit":
            from fallguy.world import edit as E
            rec = E.edit(os.path.abspath(a.world), a.prompt)
            print(json.dumps({"world_dir": os.path.abspath(a.world), "edit": rec["n"], "summary": rec["summary"],
                              "applied": rec["applied"]}))
        elif a.cmd == "render":
            from fallguy.world import build as B
            w = B.read_json(os.path.join(a.world, "world.json"))
            B.save_world(w, a.world)
            print(json.dumps(B.render_views(w, a.world, "rr")))
    except A.ArgonError as ex:
        print(f"world: Argon error: {A.redact(str(ex))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
