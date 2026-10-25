"""python -m fallguy.viewer --world DIR [--stunt DIR] [--port 8765] [--no-open]"""
import argparse

from .server import serve


def main():
    ap = argparse.ArgumentParser(prog="python -m fallguy.viewer", description="FALLGUY local WebGL viewer")
    ap.add_argument("--world", help="worlds/<slug>-<ts> dir (world.json, scene.xml, mesh/env.glb, frames/)")
    ap.add_argument("--stunt", help="stunts/<slug>-<ts> dir (stunt.json, v1/, v2/ ...)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    a = ap.parse_args()
    if not a.world and not a.stunt:
        ap.error("need --world and/or --stunt")
    serve(a.world, a.stunt, host=a.host, port=a.port, open_browser=not a.no_open)


if __name__ == "__main__":
    main()
