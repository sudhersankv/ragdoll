"""python -m fallguy.app [--host 127.0.0.1] [--port 8800]"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="python -m fallguy.app", description="FALLGUY web app (API + frontend)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    import uvicorn
    print(f"FALLGUY app: http://{a.host}:{a.port}/", flush=True)
    try:
        uvicorn.run("fallguy.app.server:app", host=a.host, port=a.port, reload=a.reload, log_level="warning")
    finally:
        try:
            from .server import JOBS
            JOBS.kill_all()
        except Exception:
            pass


if __name__ == "__main__":
    main()
