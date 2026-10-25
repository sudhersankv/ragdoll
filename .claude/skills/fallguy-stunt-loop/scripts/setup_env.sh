#!/usr/bin/env bash
# FALLGUY stunt loop setup: find (or make) a Python with mujoco, numpy, imageio, imageio-ffmpeg and
# Pillow, find (or download) the Unitree H1 model, and check the Argon key without printing it.
#
# stdout gets exactly two lines, everything else goes to stderr:
#   FALLGUY_PYTHON=<python>
#   FALLGUY_H1_DIR=<dir with h1.xml>
# Use:  eval "$(bash setup_env.sh)" && export FALLGUY_PYTHON FALLGUY_H1_DIR
# Exits 1 with a specific message when something is missing.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$SKILL_DIR/../../.." && pwd)"
REQ="$SKILL_DIR/scripts/requirements.txt"
MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie"

say() { printf '%s\n' "$*" >&2; }
die() { printf 'setup_env.sh: error: %s\n' "$*" >&2; exit 1; }

has_modules() {  # $1: python. True when every runtime package imports and the bundled ffmpeg exists.
  "$1" - >/dev/null 2>&1 <<'EOF'
import mujoco, numpy, imageio, imageio_ffmpeg, PIL
import os
assert os.path.isfile(imageio_ffmpeg.get_ffmpeg_exe())
EOF
}

# ------------------------------------------------------------------ Python
PY=""
if [ -n "${FALLGUY_PYTHON:-}" ]; then
  command -v "$FALLGUY_PYTHON" >/dev/null 2>&1 || die "FALLGUY_PYTHON=$FALLGUY_PYTHON is not an executable"
  has_modules "$FALLGUY_PYTHON" || die "FALLGUY_PYTHON=$FALLGUY_PYTHON cannot import mujoco, numpy, imageio, imageio_ffmpeg and PIL; run: $FALLGUY_PYTHON -m pip install -r $REQ (or unset FALLGUY_PYTHON)"
  PY="$FALLGUY_PYTHON"
elif [ -x "$REPO/.venv/bin/python" ] && has_modules "$REPO/.venv/bin/python"; then
  PY="$REPO/.venv/bin/python"
else
  VENV="$REPO/.venv"
  if [ ! -x "$VENV/bin/python" ]; then
    say "creating $VENV ..."
    if command -v uv >/dev/null 2>&1; then
      uv venv --python 3.12 "$VENV" >&2 || uv venv "$VENV" >&2 || die "uv could not create $VENV"
    else
      command -v python3 >/dev/null 2>&1 || die "neither uv nor python3 is installed; install uv (https://docs.astral.sh/uv/) or Python 3.10+"
      python3 -m venv "$VENV" >&2 || die "python3 -m venv could not create $VENV"
    fi
  fi
  say "installing $REQ into $VENV ..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" -r "$REQ" >&2 || die "uv pip install -r $REQ failed (see above)"
  else
    "$VENV/bin/python" -m pip install -r "$REQ" >&2 || die "pip install -r $REQ failed (see above)"
  fi
  has_modules "$VENV/bin/python" || die "the packages still do not import in $VENV/bin/python"
  PY="$VENV/bin/python"
fi
MJ_VERSION="$("$PY" -c 'import mujoco; print(mujoco.__version__)')"
[ "$MJ_VERSION" = "3.17.0" ] || say "warning: $PY has mujoco $MJ_VERSION; the skill is tested with 3.17.0"

GL_ERR="$("$PY" - 2>&1 <<'EOF' || true
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size="0.1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 48, 64)
r.update_scene(d)
assert r.render().shape == (48, 64, 3)
print("GL_OK")
EOF
)"
case "$GL_ERR" in
  *GL_OK*) ;;
  *) die "MuJoCo offscreen rendering fails with $PY (on headless Linux set MUJOCO_GL=egl or osmesa): $(printf '%s' "$GL_ERR" | tail -n 1)" ;;
esac

# ------------------------------------------------------------------ Unitree H1 model
h1_ok() { [ -f "$1/h1.xml" ] && [ -d "$1/assets" ]; }
H1=""
if [ -n "${FALLGUY_H1_DIR:-}" ]; then
  h1_ok "$FALLGUY_H1_DIR" || die "FALLGUY_H1_DIR=$FALLGUY_H1_DIR has no h1.xml and assets/ (point it at mujoco_menagerie/unitree_h1)"
  H1="$FALLGUY_H1_DIR"
else
  for d in "$REPO/models/mujoco_menagerie/unitree_h1" "$PWD/models/mujoco_menagerie/unitree_h1" \
           "$HOME/.cache/fallguy/mujoco_menagerie/unitree_h1"; do
    if h1_ok "$d"; then H1="$d"; break; fi
  done
  if [ -z "$H1" ]; then
    CACHE="$HOME/.cache/fallguy/mujoco_menagerie"
    command -v git >/dev/null 2>&1 || die "the Unitree H1 model is missing and git is not installed to fetch it; set FALLGUY_H1_DIR"
    say "downloading the Unitree H1 model (blob-less sparse clone of mujoco_menagerie) into $CACHE ..."
    rm -rf "$CACHE"
    mkdir -p "$(dirname "$CACHE")"
    git clone --depth 1 --filter=blob:none --sparse "$MENAGERIE_URL" "$CACHE" >&2 \
      || die "git clone of $MENAGERIE_URL failed; check the network or set FALLGUY_H1_DIR"
    git -C "$CACHE" sparse-checkout set unitree_h1 >&2 || die "git sparse-checkout of unitree_h1 failed in $CACHE"
    h1_ok "$CACHE/unitree_h1" || die "the clone in $CACHE has no unitree_h1/h1.xml"
    H1="$CACHE/unitree_h1"
  fi
fi
H1_ERR="$("$PY" -c 'import sys, mujoco; mujoco.MjModel.from_xml_path(sys.argv[1]); print("H1_OK")' "$H1/h1.xml" 2>&1 || true)"
case "$H1_ERR" in
  *H1_OK*) ;;
  *) die "MuJoCo cannot load $H1/h1.xml: $(printf '%s' "$H1_ERR" | tail -n 1)" ;;
esac

# ------------------------------------------------------------------ Argon key (never printed)
TOKEN="${FALLGUY_ARGON_TOKEN_FILE:-$HOME/.argon_token}"
if [ -r "$TOKEN" ] && grep -q '[^[:space:]]' "$TOKEN" 2>/dev/null; then
  MODE="$(stat -f '%Lp' "$TOKEN" 2>/dev/null || stat -c '%a' "$TOKEN" 2>/dev/null || echo unknown)"
  case "$MODE" in
    600|400) ;;
    *) say "warning: $TOKEN has mode $MODE, wider than 600; run: chmod 600 $TOKEN" ;;
  esac
  KEY_STATE="present"
elif [ "${FALLGUY_ARGON_MOCK:-0}" = "1" ]; then
  say "warning: no Argon key in $TOKEN; FALLGUY_ARGON_MOCK=1, so Argon answers are canned"
  KEY_STATE="missing (mock mode)"
else
  die "no Argon API key: put it on one line in $TOKEN, then run chmod 600 $TOKEN"
fi

say "ok: python $PY (mujoco $MJ_VERSION, offscreen GL works), H1 model $H1, Argon key $KEY_STATE"
printf 'FALLGUY_PYTHON=%s\n' "$PY"
printf 'FALLGUY_H1_DIR=%s\n' "$H1"
