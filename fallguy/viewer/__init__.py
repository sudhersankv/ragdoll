"""FALLGUY browser viewer: `python -m fallguy.viewer --world DIR [--stunt DIR]`."""
from .server import Viewer, normalize_biomech, serve  # noqa: F401
