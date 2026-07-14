"""Intercept the engine's matplotlib preview and route it to the in-TUI braille plot.

The exporter builds a figure with ``plt.plot(...)`` + ``plt.axvline(...)`` and calls
``plt.show(block=False)`` right before asking "Are you satisfied with this
selection?".  We replace those pyplot calls so, instead of opening a window
(impossible on the worker thread), the spectrum + selected wavenumbers are
captured and published to the bridge for braille rendering.  No engine edits.
"""

import numpy as np

from bad_blood_pkg.tui.bridge import BRIDGE

_cap = {"x": None, "y": None, "markers": []}


def install():
    import matplotlib.pyplot as plt

    orig_close = plt.close

    def plot(*args, **kwargs):
        # First (x, y) positional pair is the spectrum; a new plot starts a
        # fresh set of markers.
        try:
            if len(args) >= 2 and np.ndim(args[0]) == 1 and np.ndim(args[1]) == 1:
                _cap["x"] = np.asarray(args[0], dtype=float)
                _cap["y"] = np.asarray(args[1], dtype=float)
                _cap["markers"] = []
        except Exception:
            pass
        return []

    def axvline(*args, x=None, **kwargs):
        if x is None and args:
            x = args[0]
        try:
            _cap["markers"].append(float(x))
        except Exception:
            pass

    def show(*args, **kwargs):
        if _cap["x"] is not None and _cap["y"] is not None:
            BRIDGE.set_preview({"x": _cap["x"], "y": _cap["y"],
                                "markers": list(_cap["markers"])})

    def close(*args, **kwargs):
        try:
            orig_close(*args, **kwargs)
        except Exception:
            pass
        _cap["x"] = None
        _cap["y"] = None
        _cap["markers"] = []
        BRIDGE.clear_preview()

    plt.plot = plot
    plt.axvline = axvline
    plt.show = show
    plt.close = close
