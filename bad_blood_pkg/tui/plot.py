"""Braille line plot for the in-TUI spectral preview.

Renders a 1-D spectrum as Unicode braille (2x4 dots per character cell) plus a
marker axis for the selected wavenumbers.  Returns prompt_toolkit FormattedText
fragments so the caller can colour the curve and the markers independently.
"""

import numpy as np

BRAILLE_BASE = 0x2800
# DOTS[dy][dx] -> bit for the dot at sub-cell (dx in 0..1, dy in 0..3)
DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def _grid(xs, ys, W, H):
    DW, DH = 2 * W, 4 * H
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0

    def to_px(x):  # IR convention: high wavenumber on the left
        return int(round((xmax - x) / (xmax - xmin) * (DW - 1)))

    def to_py(y):  # higher absorbance -> higher up (row 0 = top)
        return int(round((ymax - y) / (ymax - ymin) * (DH - 1)))

    grid = [[0] * H for _ in range(W)]

    def setdot(px, py):
        if 0 <= px < DW and 0 <= py < DH:
            grid[px // 2][py // 4] |= DOTS[py % 4][px % 2]

    pts = [(to_px(x), to_py(y)) for x, y in zip(xs, ys)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            setdot(int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))

    return grid, xmin, xmax


def braille_fragments(data, W, H, styles):
    """data: {'x','y','markers'}. styles: {'curve','mark','hint'} class names."""
    xs = np.asarray(data.get("x"), dtype=float)
    ys = np.asarray(data.get("y"), dtype=float)
    markers = data.get("markers", []) or []

    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size < 2:
        return [(styles["hint"], "  (no spectrum to preview)\n")]

    grid, xmin, xmax = _grid(xs, ys, W, H)

    frags = [(styles["hint"], "  Preview · random spectrum  (IR: high → low cm⁻¹)\n")]
    for cy in range(H):
        row = "".join(chr(BRAILLE_BASE + grid[cx][cy]) for cx in range(W))
        frags.append((styles["curve"], "  " + row + "\n"))

    # marker axis
    def col_of(x):
        px = int(round((xmax - x) / (xmax - xmin) * (2 * W - 1)))
        return min(W - 1, max(0, px // 2))

    axis = [" "] * W
    for mk in markers:
        if xmin <= mk <= xmax:
            axis[col_of(mk)] = "▲"
    frags.append((styles["curve"], "  "))
    for ch in axis:
        frags.append((styles["mark"] if ch == "▲" else styles["curve"], ch))
    frags.append(("", "\n"))

    if markers:
        lbl = ", ".join(f"{m:.0f}" for m in markers)
        frags.append((styles["mark"], f"  selected: {lbl} cm⁻¹\n"))
    return frags
