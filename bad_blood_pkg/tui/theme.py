"""Static visual assets and colour palette for the Bad Blood TUI.

The logo below was generated ONCE with pyfiglet (font "ansi_shadow") and pasted
here as literal text — pyfiglet is NOT a runtime dependency.
"""

# --- Palette -------------------------------------------------------------
ACCENT_TOP    = "#ff2a3a"   # bright blood red  (logo gradient top)
ACCENT_BOTTOM = "#7a0000"   # dark clot red     (logo gradient bottom)
ACCENT        = "#c1121f"   # primary accent
GOLD          = "#d4af37"   # secondary / highlights
DIM           = "#6b6b6b"   # faint / hints / status bar
FG            = "#d0d0d0"   # normal body text
OK            = "#2e9e5b"   # success green

# --- Logo (ansi_shadow, generated with pyfiglet, pasted as text) ---------
LOGO = r"""██████╗  █████╗ ██████╗     ██████╗ ██╗      ██████╗  ██████╗ ██████╗ 
██╔══██╗██╔══██╗██╔══██╗    ██╔══██╗██║     ██╔═══██╗██╔═══██╗██╔══██╗
██████╔╝███████║██║  ██║    ██████╔╝██║     ██║   ██║██║   ██║██║  ██║
██╔══██╗██╔══██║██║  ██║    ██╔══██╗██║     ██║   ██║██║   ██║██║  ██║
██████╔╝██║  ██║██████╔╝    ██████╔╝███████╗╚██████╔╝╚██████╔╝██████╔╝
╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ 
"""

LOGO_LINES = LOGO.strip("\n").split("\n")

# --- Compact banner (figlet "ansi_regular", pasted as text) for the report header ---
# Solid filled block letters — same block style as the welcome logo but one row
# shorter and without the drop shadow, so it reads as a smaller filled title.
BANNER = r"""
██████   █████  ██████      ██████  ██       ██████   ██████  ██████
██   ██ ██   ██ ██   ██     ██   ██ ██      ██    ██ ██    ██ ██   ██
██████  ███████ ██   ██     ██████  ██      ██    ██ ██    ██ ██   ██
██   ██ ██   ██ ██   ██     ██   ██ ██      ██    ██ ██    ██ ██   ██
██████  ██   ██ ██████      ██████  ███████  ██████   ██████  ██████
"""

BANNER_LINES = BANNER.strip("\n").split("\n")
# Pad every line to equal width so WindowAlign.CENTER aligns the whole block
# (otherwise each line centres on its own length and the top row drifts).
_BANNER_W = max(len(_l) for _l in BANNER_LINES)
BANNER_LINES = [_l.ljust(_BANNER_W) for _l in BANNER_LINES]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def gradient_colors(n, top=ACCENT_TOP, bottom=ACCENT_BOTTOM):
    """Return n truecolor hex strings interpolated top->bottom."""
    r1, g1, b1 = _hex_to_rgb(top)
    r2, g2, b2 = _hex_to_rgb(bottom)
    out = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        out.append(_rgb_to_hex((
            round(r1 + (r2 - r1) * t),
            round(g1 + (g2 - g1) * t),
            round(b1 + (b2 - b1) * t),
        )))
    return out


def logo_fragments():
    """FormattedText fragments: each logo line styled with its gradient colour."""
    cols = gradient_colors(len(LOGO_LINES))
    frags = []
    for line, colour in zip(LOGO_LINES, cols):
        frags.append((f"fg:{colour} bold", line + "\n"))
    return frags


def banner_fragments():
    """Compact gradient banner for the report/running header."""
    cols = gradient_colors(len(BANNER_LINES))
    frags = []
    for line, colour in zip(BANNER_LINES, cols):
        frags.append((f"fg:{colour} bold", line + "\n"))
    return frags
