"""Shared retro-TUI console styling for the Bad Blood engine steps.

The full-screen TUI strips ANSI colour from captured stdout, so prominence is
carried by structure, not colour. Section headers are printed as a full-width
bar whose text starts with ``── ``; the TUI console renderer recognises that
prefix and paints the whole line as a red header bar. Here we also colour it
(red background, white text) so it stands out in the standalone CLI too.

Import ``section`` / ``explain`` from here in every step (check, prepare,
export) so the whole program shares one look.
"""

from colorama import Fore, Back, Style

# Width of the section bar (characters). The TUI adds a small left margin.
PANEL_W = 60


def section(title: str) -> None:
    """Print a prominent section header as a full-width red bar."""
    bar = f"── {title.upper()} ".ljust(PANEL_W)
    print(f"\n{Back.RED}{Fore.WHITE}{Style.BRIGHT}{bar}{Style.RESET_ALL}")


def explain(*lines: str) -> None:
    """Print dim, plain-language explanatory text (one argument per line)."""
    for ln in lines:
        print(f"{Fore.LIGHTBLACK_EX}{ln}{Style.RESET_ALL}")
