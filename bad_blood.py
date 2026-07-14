#!/usr/bin/env python3
"""Convenience launcher for running the Bad Blood TUI from a checkout.

    python bad_blood.py

Once installed (`pip install -e .`) the same thing is available as the
`bad-blood` command.  The engine is unchanged; this only starts the UI.
"""
from bad_blood_pkg.tui.cli import main

if __name__ == "__main__":
    main()
