"""Runs the untouched engine handlers under the TUI bridge.

`inject_shim()` swaps each engine module's ``questionary`` reference for the
bridge shim, so every prompt the engine raises becomes a native full-screen
widget.  The step functions simply call the existing handlers.
"""

import os

from bad_blood_pkg.tui import bridge


def inject_shim():
    """Point every engine module's `questionary` at the bridge shim (idempotent)."""
    # Force a non-interactive backend BEFORE the engine imports pyplot, so the
    # export preview cannot crash on a worker thread (plt.show becomes a no-op).
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    # Import each engine module independently so a missing optional dependency
    # (e.g. tabulate in file_checker) degrades that one step instead of the TUI.
    import importlib
    for name in ("data_loader", "data_preparer", "data_exporter", "file_checker"):
        try:
            mod = importlib.import_module(f"bad_blood_pkg.{name}")
            mod.questionary = bridge
        except Exception:
            pass
    # Route the exporter's matplotlib preview to the in-TUI braille plot.
    try:
        from bad_blood_pkg.tui import mpl_preview
        mpl_preview.install()
    except Exception:
        pass


# ------------------------------------------------------------------ path helpers
def clean_dir(p):
    """Normalise a pasted directory path.

    Trims whitespace, strips a single pair of surrounding quotes (macOS' "Copy as
    Pathname" wraps paths with spaces in single quotes, e.g.
    '/Users/me/2025 - 02 - Mosquitos database') and expands a leading ``~``.
    """
    p = (p or "").strip()
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
        p = p[1:-1].strip()
    return os.path.expanduser(p)


def valid_dir(p):
    """questionary-style validator: True or an error message."""
    cleaned = clean_dir(p)
    if not cleaned:
        return "Please enter a path."
    if not os.path.isdir(cleaned):
        return "Directory not found."
    return True


# ------------------------------------------------------------------ guided wizard
def run_pipeline(tui):
    """Guided end-to-end flow: ask the folder once, then run
    check -> (confirm) -> load -> prepare -> export in a single pass.

    The directory is fed to both `check` and `load` so neither re-prompts for it.
    It may be supplied up front (``tui.pipeline_dir``, set by the welcome-screen
    path field); otherwise we ask for it here (e.g. "New analysis" from the menu).
    """
    from bad_blood_pkg import file_checker, data_loader, data_preparer, data_exporter

    directory = tui.pipeline_dir
    tui.pipeline_dir = None
    if not directory:
        raw = bridge.path(
            "Path to the folder with the spectra:", qmark="📂", validate=valid_dir
        ).ask()
        if not raw:
            return "Cancelled."
        directory = clean_dir(raw)
    else:
        directory = clean_dir(directory)
    tui.directory = directory

    # 1. Audit file names & date formats.
    file_checker.handle_check_command([directory])

    # 2. Gate before the heavy work.
    if not bridge.confirm("Continue and load the spectra?", default=True).ask():
        return "Stopped after the file check."

    # 3. Load spectra into the unified matrix.
    result = data_loader.handle_load_command([directory])
    if result is None:
        return "Load failed — see the log above."
    df, src = result
    tui.df = df
    tui.directory = src
    tui.prepared_df = None
    tui.n_loaded = len(df)
    tui.n_prepared = None

    # 4. Quality-control filters.
    prepared = data_preparer.handle_prepare_command(df, tui.directory)
    if prepared is None:
        return "Prepare produced no data."
    tui.prepared_df = prepared
    tui.n_prepared = len(prepared)

    # 5. Export.
    data_exporter.handle_export_command(prepared, tui.directory)
    return f"Done — {len(prepared)} spectra prepared and exported."


# ------------------------------------------------------------------ steps
def run_check(tui):
    from bad_blood_pkg import file_checker
    args = [tui.directory] if tui.directory else []
    file_checker.handle_check_command(args)
    return "File check finished."


def run_load(tui):
    from bad_blood_pkg import data_loader
    result = data_loader.handle_load_command([])   # engine prompts for the path
    if result is not None:
        df, src = result
        tui.df = df
        tui.directory = src
        tui.prepared_df = None
        tui.n_loaded = len(df)
        tui.n_prepared = None
        return f"Loaded {len(df)} spectra."
    return "Load cancelled / failed."


def run_prepare(tui):
    from bad_blood_pkg import data_preparer
    prepared = data_preparer.handle_prepare_command(tui.df, tui.directory)
    if prepared is not None:
        tui.prepared_df = prepared
        tui.n_prepared = len(prepared)
        return f"Prepared: {len(prepared)} spectra remain."
    return "Prepare produced no data."


def run_export(tui):
    from bad_blood_pkg import data_exporter
    data_exporter.handle_export_command(tui.prepared_df, tui.directory)
    return "Export finished."


STEP_RUNNERS = {
    "pipeline": run_pipeline,
    "check": run_check,
    "load": run_load,
    "prepare": run_prepare,
    "export": run_export,
}
