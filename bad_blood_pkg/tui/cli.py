"""Console entry point for Bad Blood.

Launches the full-screen TUI after a friendly dependency pre-check.  This is the
target of the ``bad-blood`` console script.
"""

from colorama import Fore, Style


def check_dependencies():
    """
    Checks that all required third-party libraries are installed before
    starting the analysis. This prevents wasting time processing thousands
    of files only to discover a missing dependency at export time.
    """
    import importlib

    # (import_name, pip_name, used_by)
    required = [
        ("colorama",    "colorama",     "terminal colours"),
        ("pandas",      "pandas",       "data handling"),
        ("numpy",       "numpy",        "numerical operations"),
        ("prompt_toolkit", "prompt_toolkit", "full-screen interface"),
        ("scipy",       "scipy",        "spectral interpolation"),
        ("tqdm",        "tqdm",         "progress bars"),
        ("sklearn",     "scikit-learn", "quality filtering (R² score)"),
        ("matplotlib",  "matplotlib",   "spectral preview plots"),
        ("tabulate",    "tabulate",     "file checker reports"),
    ]

    # pyarrow or fastparquet needed by pandas.to_parquet()
    parquet_engines = [
        ("pyarrow",     "pyarrow"),
        ("fastparquet", "fastparquet"),
    ]

    missing = []
    for import_name, pip_name, purpose in required:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((pip_name, purpose))

    # Check that at least one parquet engine is available
    has_parquet_engine = False
    for import_name, pip_name in parquet_engines:
        try:
            importlib.import_module(import_name)
            has_parquet_engine = True
            break
        except ImportError:
            continue

    if not has_parquet_engine:
        missing.append(("pyarrow", "parquet export (install pyarrow or fastparquet)"))

    if missing:
        print(f"\n{Fore.RED}Missing required dependencies:{Style.RESET_ALL}\n")
        for pip_name, purpose in missing:
            print(f"  {Fore.YELLOW}•{Style.RESET_ALL} {pip_name}  ({purpose})")
        print(f"\n{Fore.CYAN}Install them with:{Style.RESET_ALL}")
        pip_names = " ".join(pip_name for pip_name, _ in missing)
        print(f"  pip install {pip_names}\n")
        raise SystemExit(1)


def main():
    """Entry point for the ``bad-blood`` command."""
    from colorama import init
    init(autoreset=True)
    check_dependencies()
    from bad_blood_pkg.tui.app import run
    run()


if __name__ == "__main__":
    main()
