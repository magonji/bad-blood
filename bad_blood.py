from colorama import init, Fore, Style
import shlex
import pandas as pd

from bad_blood_pkg import welcome
from bad_blood_pkg import data_loader
from bad_blood_pkg import data_preparer
from bad_blood_pkg import data_exporter
from bad_blood_pkg import file_checker


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
        ("questionary", "questionary",  "interactive prompts"),
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
    """
    Main entry point for the Bad Blood program.
    Handles user interaction and command execution.
    """
    init(autoreset=True)  # Initialise colorama for cross-platform terminal colors

    check_dependencies()

    welcome.welcome_message()

    df = pd.DataFrame()
    source_directory = None  # Shared directory between commands

    while True:
        try:
            command_input = input(f"{Fore.MAGENTA}\n> {Style.RESET_ALL}").strip()  
            command_parts = shlex.split(command_input)
            command = command_parts[0].lower()
            args = command_parts[1:]
            

            if command == "exit":
                print(f"{Fore.MAGENTA}Exiting the program... Goodbye!{Style.RESET_ALL}")
                break

            elif command == "load":    
                result = data_loader.handle_load_command(args)
                if result is not None:
                    df, source_directory = result
                    print(f"\n{Fore.GREEN}Spectra loaded successfully.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}You can now proceed with the command {Fore.GREEN}'prepare'{Fore.YELLOW} or repeat this step if not satisfied.{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.RED}Failed to load spectra. Please try again.{Style.RESET_ALL}")

            elif command == "check":
                # Use source_directory if available and no args provided
                if not args and source_directory:
                    print(f"{Fore.CYAN}Using previously loaded directory: {source_directory}{Style.RESET_ALL}")
                    args = [source_directory]
                result = file_checker.handle_check_command(args)
                # If check was run first, save the directory for later use
                if result and 'root_directory' in result:
                    source_directory = result.get('root_directory')   

            elif command == "prepare":
                if 'df' in locals() and df is not None and not df.empty:
                    prepared_df = data_preparer.handle_prepare_command(df)
                    print(f"\n{Fore.GREEN}Spectra prepared successfully.{Style.RESET_ALL}")
                    print(f"{Fore.YELLOW}You can now proceed with the command {Fore.GREEN}'export'{Fore.YELLOW} or repeat this step if not satisfied.{Style.RESET_ALL}")
                else:
                    print(f"\n{Fore.RED}You haven't loaded your spectra yet. Please use the command {Fore.GREEN}'load'{Fore.RED} first.{Style.RESET_ALL}")

            elif command == "export":
                if ('prepared_df' in locals() and prepared_df is not None and not prepared_df.empty and 
                    'source_directory' in locals() and source_directory is not None):
                    data_exporter.handle_export_command(prepared_df, source_directory)
                else:
                    print(f"\n{Fore.RED}You haven't prepared your spectra yet. Please use the commands {Fore.GREEN}'load'{Fore.RED} and {Fore.GREEN}'prepare'{Fore.RED} first.{Style.RESET_ALL}")

            elif command == "help":
                print(f"\n{Fore.LIGHTBLACK_EX}Available commands:{Style.RESET_ALL}")
                print(f'  {Fore.LIGHTCYAN_EX}load "directory"{Style.RESET_ALL}      - Load and analyse .mzz spectral files from a directory')
                print(f"                            Creates unified matrix with metadata and interpolated spectra")
                print(f"  {Fore.GREEN}prepare{Style.RESET_ALL}               - Apply quality control filters to remove problematic spectra")
                print(f"                            Removes duplicates; detects abnormal backgrounds, atmospheric interference, and low signals")
                print(f"  {Fore.LIGHTGREEN_EX}export{Style.RESET_ALL}                - Export processed data with optional wavenumber selection")
                print(f"                            Interactive selection of specific wavenumbers and file formats")
                print(f'  {Fore.YELLOW}check "directory"{Style.RESET_ALL}     - Analyse file naming schemes and detect date format errors')
                print(f"                            Can rename files with wrong date formats (DDMMYY→YYMMDD, YYYY→YY)")
                print(f"                            Options: --recheck (verify renamed files)")
                print(f"                                     --show-log (show rename history)")
                print(f"                                     --clear-log (clear rename history)")
                print(f"  {Fore.LIGHTRED_EX}help{Style.RESET_ALL}                  - Display this help message")
                print(f"  {Fore.LIGHTMAGENTA_EX}exit{Style.RESET_ALL}                  - Exit the program")
                print(f"\n{Fore.CYAN}Command workflow:{Style.RESET_ALL} check → load → prepare → export")
                print(f"                  {Fore.LIGHTBLACK_EX}(check is optional but recommended first){Style.RESET_ALL}")

            else:
                print(f"{Fore.RED}Unrecognised command. Use '{Fore.GREEN}help{Fore.RED}' for available commands.{Style.RESET_ALL}")

        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Program interrupted by user.{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"{Fore.RED}An error occurred: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please check your input or consult the documentation.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()