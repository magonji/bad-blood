from colorama import init, Fore, Style
import shlex
import pandas as pd

from bad_blood_pkg import welcome
from bad_blood_pkg import data_loader
from bad_blood_pkg import data_preparer
from bad_blood_pkg import data_exporter


def main():
    """
    Main entry point for the Bad Blood program.
    Handles user interaction and command execution.
    """
    init(autoreset=True)  # Initialise colorama for cross-platform terminal colors

    welcome.welcome_message()

    df = pd.DataFrame()

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
                print(f'  {Fore.GREEN}load "directory"{Style.RESET_ALL}    - Load and analyse .mzz spectral files from a directory')
                print(f"                            Creates unified matrix with metadata and interpolated spectra")
                print(f"  {Fore.LIGHTGREEN_EX}prepare{Style.RESET_ALL}               - Apply quality control filters to remove problematic spectra")
                print(f"                            Detects abnormal backgrounds, atmospheric interference, and low signals")
                print(f"  {Fore.YELLOW}export{Style.RESET_ALL}                - Export processed data with optional wavenumber selection")
                print(f"                            Interactive selection of specific wavenumbers and file formats")
                print(f"  {Fore.LIGHTRED_EX}help{Style.RESET_ALL}                  - Display this help message")
                print(f"  {Fore.MAGENTA}exit{Style.RESET_ALL}                  - Exit the program")
                print(f"\n{Fore.CYAN}Command workflow:{Style.RESET_ALL} load → prepare → export")

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