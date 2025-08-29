from colorama import init, Fore, Style
import time

def welcome_message():
      """
      Prints the welcome message for OkePy.
      """
      print(f"\n\n{Fore.YELLOW}**********************************************************************{Style.RESET_ALL}")
      print(f"{Fore.YELLOW}* Welcome to Bad Blood — an IR spectra and metadata processing tool. *{Style.RESET_ALL}")
      print(f"{Fore.YELLOW}**********************************************************************{Style.RESET_ALL}")
      drop = rf"""
             ..   
           ..  ..   
          ..    ..  
          ..    ..  
         ..      .. 
         ..      .. 
        ..   \/   ..
        ..  @  @  ..
        ..        ..    {Fore.LIGHTRED_EX}  Bad Blood 2.0{Style.RESET_ALL}
        ..   /\   ..    {Fore.LIGHTRED_EX}  Mario González-Jiménez{Style.RESET_ALL}
        ..        ..    {Fore.LIGHTRED_EX}  25 Aug 2025 - University of Glasgow{Style.RESET_ALL}  
         ...    ... 
           ......    
            """   
      print(drop)

      print(f"{Fore.YELLOW}The program for processing FTIR spectral data.{Style.RESET_ALL}\n")
      # Wait 2 seconds before showing the rest
      time.sleep(4)

      print(f"\n\n{Fore.YELLOW}The analysis follows three main steps:{Style.RESET_ALL}\n")

      print(f"  {Fore.GREEN}1. 'load \"directory\"'{Style.RESET_ALL}")
      print(f"     Loads .mzz spectral files from a directory, analyses naming schemes,")
      print(f"     and creates a unified data matrix with metadata and interpolated spectra.")
      print(f"     {Fore.CYAN}If no directory is specified, a file browser will appear.{Style.RESET_ALL}\n")

      print(f"  {Fore.GREEN}2. 'prepare'{Style.RESET_ALL}")
      print(f"     Applies quality control filters to identify and remove problematic spectra:")
      print(f"     • Abnormal background detection (interferometer errors)")
      print(f"     • Atmospheric interference detection (water vapour, CO₂)")
      print(f"     • Low-intensity signal detection (insufficient signal strength)")
      print(f"     {Fore.CYAN}Interactive options to remove or flag identified issues.{Style.RESET_ALL}\n")

      print(f"  {Fore.GREEN}3. 'export'{Style.RESET_ALL}")
      print(f"     Exports the processed data with interactive options:")
      print(f"     • Select specific wavenumbers or export all spectral data")
      print(f"     • Visual preview of selected wavenumbers on sample spectra")
      print(f"     • Multiple export formats (Parquet, CSV, Excel)")
      print(f"     {Fore.CYAN}Files are saved to the original data directory by default.{Style.RESET_ALL}\n")

      print(f"{Fore.YELLOW}Additional commands:{Style.RESET_ALL}")
      print(f"  {Fore.CYAN}'help'{Style.RESET_ALL} - Display available commands and their descriptions")
      print(f"  {Fore.CYAN}'exit'{Style.RESET_ALL} - Exit the program safely")

      print(f"\n{Fore.MAGENTA}Recommended workflow: load → prepare → export{Style.RESET_ALL}")
      print(f"\n{Fore.YELLOW}Each step can be repeated if you want to adjust parameters or try different settings.{Style.RESET_ALL}")