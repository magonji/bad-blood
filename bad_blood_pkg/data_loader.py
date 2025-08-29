"""
Data loader module for Bad Blood FTIR spectra processing.

This module handles loading and processing of .mzz spectral files,
including metadata analysis and creation of unified data matrices.
"""

import os
import re
import zipfile
import threading
from typing import Union, List, Tuple, Dict, Optional, Counter as CounterType
from collections import Counter

import numpy as np
import pandas as pd
import questionary
from colorama import Fore, Style
from scipy.interpolate import interp1d
from tqdm import tqdm

from bad_blood_pkg import utils


def find_spectra_files() -> Optional[str]:
    """
    Prompts the user to select a directory containing spectral files.
    
    Returns:
        Path to selected directory or None if cancelled
    """
    target_directory = questionary.path(
        "Please select the root directory to analyse:",
        default=os.getcwd(),
        qmark="📂"
    ).ask()
    
    return target_directory


def analyse_mzz_files(root_directory: str) -> Optional[Tuple[int, int, bool]]:
    """
    Analyses .mzz files in subdirectories to determine naming scheme consistency.
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        Tuple of (metadata_length, num_dates, has_days) representing the most
        common naming scheme, or None if no files found
    """
    total_files = 0
    metadata_counts: CounterType[Tuple[int, int, bool]] = Counter()
    
    # Regex patterns for extracting metadata elements
    date_pattern = re.compile(r'\d{6}')  # YYMMDD format
    days_pattern = re.compile(r'\d{2}D')  # ##D format
    
    print(f"\n{Fore.MAGENTA}Analysing files in directory: {root_directory}{Style.RESET_ALL}\n")
    
    # Start progress spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=utils.spinner, args=(stop_spinner,))
    spinner_thread.start()
    
    try:
        # Recursively search for .mzz files
        for current_path, _, files in os.walk(root_directory):
            for filename in files:
                if filename.lower().endswith('.mzz'):
                    total_files += 1
                    
                    # Extract metadata from filename (excluding extension and indicator code)
                    parts = filename.split('-')
                    metadata = parts[:-1]
                    
                    # Count dates and check for days notation
                    num_dates = len(date_pattern.findall(filename))
                    has_days = bool(days_pattern.search(filename))
                    
                    # Create metadata signature
                    metadata_key = (len(metadata), num_dates, has_days)
                    metadata_counts[metadata_key] += 1
    
    finally:
        # Ensure spinner is stopped
        stop_spinner.set()
        spinner_thread.join()
    
    # Handle case where no files are found
    if not metadata_counts:
        print(f"{Fore.RED}No .mzz files were found in the selected directory.{Style.RESET_ALL}")
        return None
    
    # Determine most common naming scheme
    most_common_scheme = metadata_counts.most_common(1)[0][0]
    
    # Display analysis report
    _print_metadata_report(total_files, metadata_counts, most_common_scheme)
    
    return most_common_scheme


def _print_metadata_report(total_files: int, 
                          metadata_counts: CounterType[Tuple[int, int, bool]], 
                          most_common_scheme: Tuple[int, int, bool]) -> None:
    """
    Prints a formatted report of metadata analysis results.
    
    Args:
        total_files: Total number of .mzz files found
        metadata_counts: Counter of different metadata schemes
        most_common_scheme: The most frequently occurring scheme
    """
    print(f"{Fore.YELLOW}Metadata Analysis Report{Style.RESET_ALL}")
    print(f"Total .mzz files found: {total_files}\n")
    
    for scheme, count in metadata_counts.items():
        length, num_dates, has_days = scheme
        
        is_most_common = scheme == most_common_scheme
        prefix = "Most common" if is_most_common else "Less common"
        
        print(f"{prefix} naming scheme ({count} files):")
        print(f"   • Number of metadata parts: {length}")
        print(f"   • Number of dates (YYMMDD): {num_dates}")
        print(f"   • Contains days (##D): {'Yes' if has_days else 'No'}\n")


def load_and_process_spectra(root_directory: str, 
                           most_common_scheme: Optional[Tuple[int, int, bool]]) -> Optional[pd.DataFrame]:
    """
    Loads and processes spectra from files matching the most common naming scheme.
    Creates a unified pandas DataFrame with interpolated spectra.
    
    Args:
        root_directory: Root directory containing .mzz files
        most_common_scheme: Naming scheme to match (from analyse_mzz_files)
        
    Returns:
        pandas DataFrame with processed spectra or None if processing failed
    """
    if most_common_scheme is None:
        print(f"\n{Fore.RED}Cannot process spectra without a common naming scheme.{Style.RESET_ALL}")
        return None
    
    expected_metadata_length, _, _ = most_common_scheme
    
    # Find matching files
    matching_files = _find_matching_files(root_directory, expected_metadata_length)
    
    if not matching_files:
        print(f"\n{Fore.RED}No files found matching the common naming scheme.{Style.RESET_ALL}")
        return None
    
    print(f"\n{Fore.MAGENTA}Loading and Processing Spectra ({len(matching_files)} files){Style.RESET_ALL}")
    
    # Process all spectral files
    spectra_data, spectral_properties = _process_spectral_files(matching_files)
    
    if not spectra_data:
        print(f"\n{Fore.RED}No valid spectra were processed.{Style.RESET_ALL}")
        return None
    
    # Display consistency report
    _print_spectral_consistency_report(spectral_properties)
    
    # Create unified matrix
    common_wavenumbers, unified_matrix = _create_unified_matrix(spectra_data)
    
    # Convert to DataFrame with user-defined column names
    df = _create_dataframe(unified_matrix, common_wavenumbers, spectra_data)
    
    return df


def _find_matching_files(root_directory: str, expected_length: int) -> List[str]:
    """
    Finds all .mzz files with metadata matching the expected length.
    
    Args:
        root_directory: Directory to search
        expected_length: Expected number of metadata parts
        
    Returns:
        List of file paths matching the criteria
    """
    matching_files = []
    
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                parts = filename.split('-')
                metadata = parts[:-1]  # Exclude extension and indicator code
                
                if len(metadata) == expected_length:
                    matching_files.append(os.path.join(current_path, filename))
    
    return matching_files


def _process_spectral_files(file_paths: List[str]) -> Tuple[List[Dict], CounterType]:
    """
    Processes individual spectral files and extracts data.
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        Tuple of (processed_spectra_data, spectral_properties_counter)
    """
    spectra_data = []
    spectral_properties = Counter()
    
    for file_path in tqdm(file_paths, desc="Processing .mzz files"):
        try:
            spectrum_data = _extract_spectrum_from_file(file_path)
            if spectrum_data:
                spectra_data.append(spectrum_data)
                
                # Track spectral properties for consistency checking
                wavenumbers, _ = spectrum_data['spectrum']
                min_wn, max_wn = min(wavenumbers), max(wavenumbers)
                num_points = len(wavenumbers)
                resolution = abs(wavenumbers[1] - wavenumbers[0]) if len(wavenumbers) > 1 else 0
                
                spectrum_key = (min_wn, max_wn, num_points, round(resolution, 5))
                spectral_properties[spectrum_key] += 1
                
        except Exception as e:
            filename = os.path.basename(file_path)
            tqdm.write(f"Error processing '{filename}': {e}")
            continue
    
    return spectra_data, spectral_properties


def _extract_spectrum_from_file(file_path: str) -> Optional[Dict]:
    """
    Extracts spectral data from a single .mzz file.
    
    Args:
        file_path: Path to the .mzz file
        
    Returns:
        Dictionary containing metadata and spectral data, or None if failed
    """
    filename = os.path.basename(file_path)
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Find the .tmp file containing spectral data
            tmp_filename = None
            for name in zf.namelist():
                if name.endswith('.tmp'):
                    tmp_filename = name
                    break
            
            if not tmp_filename:
                tqdm.write(f"Warning: '{filename}' contains no .tmp file. Skipping.")
                return None
            
            # Read spectral data
            with zf.open(tmp_filename) as spectrum_file:
                data = spectrum_file.read().decode('utf-8').splitlines()
                
                max_wn = float(data[0].strip())
                min_wn = float(data[1].strip())
                num_points = int(float(data[2].strip()))
                absorbances = np.array([float(x.strip()) for x in data[3:]])
                
                # Validate data consistency
                if len(absorbances) != num_points:
                    tqdm.write(f"Warning: Point count mismatch in '{filename}'. Skipping.")
                    return None
                
                # Create wavenumber array
                wavenumbers = np.linspace(max_wn, min_wn, num_points)
                
                # Extract metadata from filename
                metadata_parts = filename.split('-')
                # Remove extension from last part
                metadata_parts[-1] = metadata_parts[-1].replace('.mzz', '')
                # Remove the sample identifier (last part after removing extension)
                metadata_parts = metadata_parts[:-1]
                
                # Create metadata dictionary
                file_metadata = {f'metadata_{i}': val for i, val in enumerate(metadata_parts)}
                file_metadata['spectrum'] = (wavenumbers, absorbances)
                
                return file_metadata
                
    except zipfile.BadZipFile:
        tqdm.write(f"Error: '{filename}' is not a valid zip file. Skipping.")
    except (ValueError, IndexError) as e:
        tqdm.write(f"Error: Invalid data format in '{filename}': {e}. Skipping.")
    
    return None


def _print_spectral_consistency_report(spectral_properties: CounterType) -> None:
    """
    Prints a report of spectral properties consistency.
    
    Args:
        spectral_properties: Counter of different spectral property combinations
    """
    print(f"\n{Fore.YELLOW}Spectral Consistency Report {Style.RESET_ALL}")
    
    if not spectral_properties:
        print("No valid spectra found with the common naming scheme.")
        return
    
    for properties, count in spectral_properties.items():
        min_wn, max_wn, num_points, resolution = properties
        print(f"• {count} spectra with range [{max_wn:.1f}, {min_wn:.1f}] cm⁻¹ "
              f"and resolution {resolution:.5f} cm⁻¹")


def _create_unified_matrix(spectra_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a unified data matrix from processed spectral data.
    
    Args:
        spectra_data: List of dictionaries containing spectral data
        
    Returns:
        Tuple of (common_wavenumbers, unified_data_matrix)
    """
    print(f"\n{Fore.MAGENTA}Creating Unified Data Matrix{Style.RESET_ALL}")
    
    # Determine common wavenumber range and resolution
    all_wavenumbers = []
    for spec_data in spectra_data:
        wavenumbers, _ = spec_data['spectrum']
        all_wavenumbers.extend(wavenumbers)
    
    min_wavenumber = max(min(spec_data['spectrum'][0]) for spec_data in spectra_data)
    max_wavenumber = min(max(spec_data['spectrum'][0]) for spec_data in spectra_data)
    
    # Calculate highest resolution (smallest step size)
    resolutions = []
    for spec_data in spectra_data:
        wavenumbers, _ = spec_data['spectrum']
        if len(wavenumbers) > 1:
            resolution = abs(wavenumbers[1] - wavenumbers[0])
            resolutions.append(resolution)
    
    highest_resolution = min(resolutions) if resolutions else 1.0
    
    # Create common wavenumber grid
    num_points = int((max_wavenumber - min_wavenumber) / highest_resolution) + 1
    common_wavenumbers = np.linspace(max_wavenumber, min_wavenumber, num_points)
    
    # Build unified matrix
    num_metadata_cols = len([key for key in spectra_data[0].keys() if key.startswith('metadata_')])
    unified_matrix = []
    
    for spec_data in tqdm(spectra_data, desc="Creating unified matrix"):
        original_wavenumbers, absorbances = spec_data['spectrum']
        
        # Interpolate to common grid
        interp_func = interp1d(original_wavenumbers, absorbances, 
                              kind='linear', fill_value='extrapolate')
        interpolated_absorbances = interp_func(common_wavenumbers)
        
        # Combine metadata and spectral data
        row = [spec_data[f'metadata_{i}'] for i in range(num_metadata_cols)]
        row.extend(interpolated_absorbances.tolist())
        unified_matrix.append(row)
    
    final_matrix = np.array(unified_matrix, dtype=object)
    
    print(f"Unified matrix created: {len(unified_matrix)} rows × {len(unified_matrix[0])} columns")
    print(f"Spectra interpolated to range [{max_wavenumber:.1f}, {min_wavenumber:.1f}] cm⁻¹ "
          f"with {num_points} points")
    
    return common_wavenumbers, final_matrix


def _display_metadata_categories(spectra_data: List[Dict]) -> None:
    """
    Displays unique categories for each metadata column, helping users understand 
    what each column represents before naming them.
    
    Args:
        spectra_data: List of processed spectral data dictionaries
    """
    if not spectra_data:
        return
    
    # Get number of metadata columns
    num_metadata_cols = len([key for key in spectra_data[0].keys() if key.startswith('metadata_')])
    
    # Regex patterns for automatic detection
    date_pattern = re.compile(r'^\d{6}$')  # YYMMDD format
    days_pattern = re.compile(r'^\d{1,3}D$')  # #D, ##D, ###D format
    
    print(f"\n{Fore.YELLOW}Metadata Categories Analysis{Style.RESET_ALL}")
    print("Unique categories found in each metadata column:\n")
    
    # Analyse each metadata column
    for col_idx in range(num_metadata_cols):
        # Collect all values for this column
        values = [spec_data[f'metadata_{col_idx}'] for spec_data in spectra_data]
        unique_values = sorted(set(values))
        
        # Check if it's a date or age column
        is_date = all(date_pattern.match(str(val)) for val in unique_values)
        is_age = all(days_pattern.match(str(val)) for val in unique_values)
        
        print(f"{Fore.CYAN}metadata_{col_idx}:{Style.RESET_ALL}")
        
        if is_date:
            print(f"  {Fore.GREEN}[Detected: DATE]{Style.RESET_ALL}")
            print(f"  Unique dates: {len(unique_values)}")
            if len(unique_values) <= 10:
                print(f"  Values: {', '.join(map(str, unique_values))}")
            else:
                print(f"  Range: {unique_values[0]} to {unique_values[-1]}")
                print(f"  First 10: {', '.join(map(str, unique_values[:10]))}")
        elif is_age:
            print(f"  {Fore.GREEN}[Detected: AGE]{Style.RESET_ALL}")
            print(f"  Values: {', '.join(map(str, unique_values))}")
        else:
            print(f"  {Fore.LIGHTBLUE_EX}[Categorical Data]{Style.RESET_ALL}")
            print(f"  Categories ({len(unique_values)}): {', '.join(map(str, unique_values))}")
        
        print()  # Empty line for readability


def _generate_smart_column_names(spectra_data: List[Dict]) -> List[str]:
    """
    Generates smart column names based on automatic detection of date and age columns.
    
    Args:
        spectra_data: List of processed spectral data dictionaries
        
    Returns:
        List of smart column names
    """
    if not spectra_data:
        return []
    
    # Get number of metadata columns
    num_metadata_cols = len([key for key in spectra_data[0].keys() if key.startswith('metadata_')])
    
    # Regex patterns for automatic detection
    date_pattern = re.compile(r'^\d{6}$')  # YYMMDD format
    days_pattern = re.compile(r'^\d{1,3}D$')  # #D, ##D, ###D format
    
    # Counters for naming
    date_counter = 1
    days_counter = 1
    metadata_counter = 1
    
    column_names = []
    
    # Analyse each metadata column and assign smart names
    for col_idx in range(num_metadata_cols):
        # Collect all values for this column
        values = [spec_data[f'metadata_{col_idx}'] for spec_data in spectra_data]
        unique_values = sorted(set(values))
        
        # Check if it's a date or age column
        is_date = all(date_pattern.match(str(val)) for val in unique_values)
        is_age = all(days_pattern.match(str(val)) for val in unique_values)
        
        if is_date:
            column_names.append(f"date_{date_counter}")
            date_counter += 1
        elif is_age:
            column_names.append(f"days_{days_counter}")
            days_counter += 1
        else:
            column_names.append(f"metadata_{metadata_counter}")
            metadata_counter += 1
    
    return column_names

def _validate_column_names(input_str: str, expected_count: int) -> Union[bool, str]:
    """
    Validates column names input for questionary.
    
    Args:
        input_str: User input string
        expected_count: Expected number of column names
        
    Returns:
        True if valid, or error message string if invalid
    """
    if not input_str or not input_str.strip():
        return "No input provided. Please enter column names."
    
    # Parse input
    column_names = [name.strip() for name in input_str.split(',')]
    
    # Check count
    if len(column_names) != expected_count:
        return f"Expected {expected_count} names, got {len(column_names)}. Please try again."
    
    # Check for empty names
    if any(not name for name in column_names):
        return "Column names cannot be empty. Please try again."
    
    # Check for duplicates
    if len(set(column_names)) != len(column_names):
        return "Column names must be unique. Please try again."
    
    return True

def _get_column_names_from_user(num_metadata_cols: int, spectra_data: List[Dict]) -> List[str]:
    """
    Prompts user to provide custom names for metadata columns or uses smart defaults.
    
    Args:
        num_metadata_cols: Number of metadata columns to name
        spectra_data: List of processed spectral data dictionaries
        
    Returns:
        List of column names (either custom or smart defaults)
    """
    # Ask if user wants to provide custom names
    use_custom_names = questionary.confirm(
        "Would you like to provide custom names for the metadata columns?",
        default=False
    ).ask()
    
    if not use_custom_names:
        # Generate and return smart column names
        smart_names = _generate_smart_column_names(spectra_data)
        
        print(f"\n{Fore.GREEN}Using smart column names:{Style.RESET_ALL}")
        for i, name in enumerate(smart_names):
            print(f"  metadata_{i} → {name}")
        
        return smart_names
    
    # Get custom names from user
    print(f"\nPlease provide {num_metadata_cols} column names (separated by commas):")
    print("Example: experiment, country, replica, collection_date, killing_date, measurement_date, species, age, status, rear, parity, insecticide_resistance, insecticide_exposure")
    
    while True:
        user_input = questionary.text(
            f"Enter {num_metadata_cols} column names:",
            qmark="📝",
            validate=lambda x: _validate_column_names(x, num_metadata_cols)
        ).ask()
        
        if not user_input:
            print(f"{Fore.RED}No input provided. Please enter column names.{Style.RESET_ALL}")
            continue
        
        # Parse and validate input
        column_names = [name.strip() for name in user_input.split(',')]
        
        if len(column_names) != num_metadata_cols:
            print(f"{Fore.RED}Expected {num_metadata_cols} names, got {len(column_names)}. "
                  f"Please try again.{Style.RESET_ALL}")
            continue
        
        # Check for empty names
        if any(not name for name in column_names):
            print(f"{Fore.RED}Column names cannot be empty. Please try again.{Style.RESET_ALL}")
            continue
        
        # Check for duplicate names
        if len(set(column_names)) != len(column_names):
            print(f"{Fore.RED}Column names must be unique. Please try again.{Style.RESET_ALL}")
            continue
        
        print(f"\n{Fore.GREEN}Column names accepted:{Style.RESET_ALL}")
        for i, name in enumerate(column_names):
            print(f"  metadata_{i} → {name}")
        
        confirm = questionary.confirm(
            "Are these names correct?",
            default=True
        ).ask()
        
        if confirm:
            return column_names


def _create_dataframe(unified_matrix: np.ndarray, 
                     common_wavenumbers: np.ndarray,
                     spectra_data: List[Dict]) -> pd.DataFrame:
    """
    Converts the unified matrix to a pandas DataFrame with appropriate column names.
    
    Args:
        unified_matrix: The processed data matrix
        common_wavenumbers: Array of wavenumber values
        spectra_data: Original spectral data for metadata analysis
        
    Returns:
        pandas DataFrame with named columns
    """
    print(f"\n{Fore.MAGENTA}Creating DataFrame{Style.RESET_ALL}")
    
    # Determine number of metadata columns
    num_metadata_cols = len([key for key in spectra_data[0].keys() if key.startswith('metadata_')])
    
    # Show metadata categories to user
    _display_metadata_categories(spectra_data)
    
    # Get column names for metadata (now includes smart naming)
    metadata_column_names = _get_column_names_from_user(num_metadata_cols, spectra_data)
    
    # Create wavenumber column names (rounded to reasonable precision)
    wavenumber_columns = [f"{wn:.2f}" for wn in common_wavenumbers]
    
    # Combine all column names
    all_column_names = metadata_column_names + wavenumber_columns
    
    # Create DataFrame
    df = pd.DataFrame(unified_matrix, columns=all_column_names)
    
    # Convert spectral data columns to numeric (metadata columns remain as object/string)
    spectral_columns = df.columns[num_metadata_cols:]
    df[spectral_columns] = df[spectral_columns].astype(float)
    
    print(f"\n{Fore.GREEN}DataFrame created successfully:{Style.RESET_ALL}")
    print(f"  • Shape: {df.shape}")
    print(f"  • Metadata columns: {num_metadata_cols}")
    print(f"  • Spectral columns: {len(spectral_columns)}")
    print(f"  • Wavenumber range: {common_wavenumbers[-1]:.2f} to {common_wavenumbers[0]:.2f} cm⁻¹")
    
    return df


def handle_load_command(args: List[str]) -> Optional[pd.DataFrame]:
    """
    Handles the logic for the 'load' command.
    
    Args:
        args: Command line arguments (optional directory path)
        
    Returns:
        pandas DataFrame with processed spectra if successful, None otherwise
    """
    # Determine target directory
    if args:
        target_directory = args[0]
    else:
        target_directory = find_spectra_files()
    
    # Validate directory and process spectra
    if target_directory and os.path.exists(target_directory):
        most_common_scheme = analyse_mzz_files(target_directory)
        df = load_and_process_spectra(target_directory, most_common_scheme)
        
        if df is not None:
            print(f"\n{Fore.GREEN}Final DataFrame created successfully.{Style.RESET_ALL}")
            return df, target_directory
        else:
            print(f"\n{Fore.RED}Failed to create DataFrame.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}Invalid directory selected.{Style.RESET_ALL}")
    
    return None