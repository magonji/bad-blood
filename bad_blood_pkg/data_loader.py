"""
Data loader module for Bad Blood FTIR spectra processing.

This module handles loading and processing of .mzz spectral files,
including metadata analysis and creation of unified data matrices.

Uses a memory-efficient two-pass approach:
  - Pass 1: Scan metadata and spectral properties (header only, no spectra stored)
  - Pass 2: Re-read files and interpolate directly into pre-allocated float32 matrix
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
from bad_blood_pkg.console_ui import section, explain


def find_spectra_files() -> Optional[str]:
    """
    Prompts the user to select a directory containing spectral files.

    Returns:
        Path to selected directory or None if cancelled
    """
    target_directory = questionary.path(
        "Please select the root directory to analyse:",
    ).ask()

    return target_directory


def analyse_mzz_files(root_directory: str) -> Optional[Tuple[int, int, bool]]:
    """
    Analyses .mzz files in subdirectories to determine naming scheme consistency.

    Args:
        root_directory: Root directory path to search for .mzz files

    Returns:
        Tuple of (metadata_length, num_dates, has_age) representing the most
        common naming scheme, or None if no files found
    """
    total_files = 0
    metadata_counts: CounterType[Tuple[int, int, bool]] = Counter()

    # Regex patterns for extracting metadata elements
    # Date pattern: 6 or 8 digits between hyphens or before extension (avoids serial numbers)
    date_pattern = re.compile(r'(?:^|-)(\d{8}|\d{6})(?=-|\.)')  # YYYYMMDD or YYMMDD format
    age_pattern = utils.AGE_TAG_RE  # ##D / ##W / ##M / ##Y format

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

                    # Extract metadata from filename (excluding extension)
                    name_without_ext = filename[:-4] if filename.lower().endswith('.mzz') else filename
                    parts = name_without_ext.split('-')
                    metadata = parts

                    # Count dates and check for an age tag (days/weeks/months/years)
                    num_dates = len(date_pattern.findall(filename))
                    has_age = bool(age_pattern.search(filename))

                    # Create metadata signature
                    metadata_key = (len(metadata), num_dates, has_age)
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
        length, num_dates, has_age = scheme

        is_most_common = scheme == most_common_scheme
        prefix = "Most common" if is_most_common else "Less common"

        print(f"{prefix} naming scheme ({count} files):")
        print(f"   • Number of metadata parts: {length}")
        print(f"   • Number of dates (YYMMDD/YYYYMMDD): {num_dates}")
        print(f"   • Contains an age tag ({utils.AGE_TAG_EXAMPLES}): "
              f"{'Yes' if has_age else 'No'}\n")


def load_and_process_spectra(root_directory: str,
                           most_common_scheme: Optional[Tuple[int, int, bool]]) -> Optional[pd.DataFrame]:
    """
    Loads and processes spectra from files matching the most common naming scheme.
    Uses a memory-efficient two-pass approach.

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

    # =================================================================
    # PASS 1: Scan metadata and spectral properties (no spectra stored)
    # =================================================================
    valid_files, metadata_list, spectral_props, spectral_properties = (
        _scan_spectral_properties(matching_files)
    )

    if not valid_files:
        print(f"\n{Fore.RED}No valid spectra were processed.{Style.RESET_ALL}")
        return None

    # Display consistency report
    _print_spectral_consistency_report(spectral_properties)

    # Calculate common wavenumber grid from properties
    grid = _calculate_common_grid(spectral_props)
    if grid is None:
        return None
    common_wavenumbers, num_points = grid

    # =================================================================
    # PASS 2: Read spectra and interpolate directly into matrix
    # =================================================================
    spectral_matrix, failed_indices = _build_spectral_matrix(
        valid_files, common_wavenumbers, num_points
    )

    if spectral_matrix is None:
        print(f"\n{Fore.RED}Failed to build spectral matrix.{Style.RESET_ALL}")
        return None

    # Remove rows that failed in pass 2
    if failed_indices:
        spectral_matrix = np.delete(spectral_matrix, failed_indices, axis=0)
        for idx in sorted(failed_indices, reverse=True):
            metadata_list.pop(idx)
            valid_files.pop(idx)
        print(f"{Fore.YELLOW}Removed {len(failed_indices)} spectra that failed during loading.{Style.RESET_ALL}")

    # Keep the file each row came from (path relative to the scanned folder) so
    # later steps can name the offending files (e.g. the duplicate report).
    source_files = [os.path.relpath(p, root_directory) for p in valid_files]

    # Convert to DataFrame with user-defined column names
    df = _create_dataframe(metadata_list, spectral_matrix, common_wavenumbers, source_files)

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
                name_without_ext = filename[:-4]
                parts = name_without_ext.split('-')
                metadata = parts

                if len(metadata) == expected_length:
                    matching_files.append(os.path.join(current_path, filename))

    return matching_files


# =========================================================================
# PASS 1: Lightweight scan (metadata + spectral properties only)
# =========================================================================

def _scan_spectral_properties(file_paths: List[str]) -> Tuple[List[str], List[List[str]], List[Dict], CounterType]:
    """
    Pass 1: Scans files to extract metadata and spectral properties
    without loading the full spectra into memory.

    Args:
        file_paths: List of file paths to scan

    Returns:
        Tuple of (valid_file_paths, metadata_list, spectral_props, spectral_properties_counter)
    """
    valid_files = []
    metadata_list = []
    spectral_props = []
    spectral_properties = Counter()

    for file_path in tqdm(file_paths, desc="Scanning files (pass 1)"):
        try:
            result = _scan_single_file(file_path)
            if result:
                metadata, props = result
                valid_files.append(file_path)
                metadata_list.append(metadata)
                spectral_props.append(props)

                spectrum_key = (props['min_wn'], props['max_wn'],
                               props['num_points'], round(props['resolution'], 5))
                spectral_properties[spectrum_key] += 1

        except Exception as e:
            filename = os.path.basename(file_path)
            tqdm.write(f"Error scanning '{filename}': {e}")
            continue

    return valid_files, metadata_list, spectral_props, spectral_properties


def _scan_single_file(file_path: str) -> Optional[Tuple[List[str], Dict]]:
    """
    Extracts metadata and spectral properties from a single .mzz file
    WITHOUT loading the full spectrum data. Only reads the 3-line header.

    Args:
        file_path: Path to the .mzz file

    Returns:
        Tuple of (metadata_parts, spectral_properties_dict) or None if failed
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

            with zf.open(tmp_filename) as spectrum_file:
                data = spectrum_file.read().decode('utf-8').splitlines()

                # The two header bounds may be stored in either order: most
                # instruments write the highest wavenumber first (descending
                # spectra), but some export ascending files (lowest first).
                # Sort them so the range is always read correctly.
                bound_a = float(data[0].strip())
                bound_b = float(data[1].strip())
                max_wn = max(bound_a, bound_b)
                min_wn = min(bound_a, bound_b)
                num_points = int(float(data[2].strip()))

                # Validate data consistency (count lines, don't parse floats)
                if len(data) - 3 != num_points:
                    tqdm.write(f"Warning: Point count mismatch in '{filename}'. Skipping.")
                    return None

                # Calculate resolution from header values (no need to build arrays)
                resolution = abs(max_wn - min_wn) / (num_points - 1) if num_points > 1 else 0

                # Extract metadata from filename
                name_without_ext = filename[:-4] if filename.lower().endswith('.mzz') else filename
                metadata_parts = name_without_ext.split('-')

                props = {
                    'min_wn': min_wn,
                    'max_wn': max_wn,
                    'num_points': num_points,
                    'resolution': resolution
                }

                return metadata_parts, props

    except zipfile.BadZipFile:
        tqdm.write(f"Error: '{filename}' is not a valid zip file. Skipping.")
    except (ValueError, IndexError) as e:
        tqdm.write(f"Error: Invalid data format in '{filename}': {e}. Skipping.")

    return None


# =========================================================================
# Common wavenumber grid calculation
# =========================================================================

def _calculate_common_grid(spectral_props: List[Dict]) -> Optional[Tuple[np.ndarray, int]]:
    """
    Calculates the common wavenumber grid from spectral properties
    gathered during pass 1.

    Args:
        spectral_props: List of property dictionaries from pass 1

    Returns:
        Tuple of (common_wavenumbers_array, num_points), or None if the spectra
        share no common wavenumber range.
    """
    # Find the overlapping range (tightest bounds)
    min_wavenumber = max(props['min_wn'] for props in spectral_props)
    max_wavenumber = min(props['max_wn'] for props in spectral_props)

    # Spectra covering disjoint ranges have no common grid to interpolate onto.
    if max_wavenumber <= min_wavenumber:
        print(f"\n{Fore.RED}The spectra share no common wavenumber range.{Style.RESET_ALL}")
        print(f"The narrowest range would be [{max_wavenumber:.1f}, {min_wavenumber:.1f}] cm⁻¹, "
              f"which is empty.")
        print("Check the ranges listed above: some files cover a region the others do not.")
        return None

    # Use highest resolution (smallest step)
    resolutions = [props['resolution'] for props in spectral_props if props['resolution'] > 0]
    highest_resolution = min(resolutions) if resolutions else 1.0

    num_points = int((max_wavenumber - min_wavenumber) / highest_resolution) + 1
    common_wavenumbers = np.linspace(max_wavenumber, min_wavenumber, num_points)

    return common_wavenumbers, num_points


# =========================================================================
# PASS 2: Read spectra and build matrix
# =========================================================================

def _build_spectral_matrix(valid_files: List[str],
                           common_wavenumbers: np.ndarray,
                           num_points: int) -> Tuple[Optional[np.ndarray], List[int]]:
    """
    Pass 2: Re-reads spectra files and interpolates directly into a
    pre-allocated float32 matrix. Only one spectrum is in memory at a time.

    Args:
        valid_files: List of validated file paths from pass 1
        common_wavenumbers: Target wavenumber grid
        num_points: Number of points in target grid

    Returns:
        Tuple of (float32 matrix of shape (num_spectra, num_points), list of failed row indices)
    """
    num_spectra = len(valid_files)

    print(f"\n{Fore.MAGENTA}Creating Unified Data Matrix{Style.RESET_ALL}")

    # Pre-allocate the full matrix
    spectral_matrix = np.zeros((num_spectra, num_points), dtype=np.float32)
    failed_indices = []

    for idx, file_path in enumerate(tqdm(valid_files, desc="Loading spectra (pass 2)")):
        result = _read_spectrum(file_path)
        if result is None:
            failed_indices.append(idx)
            tqdm.write(f"Error reading '{os.path.basename(file_path)}' in pass 2. Skipping.")
            continue

        original_wavenumbers, absorbances = result

        # Interpolate directly into the matrix row
        interp_func = interp1d(original_wavenumbers, absorbances,
                              kind='linear', fill_value='extrapolate')
        spectral_matrix[idx, :] = interp_func(common_wavenumbers)

    if failed_indices:
        print(f"{Fore.YELLOW}Warning: {len(failed_indices)} files had errors during pass 2.{Style.RESET_ALL}")

    max_wavenumber = common_wavenumbers[0]
    min_wavenumber = common_wavenumbers[-1]
    effective_spectra = num_spectra - len(failed_indices)
    print(f"Unified matrix created: {effective_spectra} rows × {num_points} columns")
    print(f"Spectra interpolated to range [{max_wavenumber:.1f}, {min_wavenumber:.1f}] cm⁻¹ "
          f"with {num_points} points")

    return spectral_matrix, failed_indices


def _read_spectrum(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Reads the full spectrum from a single .mzz file.

    Args:
        file_path: Path to the .mzz file

    Returns:
        Tuple of (wavenumbers, absorbances) as numpy arrays, or None if failed
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            tmp_filename = None
            for name in zf.namelist():
                if name.endswith('.tmp'):
                    tmp_filename = name
                    break

            if not tmp_filename:
                return None

            with zf.open(tmp_filename) as spectrum_file:
                data = spectrum_file.read().decode('utf-8').splitlines()

                # Keep the header bounds in the order they are stored: the
                # absorbances follow that same direction (descending for most
                # instruments, ascending for the files that list the lowest
                # wavenumber first), so the axis must not be re-sorted here.
                first_wn = float(data[0].strip())
                last_wn = float(data[1].strip())
                num_points = int(float(data[2].strip()))
                absorbances = np.array([float(x.strip()) for x in data[3:]], dtype=np.float32)

                if len(absorbances) != num_points:
                    return None

                wavenumbers = np.linspace(first_wn, last_wn, num_points)
                return wavenumbers, absorbances

    except (zipfile.BadZipFile, ValueError, IndexError):
        return None


# =========================================================================
# DataFrame creation (memory-efficient)
# =========================================================================

# Detection patterns / thresholds for classifying a metadata column.
_DATE_RE = re.compile(r'^\d{6}$|^\d{8}$')   # YYMMDD or YYYYMMDD
_AGE_RE = utils.AGE_VALUE_RE                # #D/##D/###D and the W, M, Y units
# A "sample identifier" is a near-unique code, not a grouping category. Spotting
# it avoids listing (and treating as categories) hundreds of one-off codes.
_ID_MIN_UNIQUE = 10        # need at least this many distinct values to be an ID
# Reserved column holding the file each spectrum was read from. Downstream steps
# (duplicate report, export) look it up by this exact name.
SOURCE_FILE_COLUMN = 'source_file'
_ID_UNIQUE_RATIO = 0.5     # ...and this fraction of rows distinct (few repeats)
_MAX_CATEGORIES = 30       # more distinct values than this is never a real category
_CATEGORY_DISPLAY_CAP = 15  # how many category values to list before summarising


def _classify_metadata_column(values: List[str]) -> Tuple[str, List[str]]:
    """Classify a metadata column.

    Returns (kind, sorted_unique_values) where kind is one of:
    'date', 'age', 'id' (sample identifier / high-cardinality code), 'categorical'.
    """
    unique_values = sorted(set(values))
    n_total = len(values)
    n_unique = len(unique_values)

    if n_unique and all(_DATE_RE.match(str(v)) for v in unique_values):
        return 'date', unique_values
    if n_unique and all(_AGE_RE.match(str(v)) for v in unique_values):
        return 'age', unique_values

    ratio = n_unique / n_total if n_total else 0.0
    if n_unique > _MAX_CATEGORIES or (n_unique >= _ID_MIN_UNIQUE and ratio >= _ID_UNIQUE_RATIO):
        return 'id', unique_values
    return 'categorical', unique_values


def _display_metadata_categories(metadata_list: List[List[str]], num_metadata_cols: int) -> None:
    """
    Displays unique categories for each metadata column, helping users understand
    what each column represents before naming them.

    Args:
        metadata_list: List of metadata row lists (strings)
        num_metadata_cols: Number of metadata columns
    """
    if not metadata_list:
        return

    section("METADATA")
    explain(
        "Each part of the file name becomes a column. Columns are auto-detected",
        f"as dates, ages ({utils.AGE_TAG_EXAMPLES}), sample identifiers",
        "(near-unique codes) or categories. Identifiers are summarised instead",
        "of listed in full.",
    )
    print()

    for col_idx in range(num_metadata_cols):
        values = [row[col_idx] for row in metadata_list]
        kind, unique_values = _classify_metadata_column(values)

        print(f"{Fore.CYAN}metadata_{col_idx}:{Style.RESET_ALL}")

        if kind == 'date':
            print(f"  {Fore.GREEN}[Detected: DATE]{Style.RESET_ALL}")
            print(f"  Unique dates: {len(unique_values)}")
            if len(unique_values) <= 10:
                print(f"  Values: {', '.join(map(str, unique_values))}")
            else:
                print(f"  Range: {unique_values[0]} to {unique_values[-1]}")
                print(f"  First 10: {', '.join(map(str, unique_values[:10]))}")
        elif kind == 'age':
            unit = utils.age_unit(unique_values)
            unit_label = f" in {utils.AGE_UNIT_NAMES[unit]}" if unit else " (mixed units)"
            print(f"  {Fore.GREEN}[Detected: AGE{unit_label}]{Style.RESET_ALL}")
            print(f"  Values: {', '.join(map(str, unique_values))}")
        elif kind == 'id':
            print(f"  {Fore.MAGENTA}[Detected: SAMPLE ID]{Style.RESET_ALL}")
            print(f"  {len(unique_values)} unique codes — treated as an identifier, "
                  f"not a category")
            print(f"  Examples: {', '.join(map(str, unique_values[:5]))}, ...")
        else:
            print(f"  {Fore.LIGHTBLUE_EX}[Categorical Data]{Style.RESET_ALL}")
            if len(unique_values) <= _CATEGORY_DISPLAY_CAP:
                print(f"  Categories ({len(unique_values)}): "
                      f"{', '.join(map(str, unique_values))}")
            else:
                shown = ', '.join(map(str, unique_values[:_CATEGORY_DISPLAY_CAP]))
                print(f"  Categories ({len(unique_values)}): {shown}, "
                      f"... (+{len(unique_values) - _CATEGORY_DISPLAY_CAP} more)")

        print()  # Empty line for readability


def _generate_smart_column_names(metadata_list: List[List[str]], num_metadata_cols: int) -> List[str]:
    """
    Generates smart column names based on automatic detection of date and age columns.

    Args:
        metadata_list: List of metadata row lists (strings)
        num_metadata_cols: Number of metadata columns

    Returns:
        List of smart column names
    """
    if not metadata_list:
        return []

    # Counters for naming.  Age columns are counted per unit, so a file name
    # carrying both 03D and 12M yields "days_1" and "months_1".
    date_counter = 1
    age_counters: CounterType[str] = Counter()
    id_counter = 1
    metadata_counter = 1

    column_names = []

    # Analyse each metadata column and assign smart names
    for col_idx in range(num_metadata_cols):
        values = [row[col_idx] for row in metadata_list]
        kind, unique_values = _classify_metadata_column(values)

        if kind == 'date':
            column_names.append(f"date_{date_counter}")
            date_counter += 1
        elif kind == 'age':
            unit = utils.age_unit(unique_values)
            base = utils.AGE_UNIT_NAMES[unit] if unit else 'age'
            age_counters[base] += 1
            column_names.append(f"{base}_{age_counters[base]}")
        elif kind == 'id':
            column_names.append(f"id_{id_counter}")
            id_counter += 1
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

    # '<SOURCE_FILE_COLUMN>' is added automatically with the file each row came from
    if SOURCE_FILE_COLUMN in column_names:
        return f"'{SOURCE_FILE_COLUMN}' is reserved by Bad Blood. Please use another name."

    return True

def _get_column_names_from_user(num_metadata_cols: int, metadata_list: List[List[str]]) -> List[str]:
    """
    Prompts user to provide custom names for metadata columns or uses smart defaults.

    Args:
        num_metadata_cols: Number of metadata columns to name
        metadata_list: List of metadata row lists (strings)

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
        smart_names = _generate_smart_column_names(metadata_list, num_metadata_cols)

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

        # Reserved: Bad Blood adds this column itself (source file of each row)
        if SOURCE_FILE_COLUMN in column_names:
            print(f"{Fore.RED}'{SOURCE_FILE_COLUMN}' is reserved by Bad Blood. "
                  f"Please use another name.{Style.RESET_ALL}")
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


def _create_dataframe(metadata_list: List[List[str]],
                     spectral_matrix: np.ndarray,
                     common_wavenumbers: np.ndarray,
                     source_files: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Creates a pandas DataFrame from separate metadata and spectral data,
    avoiding the expensive conversion to object dtype.

    Metadata columns use 'category' dtype (very compact for repeated strings).
    Spectral columns remain as float32 (4 bytes per value instead of 28 for object).

    Args:
        metadata_list: List of metadata row lists (strings)
        spectral_matrix: Float32 numpy array with interpolated spectra
        common_wavenumbers: Array of wavenumber values
        source_files: Optional list of source file paths (one per row), added as
            the leading SOURCE_FILE_COLUMN column

    Returns:
        pandas DataFrame with named columns
    """
    num_metadata_cols = len(metadata_list[0]) if metadata_list else 0

    print(f"\n{Fore.MAGENTA}Creating DataFrame{Style.RESET_ALL}")

    # Show metadata categories to user
    _display_metadata_categories(metadata_list, num_metadata_cols)

    # Get column names for metadata (now includes smart naming)
    metadata_column_names = _get_column_names_from_user(num_metadata_cols, metadata_list)

    # Build metadata DataFrame with category dtype for memory efficiency
    metadata_df = pd.DataFrame(metadata_list, columns=metadata_column_names)
    for col in metadata_df.columns:
        metadata_df[col] = metadata_df[col].astype('category')

    # Prepend the source file path (kept as plain strings: paths are unique, so
    # the category dtype would only add overhead).
    if source_files is not None:
        metadata_df.insert(0, SOURCE_FILE_COLUMN, source_files)

    # Build spectral DataFrame directly from float32 matrix (no object conversion)
    wavenumber_columns = [f"{wn:.2f}" for wn in common_wavenumbers]
    spectral_df = pd.DataFrame(spectral_matrix, columns=wavenumber_columns)

    # Concatenate metadata + spectra
    df = pd.concat([metadata_df, spectral_df], axis=1)

    print(f"\n{Fore.GREEN}DataFrame created successfully:{Style.RESET_ALL}")
    print(f"  • Shape: {df.shape}")
    print(f"  • Metadata columns: {num_metadata_cols} (category dtype)")
    print(f"  • Spectral columns: {len(wavenumber_columns)} (float32 dtype)")
    print(f"  • Wavenumber range: {common_wavenumbers[-1]:.2f} to {common_wavenumbers[0]:.2f} cm⁻¹")

    return df


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
