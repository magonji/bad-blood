"""
Data exporter module for Bad Blood FTIR spectra processing.

This module handles exporting processed spectral data with optional wavenumber 
selection and metadata filtering capabilities.
"""

import pandas as pd
import questionary
from typing import List, Optional
from colorama import Fore, Style
import threading
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use('dark_background')
colour_palette = plt.style.library['dark_background']['axes.prop_cycle'].by_key()['color'] # We extract the colour palette of the current style

from bad_blood_pkg import data_loader
from bad_blood_pkg import data_preparer
from bad_blood_pkg import utils
from bad_blood_pkg.console_ui import section



def select_wavenumbers_interactive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive function for selecting specific wavenumbers in an IR spectra DataFrame.
    
    Allows users to either export all wavenumbers or select specific ones with
    visual preview functionality.
    
    Args:
        df: DataFrame containing IR spectral data with metadata
        
    Returns:
        DataFrame with selected wavenumber columns (or original if all selected)
    """
    # Identify spectral columns and metadata boundaries
    spectral_info = data_preparer._get_spectral_columns_info(df)
    
    if spectral_info is None:
        print(f"{Fore.RED}Could not identify spectral columns.{Style.RESET_ALL}")
        return df
    
    numeric_cols, first_spectral_col_idx = spectral_info
    metadata_cols = df.columns[:first_spectral_col_idx].tolist()
    
    # Convert column names to wavenumbers
    try:
        wavenumbers = [float(col) for col in numeric_cols.columns]
    except ValueError:
        print(f"{Fore.RED}Error: Could not convert column names to wavenumbers.{Style.RESET_ALL}")
        return df
    
    # Display current DataFrame information
    _print_dataframe_summary(df, wavenumbers)
    
    # Ask user what they want to do
    choice = questionary.select(
        "Which wavenumbers would you like to export?",
        choices=[
            "Export all wavenumbers",
            "Select specific wavenumbers (closest matches)"
        ]
    ).ask()
    
    if choice == "Export all wavenumbers":
        print(f"{Fore.GREEN}Exporting all wavenumbers.{Style.RESET_ALL}")
        return df
    else:
        return _handle_wavenumber_selection(df, wavenumbers, metadata_cols, first_spectral_col_idx)


def _print_dataframe_summary(df: pd.DataFrame, wavenumbers: List[float]) -> None:
    """
    Prints a summary of the current DataFrame.
    
    Args:
        df: DataFrame to summarise
        wavenumbers: List of wavenumber values
    """
    print(f"\n{Fore.YELLOW}Current DataFrame Summary:{Style.RESET_ALL}")
    print(f"  • Rows: {len(df)}")
    print(f"  • Wavenumber range: {max(wavenumbers):.1f} - {min(wavenumbers):.1f} cm⁻¹")
    print(f"  • Total spectral points: {len(wavenumbers)}")


def _handle_wavenumber_selection(df: pd.DataFrame, 
                                wavenumbers: List[float],
                                metadata_cols: List[str],
                                first_spectral_col_idx: int) -> pd.DataFrame:
    """
    Handles the interactive wavenumber selection process.
    
    Args:
        df: Original DataFrame
        wavenumbers: List of available wavenumbers
        metadata_cols: List of metadata column names
        first_spectral_col_idx: Index of first spectral column
        
    Returns:
        DataFrame with selected wavenumber columns
    """
    while True:
        # Request specific wavenumbers from user
        target_wavenumbers_input = questionary.text(
            "Enter desired wavenumbers separated by commas (e.g: 1650, 1450, 1000):",
            validate=lambda x: _validate_wavenumbers(x, wavenumbers)
        ).ask()
        
        # Parse user input
        target_wavenumbers = [float(x.strip()) for x in target_wavenumbers_input.split(',')]
        
        # Find closest matching wavenumbers
        raw_selected_columns = data_preparer._find_closest_wavenumber_column(df, target_wavenumbers)
            
        # Remove duplicates while preserving order
        selected_columns = list(dict.fromkeys(raw_selected_columns))
        
        # Show spectral preview and get user confirmation
        confirmation_choice = _show_spectral_preview(
            df, selected_columns, first_spectral_col_idx
        )
        
        if confirmation_choice == "Yes, use this selection":
            
            final_columns = metadata_cols + selected_columns
            filtered_df = df[final_columns].copy()
            
            print(f"\n{Fore.GREEN}Filtered DataFrame created:{Style.RESET_ALL}")
            print(f"  • Final columns: {len(final_columns)}")
            print(f"  • Metadata columns: {len(metadata_cols)}")
            print(f"  • Selected wavenumbers: {len(selected_columns)}")
            print(f"  • Wavenumbers: {', '.join(selected_columns)} cm⁻¹")

            return filtered_df
        elif confirmation_choice == "No, select different wavenumbers":
            print(f"\n{Fore.MAGENTA}Starting new selection...{Style.RESET_ALL}\n")
            continue
        else:  # Cancel and export all wavenumbers
            print(f"{Fore.MAGENTA}Cancelling selection. Exporting all wavenumbers.{Style.RESET_ALL}")
            return df


def _show_spectral_preview(df: pd.DataFrame, 
                          selected_columns: List[str],
                          first_spectral_col_idx: int) -> str:
    """
    Shows a spectral preview with selected wavenumbers highlighted.
    
    Args:
        df: DataFrame containing spectral data
        selected_columns: List of selected wavenumber column names
        first_spectral_col_idx: Index of first spectral column
        
    Returns:
        User's choice for how to proceed
    """
    # Get wavenumber values and random spectrum
    wavenumber_values = df.columns[first_spectral_col_idx:].astype(float).values
    random_row = df.sample(n=1)
    absorbance_values = random_row.iloc[:, first_spectral_col_idx:].values.flatten()
    
    # Create spectral plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(wavenumber_values, absorbance_values, linewidth=1, label='IR Spectrum')
    
    # Add vertical lines for selected columns
    for i, col in enumerate(selected_columns):
        plt.axvline(x=float(col), color=colour_palette[3], label=f'{col} cm⁻¹' if i == 0 else "")
    
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('Absorbance')
    plt.title(f'IR Spectrum - Row {random_row.index[0]} (Selection Preview)')
    plt.gca().invert_xaxis()  # Invert x-axis (IR convention)
    plt.tight_layout()
    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

    
    # Get user confirmation
    confirmation = questionary.select(
        "Are you satisfied with this selection?",
        choices=[
            "Yes, use this selection",
            "No, select different wavenumbers", 
            "Cancel and export all wavenumbers"
        ]
    ).ask()
    
    plt.close(fig)

    return confirmation


def _create_filtered_dataframe(df: pd.DataFrame, 
                              metadata_cols: List[str],
                              selected_columns: List[str]) -> pd.DataFrame:
    """
    Creates a filtered DataFrame with selected columns.
    
    Args:
        df: Original DataFrame
        metadata_cols: List of metadata column names
        selected_columns: List of selected wavenumber column names
        
    Returns:
        Filtered DataFrame
    """
   
    final_columns = metadata_cols + selected_columns
    filtered_df = df[final_columns].copy()
    
    print(f"\n{Fore.GREEN}Filtered DataFrame created:{Style.RESET_ALL}")
    print(f"  • Final columns: {len(final_columns)}")
    print(f"  • Metadata columns: {len(metadata_cols)}")
    print(f"  • Selected wavenumbers: {len(selected_columns)}")
    print(f"  • Wavenumbers: {', '.join(selected_columns)} cm⁻¹")
    
    return filtered_df


def _validate_wavenumbers(input_str: str, wavenumbers: List[float]) -> bool:
    """
    Validates that user input is a valid list of wavenumber values.
    
    Args:
        input_str: User input string
        wavenumbers: List of available wavenumbers from the spectra
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Attempt to parse the numbers
        numbers = [float(x.strip()) for x in input_str.split(',')]
        
        # Check that there's at least one number
        if len(numbers) == 0:
            return False
        
        # Check for duplicates
        if len(numbers) != len(set(numbers)):
            return False
            
        # Check that all values are within the available range
        min_wn, max_wn = min(wavenumbers), max(wavenumbers)
        
        if any(num < min_wn or num > max_wn for num in numbers):
            return False
            
        return True
    except (ValueError, AttributeError):
        return False


def apply_metadata_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allows users to filter spectra based on metadata criteria.
    
    Args:
        df: DataFrame to filter
        
    Returns:
        Filtered DataFrame based on user-selected criteria
    """
    import re
    
    # Identify metadata columns (non-numeric column names). The source file is
    # a near-unique path per row: useful to keep in the export, useless to filter by.
    numeric_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
    all_metadata_columns = [col for col in df.columns
                            if col not in numeric_columns
                            and col != data_loader.SOURCE_FILE_COLUMN]
    
    if not all_metadata_columns:
        print(f"{Fore.YELLOW}No metadata columns found. Skipping filtering step.{Style.RESET_ALL}")
        return df
    
    # Date pattern detection (YYMMDD format)
    date_pattern = re.compile(r'^\d{6}$')
    
    # Filter out columns with only one unique value or date columns
    filterable_columns = []
    single_value_columns = []
    date_columns = []
    
    print(f"\n{Fore.CYAN}Available metadata columns for filtering:{Style.RESET_ALL}")
    for col in all_metadata_columns:
        value_counts = df[col].value_counts().sort_index()
        
        # Check if column contains only dates
        is_date_column = all(date_pattern.match(str(val)) for val in value_counts.index)
        
        if len(value_counts) == 1:
            # Column has only one unique value
            single_value_columns.append(col)
            continue
        elif is_date_column:
            # Column contains dates - not useful for filtering
            date_columns.append(col)
            continue
        
        # Column has multiple non-date values - can be used for filtering
        filterable_columns.append(col)
        
        if len(value_counts) <= 10:
            # Show all values with counts
            counts_str = ", ".join([f"{val} ({count})" for val, count in value_counts.items()])
            print(f"  • {col}: {counts_str}")
        else:
            # Show summary for columns with many unique values
            print(f"  • {col}: {len(value_counts)} unique values")
            print(f"    Most common: {', '.join([f'{val} ({count})' for val, count in value_counts.head(3).items()])}")
        
    # Check if we have any filterable columns
    if not filterable_columns:
        print(f"{Fore.YELLOW}No columns available for filtering. Skipping filtering step.{Style.RESET_ALL}")
        return df
    
    # Ask if user wants to apply filters
    apply_filters = questionary.confirm(
        "Would you like to apply metadata filters to select specific spectra?",
        default=False
    ).ask()
    
    if not apply_filters:
        return df
    
    filtered_df = df.copy()
    
    finish_option = "-- FINISH FILTERING"
    while True:
        # Select column to filter (only show filterable columns)
        column_to_filter = questionary.select(
            "Select a metadata column to filter by:",
            choices=filterable_columns + [finish_option]
        ).ask()
        
        if column_to_filter == finish_option:
            break
        
        # Show unique values with counts for selected column
        value_counts = filtered_df[column_to_filter].value_counts().sort_index()
        
        # Create choices with counts for checkbox
        choices_with_counts = [f"{val} ({count})" for val, count in value_counts.items()]
        
        # Select values to keep
        selected_values_with_counts = questionary.checkbox(
            f"Select values to keep for '{column_to_filter}':",
            choices=choices_with_counts
        ).ask()
        
        if selected_values_with_counts:
            # Extract original values (remove count information)
            selected_values = []
            for choice in selected_values_with_counts:
                # Extract value before the " (" part
                original_value = choice.split(" (")[0]
                selected_values.append(original_value)
            
            # Convert back to original types if necessary
            if filtered_df[column_to_filter].dtype == 'object':
                filter_values = selected_values
            else:
                # Convert to original type
                original_type = type(filtered_df[column_to_filter].iloc[0])
                filter_values = [original_type(val) for val in selected_values]
            
            filtered_df = filtered_df[filtered_df[column_to_filter].isin(filter_values)]
            
            print(f"{Fore.GREEN}Applied filter: {len(filtered_df)} spectra remaining{Style.RESET_ALL}")
        
        if len(filtered_df) == 0:
            print(f"{Fore.RED}No spectra remaining after filtering. Returning to previous state.{Style.RESET_ALL}")
            return df
    
    print(f"\n{Fore.YELLOW}Filtering complete: {len(filtered_df)} spectra selected{Style.RESET_ALL}")
    return filtered_df


def _default_output_path(source_directory: Optional[str]) -> Path:
    """
    Builds the suggested destination for the export.

    The file is named after the scanned folder and written beside it, so
    exporting two folders in a row cannot silently overwrite the first result.

    Args:
        source_directory: Directory where the original data was loaded from

    Returns:
        Path of the suggested .parquet file
    """
    if not source_directory:
        return Path.cwd() / "data.parquet"

    p = Path(source_directory).expanduser().resolve()
    return p.parent / f"{p.name}.parquet"


def _resolve_output_path(raw: str, default_path: Path) -> Path:
    """
    Turns whatever the user typed into a concrete .parquet path.

    Accepts an absolute path, a path relative to the default folder, a bare
    file name, or a directory (the default file name is used inside it).
    Surrounding quotes — as pasted by macOS' "Copy as Pathname" — are stripped
    and a missing .parquet extension is added.

    Args:
        raw: Text entered by the user
        default_path: Destination used when the input is empty

    Returns:
        Resolved output Path
    """
    raw = (raw or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()

    if not raw:
        return default_path

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = default_path.parent / p
    if p.is_dir():
        p = p / default_path.name
    if p.suffix.lower() != '.parquet':
        p = p.parent / (p.name + '.parquet')

    return p


def _validate_output_path(raw: str, default_path: Path):
    """
    questionary-style validator for the destination field.

    Args:
        raw: Text entered by the user
        default_path: Destination used when the input is empty

    Returns:
        True if usable, otherwise the error message to display
    """
    try:
        output_file = _resolve_output_path(raw, default_path)
    except (OSError, ValueError):
        return "That is not a valid path."

    if not output_file.parent.is_dir():
        return f"Folder not found: {output_file.parent}"

    return True


def _ask_output_path(default_path: Path) -> Optional[Path]:
    """
    Asks where to write the export, offering the default name.

    Args:
        default_path: Suggested destination, pre-filled in the field

    Returns:
        Chosen Path, or None if the user cancelled
    """
    print(f"\n{Fore.YELLOW}Destination:{Style.RESET_ALL}")
    print(f"  Suggested file: {Fore.CYAN}{default_path.name}{Style.RESET_ALL}")
    print(f"  Folder: {default_path.parent}")
    print("  Leave empty and press ⏎ to accept it, or type another file name")
    print("  (saved in the same folder), a folder, or a full path.")

    while True:
        # The field is left empty on purpose: accepting the suggestion is one
        # keystroke, and a new name needs no deleting of a pre-filled path.
        raw = questionary.text(
            "Save the exported file as:",
            validate=lambda x: _validate_output_path(x, default_path),
        ).ask()

        if raw is None:
            return None

        output_file = _resolve_output_path(raw, default_path)

        if output_file.exists():
            overwrite = questionary.confirm(
                f"'{output_file.name}' already exists. Overwrite it?",
                default=False,
            ).ask()
            if not overwrite:
                continue

        return output_file


def export_data(df: pd.DataFrame, source_directory: Optional[str] = None) -> None:
    """
    Exports the DataFrame to various file formats.

    Args:
        df: DataFrame to export
        source_directory: Directory where the original data was loaded from
    """

    output_file = _ask_output_path(_default_output_path(source_directory))

    if output_file is None:
        print(f"\n{Fore.YELLOW}Export cancelled. Nothing was written.{Style.RESET_ALL}")
        return

    print(f"\n{Fore.CYAN}Exporting data to: {output_file}{Style.RESET_ALL}\n")

    # Start progress spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=utils.spinner, args=(stop_spinner,))
    spinner_thread.start()
    
    try:
        df.to_parquet(output_file, index=False)
        
        # Stop spinner before showing results
        stop_spinner.set()
        spinner_thread.join()
        
        print(f"\n{Fore.GREEN}Data successfully exported to: {output_file}{Style.RESET_ALL}")
        print(f"  • Rows exported: {len(df)}")
        print(f"  • Columns exported: {len(df.columns)}")
        
    except Exception as e:
        # Ensure spinner is stopped even if there's an error
        stop_spinner.set()
        spinner_thread.join()
        
        print(f"\n{Fore.RED}Export failed: {e}{Style.RESET_ALL}")


def handle_export_command(prepared_df: pd.DataFrame, source_directory) -> None:
    """
    Main handler for the export command.
    
    Orchestrates the complete export workflow:
    1. Wavenumber selection
    2. Metadata filtering (optional)
    4. File export
    
    Args:
        prepared_df: Prepared DataFrame ready for export
    """
    # Step 1: Select wavenumbers
    section("WAVENUMBER SELECTION")
    filtered_df = select_wavenumbers_interactive(prepared_df)

    # Step 2: Optional metadata filtering
    section("METADATA FILTERING")
    final_df = apply_metadata_filters(filtered_df)

    # Step 3: Export data
    section("WRITING FILE")
    export_data(final_df, source_directory)

    section("EXPORT COMPLETE")