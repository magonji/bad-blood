"""
Quality filtering module for Bad Blood FTIR spectra processing.

This module handles quality control of spectral data, including detection and 
removal of spectra with abnormal backgrounds, atmospheric interference, and 
low intensity signals.
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import questionary
from colorama import Fore, Style

pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.width', None)       # Sin límite de ancho
pd.set_option('display.max_colwidth', 100)


def abnormal_background_filter(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Filters spectra with abnormal backgrounds based on absorbance intensity at 1900 cm⁻¹.
    
    Uses interquartile range (IQR) method to identify outliers in background absorbance,
    which typically indicates interferometer errors or sample preparation issues.
    
    Args:
        df: DataFrame containing metadata and spectral data
        threshold: IQR multiplier for outlier detection (default: 2.5)
        
    Returns:
        Filtered DataFrame with abnormal background spectra removed
    """
    print(f"{Fore.MAGENTA}Filtering spectra with abnormal background...{Style.RESET_ALL}")
    
    # Find spectral columns and locate 1900 cm⁻¹
    target_wavenumber = _find_closest_wavenumber_column(df, 1900.0)
    
    if target_wavenumber is None:
        print(f"{Fore.RED}No spectral data columns found.{Style.RESET_ALL}")
        return df
    
    # Calculate IQR-based outlier boundaries
    q1 = df[target_wavenumber].quantile(0.25)
    q3 = df[target_wavenumber].quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Apply filter
    original_count = len(df)
    filtered_df = df[
        (df[target_wavenumber] >= lower_bound) & 
        (df[target_wavenumber] <= upper_bound)
    ]
    
    removed_count = original_count - len(filtered_df)
    _print_removal_summary(removed_count, "abnormal background")
    
    return filtered_df


def filter_atmospheric_interference(df: pd.DataFrame, 
                                   wavenumber_start: float = 3900.0, 
                                   wavenumber_end: float = 3500.0, 
                                   r2_threshold: float = 0.96) -> pd.DataFrame:
    """
    Identifies and optionally removes spectra with atmospheric interference.
    
    Detects atmospheric water vapour and CO₂ interference by fitting polynomials
    to the 3900-3500 cm⁻¹ region and identifying poor fits (R² < threshold).
    
    Args:
        df: DataFrame containing metadata and spectral data
        wavenumber_start: Upper wavenumber limit for analysis region
        wavenumber_end: Lower wavenumber limit for analysis region  
        r2_threshold: R² threshold below which interference is suspected
        
    Returns:
        Modified DataFrame (filtered or with interference column added)
    """
    print(f"{Fore.MAGENTA}Analysing spectra for atmospheric interference...{Style.RESET_ALL}")
    
    # Get spectral wavenumber columns and validate range
    wavenumber_range = _get_wavenumber_range(df, wavenumber_start, wavenumber_end)
    
    if wavenumber_range is None:
        print(f"{Fore.RED}Error: Invalid wavenumber range for atmospheric interference detection.{Style.RESET_ALL}")
        return df
    
    wavenumbers_slice, first_spectral_col_idx = wavenumber_range
    
    # Identify spectra with poor polynomial fits (atmospheric interference)
    interference_indices = _detect_atmospheric_interference(
        df, wavenumber_end, wavenumber_start, r2_threshold
    )
    
    num_interfered = len(interference_indices)
    _print_removal_summary(num_interfered, "atmospheric interference")
    
    if num_interfered == 0:
        return df
    
    # Ask user how to handle interfered spectra
    return _handle_interfered_spectra(df, interference_indices, first_spectral_col_idx)


def filter_low_intensity_spectra(df: pd.DataFrame, 
                                intensity_threshold: float = 0.11,
                                region_start: float = 600.0,
                                region_end: float = 400.0) -> pd.DataFrame:
    """
    Filters spectra with low intensity signals in the fingerprint region.
    
    Identifies spectra with insufficient signal strength by calculating average 
    absorbance in the 600-400 cm⁻¹ plateau region.
    
    Args:
        df: DataFrame containing metadata and spectral data
        intensity_threshold: Minimum average absorbance threshold
        region_start: Upper wavenumber limit for intensity calculation
        region_end: Lower wavenumber limit for intensity calculation
        
    Returns:
        Modified DataFrame (filtered or with intensity column added)
    """
    print(f"{Fore.MAGENTA}Filtering low-intensity spectra...{Style.RESET_ALL}")
    
    # Identify spectral columns
    spectral_info = _get_spectral_columns_info(df)
    
    if spectral_info is None:
        print(f"{Fore.RED}Could not identify spectral columns.{Style.RESET_ALL}")
        return df
    
    numeric_cols, first_spectral_col_idx = spectral_info
    
    # Calculate average absorbance in plateau region
    df_with_averages = _calculate_plateau_averages(
        df, numeric_cols, region_start, region_end, first_spectral_col_idx
    )
    
    # Identify low-intensity spectra
    low_intensity_mask = df_with_averages['average_plateau_absorbance'] < intensity_threshold
    low_intensity_count = low_intensity_mask.sum()
    
    if low_intensity_count == 0:
        filtered_df = df_with_averages.drop(columns=['average_plateau_absorbance'])
        print(f"{Fore.GREEN}No low-intensity spectra found. {Style.RESET_ALL}")
        return filtered_df
    
    print(f"{Fore.YELLOW}Identified {low_intensity_count} spectra with low intensity "
          f"(average < {intensity_threshold} in {region_start}-{region_end} cm⁻¹ region).{Style.RESET_ALL}")
    
    # Ask user how to handle low-intensity spectra
    return _handle_low_intensity_spectra(df_with_averages, low_intensity_mask, low_intensity_count)


def handle_prepare_command(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive quality filtering to spectral data.
    
    Sequentially applies three quality filters:
    1. Abnormal background detection (interferometer errors)
    2. Atmospheric interference detection (water vapour, CO₂)
    3. Low-intensity signal detection (insufficient signal strength)
    
    Args:
        df: Original DataFrame with spectral data
        
    Returns:
        Quality-filtered DataFrame
    """
    print(f"\n{Fore.MAGENTA}Starting Quality Control Analysis{Style.RESET_ALL}")
    print(f"Initial dataset: {len(df)} spectra")
    
    # Stage 1: Filter abnormal backgrounds
    print(f"\n{Fore.LIGHTBLUE_EX}Stage 1: Abnormal Background Detection{Style.RESET_ALL}")
    filtered_df = abnormal_background_filter(df)
    
    # Stage 2: Filter atmospheric interference
    print(f"\n{Fore.LIGHTBLUE_EX}Stage 2: Atmospheric Interference Detection{Style.RESET_ALL}")
    filtered_df = filter_atmospheric_interference(filtered_df)
    
    # Stage 3: Filter low-intensity spectra
    print(f"\n{Fore.LIGHTBLUE_EX}Stage 3: Low-Intensity Signal Detection{Style.RESET_ALL}")
    filtered_df = filter_low_intensity_spectra(filtered_df)
    
    print(f"\n{Fore.GREEN}Quality Control Complete{Style.RESET_ALL}")
    print(f"Final dataset: {len(filtered_df)} spectra")
    print(f"Total removed: {len(df) - len(filtered_df)} spectra")

    return filtered_df


# Helper functions for improved code organisation and efficiency
def _find_closest_wavenumber_column(df: pd.DataFrame, target_wavenumber: Union[float, List[float]]) -> Union[Optional[str], Optional[List[str]]]:
    """
    Finds the column name(s) closest to the target wavenumber(s).
    """
    # Identify numeric column names (spectral data)
    numeric_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
    
    if numeric_columns.empty:
        return None
    
    numeric_values = pd.to_numeric(numeric_columns)
    
    # Handle single value
    if isinstance(target_wavenumber, (int, float)):
        closest_idx = np.abs(numeric_values - target_wavenumber).argmin()
        return numeric_columns[closest_idx]  # Devolver el nombre original de la columna
    
    # Handle list of values
    elif isinstance(target_wavenumber, list):
        closest_columns = []
        for target in target_wavenumber:
            closest_idx = np.abs(numeric_values - target).argmin()
            closest_columns.append(numeric_columns[closest_idx])  # Nombre original
        return closest_columns
    
    else:
        raise ValueError("target_wavenumber must be a float or a list of floats")


def _get_wavenumber_range(df: pd.DataFrame, 
                         start_wn: float, 
                         end_wn: float) -> Optional[Tuple[np.ndarray, int]]:
    """
    Extracts wavenumber range and validates spectral columns.
    
    Args:
        df: DataFrame containing spectral data
        start_wn: Starting wavenumber
        end_wn: Ending wavenumber
        
    Returns:
        Tuple of (wavenumber_array, first_spectral_column_index) or None if invalid
    """
    try:
        # Get numeric column names (wavenumbers)
        numeric_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
        
        if numeric_columns.empty:
            return None
            
        wavenumbers = pd.to_numeric(numeric_columns)
        
        # Filter to specified range
        mask = (wavenumbers <= start_wn) & (wavenumbers >= end_wn)
        wavenumbers_in_range = wavenumbers[mask]
        
        if wavenumbers_in_range.empty:
            return None
        
        # Find first spectral column index
        all_numeric_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
        first_spectral_col_idx = df.columns.get_loc(all_numeric_columns[0])
        
        return wavenumbers_in_range, first_spectral_col_idx
        
    except (ValueError, KeyError):
        return None


def _detect_atmospheric_interference(df: pd.DataFrame,
                                   start_wn: float,
                                   end_wn: float,
                                   r2_threshold: float) -> List[int]:
    """
    Detects atmospheric interference using polynomial fitting.
    
    Args:
        df: DataFrame with spectral data
        wavenumbers: Wavenumber array for the analysis region
        start_wn: Starting wavenumber
        end_wn: Ending wavenumber  
        r2_threshold: R² threshold for interference detection
        
    Returns:
        List of row indices with suspected atmospheric interference
    """
    interference_indices = []
    
    # Find the actual column names closest to our target wavenumbers
    start_col_actual = _find_closest_wavenumber_column(df, start_wn)
    end_col_actual = _find_closest_wavenumber_column(df, end_wn)
    
    if start_col_actual is None or end_col_actual is None:
        return interference_indices
    
    for index, row in df.iterrows():
        try:
            # Get all spectral columns between start and end wavenumbers
            spectral_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
            spectral_values = pd.to_numeric(spectral_columns)
            
            # Filter columns in the range
            low, high = sorted([start_wn, end_wn])
            mask = (spectral_values >= low) & (spectral_values <= high)
            columns_in_range = spectral_columns[mask]
            
            if columns_in_range.empty:
                continue
                
            # Extract absorbance values for the wavenumber range
            absorbance_slice = row[columns_in_range].values.astype(float)
            wavenumbers_slice = pd.to_numeric(columns_in_range)
            
            # Fit 5th-degree polynomial
            poly_coeffs = np.polyfit(wavenumbers_slice, absorbance_slice, 5)
            absorbance_predicted = np.polyval(poly_coeffs, wavenumbers_slice)
            
            # Calculate R² goodness of fit
            r_squared = r2_score(absorbance_slice, absorbance_predicted)
            
            # Poor fit suggests atmospheric interference
            if r_squared < r2_threshold:
                interference_indices.append(index)
                
        except (ValueError, KeyError) as e:
            # Skip this spectrum if there's an error
            continue

    return interference_indices

def _get_spectral_columns_info(df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, int]]:
    """
    Identifies spectral columns and metadata boundary.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (spectral_columns_df, first_spectral_column_index) or None if error
    """
    try:
        # Find columns that can be converted to numeric (wavenumbers) - same method as other functions
        numeric_columns = df.columns[pd.to_numeric(df.columns, errors='coerce').notna()]
        
        if numeric_columns.empty:
            return None
        
        numeric_cols_df = df[numeric_columns]
        first_spectral_col_idx = df.columns.get_loc(numeric_columns[0])
        
        return numeric_cols_df, first_spectral_col_idx
        
    except (ValueError, IndexError, KeyError):
        return None


def _calculate_plateau_averages(df: pd.DataFrame,
                               spectral_cols: pd.DataFrame,
                               region_start: float,
                               region_end: float,
                               first_spectral_idx: int) -> pd.DataFrame:
    """
    Calculates average absorbance in the plateau region for each spectrum.
    
    Args:
        df: Original DataFrame
        spectral_cols: DataFrame containing only spectral columns
        region_start: Upper wavenumber limit
        region_end: Lower wavenumber limit
        first_spectral_idx: Index of first spectral column
        
    Returns:
        DataFrame with added 'average_plateau_absorbance' column
    """
    plateau_averages = []
    
    # Get spectral column names that are numeric
    spectral_column_names = spectral_cols.columns
    
    for _, row in spectral_cols.iterrows():
        # Find relevant columns in the plateau region
        relevant_cols = []
        for col in spectral_column_names:
            try:
                col_value = float(col)
                if region_start >= col_value >= region_end:
                    relevant_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        # If no columns in range, use all columns up to region_start
        if not relevant_cols:
            for col in spectral_column_names:
                try:
                    col_value = float(col)
                    if col_value <= region_start:
                        relevant_cols.append(col)
                except (ValueError, TypeError):
                    continue
        
        # Calculate average (0 if no relevant columns)
        if relevant_cols:
            average = row[relevant_cols].mean()
        else:
            average = 0.0
            
        plateau_averages.append(average)
    
    # Add the new column and reorder
    df_copy = df.copy()
    df_copy['average_plateau_absorbance'] = plateau_averages
    
    # Move the new column to just before spectral data
    cols = list(df_copy.columns)
    cols.insert(first_spectral_idx, cols.pop(cols.index('average_plateau_absorbance')))
    
    return df_copy[cols]


def _handle_interfered_spectra(df: pd.DataFrame, 
                              interference_indices: List[int],
                              first_spectral_idx: int) -> pd.DataFrame:
    """
    Handles user choice for spectra with atmospheric interference.
    
    Args:
        df: Original DataFrame
        interference_indices: List of interfered spectrum indices
        first_spectral_idx: Index of first spectral column
        
    Returns:
        Modified DataFrame based on user choice
    """
    action = questionary.select(
        "How would you like to handle spectra with atmospheric interference?",
        choices=[
            "Remove them from the DataFrame",
            "Add metadata column and keep them"
        ]
    ).ask()
    
    if action == "Remove them from the DataFrame":
        filtered_df = df.drop(index=interference_indices)
        print(f"\n{Fore.GREEN}DataFrame updated. Interfered spectra have been removed.{Style.RESET_ALL}")
        return filtered_df
    
    else:  # Add metadata column
        df_copy = df.copy()
        df_copy['atmospheric_interference'] = False
        df_copy.loc[interference_indices, 'atmospheric_interference'] = True
        
        # Reorder columns to place new column before spectral data
        cols = list(df_copy.columns)
        cols.insert(first_spectral_idx, cols.pop(cols.index('atmospheric_interference')))
        
        print(f"\n{Fore.GREEN}Atmospheric interference column added to DataFrame.{Style.RESET_ALL}")
        return df_copy[cols]


def _handle_low_intensity_spectra(df: pd.DataFrame,
                                 low_intensity_mask: pd.Series,
                                 low_intensity_count: int) -> pd.DataFrame:
    """
    Handles user choice for low-intensity spectra.
    
    Args:
        df: DataFrame with plateau averages added
        low_intensity_mask: Boolean mask identifying low-intensity spectra
        low_intensity_count: Number of low-intensity spectra
        
    Returns:
        Modified DataFrame based on user choice
    """
    choice = questionary.select(
        "How would you like to handle these low-intensity spectra?",
        choices=[
            "Remove low-intensity spectra",
            "Keep spectra and add average column"
        ]
    ).ask()
    
    if choice == "Remove low-intensity spectra":
        filtered_df = df[~low_intensity_mask].drop(columns=['average_plateau_absorbance'])
        print(f"\n{Fore.YELLOW}Removed {low_intensity_count} spectra. "
              f"New DataFrame has {len(filtered_df)} rows.{Style.RESET_ALL}")
        return filtered_df
    
    else:  # Keep spectra with average column
        print(f"\n{Fore.YELLOW}Low-intensity spectra retained. "
              f"'average_plateau_absorbance' column added.{Style.RESET_ALL}")
        return df


def _print_removal_summary(removed_count: int, filter_type: str) -> None:
    """
    Prints a formatted summary of removed spectra.
    
    Args:
        removed_count: Number of spectra removed
        filter_type: Type of filter applied
    """
    if removed_count == 0:
        print(f"     {Fore.GREEN}No spectra removed due to {filter_type}.{Style.RESET_ALL}")
    elif removed_count == 1:
        print(f"     {Fore.YELLOW}1 spectrum removed due to {filter_type}.{Style.RESET_ALL}")
    else:
        print(f"     {Fore.YELLOW}{removed_count} spectra removed due to {filter_type}.{Style.RESET_ALL}")