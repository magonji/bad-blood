"""
File checker module for Bad Blood FTIR spectra processing.

This module provides utilities to analyze file naming schemes before processing,
identifying majority and minority naming formats and listing problematic files
for user correction.
"""

import os
import re
import datetime
import json
from typing import Tuple, Dict, List, Optional
from collections import Counter, defaultdict

import questionary
from colorama import Fore, Style
from tabulate import tabulate
from tqdm import tqdm

from bad_blood_pkg import utils
from bad_blood_pkg.console_ui import section as _section, explain as _explain

# File to store recently renamed files for quick re-verification
RENAMED_FILES_LOG = os.path.join(os.path.expanduser("~"), ".bad_blood_renamed_files.json")

# Console section headers / explanatory text live in console_ui (shared with
# the prepare and export steps).  _section and _explain are imported at the top.
_MINORITY_PREVIEW = 8   # max filenames shown per minority group (lists can be huge)


def _describe_scheme(scheme: Tuple[int, int, bool]) -> str:
    """Turn a (parts, dates, has_age) signature into plain words."""
    length, num_dates, has_age = scheme
    parts = [f"{length} name part" + ("" if length == 1 else "s")]
    parts.append(f"{num_dates} date" + ("" if num_dates == 1 else "s"))
    parts.append("an age tag" if has_age else "no age tag")
    return ", ".join(parts)


def save_renamed_files(renamed_files: List[Dict]) -> None:
    """
    Saves the list of renamed files to a JSON log for later re-verification.
    
    Args:
        renamed_files: List of dicts with 'old_path', 'new_path', 'timestamp'
    """
    try:
        # Load existing data if present
        existing = []
        if os.path.exists(RENAMED_FILES_LOG):
            with open(RENAMED_FILES_LOG, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        
        # Append new entries
        existing.extend(renamed_files)
        
        # Keep only the last 1000 entries to prevent unbounded growth
        existing = existing[-1000:]
        
        with open(RENAMED_FILES_LOG, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Could not save rename log: {e}{Style.RESET_ALL}")


def load_renamed_files() -> List[Dict]:
    """
    Loads the list of recently renamed files.
    
    Returns:
        List of dicts with 'old_path', 'new_path', 'timestamp'
    """
    if not os.path.exists(RENAMED_FILES_LOG):
        return []
    
    try:
        with open(RENAMED_FILES_LOG, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def clear_renamed_files_log() -> None:
    """Clears the renamed files log."""
    if os.path.exists(RENAMED_FILES_LOG):
        os.remove(RENAMED_FILES_LOG)


def verify_renamed_files(renamed_files: List[Dict]) -> Dict:
    """
    Re-verifies only the files that were recently renamed.
    
    Args:
        renamed_files: List of dicts with 'old_path', 'new_path', 'timestamp'
        
    Returns:
        Dictionary with verification results
    """
    date_extractor = re.compile(r'(?:^|-)(\d{6})(?=-|\.)')
    
    results = {
        'total_checked': 0,
        'verified_ok': [],
        'still_problematic': [],
        'not_found': [],
        'details': {}
    }
    
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Re-verifying Recently Renamed Files{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    for entry in renamed_files:
        new_path = entry.get('new_path')
        old_path = entry.get('old_path')
        
        if not new_path:
            continue
            
        results['total_checked'] += 1
        
        if not os.path.exists(new_path):
            results['not_found'].append(new_path)
            print(f"  {Fore.YELLOW}?{Style.RESET_ALL} {os.path.basename(new_path)} - File not found (moved or deleted?)")
            continue
        
        filename = os.path.basename(new_path)
        dates = date_extractor.findall(filename)
        
        if dates:
            analysis = _analyze_multiple_dates(dates, filename)
            if analysis['classification'] == 'DEFINITELY_DDMMYY':
                results['still_problematic'].append(new_path)
                results['details'][new_path] = [analysis['reason']]
                print(f"  {Fore.RED}✗{Style.RESET_ALL} {filename} - Still has date format issues")
            else:
                results['verified_ok'].append(new_path)
                print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {filename} - OK")
        else:
            results['verified_ok'].append(new_path)
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {filename} - OK (no dates to check)")
    
    # Summary
    print(f"\n{Fore.YELLOW}{'─'*70}{Style.RESET_ALL}")
    print(f"Re-verification Summary:")
    print(f"  Total checked: {results['total_checked']}")
    print(f"  {Fore.GREEN}Verified OK: {len(results['verified_ok'])}{Style.RESET_ALL}")
    if results['still_problematic']:
        print(f"  {Fore.RED}Still problematic: {len(results['still_problematic'])}{Style.RESET_ALL}")
    if results['not_found']:
        print(f"  {Fore.YELLOW}Not found: {len(results['not_found'])}{Style.RESET_ALL}")
    print()
    
    return results


def check_four_digit_year_dates(root_directory: str) -> Dict:
    """
    Detects files using YYYY-MM-DD format (4-digit year) instead of YY-MM-DD (2-digit year).
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        Dictionary containing:
        - 'files_with_4digit_years': List of tuples (file_path, detected_dates)
        - 'total_found': Total number of files with this issue
    """
    # Pattern for YYYYMMDD (8-digit dates) between delimiters
    # Must be exactly 8 digits to avoid matching 6-digit YYMMDD dates
    # Matches: 20240515, 2024-05-15, 2024_05_15 but NOT 201023 (which is YYMMDD)
    # The pattern requires the date to be between hyphens/underscores or at boundaries
    eight_digit_date_pattern = re.compile(
        r'(?:^|[-_])(\d{8})(?=[-_.]|$)'  # 8 consecutive digits between delimiters
    )
    # Also match dates with separators: YYYY-MM-DD or YYYY_MM_DD
    separated_date_pattern = re.compile(
        r'(?:^|[-_])(20[1-3]\d)[-_](\d{2})[-_](\d{2})(?=[-_.]|$)'
    )
    
    files_with_issues = []
    
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                detected_dates = []
                
                # Check for 8-digit continuous dates (YYYYMMDD)
                for match in eight_digit_date_pattern.finditer(filename):
                    date_str = match.group(1)
                    year = date_str[0:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    # Validate it looks like a real date (year 2010-2039, month 01-12, day 01-31)
                    if (year.startswith('20') and 
                        10 <= int(year[2:4]) <= 39 and
                        1 <= int(month) <= 12 and 
                        1 <= int(day) <= 31):
                        detected_dates.append((year, month, day))
                
                # Check for separated dates (YYYY-MM-DD or YYYY_MM_DD)
                for match in separated_date_pattern.finditer(filename):
                    year, month, day = match.groups()
                    if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                        detected_dates.append((year, month, day))
                
                if detected_dates:
                    file_path = os.path.join(current_path, filename)
                    files_with_issues.append((file_path, detected_dates))
    
    return {
        'files_with_4digit_years': files_with_issues,
        'total_found': len(files_with_issues)
    }


def rename_4digit_to_2digit_year(file_path: str) -> Tuple[str, str]:
    """
    Generates the corrected filename with 2-digit year format.
    
    Args:
        file_path: Original file path with 4-digit year
        
    Returns:
        Tuple of (original_path, new_path)
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Pattern to match various 4-digit year formats
    # YYYY-MM-DD, YYYY_MM_DD, or YYYYMMDD
    def replace_year(match):
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        # Get separator used (if any)
        full_match = match.group(0)
        
        # Determine the separator used
        if f"{year}-{month}-{day}" in full_match or f"{year}-{month}" in full_match:
            sep = "-"
        elif f"{year}_{month}_{day}" in full_match or f"{year}_{month}" in full_match:
            sep = "_"
        else:
            sep = ""
        
        # Convert to 2-digit year
        yy = year[2:]  # Take last 2 digits
        return f"{yy}{sep}{month}{sep}{day}"
    
    pattern = re.compile(r'(20[1-3]\d)([-_]?)(\d{2})([-_]?)(\d{2})')
    
    def smart_replace(match):
        year = match.group(1)
        sep1 = match.group(2)
        month = match.group(3)
        sep2 = match.group(4)
        day = match.group(5)
        yy = year[2:]
        return f"{yy}{sep1}{month}{sep2}{day}"
    
    new_filename = pattern.sub(smart_replace, filename)
    new_path = os.path.join(directory, new_filename)
    
    return file_path, new_path


def print_4digit_year_report(results: Dict) -> None:
    """
    Prints a formatted report of files with 4-digit year dates.
    
    Args:
        results: Dictionary from check_four_digit_year_dates()
    """
    files = results['files_with_4digit_years']

    if not files:
        print(f"{Fore.GREEN}✓ No four-digit-year dates found.{Style.RESET_ALL}\n")
        return

    print(f"\n{Fore.CYAN}Preview — YYYYMMDD → YYMMDD:{Style.RESET_ALL}\n")
    for i, (file_path, dates) in enumerate(sorted(files), 1):
        filename = os.path.basename(file_path)
        _, new_path = rename_4digit_to_2digit_year(file_path)
        new_filename = os.path.basename(new_path)
        print(f"  {i:3d}. {filename}")
        if new_filename != filename:
            print(f"       {Fore.GREEN}→ {new_filename}{Style.RESET_ALL}")
    print()


def rename_files_4digit_to_2digit(files_list: List[Tuple[str, List]]) -> Tuple[int, int]:
    """
    Renames files from 4-digit year to 2-digit year format.
    
    Args:
        files_list: List of (file_path, detected_dates) tuples
        
    Returns:
        Tuple of (successful_renames, failed_renames)
    """
    successful = 0
    failed = 0
    renamed_entries = []
    timestamp = datetime.datetime.now().isoformat()
    
    for file_path, _ in files_list:
        old_path, new_path = rename_4digit_to_2digit_year(file_path)
        
        if old_path == new_path:
            continue
            
        try:
            os.rename(old_path, new_path)
            successful += 1
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {os.path.basename(old_path)} → {os.path.basename(new_path)}")
            renamed_entries.append({
                'old_path': old_path,
                'new_path': new_path,
                'timestamp': timestamp,
                'type': '4digit_to_2digit'
            })
        except Exception as e:
            failed += 1
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {os.path.basename(old_path)}: {str(e)}")
    
    # Save renamed files log
    if renamed_entries:
        save_renamed_files(renamed_entries)
    
    return successful, failed


def rename_yymmdd_to_yyyymmdd(file_path: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """
    Generates the filename converting YYMMDD dates to YYYYMMDD format.
    
    Args:
        file_path: Original file path with YYMMDD dates
        
    Returns:
        Tuple of (original_path, new_path, list of (old_date, new_date) conversions)
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Pattern for 6-digit dates between hyphens or before extension
    date_pattern = re.compile(r'(?<=-)(\d{6})(?=-|\.)|^(\d{6})(?=-)')
    
    conversions = []
    
    def convert_date(match):
        date_str = match.group(1) or match.group(2)
        # Parse as YYMMDD
        yy = date_str[0:2]
        mm = date_str[2:4]
        dd = date_str[4:6]
        
        # Convert to YYYYMMDD (assume 20xx for years)
        new_date = f"20{yy}{mm}{dd}"
        conversions.append((date_str, new_date))
        return new_date
    
    new_filename = date_pattern.sub(convert_date, filename)
    new_path = os.path.join(directory, new_filename)
    
    return file_path, new_path, conversions


def preview_yymmdd_to_yyyymmdd_rename(files_list: List[str], max_preview: int = 10) -> None:
    """
    Shows a preview of how files would be renamed from YYMMDD to YYYYMMDD.
    
    Args:
        files_list: List of file paths with YYMMDD dates
        max_preview: Maximum number of files to preview
    """
    print(f"\n{Fore.CYAN}Preview of YYMMDD → YYYYMMDD conversions:{Style.RESET_ALL}\n")
    
    for i, file_path in enumerate(sorted(files_list)[:max_preview], 1):
        old_path, new_path, conversions = rename_yymmdd_to_yyyymmdd(file_path)
        old_filename = os.path.basename(old_path)
        new_filename = os.path.basename(new_path)
        
        print(f"  {i:3d}. {Fore.YELLOW}{old_filename}{Style.RESET_ALL}")
        print(f"       {Fore.GREEN}→ {new_filename}{Style.RESET_ALL}")
        
        for old_date, new_date in conversions:
            print(f"       {Fore.CYAN}({old_date} → {new_date}){Style.RESET_ALL}")
        print()
    
    if len(files_list) > max_preview:
        print(f"  {Fore.YELLOW}... and {len(files_list) - max_preview} more files{Style.RESET_ALL}\n")


def rename_files_yymmdd_to_yyyymmdd(files_list: List[str]) -> Tuple[int, int]:
    """
    Renames files from YYMMDD to YYYYMMDD date format.
    
    Args:
        files_list: List of file paths with YYMMDD dates
        
    Returns:
        Tuple of (successful_renames, failed_renames)
    """
    successful = 0
    failed = 0
    renamed_entries = []
    timestamp = datetime.datetime.now().isoformat()
    
    for file_path in files_list:
        old_path, new_path, conversions = rename_yymmdd_to_yyyymmdd(file_path)
        
        if old_path == new_path:
            continue
        
        # Check if target already exists
        if os.path.exists(new_path):
            failed += 1
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {os.path.basename(old_path)}: Target file already exists")
            continue
            
        try:
            os.rename(old_path, new_path)
            successful += 1
            conversion_str = ", ".join([f"{old}→{new}" for old, new in conversions])
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {os.path.basename(old_path)} → {os.path.basename(new_path)}")
            print(f"      {Fore.CYAN}Dates converted: {conversion_str}{Style.RESET_ALL}")
            renamed_entries.append({
                'old_path': old_path,
                'new_path': new_path,
                'timestamp': timestamp,
                'type': 'yymmdd_to_yyyymmdd',
                'conversions': conversions
            })
        except Exception as e:
            failed += 1
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {os.path.basename(old_path)}: {str(e)}")
    
    # Save renamed files log
    if renamed_entries:
        save_renamed_files(renamed_entries)
    
    return successful, failed


def get_files_with_yymmdd_dates(root_directory: str) -> List[str]:
    """
    Gets all .mzz files that have 6-digit dates (YYMMDD format).
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        List of file paths with YYMMDD dates
    """
    date_pattern = re.compile(r'(?:^|-)(\d{6})(?=-|\.)')
    files_with_dates = []
    
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                if date_pattern.search(filename):
                    files_with_dates.append(os.path.join(current_path, filename))
    
    return files_with_dates


def rename_ddmmyy_to_yymmdd(file_path: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """
    Generates the corrected filename converting DDMMYY dates to YYMMDD format.
    
    Args:
        file_path: Original file path with DDMMYY dates
        
    Returns:
        Tuple of (original_path, new_path, list of (old_date, new_date) conversions)
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Pattern for 6-digit dates between hyphens or before extension
    # Uses lookahead/lookbehind to avoid matching serial numbers
    date_pattern = re.compile(r'(?<=-)(\d{6})(?=-|\.)|^(\d{6})(?=-)')
    
    conversions = []
    
    def convert_date(match):
        date_str = match.group(1) or match.group(2)
        # Parse as DDMMYY
        dd = date_str[0:2]
        mm = date_str[2:4]
        yy = date_str[4:6]
        
        # Convert to YYMMDD
        new_date = f"{yy}{mm}{dd}"
        conversions.append((date_str, new_date))
        return new_date
    
    new_filename = date_pattern.sub(convert_date, filename)
    new_path = os.path.join(directory, new_filename)
    
    return file_path, new_path, conversions


def preview_ddmmyy_to_yymmdd_rename(files_list: List[str], max_preview: int = 10) -> None:
    """
    Shows a preview of how files would be renamed from DDMMYY to YYMMDD.
    
    Args:
        files_list: List of file paths with DDMMYY dates
        max_preview: Maximum number of files to preview
    """
    print(f"\n{Fore.CYAN}Preview of DDMMYY → YYMMDD conversions:{Style.RESET_ALL}\n")
    
    for i, file_path in enumerate(sorted(files_list)[:max_preview], 1):
        old_path, new_path, conversions = rename_ddmmyy_to_yymmdd(file_path)
        old_filename = os.path.basename(old_path)
        new_filename = os.path.basename(new_path)
        
        print(f"  {i:3d}. {Fore.RED}{old_filename}{Style.RESET_ALL}")
        print(f"       {Fore.GREEN}→ {new_filename}{Style.RESET_ALL}")
        
        for old_date, new_date in conversions:
            # Show interpreted dates
            dd, mm, yy = old_date[0:2], old_date[2:4], old_date[4:6]
            print(f"       {Fore.YELLOW}({old_date} = {dd}/{mm}/20{yy} → {new_date} = 20{yy}/{mm}/{dd}){Style.RESET_ALL}")
        print()
    
    if len(files_list) > max_preview:
        print(f"  {Fore.YELLOW}... and {len(files_list) - max_preview} more files{Style.RESET_ALL}\n")


def rename_files_ddmmyy_to_yymmdd(files_list: List[str], date_issue_details: Dict) -> Tuple[int, int]:
    """
    Renames files from DDMMYY to YYMMDD date format.
    
    Args:
        files_list: List of file paths with DDMMYY dates
        date_issue_details: Dictionary with issue details for context
        
    Returns:
        Tuple of (successful_renames, failed_renames)
    """
    successful = 0
    failed = 0
    renamed_entries = []
    timestamp = datetime.datetime.now().isoformat()
    
    for file_path in files_list:
        old_path, new_path, conversions = rename_ddmmyy_to_yymmdd(file_path)
        
        if old_path == new_path:
            continue
        
        # Check if target already exists
        if os.path.exists(new_path):
            failed += 1
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {os.path.basename(old_path)}: Target file already exists")
            continue
            
        try:
            os.rename(old_path, new_path)
            successful += 1
            conversion_str = ", ".join([f"{old}→{new}" for old, new in conversions])
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {os.path.basename(old_path)} → {os.path.basename(new_path)}")
            print(f"      {Fore.CYAN}Dates converted: {conversion_str}{Style.RESET_ALL}")
            renamed_entries.append({
                'old_path': old_path,
                'new_path': new_path,
                'timestamp': timestamp,
                'type': 'ddmmyy_to_yymmdd',
                'conversions': conversions
            })
        except Exception as e:
            failed += 1
            print(f"  {Fore.RED}✗{Style.RESET_ALL} {os.path.basename(old_path)}: {str(e)}")
    
    # Save renamed files log
    if renamed_entries:
        save_renamed_files(renamed_entries)
    
    return successful, failed


def check_date_format_issues(root_directory: str) -> Dict:
    """
    Analyzes date formats in .mzz files to detect potential DDMMYY vs YYMMDD issues.
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        Dictionary containing:
        - 'suspicious_files': List of files with potential date format issues
        - 'issue_details': Details about each suspicious file
    """
    suspicious_files = []
    issue_details = defaultdict(list)
    
    # Regex pattern for 6-digit sequences (potential dates) between hyphens or before extension
    date_pattern = re.compile(r'(?:^|-)(\d{6})(?=-|\.)')
    
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Date Format Checker - Detecting DDMMYY vs YYMMDD issues{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    print(f"{Fore.MAGENTA}Scanning: {root_directory}{Style.RESET_ALL}\n")
    
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                file_path = os.path.join(current_path, filename)
                dates = date_pattern.findall(filename)
                
                if not dates:
                    continue
                
                # Analyze all dates together for this file
                file_issues = _analyze_multiple_dates(dates, filename)
                
                if file_issues:
                    suspicious_files.append(file_path)
                    issue_details[file_path].extend(file_issues)
    
    return {
        'suspicious_files': suspicious_files,
        'issue_details': issue_details
    }


def _analyze_multiple_dates(dates: List[str], filename: str) -> Dict:
    """
    Analyzes multiple dates from the same file together using the new rules.
    
    Rules:
    1. All dates in a file have the same format (either all YYMMDD or all DDMMYY)
    2. Valid year range: 16 to current year (2-digit)
    3. DEFINITELY DDMMYY: if ANY date starts with value outside [16, current_year]
    4. DEFINITELY YYMMDD: if ALL dates start with value in [16, current_year] 
       AND at least one date ends with value outside [16, current_year]
    5. AMBIGUOUS: otherwise (requires global consensus to resolve)
    
    Args:
        dates: List of 6-digit date strings from the same file
        filename: Name of the file (for context)
        
    Returns:
        Dictionary with:
        - 'classification': 'DEFINITELY_DDMMYY', 'DEFINITELY_YYMMDD', or 'AMBIGUOUS'
        - 'reason': Human-readable explanation
        - 'dates_analysis': List of individual date analyses
    """
    MIN_YEAR = 16  # Experiments started in 2016
    current_year = int(datetime.datetime.now().strftime('%y'))
    
    # Analyze each date
    dates_analysis = [_analyze_single_date(d) for d in dates]
    
    # Check Rule 4: DEFINITELY DDMMYY
    # If ANY date starts with value outside valid year range
    dates_with_invalid_start = [
        d for d in dates_analysis 
        if d['first_two'] < MIN_YEAR or d['first_two'] > current_year
    ]
    
    if dates_with_invalid_start:
        invalid_examples = [
            f"'{d['date_str']}' starts with {d['first_two']:02d}" 
            for d in dates_with_invalid_start
        ]
        return {
            'classification': 'DEFINITELY_DDMMYY',
            'reason': (
                f"DDMMYY FORMAT (INCORRECT) - Date(s) start with value outside "
                f"valid year range [{MIN_YEAR}-{current_year}]: {', '.join(invalid_examples)}"
            ),
            'dates_analysis': dates_analysis
        }
    
    # Check Rule 5: DEFINITELY YYMMDD
    # All dates start with valid year AND at least one ends outside valid range
    all_start_valid = all(
        MIN_YEAR <= d['first_two'] <= current_year 
        for d in dates_analysis
    )
    
    dates_with_invalid_end = [
        d for d in dates_analysis 
        if d['last_two'] < MIN_YEAR or d['last_two'] > current_year
    ]
    
    if all_start_valid and dates_with_invalid_end:
        invalid_examples = [
            f"'{d['date_str']}' ends with {d['last_two']:02d}" 
            for d in dates_with_invalid_end
        ]
        return {
            'classification': 'DEFINITELY_YYMMDD',
            'reason': (
                f"YYMMDD FORMAT (CORRECT) - All dates start with valid year "
                f"[{MIN_YEAR}-{current_year}] and date(s) end outside this range: "
                f"{', '.join(invalid_examples)}"
            ),
            'dates_analysis': dates_analysis
        }
    
    # Rule 6: AMBIGUOUS - needs global consensus
    return {
        'classification': 'AMBIGUOUS',
        'reason': (
            f"AMBIGUOUS - All values could be valid in either format. "
            f"Dates: {', '.join(d['date_str'] for d in dates_analysis)}"
        ),
        'dates_analysis': dates_analysis
    }


def _analyze_single_date(date_str: str) -> Dict:
    """
    Analyzes a single 6-digit date string and returns its classification.
    
    Args:
        date_str: 6-digit date string
        
    Returns:
        Dictionary with:
        - 'first_two': int value of first two digits
        - 'last_two': int value of last two digits
        - 'middle_two': int value of middle two digits
        - 'as_yymmdd': formatted string if interpreted as YYMMDD
        - 'as_ddmmyy': formatted string if interpreted as DDMMYY
    """
    first_two = int(date_str[0:2])
    middle_two = int(date_str[2:4])
    last_two = int(date_str[4:6])
    
    return {
        'date_str': date_str,
        'first_two': first_two,
        'middle_two': middle_two,
        'last_two': last_two,
        'as_yymmdd': f"20{first_two:02d}-{middle_two:02d}-{last_two:02d}",
        'as_ddmmyy': f"{first_two:02d}/{middle_two:02d}/20{last_two:02d}"
    }


def print_date_format_report(date_check_results: Dict) -> None:
    """
    Prints a formatted report of date format issues.
    
    Args:
        date_check_results: Dictionary from check_date_format_issues()
    """
    suspicious_files = date_check_results['suspicious_files']
    issue_details = date_check_results['issue_details']

    if not suspicious_files:
        print(f"{Fore.GREEN}✓ Nothing to show.{Style.RESET_ALL}\n")
        return

    print()
    for i, file_path in enumerate(sorted(suspicious_files), 1):
        print(f"{Fore.CYAN}  {i:3d}. {os.path.basename(file_path)}{Style.RESET_ALL}")
        for issue in issue_details.get(file_path, []):
            print(f"       {issue}")
    print()


def export_date_issues(date_check_results: Dict, export_path: Optional[str] = None) -> None:
    """
    Exports date format issues to a text file.
    
    Args:
        date_check_results: Dictionary from check_date_format_issues()
        export_path: Path for export file
    """
    suspicious_files = date_check_results['suspicious_files']
    issue_details = date_check_results['issue_details']
    
    if not suspicious_files:
        print(f"{Fore.GREEN}No issues to export.{Style.RESET_ALL}\n")
        return
    
    if not export_path:
        export_path = os.path.join(os.getcwd(), "date_format_issues.txt")
    
    export_lines = []
    export_lines.append("=" * 80)
    export_lines.append("Bad Blood - Date Format Issues Report")
    export_lines.append("=" * 80)
    export_lines.append(f"\nTotal files with potential issues: {len(suspicious_files)}\n")
    export_lines.append("Legend:")
    export_lines.append("  YYMMDD = Year-Month-Day (CORRECT format)")
    export_lines.append("  DDMMYY = Day-Month-Year (INCORRECT format)\n")
    export_lines.append("=" * 80)
    
    for i, file_path in enumerate(sorted(suspicious_files), 1):
        export_lines.append(f"\n{i}. {file_path}")
        for issue in issue_details[file_path]:
            export_lines.append(f"   - {issue}")
    
    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(export_lines))
        print(f"{Fore.GREEN}✓ Date issues report exported to: {export_path}{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}✗ Error exporting report: {str(e)}{Style.RESET_ALL}\n")


def _looks_like_yyyymmdd(s: str) -> bool:
    """True if an 8-digit token is a plausible YYYYMMDD date (year 20xx)."""
    return (len(s) == 8 and s.startswith('20') and
            10 <= int(s[2:4]) <= 39 and
            1 <= int(s[4:6]) <= 12 and
            1 <= int(s[6:8]) <= 31)


def check_files(root_directory: str) -> Dict:
    """
    Analyzes all .mzz files in a directory to identify naming scheme patterns.
    
    Date format detection rules:
    1. All dates in a file have the same format (DDMMYY or YYMMDD)
    2. Valid year range: 16 to current year
    3. DEFINITELY DDMMYY: if ANY date starts with value outside [16, current_year]
    4. DEFINITELY YYMMDD: if ALL dates start with valid year AND ANY ends outside [16, current_year]
    5. AMBIGUOUS files assume the majority format from the ENTIRE directory
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'total_files': Total number of .mzz files found
        - 'scheme_counts': Counter of different naming schemes
        - 'scheme_files': Dictionary mapping schemes to file paths
        - 'most_common_scheme': The most frequently occurring scheme
        - 'files_with_definite_date_errors': Files with DEFINITE DDMMYY format
        - 'files_with_ambiguous_dates': Files with ambiguous dates (after global resolution)
        - 'date_issue_details': Details of all date issues
        - 'files_resolved_by_global_consensus': Files resolved by global majority
    """
    total_files = 0
    metadata_counts = Counter()
    scheme_files = defaultdict(list)
    
    # Date classification storage
    files_definitely_ddmmyy = []  # Definite DDMMYY (incorrect)
    files_definitely_yymmdd = []  # Definite YYMMDD (correct)
    files_ambiguous = []          # Ambiguous (need global consensus)
    date_issue_details = defaultdict(list)
    date_analysis_results = {}    # Store full analysis for each file
    
    # Regex patterns
    date_pattern = re.compile(r'(?:^|-)(\d{6})(?=-|\.)')          # YYMMDD  (2-digit year)
    date8_pattern = re.compile(r'(?:^|[-_])(\d{8})(?=[-_.]|$)')   # YYYYMMDD (4-digit year)
    age_pattern = utils.AGE_TAG_RE                                # ##D/##W/##M/##Y

    _section("CHECKING FILES")
    print(f"{Fore.LIGHTBLACK_EX}Scanning {root_directory}{Style.RESET_ALL}\n")
    
    # =========================================================================
    # FIRST PASS: Collect all .mzz files
    # =========================================================================
    print(f"{Fore.CYAN}Collecting files...{Style.RESET_ALL}")
    all_mzz_files = []
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                all_mzz_files.append((current_path, filename))
    
    if not all_mzz_files:
        print(f"{Fore.RED}No .mzz files were found in the selected directory.{Style.RESET_ALL}")
        return None
    
    print(f"{Fore.GREEN}Found {len(all_mzz_files)} .mzz files{Style.RESET_ALL}\n")
    
    # =========================================================================
    # SECOND PASS: Analyze files with progress bar
    # =========================================================================
    for current_path, filename in tqdm(all_mzz_files, desc="Analyzing files", unit="file", 
                                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        total_files += 1
        file_path = os.path.join(current_path, filename)
        
        # Extract metadata from filename (excluding extension)
        name_without_ext = filename[:-4] if filename.lower().endswith('.mzz') else filename
        parts = name_without_ext.split('-')
        metadata = parts
        
        # Count dates and check for an age tag (days/weeks/months/years). Both
        # YYMMDD and YYYYMMDD count as one date each, so a 4-digit-year file
        # shares the same naming scheme as its 2-digit peers instead of being
        # flagged as a minority.
        six_digit_dates = date_pattern.findall(filename)
        eight_digit_dates = [d for d in date8_pattern.findall(filename)
                             if _looks_like_yyyymmdd(d)]
        num_dates = len(six_digit_dates) + len(eight_digit_dates)
        has_age = bool(age_pattern.search(filename))

        # Create metadata signature
        metadata_key = (len(metadata), num_dates, has_age)
        metadata_counts[metadata_key] += 1
        scheme_files[metadata_key].append(file_path)

        # Date-order checks (DDMMYY vs YYMMDD) apply only to 2-digit-year dates;
        # 4-digit-year dates are unambiguous and handled by the format step.
        if six_digit_dates:
            analysis = _analyze_multiple_dates(six_digit_dates, filename)
            date_analysis_results[file_path] = analysis
            
            if analysis['classification'] == 'DEFINITELY_DDMMYY':
                files_definitely_ddmmyy.append(file_path)
                date_issue_details[file_path].append(analysis['reason'])
            elif analysis['classification'] == 'DEFINITELY_YYMMDD':
                files_definitely_yymmdd.append(file_path)
                # No issue to report for correct files
            else:  # AMBIGUOUS
                files_ambiguous.append(file_path)
                date_issue_details[file_path].append(analysis['reason'])
    
    # =========================================================================
    # THIRD PASS: Resolve ambiguous files using GLOBAL consensus
    # =========================================================================
    files_resolved_by_global = []
    final_definite_errors = list(files_definitely_ddmmyy)  # Start with definite errors
    final_ambiguous = []
    
    if files_ambiguous:
        num_definite_ddmmyy = len(files_definitely_ddmmyy)
        num_definite_yymmdd = len(files_definitely_yymmdd)
        
        # Determine global majority format
        if num_definite_ddmmyy > num_definite_yymmdd:
            # Majority is DDMMYY (incorrect) - ambiguous files are also DDMMYY
            global_consensus = 'DDMMYY'
            for file_path in files_ambiguous:
                final_definite_errors.append(file_path)
                files_resolved_by_global.append(file_path)
                original_reason = date_issue_details[file_path][0] if date_issue_details[file_path] else ""
                date_issue_details[file_path] = [
                    f"RESOLVED BY GLOBAL CONSENSUS: Marked as DDMMYY (incorrect) - "
                    f"directory has {num_definite_ddmmyy} definite DDMMYY vs {num_definite_yymmdd} definite YYMMDD files",
                    f"   Original: {original_reason}"
                ]
        elif num_definite_yymmdd > num_definite_ddmmyy:
            # Majority is YYMMDD (correct) - ambiguous files are fine
            global_consensus = 'YYMMDD'
            for file_path in files_ambiguous:
                files_resolved_by_global.append(file_path)
                # Remove from issue details since they're resolved as correct
                if file_path in date_issue_details:
                    del date_issue_details[file_path]
        elif num_definite_ddmmyy == 0 and num_definite_yymmdd == 0:
            # No definite files at all - all remain ambiguous
            global_consensus = None
            final_ambiguous = files_ambiguous
            print(f"{Fore.YELLOW}No definite date formats found in directory. "
                  f"{len(files_ambiguous)} file(s) remain ambiguous.{Style.RESET_ALL}\n")
        else:
            # Equal number of definite files - ambiguous remain ambiguous
            global_consensus = None
            final_ambiguous = files_ambiguous
            print(f"{Fore.YELLOW}Equal number of DDMMYY ({num_definite_ddmmyy}) and YYMMDD ({num_definite_yymmdd}) files. "
                  f"{len(files_ambiguous)} file(s) remain ambiguous.{Style.RESET_ALL}\n")
        
        if global_consensus and files_resolved_by_global:
            print(f"{Fore.CYAN}Resolved {len(files_resolved_by_global)} ambiguous file(s) by global consensus "
                  f"(majority format: {global_consensus}).{Style.RESET_ALL}\n")
    
    # Determine most common naming scheme
    most_common_scheme = metadata_counts.most_common(1)[0][0]
    
    return {
        'total_files': total_files,
        'scheme_counts': metadata_counts,
        'scheme_files': scheme_files,
        'most_common_scheme': most_common_scheme,
        'files_with_definite_date_errors': final_definite_errors,
        'files_with_ambiguous_dates': final_ambiguous,
        'date_issue_details': date_issue_details,
        'files_resolved_by_global_consensus': files_resolved_by_global,
        'root_directory': root_directory
    }


def print_analysis_summary(analysis_results: Dict) -> None:
    """
    Prints a formatted summary of the file analysis.
    
    Args:
        analysis_results: Dictionary from check_files() containing analysis data
    """
    total_files = analysis_results['total_files']
    scheme_counts = analysis_results['scheme_counts']
    most_common_scheme = analysis_results['most_common_scheme']

    _section("NAMING SCHEMES")
    _explain(
        "Bad Blood reads data (sample, date, ...) from each file name, split",
        "on \"-\". Files are grouped by structure: how many parts they have,",
        "how many dates, and whether they carry an age tag such as 03D (days),",
        "6W (weeks), 12M (months) or 2Y (years). The structure most files share",
        "is treated as the expected one; files that differ from it may fail to",
        "load.",
    )
    print(f"\nScanned {Fore.GREEN}{total_files}{Style.RESET_ALL} .mzz files.\n")

    # One row per structure — date problems get their own section later.
    table_data = []
    for scheme, count in scheme_counts.most_common():
        length, num_dates, has_age = scheme
        percentage = (count / total_files) * 100
        is_majority = scheme == most_common_scheme
        status = "✓ expected" if is_majority else "differs"
        table_data.append([
            status, count, f"{percentage:.0f}%", length, num_dates,
            "yes" if has_age else "no",
        ])

    headers = ["", "Files", "Share", "Name parts", "Dates", "Age tag"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    if len(scheme_counts) == 1:
        print(f"\n{Fore.GREEN}✓ All files share the same structure.{Style.RESET_ALL}")
    else:
        n_diff = total_files - scheme_counts[most_common_scheme]
        print(f"\n{Fore.YELLOW}{n_diff} file(s) differ from the expected structure "
              f"({_describe_scheme(most_common_scheme)}).{Style.RESET_ALL}")
    print()


def list_minority_files(analysis_results: Dict, export: bool = False,
                       export_path: Optional[str] = None, show: bool = True) -> None:
    """
    Lists all files with minority naming schemes and optionally exports them.

    Args:
        analysis_results: Dictionary from check_files() containing analysis data
        export: Whether to export the list to a text file
        export_path: Path for the export file (if export=True)
        show: Whether to print the list to the console (False when only saving)
    """
    scheme_counts = analysis_results['scheme_counts']
    scheme_files = analysis_results['scheme_files']
    most_common_scheme = analysis_results['most_common_scheme']
    total_files = analysis_results['total_files']
    root = analysis_results.get('root_directory', '')

    minority_schemes = {k: v for k, v in scheme_counts.items() if k != most_common_scheme}
    if not minority_schemes:
        if show:
            print(f"{Fore.GREEN}✓ All files follow the expected structure.{Style.RESET_ALL}\n")
        return

    total_minority = sum(minority_schemes.values())
    expected_desc = _describe_scheme(most_common_scheme)

    if show:
        print(f"{Fore.YELLOW}The {total_minority} file(s) below don't match the expected "
              f"structure{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}(expected: {expected_desc}). Each group shows how it "
              f"differs.{Style.RESET_ALL}\n")

    export_lines = ["Bad Blood — files that differ from the expected naming structure",
                    f"Expected: {expected_desc}",
                    f"{total_minority} of {total_files} files differ", ""]

    truncated = False
    for i, (scheme, count) in enumerate(sorted(minority_schemes.items(),
                                               key=lambda x: x[1], reverse=True), 1):
        header = f"  Group {i} — {_describe_scheme(scheme)}  ({count} file(s)):"
        if show:
            print(f"{Fore.CYAN}{header}{Style.RESET_ALL}")
        export_lines.append(header)

        for j, file_path in enumerate(sorted(scheme_files[scheme])):
            rel = os.path.relpath(file_path, root) if root else file_path
            export_lines.append(f"      {rel}")
            # Only a bounded sample is printed — the list can be huge.
            if show and j < _MINORITY_PREVIEW:
                print(f"      {rel}")
        if show and count > _MINORITY_PREVIEW:
            truncated = True
            print(f"      {Fore.LIGHTBLACK_EX}... and {count - _MINORITY_PREVIEW} more"
                  f"{Style.RESET_ALL}")
        if show:
            print()
        export_lines.append("")

    if show and truncated:
        _explain(f"Only the first {_MINORITY_PREVIEW} of each group are shown above — "
                 f"save to a file to see them all.")

    if export:
        if not export_path:
            export_path = os.path.join(os.getcwd(), "minority_files_report.txt")
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(export_lines))
            print(f"{Fore.GREEN}✓ Saved to: {export_path}{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Could not save the file: {str(e)}{Style.RESET_ALL}\n")


def interactive_file_check(root_directory: Optional[str] = None) -> Dict:
    """
    Interactive mode for file checking with user prompts.
    
    Args:
        root_directory: Optional root directory path. If None, prompts user.
        
    Returns:
        Dictionary containing analysis results
    """
    # Get directory if not provided
    if not root_directory:
        root_directory = questionary.path(
            "Select the root directory to analyze:",
        ).ask()
        
        if not root_directory:
            print(f"{Fore.RED}No directory selected. Exiting.{Style.RESET_ALL}")
            return None
    
    # Validate directory
    if not os.path.exists(root_directory):
        print(f"{Fore.RED}Directory does not exist: {root_directory}{Style.RESET_ALL}")
        return None
    
    # Perform analysis
    analysis_results = check_files(root_directory)
    
    if not analysis_results:
        return None
    
    # Print summary
    print_analysis_summary(analysis_results)
    
    # --- Minority naming schemes: show them, then one optional save -------
    scheme_counts = analysis_results['scheme_counts']
    has_minority = len(scheme_counts) > 1

    if has_minority:
        list_minority_files(analysis_results)          # always show the odd files
        save_list = questionary.confirm(
            "Save this list to a text file (in the scanned folder)?",
            default=False,
        ).ask()
        if save_list:
            export_path = os.path.join(root_directory, "minority_files_report.txt")
            list_minority_files(analysis_results, export=True, export_path=export_path, show=False)

    # --- Date formats ------------------------------------------------------
    _section("DATE FORMATS")
    _explain(
        "Dates in file names are read as year-month-day. Two-digit years",
        "(YYMMDD, e.g. 240515) and four-digit years (YYYYMMDD, e.g. 20240515)",
        "are both fine. Bad Blood flags dates that look like DAY-month-year by",
        "mistake, and can standardise everything onto one style.",
    )

    files_with_definite_errors = analysis_results.get('files_with_definite_date_errors', [])
    files_with_ambiguous = analysis_results.get('files_with_ambiguous_dates', [])

    if not files_with_definite_errors and not files_with_ambiguous:
        print(f"\n{Fore.GREEN}✓ No day/month-order problems found.{Style.RESET_ALL}")

    # Files that look like DAY-month-year (DDMMYY) — very likely wrong.
    if files_with_definite_errors:
        n = len(files_with_definite_errors)
        print(f"\n{Fore.RED}✗ {n} file(s) look like DAY-month-year (DDMMYY).{Style.RESET_ALL}")
        _explain(
            "  e.g. 150524 meaning 15 May 2024, where Bad Blood expects 240515.",
            "  Left as they are, these dates will be read incorrectly.",
        )
        date_results = {
            'suspicious_files': files_with_definite_errors,
            'issue_details': {k: v for k, v in analysis_results.get('date_issue_details', {}).items()
                             if k in files_with_definite_errors},
        }
        if questionary.confirm(f"Show the {n} affected file(s)?", default=True).ask():
            print_date_format_report(date_results)

        if questionary.confirm(
            f"Rename them to YYMMDD (fix the day/month order)?", default=False
        ).ask():
            preview_ddmmyy_to_yymmdd_rename(files_with_definite_errors)
            if questionary.confirm(f"Proceed and rename {n} file(s)?", default=False).ask():
                print(f"\n{Fore.CYAN}Renaming files...{Style.RESET_ALL}\n")
                successful, failed = rename_files_ddmmyy_to_yymmdd(
                    files_with_definite_errors,
                    analysis_results.get('date_issue_details', {}),
                )
                print(f"\n{Fore.GREEN}✓ Renamed: {successful}{Style.RESET_ALL}")
                if failed > 0:
                    print(f"{Fore.RED}✗ Failed: {failed}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Renaming cancelled.{Style.RESET_ALL}")

    # Files whose date can't be resolved automatically — need a human eye.
    if files_with_ambiguous:
        n = len(files_with_ambiguous)
        print(f"\n{Fore.YELLOW}? {n} file(s) have dates that can't be resolved automatically.{Style.RESET_ALL}")
        _explain(
            "  Both readings are plausible (e.g. 120324) and the folder has no",
            "  clear majority, so Bad Blood won't guess. Review them by hand.",
        )
        date_results = {
            'suspicious_files': files_with_ambiguous,
            'issue_details': {k: v for k, v in analysis_results.get('date_issue_details', {}).items()
                             if k in files_with_ambiguous},
        }
        if questionary.confirm(f"Show the {n} file(s) to review?", default=False).ask():
            print_date_format_report(date_results)

    # =====================================================================
    # Date-format preference: YYMMDD (2-digit year) vs YYYYMMDD (4-digit year)
    #
    # Both are treated as valid formats — neither is an error. If BOTH coexist
    # in the directory we ask which one to standardise on and rename the others.
    # Only offered once the DDMMYY / ambiguous checks above are clean, so a
    # mis-ordered date is never blindly converted.
    # =====================================================================
    files_with_definite_errors = analysis_results.get('files_with_definite_date_errors', [])
    files_with_ambiguous = analysis_results.get('files_with_ambiguous_dates', [])

    if not files_with_definite_errors and not files_with_ambiguous:
        four_digit_results = check_four_digit_year_dates(root_directory)
        files_yyyymmdd = four_digit_results['files_with_4digit_years']   # list of (path, dates)
        files_yymmdd = get_files_with_yymmdd_dates(root_directory)       # list of paths
        n8, n6 = len(files_yyyymmdd), len(files_yymmdd)

        if n6 > 0 and n8 > 0:
            print(f"\n{Fore.YELLOW}This folder mixes two date styles:{Style.RESET_ALL}")
            print(f"    YYMMDD   (2-digit year): {Fore.CYAN}{n6}{Style.RESET_ALL} file(s)")
            print(f"    YYYYMMDD (4-digit year): {Fore.CYAN}{n8}{Style.RESET_ALL} file(s)")
            _explain("  You can rename the files so they all share one style.")

            choice_yy = f"Use YYMMDD   — rename the {n8} four-digit file(s)"
            choice_yyyy = f"Use YYYYMMDD — rename the {n6} two-digit file(s)"
            choice_keep = "Leave the names as they are"
            preferred = questionary.select(
                "Which date style should all files use?",
                choices=[choice_yy, choice_yyyy, choice_keep],
            ).ask()

            if preferred == choice_yy:
                # Standardise on 2-digit year: rename the YYYYMMDD files down.
                print_4digit_year_report(four_digit_results)
                if questionary.confirm(f"Rename {n8} file(s) to YYMMDD?", default=False).ask():
                    print(f"\n{Fore.CYAN}Renaming files...{Style.RESET_ALL}\n")
                    successful, failed = rename_files_4digit_to_2digit(files_yyyymmdd)
                    print(f"\n{Fore.GREEN}✓ Renamed: {successful}{Style.RESET_ALL}")
                    if failed > 0:
                        print(f"{Fore.RED}✗ Failed: {failed}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Left unchanged.{Style.RESET_ALL}")

            elif preferred == choice_yyyy:
                # Standardise on 4-digit year: rename the YYMMDD files up.
                preview_yymmdd_to_yyyymmdd_rename(files_yymmdd)
                if questionary.confirm(f"Rename {n6} file(s) to YYYYMMDD?", default=False).ask():
                    print(f"\n{Fore.CYAN}Renaming files...{Style.RESET_ALL}\n")
                    successful, failed = rename_files_yymmdd_to_yyyymmdd(files_yymmdd)
                    print(f"\n{Fore.GREEN}✓ Renamed: {successful}{Style.RESET_ALL}")
                    if failed > 0:
                        print(f"{Fore.RED}✗ Failed: {failed}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Left unchanged.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Left unchanged.{Style.RESET_ALL}")

        elif n6 > 0:
            print(f"\n{Fore.GREEN}✓ All dates use one style (YYMMDD).{Style.RESET_ALL}")
        elif n8 > 0:
            print(f"\n{Fore.GREEN}✓ All dates use one style (YYYYMMDD).{Style.RESET_ALL}")

    print()
    return analysis_results


def handle_check_command(args: List[str]) -> Optional[Dict]:
    """
    Handles the logic for the 'check' command.
    
    Args:
        args: Command line arguments. Supports:
              - directory path
              - --recheck: Re-verify only recently renamed files
              - --clear-log: Clear the renamed files log
        
    Returns:
        Dictionary with analysis results if successful, None otherwise
    """
    # Check for special flags
    if '--recheck' in args:
        renamed_files = load_renamed_files()
        if not renamed_files:
            print(f"{Fore.YELLOW}No recently renamed files found. Run a full check first.{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.CYAN}Found {len(renamed_files)} recently renamed files in log.{Style.RESET_ALL}")
        results = verify_renamed_files(renamed_files)
        
        # Ask if user wants to clear the log after successful verification
        if results['verified_ok'] and not results['still_problematic']:
            clear_log = questionary.confirm(
                "All files verified OK. Would you like to clear the rename log?",
                default=True
            ).ask()
            if clear_log:
                clear_renamed_files_log()
                print(f"{Fore.GREEN}✓ Rename log cleared.{Style.RESET_ALL}")
        
        return results
    
    if '--clear-log' in args:
        clear_renamed_files_log()
        print(f"{Fore.GREEN}✓ Rename log cleared.{Style.RESET_ALL}")
        return None
    
    if '--show-log' in args:
        renamed_files = load_renamed_files()
        if not renamed_files:
            print(f"{Fore.YELLOW}No recently renamed files in log.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.CYAN}Recently renamed files ({len(renamed_files)} entries):{Style.RESET_ALL}\n")
            for i, entry in enumerate(renamed_files[-20:], 1):  # Show last 20
                print(f"  {i}. {os.path.basename(entry.get('old_path', '?'))} → {os.path.basename(entry.get('new_path', '?'))}")
                print(f"     {Fore.LIGHTBLACK_EX}Type: {entry.get('type', '?')} | {entry.get('timestamp', '?')}{Style.RESET_ALL}")
            if len(renamed_files) > 20:
                print(f"\n  {Fore.YELLOW}... and {len(renamed_files) - 20} more entries{Style.RESET_ALL}")
        return None
    
    # Determine target directory (filter out flags)
    target_directory = None
    for arg in args:
        if not arg.startswith('--'):
            target_directory = arg
            break
    
    # Run interactive file check
    return interactive_file_check(target_directory)


# Quick standalone execution
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║        Bad Blood - File Checker (Standalone Mode)                ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    interactive_file_check()