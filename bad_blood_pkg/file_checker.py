"""
File checker module for Bad Blood FTIR spectra processing.

This module provides utilities to analyze file naming schemes before processing,
identifying majority and minority naming formats and listing problematic files
for user correction.
"""

import os
import re
import datetime
from typing import Tuple, Dict, List, Optional
from collections import Counter, defaultdict

import questionary
from colorama import Fore, Style
from tabulate import tabulate


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
    
    # Regex pattern for 6-digit sequences (potential dates)
    date_pattern = re.compile(r'\b(\d{6})\b')
    
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


def _analyze_multiple_dates(dates: List[str], filename: str) -> List[str]:
    """
    Analyzes multiple dates from the same file together.
    Files can have up to 3 dates, all in the same format (either all YYMMDD or all DDMMYY).
    If any date is definitively wrong, the whole file is wrong.
    If any date is definitively correct, the whole file is correct.
    
    Args:
        dates: List of 6-digit date strings from the same file
        filename: Name of the file (for context)
        
    Returns:
        List of issue descriptions (empty if no issues or file is correct)
    """
    issues = []
    
    # Analyze each date individually
    definite_errors = []  # Dates that are DEFINITELY DDMMYY format (wrong)
    definite_correct = []  # Dates that are DEFINITELY YYMMDD format (correct)
    ambiguous = []  # Dates that could be either
    
    for date_str in dates:
        date_issues = _analyze_date_format(date_str)
        
        if date_issues:
            # Check if it's a definite error
            if any('DEFINITELY' in issue for issue in date_issues):
                definite_errors.append((date_str, date_issues))
            else:
                ambiguous.append((date_str, date_issues))
        else:
            # No issues means it's consistent with YYMMDD format
            # But we need to check if day >31, which would confirm YYMMDD
            yy_format_dd = int(date_str[4:6])
            if yy_format_dd > 31:
                definite_correct.append(date_str)
    
    # Decision logic based on all dates
    if definite_errors:
        # At least one date is definitely wrong (DDMMYY)
        # This means ALL dates in the file are DDMMYY format
        issues.append(
            f"🔴 FILE HAS DDMMYY FORMAT (INCORRECT) - "
            f"Found {len(definite_errors)} definite error(s) out of {len(dates)} date(s)"
        )
        for date_str, date_issues in definite_errors:
            for issue in date_issues:
                issues.append(f"   {issue}")
        
        # If there are other dates that looked ambiguous, they're also wrong
        if ambiguous:
            issues.append(
                f"   → The other {len(ambiguous)} date(s) in this file are also DDMMYY (same format)"
            )
            for date_str, _ in ambiguous:
                issues.append(f"      '{date_str}' is also DDMMYY")
    
    elif definite_correct:
        # At least one date is definitely correct (YYMMDD with day >31)
        # This means ALL dates are in correct YYMMDD format - no issue to report
        return []
    
    elif ambiguous and len(dates) > 1:
        # All dates are ambiguous, but we have multiple dates
        # Check if they're all the same pattern - if so, it's suspicious
        issues.append(
            f"⚠️  AMBIGUOUS - {len(dates)} dates, all individually ambiguous - verify manually"
        )
        for date_str, date_issues in ambiguous:
            for issue in date_issues:
                issues.append(f"   {issue}")
    
    elif ambiguous:
        # Single ambiguous date
        for date_str, date_issues in ambiguous:
            issues.extend(date_issues)
    
    return issues


def _analyze_date_format(date_str: str) -> List[str]:
    """
    Analyzes a 6-digit date string to detect format issues.
    
    Args:
        date_str: 6-digit date string
        
    Returns:
        List of issue descriptions (empty if no issues detected)
    """
    issues = []
    
    # Parse as YYMMDD (correct format)
    yy_format_yy = int(date_str[0:2])
    yy_format_mm = int(date_str[2:4])
    yy_format_dd = int(date_str[4:6])
    
    # Parse as DDMMYY (incorrect format)
    dd_format_dd = int(date_str[0:2])
    dd_format_mm = int(date_str[2:4])
    dd_format_yy = int(date_str[4:6])
    
    # Get current year (2-digit format)
    now = datetime.datetime.now()
    current_year = int(now.strftime('%y'))
    
    # Check 0: Year range validation (MOST DEFINITIVE CHECK)
    # Experiments started in 2016 (YY=16) and should not be in the future
    # If first two digits are <16 OR >current_year, it's definitely a day (DDMMYY format error)
    if yy_format_yy < 16 or yy_format_yy > current_year:
        reason = ""
        if yy_format_yy < 16:
            reason = f"Year field ({yy_format_yy}) < 16 (pre-2016)"
        else:
            reason = f"Year field ({yy_format_yy}) > {current_year} (future year)"
        
        issues.append(
            f"'{date_str}': {reason} if YYMMDD format - "
            f"DEFINITELY DDMMYY ({dd_format_dd:02d}/{dd_format_mm:02d}/20{dd_format_yy:02d})"
        )
        return issues  # This is definitive, no need to check further
    
    # Check 1: Month field validation (YYMMDD format)
    # If interpreting as YYMMDD, the month (positions 2-4) should be 01-12
    if yy_format_mm > 12:
        issues.append(
            f"'{date_str}': Month field ({yy_format_mm}) > 12 if YYMMDD format - "
            f"possibly DDMMYY ({dd_format_dd:02d}/{dd_format_mm:02d}/20{dd_format_yy:02d})"
        )
    
    # Check 2: Day field validation (YYMMDD format)
    # If interpreting as YYMMDD, the day (positions 4-6) should be 01-31
    elif yy_format_dd > 31:
        issues.append(
            f"'{date_str}': Day field ({yy_format_dd}) > 31 if YYMMDD format - "
            f"possibly DDMMYY ({dd_format_dd:02d}/{dd_format_mm:02d}/20{dd_format_yy:02d})"
        )
    
    # Check 3: Suspicious year field (near-future dates)
    # If first two digits are plausible day (01-31) and last two could be year
    # Only flag if it's suspiciously close to current year (within 1 year in future)
    elif dd_format_dd <= 31 and dd_format_yy == current_year + 1:
        # This is ambiguous - could be either format
        # Add warning only if month is also valid for DDMMYY interpretation
        if dd_format_mm <= 12:
            issues.append(
                f"'{date_str}': Ambiguous - could be either "
                f"YYMMDD (20{yy_format_yy:02d}/{yy_format_mm:02d}/{yy_format_dd:02d}) or "
                f"DDMMYY ({dd_format_dd:02d}/{dd_format_mm:02d}/20{dd_format_yy:02d}) - verify manually"
            )
    
    return issues


def print_date_format_report(date_check_results: Dict) -> None:
    """
    Prints a formatted report of date format issues.
    
    Args:
        date_check_results: Dictionary from check_date_format_issues()
    """
    suspicious_files = date_check_results['suspicious_files']
    issue_details = date_check_results['issue_details']
    
    if not suspicious_files:
        print(f"{Fore.GREEN}✓ No date format issues detected!{Style.RESET_ALL}\n")
        return
    
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Potential Date Format Issues Found{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")
    print(f"⚠ Found {Fore.RED}{len(suspicious_files)}{Style.RESET_ALL} files with potential date format issues:\n")
    
    for i, file_path in enumerate(sorted(suspicious_files), 1):
        print(f"{Fore.CYAN}{i}. {file_path}{Style.RESET_ALL}")
        for issue in issue_details[file_path]:
            print(f"   {Fore.YELLOW}⚠{Style.RESET_ALL} {issue}")
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
            export_lines.append(f"   ⚠ {issue}")
    
    try:
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(export_lines))
        print(f"{Fore.GREEN}✓ Date issues report exported to: {export_path}{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"{Fore.RED}✗ Error exporting report: {str(e)}{Style.RESET_ALL}\n")


def check_files(root_directory: str) -> Dict:
    """
    Analyzes all .mzz files in a directory to identify naming scheme patterns.
    
    Args:
        root_directory: Root directory path to search for .mzz files
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'total_files': Total number of .mzz files found
        - 'scheme_counts': Counter of different naming schemes
        - 'scheme_files': Dictionary mapping schemes to file paths
        - 'most_common_scheme': The most frequently occurring scheme
        - 'files_with_definite_date_errors': Files with DEFINITE DDMMYY format
        - 'files_with_ambiguous_dates': Files with ambiguous dates
        - 'date_issue_details': Details of all date issues
    """
    total_files = 0
    metadata_counts = Counter()
    scheme_files = defaultdict(list)
    files_with_definite_errors = []
    files_with_ambiguous = []
    date_issue_details = defaultdict(list)
    
    # Regex patterns for extracting metadata elements
    date_pattern = re.compile(r'\d{6}')  # YYMMDD format
    days_pattern = re.compile(r'\d{2}D')  # ##D format
    date_extractor = re.compile(r'\b(\d{6})\b')  # For date format checking
    
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}File Checker - Analyzing directory structure{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    print(f"{Fore.MAGENTA}Scanning: {root_directory}{Style.RESET_ALL}\n")
    
    # Recursively search for .mzz files
    for current_path, _, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith('.mzz'):
                total_files += 1
                file_path = os.path.join(current_path, filename)
                
                # Extract metadata from filename
                parts = filename.split('-')
                metadata = parts[:-1]
                
                # Count dates and check for days notation
                num_dates = len(date_pattern.findall(filename))
                has_days = bool(days_pattern.search(filename))
                
                # Create metadata signature
                metadata_key = (len(metadata), num_dates, has_days)
                metadata_counts[metadata_key] += 1
                scheme_files[metadata_key].append(file_path)
                
                # Check for date format issues
                dates = date_extractor.findall(filename)
                if dates:
                    issues = _analyze_multiple_dates(dates, filename)
                    if issues:
                        date_issue_details[file_path].extend(issues)
                        
                        # Determine if definite error or ambiguous
                        is_definite = any('🔴' in issue or 'DEFINITELY' in issue for issue in issues)
                        is_ambiguous = any('⚠️' in issue and 'AMBIGUOUS' in issue for issue in issues)
                        
                        if is_definite:
                            files_with_definite_errors.append(file_path)
                        elif is_ambiguous:
                            files_with_ambiguous.append(file_path)
    
    # Handle case where no files are found
    if not metadata_counts:
        print(f"{Fore.RED}❌ No .mzz files were found in the selected directory.{Style.RESET_ALL}")
        return None
    
    # Determine most common naming scheme
    most_common_scheme = metadata_counts.most_common(1)[0][0]
    
    return {
        'total_files': total_files,
        'scheme_counts': metadata_counts,
        'scheme_files': scheme_files,
        'most_common_scheme': most_common_scheme,
        'files_with_definite_date_errors': files_with_definite_errors,
        'files_with_ambiguous_dates': files_with_ambiguous,
        'date_issue_details': date_issue_details
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
    files_with_definite_errors = analysis_results.get('files_with_definite_date_errors', [])
    files_with_ambiguous = analysis_results.get('files_with_ambiguous_dates', [])
    
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}File Analysis Summary{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")
    print(f"Total .mzz files found: {Fore.GREEN}{total_files}{Style.RESET_ALL}")
    
    # Show date issues summary if any
    if files_with_definite_errors or files_with_ambiguous:
        if files_with_definite_errors:
            print(f"Files with DEFINITE date errors: {Fore.RED}{len(files_with_definite_errors)}{Style.RESET_ALL}")
        if files_with_ambiguous:
            print(f"Files with AMBIGUOUS dates: {Fore.YELLOW}{len(files_with_ambiguous)}{Style.RESET_ALL}")
        print()
    else:
        print()
    
    # Prepare table data
    table_data = []
    for scheme, count in scheme_counts.most_common():
        length, num_dates, has_days = scheme
        percentage = (count / total_files) * 100
        
        is_majority = scheme == most_common_scheme
        status = f"{Fore.GREEN}✓ MAJORITY{Style.RESET_ALL}" if is_majority else f"{Fore.YELLOW}⚠ MINORITY{Style.RESET_ALL}"
        
        table_data.append([
            status,
            count,
            f"{percentage:.1f}%",
            length,
            num_dates,
            "Yes" if has_days else "No"
        ])
    
    # Add definite date errors row if there are any
    if files_with_definite_errors:
        definite_percentage = (len(files_with_definite_errors) / total_files) * 100
        table_data.append([
            f"{Fore.RED}❌ DATE ERRORS{Style.RESET_ALL}",
            len(files_with_definite_errors),
            f"{definite_percentage:.1f}%",
            "—",
            "—",
            "—"
        ])
    
    # Add ambiguous dates row if there are any
    if files_with_ambiguous:
        ambiguous_percentage = (len(files_with_ambiguous) / total_files) * 100
        table_data.append([
            f"{Fore.YELLOW}⚠️  AMBIGUOUS{Style.RESET_ALL}",
            len(files_with_ambiguous),
            f"{ambiguous_percentage:.1f}%",
            "—",
            "—",
            "—"
        ])
    
    headers = ["Status", "Files", "%", "Metadata Parts", "Dates (YYMMDD)", "Has Days (##D)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print()


def list_minority_files(analysis_results: Dict, export: bool = False, 
                       export_path: Optional[str] = None) -> None:
    """
    Lists all files with minority naming schemes and optionally exports them.
    
    Args:
        analysis_results: Dictionary from check_files() containing analysis data
        export: Whether to export the list to a text file
        export_path: Path for the export file (if export=True)
    """
    scheme_counts = analysis_results['scheme_counts']
    scheme_files = analysis_results['scheme_files']
    most_common_scheme = analysis_results['most_common_scheme']
    total_files = analysis_results['total_files']
    
    # Get minority schemes
    minority_schemes = {k: v for k, v in scheme_counts.items() if k != most_common_scheme}
    
    if not minority_schemes:
        print(f"{Fore.GREEN}✓ All files follow the majority naming scheme!{Style.RESET_ALL}\n")
        return
    
    total_minority = sum(minority_schemes.values())
    
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Files with Minority Naming Schemes{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")
    print(f"⚠ Found {Fore.RED}{total_minority}{Style.RESET_ALL} files ({(total_minority/total_files)*100:.1f}%) "
          f"with non-standard naming:\n")
    
    # Prepare export content if needed
    export_lines = []
    if export:
        export_lines.append("=" * 80)
        export_lines.append("Bad Blood - Files with Minority Naming Schemes")
        export_lines.append("=" * 80)
        export_lines.append(f"\nTotal minority files: {total_minority} ({(total_minority/total_files)*100:.1f}%)")
        export_lines.append(f"Total files: {total_files}\n")
    
    # List files grouped by scheme
    for i, (scheme, count) in enumerate(sorted(minority_schemes.items(), 
                                               key=lambda x: x[1], reverse=True), 1):
        length, num_dates, has_days = scheme
        
        print(f"{Fore.CYAN}Minority Scheme #{i}{Style.RESET_ALL} ({count} files):")
        print(f"  • Metadata parts: {length}")
        print(f"  • Dates (YYMMDD): {num_dates}")
        print(f"  • Has days (##D): {'Yes' if has_days else 'No'}\n")
        
        if export:
            export_lines.append(f"\n{'─' * 80}")
            export_lines.append(f"Minority Scheme #{i} ({count} files):")
            export_lines.append(f"  Metadata parts: {length}")
            export_lines.append(f"  Dates (YYMMDD): {num_dates}")
            export_lines.append(f"  Has days (##D): {'Yes' if has_days else 'No'}\n")
        
        # List files for this scheme
        files = scheme_files[scheme]
        for j, file_path in enumerate(sorted(files), 1):
            print(f"    {j:3d}. {file_path}")
            if export:
                export_lines.append(f"    {j:3d}. {file_path}")
        
        print()
    
    # Export if requested
    if export:
        if not export_path:
            export_path = os.path.join(os.getcwd(), "minority_files_report.txt")
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(export_lines))
            print(f"{Fore.GREEN}✓ Report exported to: {export_path}{Style.RESET_ALL}\n")
        except Exception as e:
            print(f"{Fore.RED}✗ Error exporting report: {str(e)}{Style.RESET_ALL}\n")


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
            qmark="📂"
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
    
    # Check if there are minority files
    scheme_counts = analysis_results['scheme_counts']
    most_common_scheme = analysis_results['most_common_scheme']
    has_minority = len(scheme_counts) > 1
    
    if has_minority:
        # Ask if user wants to see minority files
        show_minority = questionary.confirm(
            "Would you like to see the list of files with minority naming schemes?",
            default=True
        ).ask()
        
        if show_minority:
            # Ask if user wants to export
            export_list = questionary.confirm(
                "Would you like to export this list to a text file?",
                default=False
            ).ask()
            
            export_path = None
            if export_list:
                export_path = questionary.text(
                    "Enter export file path (or press Enter for default 'minority_files_report.txt'):",
                    default="minority_files_report.txt"
                ).ask()
            
            list_minority_files(analysis_results, export=export_list, export_path=export_path)
    else:
        print(f"{Fore.GREEN}✓ All files follow the same naming scheme!{Style.RESET_ALL}\n")
    
    # Check for DEFINITE date format errors
    files_with_definite_errors = analysis_results.get('files_with_definite_date_errors', [])
    if files_with_definite_errors:
        check_definite = questionary.confirm(
            f"Would you like to see the list of {len(files_with_definite_errors)} files with DEFINITE date errors (DDMMYY format)?",
            default=True
        ).ask()
        
        if check_definite:
            # Create date results for definite errors
            date_results = {
                'suspicious_files': files_with_definite_errors,
                'issue_details': {k: v for k, v in analysis_results.get('date_issue_details', {}).items() 
                                 if k in files_with_definite_errors}
            }
            print_date_format_report(date_results)
            
            # Offer to export
            export_definite = questionary.confirm(
                "Would you like to export the definite date errors report?",
                default=False
            ).ask()
            
            if export_definite:
                definite_export_path = questionary.text(
                    "Enter export file path (or press Enter for default 'definite_date_errors.txt'):",
                    default="definite_date_errors.txt"
                ).ask()
                export_date_issues(date_results, export_path=definite_export_path)
    
    # Check for AMBIGUOUS dates
    files_with_ambiguous = analysis_results.get('files_with_ambiguous_dates', [])
    if files_with_ambiguous:
        check_ambiguous = questionary.confirm(
            f"Would you like to see the list of {len(files_with_ambiguous)} files with AMBIGUOUS dates (require manual verification)?",
            default=False  # Default to No since these require manual review
        ).ask()
        
        if check_ambiguous:
            # Create date results for ambiguous dates
            date_results = {
                'suspicious_files': files_with_ambiguous,
                'issue_details': {k: v for k, v in analysis_results.get('date_issue_details', {}).items() 
                                 if k in files_with_ambiguous}
            }
            print_date_format_report(date_results)
            
            # Offer to export
            export_ambiguous = questionary.confirm(
                "Would you like to export the ambiguous dates report?",
                default=False
            ).ask()
            
            if export_ambiguous:
                ambiguous_export_path = questionary.text(
                    "Enter export file path (or press Enter for default 'ambiguous_dates.txt'):",
                    default="ambiguous_dates.txt"
                ).ask()
                export_date_issues(date_results, export_path=ambiguous_export_path)
    
    return analysis_results


def handle_check_command(args: List[str]) -> Optional[Dict]:
    """
    Handles the logic for the 'check' command.
    
    Args:
        args: Command line arguments (optional directory path)
        
    Returns:
        Dictionary with analysis results if successful, None otherwise
    """
    # Determine target directory
    if args:
        target_directory = args[0]
    else:
        target_directory = None
    
    # Run interactive file check
    return interactive_file_check(target_directory)


# Quick standalone execution
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║        Bad Blood - File Checker (Standalone Mode)                ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}\n")
    
    interactive_file_check()