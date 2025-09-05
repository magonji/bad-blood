# Bad Blood 2.0

**Bad Blood** is a comprehensive command-line tool for processing FTIR (Fourier Transform Infrared) spectral data from `.mzz` files. Developed at the University of Glasgow, it provides automated quality control, metadata extraction, and flexible data export capabilities for spectroscopic research.

## Features

### Core Functionality
- **Automated File Analysis**: Recursively scans directories for `.mzz` spectral files and analyzes naming schemes
- **Unified Data Matrix**: Creates interpolated spectral matrices with consistent wavenumber grids
- **Metadata Extraction**: Intelligent parsing of filename metadata with automatic date and age detection
- **Quality Control**: Multi-stage filtering system for problematic spectra

### Quality Control Filters
1. **Abnormal Background Detection**: Identifies interferometer errors using IQR-based outlier detection at 1900 cm⁻¹
2. **Atmospheric Interference Detection**: Detects water vapor and CO₂ contamination using polynomial fitting (3900-3500 cm⁻¹)
3. **Low-Intensity Signal Detection**: Filters spectra with insufficient signal strength in the fingerprint region (600-400 cm⁻¹)

### Interactive Features
- **Wavenumber Selection**: Visual preview and selection of specific spectral regions
- **Metadata Filtering**: Interactive filtering based on categorical metadata
- **Export Options**: Multiple file formats (Parquet, CSV, Excel) with customizable column selection

## Installation

### From Source
```bash
git clone https://github.com/magonji/bad-blood.git
cd bad-blood
pip install -e .
```

### Requirements
- Python 3.8 or higher
- See `requirements.txt` for detailed dependencies

## Quick Start

After installation, launch Bad Blood from the command line:

```bash
bad-blood
```

### Basic Workflow

1. **Load spectral data**:
   ```
   > load "/path/to/your/data"
   ```
   Or simply use `load` to open a file browser.

2. **Apply quality control**:
   ```
   > prepare
   ```

3. **Export processed data**:
   ```
   > export
   ```

### Command Reference

| Command | Description |
|---------|-------------|
| `load "directory"` | Load and analyze `.mzz` files from specified directory |
| `prepare` | Apply quality control filters to remove problematic spectra |
| `export` | Export processed data with interactive options |
| `help` | Display available commands |
| `exit` | Exit the program |

## File Format Support

### Input
- **`.mzz` files**: Compressed spectral data files containing:
  - Wavenumber range and resolution information
  - Absorbance data arrays
  - Embedded metadata in filenames

### Output
- **Parquet file**

## Metadata Naming Schemes

Bad Blood automatically detects and processes various metadata naming schemes:

### Supported Patterns
- **Date Format**: `YYMMDD` (e.g., `240815` for August 15, 2024)
- **Age Format**: `##D` (e.g., `5D` for 5 days old)
- **Categorical Data**: Any alphanumeric identifiers

### Example Filename
```
experiment1-UK-rep1-240815-240820-240825-mosquito-5D-alive-rear1-nulliparous-susceptible-unexposed-001.mzz
```

This would be parsed as:
- experiment: experiment1
- country: UK
- replica: rep1
- collection_date: 240815
- killing_date: 240820
- measurement_date: 240825
- species: mosquito
- age: 5D
- status: alive
- rear: rear1
- parity: nulliparous
- resistance: susceptible
- exposure: unexposed

## Quality Control Details

### Abnormal Background Filter
- **Method**: Interquartile Range (IQR) outlier detection
- **Target**: 1900 cm⁻¹ absorbance values
- **Threshold**: 2.5 × IQR (adjustable)
- **Purpose**: Remove spectra with interferometer errors

### Atmospheric Interference Filter
- **Method**: 5th-degree polynomial fitting
- **Region**: 3900-3500 cm⁻¹
- **Threshold**: R² < 0.96
- **Purpose**: Detect water vapor and CO₂ contamination

### Low-Intensity Filter
- **Method**: Average absorbance calculation
- **Region**: 600-400 cm⁻¹ plateau
- **Threshold**: < 0.11 absorbance units
- **Purpose**: Remove spectra with insufficient signal

## Configuration

### Default Parameters
- Background filter threshold: 2.5 × IQR
- Atmospheric interference R² threshold: 0.96
- Low-intensity threshold: 0.11 absorbance units
- Fingerprint region: 600-400 cm⁻¹
- Atmospheric region: 3900-3500 cm⁻¹

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
git clone https://github.com/magonji/bad-blood.git
cd bad-blood
pip install -e ".[dev]"
```

## Citation

If you use Bad Blood in your research, please cite:

```bibtex
@software{gonzalez_jimenez_2025_bad_blood,
  author       = {González-Jiménez, Mario},
  title        = {Bad Blood: FTIR Spectral Analysis Tool},
  version      = {2.0.0},
  year         = {2025},
  institution  = {University of Glasgow},
  url          = {https://github.com/magonji/bad-blood}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Mario González-Jiménez**  
University of Glasgow  
Email: mario.gonzalezjimenez@glasgow.ac.uk

## Changelog

### Version 2.0.0 (August 2025)
- Complete rewrite with modular architecture
- Interactive quality control with visual feedback
- Smart metadata detection and naming
- Multiple export formats support
- Comprehensive error handling
- Improved user experience with colored terminal output

## Support

For questions, bug reports, or feature requests, please:
1. Check the [documentation](https://github.com/magonji/bad-blood/wiki)
2. Search [existing issues](https://github.com/magonji/bad-blood/issues)
3. Create a [new issue](https://github.com/magonji/bad-blood/issues/new) if needed



