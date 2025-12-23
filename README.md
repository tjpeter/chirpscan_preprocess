# chirpscan_preprocess

Preprocessing tools for bioacoustic data, including downloading, extracting, and exploring audio datasets for insect sound analysis.

## Overview

This repository provides utilities for:

- **Downloading** the ECOSoundSet dataset from Zenodo with checksum verification
- **Extracting** zip archives while preserving folder structure
- **Exploring** audio segments with an interactive spectrogram viewer

## Installation

Create the conda environment:

```bash
mamba env create -f environment.yml
conda activate preprocess_chirpscan
```

## Project Structure

```
├── data/                      # Audio files and metadata (gitignored)
├── download_unzip/            # Download and extraction scripts
│   ├── download_ecosoundset.py/sh   # ECOSoundSet downloader
│   └── unzip_files.py/sh            # Generic zip extractor
├── exploration/               # Data exploration tools
│   ├── explore_ecosoundset.py       # Panel-based spectrogram viewer
│   └── run_ecosoundset_explorer.sh  # SLURM launcher for viewer
├── environment.yml            # Conda environment specification
└── README.md
```

## Usage

### Download ECOSoundSet

```bash
# Local execution
python download_unzip/download_ecosoundset.py --out ./data/ECOSoundSet

# SLURM cluster
sbatch download_unzip/download_ecosoundset.sh
```

Options:
- `--out <path>` — Output directory
- `--no-extract` — Download only, skip extraction
- `--verify-only` — Verify checksums of existing files

### Extract Zip Archives

```bash
# Single file or directory
python download_unzip/unzip_files.py --source <path_to_zip_or_dir> --target <output_dir>

# SLURM cluster
sbatch download_unzip/unzip_files.sh
```

### Explore Audio Data

Launch the interactive spectrogram explorer:

```bash
# Local
panel serve exploration/explore_ecosoundset.py

# SLURM cluster (creates SSH tunnel instructions)
sbatch exploration/run_ecosoundset_explorer.sh
```

The explorer provides:
- Multi-segment spectrogram visualization
- Audio playback
- Filtering by label and category
- Annotation overlay boxes

## Dependencies

Key packages (see `environment.yml`):
- `librosa` — Audio loading and spectrogram computation
- `panel` / `holoviews` / `bokeh` — Interactive visualization
- `pandas` — Metadata handling
- `tqdm` / `coloredlogs` — CLI utilities

## Data

The `data/` folder contains:
- CSV metadata files for ECOSoundSet annotations
- Audio recordings organized by license type
- Additional field recordings

## License

[Add your license here]