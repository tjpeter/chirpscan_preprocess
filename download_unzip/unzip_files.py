#!/usr/bin/env python3
"""
Unzip archives from a source directory or single zip file into a target directory,
preserving subfolder structure. 

Usage:
  python unzip_subfolders.py --source <source_dir_or_zip> --target <target_dir>
  python unzip_subfolders.py  # Uses default directories
"""

import logging
import coloredlogs
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm


# =============================================
# Configuration
# =============================================

DEFAULT_SOURCE_DIR = Path("/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/zip-archives/Kanton_Aargau_zips")
DEFAULT_TARGET_DIR = Path("/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/Kanton_Aargau")

# =============================================
# Parser Arguments
# =============================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unzip archives from source directory or single zip file to target directory, preserving subfolder structure."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source directory containing zip files or path to single zip file (default: {DEFAULT_SOURCE_DIR})"
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Target directory for extracted files (default: {DEFAULT_TARGET_DIR})"
    )
    return parser.parse_args()


# =============================================
# Functions
# =============================================

def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """
    Extract zip defensively (avoid path traversal).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Extracting {zip_path.name} to {target_dir}")
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        
        for member in tqdm(members, desc=f"Extracting {zip_path.name}", unit="file"):
            member_path = target_dir / member
            # Security check: ensure extracted path is within target directory
            if not str(member_path.resolve()).startswith(str(target_dir.resolve())):
                raise Exception(f"Unsafe path in zip file: {member}")
            zf.extract(member, target_dir)


def process_single_zip(zip_path: Path, target_dir: Path, source_base: Path = None) -> None:
    """
    Process a single zip file and extract to target directory.
    
    Args:
        zip_path: Path to the zip file
        target_dir: Base target directory
        source_base: Optional base directory for preserving relative structure
    """
    try:
        # If source_base is provided, preserve relative structure
        if source_base and source_base.is_dir():
            relative_path = zip_path.parent.relative_to(source_base)
            output_dir = target_dir / relative_path / zip_path.stem
        else:
            # For single file, just use zip filename as subdirectory
            output_dir = target_dir / zip_path.stem
        
        safe_extract_zip(zip_path, output_dir)
        logging.info(f"✓ Successfully extracted {zip_path.name}")
        
    except Exception as e:
        logging.error(f"✗ Failed to extract {zip_path.name}: {e}")
        raise


def process_all_zips(source: Path, target_dir: Path) -> None:
    """
    Process zip file(s) from source and extract to target directory.
    
    Args:
        source: Path to directory containing zip files OR path to single zip file
        target_dir: Target directory for extracted files
    """
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    
    # Check if source is a single zip file
    if source.is_file():
        if source.suffix.lower() != '.zip':
            raise ValueError(f"Source file is not a zip file: {source}")
        
        logging.info(f"Processing single zip file: {source.name}")
        process_single_zip(source, target_dir)
        return
    
    # Source is a directory - find all zip files
    if not source.is_dir():
        raise ValueError(f"Source is neither a file nor a directory: {source}")
    
    zip_files = list(source.rglob("*.zip"))
    
    if not zip_files:
        logging.warning(f"No zip files found in {source}")
        return
    
    logging.info(f"Found {len(zip_files)} zip file(s) to extract")
    
    for zip_path in zip_files:
        try:
            process_single_zip(zip_path, target_dir, source_base=source)
        except Exception as e:
            logging.error(f"Continuing with next file...")
            continue


# =============================================
# Main
# =============================================

def main():
    args = parse_args()
    
    source_type = "file" if args.source.is_file() else "directory"
    logging.info(f"Source {source_type}: {args.source}")
    logging.info(f"Target directory: {args.target}")
    logging.info("")
    
    process_all_zips(args.source, args.target)
    
    logging.info("\nDone.")


# =============================================
# Entry
# =============================================

if __name__ == "__main__":
    # Configure logging
    coloredlogs.install(level=logging.INFO)
    
    main()