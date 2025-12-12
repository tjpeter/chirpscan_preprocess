#!/usr/bin/env python3
"""
Unzip all archives from a source directory into a target directory,
preserving subfolder structure. 

Usage:
  python unzip_subfolders.py --source <source_dir> --target <target_dir>
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
        description="Unzip all archives from source directory to target directory, preserving subfolder structure."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Source directory containing zip files (default: {DEFAULT_SOURCE_DIR})"
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


def process_all_zips(source_dir: Path, target_dir: Path) -> None:
    """
    Process all zip files in source directory and extract to target directory.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    # Find all zip files (including in subdirectories)
    zip_files = list(source_dir.rglob("*.zip"))
    
    if not zip_files:
        logging.warning(f"No zip files found in {source_dir}")
        return
    
    logging.info(f"Found {len(zip_files)} zip file(s) to extract")
    
    for zip_path in zip_files:
        try:
            # Create subdirectory structure in target matching source structure
            relative_path = zip_path.parent.relative_to(source_dir)
            output_dir = target_dir / relative_path / zip_path.stem
            
            safe_extract_zip(zip_path, output_dir)
            logging.info(f"✓ Successfully extracted {zip_path.name}")
            
        except Exception as e:
            logging.error(f"✗ Failed to extract {zip_path.name}: {e}")
            continue


# =============================================
# Main
# =============================================

def main():
    args = parse_args()
    
    logging.info(f"Source directory: {args.source}")
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