#!/usr/bin/env python3
"""
Download and unpack the ECOSoundSet dataset from Zenodo.

Downloads files to:
/cfs/earth/scratch/icls/shared/icls-02092025-bioacoustics/insects/data/ECOSoundSet

Usage:
  python download_ecosoundset.py
  python download_ecosoundset.py --out ./my_data
  python download_ecosoundset.py --verify-only
"""

import argparse
import hashlib
import os
import shutil
import zipfile
import logging
import coloredlogs
import sys
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# =============================================
# Configuration
# =============================================

DEFAULT_OUT_DIR = "/cfs/earth/scratch/icls/shared/icls-02092025-bioacoustics/insects/data/ECOSoundSet"

RESOURCES = [
    {
        "filename": "abiotic_sound_labels.csv",
        "url": "https://zenodo.org/records/17086328/files/abiotic_sound_labels.csv?download=1",
        "md5": "e6c71ccb5d4a1de90d1eb72157e833ff",
        "extract": False,
    },
    {
        "filename": "annotated_audio_segments.csv",
        "url": "https://zenodo.org/records/17086328/files/annotated_audio_segments.csv?download=1",
        "md5": "b69828e29e5bd90a8c483f9f63c72e13",
        "extract": False,
    },
    {
        "filename": "annotated_audio_segments_by_label_summary.csv",
        "url": "https://zenodo.org/records/17086328/files/annotated_audio_segments_by_label_summary.csv?download=1",
        "md5": "df7051c08eaf0c0ba553df35db921d60",
        "extract": False,
    },
    {
        "filename": "online_recordings_metadata.csv",
        "url": "https://zenodo.org/records/17086328/files/online_recordings_metadata.csv?download=1",
        "md5": "886aae89b619bf106b609b143a2bcfac",
        "extract": False,
    },
    {
        "filename": "recording_metadata.csv",
        "url": "https://zenodo.org/records/17086328/files/recording_metadata.csv?download=1",
        "md5": "cb9f9c46591393d81f404acf0c56b2b9",
        "extract": False,
    },
    {
        "filename": "Split recordings.zip",
        "url": "https://zenodo.org/records/17086328/files/Split%20recordings.zip?download=1",
        "md5": "397f657f60e7f7b199b0ec80230f9773",
        "extract": True,
    },
    {
        "filename": "Whole recordings.zip",
        "url": "https://zenodo.org/records/17086328/files/Whole%20recordings.zip?download=1",
        "md5": "870860bead4eea6249f033e1ec96a5b5",
        "extract": True,
    },
]

# =============================================
# Functions
# =============================================

def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path, desc: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    # Resume if partial exists (basic HTTP Range support)
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total = resp.getheader("Content-Length")
            total = int(total) + resume_pos if total is not None else None

            if tqdm:
                bar = tqdm(
                    total=total, initial=resume_pos, unit="B", unit_scale=True, desc=desc or dest.name
                )
            else:
                logging.info(f"Downloading {url} -> {dest}")

            mode = "ab" if resume_pos > 0 else "wb"
            with tmp.open(mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    if tqdm:
                        bar.update(len(chunk))

            if tqdm:
                bar.close()
    except urllib.error.HTTPError as e:
        # Handle case where range is not satisfiable (e.g. file fully downloaded in part file but not renamed)
        if e.code == 416:
            logging.warning(f"HTTP 416 Range Not Satisfiable for {url}. Assuming download complete.")
        else:
            raise

    if tmp.exists():
        tmp.replace(dest)


def safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    """
    Extract zip defensively (avoid path traversal).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

        for member in zf.infolist():
            member_path = target_dir / member.filename
            if not is_within_directory(target_dir, member_path):
                raise Exception("Unsafe path in zip file: %s" % member.filename)
            zf.extract(member, target_dir)


def ensure_md5(path: Path, expected: str) -> None:
    logging.info(f"Verifying md5: {path.name}")
    actual = md5sum(path)
    if actual != expected:
        raise RuntimeError(
            f"MD5 mismatch for {path.name}\n expected: {expected}\n   actual: {actual}"
        )


def process_resource(
    resource: dict,
    out_dir: Path,
    extract: bool,
    verify_only: bool,
) -> None:
    filename = resource["filename"]
    url = resource["url"]
    md5 = resource["md5"]
    should_extract = resource["extract"]
    
    dest_path = out_dir / filename
    
    if not verify_only:
        if not dest_path.exists():
            download(url, dest_path, desc=filename)
            
    if dest_path.exists():
        ensure_md5(dest_path, md5)
        
        if extract and should_extract and not verify_only:
            logging.info(f"Extracting {filename}...")
            safe_extract_zip(dest_path, out_dir)
    else:
        if verify_only:
            logging.warning(f"File {filename} not found, skipping verification.")
        else:
            logging.error(f"File {filename} failed to download.")


# =============================================
# Parse CMD arguments
# =============================================    

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""

    p = argparse.ArgumentParser(description="Download ECOSoundSet dataset.")

    p.add_argument(
        "--out",
        type=Path,
        default=Path(DEFAULT_OUT_DIR),
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Do not extract archives (only download & verify checksums).",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify md5 of already-downloaded archives.",
    )

    return p.parse_args()


# =============================================
# Main
# =============================================    

def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    
    for res in RESOURCES:
        logging.info(f"\n=== Processing: {res['filename']} ===")
        process_resource(
            resource=res,
            out_dir=args.out,
            extract=not args.no_extract,
            verify_only=args.verify_only,
        )

    logging.info("\nDone.")


# =============================================
# Entry
# =============================================   

if __name__ == "__main__":
    # Parse the command line arguments
    arguments = parse_args()
    # configure logging
    logging_level = logging.INFO
    coloredlogs.install(level=logging_level)

    main(arguments)
