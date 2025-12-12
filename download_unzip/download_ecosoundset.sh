#!/usr/bin/env bash

#
#---------------------- SLURM DIRECTIVES ----------------------#
#SBATCH --job-name=download_ecosoundset
#SBATCH --mail-type=end
#SBATCH --mail-user=peeb@zhaw.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --partition=earth-3
#SBATCH --constraint=rhel8
#SBATCH --mem=32G
#SBATCH --output=download_ecosoundset.%j.out
#SBATCH --error=download_ecosoundset.%j.err
#SBATCH --chdir=/cfs/earth/scratch/peeb/jobs/insects
#--------------------------------------------------------------#

set -euo pipefail

# Load environment (USS 2022 stack on RHEL8)
module load USS/2022
module load gcc/9.4.0-pe5.34
module load unzip/6.0

# Base data directory
DATA_DIR="/cfs/earth/scratch/icls/shared/icls-02092025-bioacoustics/insects/data/ECOSoundSet"
mkdir -p "$DATA_DIR"

# Files to download
FILES=(
  "abiotic_sound_labels.csv"
  "annotated_audio_segments.csv"
  "annotated_audio_segments_by_label_summary.csv"
  "online_recordings_metadata.csv"
  "recording_metadata.csv"
  "Split recordings.zip"
  "Whole recordings.zip"
)

# Expected MD5s hashes (from Zenodo)
declare -A MD5S=(
  ["abiotic_sound_labels.csv"]="e6c71ccb5d4a1de90d1eb72157e833ff"
  ["annotated_audio_segments.csv"]="b69828e29e5bd90a8c483f9f63c72e13"
  ["annotated_audio_segments_by_label_summary.csv"]="df7051c08eaf0c0ba553df35db921d60"
  ["online_recordings_metadata.csv"]="886aae89b619bf106b609b143a2bcfac"
  ["recording_metadata.csv"]="cb9f9c46591393d81f404acf0c56b2b9"
  ["Split recordings.zip"]="397f657f60e7f7b199b0ec80230f9773"
  ["Whole recordings.zip"]="870860bead4eea6249f033e1ec96a5b5"
)

BASE_URL="https://zenodo.org/records/17086328/files"

for FILENAME in "${FILES[@]}"; do
  OUT="${DATA_DIR}/${FILENAME}"

  # Handle URL encoding for filenames with spaces
  URL_FILENAME="${FILENAME// /%20}"
  URL="${BASE_URL}/${URL_FILENAME}?download=1"

  echo "=== Downloading: $FILENAME ==="
  wget -c "$URL" -O "$OUT"

  echo "Verifying MD5 for $FILENAME..."
  EXPECTED="${MD5S[$FILENAME]}"
  ACTUAL="$(md5sum "$OUT" | awk '{print $1}')"
  if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "MD5 mismatch for $FILENAME"
    echo "   expected: $EXPECTED"
    echo "   actual:   $ACTUAL"
    rm -f "$OUT"
    exit 1
  fi
  echo "MD5 OK"

  if [[ "$FILENAME" == *.zip ]]; then
    echo "Unzipping $FILENAME..."
    unzip -o "$OUT" -d "$DATA_DIR"
    rm -f "$OUT"   # delete zip after extraction
  fi

  echo "Done: $FILENAME"
  echo
done

echo "All downloads, verifications, and extractions completed!"
