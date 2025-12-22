#!/bin/bash
#
#---------------------- SLURM DIRECTIVES ----------------------#
#SBATCH --job-name=download_ecosoundset
#SBATCH --mail-type=end
#SBATCH --mail-user=peeb@zhaw.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --partition=earth-3
#SBATCH --constraint=rhel8
#SBATCH --mem=8G
#SBATCH --output=download_ecosoundset.%x.%j.out
#SBATCH --error=download_ecosoundset.%x.%j.err
#SBATCH --chdir=/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/logs
#--------------------------------------------------------------#

set -euo pipefail

# ---------------------- CONFIG ---------------------- #

# Paths
PY_SCRIPT="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/download_unzip/download_ecosoundset.py"
REPO_DIR="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess"
OUTPUT_DIR="/cfs/earth/scratch/icls/shared/icls-02092025-bioacoustics/insects/data/ECOSoundSet"

CONDA_ENV="tl_bioac_mamba"

# Script options (uncomment/modify as needed)
# VERIFY_ONLY="--verify-only"      # Only verify checksums, don't download
# NO_EXTRACT="--no-extract"        # Download but don't extract zips
# CUSTOM_OUT="--out /path/to/dir"  # Override output directory

# ---------------------- ENV ------------------------- #
module load USS/2022
module load gcc/9.4.0-pe5.34
module load lsfm-init-miniconda/1.0.0
module load slurm

set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SLURM_JOB_TMP:-/tmp}"
mkdir -p "/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/logs"

# ---------------------- RUN ------------------------- #
echo "============================================================"
echo " Job:        ${SLURM_JOB_NAME}  (JobID: ${SLURM_JOB_ID})"
echo " Host:       $(hostname)"
echo " Started:    $(date -Is)"
echo " Output dir: ${OUTPUT_DIR}"
echo " Conda env:  ${CONDA_ENV}"
echo " Python:     $(which python3 || true)"
python3 --version || true
echo "============================================================"

cd "${REPO_DIR}"

# Run the download script with default settings
srun python3 "${PY_SCRIPT}" \
  --out "${OUTPUT_DIR}"

# Alternative: run with custom options (uncomment if needed)
# srun python3 "${PY_SCRIPT}" \
#   ${VERIFY_ONLY:-} \
#   ${NO_EXTRACT:-} \
#   ${CUSTOM_OUT:---out "${OUTPUT_DIR}"}

STATUS=$?
echo "------------------------------------------------------------"
echo " Finished: $(date -Is)  |  Exit code: ${STATUS}"
echo "============================================================"
exit "${STATUS}"