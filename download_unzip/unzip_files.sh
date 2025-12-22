#!/bin/bash
#
#---------------------- SLURM DIRECTIVES ----------------------#
#SBATCH --job-name=unzip_files
#SBATCH --mail-type=end
#SBATCH --mail-user=peeb@zhaw.ch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=earth-3
#SBATCH --constraint=rhel8
#SBATCH --mem=16G
#SBATCH --output=unzip_files.%x.%j.out
#SBATCH --error=unzip_files.%x.%j.err
#SBATCH --chdir=/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/logs
#--------------------------------------------------------------#

set -euo pipefail

# ---------------------- CONFIG ---------------------- #

# Paths
SOURCE_DIR="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/zip-archives/Kanton_Aargau_zips/Audio 2025 von Bruno.zip"
TARGET_DIR="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/data/Kanton_Aargau"
PY_SCRIPT="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/download_unzip/unzip_files.py"
REPO_DIR="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess"

CONDA_ENV="tl_bioac_mamba"

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
mkdir -p "${TARGET_DIR}"
mkdir -p "${SLURM_JOB_TMP:-/tmp}"
mkdir -p "/cfs/earth/scratch/peeb/projects/chirpscan_preprocess/logs"

# ---------------------- RUN ------------------------- #
echo "============================================================"
echo " Job:        ${SLURM_JOB_NAME}  (JobID: ${SLURM_JOB_ID})"
echo " Host:       $(hostname)"
echo " Started:    $(date -Is)"
echo " Source:     ${SOURCE_DIR}"
echo " Target dir: ${TARGET_DIR}"
echo " Conda env:  ${CONDA_ENV}"
echo " Python:     $(which python3 || true)"
python3 --version || true
echo "============================================================"

cd "${REPO_DIR}"

srun python3 "${PY_SCRIPT}" \
  --source "${SOURCE_DIR}" \
  --target "${TARGET_DIR}"

STATUS=$?
echo "------------------------------------------------------------"
echo " Finished: $(date -Is)  |  Exit code: ${STATUS}"
echo "============================================================"
exit "${STATUS}"