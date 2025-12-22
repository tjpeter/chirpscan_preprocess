#!/bin/bash
#
#SBATCH --job-name=panel_ecosoundset
# (mail settings are configured dynamically after loading .env)
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=earth-3
#SBATCH --constraint=rhel8

# -----------------------------------------------------------------------------
# Set project root
# -----------------------------------------------------------------------------
PROJECT_ROOT="/cfs/earth/scratch/peeb/projects/chirpscan_preprocess"
APP_FILE="${PROJECT_ROOT}/exploration/explore_ecosoundset.py"

cd "${PROJECT_ROOT}" || { echo "[error] cannot cd to ${PROJECT_ROOT}"; exit 1; }

# -----------------------------------------------------------------------------
# Load environment variables from .env file
# -----------------------------------------------------------------------------
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    echo "[info] Loading environment variables from ${ENV_FILE}"
    # Export variables from .env, ignoring comments and empty lines
    set -a
    source <(grep -v '^#' "${ENV_FILE}" | grep -v '^$' | sed 's/^/export /')
    set +a
else
    echo "[warn] .env file not found at ${ENV_FILE}. Using defaults."
fi

# -----------------------------------------------------------------------------
# Derive config from environment (with sensible defaults)
# -----------------------------------------------------------------------------
SSH_USER="${SSH_USER:-peeb}"
CONDA_ENV="${CONDA_ENV:-preprocess_chirpscan}"
JOB_EMAIL="${JOB_EMAIL:-peeb@zhaw.ch}"

# -----------------------------------------------------------------------------
# Define log directory
# -----------------------------------------------------------------------------
JOB_NAME="${SLURM_JOB_NAME:-job}"
JOB_ID="${SLURM_JOB_ID:-$$}"
LOG_DIR="${PROJECT_ROOT}/logs/slurm_${JOB_NAME}_${JOB_ID}"
mkdir -p "${LOG_DIR}"

# Mirror stdout/stderr to log files (use tee to see output in SLURM logs too)
exec 1> >(tee -a "${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out")
exec 2> >(tee -a "${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err" >&2)

echo "[info] ============================================================"
echo "[info] Job:        ${JOB_NAME} (JobID: ${JOB_ID})"
echo "[info] Node:       $(hostname)"
echo "[info] Started:    $(date -Is)"
echo "[info] Project:    ${PROJECT_ROOT}"
echo "[info] App:        ${APP_FILE}"
echo "[info] Logging:    ${LOG_DIR}"
echo "[info] SSH user:   ${SSH_USER}"
echo "[info] Conda env:  ${CONDA_ENV}"
if [[ -n "${JOB_EMAIL}" ]]; then
  echo "[info] Job e-mail: ${JOB_EMAIL}"
else
  echo "[info] Job e-mail: <none configured>"
fi
echo "[info] ============================================================"

# -----------------------------------------------------------------------------
# Load modules
# -----------------------------------------------------------------------------
module load USS/2022
module load gcc/9.4.0-pe5.34 
module load lsfm-init-miniconda/1.0.0
module load slurm

# -----------------------------------------------------------------------------
# Configure SLURM e-mail notifications only if an e-mail is provided
# -----------------------------------------------------------------------------
if [[ -n "${JOB_EMAIL}" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[info] Configuring SLURM mail user to ${JOB_EMAIL}"
    # Set mail user and types dynamically
    scontrol update JobId="${SLURM_JOB_ID}" MailUser="${JOB_EMAIL}" MailType=END,FAIL || {
        echo "[warn] Failed to configure SLURM mail settings via scontrol"
    }
else
    echo "[info] No JOB_EMAIL set; SLURM notifications will not be configured."
fi

# -----------------------------------------------------------------------------
# Activate conda env (from .env or default)
# -----------------------------------------------------------------------------
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" != "${CONDA_ENV}" ]]; then
    echo "[error] Failed to activate ${CONDA_ENV} environment"
    exit 1
fi
echo "[info] Active conda environment: $CONDA_DEFAULT_ENV"

# -----------------------------------------------------------------------------
# Set number of threads
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "[info] Thread configuration: OMP_NUM_THREADS=${OMP_NUM_THREADS}, MKL_NUM_THREADS=${MKL_NUM_THREADS}"

# -----------------------------------------------------------------------------
# Panel server configuration
# -----------------------------------------------------------------------------
PORT="${PORT:-5006}"
HOST="localhost"

# Identify the compute node
NODE="$(hostname)"
TUNNEL_FILE="${LOG_DIR}/ssh_tunnel_instructions.txt"
LOGIN_NODE="${LOGIN_NODE:-<login-node>}"

# -----------------------------------------------------------------------------
# Verify app file exists
# -----------------------------------------------------------------------------
if [[ ! -f "${APP_FILE}" ]]; then
    echo "[error] App file not found: ${APP_FILE}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Run Panel app with automatic port finding
# -----------------------------------------------------------------------------
echo "[info] Starting Panel app..."
echo "[info] Initial port attempt: ${PORT}"
echo "[info] ============================================================"
echo ""

set -euo pipefail

# Function to check if port is in use
check_port() {
    local port=$1
    lsof -Pi :${port} -sTCP:LISTEN -t >/dev/null 2>&1
    return $?
}

# Try to find an available port
MAX_PORT_ATTEMPTS=10
ATTEMPT=0
while check_port ${PORT} && [ ${ATTEMPT} -lt ${MAX_PORT_ATTEMPTS} ]; do
    echo "[warn] Port ${PORT} is already in use, trying next port..."
    PORT=$((PORT + 1))
    ATTEMPT=$((ATTEMPT + 1))
done

if [ ${ATTEMPT} -ge ${MAX_PORT_ATTEMPTS} ]; then
    echo "[error] Could not find available port after ${MAX_PORT_ATTEMPTS} attempts"
    exit 1
fi

echo "[info] Using port: ${PORT}"

# Generate SSH tunnel instructions with actual port
cat > "${TUNNEL_FILE}" <<EOF
========================================
Panel App Connection Instructions
========================================

The Panel app is running on: ${NODE}:${PORT}

To access it from your local machine:

METHOD 1: Direct SSH tunnel to compute node (if allowed)
  ssh -J ${SSH_USER}@${LOGIN_NODE} -L ${PORT}:localhost:${PORT} ${SSH_USER}@${NODE}

METHOD 2: Port forward via login node (recommended)
  ssh -L ${PORT}:${NODE}:${PORT} ${SSH_USER}@${LOGIN_NODE}

METHOD 3: Two-step tunnel (most compatible)
  # Step 1: On your local machine
  ssh -L ${PORT}:localhost:${PORT} ${SSH_USER}@${LOGIN_NODE}
  
  # Step 2: On login node (in the SSH session from step 1)
  ssh -L ${PORT}:localhost:${PORT} ${SSH_USER}@${NODE}

Then open in your browser: http://localhost:${PORT}

Job ID: ${JOB_ID}
Log directory: ${LOG_DIR}

----------------------------------------
Stopping the app and freeing ports
----------------------------------------

To stop the Panel app and free the compute node port (${NODE}:${PORT}):

  scancel ${JOB_ID}

To free the local port on your own machine (localhost:${PORT}):

  - Go to the terminal where the SSH tunnel is running
  - Press:  Ctrl+C
  - Close the terminal if desired

Note:
  Both the SLURM job AND the SSH tunnel must be terminated
  for the port to become fully available for future jobs.

========================================
EOF

echo ""
echo "[info] SSH tunnel instructions:"
cat "${TUNNEL_FILE}"
echo ""
echo "[info] Server will be available at: http://${HOST}:${PORT}"
echo "[info] ============================================================"
echo ""

# Run with srun to keep it attached to the SLURM allocation
srun --ntasks=1 \
  panel serve "${APP_FILE}" \
    --address "${HOST}" \
    --port "${PORT}" \
    --allow-websocket-origin="*" \
    --num-procs 1 \
    2>&1 | tee -a "${LOG_DIR}/panel_output.log"

# Capture exit status
STATUS=$?

echo ""
echo "[info] ============================================================"
echo "[info] Panel app stopped"
echo "[info] Exit code: ${STATUS}"
echo "[info] Finished: $(date -Is)"
echo "[info] ============================================================"

# -----------------------------------------------------------------------------
# Cleanup SLURM step directories
# -----------------------------------------------------------------------------
relocate_step_dir() {
  local d=""
  echo "[info] Relocating SLURM step directory contents to ${LOG_DIR}"
  for d in "Slurm-${SLURM_JOB_ID}" "slurm-${SLURM_JOB_ID}"; do
    if [[ -d "$d" ]]; then
      echo "[info] Found directory: $d"
      rsync -a --remove-source-files "$d"/ "${LOG_DIR}/" 2>/dev/null || true
      rm -rf "$d" 2>/dev/null || true
      echo "[info] Removed directory: $d"
    fi
  done
}

relocate_step_dir
trap 'relocate_step_dir' EXIT

exit ${STATUS}