#!/bin/bash -l
# Example only: sbatch --chdir=RUN_DIR/logs THIS_FILE CFG_LIST [ENSEMBLE] [RUN_LABEL]
#SBATCH --job-name=gff-disc-1dv
#SBATCH --account=renormparton
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=slurm-disconnected_3pt-%j.out

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_JOB_NODELIST:-} ]]; then
    echo 'Submit this example with sbatch; do not run it on a login node.' >&2
    exit 2
fi

# Initialize the CPU software stack here, not in the submitting login session.
module --force purge
module load Stages/2026 GCCcore/14.3.0
module load h5py/3.14.0 sympy/1.14.0 SciPy-bundle/2025.07
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

CODE=/p/project1/ngff/li47/code/glwc3/project2/09_gravitationalFFs/codex
OUTPUT_ROOT=/p/project1/ngff/li47/code/scratch/run/09_gravitationalFFs
CFG_LIST=${1:?Pass a readable configuration-list file as the first argument}
ENSEMBLE=${2:-cB211.072.64}
RUN_LABEL=${3:-K16_stout10}
WORKERS=10
MEMORY_MIB=1024

case "$ENSEMBLE" in
    cB211.072.64|cC211.060.80|cD211.054.96|cE211.044.112) ;;
    *) echo "Unsupported ensemble: $ENSEMBLE" >&2; exit 2 ;;
esac
if [[ ! $RUN_LABEL =~ ^[A-Za-z0-9_.-]+$ || $RUN_LABEL == . || $RUN_LABEL == .. ]]; then
    echo 'RUN_LABEL must be a plain name.' >&2
    exit 2
fi
if [[ ! -r $CFG_LIST || ${SLURM_CPUS_PER_TASK:-0} -lt $WORKERS ]]; then
    echo 'A readable CFG_LIST and at least ten allocated CPUs are required.' >&2
    exit 2
fi

# Validate the whole list before starting writers, including duplicate names.
mapfile -t LINES < "$CFG_LIST"
CFGS=()
declare -A SEEN=()
for cfg in "${LINES[@]}"; do
    cfg=${cfg%$'\r'}
    [[ -z $cfg || $cfg == \#* ]] && continue
    if [[ ! $cfg =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "Invalid configuration name: $cfg" >&2
        exit 2
    fi
    if [[ -n ${SEEN[$cfg]+present} ]]; then
        echo "Repeated configuration name: $cfg" >&2
        exit 2
    fi
    SEEN[$cfg]=1
    CFGS+=("$cfg")
done
if [[ ${#CFGS[@]} -eq 0 ]]; then
    echo 'CFG_LIST selects no configurations.' >&2
    exit 2
fi

LOG_DIR=$OUTPUT_ROOT/$ENSEMBLE/$RUN_LABEL/logs
mkdir -p "$LOG_DIR"
export CODE OUTPUT_ROOT ENSEMBLE RUN_LABEL MEMORY_MIB LOG_DIR
printf 'node=%s job=%s configurations=%s workers=%s\n' \
    "$(hostname)" "$SLURM_JOB_ID" "${#CFGS[@]}" "$WORKERS"

# sbatch runs this script on an allocated compute node; these children stay there.
# Each process owns one CFG.h5 and its persistent CFG.lock. Never remove lock files.
printf '%s\0' "${CFGS[@]}" | xargs -0 -r -n 1 -P "$WORKERS" bash -c '
    cfg=$1
    exec python3 -u "$CODE/disconnected_3pt.py" \
        --cfg "$cfg" --ensemble "$ENSEMBLE" \
        --output-root "$OUTPUT_ROOT" --run-label "$RUN_LABEL" \
        --sink-n2-max 16 --transfer-n2-max 16 \
        --currents j+,js,jc --stouts 5,7,10,13,15,20,25,30,35,40 \
        --tfs 2:22 --threads 1 --kernel direct --memory-mib "$MEMORY_MIB" \
        >"$LOG_DIR/$cfg.job$SLURM_JOB_ID.log" 2>&1
' _
