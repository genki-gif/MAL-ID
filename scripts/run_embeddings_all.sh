#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=12:00:00
#$ -N run_embedding_all
#$ -o job_logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e job_logs/$JOB_NAME.$JOB_ID.$TASK_ID.err
#$ -m abe
#$ -M masuda@li.comp.isct.ac.jp
#$ -t 1-8

# --- 環境設定 ---
. /etc/profile.d/modules.sh
source ~/.bashrc
mamba activate cuda-env-py39

set -euo pipefail

# Configuration (can be overridden via environment)
MALID_DATASET_VERSION="${MALID_DATASET_VERSION:-20231109}"
MALID_CV_SPLIT="${MALID_CV_SPLIT:-in_house_peak_disease_timepoints}"
LOG_DIR="data/logs"

mkdir -p "${LOG_DIR}"

echo "Using MALID_DATASET_VERSION=${MALID_DATASET_VERSION}"
echo "Using MALID_CV_SPLIT=${MALID_CV_SPLIT}"

# Loci and CV folds to run
LOCI=("BCR" "TCR")
FOLDS=("0" "1" "2" "-1")

run_one() {
  local locus="$1"
  local fold="$2"

  echo "=== Running embedding for locus=${locus}, fold_id=${fold} ==="
  MALID_DATASET_VERSION="${MALID_DATASET_VERSION}" MALID_CV_SPLIT="${MALID_CV_SPLIT}" \
    python scripts/run_embedding.py \
      --locus "${locus}" \
      --fold_id "${fold}" \
    2>&1 | tee "${LOG_DIR}/run_embedding.fold${fold}.${locus}.log"

  echo "=== Scaling anndatas for locus=${locus}, fold_id=${fold} ==="
  MALID_DATASET_VERSION="${MALID_DATASET_VERSION}" MALID_CV_SPLIT="${MALID_CV_SPLIT}" \
    python scripts/scale_embedding_anndatas.py \
      --locus "${locus}" \
      --fold_id "${fold}" \
    2>&1 | tee "${LOG_DIR}/scale_anndatas.fold${fold}.${locus}.log"
}

# SGE array: map SGE_TASK_ID -> (locus, fold)
TASKS=()
for _locus in "${LOCI[@]}"; do
  for _fold in "${FOLDS[@]}"; do
    TASKS+=("${_locus},${_fold}")
  done
done

if [ -n "${SGE_TASK_ID:-}" ]; then
  idx=$((SGE_TASK_ID - 1))
  if [ "${idx}" -lt 0 ] || [ "${idx}" -ge "${#TASKS[@]}" ]; then
    echo "Invalid SGE_TASK_ID=${SGE_TASK_ID} (TASKS has ${#TASKS[@]} items)"
    exit 1
  fi
  IFS=',' read -r locus fold <<< "${TASKS[$idx]}"
  echo "SGE task ${SGE_TASK_ID}/${#TASKS[@]}: locus=${locus}, fold=${fold}"
  run_one "${locus}" "${fold}"
else
  # Fallback: iterate all combinations sequentially
  for locus in "${LOCI[@]}"; do
    for fold in "${FOLDS[@]}"; do
      run_one "${locus}" "${fold}"
    done
  done
fi

echo "All embedding + scaling runs completed."


