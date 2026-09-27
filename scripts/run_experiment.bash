#!/bin/bash
# =================================================================================================
# Run one paper experiment (all model seeds of one model on one dataset), exactly as released.
#
# Usage (inside the docker container, any working directory, e.g. /opt/MTM-Pro/scripts/run_experiment.bash):
#   $ bash scripts/run_experiment.bash <dataset> <model> [extra hydra overrides...]
#
#   <dataset>: lorenz | husky | neurobem | pi_tcn
#   <model>  : ar-mlp-det | ar-mlp-prob | gru | lstm | e2e-tcn | m3 | tcn | tcn-bf16 | mtm-pro-ms+cp | mtm-pro-ms
#
# Example:
#   $ bash scripts/run_experiment.bash lorenz mtm-pro-ms+cp
#   $ bash scripts/run_experiment.bash husky tcn trial_nb=1        # a single seed instead of the 4 released ones
#
# See src/launcher/configs/multirun_overrides/ICLR2027_paper_release_experiments_list.md for the
# mapping between the hydra configuration files and the model labels used in the paper.
# =================================================================================================
set -euo pipefail

DATASET="${1:?'usage: run_experiment.bash <dataset> <model> [hydra overrides...]'}"
MODEL="${2:?'usage: run_experiment.bash <dataset> <model> [hydra overrides...]'}"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CFG_ROOT="launcher/configs/multirun_overrides"

case "${DATASET}" in
  lorenz)
    LAUNCHER="launcher/math_env_main.py"
    CFG_DIR="${CFG_ROOT}/math_env/RLRP-757-E4b-low-global-noise"
    PREFIX="math_env_main_multirun_RLRP-757-"
    declare -A MODELS=(
      [ar-mlp-det]="ar-mlp-det" [ar-mlp-prob]="ar-mlp-prob" [gru]="gru" [lstm]="lstm"
      [e2e-tcn]="e2e-tcn-F1000" [m3]="m3-F1000" [tcn]="tcn" [tcn-bf16]="tcn-bf16"
      [mtm-pro-ms+cp]="DMtmPro-MS+CP" [mtm-pro-ms]="DMtmPro-MS"
    )
    ;;
  husky)
    LAUNCHER="launcher/robotic_3d_env_main_multirun.py"
    CFG_DIR="${CFG_ROOT}/robotic_3d_env/RLRP-757-Husky-F2-online"
    PREFIX="multirun-"
    declare -A MODELS=(
      [ar-mlp-det]="ar-mlp-det" [ar-mlp-prob]="ar-mlp-prob" [gru]="gru" [lstm]="lstm"
      [e2e-tcn]="e2e-tcn-F50" [m3]="m3-F50" [tcn]="tcn" [tcn-bf16]="tcn-bf16"
      [mtm-pro-ms+cp]="DMtm-Pro-MS+CP" [mtm-pro-ms]="DMtm-Pro-MS"
    )
    ;;
  neurobem)
    LAUNCHER="launcher/robotic_3d_env_main_multirun.py"
    CFG_DIR="${CFG_ROOT}/robotic_3d_env/RLRP-757-Neurobem-F1"
    PREFIX="multirun-"
    declare -A MODELS=(
      [ar-mlp-det]="ar-mlp-det" [ar-mlp-prob]="ar-mlp-prob" [gru]="gru" [lstm]="lstm"
      [e2e-tcn]="e2e-tcn-F500" [m3]="m3-F500" [tcn]="tcn" [tcn-bf16]="tcn-bf16"
      [mtm-pro-ms+cp]="DMtm-Pro-MS+CP" [mtm-pro-ms]="DMtm-Pro-MS"
    )
    ;;
  pi_tcn)
    LAUNCHER="launcher/robotic_3d_env_main_multirun.py"
    CFG_DIR="${CFG_ROOT}/robotic_3d_env/RLRP-757-PI-TCN-B1"
    PREFIX="multirun-"
    declare -A MODELS=(
      [ar-mlp-det]="ar-mlp-det" [ar-mlp-prob]="ar-mlp-prob" [gru]="gru" [lstm]="lstm"
      [e2e-tcn]="e2e-tcn-F500" [m3]="m3-F500" [tcn]="tcn" [tcn-bf16]="tcn-bf16"
      [mtm-pro-ms+cp]="DMtm-Pro-MS+CP+no-enf-continuity" [mtm-pro-ms]="DMtm-Pro-MS+no-enf-continuity"
    )
    ;;
  *)
    echo "Unknown dataset '${DATASET}' (expected: lorenz | husky | neurobem | pi_tcn)" >&2; exit 1 ;;
esac

if [[ -z "${MODELS[${MODEL}]+x}" ]]; then
  echo "Unknown model '${MODEL}' (expected: ${!MODELS[*]})" >&2; exit 1
fi
CONFIG_NAME="${PREFIX}${MODELS[${MODEL}]}"

cd "${REPO_ROOT}/src"
echo "[MTM-Pro] python ${LAUNCHER} --multirun --config-dir=${CFG_DIR} --config-name=${CONFIG_NAME} $*"
exec python "${LAUNCHER}" --multirun --config-dir="${CFG_DIR}" --config-name="${CONFIG_NAME}" "$@"
