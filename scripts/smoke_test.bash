#!/bin/bash
# =================================================================================================
# Fast end-to-end smoke test of the standalone repository (a few minutes on CPU).
#
# Runs the real experiment configuration of each model family on each dataset with the `dev`
# hydra overlay (tiny models, 2 epochs, one seed) and, for the robotic datasets, a small subset
# of trajectories. Nothing here is a paper result: it only validates that the environment, the
# data and every pipeline stage (data ingestion, training, test-time rollouts, artifact
# recording) work.
#
# Usage (inside the docker container):
#   $ bash scripts/smoke_test.bash            # every family on every dataset (~15-25 min CPU)
#   $ bash scripts/smoke_test.bash quick      # one Lorenz run + one robotic run (~3 min CPU)
# =================================================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/src"

MODE="${1:-full}"
PASS=0; FAIL=0; FAILED=()

COMMON=(
  trial_nb=1
  hydra.sweep.subdir=smoke_test
  pipeline.tensorboard.clear_tmp_directory_run_artifact_on_teardown=true
)
# Never write a dataset cache from a smoke test run (they are spec guarded but keep the data dir clean).
ROBOTIC_NOCACHE=(
  pipeline.force_regenerating_saved_ms_replaybuffer=true
  pipeline.enable_ms_replaybuffer_saving_to_data_dir=false
  pipeline.force_regenerating_saved_ms_hdf5=true
  pipeline.enable_ms_hdf5_saving_to_data_dir=false
)

run() {
  local label="$1"; shift
  echo; echo "================================================================================"
  echo "[smoke_test] ${label}"
  echo "================================================================================"
  if "$@" > "/tmp/mtm_pro_smoke_${label//[^A-Za-z0-9_+-]/_}.log" 2>&1; then
    echo "[smoke_test] ${label} ... OK"; PASS=$((PASS + 1))
  else
    echo "[smoke_test] ${label} ... FAILED (see /tmp/mtm_pro_smoke_${label//[^A-Za-z0-9_+-]/_}.log)"
    tail -n 30 "/tmp/mtm_pro_smoke_${label//[^A-Za-z0-9_+-]/_}.log"
    FAIL=$((FAIL + 1)); FAILED+=("${label}")
  fi
}

# The released MTM-Pro configurations use a single ensemble member (the dev overlay default of 2 is
# not supported together with the compounded-prediction loss).
extra_for() { [[ "$1" == *mtm_pro* ]] && echo "ms_model.ensemble_size=1" || echo "comment=smoke_test"; }

lorenz() { # $1 model, $2 dev overlay
  run "lorenz/$1" bash "${SCRIPT_DIR}/run_experiment.bash" lorenz "$1" \
    "+dev@_global_=$2" "$(extra_for "$2")" \
    simulator@environment=lorenz_v6_5XinitC_low_global_noise "${COMMON[@]}"
}
robotic() { # $1 dataset, $2 model, $3 dev overlay, $4 small simulator config
  run "$1/$2" bash "${SCRIPT_DIR}/run_experiment.bash" "$1" "$2" \
    "+dev@_global_=$3" "$(extra_for "$3")" \
    "simulator@environment=$4" "${COMMON[@]}" "${ROBOTIC_NOCACHE[@]}"
}

if [[ "${MODE}" == "quick" ]]; then
  lorenz  mtm-pro-ms+cp math_env_debug_mtm_pro
  robotic pi_tcn mtm-pro-ms+cp robotic_3d_env_debug_mtm_pro pi_tcn_3x_trajectories
else
  # ....Lorenz (every model family).................................................................
  lorenz tcn           math_env_debug_cp_ar_tcn
  lorenz ar-mlp-det    math_env_debug
  lorenz gru           math_env_debug
  lorenz e2e-tcn       math_env_debug_ms2ms_e2e_tcn
  lorenz m3            math_env_debug_ms2ms_m3
  lorenz mtm-pro-ms+cp math_env_debug_mtm_pro
  # ....Robotic datasets (UAV PI-TCN, UAV NeuroBEM, UGV Husky)......................................
  robotic pi_tcn   tcn           robotic_3d_env_debug_cp_ar_tcn   pi_tcn_3x_trajectories
  robotic pi_tcn   m3            robotic_3d_env_debug_ms2ms_m3    pi_tcn_3x_trajectories
  robotic pi_tcn   mtm-pro-ms+cp robotic_3d_env_debug_mtm_pro     pi_tcn_3x_trajectories
  robotic neurobem e2e-tcn       robotic_3d_env_debug_ms2ms_e2e_tcn neurobem_3x_named_trajectories
  robotic neurobem mtm-pro-ms    robotic_3d_env_debug_mtm_pro     neurobem_3x_named_trajectories
  robotic husky    tcn           robotic_3d_env_debug_cp_ar_tcn   husky_gravel_and_grass_smoke_test_vel_only_online
  robotic husky    mtm-pro-ms+cp robotic_3d_env_debug_mtm_pro     husky_gravel_and_grass_smoke_test_vel_only_online
fi

echo; echo "================================================================================"
echo "[smoke_test] passed: ${PASS}   failed: ${FAIL}"
if [[ ${FAIL} -gt 0 ]]; then
  printf '[smoke_test]   - %s\n' "${FAILED[@]}"
  exit 1
fi
