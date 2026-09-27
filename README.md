# FROM CONTRACTION TO CHAOS: COMPOUNDING ERROR RESISTANT WORLD MODELS FOR ROBOTS

> This anonymized repository accompanies our paper under review at ICLR 2027 on OpenReview:
> [https://openreview.net/forum?id=IKsP4J04TN](https://openreview.net/forum?id=IKsP4J04TN&invitationId=ICLR.cc%2F2027%2FConference%2FSubmission37661%2F-%2FFull_Submission&referrer=%5BTasks%5D%28%2Ftasks%29)

Standalone, anonymized release of the source code, configurations and data required to reproduce the
experiments of the paper (ICLR 2027 submission). It is a consolidated extract of a larger research
codebase: only the code paths, hydra configurations and dataset files exercised by the released
experiments are included, with their logic and naming kept identical.

> Note: the paper initially targeted ICRA, so several directory / experiment names still carry the
> `ICRA2026` / `RLRP-757` tokens. Likewise a few shared helper packages keep their historical names from
> the parent codebase (e.g. `tools/f110_gym_env_tools/plot_tensorboard.py`, the tensorboard writer used
> by every pipeline); no simulator other than the Lorenz attractor and the three csv datasets is included.

## Contents

```
MTM-Pro/
├── docker/                    Dockerfile, docker-compose.yaml, requirements.txt (the ONLY supported runtime)
├── scripts/                   run_experiment.bash (paper runs), smoke_test.bash (fast end-to-end check)
├── src/
│   ├── launcher/              hydra entry points + configs (see launcher/configs/multirun_overrides/)
│   ├── algorithm/             experience-replay learning loop, motion-model learning/deployment
│   ├── pipeline/              Lorenz (math_env) and robotic (robotic_3d_env) experiment pipelines
│   ├── tools/                 models (tools/multistep_tools/models: MTM-Pro + baselines), data, plotting
│   ├── simulator/, custom_types/
├── data/
│   ├── external_data/         the three robotic datasets (csv trajectories used by the paper configs)
│   │   ├── long_horizon/data/neurobem   UAV NeuroBEM
│   │   ├── long_horizon/data/pi_tcn     UAV PI-TCN
│   │   └── ugv/husky_dataset_src/...    UGV Husky (gravel and grass, online split)
│   └── data_consolidated/     the numeric series behind every paper figure (csv + provenance meta.txt)
├── utilities/                 git submodules: mbrl-lib (fork), math-gymnasium, trajectory-container-tools
└── artifact/                  experiment outputs (created at runtime, VCS ignored)
```

The Lorenz attractor data is generated on the fly (`utilities/math-gymnasium`).

## Quick start

Requirements: `git`, `docker` (>= 24, with the compose plugin). NVIDIA GPU runs additionally need the
[NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone <this repository> MTM-Pro          # the directory MUST be named MTM-Pro
cd MTM-Pro
git submodule update --init --recursive

# 1. Build the image (~10-20 min, downloads PyTorch)
docker compose -f docker/docker-compose.yaml build

# 2. Fast end-to-end check (CPU, ~3 min): one MTM-Pro run on Lorenz + one on the UAV dataset,
#    with tiny models and 2 epochs
docker compose -f docker/docker-compose.yaml run --rm mtm-pro bash /opt/MTM-Pro/scripts/smoke_test.bash quick

# 3. A paper experiment (all 4 seeds of one model on one dataset), on GPU
docker compose -f docker/docker-compose.yaml run --rm mtm-pro-gpu bash /opt/MTM-Pro/scripts/run_experiment.bash lorenz mtm-pro-ms+cp
```

Use the `mtm-pro` service instead of `mtm-pro-gpu` to run on CPU. Both services bind mount the repository
at `/opt/MTM-Pro`, so outputs land in `./artifact/` on the host. An interactive shell:

```bash
docker compose -f docker/docker-compose.yaml run --rm mtm-pro-gpu   # working dir: /opt/MTM-Pro/src
```

## Running the paper experiments

Every experiment is a hydra multirun (`trial_nb=range(1,5)`, i.e. 4 model seeds) defined by ONE
configuration file. `scripts/run_experiment.bash <dataset> <model> [hydra overrides...]` maps the paper
labels to those files:

| `<dataset>` | Configuration directory (`src/launcher/configs/multirun_overrides/`) | Launcher |
|---|---|---|
| `lorenz`   | `math_env/RLRP-757-E4b-low-global-noise` | `launcher/math_env_main.py` |
| `husky`    | `robotic_3d_env/RLRP-757-Husky-F2-online` | `launcher/robotic_3d_env_main_multirun.py` |
| `neurobem` | `robotic_3d_env/RLRP-757-Neurobem-F1` | `launcher/robotic_3d_env_main_multirun.py` |
| `pi_tcn`   | `robotic_3d_env/RLRP-757-PI-TCN-B1` | `launcher/robotic_3d_env_main_multirun.py` |

| `<model>` | Paper label | Configuration file (robotic datasets / Lorenz) |
|---|---|---|
| `ar-mlp-det`    | (AR) Det MLP        | `multirun-ar-mlp-det` / `math_env_main_multirun_RLRP-757-ar-mlp-det` |
| `ar-mlp-prob`   | (AR) Prob MLP       | `multirun-ar-mlp-prob` / `..._RLRP-757-ar-mlp-prob` |
| `gru`           | (AR) GRU            | `multirun-gru` / `..._RLRP-757-gru` |
| `lstm`          | (AR) LSTM           | `multirun-lstm` / `..._RLRP-757-lstm` |
| `tcn`           | (AR) TCN            | `multirun-tcn` / `..._RLRP-757-tcn` |
| `tcn-bf16`      | (AR) TCN-bf16       | `multirun-tcn-bf16` / `..._RLRP-757-tcn-bf16` |
| `e2e-tcn`       | (MS) E2E-TCN        | `multirun-e2e-tcn-F<horizon>` / `..._RLRP-757-e2e-tcn-F1000` |
| `m3`            | (MS) M3             | `multirun-m3-F<horizon>` / `..._RLRP-757-m3-F1000` |
| `mtm-pro-ms+cp` | MTM-Pro-MS+CP (ours)| `multirun-DMtm-Pro-MS+CP` (`+no-enf-continuity` on PI-TCN) / `..._RLRP-757-DMtmPro-MS+CP` |
| `mtm-pro-ms`    | MTM-Pro-MS (ours)   | `multirun-DMtm-Pro-MS` (`+no-enf-continuity` on PI-TCN) / `..._RLRP-757-DMtmPro-MS` |

The full list is in `src/launcher/configs/multirun_overrides/ICLR2027_paper_release_experiments_list.md`.
The equivalent raw command of e.g. `scripts/run_experiment.bash husky tcn` is (from `src/`):

```bash
python launcher/robotic_3d_env_main_multirun.py --multirun \
    --config-dir=launcher/configs/multirun_overrides/robotic_3d_env/RLRP-757-Husky-F2-online \
    --config-name=multirun-tcn
```

Any hydra override can be appended, e.g. `trial_nb=1` for a single seed, or `hydra.launcher.n_jobs=2`
(also `HPO_N_JOBS=2` in the environment) to train two seeds concurrently on one GPU.

Outputs (tensorboard logs, checkpoints, test-time rollout records and plots) are written under
`artifact/ICRA2026/<experiment>/<date>/<time>/`. The robotic pipelines cache the ingested dataset next
to the csv files (`<data_path>/ms_hdf5/` and `<data_path>/ms_replaybuffer/`) on first use.

### Figures (test-time rollout comparison across models)

Once the multiruns of one dataset are recorded, the paper figures are rendered by the plot pipeline,
which reads the recorded rollouts from `artifact/` (one configuration per dataset in
`src/launcher/configs/multirun_testtime_rollout_plot/`):

```bash
python launcher/multirun_testtime_rollout_plot.py multirun_testtime_rollout_plot@_global_=icra2026_Lorenz_RLRP-757-E4B-low-global-noise
python launcher/multirun_testtime_rollout_plot.py multirun_testtime_rollout_plot@_global_=icra2026_ugv_RLRP-757-F2c-obsD6-online
python launcher/multirun_testtime_rollout_plot.py multirun_testtime_rollout_plot@_global_=icra2026_neurobem_RLRP-757-F1
python launcher/multirun_testtime_rollout_plot.py multirun_testtime_rollout_plot@_global_=icra2026_pi_tcn_RLRP-757-B1a
```

Those configurations point to the experiment directories of the original runs
(`multirun_experiments_path`): update them to the `artifact/...` paths of your runs. With
`consolidate_data.enable=true` the pipeline also dumps the exact numeric series rendered in each figure
to `data/data_consolidated/<experiment>/` -- the version shipped here (`data/data_consolidated/`, see its
`README.md`) is the one produced from the paper runs, so every figure can be rebuilt from csv without
re-running the pipelines.

## Smoke test / development configurations

`scripts/smoke_test.bash [quick]` runs the real experiment configurations of every model family on every
dataset with the `dev` hydra overlay (`src/launcher/configs/dev/`): tiny models, 2 epochs, one seed and,
for the robotic datasets, a handful of trajectories. It validates the environment, the data and every
pipeline stage in minutes on CPU but is NOT a paper result. Any experiment can be run this way by
appending, e.g., `+dev@_global_=math_env_debug_mtm_pro ms_model.ensemble_size=1 trial_nb=1`.

## Datasets and credits

- **UAV PI-TCN**: the quadrotor dataset of *Physics-Inspired Temporal Learning of Quadrotor Dynamics for
  Accurate Model Predictive Trajectory Tracking* ([arplaboratory/pi-tcn](https://github.com/arplaboratory/pi-tcn)),
  pre-processed to csv. See `data/external_data/Quadcopter_README.md`.
- **UAV NeuroBEM**: the [NeuroBEM](https://rpg.ifi.uzh.ch/NeuroBEM.html) quadrotor dataset (RPG, University of
  Zurich), pre-processed to csv. See `data/external_data/Quadcopter_README.md`.
- **UGV Husky** (gravel and grass, online split): our own Clearpath Husky A200 recordings, pre-processed to csv.
- **Baselines**: the baseline network implementations in `src/tools/baseline_models/` derive from the
  [long-horizon-dynamics](https://github.com/arplaboratory/long-horizon-dynamics) codebase of *Learning Long-Horizon
  Predictions for Quadrotor Dynamics* (IROS 2024), see `src/tools/baseline_models/README.md`. Every model of the
  paper (MTM-Pro and the AR / MS baselines) is wrapped in `src/tools/multistep_tools/models/`.
- `utilities/mbrl-lib` is a fork of Facebook Research's [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib)
  (MIT license).

## License

See the `LICENSE` file of each submodule for the third-party code. The MTM-Pro source code is released for
review purposes; a license will be attached to the de-anonymized release.
