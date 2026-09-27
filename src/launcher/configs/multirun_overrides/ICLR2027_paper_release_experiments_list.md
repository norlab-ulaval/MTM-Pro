## ICLR 2027 conference paper, hydra configuration files per experiment and per model
Note: Our paper was initially aiming for publishing in ICRA so directory name might still be named after that. We are publishing in ICLR2027 in the end.   

Research paper associated codebase repository local path: `/path/to/MTM-Pro` 

### Lorenz experiments
Experiment directory path: `src/launcher/configs/multirun_overrides/math_env/RLRP-757-E4b-low-global-noise`

Multirun experiment hydra configuration files per model:
- `math_env_main_multirun_RLRP-757-ar-mlp-det.yaml`
- `math_env_main_multirun_RLRP-757-ar-mlp-prob.yaml`
- `math_env_main_multirun_RLRP-757-gru.yaml`
- `math_env_main_multirun_RLRP-757-lstm.yaml`
- `math_env_main_multirun_RLRP-757-e2e-tcn-F1000.yaml`
- `math_env_main_multirun_RLRP-757-m3-F1000.yaml`
- `math_env_main_multirun_RLRP-757-tcn.yaml`
- `math_env_main_multirun_RLRP-757-tcn-bf16.yaml`
- `math_env_main_multirun_RLRP-757-DMtmPro-MS+CP.yaml`
- `math_env_main_multirun_RLRP-757-DMtmPro-MS.yaml`

### Husky (UGV) experiments
Experiment directory path: `src/launcher/configs/multirun_overrides/robotic_3d_env/RLRP-757-Husky-F2-online`

Multirun experiment hydra configuration files per model:
- `multirun-ar-mlp-det.yaml`
- `multirun-ar-mlp-prob.yaml`
- `multirun-gru.yaml`
- `multirun-lstm.yaml`
- `multirun-e2e-tcn-F50.yaml`
- `multirun-m3-F50.yaml`
- `multirun-tcn.yaml`
- `multirun-tcn-bf16.yaml`
- `multirun-DMtm-Pro-MS+CP.yaml`
- `multirun-DMtm-Pro-MS.yaml`

### Neurobem (UAV) experiments
Experiment directory path: `src/launcher/configs/multirun_overrides/robotic_3d_env/RLRP-757-Neurobem-F1`

Multirun experiment hydra configuration files per model:
- `multirun-ar-mlp-det.yaml`
- `multirun-ar-mlp-prob.yaml`
- `multirun-gru.yaml`
- `multirun-lstm.yaml`
- `multirun-e2e-tcn-F500.yaml`
- `multirun-m3-F500.yaml`
- `multirun-tcn.yaml`
- `multirun-tcn-bf16.yaml`
- `multirun-DMtm-Pro-MS+CP.yaml`
- `multirun-DMtm-Pro-MS.yaml`


### PI-TCN (UAV) experiments
Experiment directory path: `src/launcher/configs/multirun_overrides/robotic_3d_env/RLRP-757-PI-TCN-B1`

Multirun experiment hydra configuration files per model:
- `multirun-ar-mlp-det.yaml`
- `multirun-ar-mlp-prob.yaml`
- `multirun-gru.yaml`
- `multirun-lstm.yaml`
- `multirun-e2e-tcn-F500.yaml`
- `multirun-m3-F500.yaml`
- `multirun-tcn.yaml`
- `multirun-tcn-bf16.yaml`
- `multirun-DMtm-Pro-MS+CP+no-enf-continuity.yaml` <-- equivalent to `multirun-DMtm-Pro-MS+CP.yaml` in the other experiment
- `multirun-DMtm-Pro-MS+no-enf-continuity.yaml` <-- equivalent to `multirun-DMtm-Pro-MS.yaml` in the other experiment

## Instructions on labeling
For every model of each experiment, use the following label mapping
- `multirun-ar-mlp-det.yaml` -> `(AR) Det MLP`
- `multirun-ar-mlp-prob.yaml` -> `(AR) Prob MLP`
- `multirun-gru.yaml` -> `(AR) GRU`
- `multirun-lstm.yaml` -> `(AR) LSTM`
- `multirun-e2e-tcn-F500.yaml` -> `(MS) E2E-TCN`
- `multirun-m3-F500.yaml` -> `(MS) M3`
- `multirun-tcn.yaml` -> `(AR) TCN `
- `multirun-tcn-bf16.yaml` -> `(AR) TCN-bf16`
- `multirun-DMtm-Pro-MS+CP.yaml` -> `MTM-Pro-MS+CP (ours)`
- `multirun-DMtm-Pro-MS.yaml` -> `MTM-Pro-MS (ours)`

