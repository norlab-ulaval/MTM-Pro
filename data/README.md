# Data

```
data/
├── external_data/                       robotic datasets used by the released experiments (csv trajectories)
│   ├── Quadcopter_README.md             provenance / pre-processing notes of the two UAV datasets
│   ├── long_horizon/data/neurobem/      UAV NeuroBEM      -> simulator config `neurobem_*`
│   ├── long_horizon/data/pi_tcn/        UAV PI-TCN        -> simulator config `pi_tcn_*`
│   └── ugv/husky_dataset_src/pre_processed_post_rlrp-780/sets/online/
│                                        UGV Husky, gravel and grass, online split -> `husky_*_online`
└── data_consolidated/                   numeric series behind every paper figure (see its README.md)
```

Only the trajectories referenced by the simulator configurations of the released experiments
(`src/launcher/configs/simulator/`) are shipped. The Lorenz attractor data is generated on the fly.

On first use, the robotic pipelines ingest the csv files and cache the result next to them in
`<data_path>/ms_hdf5/` (versioned, spec guarded: regenerated automatically when the configuration changes)
and `<data_path>/ms_replaybuffer/` (VCS ignored).
