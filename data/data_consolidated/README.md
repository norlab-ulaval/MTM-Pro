# `data_consolidated` — consolidated experiment data (RLRP-818)

Version-controlled home of the numeric series behind the ICRA 2026 figures.

Each CSV holds **exactly what a figure rendered**, so the standalone plot functions (RLRP-817) can rebuild
an identical figure without re-running the `MTM-Pro` (RLRC) pipeline.

> ⚠️ This directory is **produced by RLRC**, not by hand. It is written only when a run explicitly opts in
> with `consolidate_data.enable: true`.

## Layout

```
data_consolidated/
└── <EXPERIMENT-NAME>/                    # == cfg.overrides.experiment
    ├── meta.txt                          # EXPERIMENT-level provenance
    ├── drift_rate/
    │   ├── meta.txt                      # METRIC-level provenance
    │   └── <grp_name>/                   # one sub-directory per group (slugified)
    │       ├── test_time_models_comparaison_drift_rate.csv            # deployed (main) rollout
    │       └── <E>_epoch_test_time_models_comparaison_drift_rate.csv  # one per epoch checkpoint
    ├── drift_rate_samples/               # OPTIONAL, schema 2.2 -- the per-rollout population
    │   ├── meta.txt
    │   └── <grp_name>/                   # one WIDE CSV per model seed (trial)
    │       └── seed_<k>_test_time_models_comparaison_drift_rate_samples.csv
    ├── training_wall_clock_time/
    │   ├── meta.txt
    │   └── <grp_name>/
    │       └── test_time_models_comparaison_training_wall_clock_time_barplot.csv
    └── inference_benchmark/
        ├── meta.txt
        └── <model_config>/
            └── inference_benchmark_<level>_<timing_pass>_<metric>_<plot_type>.csv
```

- The **directory** carries the group identity; the **file name mirrors the PNG file name** of the figure
  it belongs to, so a CSV and its figure can always be paired by name.
- Only the groups **enabled in `show.grp_names`** are ever collected.
- The drift rate is consolidated for the deployed-model rollout **and every `epoch_<E>` checkpoint rollout
  found on disk**, regardless of `show.generate_drift_rate_plot.train_epochs` (which only drives *rendering*).

## The `drift_rate_samples` family (schema `2.2`)

`drift_rate/` stores the **reduced** series (one median + a std-err band per timestep); the per-rollout
population behind it is reduced away. `drift_rate_samples/` is the family that exports that population, so
a consumer can compute a genuine boxplot (quartiles, 1.5·IQR whiskers, fliers) instead of re-drawing a
dispersion band with box geometry.

> ⚠️ **The schema-`2.1` layout of this family is RETIRED.** It was one *tall* CSV per group
> (`timestep,timestamp,sample_index,trajectory_name,seed_label,drift_rate,schema_version`) holding only the
> four horizon-ratio abscissae. Schema `2.2` replaces it with the per-seed **wide** layout below. This is a
> breaking change **of this family only** — `drift_rate/`, `training_wall_clock_time/` and
> `inference_benchmark/` are untouched.

One **wide** CSV per **model seed (trial)** of a group, one column per **trajectory id**, one row per
timestep of the **whole trimmed grid**:

```text
drift_rate_samples/
├── meta.txt
└── <grp_name>/
    ├── seed_1_test_time_models_comparaison_drift_rate_samples.csv
    ├── …
    └── seed_<n_seeds>_test_time_models_comparaison_drift_rate_samples.csv
```

Header and first row of
`RLRP-757-B1a-PI-TCN-multirun-test-time-plot/drift_rate_samples/AR-GRU/seed_1_….csv`:

```
timestep,timestamp,1,2,3,4,5,6,7,8
200,2.0,0.001333742404400938,0.0020274539703373394,0.002295973720526038,0.0032996934323938198,0.002439011663919706,0.0020615959558071455,0.002282648114920687,0.0031875873609185897
```

| Column | Content |
|---|---|
| `timestep` | the abscissa — **every** timestep of the trimmed grid, bit-identical to the `timestep` column of the reduced `drift_rate` CSV of the same group |
| `timestamp` | `timestep × timestamp_scale`, **empty** when the experiment has no timestamp axis (Lorenz) |
| `1`, `2`, … `K` | one column per **trajectory id**, resolved to a name through `param.trajectory_name_mapping`; the cell is that single rollout's drift value at that timestep (**no** reduction applied) |

There is **no `schema_version` column** — the version lives once, in `meta.txt`. The seed ordinal lives in
the **file name**, the trajectory id in the **column header**, so a rollout is never identified by row order.

> ⚠️ **A `(seed, trajectory)` pair with no rollout is an EMPTY cell, never `0.0`.** A ragged group
> (fewer seeds, or a missing rollout) writes nothing in that cell; a zero would silently inject a fake
> perfect rollout into the boxplot. Consumers must drop the empty cells, not fill them.

The metric `meta.txt` carries, in addition to the usual RLRP-818 provenance:

| Param | Meaning |
|---|---|
| `param.trajectory_name_mapping` | `{id: name}` of the CSV columns, e.g. `{1: 'ellipse_1', 2: 'ellipse_2', …}` — **per experiment**, so column `3` of another experiment is a different trajectory |
| `param.trajectory_ids_per_group` | the column ids actually written, per group |
| `param.seed_key_mapping` | `{grp: {seed_ordinal: [experiment_path, multirun_path]}}` — which run each `seed_<k>` file came from |
| `param.n_seeds` | per group, the number of model seeds (trials) = the number of CSV files |
| `param.n_rollouts` | per group, the number of `(seed, trajectory)` cells |
| `param.missing_cells` | per group, the number of empty cells (`0` everywhere in the shipped tree) |
| `param.sample_identity` | what one **cell** is — `rollout` |
| `param.sample_reduction` | `none` — the values are raw, not reduced |
| `param.float_format` | `repr` by default (bit-exact round-trip), or the `%`-format that was applied |
| `param.horizon_ratio_map` | **provenance only** since `2.2` — the resolved `{ratio: timestep}` map, kept for the producer/consumer cross-check. It no longer gates which abscissae exist |
| `param.timestep_stride` | `None` by default — the full grid; an integer keeps every *n*-th timestep |
| `param.warmup_end` / `param.horizon_length` | the usable window the ratios are measured in (`warmup_end … warmup_end + horizon_length`) |

> 🆕 **The whole grid is dumped, so a downstream consumer can freeze a figure at ANY timestep, horizon
> ratio or timestamp with no upstream re-consolidation.** That is the point of `2.2`: picking a new
> abscissa is a consumer-side one-line edit, never an RLRC re-run. Asking for an abscissa outside the
> trimmed window is a loud error, never a silent interpolation or clamp.

**Per-group seed and rollout counts are NOT uniform** — read them from `meta.txt`, never assume one:

| Experiment | seeds × trajectories = rollouts per group | On-disk size |
|---|---|---|
| `RLRP-757-B1a-PI-TCN-…` | 8 × 8 = 64 | 16 MB |
| `RLRP-757-F1-neurobem-…` | 8 × 12 = 96 | 27 MB |
| `RLRP-757-Fa1-ugv-…` | 8 × 2 = 16, except `Det TCN` 4 × 2 = 8 | 1.1 MB |
| `RLRP-757-Lorenz-E4-…` | 8 × 3 = 24, except `AR-TCN` and `(ours)` 6 × 3 = 18 | 11 MB |

≈ **55 MB** total. Full `repr` precision ships by default; setting
`consolidate_data.drift_rate_sample_float_format='%.9g'` roughly **halves** it, at 9 significant digits —
far beyond any plot's resolution. Dumping every epoch checkpoint
(`consolidate_data.drift_rate_sample_epochs=all`) multiplies the dump by the epoch count (hundreds of MB)
and is an opt-in for a per-epoch box study only.

## The two `meta.txt` levels

| File | Content |
|---|---|
| `<EXPERIMENT-NAME>/meta.txt` | generation date, experiment name, originating Hydra config + overrides, RLRC git sha, schema version, run-wide figure-variant tokens, and the `produced_by` list of metric sub-directories written so far |
| `<EXPERIMENT-NAME>/<metric>/meta.txt` | the producing function + hook, the metric parameters, the per-group source experiment paths, the raw `grp_name` → directory-slug mapping, the CSV column list and the CSV inventory |

## Schema

The authoritative column contract lives in
`src/tools/experiment_data_consolidation/consolidation_schema.py` of the RLRC repository; see
`src/tools/experiment_data_consolidation/README.md` for the per-metric column tables.
For `drift_rate`, CSVs contain only `timestep,timestamp,drift_rate,drift_rate_lower,drift_rate_upper`
with all static group and experiment metadata consolidated in `meta.txt`.

The current schema version is **`2.2`** (RLRP-819). `2.0 → 2.1` introduced the `drift_rate_samples` family;
`2.1 → 2.2` **redesigns that family only**, from the tall per-group CSV to the wide per-seed layout above.
The `drift_rate`, `training_wall_clock_time` and `inference_benchmark` CSVs are **byte-unchanged** across
both bumps, so schema-`2.0` and schema-`2.1` files remain valid and are read without a warning (the
standalone loader ships `COMPATIBLE_SCHEMA_VERSIONS = {"2.0", "2.1", "2.2"}` and
`EXPECTED_SCHEMA_VERSION = "2.2"`).

## How to regenerate

From the RLRC repository, **inside the DNA container** and from `src/`, run the producing app with the
consolidation enabled, e.g.:

```shell
cd src && python launcher/multirun_testtime_rollout_plot.py \
    multirun_testtime_rollout_plot@_global_=icra2026_pi_tcn_RLRP-757-B1a \
    consolidate_data.enable=true
```

The `drift_rate_samples` family is written **only** when the sample export is also enabled:

```shell
cd src && python launcher/multirun_testtime_rollout_plot.py \
    multirun_testtime_rollout_plot@_global_=<preset> \
    consolidate_data.enable=true \
    consolidate_data.emit_samples=true
```

The RLRC keys that drive the family are `consolidate_data.{enable, emit_samples,
drift_rate_sample_epochs, drift_rate_sample_float_format, drift_rate_sample_timestep_stride,
drift_rate_sample_horizon_ratios}`. The `2.1`-era `drift_rate_sample_timesteps` and `sample_stride` keys
are **retired**.

```shell
python3 launcher/model_inference_benchmark_plot.py \
    --config-name=model_inference_benchmark_plot \
    plot.input=<path/to/inference_benchmark.json> \
    consolidate_data.enable=true
```

Re-running overwrites the affected CSVs and rewrites the meta files in place.
