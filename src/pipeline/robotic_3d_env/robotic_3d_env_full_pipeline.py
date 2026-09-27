# coding=utf-8
import os
from typing import Optional

import omegaconf
from mbrl.util import ReplayBuffer

from algorithm.utils import seed_me
from pipeline.pipeline_utils.general.setup import (
    assert_ood_objective_has_ood_trajectories,
    get_model_description,
    setup_multistep_step_model_and_trainer,
    uder_cfg_validation,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.setup_utils import (
    create_validation_trajectory_rollouts,
    setup_test_target_rollouts,
)
from pipeline.pipeline_utils.robotic_env_pipeline_utils.train_and_deploy_utils import (
    robotic_3d_env_ms_model_train_and_deploy,
)
from pipeline.robotic_3d_env import (
    robotic_3d_env_generate_ms_hdf5_pipeline,
    robotic_3d_env_generate_ms_replaybuffer_pipeline,
)

from tools.hydra_apps_tools.hparam_optimization import HyperparamObjectives
from tools.hydra_apps_tools.hydra_utils import (
    get_hydra_experiment_cwd,
)

from tools.dna_dev_tools.dn_CI_tools import show_plot_unless_CI_server_runned
from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run

import matplotlib.pyplot as plt

# .... Matplotlib configuration ...................................................................
import matplotlib as mpl

from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp
from tools.multistep_tools.ms_replaybuffer_save_load_utils import (
    load_multistep_replaybuffer_with_spec,
)
from tools.multistep_tools.window_dataset.pipeline_utils import (
    build_window_data_source,
    build_window_dataset,
    describe_window_path,
    is_dataloader_data_manager,
    move_store_to_training_device,
    stamp_original_dataset_size,
)
from tools.console_tools.message import consol_msg_universal_one_liner

plt.style.use("classic")
mpl.rcParams["figure.facecolor"] = "white"

# mpl.rcParams["legend.markerscale"] = 5
# mpl.rcParams["legend.numpoints"] = 3
# mpl.rcParams["legend.scatterpoints"] = 5

mpl.rcParams["font.size"] = 14
mpl.rcParams["legend.loc"] = "best"
# mpl.rcParams['legend.loc'] = 'lower right'
mpl.rcParams["axes.prop_cycle"] = plt.cycler(color=["b", "g", "r", "y"])


def execute(
    cfg: omegaconf.DictConfig,
    headless: bool = False,
    pipeline_app: Optional[R2S2RPipelineHydraApp] = None,
) -> HyperparamObjectives:
    """Execute math toy environment Sim2Sim motion model learner feedback loop

    :param cfg: Hydra configuration file
    :param headless: turn environment rendering off (param for commandline flag)
    :param pipeline_app:
    :return: a HyperparamObjectives object
    """

    exp_dir_relative_path = get_hydra_experiment_cwd(cfg)
    seed, torch_rng = seed_me(cfg, output_torch_rdn_generator=True)

    # .... Configuration setting validation .......................................................
    uder_cfg_validation(cfg)

    # :::: Define environment dynamic :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # RLRP-824 (FR1): ``pipeline.data_manager`` selects the training data path. ``replay-buffer``
    # (default) is the legacy materialized multistep buffer, untouched below; ``dataloader`` keeps
    # the single-step trajectories ONCE (HDF5-cached ``SingleStepTrajectoryStore``) and composes
    # the ``(H, F)`` windows lazily per batch (``MultistepWindowDataset`` ->
    # ``WindowDataLoaderDataSource``), which is what makes ``F > H`` and the ``F=500`` / ``F=1000``
    # horizons fit in memory (plan KD2 / KD3).
    use_dataloader = is_dataloader_data_manager(cfg)
    ms_replay_buffer_explorable_region = None
    single_step_store = None

    if use_dataloader:
        single_step_store = robotic_3d_env_generate_ms_hdf5_pipeline.load_or_generate(
            cfg, headless
        )
    else:
        if not cfg.pipeline.get("force_regenerating_saved_ms_replaybuffer", False):
            ms_replay_buffer_explorable_region = load_multistep_replaybuffer_with_spec(
                cfg,
                override_load_dir=(
                    os.path.realpath(
                        os.path.join(
                            cfg.project_root_path,
                            cfg.environment.data_path,
                        )
                    )
                ),
            )

        if ms_replay_buffer_explorable_region is None:
            # # .... Create source single step replay buffer ................................................
            # ss_full_time_space_replay_buffers = create_ss_trajectory_replaybuffer_from_csv(
            #     cfg, exp_dir_relative_path, headless
            # )
            #
            # # .... Create multi-step replay buffer ........................................................
            # ms_replay_buffer_explorable_region = setup_source_multi_step_replay_buffer(
            #     cfg, ss_full_time_space_replay_buffers
            # )

            ms_replay_buffer_explorable_region = (
                robotic_3d_env_generate_ms_replaybuffer_pipeline.execute(cfg, headless)
            )

    # .... Create validation and tests trajectory rollouts ........................................
    equal_length_val_env_trjs = create_validation_trajectory_rollouts(cfg)

    (
            target_InD_rollouts,
            target_OOD_rollouts,
    ) = setup_test_target_rollouts(cfg, exp_dir_relative_path, headless)

    # Fail loudly/early when an HPO study optimises the OOD deploy metric but the
    # dataset declares zero OOD test trajectories (RLRP-778, plan D1).
    assert_ood_objective_has_ood_trajectories(cfg, len(target_OOD_rollouts))

    # .... Setup model ............................................................................
    motion_model_container, ms_trainer = setup_multistep_step_model_and_trainer(cfg)
    if use_dataloader:
        # Lazy window dataset sized on the model's composed output window ``W`` (``max(H, F)`` for
        # the MS->MS forecast family), validated against the container like a replay buffer, then
        # wrapped in the ERLL data source (one constant-batch-size ``DataLoader`` pair, FR12).
        # The store is placed on the training device first (``pipeline.dataloader.store_device``,
        # default ``auto`` -> CUDA model device) so the windows are composed on-device: no host
        # gather, no per-batch H2D copy (Valeria A100 host-bound profile, 2026-09-17).
        single_step_store = move_store_to_training_device(
            cfg, single_step_store, motion_model_container.dynamics_model.device
        )
        window_dataset = build_window_dataset(
            cfg, single_step_store, output_window_len=motion_model_container.output_window_len
        )
        motion_model_container.validate_with_window_dataset(window_dataset)
        stamp_original_dataset_size(cfg, len(window_dataset))
        consol_msg_universal_one_liner(describe_window_path(single_step_store, window_dataset))
        erll_source = build_window_data_source(
            cfg, window_dataset, device=motion_model_container.dynamics_model.device
        )
    else:
        motion_model_container.validate_with_replay_buffer(
            ms_replay_buffer_explorable_region
        )
        erll_source = ms_replay_buffer_explorable_region

    ms_model_description = get_model_description(cfg.ms_model, motion_model_container)

    # :::: Learn models :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    (
        best_pred_mae_score,
        ms_deploy_result,
        ms_tensorboard_writer,
        best_pred_val_loss,
    ) = robotic_3d_env_ms_model_train_and_deploy(
        cfg,
        erll_source,
        motion_model_container,
        ms_model_description,
        target_InD_rollouts,
        target_OOD_rollouts,
        ms_trainer,
        equal_length_val_env_trjs,
        exp_dir_relative_path,
        headless,
        torch_rng,
        pipeline_app=pipeline_app,
    )

    # ==== Post: Metric ===========================================================================

    # ==== Teardown ===============================================================================
    if show_plot_unless_CI_server_runned(True) and not is_pytest_run():
        ms_tensorboard_writer.close()

    plt.close("all")

    # Always-recorded deploy-stage compounded-prediction-mode MAE on the InD/OOD target
    # rollout sets, available as selectable HPO objectives (see cfg.hparam_optimizer.objectives_name).
    deploy_compounded_pred_mae_target_InD = ms_deploy_result.compounded_pred_mae_target_InD
    deploy_compounded_pred_mae_target_OOD = ms_deploy_result.compounded_pred_mae_target_OOD

    hyperparam_objectives = HyperparamObjectives(cfg)
    hyperparam_objectives.record("best_pred_mae_score", best_pred_mae_score)
    hyperparam_objectives.record("best_pred_val_loss", best_pred_val_loss)
    hyperparam_objectives.record(
        "deploy_compounded_pred_mae_target_InD", deploy_compounded_pred_mae_target_InD
    )
    hyperparam_objectives.record(
        "deploy_compounded_pred_mae_target_OOD", deploy_compounded_pred_mae_target_OOD
    )
    return hyperparam_objectives.auto_record(locals())
