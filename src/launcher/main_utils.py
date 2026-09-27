# coding=utf-8
import os
import sys
import traceback
import warnings
from typing import Optional

import omegaconf
import optuna

from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from pipeline import multirun_testtime_rollout_plot_pipeline
from pipeline.math_env import (
    math_env_deploy_only_pipeline,
    math_env_full_pipeline,
    math_env_plotter_pipeline,
    math_env_ss_and_ms_full_pipeline,
)
from pipeline.robotic_3d_env import (
    robotic_3d_env_deploy_only_pipeline,
    robotic_3d_env_full_pipeline,
    robotic_3d_env_sanitize_dataset_pipeline,
    robotic_3d_env_generate_ms_hdf5_pipeline,
    robotic_3d_env_generate_ms_replaybuffer_pipeline,
    robotic_3d_env_plotter_pipeline,
    robotic_3d_env_train_step_dev_profiling_pipeline,
)
from tools.hydra_apps_tools.hparam_optimization import HyperparamObjectives
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import (
    consol_msg_f1tenth_vaul_main_one_liner,
    consol_msg_universal,
    consol_msg_universal_one_liner,
)
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp


def select_math_env_pipeline_and_execute(
    cfg: omegaconf.DictConfig, pipeline_app: R2S2RPipelineHydraApp
) -> Optional[HyperparamObjectives]:
    """
    Selects and executes the appropriate pipeline based on the configuration provided.

    :param cfg: Configuration object specifying the pipeline name and other parameters.
    :param pipeline_app: Application context for the pipeline, typically containing settings
    such as headless execution and other attributes.
    :return: None
    """
    hyperparam_objective = None
    try:
        if cfg.pipeline.name == "math_env_full_pipeline":
            math_env_full_pipeline.execute(
                cfg, headless=pipeline_app.headless, pipeline_app=pipeline_app
            )
        elif cfg.pipeline.name == "math_env_full_pipeline_multirun":
            math_env_full_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "math_env_full_pipeline_hyperparm_optimization":
            hyperparam_objective = math_env_full_pipeline.execute(
                cfg, headless=pipeline_app.headless, pipeline_app=pipeline_app
            )
        elif cfg.pipeline.name == "math_env_plotter_pipeline":
            math_env_plotter_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "math_env_deploy_only_pipeline":
            math_env_deploy_only_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "math_env_main_ss_and_ms_uder_pipeline":
            math_env_ss_and_ms_full_pipeline.execute(
                cfg, headless=pipeline_app.headless, pipeline_app=pipeline_app
            )
        else:
            consol_msg_universal_one_liner(
                f"{ConsoleFormat.MSG_ERROR_FORMAT}pipeline not implemented"
                f"{ConsoleFormat.MSG_END_FORMAT}"
            )
            exit(1)
    except KeyboardInterrupt:
        pass
    except optuna.TrialPruned:
        # RLRP-624 Phase B — pruner-driven trial termination. Mark PRUNED
        # in storage and emit a sentinel objective so the sweeper can finish
        # the job cleanly.
        pipeline_app.finalize_pruned()
        hyperparam_objective = HyperparamObjectives(cfg)
        hyperparam_objective.set_registred_objectives_with_optimum_worst()
    except Exception as e:
        show_debug_information(cfg)

        # Note: The exception scope is large on purpose
        if os.getenv("IS_SLURM_RUN") and not is_pytest_run():
            traceback.print_exc(file=sys.stderr)
            warnings.warn(str(e))
            hyperparam_objective = HyperparamObjectives(cfg)
            hyperparam_objective.set_registred_objectives_with_optimum_worst()
        else:
            raise
    finally:
        pipeline_app.teardown()

    return hyperparam_objective

def select_multirun_testtime_rollout_plot_and_execute(
    cfg: omegaconf.DictConfig, pipeline_app: R2S2RPipelineHydraApp
) -> Optional[HyperparamObjectives]:
    """
    Selects and executes the appropriate pipeline based on the configuration provided.

    :param cfg: Configuration object specifying the pipeline name and other parameters.
    :param pipeline_app: Application context for the pipeline, typically containing settings
    such as headless execution and other attributes.
    :return: None
    """
    hyperparam_objective = None
    try:
        if cfg.pipeline.name == "multirun_testtime_rollout_plot_pipeline":
            multirun_testtime_rollout_plot_pipeline.execute(cfg, headless=pipeline_app.headless)
        else:
            consol_msg_universal_one_liner(
                f"{ConsoleFormat.MSG_ERROR_FORMAT}pipeline not implemented"
                f"{ConsoleFormat.MSG_END_FORMAT}"
            )
            exit(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        show_debug_information(cfg)

        # Note: The exception scope is large on purpose
        if os.getenv("IS_SLURM_RUN") and not is_pytest_run():
            traceback.print_exc(file=sys.stderr)
            warnings.warn(str(e))
            hyperparam_objective = HyperparamObjectives(cfg)
            hyperparam_objective.set_registred_objectives_with_optimum_worst()
        else:
            raise
    finally:
        pipeline_app.teardown()

    return hyperparam_objective

def select_robotic_3d_env_pipeline_and_execute(
    cfg: omegaconf.DictConfig, pipeline_app: R2S2RPipelineHydraApp
) -> Optional[HyperparamObjectives]:
    """
    Selects and executes the appropriate pipeline based on the configuration provided.

    :param cfg: Configuration object specifying the pipeline name and other parameters.
    :param pipeline_app: Application context for the pipeline, typically containing settings
    such as headless execution and other attributes.
    :return: None
    """
    hyperparam_objective = None
    try:
        if cfg.pipeline.name == "robotic_3d_env_full_pipeline":
            robotic_3d_env_full_pipeline.execute(
                cfg, headless=pipeline_app.headless, pipeline_app=pipeline_app
            )
        elif cfg.pipeline.name == "robotic_3d_env_full_pipeline_multirun":
            robotic_3d_env_full_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_full_pipeline_hyperparm_optimization":
            hyperparam_objective = robotic_3d_env_full_pipeline.execute(
                cfg, headless=pipeline_app.headless, pipeline_app=pipeline_app
            )
        elif cfg.pipeline.name == "robotic_3d_env_sanitize_dataset_pipeline":
            robotic_3d_env_sanitize_dataset_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_generate_ms_replaybuffer_pipeline":
            robotic_3d_env_generate_ms_replaybuffer_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_generate_ms_hdf5_pipeline":
            robotic_3d_env_generate_ms_hdf5_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_train_step_dev_profiling_pipeline":
            robotic_3d_env_train_step_dev_profiling_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_plotter_pipeline":
            robotic_3d_env_plotter_pipeline.execute(cfg, headless=pipeline_app.headless)
        elif cfg.pipeline.name == "robotic_3d_env_deploy_only_pipeline":
            robotic_3d_env_deploy_only_pipeline.execute(cfg, headless=pipeline_app.headless)
        else:
            consol_msg_universal_one_liner(
                f"{ConsoleFormat.MSG_ERROR_FORMAT}pipeline not implemented"
                f"{ConsoleFormat.MSG_END_FORMAT}"
            )
            exit(1)
    except KeyboardInterrupt:
        pass
    except optuna.TrialPruned:
        # RLRP-624 Phase B — pruner-driven trial termination. Mark PRUNED
        # in storage and emit a sentinel objective so the sweeper can finish
        # the job cleanly.
        pipeline_app.finalize_pruned()
        hyperparam_objective = HyperparamObjectives(cfg)
        hyperparam_objective.set_registred_objectives_with_optimum_worst()
    except Exception as e:
        show_debug_information(cfg)

        # Note: The exception scope is large on purpose
        if os.getenv("IS_SLURM_RUN") and not is_pytest_run():
            traceback.print_exc(file=sys.stderr)
            warnings.warn(str(e))
            hyperparam_objective = HyperparamObjectives(cfg)
            hyperparam_objective.set_registred_objectives_with_optimum_worst()
        else:
            raise
    finally:
        pipeline_app.teardown()

    return hyperparam_objective


def show_debug_information(cfg: omegaconf.DictConfig) -> None:
    """Show debug information about the current environment and configuration.

    .. important:: This helper runs **inside the ``except`` block** of the
       pipeline dispatchers, so it must NEVER raise -- an exception raised here
       replaces the original traceback and masks the true failure.

       ``OmegaConf.to_yaml(cfg, resolve=True)`` resolves every interpolation,
       and a config in a partially-built state (e.g. ``ms_model.singlestep_obs_len
       = ${environment.obs_shape[0]}`` before ``environment.obs_shape`` is
       populated) makes it raise :class:`omegaconf.errors.InterpolationKeyError`.
       That is exactly the symptom that used to surface as the misleading
       top-level error while the real cause (e.g. a failed trajectory load) was
       hidden. We therefore fall back to the UNRESOLVED dump on any OmegaConf
       error so the original exception is preserved and re-raised by the caller.
    """
    consol_msg_universal("General debug information:")
    print(f"- Current working directory: {os.getcwd()}")
    try:
        print(f"- Pipeline name: {cfg.pipeline.name}")
    except Exception:  # never mask the original failure from a debug dump
        print("- Pipeline name: <unavailable>")
    try:
        cfg_yaml = omegaconf.OmegaConf.to_yaml(cfg, resolve=True)
    except Exception as dump_error:  # e.g. InterpolationKeyError on a partial cfg
        try:
            cfg_yaml = omegaconf.OmegaConf.to_yaml(cfg, resolve=False)
        except Exception:
            cfg_yaml = "<config could not be rendered>"
        cfg_yaml = (
            f"[debug-dump] resolved dump failed ({type(dump_error).__name__}: "
            f"{dump_error}); showing UNRESOLVED config instead so the original "
            f"error is not masked.\n\n{cfg_yaml}"
        )
    consol_msg_universal(f"Show hydra cfg:\n\n{cfg_yaml}")
    consol_msg_universal(f"Show environment variables\n\n{str(os.environ)}")
    return None
