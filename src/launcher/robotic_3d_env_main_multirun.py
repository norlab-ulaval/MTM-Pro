# coding=utf-8
from typing import Optional

import omegaconf
import hydra

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp
from launcher.main_utils import (
    select_math_env_pipeline_and_execute,
    select_robotic_3d_env_pipeline_and_execute,
)


# (NICE TO HAVE) ToDo: unit-test/integration test (ref task RLRP-319)


@hydra.main(
    config_path="configs", config_name="robotic_3d_env_main_multirun", version_base=None
)
def multirun_pipeline(cfg: omegaconf.DictConfig) -> None:
    # .... Overriding cfg for multirun ............................................................
    omegaconf.OmegaConf.update(
        cfg, "simulation_mode.rendering", "headless_fast", merge=False
    )
    consol_msg_universal_one_liner(f"Switching off rendering for headless run")

    # .... Execute multirun .......................................................................
    pipeline_app = R2S2RPipelineHydraApp(cfg)
    select_robotic_3d_env_pipeline_and_execute(cfg, pipeline_app)
    return None


if __name__ == "__main__":
    multirun_pipeline()
