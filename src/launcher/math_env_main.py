# coding=utf-8

import omegaconf
import hydra

from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp
from launcher.main_utils import select_math_env_pipeline_and_execute


@hydra.main(config_path="configs", config_name="math_env_main_default", version_base=None)
def run_pipeline(cfg: omegaconf.DictConfig) -> None:
    pipeline_app = R2S2RPipelineHydraApp(cfg)
    select_math_env_pipeline_and_execute(cfg, pipeline_app)
    return None


if __name__ == "__main__":
    run_pipeline()
