# coding=utf-8

import omegaconf
import hydra

from launcher.main_utils import select_multirun_testtime_rollout_plot_and_execute
from tools.hydra_apps_tools.r2s2r_apps_utils import R2S2RPipelineHydraApp


@hydra.main(config_path="configs", config_name="multirun_testtime_rollout_plot", version_base=None)
def run_pipeline(cfg: omegaconf.DictConfig) -> None:
    pipeline_app = R2S2RPipelineHydraApp(cfg)
    select_multirun_testtime_rollout_plot_and_execute(cfg, pipeline_app)
    return None


if __name__ == "__main__":
    run_pipeline()
