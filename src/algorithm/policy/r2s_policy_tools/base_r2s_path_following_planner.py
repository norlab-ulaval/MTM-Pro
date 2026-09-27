# coding=utf-8
import abc
import os
from typing import Any

import omegaconf

from algorithm.policy.r2s_policy_tools.base_r2s_policy import BaseR2SPolicy
from tools.console_tools.format import ConsoleFormat
from tools.domain_randomization_tools.spec_containers import EnvironmentSpec


class BaseR2SPathFollowingPlanner(BaseR2SPolicy):
    active_wpt_path: str

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        policy_cfg_key: str = "dynamic_sampler",
        simulator_cfg_key: str = "simulator",
    ):
        super().__init__(cfg, policy_cfg_key, simulator_cfg_key)

    def _init_configuration(
        self,
        cfg: omegaconf.DictConfig,
        policy_cfg_key: str,
        simulator_cfg_key: str,
    ) -> None:
        super()._init_configuration(cfg, policy_cfg_key, simulator_cfg_key)
        self._check_wpt_path_exist(self.simulator_cfg.wpt_path)
        self.active_wpt_path = self.simulator_cfg.wpt_path
        return None

    @abc.abstractmethod
    def act(self, obs_poses_x, obs_poses_y, obs_poses_theta) -> Any:
        """Adapter method mimicking the gym API act method.

        :param obs_poses_x:
        :param obs_poses_y:
        :param obs_poses_theta:
        :return: (steer, speed)
        """
        pass

    def show_full_spec(self) -> None:
        """Print the complette specification to console."""
        self._console_msg(
            "Show specs:\n\n"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
            f"active waypoint path: {self.active_wpt_path}\n"
            f"{omegaconf.OmegaConf.to_yaml({ 'actor': self.policy_cfg })}"
            f"{omegaconf.OmegaConf.to_yaml({ 'simulator': self.simulator_cfg })}"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
        )
        return None

    def show_essential_spec(self) -> None:
        """Print the essential specification to console.
        i.e. active_wpt_path, planner.planner_finetuning
        """
        planner_finetuning = omegaconf.OmegaConf.select(self.policy_cfg, "planner_finetuning")
        cfg_ = omegaconf.OmegaConf.to_yaml(
            {
                "active_waypoint_path": self.active_wpt_path,
                "actor": {"planner_finetuning": planner_finetuning},
            }
        )

        self._console_msg(
            "Show specs:\n\n"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
            f"{cfg_}"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
        )
        return None

    @abc.abstractmethod
    def set_waypoints(self, wpt_path: str) -> None:
        """Set policy path following waypoints manualy.

        :param wpt_path: the path to the waypoint file
        """
        pass

    def set_waypoints_from_spec(self, env_config_specs: EnvironmentSpec) -> None:
        """Set the planner path using the specifications of an EnvironmentSpec subclassed object.

        :param env_config_specs: the environment specification object
        :return: None
        """
        assert isinstance(env_config_specs, EnvironmentSpec)
        self._check_wpt_path_exist(env_config_specs.wpt_path)
        self.active_wpt_path = env_config_specs.wpt_path

        self.set_waypoints(wpt_path=self.active_wpt_path)
        return None

    def update_sim_config_from_spec(self, env_config_specs: EnvironmentSpec) -> None:
        super().update_sim_config_from_spec(env_config_specs)
        self.set_waypoints_from_spec(env_config_specs)
        return None

    def _check_wpt_path_exist(self, wpt_path) -> None:
        assert self._project_src_root_path is not None, (
            f"Missing required configuration " f"`cfg.project_src_root_path`"
        )
        assert os.path.exists(
            os.path.realpath(os.path.join(self._project_src_root_path, wpt_path))
        )
        return None
