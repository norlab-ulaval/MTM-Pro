# coding=utf-8
import abc
import os
from typing import Any, Callable, List, Optional, Union

import omegaconf
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import consol_msg, MSG_PREFIX
from tools.domain_randomization_tools.spec_containers import (
    EnvironmentSpec,
)
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist


class BaseR2SPolicy:
    simulator_cfg: Union[dict, omegaconf.DictConfig]
    policy_cfg: Union[dict, omegaconf.DictConfig]
    policy: Any
    _project_src_root_path: Optional[str]

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        policy_cfg_key: str = "dynamic_sampler",
        simulator_cfg_key: str = "simulator",
    ):
        """Convenient policy class mimicking part of the gym API (i.e. implement the 'act'
        method) and support fetching configuration specification from a EnvironmentSpec object.

        :param cfg: hydra configuration
        :param policy_cfg_key: policy related key to fetch in the cfg object
        :param simulator_cfg_key: simulator related key to fetch in the cfg object
        """
        self._init_configuration(cfg, policy_cfg_key, simulator_cfg_key)
        self._init_policy()

    def _init_configuration(
        self, cfg: omegaconf.DictConfig, policy_cfg_key: str, simulator_cfg_key: str
    ) -> None:
        self._project_src_root_path = cfg.get("project_src_root_path", None)

        self._policy_cfg_main_key = policy_cfg_key
        self._simulator_cfg_main_key = simulator_cfg_key

        if not is_cfg_key_exist(cfg, policy_cfg_key, raise_error=False):
            raise ValueError(f"'{policy_cfg_key=}' does not exist in hydra cfg")
        else:
            self.policy_cfg = omegaconf.OmegaConf.select(cfg, policy_cfg_key)
            self._policy_cfg_sanity_check(self.policy_cfg)

        if not is_cfg_key_exist(cfg, simulator_cfg_key, raise_error=False):
            raise ValueError(f"'{simulator_cfg_key=}' does not exist in hydra cfg")
        else:
            self.simulator_cfg = omegaconf.OmegaConf.select(cfg, simulator_cfg_key)
            self._sim_cfg_sanity_check(self.simulator_cfg)
        return None

    @abc.abstractmethod
    def _init_policy(self) -> None:
        """Policy initialization
        Note: The policy variable accept an arbitrary object type.
        """
        self.policy = None
        pass

    @abc.abstractmethod
    def act(self, *obs: Any) -> Any:
        """Adapter method mimicking the gym API act method."""
        pass

    @property
    @abc.abstractmethod
    def _sim_cfg_keys(self) -> Optional[List[str]]:
        """Specify any required simulator configuration key"""
        pass

    @property
    @abc.abstractmethod
    def _policy_cfg_keys(self) -> Optional[List[str]]:
        """Specify any required planner configuration key"""
        pass

    def _sim_cfg_sanity_check(self, sim_config: omegaconf.DictConfig) -> None:
        if self._sim_cfg_keys is not None:
            for each_sim_key in self._sim_cfg_keys:
                is_cfg_key_exist(
                    sim_config,
                    each_sim_key,
                    raise_error=True,
                    error_msg_prepend_key_parent=self._simulator_cfg_main_key,
                )
        return None

    def _policy_cfg_sanity_check(self, policy_config: omegaconf.DictConfig) -> None:
        if self._policy_cfg_keys is not None:
            for each_policy_key in self._policy_cfg_keys:
                is_cfg_key_exist(
                    policy_config,
                    each_policy_key,
                    raise_error=True,
                    error_msg_prepend_key_parent=self._policy_cfg_main_key,
                )
        return None

    def show_full_spec(self) -> None:
        """Print the complette specification to console."""
        self._console_msg(
            "Show specs:\n\n"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
            f"{omegaconf.OmegaConf.to_yaml({ 'actor': self.policy_cfg })}"
            f"{omegaconf.OmegaConf.to_yaml({ 'simulator': self.simulator_cfg })}"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
        )
        return None

    def update_sim_config_from_spec(
        self, env_config_specs: EnvironmentSpec, project_src_root_path: Optional[str] = None
    ) -> None:
        """Update the stored simulator configuration `simulator_cfg` using the specifications of an
        EnvironmentSpec object.

        Note: it wont modify the stored planner configuration `planner_cfg`.

        :param env_config_specs: the environment specification object
        :param project_src_root_path: (optional) Required if cfg.project_src_root_path is not set
        :return: None
        """
        assert isinstance(env_config_specs, EnvironmentSpec)

        if self._project_src_root_path is None:
            assert project_src_root_path is not None, (
                f"Param `project_src_root_path` is "
                f"required since "
                f"`cfg.project_src_root_path` was not set"
            )
            self._project_src_root_path = project_src_root_path

        simulator_cfg_path = env_config_specs.simulator_cfg_path
        abs_simulator_cfg_path = os.path.join(self._project_src_root_path, simulator_cfg_path)
        with open(abs_simulator_cfg_path) as file:
            loaded_sim_cfg = omegaconf.OmegaConf.load(file)
            cfg_ = omegaconf.OmegaConf.create(
                f"""
                project_src_root_path: {self._project_src_root_path}
                sim_cfg: {loaded_sim_cfg}
                """
            )
            resolved_cfg = omegaconf.OmegaConf.to_yaml(cfg_, resolve=True)
            cfg_ = omegaconf.OmegaConf.create(resolved_cfg)
            self._sim_cfg_sanity_check(cfg_.sim_cfg)
            self.simulator_cfg = cfg_.sim_cfg
        return None

    def render_callback(self) -> Optional[Callable]:
        """Rendering callback passed to F110-gym based environment render method

        :return: a callable taking one argument or None
        """
        return lambda x: x

    def _console_msg(self, msg) -> None:
        consol_msg(
            msg=msg,
            who_am_i=f"{MSG_PREFIX} {self.__class__.__name__}",
            space_before=False,
            space_after=False,
        )
        return None


