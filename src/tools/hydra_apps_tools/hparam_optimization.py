# coding=utf-8
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra.utils
from hydra.core.hydra_config import HydraConf
import omegaconf
import numpy as np

from tools.console_tools.message import consol_msg_DEV_universal, consol_msg_universal_one_liner
from tools.hydra_apps_tools.hydra_utils import is_hydra_optuna_run
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist


class HyperparamObjectives:
    # (Priority) ToDo: implement unit-test
    cfg: omegaconf.DictConfig
    _records: Dict

    def __init__(self, cfg: omegaconf.DictConfig):
        """Object that handle hyperparameter objectives data such that experiment worklow be
        controlled easily from hydra config without having to change optimization objective
        manualy in the algorithm script each time we want to optimize for different objective(
        s). Handle single and multi objective.

        Require the following hydra config keys be set:

        >>>     hparam_optimizer:
        >>>         objectives_name: "[<the-objectives-key>,...]"
        >>>     hydra:
        >>>         sweeper:
        >>>             direction: [[minimize|maximize],...]

        An object can record multiple objective value using `record` method, only those that are
        set as objective in `cfg.hparam_optimizer.objectives_name` config will be considered at
        runtime.

        For non hydra sweep run, the object methods will simply return None value.

        :param cfg: hydra configuration
        """
        self.cfg = cfg
        self._records = dict()
        if self.is_hydra_hyperparameter_optimization_run:
            self._optimization_config_sanity_check()

        # Note: dont initialize `_records` dict with default value (for now) so that we can easily
        #       validate that `get_objective_records()` returned values were set with using
        #       `record(...)` or `auto_record(...)`.

    def record(self, name: str, value: float):
        """Record potential objective value.

        :param name: name as defined in `cfg.hparam_optimizer.objectives_name`
        :param value: the recorded value
        :return: self
        """
        self._records[name] = np.nan_to_num(value)

        return self

    def auto_record(self, local_dict: Dict):
        """Automaticly record objective value(s)

        Usage:
        >>> hparam_obj = HyperparamObjectives(cfg)
        >>> hparam_obj.auto_record(local_dict=locals())

        :param local_dict: current scope's local variables
        :return: self
        """
        if self.is_hydra_hyperparameter_optimization_run:
            for each in self.get_objectives_name():

                assert each in local_dict, f"Objective '{each}' not found in local_dict"

                self.record(each, local_dict[each])

            # consol_msg_universal_one_liner(
            #     f"Recorded objective(s) value for {self.get_objectives_name()}"
            # )
        return self

    @property
    def is_hydra_hyperparameter_optimization_run(self) -> bool:
        return is_hydra_optuna_run(self.cfg)

    def get_objective_records(self) -> Union[float, Tuple[float], None]:
        opt_objectives = None
        if self.is_hydra_hyperparameter_optimization_run:
            try:
                opt_objectives = tuple(
                    self._records.get(hp_obj) for hp_obj in self.get_objectives_name()
                )

                if len(opt_objectives) == 1:
                    opt_objectives = opt_objectives[0]
            except KeyError as e:
                raise KeyError("Missing optimization objective record") from e
        return opt_objectives

    def set_registred_objectives_with_optimum_worst(self) -> None:
        if self.is_hydra_hyperparameter_optimization_run:
            obj_name = self.get_objectives_name()
            limits_ = self.get_optimium_worst_values()
            for idx, hp_obj in enumerate(obj_name):
                self._records.setdefault(hp_obj, limits_[idx])
        return None

    def get_optimium_worst_values(self) -> Union[List[float], None]:
        direction_limits = None
        if self.is_hydra_hyperparameter_optimization_run:
            sweeper_direction = self.get_sweeper_direction()
            direction_limits = []
            for each in sweeper_direction:
                if each == "minimize":
                    direction_limits.append(np.nan_to_num(+math.inf))
                elif each == "maximize":
                    direction_limits.append(np.nan_to_num(-math.inf))
                else:
                    raise NotImplementedError(f"Sweeper direction {each} not supported")

        return direction_limits

    def get_sweeper_direction(self) -> List[str]:
        hydra_config = hydra.utils.HydraConfig.get()
        sweeper_direction_cfg = hydra_config.sweeper.direction
        if omegaconf.OmegaConf.is_list(sweeper_direction_cfg):
            sweeper_direction_cfg = omegaconf.OmegaConf.to_object(sweeper_direction_cfg)
        else:
            sweeper_direction_cfg = [sweeper_direction_cfg]
        return sweeper_direction_cfg

    def get_objectives_name(self) -> List[str]:
        obj_name = self.cfg.hparam_optimizer.objectives_name
        if omegaconf.OmegaConf.is_list(obj_name):
            obj_name = omegaconf.OmegaConf.to_object(obj_name)
        else:
            obj_name = [obj_name]
        return obj_name

    def _optimization_config_sanity_check(self) -> None:
        obj_name = self.get_objectives_name()
        sweeper_direction = self.get_sweeper_direction()

        assert len(sweeper_direction) == len(obj_name), (
            f"`cfg.hydra.sweeper.direction` config field must match len of "
            f"`cfg.hparam_optimizer.objectives_name` config field "
            f"({len(sweeper_direction)} != "
            f"{len(obj_name)})"
        )
        return None
