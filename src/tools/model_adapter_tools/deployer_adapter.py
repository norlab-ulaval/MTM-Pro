# coding=utf-8

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch

from tools.model_adapter_tools.base import ModelInputOutputAdapterBase


class Env2ModelObservationAdapter(ModelInputOutputAdapterBase):
    def __init__(self, array_in_len: int, array_out_len: int, adapter_fct: Callable):
        """
        Class for carying and validating environment to model adapter.
        Example usage: run-time rollout deployer.

        Expected pipeline: ... -> environment state out -> adapter -> model in -> ...

        Attribute `adapter_fct` expect a callable with the following function signature:

            model_adapter(obs: Union[np.ndarray,torch.Tensor],
                          ) -> Union[np.ndarray,torch.Tensor]

        Example: `adapter_fct = lambda obs: obs`.

        :param array_in_len: The environment observation length
        :param array_out_len: The customized environment observation length i.e. subset,
        transformed, ...
        :param adapter_fct: Fct that interface the target environment with the motion model
        """
        super().__init__(array_in_len, array_out_len, adapter_fct)

    def _set_adapter_kwarg_attribute(self) -> None:
        pass

    def _adapter_output_type(self) -> Tuple[Type[np.ndarray], Type[torch.Tensor]]:
        return np.ndarray, torch.Tensor


class Model2EnvNextObservationAdapter(ModelInputOutputAdapterBase):
    def __init__(self, array_in_len: int, array_out_len: int, adapter_fct: Callable):
        """
        Class for carrying and validating model to environment adapter.
        Example usage: run-time rollout deployer.

        Expected pipeline: ... -> model out -> adapter -> environment state in -> ...

        Attribute `adapter_fct` expect a callable with the following function signature:

            model_adapter( next_obs: Union[np.ndarray,torch.Tensor], info: Dict,
                           last_env_state: Union[None,Any],
                          ) -> Tuple[Union[Type[np.ndarray], Type[torch.Tensor]], Type[Dict]]

        Example: `adapter_fct = lambda next_obs, info, last_env_state: (next_obs, info)`.

        The `last_env_state=<last-env-state-value>` attribute should be the same one that was
        passed to `Env2ModelObservationAdapter.adapter_fct(in_arg=<last-env-state-value>)`.

        :param array_in_len: Is the model singlestep output size
        :param array_out_len: Is the target environment next observation length
        :param adapter_fct: Fct that interface the motion model with the target environment
        """
        super().__init__(array_in_len, array_out_len, adapter_fct)

    def _set_adapter_kwarg_attribute(self) -> None:
        self.adapter_kwarg_attribute.setdefault("last_env_state")
        self.adapter_kwarg_attribute.setdefault("info", {})
        return None

    def _adapter_output_type(
        self,
    ) -> Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Type[Dict]]:
        return (np.ndarray, torch.Tensor), dict


class Model2ModelSymmetricObservationAdapter(ModelInputOutputAdapterBase):
    def __init__(self, array_in_len: int, array_out_len: int, adapter_fct: Callable):
        """
        Class for carying and validating model-to-model adapter.
        Example usage: test-time rollout deployer.

        Expected pipeline: ... -> model out -> adapter -> model in -> ...

        Attribute `adapter_fct` expect a callable with the following function signature:

            model_adapter( next_obs: Union[np.ndarray,torch.Tensor], info: Dict,
                         ) -> Tuple[Union[Type[np.ndarray], Type[torch.Tensor]], Type[Dict]]

        Example: `adapter_fct = lambda next_obs, info: (next_obs, info)`.

        :param array_in_len: Is the model output size
        :param array_out_len: Is the model input size
        :param adapter_fct: Fct that interface the motion model with the target environment
        """
        assert (
            array_in_len == array_out_len
        ), f"array_in_len={array_in_len} != array_out_len={array_out_len}"

        super().__init__(array_in_len, array_out_len, adapter_fct)

    def _set_adapter_kwarg_attribute(self) -> None:
        self.adapter_kwarg_attribute.setdefault("info", {})
        pass

    def _adapter_output_type(
        self,
    ) -> Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Type[Dict]]:
        return (np.ndarray, torch.Tensor), dict

