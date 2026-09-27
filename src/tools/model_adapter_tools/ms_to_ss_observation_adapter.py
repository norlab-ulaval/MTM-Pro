# coding=utf-8
from typing import Any, Callable, Tuple, Type, Union

import numpy as np
import torch

from tools.model_adapter_tools.base import ModelInputOutputAdapterBase

from tools.multistep_tools.utils import (
    convert_compose_next_obs_multistep_obs_to_next_single_step_obs,
)


class MultistepObservationToSinglestepObservationAdapter(ModelInputOutputAdapterBase):
    def __init__(
        self, array_in_len: int, array_out_len: int, multistep_obs_to_singlestep_obs_slice: slice
    ):
        """
        Adapter class to convert multistep observations into single-step observations.

        This class serves as an adapter to handle transformations from multistep
        observations to single-step observations for a given MultiSTep model. It leverages
        a slicing mechanism and a callable adapter function to achieve this conversion.

        The class is configured upon instantiation and integrates seamlessly with its
        parent class functionalities for observation handling.

        :param array_in_len: The model multistep output size
        :param array_out_len: The target environment next observation single-step length
        :param multistep_obs_to_singlestep_obs_slice: The slice object used
            to specify how multi-step observations are sliced and mapped to single-step
            observations.
        """
        self.multistep_obs_to_singlestep_obs_slice = multistep_obs_to_singlestep_obs_slice

        super().__init__(
            array_in_len,
            array_out_len,
            adapter_fct=convert_compose_next_obs_multistep_obs_to_next_single_step_obs,
        )

    def _set_adapter_kwarg_attribute(self) -> None:
        self.adapter_kwarg_attribute["compose_next_multistep_obs_to_next_singlestep_obs_slice"] = (
            self.multistep_obs_to_singlestep_obs_slice
        )
        return None

    def _adapter_output_type(self) -> Union[
        Tuple[Type[np.ndarray], Type[torch.Tensor]],
        Tuple[Tuple[Type[np.ndarray], Type[torch.Tensor]], Any],
    ]:
        return np.ndarray, torch.Tensor
