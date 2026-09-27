# coding=utf-8
from dataclasses import dataclass
from typing import Callable, Type, Union

from tools.model_adapter_tools.base import ModelInputOutputAdapterBase
from tools.model_adapter_tools.deployer_adapter import (
    Env2ModelObservationAdapter,
    Model2EnvNextObservationAdapter,
    Model2ModelSymmetricObservationAdapter,
)


def check_deployer_adapter_legal_type(
    instance: ModelInputOutputAdapterBase, adapter_type: Type[ModelInputOutputAdapterBase]
) -> None:
    assert issubclass(adapter_type, ModelInputOutputAdapterBase)
    assert isinstance(instance, adapter_type) or callable(
        instance
    ), f"{type(instance)} should be either callable or type {adapter_type.__name__}."
    return None


@dataclass()
class DeployerAdapter:
    """
    Represents adapter for deploying and mapping input/output between environmental observations
    and model observations.

    Note: Passe callable attribute only if you know what your doing, use factory function
    `motion_model_container_factory_from_ms_data_buffer_processor` or
    `r2s_motion_model_container_factory` otherwise.

    :ivar target_env_to_model_ss_in_obs: A callable or adapter that translates target
        environment observations to model state-space inputs.
    :type target_env_to_model_ss_in_obs: Union[Callable, Env2ModelObservationAdapter]
    :ivar model_ss_out_to_target_env_next_obs: A callable or adapter that transforms model
        state-space outputs to target environment's next observations.
    :type model_ss_out_to_target_env_next_obs: Union[Callable, Model2EnvNextObservationAdapter]
    :ivar model_to_model_ss_obs: A callable or adapter for translating model-specific state-space
        data between output and input forms symmetrically.
    :type model_to_model_ss_obs: Union[Callable, Model2ModelSymmetricObservationAdapter]
    """

    target_env_to_model_ss_in_obs: Union[Callable, Env2ModelObservationAdapter] = lambda obs: obs
    model_ss_out_to_target_env_next_obs: Union[Callable, Model2EnvNextObservationAdapter] = (
        lambda next_obs, info, last_env_state: (next_obs, info)
    )
    model_to_model_ss_obs: Union[Callable, Model2ModelSymmetricObservationAdapter] = (
        lambda next_obs, info: (next_obs, info)
    )
    # model_ss_out_body_frame_to_world_frame: Union[
    #     Callable, Model2EnvNextObservationAdapter
    # ] = lambda next_obs, info, last_env_state: (next_obs, info)
