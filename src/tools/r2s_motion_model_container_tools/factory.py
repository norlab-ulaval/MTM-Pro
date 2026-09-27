# coding=utf-8
from dataclasses import dataclass
from typing import Callable, Optional, Union

from tools.multistep_tools.data_buffer_processor import MultistepDataBufferProcessorAbstract

from tools.r2s_motion_model_container_tools.container import (
    R2SMotionModelContainer,
)
from tools.r2s_motion_model_container_tools.utils import DeployerAdapter
from tools.model_adapter_tools.base import (
    ModelInputOutputAdapterBase,
)
from tools.model_adapter_tools.deployer_adapter import (
    Env2ModelObservationAdapter,
    Model2EnvNextObservationAdapter,
    Model2ModelSymmetricObservationAdapter,
)
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)


@dataclass()
class TargetEnvironmentSpec:
    target_env_obs_shape: tuple
    target_env_next_obs_shape: tuple
    target_env_act_shape: tuple
    learned_rewards: bool = False


@dataclass()
class R2SMotionModelFactorySpec:
    singlestep_obs_len: int
    singlestep_act_len: int
    multistep_len: int
    target_env_obs_shape: tuple
    target_env_next_obs_shape: tuple
    target_env_act_shape: tuple
    learned_rewards: bool = False
    # RLRP-824: composed OUTPUT window length ``W`` in obs blocks. ``None`` => ``multistep_len``
    # (legacy ``obs*H + act*(H-1)`` layout, bit-exact). Resolve it from the model class with
    # ``MultiStepMLP.resolve_output_window_len(history_len, horizon_len)``.
    output_window_len: Optional[int] = None


def motion_model_container_factory_from_ms_data_buffer_processor(
    multistep_data_buffer_processor: MultistepDataBufferProcessorAbstract,
    target_env_spec: TargetEnvironmentSpec,
    target_env_to_model_ss_in_obs_adapter: Union[
        Callable, Env2ModelObservationAdapter
    ] = lambda obs: obs,
    model_ss_out_to_target_env_next_obs_adapter: Union[
        Callable, Model2EnvNextObservationAdapter
    ] = lambda next_obs, info, last_env_state: (next_obs, info),
    model_to_model_ss_obs_adapter: Union[
        Callable, Model2ModelSymmetricObservationAdapter
    ] = lambda next_obs, info: (next_obs, info),
) -> R2SMotionModelContainer:
    # RLRP-402 feat(MotionModelContainer): add support for MS data buffer processor
    #   ↳ (NICE TO HAVE) ToDo: Implement factory test case for codebase stability
    """
    Factory function to create an R2SMotionModelContainer from a multistep data buffer
    processor and various deployer adapters for transforming between different observation formats.

    This function constructs an R2SMotionModelContainer by using the specifications
    and configurations provided through the multistep data buffer processor, target
    environment specifications, and multiple observation adapters. It ensures the
    compatibility of the generated container by validating it with the given data processor.

    :param multistep_data_buffer_processor: The data buffer processor containing
        multistep histories and configurations, used to extract essential specifications
        for the motion model creation.
    :param target_env_spec: Specifications of the target environment, including shapes of
        observations, actions, next states, and other environment-related attributes.
    :param target_env_to_model_ss_in_obs_adapter: Obs array adapter used by run-time deployer
    :param model_ss_out_to_target_env_next_obs_adapter: Obs array adapter used by run-time deployer
    :param model_to_model_ss_obs_adapter: Obs array adapter used by test-time deployer
    :return: An instance of R2SMotionModelContainer, which encapsulates the motion model associated
        configurations and deployer adapters.
    """
    ms_db_proc = multistep_data_buffer_processor
    assert isinstance(ms_db_proc, MultistepDataBufferProcessorAbstract)
    assert isinstance(target_env_spec, TargetEnvironmentSpec)

    r2s_model_container = r2s_motion_model_container_factory(
        motion_model_spec=R2SMotionModelFactorySpec(
            singlestep_obs_len=ms_db_proc.target_singlestep_obs_len,
            singlestep_act_len=ms_db_proc.target_singlestep_act_len,
            multistep_len=ms_db_proc.history_len,
            target_env_obs_shape=target_env_spec.target_env_obs_shape,
            target_env_next_obs_shape=target_env_spec.target_env_next_obs_shape,
            target_env_act_shape=target_env_spec.target_env_act_shape,
            learned_rewards=target_env_spec.learned_rewards,
        ),
        target_env_to_model_ss_in_obs_adapter=target_env_to_model_ss_in_obs_adapter,
        model_ss_out_to_target_env_next_obs_adapter=model_ss_out_to_target_env_next_obs_adapter,
        model_to_model_ss_obs_adapter=model_to_model_ss_obs_adapter,
    )

    r2s_model_container.validate_with_ms_data_buffer_processor(ms_db_proc)

    return r2s_model_container


def r2s_motion_model_container_factory(
    motion_model_spec: R2SMotionModelFactorySpec,
    target_env_to_model_ss_in_obs_adapter: Union[
        Callable, Env2ModelObservationAdapter
    ] = lambda obs: obs,
    model_ss_out_to_target_env_next_obs_adapter: Union[
        Callable, Model2EnvNextObservationAdapter
    ] = lambda next_obs, info, last_env_state: (next_obs, info),
    model_to_model_ss_obs_adapter: Union[
        Callable, Model2ModelSymmetricObservationAdapter
    ] = lambda next_obs, info: (next_obs, info),
) -> R2SMotionModelContainer:
    """
    Factory function for seting up a R2SMotionModelContainer from a R2SMotionModelFactorySpec
    dataclass and  multistep model deployer input/output observation adapter.

    :param motion_model_spec: A motion model specification dataclass
    :param target_env_to_model_ss_in_obs_adapter: Obs array adapter used by run-time deployer
    :param model_ss_out_to_target_env_next_obs_adapter: Obs array adapter used by run-time deployer
    :param model_to_model_ss_obs_adapter: Obs array adapter used by test-time deployer
    :return: An instance of R2SMotionModelContainer, which encapsulates the motion model associated
        configurations and deployer adapters.
    """
    assert isinstance(motion_model_spec, R2SMotionModelFactorySpec)

    # .... Compute model ms input/output size .....................................................
    motion_model_ss_in_size = (
        motion_model_spec.singlestep_obs_len + motion_model_spec.singlestep_act_len
    )
    # (NICE TO HAVE) iceboxed: add assymetric logic to `motion_model_ss_out_size`
    #  i.e. motion_model_ss_out_size = motion_model_spec.singlestep_next_obs_len
    motion_model_ss_out_size = motion_model_spec.singlestep_obs_len

    motion_model_ms_in_size = compute_multistep_model_in_size(
        singlestep_obs_len=motion_model_spec.singlestep_obs_len,
        singlestep_act_len=motion_model_spec.singlestep_act_len,
        multistep_len=motion_model_spec.multistep_len,
    )

    # RLRP-824: the OUTPUT window is ``W = output_window_len`` obs blocks (``== multistep_len``
    # for every legacy spec).
    output_window_len = (
        motion_model_spec.multistep_len
        if motion_model_spec.output_window_len is None
        else motion_model_spec.output_window_len
    )
    motion_model_ms_out_size = compute_multistep_model_out_size(
        singlestep_obs_len=motion_model_spec.singlestep_obs_len,
        singlestep_act_len=motion_model_spec.singlestep_act_len,
        multistep_len=output_window_len,
        learned_reward=motion_model_spec.learned_rewards,
    )

    # .... Wrap adapter fct to respective ModelInputOutputAdapterBase child ..................................
    # Note: Will create adapter with default input/output value

    if not isinstance(target_env_to_model_ss_in_obs_adapter, ModelInputOutputAdapterBase):
        target_env_to_model_ss_in_obs_adapter = Env2ModelObservationAdapter(
            array_in_len=motion_model_spec.target_env_obs_shape[0],
            array_out_len=motion_model_spec.singlestep_obs_len,
            adapter_fct=target_env_to_model_ss_in_obs_adapter,
        )

    if not isinstance(model_ss_out_to_target_env_next_obs_adapter, ModelInputOutputAdapterBase):
        model_ss_out_to_target_env_next_obs_adapter = Model2EnvNextObservationAdapter(
            array_in_len=motion_model_ss_out_size,
            array_out_len=motion_model_spec.target_env_next_obs_shape[0],
            adapter_fct=model_ss_out_to_target_env_next_obs_adapter,
        )

    if not isinstance(model_to_model_ss_obs_adapter, ModelInputOutputAdapterBase):
        model_to_model_ss_obs_adapter = Model2ModelSymmetricObservationAdapter(
            array_in_len=motion_model_spec.singlestep_obs_len,
            array_out_len=motion_model_spec.singlestep_obs_len,
            adapter_fct=model_to_model_ss_obs_adapter,
        )

    return R2SMotionModelContainer(
        target_env_obs_shape=motion_model_spec.target_env_obs_shape,
        target_env_next_obs_shape=motion_model_spec.target_env_next_obs_shape,
        target_env_act_shape=motion_model_spec.target_env_act_shape,
        motion_model_ss_in_size=motion_model_ss_in_size,
        motion_model_ss_out_size=motion_model_ss_out_size,
        motion_model_ms_in_size=motion_model_ms_in_size,
        motion_model_ms_out_size=motion_model_ms_out_size,
        deployer_adapter=DeployerAdapter(
            model_to_model_ss_obs=model_to_model_ss_obs_adapter,
            target_env_to_model_ss_in_obs=target_env_to_model_ss_in_obs_adapter,
            model_ss_out_to_target_env_next_obs=model_ss_out_to_target_env_next_obs_adapter,
        ),
        multistep_len=motion_model_spec.multistep_len,
        learned_rewards=motion_model_spec.learned_rewards,
        output_window_len=output_window_len,
    )
