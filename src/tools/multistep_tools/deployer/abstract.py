# coding=utf-8
import abc
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2, \
    assert_is_OneDTransitionRewardModelV2
from tools.multistep_tools.buffer_unit import MultistepBuffer
from tools.multistep_tools.models import (
    AutoRegressiveSequenceIterator,
    MultiStepMLP,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer
from tools.r2s_motion_model_container_tools.utils import DeployerAdapter

from functools import wraps


if torch.cuda.is_available():
    CUDA_STREAM = torch.cuda.Stream()
else:
    CUDA_STREAM = None


def use_cuda_stream(stream: Optional[torch.cuda.Stream] = None):
    """
    Applies a CUDA stream to the decorated function if CUDA is available.

    Usage example:
    >>> @use_cuda_stream(CUDA_STREAM)
    >>> def run_inference(model, data):
    >>>     return model(data)

    :param stream: The CUDA stream to be used for the function's execution.
    :type stream: torch.cuda.Stream
    :return: A decorator that wraps the provided function to optionally use the
             specified CUDA stream for execution.
    :rtype: Callable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and stream is not None:
                with torch.cuda.stream(stream):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


class AbstractMultistepMotionModelDeployer(abc.ABC):
    """
    Base class for deploying multistep motion models using abstract methods and common utilities.

    This class serves as an abstract base for deploying multistep motion models encapsulated
    within a OneDTransitionRewardModelV2. It facilitates interaction with an external environment
    via action and observation buffers, and provides utilities for predicting environment state
    transitions. Subclasses are expected to implement specific predictive functionalities and
    interfacing logic. It is tailored for integration with reinforcement learning pipelines or
    custom simulation models. Includes configuration for deterministic sampling, state and action
    buffers, and random generator initialization.

    """

    one_d_tr_model: OneDTransitionRewardModelV2
    _adapter: DeployerAdapter
    _multistep_state_buffer: MultistepBuffer
    _multistep_act_buffer: MultistepBuffer
    mbrl_rng: torch.Generator
    next_state_deterministic_selection: bool
    next_state_sampling_size: int
    device: torch.device
    # RLRP-785 (A8): optional per-step benchmark timer set by the benchmark harness (default
    # ``None`` -> the deploy path stays byte-identical). ``Optional[DeployStepTimer]``; typed as
    # ``Any`` to keep the production import graph free of ``benchmark_tools`` when timing is OFF.
    step_timer: Optional[Any]

    def __init__(
        self,
        motion_model_container: R2SMotionModelContainer,
        next_state_deterministic_selection: bool = True,
        next_state_sampling_size: int = 1,
        consol_log: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        """
        Class for deploying a MultiStepMLP model wrapped in a OneDTransitionRewardModelV2
        from a R2SMotionModelContainer into a simulator.

        Example pipeline in mbrl-f110-gym MBRLRaceCar:
            ... -> SS motion-state -> MS motion model -> next SS motion-state -> ...

        :param motion_model_container: the motion model components and specification
        :param next_state_deterministic_selection: Force probabilistic model to use next state
             deterministic sampling
        :param next_state_sampling_size: The number of samples draws if next_state_deterministic_selection=False
        :param rng: Set random number generator. Generate a new one otherwise.
        """
        super().__init__()

        self.next_state_sampling_size = next_state_sampling_size
        self.next_state_deterministic_selection = next_state_deterministic_selection

        self.one_d_tr_model = motion_model_container.dynamics_model

        assert_is_OneDTransitionRewardModelV2(self.one_d_tr_model)
        assert isinstance(
            self.one_d_tr_model.model, MultiStepMLP
        ), f"Expected MultiStepMLP, got {type(self.one_d_tr_model.model)=} instead"

        # Those adapter roles are to interface the motion model with the target environment.
        self._adapter = motion_model_container.deployer_adapter
        # iceboxed: RLRP-154 feat: add obs-to-state and state-to-obs adapter management logic

        if consol_log:
            consol_msg_universal_one_liner(
                "\n   MBRL motion model: "
                f"\n    {motion_model_container.motion_model_name[0]}"
                f"\n        ↳ {motion_model_container.motion_model_name[1]}"
                f"\n        multistep_len={motion_model_container.multistep_len}"
                f"\n        num_elites={self.one_d_tr_model.num_elites}"
                "\n   Deployer:"
                "\n      next_state_deterministic_selection="
                f"{self.next_state_deterministic_selection}"
                "\n      next_state_sampling_size="
                f"{self.next_state_sampling_size}"
                f"\n    target_env_obs_shape={motion_model_container.target_env_obs_shape}"
                f"\n    target_env_act_shape={motion_model_container.target_env_act_shape}"
                "\n      target_env_next_obs_shape="
                f"{motion_model_container.target_env_next_obs_shape}"
                f"\n"
            )

        self.device = self.one_d_tr_model.device

        self._multistep_state_buffer = MultistepBuffer(
            multistep_len=motion_model_container.multistep_len,
            single_step_len=motion_model_container.singlestep_obs_len,
            info="Observation history",
            consol_log=consol_log,
            device=self.device,
        )

        self._multistep_act_buffer = MultistepBuffer(
            multistep_len=motion_model_container.multistep_len,
            single_step_len=motion_model_container.singlestep_act_len,
            info="Action history",
            consol_log=consol_log,
            device=self.device,
        )
        if rng:
            self.mbrl_rng = rng
        else:
            self.mbrl_rng = torch.Generator(device=self.device)

        # RLRP-785 (A8): benchmark hook, OFF by default. The benchmark harness sets this to a
        # ``DeployStepTimer`` per rollout and clears it in a ``finally`` (plan R11 lifecycle).
        self.step_timer = None

    @property
    def model_adapter(self) -> DeployerAdapter:
        return self._adapter

    @abc.abstractmethod
    # @use_cuda_stream(CUDA_STREAM)
    def predict_next_state(
        self,
        env_state: Union[torch.Tensor, np.ndarray],
        action: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[
        Union[
            Union[np.ndarray, torch.Tensor],
            Optional[torch.Tensor],
            Optional[Dict[str, torch.Tensor]],
            Union[np.ndarray, torch.Tensor],
            Optional[Dict[str, torch.Tensor]],
        ]
    ]:
        """
        Predict the next state of the environment given the current state and an action.

        This method is an abstract method meant to be implemented by subclasses. It takes
        the current environment state and an action, and predicts the next state of the
        environment. The specifics of the prediction method depend on the subclass implementation.

        The input and output types are flexible and allow for multiple formats including
        NumPy arrays and PyTorch tensors. Optionally, additional information may be returned
        in the form of dictionaries.

        :param env_state: Current state of the environment. Can be a NumPy array or a PyTorch
           tensor depending on the implementation.
        :param action: Action to be taken in the environment. Similar to `env_state`, it can
           be either a NumPy array or a PyTorch tensor.
        :return: A tuple containing the predicted next state as a NumPy array or PyTorch
           tensor. Additional optional information may include another optional tensor or
           dictionaries of auxiliary data.
        """
        raise NotImplementedError()

    def reset(
        self,
        state_initialization_value: Union[np.ndarray, torch.Tensor],
        action_initialization_value: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Reset components i.e.: multistep racecar state predictor and multistep observation buffer.

        If action_initialization_value=None, then the multistep action buffer will be initialized
        with zeros ndarray. The input and output types are flexible and allow for multiple formats
        including NumPy arrays and PyTorch tensors. Additional information may be returned
        in the form of dictionaries.

        :param state_initialization_value: State used to initialize the multistep buffer
            and the mbrl-model.
        :param action_initialization_value:  Action used to initialize the multistep buffer
            and the mbrl-model. Default to no initial action.
        :return: The mbrl-model state dictionary with key 'obs' and 'propagation_indices'.
        """
        state_initialization_value = self._adapter.target_env_to_model_ss_in_obs(
            state_initialization_value
        )

        singlestep_act_len = self._multistep_act_buffer.single_step_len

        # .... Reset multistep buffers ............................................................
        # Torch-first: always initialize buffers with tensors to avoid numpy→torch roundtrips
        if isinstance(state_initialization_value, np.ndarray):
            state_initialization_value = self._ndarray_to_tensor(state_initialization_value)

        if action_initialization_value is None:
            action_initialization_value = torch.zeros(
                singlestep_act_len, device=self.device
            )
        elif isinstance(action_initialization_value, np.ndarray):
            action_initialization_value = self._ndarray_to_tensor(
                action_initialization_value
            )

        self._multistep_state_buffer.reset(state_initialization_value)
        self._multistep_act_buffer.reset(action_initialization_value)

        # .... Setup composed observation .........................................................
        multistep_state = self._multistep_state_buffer.get_buffer()
        multistep_act = self._multistep_act_buffer.get_buffer()
        singlestep_act = multistep_act[:-singlestep_act_len]

        composed_observation = torch.cat((multistep_state, singlestep_act))

        # .... Auto-regressive logic ..............................................................
        if isinstance(self.one_d_tr_model.model, AutoRegressiveSequenceIterator):
            # consol_msg_universal_one_liner(">>>>>>>>>> reset memory in deployer")
            self.one_d_tr_model.model.wipe_ar_memory()

        # .... Predict reset observation ..........................................................
        with torch.inference_mode():
            # Note: here param 'obs=<tensor>' is only used to fetch the obs shape
            mbrl_model_state = self.one_d_tr_model.reset(
                obs=composed_observation,
                rng=self.mbrl_rng,
            )
        return mbrl_model_state

    def _ndarray_to_tensor(self, ndarray: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(ndarray).to(self.device)
