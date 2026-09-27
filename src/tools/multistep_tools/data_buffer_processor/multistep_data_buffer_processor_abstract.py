# coding=utf-8
import abc
from typing import Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.multistep_tools.buffer_unit import MultistepBuffer
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)


class MultistepDataBufferProcessorAbstract:
    def __init__(
        self,
        history_len: int,
        horizon_len: Union[int, float] = 1,
        obs_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, obs: obs,
        obs_buffer_append_callback: Optional[Callable] = lambda ms_buffer, next_obs: next_obs,
        act_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, act=None: act,
        act_buffer_append_callback: Optional[Callable] = lambda ms_buffer, act: act,
        consol_log: bool = True,
    ):
        """Observation/action space data buffer for learning multistep state representation
        e.g.: F110-gym RaceCar, Gymnasium Reacher, synosoidal function

        Information is structured by timestep, meaning that for a 3 dimensions observation,
        the observation are stored and retrive such that

            `{(D1, D2, D3)_t=1, (D1, D2, D3)_t=2, ... , (D1, D2, D3)_t=HI }`

        with dimension `D`, timestep `t` and history `HI`.

        **Notes on observation/action history:**

        The observation/action at timestep t are given by `get_compose_obs()` and
        `get_compose_act()` with respect to `history_len` param. e.g.:

        - history_len=1` ⇒
          `get_compose_obs()` return `[obs_t]` and `get_compose_act()` return `[act_t]`
        - history_len=3` ⇒
          `get_compose_obs()` return `[obs_t-2, obs_t-1, obs_t, act_t-2, act_t-1]`
          and `get_compose_act()` return `[act_t]`

        **Notes on observation/action horizon:**

        The `horizon_len` param set how much futur obs and act will be part of the
        `get_compose_next_obs()` returned array. Note that the array shape is determined by
        `history_len`. e.g.:

        - `horizon_len = 1, history_len=3` ⇒
          `get_compose_next_obs()` return `[obs_t-1, obs_t, obs_t+1, act_t-1, act_t]`
        - `horizon_len = 3, history_len=3` ⇒
          `get_compose_next_obs()` return `[obs_t+1, obs_t+2, obs_t+3, act_t+1, act_t+2]`

        **Notes on object output's:**

        Both `get_compose_obs` and `get_compose_next_obs` output shape are determined by
        `history_len` setting

        - the composed observation lenght:
          `get_compose_obs()` ⇒ `obs_singlestep_len * history_len + act_singlestep_len * (
          history_len - 1)`
        - the composed next observation length:
          `get_compose_next_obs()` ⇒ `obs_singlestep_len * history_len + act_singlestep_len * (
          history_len - 1)`
        - the next single-step obs length:
          `get_compose_next_obs()[compose_next_multistep_obs_to_next_singlestep_obs_slice]`
          ⇒ `obs_singlestep_len`


        **Notes on constructor param's:**

        - Function signature passed to `obs_buffer_reset_callback` must take 2 arg and return the
          modified reset observation e.g.:

          >>> obs_buffer_reset_callback(
          >>>           ms_buffer, # MultistepDataBufferProcessorAbstract
          >>>           obs, # np.ndarray
          >>>       ) -> np.ndarray

        - Function signature passed to `obs_buffer_append_callback` must take 2 arg and return the
          modified observation e.g.:

          >>> obs_buffer_append_callback(
          >>>           ms_buffer, #MultistepDataBufferProcessorAbstract
          >>>           next_obs, #np.ndarray
          >>>       ) -> np.ndarray

          This callback is used by `step(act)` method and receive the `next_obs`

        - Function signature passed to `act_buffer_reset_callback` must take 1 arg and return the
          modified reset action e.g.:

          >>> act_buffer_reset_callback(
          >>>           ms_buffer, #MultistepDataBufferProcessorAbstract
          >>>           act = None, #Optional[np.ndarray]
          >>>       ) -> np.ndarray

        - Function signature passed to `act_buffer_append_callback` must take 2 arg and return the
          modified action e.g.:

          >>> act_buffer_append_callback(
          >>>           ms_buffer, #MultistepDataBufferProcessorAbstract
          >>>           act, #np.ndarray
          >>>       ) -> np.ndarray


        :param history_len: The numbers of past observations to keep in the obs and act buffer
        :param horizon_len: the lenght of the forward horizon components in the
         compose next observation. (Note: Can be a float, in which case it will be considered as
         an history_len ratio)
        :param obs_buffer_reset_callback: callback executed at env reset
        :param obs_buffer_append_callback: callback executed before appending obs to buffer
        :param act_buffer_reset_callback: callback executed at env reset
        :param act_buffer_append_callback: callback executed before appending act to buffer
        :param consol_log:
        """
        self._consol_log = consol_log
        if self._consol_log:
            consol_msg_universal_one_liner(
                f"Setting up with history_len={history_len}, horizon_len={horizon_len}"
            )

        # ....Init buffers.........................................................................
        assert history_len >= 1
        self.history_len = history_len

        if isinstance(horizon_len, float):
            horizon_len = round(history_len * horizon_len)
            if horizon_len < 1:
                horizon_len = 1
            if self._consol_log:
                consol_msg_universal_one_liner(
                    "`horizon_len` arg interpreted as an `history_len` ratio, "
                    f"converting to `horizon_len={horizon_len}`"
                )

        assert (
            1 <= horizon_len <= self.history_len
        ), f"1 <= horizon_len {horizon_len} <= history_len {self.history_len}"
        self.horizon_len = horizon_len

        self.obs_buffer = MultistepBuffer(
            multistep_len=self.history_len,
            single_step_len=self.target_singlestep_obs_len,
            multistep_extension_len=self.horizon_len,
            info="Observations",
            consol_log=self._consol_log,
        )

        self.act_buffer = MultistepBuffer(
            multistep_len=self.history_len,
            single_step_len=self.target_singlestep_act_len,
            multistep_extension_len=self.horizon_len - 1,
            info="Actions",
            consol_log=self._consol_log,
        )

        self.reward_buffer = MultistepBuffer(
            multistep_len=1,
            single_step_len=1,
            multistep_extension_len=self.horizon_len - 1,
            info="Temporaly aligned reward",
            consol_log=self._consol_log,
        )

        self.terminated_buffer = MultistepBuffer(
            multistep_len=1,
            single_step_len=1,
            multistep_extension_len=self.horizon_len - 1,
            info="Temporaly aligned terminated",
            consol_log=self._consol_log,
        )

        self.truncated_buffer = MultistepBuffer(
            multistep_len=1,
            single_step_len=1,
            multistep_extension_len=self.horizon_len - 1,
            info="Temporaly aligned truncated",
            consol_log=self._consol_log,
        )

        # ....Setup callbacks......................................................................
        self.obs_buffer_reset_callback = obs_buffer_reset_callback
        self.obs_buffer_append_callback = obs_buffer_append_callback
        self.act_buffer_reset_callback = act_buffer_reset_callback
        self.act_buffer_append_callback = act_buffer_append_callback

        if obs_buffer_reset_callback:
            assert isinstance(obs_buffer_reset_callback, Callable)

        if obs_buffer_append_callback:
            assert isinstance(obs_buffer_append_callback, Callable)

        if act_buffer_reset_callback:
            assert isinstance(act_buffer_reset_callback, Callable)

        if act_buffer_append_callback:
            assert isinstance(act_buffer_append_callback, Callable)

    @property
    @abc.abstractmethod
    def target_singlestep_obs_len(self) -> int:
        """Set the target single step observation length, i.e. the size of the learned feature"""
        pass

    @property
    @abc.abstractmethod
    def target_singlestep_act_len(self) -> int:
        """Set the target single step action length, i.e. the size of the learned feature"""
        pass

    @property
    @abc.abstractmethod
    def source_obs_len(self) -> int:
        """Set the source environment single step observation length"""
        pass

    @property
    @abc.abstractmethod
    def source_act_len(self) -> int:
        """Set the source environment single step action length"""
        pass

    @property
    @abc.abstractmethod
    def composed_observation_space(self) -> gym.spaces.Box:
        """The arbitrary composed observation space"""
        pass

    @property
    @abc.abstractmethod
    def composed_action_space(self) -> gym.spaces.Box:
        """The arbitrary composed action space"""
        pass

    @property
    def _obs_buffer_history_end_idx(self) -> int:
        return self.target_singlestep_obs_len * self.history_len

    @property
    def _act_buffer_history_end_idx(self) -> int:
        return self.target_singlestep_act_len * (self.history_len - 1)

    @property
    def _obs_buffer_prediction_ofset_start_idx(self) -> int:
        return (self.obs_buffer.capacity * self.target_singlestep_obs_len) - (
            self.target_singlestep_obs_len * self.history_len
        )

    @property
    def _act_buffer_prediction_offset_start_idx(self) -> int:
        return (self.act_buffer.capacity * self.target_singlestep_act_len) - (
            self.target_singlestep_act_len * (self.history_len - 1)
        )

    @property
    def _compose_next_obs_multistep_obs_horizon_start_idx(self):
        return self.target_singlestep_obs_len * (self.history_len - self.horizon_len)

    @property
    def _compose_next_obs_multistep_obs_horizon_end_idx(self):
        return self.target_singlestep_obs_len * self.history_len

    @property
    def compose_next_multistep_obs_to_next_singlestep_obs_slice(self) -> slice:
        """Return the slice of `obs_{t+1}` (i.e. begin and end index) in a compose next
        multistep observation.

        Usage
        >>> multistep_next_obs = env_multistep.get_compose_next_obs()
        >>> next_obs = multistep_next_obs[
        >>>     env_multistep.compose_next_multistep_obs_to_next_singlestep_obs_slice
        >>> ]

        :return: the slice object of the single step observation at timestep t
        """
        next_singlestep_obs_start_idx = self._compose_next_obs_multistep_obs_horizon_start_idx
        next_singlestep_obs_end_idx = (
            self._compose_next_obs_multistep_obs_horizon_start_idx + self.target_singlestep_obs_len
        )
        return slice(next_singlestep_obs_start_idx, next_singlestep_obs_end_idx)

    @property
    def compose_next_obs_multistep_obs_horizon_slice(self):
        """The slice corresponding to [obs_t+1: obs_t+h] in a compose next multistep obs."""
        obs_horizon_start_idx = self._compose_next_obs_multistep_obs_horizon_start_idx
        obs_horizon_end_idx = self._compose_next_obs_multistep_obs_horizon_end_idx
        return slice(obs_horizon_start_idx, obs_horizon_end_idx)

    @property
    def compose_next_obs_multistep_act_horizon_slice(self):
        """The slice corresponding to [act_t+1: act_t+h-1] in a compose next multistep obs."""
        act_horizon_start_idx = self._compose_next_obs_multistep_obs_horizon_end_idx + (
            self.target_singlestep_act_len * (self.history_len - self.horizon_len)
        )
        act_horizon_end_idx = (
            self._compose_next_obs_multistep_obs_horizon_end_idx
            + self.target_singlestep_act_len * self.history_len
        )
        return slice(act_horizon_start_idx, act_horizon_end_idx)

    def get_model_input_size_requirement_for_compose_obs(self) -> int:
        # ToDo: RLRP-232 chore: assess moving model related logic out of
        # MultistepDataBufferProcessorAbstract
        """Get the NN model required input size for composed observation i.e.
        `len([*get_compose_obs, *get_compose_act]) == (obs_len * history_len +
        act_len * history_len)`

        :return: the NN model required in_size
        """
        return compute_multistep_model_in_size(
            singlestep_obs_len=self.target_singlestep_obs_len,
            singlestep_act_len=self.target_singlestep_act_len,
            multistep_len=self.history_len,
        )

    def get_model_out_size_requirement_for_compose_obs(self, learned_rewards: bool = False) -> int:
        # ToDo: RLRP-232 chore: assess moving model related logic out of
        # MultistepDataBufferProcessorAbstract
        """Get the NN model required output size for composed observation i.e.
        `len([*get_compose_obs, reward])  == (obs_len * history_len +
        act_len * (history_len -1) + reward )`

        :return: the NN model required out_size
        """
        return compute_multistep_model_out_size(
            singlestep_obs_len=self.target_singlestep_obs_len,
            singlestep_act_len=self.target_singlestep_act_len,
            multistep_len=self.history_len,
            learned_reward=learned_rewards,
        )

    def _get_multistep_obs_history(self) -> np.ndarray:
        """Get a flattened multistep observation of len `obs_len X history_len`

        Keep only observations on interval `[t-(history_len-1):t]`
        i.e:  remove the obs prediction offset from the observation buffer view

        :return: a multistep observation ndarray
        """
        obs_buffer = self.obs_buffer.get_buffer()
        current_obs_buffer = obs_buffer[0 : self._obs_buffer_history_end_idx]
        return current_obs_buffer

    def _get_multistep_act_history(self) -> np.ndarray:
        """Get a flattened multistep action of len `act_len X history_len`

        Keep only actions on interval `[t-(history_len-1):t-1]` so that the output stay
        aligned with the `get_compose_obs` output
        i.e:  remove the obs prediction offset from the action buffer view

        :return: a multistep action ndarray
        """
        act_buffer = self.act_buffer.get_buffer()
        current_act_buffer = act_buffer[0 : self._act_buffer_history_end_idx]
        return current_act_buffer

    def get_compose_obs(self) -> np.ndarray:
        """Get a flattened composed observation+action history of len
        `(obs_len X history_len) + (act_len X history_len)`

        The return array is a flattened combination following patern

        (..., O[1:Do]_1 + ... +  O[1:Do]_MS + A[1:Da]_1 + ... +  A[1:Da]_MS-1 )

        with O[1:Do]=observation, A[1:Da]=action dimension len and MS=multistep len.
        i.e., features dim are chunked by timesteps from 1 to MS.

        :return: a multistep observation/action ndarray ordered [obs's, act's]
        """

        return np.concatenate(
            (self._get_multistep_obs_history(), self._get_multistep_act_history())
        )

    def _get_multistep_obs_prediction_offset(self) -> np.ndarray:
        """Get a flattened multi-steps next observation of len 'obs.shape' X 'history_len'

        Keep only observations on the interval
        `[t + horizon_len - history_len : t + horizon_len]`
        i.e:  adjust the obs interval from the observation buffer view with respect to the
        obs offset

        :return: a multistep next observation ndarray
        """
        obs_buffer = self.obs_buffer.get_buffer()
        next_obs_buffer = obs_buffer[self._obs_buffer_prediction_ofset_start_idx :]
        return next_obs_buffer

    def _get_multistep_act_prediction_ofset(self) -> np.ndarray:
        """Get a flattened multistep next action of len 'act.shape' X 'history_len'

        Keep only actions on the interval
        `[t + horizon_len - history_len : t + horizon_len - 1]`
        i.e:  adjust the act interval from the action buffer view with respect to the obs offset

        :return: a multistep next action ndarray
        """
        act_buffer = self.act_buffer.get_buffer()
        next_act_buffer = act_buffer[self._act_buffer_prediction_offset_start_idx :]
        return next_act_buffer

    def get_compose_next_obs(self) -> np.ndarray:
        """Get a flattened compose next observation and action of len
        `(obs_len X history_len) + (act_len X (history_len-1))`

        The return array is a flattened combination following patern

        (..., O[1:Do]_1 + ... +  O[1:Do]_MS + A[1:Da]_1 + ... +  A[1:Da]_MS-1 )

        with O[1:Do]=observation, A[1:Da]=action dimension len and MS=multistep len.
        i.e., features dim are chunked by timesteps from 1 to MS.

        Note:
        - it always returns and array of multi-steps length equal to the history length to
          simplify handling the casse where the horizon_len < history_len.
        - sequence lengths of actions dim are one timestep shorter than the observations dim, hence
          the flattened ndarray patern.

        :return: a multistep next observation/action ndarray ordered `[next obs's, next act's]`
        """
        return np.concatenate(
            (
                self._get_multistep_obs_prediction_offset(),
                self._get_multistep_act_prediction_ofset(),
            )
        )

    def get_singlestep_action_at_timestep_t(self) -> np.ndarray:
        """Return a single-step action from the action buffer that is temporaly alligned with
        the `get_compose_obs` and `get_compose_next_obs` functions. This is the action at
        timestep `t`.

        :return: action at time-step `t`
        """
        act_buffer = self.act_buffer.get_buffer()
        current_act = act_buffer[
            self.target_singlestep_act_len
            * (self.history_len - 1) : self.target_singlestep_act_len
            * self.history_len
        ]
        return current_act

    def get_reward_at_timestep_t(self) -> Union[float, int]:
        return self.reward_buffer.get_buffer()[0]

    def get_terminated_at_timestep_t(self) -> bool:
        return self.terminated_buffer.get_buffer()[0]

    def get_truncated_at_timestep_t(self) -> bool:
        return self.truncated_buffer.get_buffer()[0]

    def get_compose_act(self) -> np.ndarray:
        """Get the compose action at time-step `t`

        :return: the composed action at time-step `t`
        """

        return self.get_singlestep_action_at_timestep_t()

    def _observation(self, observation) -> None:
        if self.obs_buffer_append_callback:
            processed_observation = self.obs_buffer_append_callback(self, next_obs=observation)
            self.obs_buffer.append(processed_observation)
        else:
            self.obs_buffer.append(observation)

        # self.obs_buffer.
        return None

    def _action(self, action) -> None:
        if self.act_buffer_append_callback:
            processed_action = self.act_buffer_append_callback(self, act=action)
            self.act_buffer.append(processed_action)
        else:
            self.act_buffer.append(action)
        return None

    def add_step(
        self,
        action: Union[np.ndarray, torch.Tensor, int, float],
        next_obs: Union[np.ndarray, torch.Tensor, int, float],
        reward: Union[int, float],
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Add step information to the buffers"""
        self._action(action)
        self._observation(next_obs)
        self.reward_buffer.append(reward)
        self.terminated_buffer.append(terminated)
        self.truncated_buffer.append(truncated)

        return None

    def reset_buffers(
        self,
        obs: Union[np.ndarray, torch.Tensor, int, float],
        act: Optional[Union[np.ndarray, torch.Tensor, int, float]] = None,
    ) -> None:
        """Reset buffers and add reset output information"""
        processed_observation = None
        act_buffer_init_values = None

        if self.obs_buffer_reset_callback:
            processed_observation = self.obs_buffer_reset_callback(self, obs=obs)
            self.obs_buffer.reset(processed_observation)
        else:
            self.obs_buffer.reset(obs)

        # (CRITICAL) ToDo: validate >> act handling logic ordering change
        if self.act_buffer_reset_callback and act is not None:
            act_buffer_init_values = self.act_buffer_reset_callback(self, act=act)
            if act_buffer_init_values is not None:
                self.act_buffer.reset(act_buffer_init_values)
            else:
                self.act_buffer.reset(np.zeros((self.target_singlestep_act_len,)))
        elif act is not None:
            self.act_buffer.reset(act)
        else:
            self.act_buffer.reset(np.zeros((self.target_singlestep_act_len,)))

        self.reward_buffer.reset(0.0)
        self.terminated_buffer.reset(False)
        self.truncated_buffer.reset(False)

        # .... Sanity check .......................................................................
        if isinstance(obs, (np.ndarray, torch.Tensor)):
            assert obs.shape[0] == self.source_obs_len, f"{obs.shape[0]=} != {self.source_obs_len=}"
        else:
            assert self.source_obs_len == 1, f"{self.source_obs_len=} != 1"

        if isinstance(act, (np.ndarray, torch.Tensor)):
            assert act.shape[0] == self.source_act_len, f"{act.shape[0]=} != {self.source_act_len=}"
        elif isinstance(act, (int, float)):
            assert self.source_act_len == 1, f"{self.source_act_len=} != 1"
        else:
            assert self.source_act_len == self.target_singlestep_act_len, (
                    f"{self.source_act_len=} != {self.target_singlestep_act_len=}")

        if isinstance(processed_observation, (np.ndarray, torch.Tensor)):
            assert processed_observation.shape[0] == self.target_singlestep_obs_len, f"{processed_observation.shape[0]=} != {self.target_singlestep_obs_len=}"
        elif isinstance(processed_observation, (int, float)):
            assert self.target_singlestep_obs_len == 1, f"{self.target_singlestep_obs_len=} != 1"

        if isinstance(act_buffer_init_values, (np.ndarray, torch.Tensor)):
            assert act_buffer_init_values.shape[0] == self.target_singlestep_act_len, f"{act_buffer_init_values.shape[0]=} != {self.target_singlestep_act_len=}"
        elif isinstance(act_buffer_init_values, (int, float)):
            assert self.target_singlestep_act_len == 1, f"{self.target_singlestep_act_len=} != 1"

        return None

    def pad_horizon_by_one(self) -> None:
        """This is a utility to pad the buffer by one timestep using the data added last

        :return: True if can still be padded, False if padding as reach horizon len
        """

        self.obs_buffer.pad_extension()
        self.act_buffer.pad_extension()
        self.reward_buffer.pad_extension(0.0)
        self.terminated_buffer.pad_extension(False)
        self.truncated_buffer.pad_extension(False)
        return None

    def horizon_offset_done_is_aligned_with_timestep_t(self) -> bool:
        return self.get_terminated_at_timestep_t() or self.get_truncated_at_timestep_t()

    def __repr__(self):
        """User representation. Dynamically handle property added at run time"""
        t_sp = " " * 2
        m_sp = " " * 2
        item_space = " " * 3
        class_name = self.__class__.__name__
        repr_str = f"\n{t_sp}{class_name}(\n"
        m_sp += t_sp
        for k, v in self.__dict__.items():
            repr_str += f"{m_sp}{item_space}{k}: {v}\n"
        repr_str += f"{m_sp})"
        return repr_str
