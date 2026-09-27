# coding=utf-8
from typing import Callable, List, Optional, Union
import numpy as np

import gymnasium as gym

from tools.multistep_tools.data_buffer_processor.multistep_data_buffer_processor_abstract import (
    MultistepDataBufferProcessorAbstract,
)


class MultistepDataBufferProcessorGymnasiumFullSpace(MultistepDataBufferProcessorAbstract):
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Box

    def __init__(
        self,
        env: Union[
            gym.Wrapper,
            gym.Env,
        ],
        history_len: int,
        horizon_len: Union[int, float] = 1,
        obs_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, obs: obs,
        obs_buffer_append_callback: Optional[Callable] = lambda ms_buffer, next_obs: next_obs,
        act_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, act=None: act,
        act_buffer_append_callback: Optional[Callable] = lambda ms_buffer, act: act,
        consol_log: bool = True,
    ):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        super().__init__(
            history_len,
            horizon_len,
            obs_buffer_reset_callback,
            obs_buffer_append_callback,
            act_buffer_reset_callback,
            act_buffer_append_callback,
            consol_log=consol_log,
        )

    @property
    def target_singlestep_obs_len(self) -> int:
        return self.source_obs_len

    @property
    def target_singlestep_act_len(self) -> int:
        return self.source_act_len

    @property
    def source_obs_len(self) -> int:
        return self.observation_space.shape[0]

    @property
    def source_act_len(self) -> int:
        return self.action_space.shape[0]

    @property
    def composed_observation_space(self) -> gym.spaces.Box:
        low_m_obs = np.tile(self.observation_space.low, self.history_len)
        high_m_obs = np.tile(self.observation_space.high, self.history_len)
        len_m_obs = self.target_singlestep_obs_len * self.history_len
        low_m_act = np.tile(self.action_space.low, self.history_len - 1)
        high_m_act = np.tile(self.action_space.high, self.history_len - 1)
        len_m_act = self.target_singlestep_act_len * (self.history_len - 1)
        return gym.spaces.Box(
            low=np.concatenate((low_m_obs, low_m_act)),
            high=np.concatenate((high_m_obs, high_m_act)),
            shape=(len_m_obs + len_m_act,),
            dtype=self.observation_space.dtype,
        )

    @property
    def composed_action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.action_space.low,
            high=self.action_space.high,
            shape=self.action_space.shape,
            dtype=self.action_space.dtype,
        )


class MultistepDataBufferProcessorGymnasiumSubsetSpace(MultistepDataBufferProcessorAbstract):
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Box
    obs_subset_array_indexes: List[int]

    def __init__(
        self,
        env: Union[
            gym.Wrapper,
            gym.Env,
        ],
        history_len: int,
        obs_buffer_single_step_len: int,
        obs_subset_array_indexes: List[int],
        horizon_len: Union[int, float] = 1,
        obs_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, obs: obs,
        obs_buffer_append_callback: Optional[Callable] = lambda ms_buffer, next_obs: next_obs,
        act_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, act=None: act,
        act_buffer_append_callback: Optional[Callable] = lambda ms_buffer, act: act,
        consol_log: bool = True,
    ):
        """Observation/action space data buffer for learning multistep state representation of a
        subset of the state space feature's
        e.g.: F110-gym RaceCar with velocity only state feature's

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
          >>>           ms_buffer: MultistepDataBufferProcessorGymnasiumSubsetSpace,
          >>>           obs: np.ndarray
          >>>       ) -> np.ndarray

        - Function signature passed to `obs_buffer_append_callback` must take 2 arg and return the
          modified observation e.g.:

          >>> obs_buffer_append_callback(
          >>>           ms_buffer: MultistepDataBufferProcessorGymnasiumSubsetSpace,
          >>>           next_obs: np.ndarray
          >>>       ) -> np.ndarray

          This callback is used by `step(act)` method and receive the `next_obs`

        - Function signature passed to `act_buffer_reset_callback` must take 1 arg and return the
          modified reset action e.g.:

          >>> act_buffer_reset_callback(
          >>>           ms_buffer: MultistepDataBufferProcessorGymnasiumSubsetSpace
          >>>       ) -> np.ndarray

        - Function signature passed to `act_buffer_append_callback` must take 2 arg and return the
          modified action e.g.:

          >>> act_buffer_append_callback(
          >>>           ms_buffer: MultistepDataBufferProcessorGymnasiumSubsetSpace,
          >>>           act: np.ndarray
          >>>       ) -> np.ndarray

        :param env: gym environment
        :param history_len: The numbers of past observations to keep in the obs buffer
        :param obs_buffer_single_step_len: the length of a single observation
        :param obs_subset_array_indexes: The index of each observation to keep
        :param horizon_len: the lenght of the forward horizon components in the
         compose next observation. (Note: Can be a float, in which case it will be considered as
         an history_len ratio)
        :param obs_buffer_reset_callback: callback executed at env reset
        :param obs_buffer_append_callback: callback executed before appending obs to buffer
        :param act_buffer_reset_callback: callback executed at env reset
        :param act_buffer_append_callback: callback executed before appending act to buffer
        """

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        assert len(obs_subset_array_indexes) == obs_buffer_single_step_len
        self.obs_subset_array_indexes = obs_subset_array_indexes

        super().__init__(
            history_len,
            horizon_len,
            obs_buffer_reset_callback,
            obs_buffer_append_callback,
            act_buffer_reset_callback,
            act_buffer_append_callback,
            consol_log=consol_log,
        )

    @property
    def target_singlestep_obs_len(self) -> int:
        return len(self.obs_subset_array_indexes)

    @property
    def target_singlestep_act_len(self) -> int:
        return self.source_act_len

    @property
    def source_obs_len(self) -> int:
        return self.observation_space.shape[0]

    @property
    def source_act_len(self) -> int:
        return self.action_space.shape[0]

    @property
    def composed_observation_space(self) -> gym.spaces.Box:
        low_m_obs = np.tile(
            self.observation_space.low[self.obs_subset_array_indexes], self.history_len
        )
        high_m_obs = np.tile(
            self.observation_space.high[self.obs_subset_array_indexes], self.history_len
        )
        len_m_obs = self.target_singlestep_obs_len * self.history_len
        low_m_act = np.tile(self.action_space.low, self.history_len - 1)
        high_m_act = np.tile(self.action_space.high, self.history_len - 1)
        len_m_act = self.target_singlestep_act_len * (self.history_len - 1)
        return gym.spaces.Box(
            low=np.concatenate((low_m_obs, low_m_act)),
            high=np.concatenate((high_m_obs, high_m_act)),
            shape=(len_m_obs + len_m_act,),
            dtype=self.observation_space.dtype,
        )

    @property
    def composed_action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.action_space.low,
            high=self.action_space.high,
            shape=(self.target_singlestep_act_len,),
            dtype=self.action_space.dtype,
        )
