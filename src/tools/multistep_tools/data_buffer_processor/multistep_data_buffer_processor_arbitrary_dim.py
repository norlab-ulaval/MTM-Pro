# coding=utf-8
from typing import Callable, Optional, Type, Union

import gymnasium as gym
import gymnasium
import numpy as np

from tools.multistep_tools.data_buffer_processor import (
    MultistepDataBufferProcessorAbstract,
)


class MultistepDataBufferProcessorArbitraryDimension(
    MultistepDataBufferProcessorAbstract
):
    observation_space: gym.spaces.Box
    action_space: gym.spaces.Box

    def __init__(
        self,
        history_len: int,
        obs_dim: int,
        act_dim: int,
        obs_dtype: Type,
        act_dtype: Type,
        horizon_len: Union[int, float] = 1,
        obs_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, obs: obs,
        obs_buffer_append_callback: Optional[
            Callable
        ] = lambda ms_buffer, next_obs: next_obs,
        act_buffer_reset_callback: Optional[Callable] = lambda ms_buffer, act=None: act,
        act_buffer_append_callback: Optional[Callable] = lambda ms_buffer, act: act,
        consol_log: bool = True,
    ):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=obs_dtype,
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(act_dim,),
            dtype=act_dtype,
        )

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
