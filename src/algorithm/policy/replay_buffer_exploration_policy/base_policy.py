# coding=utf-8
import abc
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
from mbrl.util import ReplayBuffer
from omegaconf import omegaconf

from algorithm.policy.r2s_policy_tools.base_r2s_policy import BaseR2SPolicy


class BaseReplayBufferExplorationPolicy(BaseR2SPolicy):
    _source_replay_buffer: ReplayBuffer
    _dataset_visited_trj_idx: np.ndarray
    _dataset_trj_start_idx: np.ndarray
    _dataset_index: np.ndarray
    _current_uder_epoch: int = 0

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        obs_source_replay_buffer: ReplayBuffer,
        policy_cfg_key: str = "UDER.uder_exploration_policy",
        simulator_cfg_key: str = "environment",
    ):
        """Abstract base class for replay buffer exploration policy

        :param cfg: hydra configuration
        :param obs_source_replay_buffer: the observed source replay buffer
        :param policy_cfg_key: policy related key to fetch in the cfg object
        :param simulator_cfg_key: simulator related key to fetch in the cfg object
        """
        # .... Setup ..............................................................................
        self._rng = np.random.default_rng(cfg.get("seed", None))

        self._source_replay_buffer = deepcopy(obs_source_replay_buffer)
        self._dataset_visited_trj_idx = np.full(len(obs_source_replay_buffer), False)
        self._dataset_index = np.arange(0, len(obs_source_replay_buffer), dtype=int)

        scan_window_len = omegaconf.OmegaConf.select(
            cfg, f"{policy_cfg_key}.scan_window_len", default=1
        )
        scan_window_registered_len = omegaconf.OmegaConf.select(
            cfg, f"{policy_cfg_key}.scan_window_registered_len", default=1
        )

        self._dataset_trj_start_idx = self._dataset_index[
            self._dataset_index % scan_window_registered_len == 0
        ]

        # .... Pre-condition ......................................................................
        assert scan_window_registered_len <= scan_window_len, (
            f"{scan_window_registered_len} !< " f"{scan_window_len}"
        )
        assert scan_window_len % scan_window_registered_len == 0, (
            f"Cfg `{policy_cfg_key}.scan_window_len` must be a multiple of "
            f"`{policy_cfg_key}.scan_window_registered_len` i.e."
            f"{scan_window_len=} % {scan_window_registered_len=} != 0."
        )

        super().__init__(cfg, policy_cfg_key, simulator_cfg_key)

    @property
    def _policy_cfg_keys(self) -> List[str]:
        return [
            "scan_window_len",
            "scan_window_registered_len",
        ]

    @property
    def _sim_cfg_keys(self) -> Optional[List[str]]:
        return None

    def update_policy(
        self,
        obs_source_replay_buffer: ReplayBuffer,
        scan_window_len: Optional[int] = None,
        scan_window_registered_len: Optional[int] = None,
    ):
        # (CRITICAL) ToDo: implement test case
        self._source_replay_buffer = deepcopy(obs_source_replay_buffer)
        self._dataset_visited_trj_idx = np.full(len(obs_source_replay_buffer), False)
        self._dataset_index = np.arange(0, len(obs_source_replay_buffer), dtype=int)

        if scan_window_len is None:
            scan_window_len = omegaconf.OmegaConf.select(
                self.policy_cfg, "scan_window_len", default=1
            )

        if scan_window_registered_len is None:
            scan_window_registered_len = omegaconf.OmegaConf.select(
                self.policy_cfg, f"scan_window_registered_len", default=1
            )

        self._dataset_trj_start_idx = self._dataset_index[
            self._dataset_index % scan_window_registered_len == 0
        ]

        # .... Pre-condition ......................................................................
        assert scan_window_registered_len <= scan_window_len, (
            f"{scan_window_registered_len} !< " f"{scan_window_len}"
        )
        # assert len(obs_source_replay_buffer) % scan_window_len == 0, (
        #     f"Cfg `{self._policy_cfg_keys}.scan_window_len` must be a multiple of the source  "
        #     f"replay buffer size i.e. num_stored={len(obs_source_replay_buffer)} % "
        #     f"{scan_window_len=} != 0."
        # )
        assert scan_window_len % scan_window_registered_len == 0, (
            f"Cfg `{self._policy_cfg_keys}.scan_window_len` must be a multiple of "
            f"`{self._policy_cfg_keys}.scan_window_registered_len` i.e."
            f"{scan_window_len=} % {scan_window_registered_len=} != 0."
        )

        self._current_uder_epoch += 1
        self._init_policy()
        return self

    def act(self, obs_target_replay_buffer: ReplayBuffer) -> Tuple[int, int]:
        """
        Selects a trajectory start and end index from the policy and updates the registred visited
        states array. Garantee to return any `trajectory_start_idx` only once.

        :param obs_target_replay_buffer: The target replay buffer in which to record the selected
         trajectories samples.
        :return: A tuple containing the start index and the adjusted end index of the trajectory.
        :raise AssertionError: If `act` is executed while all trajectory start index where visited.
        """
        trajectory_start_idx = self.policy()

        remaining_space = obs_target_replay_buffer.capacity - obs_target_replay_buffer.num_stored

        # .... Register visited ...................................................................
        registred_trajectory_end_idx = self._enforce_source_replay_buffer_max_idx(
            trajectory_start_idx + min(self.policy_cfg.scan_window_registered_len, remaining_space)
        )
        self._register_visited_trajectory_indexes(
            trajectory_start_idx, registred_trajectory_end_idx
        )

        # .... Set returned window len policy act end index .......................................
        trajectory_end_idx = self._enforce_source_replay_buffer_max_idx(
            trajectory_start_idx + min(self.policy_cfg.scan_window_len, remaining_space)
        )
        return trajectory_start_idx, trajectory_end_idx

    @abc.abstractmethod
    def _init_policy(self) -> None:
        """Policy initialization
        Note: The policy variable accept an arbitrary object type.
        """
        self.policy: Any = None
        return None

    def _enforce_source_replay_buffer_max_idx(self, trajectory_end_idx: int) -> int:
        return min(trajectory_end_idx, len(self._source_replay_buffer))

    def _register_visited_trajectory_indexes(
        self, trajectory_start_idx: int, trajectory_end_idx: int
    ) -> None:
        assert not np.all(
            self._dataset_visited_trj_idx
        ), "All trajectory start index where already visited once!"
        self._dataset_visited_trj_idx[trajectory_start_idx:trajectory_end_idx] = True
        return None

    @property
    def _available_trj_start_idx(self) -> np.ndarray:
        available_trj_idx = self._dataset_index[self._dataset_visited_trj_idx == False]

        available_trj_start_idx = available_trj_idx[
            available_trj_idx % self.policy_cfg.scan_window_registered_len == 0
        ]
        return available_trj_start_idx

    @property
    def available_trj_start_idx_len(self) -> int:
        return len(self._available_trj_start_idx)

    @property
    def registred_visite_size(self) -> int:
        """Return the number of registred visited source replay buffer index.
        Note: that the total trajectory size returned by the `act` method might be larger as it
        depend on `policy_cfg.scan_window_registered_len`.
        """
        return self._dataset_visited_trj_idx[self._dataset_visited_trj_idx == True].size
