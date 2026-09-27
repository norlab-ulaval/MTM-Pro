# coding=utf-8
from algorithm.policy.replay_buffer_exploration_policy.base_policy import \
    BaseReplayBufferExplorationPolicy


class RandomReplayBufferExplorationPolicy(BaseReplayBufferExplorationPolicy):
    """Random replay buffer exploration policy"""

    def _init_policy(self) -> None:
        def random_replay_buffer_index() -> int:
            """Selects a trajectory start and end index randomly and updates visited states.
            Garantee to return any `trajectory_start_idx` only once
            """
            return int(self._rng.choice(self._available_trj_start_idx))

        self.policy = random_replay_buffer_index
        return None
