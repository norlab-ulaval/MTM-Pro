# coding=utf-8
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from tools.multistep_tools.deployer.abstract import (
    AbstractMultistepMotionModelDeployer,
    CUDA_STREAM,
    use_cuda_stream,
)
from tools.r2s_motion_model_container_tools.container import R2SMotionModelContainer


class MultistepMotionModel2GymEnvDeployer(AbstractMultistepMotionModelDeployer):
    def __init__(
        self,
        motion_model_container: R2SMotionModelContainer,
        next_state_deterministic_selection: bool = True,
        consol_log: bool = True,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(
            motion_model_container,
            next_state_deterministic_selection,
            consol_log,
            rng,
        )

    # @use_cuda_stream(CUDA_STREAM)
    def predict_next_state(
        self, env_state: Union[torch.Tensor, np.ndarray], action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Predict next single-step environment state.

        Handle the multistep buffers, the env-2-model adapter and the model-2-target-env adapter.

        :param env_state: The current environment state at timestep t.
        :param action: The action executed at timestep t.
        :return: The single-step next state prediction array, predicted reward (if learned)
            and the mbrl-model state dictionary.
        """
        state_subset = self._adapter.target_env_to_model_ss_in_obs(env_state)

        with torch.inference_mode():
            self.one_d_tr_model.eval()

            # Torch-first: convert numpy inputs to tensors at the gym-env boundary
            if isinstance(state_subset, np.ndarray):
                state_subset = self._ndarray_to_tensor(state_subset)
            if isinstance(action, np.ndarray):
                action = self._ndarray_to_tensor(action)

            self._multistep_state_buffer.append(state_subset)
            self._multistep_act_buffer.append(action)

            multistep_state = self._multistep_state_buffer.get_buffer()
            multistep_act = self._multistep_act_buffer.get_buffer()

            (
                next_state_pred,
                pred_rew,
                _,
                model_state_info,
            ) = self.one_d_tr_model.sample(
                act=multistep_act,
                model_state={
                    "obs": multistep_state,
                    "propagation_indices": None,
                },
                deterministic=self.next_state_deterministic_selection,
                rng=self.mbrl_rng,
            )

            # Convert to numpy at the gym-env return boundary
            next_state_pred_np = next_state_pred.detach().cpu().numpy()

            (
                next_state_pred_array,
                model_state_info,
            ) = self._adapter.model_ss_out_to_target_env_next_obs(
                next_state_pred_np.ravel(), info=model_state_info, last_env_state=env_state
            )

            return (
                next_state_pred_array,
                pred_rew,
                model_state_info,
            )
