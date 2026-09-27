# coding=utf-8
from typing import Optional, Tuple, Union

import mbrl.models.one_dim_tr_model
import mbrl.util.replay_buffer

from omegaconf import omegaconf
from torch._dynamo import OptimizedModule as torch_OptimizedModule

from tools.mbrl_lib_tools.models.one_dim_tr_model_v2 import OneDTransitionRewardModelV2
from tools.multistep_tools.data_buffer_processor import MultistepDataBufferProcessorAbstract
from tools.multistep_tools.models import MultiStepMLP
from tools.multistep_tools.utils import (
    compute_multistep_model_in_size,
    compute_multistep_model_out_size,
)
from tools.model_adapter_tools.deployer_adapter import (
    Env2ModelObservationAdapter,
    Model2EnvNextObservationAdapter,
    Model2ModelSymmetricObservationAdapter,
)
from tools.r2s_motion_model_container_tools.utils import (
    DeployerAdapter,
    check_deployer_adapter_legal_type,
)


class R2SMotionModelContainer:
    def __init__(
        self,
        target_env_obs_shape: tuple,
        target_env_next_obs_shape: tuple,
        target_env_act_shape: tuple,
        motion_model_ss_in_size: int,
        motion_model_ss_out_size: int,
        motion_model_ms_in_size: int,
        motion_model_ms_out_size: int,
        deployer_adapter: DeployerAdapter,
        multistep_len: int,
        learned_rewards: bool = False,
        output_window_len: Optional[int] = None,
    ):
        """
        Container for cariyng mbrl-lib multistep dynamic model components from instanciation
        to deployment. Also execute validation logic to make sure input/output size match over
        every stage:

            source environment sampling -> multistep data buffer processor
                -> multistep model training -> test-time rollout -> multistep model training
                -> ... -> target environment model deployer

        Goal: Simplify experimentation workflow by puting real to sim model design spec at the
        top of the experimentation script without having to modify the code low level components.

        Note:
        - Most specification validations are executed at instanciation. Further check can be
          executed:
          - against a replay buffer instance using the `validate_with_replay_buffer` methode;
          - against a multistep data buffer processor instance using the
          `validate_with_ms_data_buffer_processor` methode;
          - against a motion model using `check_motion_model_against_spec` method.
        - Required by AbstractExperienceReplayLearningLoop and by multistep deployers
          MultistepMotionModel[2GymEnv|TestTimeRollout]Deployer.

        :param target_env_obs_shape: the target env obs shape output
        :param target_env_next_obs_shape: the target env obs input shape
        :param target_env_act_shape: the target env act space shape
        :param motion_model_ss_in_size: the mbrl motion model singlestep output size
        :param motion_model_ss_out_size: the mbrl motion model singlestep input size
        :param motion_model_ms_in_size: the mbrl motion model multistep input size
        :param motion_model_ms_out_size: the mbrl motion model multistep output size
        :param deployer_adapter:  model deployement interfaces
        :param multistep_len: the number of step preserve in history buffer (also set the
         horizon buffer under the hood).
        :param learned_rewards: wheter the motion model should learn the reward or not
        :param output_window_len: (RLRP-824) the number of composed obs blocks ``W`` of the
         motion model OUTPUT window. ``None`` (default) => ``multistep_len`` (the legacy
         ``obs*H + act*(H-1)`` output layout, bit-exact). The MS->MS forecast family on an
         asymmetric ``horizon_len > history_len`` window sets ``W = horizon_len``.
        """
        self.target_env_obs_shape = target_env_obs_shape
        self.target_env_next_obs_shape = target_env_next_obs_shape
        self.target_env_act_shape = target_env_act_shape

        self.motion_model_ss_in_size = motion_model_ss_in_size
        self.motion_model_ss_out_size = motion_model_ss_out_size
        self.motion_model_ms_in_size = motion_model_ms_in_size
        self.motion_model_ms_out_size = motion_model_ms_out_size

        # For now, Assume singlestep_obs_len and singlestep_next_obs_len are symetric
        self.singlestep_obs_len = motion_model_ss_out_size
        self.singlestep_act_len = motion_model_ss_in_size - motion_model_ss_out_size

        self.multistep_len = multistep_len
        self.output_window_len = (
            int(multistep_len) if output_window_len is None else int(output_window_len)
        )

        self.learned_rewards = learned_rewards

        # iceboxed: RLRP-154 feat: add obs-to-state and state-to-obs adapter management logic
        self.deployer_adapter = deployer_adapter
        self._dynamics_model = None

        self._sanity_check()

    def validate_with_ms_data_buffer_processor(
        self, ms_data_buffer_processor: MultistepDataBufferProcessorAbstract
    ) -> None:
        # (NICE TO HAVE) ToDo: implement test case (indirectly tested)
        """Validate motion model components against a multistep data buffer processor instance.

        Note: `ms_data_buffer_processor` doesn't need to have been used yet.

        :param ms_data_buffer_processor: a multistep data buffer processor instance
        :exception AssertionError: raised if `ms_data_buffer_processor` doesn't match
            the r2s motion model container specification.
        """
        try:
            assert isinstance(ms_data_buffer_processor, MultistepDataBufferProcessorAbstract)
            assert ms_data_buffer_processor.history_len == self.multistep_len
            assert ms_data_buffer_processor.target_singlestep_obs_len == self.singlestep_obs_len
            assert ms_data_buffer_processor.target_singlestep_act_len == self.singlestep_act_len

            # .... Check in size ..............................................................
            ms_data_processor_required_model_in_size = (
                ms_data_buffer_processor.get_model_input_size_requirement_for_compose_obs()
            )
            assert ms_data_processor_required_model_in_size == self.motion_model_ms_in_size, (
                "Spec does not match `ms_data_buffer_processor` expected input feed: "
                f"{ms_data_processor_required_model_in_size} != {self.motion_model_ms_in_size}"
            )

            # .... Check out size .............................................................
            ms_data_processor_required_model_out_size = (
                ms_data_buffer_processor.get_model_out_size_requirement_for_compose_obs()
            )
            assert ms_data_processor_required_model_out_size == self.motion_model_ms_out_size, (
                "Spec does not match `ms_data_buffer_processor` expected output feed: "
                f"{ms_data_processor_required_model_out_size} != "
                f"{self.motion_model_ms_out_size}"
            )

        except AssertionError as e:
            raise AssertionError(f"{self.__class__.__name__} components misspecification: {e}")

        return None

    def validate_with_replay_buffer(self, replay_buffer: mbrl.util.ReplayBuffer) -> None:
        """Validate motion model components against a mbrl-lib replay buffer instance.
        Usefull for singlestep model which don't require usage of multistep data buffer processor.
        For multistep model case, assume `replay_buffer` is already converted to mulsistep obs/act.

        :param replay_buffer: A replay buffer to validate spec
        :exception AssertionError: raised if `replay_buffer` doesn't match the r2s motion model
            container specification.
        """
        try:
            assert isinstance(replay_buffer, mbrl.util.ReplayBuffer)
            buffer_obs_len = replay_buffer.obs_shape[0]
            buffer_act_len = replay_buffer.action_shape[0]
            buffer_next_obs_len = replay_buffer.obs_shape[0]

            # .... Check in size ..............................................................
            replay_buffer_feed_in = buffer_obs_len + buffer_act_len
            assert replay_buffer_feed_in == self.motion_model_ms_in_size, (
                "Spec does not match replay_buffer expected input feed: "
                f"{replay_buffer_feed_in} != {self.motion_model_ms_in_size}"
            )

            # .... Check out size .............................................................
            replay_buffer_feed_out = buffer_next_obs_len + int(self.learned_rewards)
            assert replay_buffer_feed_out == self.motion_model_ms_out_size, (
                "Spec does not match replay_buffer expected output feed: "
                f"{replay_buffer_feed_out} != {self.motion_model_ms_out_size}"
            )

        except AssertionError as e:
            raise AssertionError(f"{self.__class__.__name__} components misspecification: {e}")

        return None

    def validate_with_window_dataset(self, window_dataset) -> None:
        """Validate motion model components against a lazy ``MultistepWindowDataset``
        (``pipeline.data_manager: dataloader``, RLRP-824 Step 6a).

        The dataset composes rows in the replay-buffer layout, so the checks mirror
        :meth:`validate_with_replay_buffer` (``obs + act`` columns == ``in_size``, ``next_obs``
        columns == ``out_size``) plus the window lengths (``history_len == multistep_len``,
        ``output_window_len == output_window_len``) and the single-step feature sizes.

        :param window_dataset: a ``MultistepWindowDataset`` (full or split subset)
        :exception AssertionError: raised if the dataset doesn't match the container spec.
        """
        from tools.multistep_tools.window_dataset.multistep_window_dataset import (
            MultistepWindowDataset,
        )

        try:
            assert isinstance(window_dataset, MultistepWindowDataset)
            assert window_dataset.history_len == self.multistep_len, (
                f"history_len {window_dataset.history_len} != multistep_len {self.multistep_len}"
            )
            assert window_dataset.output_window_len == self.output_window_len, (
                f"output_window_len {window_dataset.output_window_len} != "
                f"{self.output_window_len}"
            )
            assert window_dataset.obs_dim == self.singlestep_obs_len, (
                f"obs_dim {window_dataset.obs_dim} != singlestep_obs_len {self.singlestep_obs_len}"
            )
            assert window_dataset.act_dim == self.singlestep_act_len, (
                f"act_dim {window_dataset.act_dim} != singlestep_act_len {self.singlestep_act_len}"
            )

            # .... Check in size ..............................................................
            dataset_feed_in = window_dataset.obs_width + window_dataset.act_dim
            assert dataset_feed_in == self.motion_model_ms_in_size, (
                "Spec does not match window dataset expected input feed: "
                f"{dataset_feed_in} != {self.motion_model_ms_in_size}"
            )

            # .... Check out size .............................................................
            dataset_feed_out = window_dataset.out_size + int(self.learned_rewards)
            assert dataset_feed_out == self.motion_model_ms_out_size, (
                "Spec does not match window dataset expected output feed: "
                f"{dataset_feed_out} != {self.motion_model_ms_out_size}"
            )

        except AssertionError as e:
            raise AssertionError(f"{self.__class__.__name__} components misspecification: {e}")

        return None

    @property
    def dynamics_model(
        self,
    ) -> Union[OneDTransitionRewardModelV2, mbrl.models.OneDTransitionRewardModel]:
        if self._dynamics_model is None:
            raise AttributeError(
                f"dynamics_model is not set yet. Use `set_dynamics_model(<my-model>)`"
            )
        else:
            return self._dynamics_model

    def set_dynamics_model(
        self,
        motion_model: Union[OneDTransitionRewardModelV2, mbrl.models.OneDTransitionRewardModel],
    ) -> None:
        """Update the container stored model with new `motion_model` if all cheks against
        motion model container specification passes.

        :param motion_model: the new mbrl motion model
        """
        self.check_motion_model_against_spec(motion_model)
        # assert (
        #     motion_model.target_is_delta is False # ToDo: experiment <--
        # ), "cfg for dynamics_model.target_is_delta should be False"

        self._dynamics_model = motion_model
        return None

    def check_motion_model_against_spec(
        self,
        motion_model: Union[OneDTransitionRewardModelV2, mbrl.models.OneDTransitionRewardModel],
    ) -> None:
        if isinstance(motion_model, torch_OptimizedModule):
            motion_model = motion_model._modules["_orig_mod"]

        assert isinstance(
            motion_model,
            (OneDTransitionRewardModelV2, mbrl.models.OneDTransitionRewardModel),
        )

        if not isinstance(motion_model.model, MultiStepMLP):
            assert self.multistep_len == 1, (
                "Only model subclassing of MultiStepMLP can handle multistep observation. "
                f"Curent spec: {type(motion_model)}, multistep_len={self.multistep_len}"
            )

        assert motion_model.model.in_size == self.motion_model_ms_in_size, (
            "dynamics_model.in_size does not match spec: "
            f"{motion_model.model.in_size} != {self.motion_model_ms_in_size}"
        )
        assert motion_model.model.out_size == self.motion_model_ms_out_size, (
            "dynamics_model.out_size does not match spec: "
            f"{motion_model.model.out_size} != {self.motion_model_ms_out_size}"
        )
        return None

    def _sanity_check(self) -> None:
        try:
            self._check_arg_type_ok()
            self._validate_spec()
            self._check_target_env2model_obs_adapter()
            self._check_model2target_env_next_obs_adapter()
            self._check_model2model_obs_adapter()

        except AssertionError as e:
            raise AssertionError(f"{self.__class__.__name__} components misspecification: {e}")

        return None

    def _check_arg_type_ok(self) -> None:
        """Goal: flag wrong attribute type early to mitigate hydra config misspecification case."""

        # .... attribute type check: tuple ........................................................
        for each in [
            self.target_env_obs_shape,
            self.target_env_next_obs_shape,
            self.target_env_act_shape,
        ]:
            if isinstance(each, (list, omegaconf.ListConfig)):
                each = tuple(each)
            assert isinstance(each, tuple), f"Wrong arg type {type(each)} != tuple"

        # .... attribute type check: int ..........................................................
        for each in [
            self.target_env_obs_shape[0],
            self.target_env_next_obs_shape[0],
            self.target_env_act_shape[0],
            self.motion_model_ss_in_size,
            self.motion_model_ss_out_size,
            self.motion_model_ms_in_size,
            self.motion_model_ms_out_size,
            self.multistep_len,
        ]:
            assert isinstance(each, int), f"Wrong arg type {each} != int"

        # .... attribute type check: deployer adapter member ......................................
        assert isinstance(self.deployer_adapter, DeployerAdapter)

        check_deployer_adapter_legal_type(
            self.deployer_adapter.target_env_to_model_ss_in_obs, Env2ModelObservationAdapter
        )

        check_deployer_adapter_legal_type(
            self.deployer_adapter.model_ss_out_to_target_env_next_obs,
            Model2EnvNextObservationAdapter,
        )

        check_deployer_adapter_legal_type(
            self.deployer_adapter.model_to_model_ss_obs, Model2ModelSymmetricObservationAdapter
        )

        return None

    def _validate_spec(self) -> None:
        target_in = compute_multistep_model_in_size(
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            multistep_len=self.multistep_len,
        )
        assert self.motion_model_ms_in_size == target_in, (
            "Expected motion_model_ms_in_size does not match spec:"
            f" {self.motion_model_ms_in_size} != {target_in}"
        )

        # RLRP-824: the OUTPUT window is ``W = output_window_len`` obs blocks (``== multistep_len``
        # for every legacy spec).
        target_out = compute_multistep_model_out_size(
            singlestep_obs_len=self.singlestep_obs_len,
            singlestep_act_len=self.singlestep_act_len,
            multistep_len=self.output_window_len,
            learned_reward=self.learned_rewards,
        )

        assert self.motion_model_ms_out_size == target_out, (
            "Expected motion_model_ms_out_size does not match spec: "
            f"{self.motion_model_ms_out_size} != {target_out}"
        )
        return None

    def _check_target_env2model_obs_adapter(self) -> None:
        if isinstance(
            self.deployer_adapter.target_env_to_model_ss_in_obs, Env2ModelObservationAdapter
        ):
            assert (
                self.deployer_adapter.target_env_to_model_ss_in_obs.array_in_len
                == self.target_env_obs_shape[0]
            ), (
                f"{self.deployer_adapter.target_env_to_model_ss_in_obs.array_in_len=} != "
                f"{self.target_env_obs_shape[0]=}"
            )
            assert (
                self.deployer_adapter.target_env_to_model_ss_in_obs.array_out_len
                == self.singlestep_obs_len
            ), (
                f"{self.deployer_adapter.target_env_to_model_ss_in_obs.array_out_len=}"
                f" != {self.singlestep_obs_len=}"
            )
        elif callable(self.deployer_adapter.target_env_to_model_ss_in_obs):
            # R2SMotionModelContainer assume you know what your doing. Example use case: testing
            pass

        return None

    def _check_model2target_env_next_obs_adapter(self) -> None:
        if isinstance(
            self.deployer_adapter.model_ss_out_to_target_env_next_obs,
            Model2EnvNextObservationAdapter,
        ):
            assert (
                self.motion_model_ss_out_size
                == self.deployer_adapter.model_ss_out_to_target_env_next_obs.array_in_len
            ), (
                f"{self.motion_model_ss_out_size=} "
                f"!= {self.deployer_adapter.model_ss_out_to_target_env_next_obs.array_in_len=}"
            )
            assert (
                self.deployer_adapter.model_ss_out_to_target_env_next_obs.array_out_len
                == self.target_env_next_obs_shape[0]
            ), (
                f"{self.deployer_adapter.model_ss_out_to_target_env_next_obs.array_out_len=} "
                f"!= {self.target_env_next_obs_shape[0]=}"
            )
        elif callable(self.deployer_adapter.model_ss_out_to_target_env_next_obs):
            # R2SMotionModelContainer assume you know what your doing. Example use case: testing
            pass

        return None

    def _check_model2model_obs_adapter(self) -> None:
        if isinstance(
            self.deployer_adapter.model_to_model_ss_obs,
            Model2EnvNextObservationAdapter,
        ):
            assert (
                self.motion_model_ss_out_size
                == self.deployer_adapter.model_to_model_ss_obs.array_in_len
            ), (
                f"{self.motion_model_ss_out_size=} "
                f"!= {self.deployer_adapter.model_to_model_ss_obs.array_in_len=}"
            )
            assert (
                self.deployer_adapter.model_to_model_ss_obs.array_out_len
                == self.motion_model_ss_in_size
            ), (
                f"{self.deployer_adapter.model_to_model_ss_obs.array_out_len=} "
                f"!= {self.motion_model_ss_in_size=}"
            )
        elif callable(self.deployer_adapter.model_to_model_ss_obs):
            # R2SMotionModelContainer assume you know what your doing. Example use case: testing
            pass

        return None

    @property
    def motion_model_name(self) -> Tuple[str, str]:
        if self._dynamics_model is not None:
            one_d_tr_model = self.dynamics_model
            if isinstance(self.dynamics_model, torch_OptimizedModule):
                one_d_tr_model = self.dynamics_model._modules["_orig_mod"]

            one_d_tr_model_name = one_d_tr_model.__class__.__name__
            motion_model_name = one_d_tr_model.model.__class__.__name__
            return one_d_tr_model_name, motion_model_name
        else:
            return "Model isn't set yet", "Model isn't set yet"

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
        if self._dynamics_model is not None:
            repr_str += f"{m_sp}oneDtrajectory model type: {self.motion_model_name[0]}"
            repr_str += f"{m_sp} ↳ motion model type: {self.motion_model_name[1]}"
        repr_str += f"{m_sp})"
        return repr_str
