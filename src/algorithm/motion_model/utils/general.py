# coding=utf-8
import omegaconf


def fetch_selected_hparam_for_tensorboard(cfg: omegaconf.DictConfig) -> str:
    # @formatter:off
    hparam_str = f"""

    overrides:
        motion_model:
            history_len: {cfg.overrides.motion_model.history_len}
            horizon_len: {cfg.overrides.motion_model.horizon_len}

        sampler_rollout:
            trajectory_max_length: {cfg.overrides.sampler_rollout.trajectory_max_length}
            nb_sampling_trials: {cfg.overrides.sampler_rollout.nb_sampling_trials}

        postprocess_replay_buffer:
            collect_trajectories: {cfg.overrides.postprocess_replay_buffer.collect_trajectories} 
            trajectory_max_length: {cfg.overrides.postprocess_replay_buffer.trajectory_max_length} 

        model_batch_size: {cfg.overrides.model_batch_size}
        model_lr: {cfg.overrides.get("model_lr", 1e-4)}
        model_wd: {cfg.overrides.get("model_wd", 5e-5)}
        patience: {cfg.overrides.get("patience", 10)}
        improvement_threshold: {cfg.overrides.get("improvement_threshold", 0.01)}

    dynamics_model:
        _target_: {str(cfg.dynamics_model.get("_target_", None))}
        ensemble_size: {cfg.dynamics_model.get("ensemble_size", None)}
        deterministic: {cfg.dynamics_model.get("deterministic", None)}
        propagation_method: {cfg.dynamics_model.get("propagation_method", None)}
        learn_logvar_bounds: {cfg.dynamics_model.get("learn_logvar_bounds", None)}

        history_len: {cfg.dynamics_model.get("history_len", None)}
        horizon_len: {cfg.dynamics_model.get("horizon_len", None)}
        gamma: {cfg.dynamics_model.get("gamma", None)}
        obs_feature_weights: {str(cfg.dynamics_model.get("obs_feature_weights", None))}
        act_feature_weights: {str(cfg.dynamics_model.get("act_feature_weights", None))}
        singlestep_obs_len: {cfg.dynamics_model.get("singlestep_obs_len", None)}
        singlestep_act_len: {cfg.dynamics_model.get("singlestep_act_len", None)}

        num_layers: {cfg.dynamics_model.get("num_layers", None)}
        hid_size: {cfg.dynamics_model.get("hid_size", None)}
        activation_fn_cfg: {str(cfg.dynamics_model.activation_fn_cfg.get("_target_", None))}

    exploration_policy
        act_space_noise_injection_steer_stdev: {cfg.overrides.exploration_policy.get("act_space_noise_injection_steer_stdev", None)}
        act_space_noise_injection_speed_stdev: {cfg.overrides.exploration_policy.get("act_space_noise_injection_speed_stdev", None)}

    """
    # @formatter:on
    return hparam_str
