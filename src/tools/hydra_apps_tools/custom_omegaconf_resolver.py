# coding=utf-8
from tools.console_tools.message import (
    consol_msg_f1tenth_vaul_main_one_liner,
    consol_msg_universal_one_liner,
)


def num_steps_summation_resolver_quickhack(cfg) -> None:
    # ToDo: RLRP-191 refactor: rename variable num_steps to sampling_epoch_num_steps
    """
    Note: Update hydra cfg in place

    :param cfg:
    """
    trajectory_max_length = cfg.overrides.sampler_rollout.get("trajectory_max_length", 1)
    nb_sampling_trials = cfg.overrides.sampler_rollout.get("nb_sampling_trials", 1)

    cfg.overrides.num_steps = trajectory_max_length * nb_sampling_trials
    return None


def dataset_size_summation_resolver_quickhack(cfg) -> None:
    # ToDo: RLRP-192 refactor: re-think spaghetti style cfg related to dataset_size
    """
    Note: Update hydra cfg in place

    :param cfg:
    """
    try:
        dataset_size = cfg.overrides.sampler_rollout.get("dataset_size", None)

        if dataset_size is None:
            trajectory_max_length = cfg.overrides.sampler_rollout.get("trajectory_max_length", 1)
            nb_sampling_trials = cfg.overrides.sampler_rollout.get("nb_sampling_trials", 1)
            domain_randomizationr_max_iteration = cfg.overrides.get(
                "domain_randomisation.max_iteration", 1
            )

            dataset_size = (
                trajectory_max_length * nb_sampling_trials * domain_randomizationr_max_iteration
            )

            exploration_policy_cfg = cfg.overrides.get("exploration_policy", None)
            if exploration_policy_cfg:
                lookahead_distance_matrix = exploration_policy_cfg.get(
                    "lookahead_distance_matrix", [0.0]
                )
                speed_gain_matrix = exploration_policy_cfg.get("speed_gain_matrix", [0.0])
                dataset_size *= len(lookahead_distance_matrix) * len(speed_gain_matrix)

        #     consol_msg_universal_one_liner(f"Set replaybuffer initial size to {dataset_size}")
        # else:
        #     consol_msg_universal_one_liner(
        #         "Set replaybuffer initial size using overrides.sampler_rollout.dataset_size="
        #         f"{dataset_size}"
        #     )

        # Note(From mbrl replay buffer doc): "Specifying replay buffer size directly takes
        # precedence over number of steps."
        cfg.algorithm.dataset_size = dataset_size
    except AttributeError as e:
        raise KeyError("hydra config missing key 'overrides.sampler_rollout'")

    return None
