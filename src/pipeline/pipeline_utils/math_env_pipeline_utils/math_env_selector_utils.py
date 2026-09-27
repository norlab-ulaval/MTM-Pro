# coding=utf-8
from functools import partial

import omegaconf

from math_gymnasium.dynamical_systems import *


def math_environment_selector(cfg: omegaconf.DictConfig, env_init_cfg) -> partial:
    if "lorenz" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_lorenz_attractor_partial_derivative,
            **{
                "s": env_init_cfg.param.s,
                "r": env_init_cfg.param.r,
                "b": env_init_cfg.param.b,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "rossler" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_rossler_attractor_partial_derivative,
            **{
                "a": env_init_cfg.param.a,
                "b": env_init_cfg.param.b,
                "c": env_init_cfg.param.c,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "aizawa" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_aizawa_attractor_partial_derivative,
            **{
                "a": env_init_cfg.param.a,
                "b": env_init_cfg.param.b,
                "c": env_init_cfg.param.c,
                "d": env_init_cfg.param.d,
                "e": env_init_cfg.param.e,
                "f": env_init_cfg.param.f,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "simple_limit_cycle_system_van_der_pol" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_simple_limit_cycle_partial_derivative,
            **{
                "mu": env_init_cfg.param.mu,
                "omega": env_init_cfg.param.omega,
                "alpha": env_init_cfg.param.alpha,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "linear_spiral" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_linear_spiral_partial_derivative,
            **{
                "a": env_init_cfg.param.a,
                "omega": env_init_cfg.param.omega,
                "c": env_init_cfg.param.c,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "damped_harmonic_oscillator" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_damped_oscillator_partial_derivative,
            **{
                "gamma": env_init_cfg.param.gamma,
                "omega0": env_init_cfg.param.omega0,
                "vz": env_init_cfg.param.vz,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "linear_debug_system" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_linear_debug_system,
            **{
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "wavy_projection_system" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_wavy_projection_partial_derivative,
            **{
                "vx": env_init_cfg.param.vx,
                "vy": env_init_cfg.param.vy,
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    elif "sombrero_projection_system" in cfg.environment.name:
        state_space_3d_fct = partial(
            rollout_sombrero_projection_partial_derivative,
            **{
                "vx": env_init_cfg.param.vx,
                "vy": env_init_cfg.param.vy,
                "A": env_init_cfg.param.get("A", 1.0),
                "sigma": env_init_cfg.param.get("sigma", 5.0),
                "k": env_init_cfg.param.get("k", 2.0),
                "initiale_coordinates": (
                    tuple(
                        omegaconf.OmegaConf.to_object(env_init_cfg.initiale_coordinates)
                    )
                ),
            },
        )
    else:
        raise NotImplementedError(
            f"Enviroment {cfg.environment.data} not supported"
        )

        # .... Create learning environment ........................................................
    return state_space_3d_fct
