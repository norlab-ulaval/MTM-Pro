# coding=utf-8
import os.path
from dataclasses import dataclass
from typing import List, Optional

import omegaconf

from tools.hydra_apps_tools.hydra_utils import get_hydra_original_cwd


@dataclass
class NoiseScaleSchedulerSpec:
    """Specification of the train-time domain randomization noise-scale scheduler.

    Introduced by task RLRP-707 of the Training-Pipeline Domain Randomization `.junie` plan
    (``feature_training_pipeline_domain_randomization_plan_20260610.md``). Drives a scalar
    multiplier applied on top of the per-feature noise scale as a function of the training epoch.
    """

    kind: str = "constant"
    start_scale: float = 1.0
    end_scale: float = 1.0
    warmup_epochs: int = 0
    total_epochs: int = 0
    # ``cyclic_cosine`` wave parameters (RLRP-707): ``cycle_length`` is the base wave length (period)
    # in epochs. ``cycle_length_growth`` makes the period get wider (``> 0``) or narrower (``< 0``)
    # as epochs progress. ``trough_growth`` lifts the wave's lowest point per completed cycle
    # (``0.0`` -> every trough returns to ``0``; ``0.25`` -> the trough scales up by 25% of the
    # current envelope each cycle). The wave peak grows/shrinks via an envelope ramping
    # ``start_scale`` -> ``end_scale`` over ``total_epochs``. Unused by the ``constant``/``linear``/
    # ``ease_in_out_ramp`` kinds.
    cycle_length: int = 0
    cycle_length_growth: float = 0.0
    trough_growth: float = 0.0


@dataclass
class TrainTimeDomainRandomizationSpec:
    """Specification of the train-time domain randomization top layer.

    Introduced by task RLRP-707 of the Training-Pipeline Domain Randomization `.junie` plan
    (``feature_training_pipeline_domain_randomization_plan_20260610.md``). A common input-side
    domain-randomization / data-augmentation scheme applied during fitting (BYOL/TD-MPC2 cited only
    as intuition; this is not a port of either). Disabled by default.
    """

    enable: bool = False
    noise_kind: str = "gaussian"
    per_feature_scale: Optional[List[float]] = None
    #: RLRP-761 S1.7 -- how to interpret :attr:`per_feature_scale`.
    #:
    #: The randomizer perturbs tensors that are ALREADY NORMALIZED, so the
    #: configured magnitude is expressed in normalizer-output units and is
    #: silently invalidated by any change to the normalization contract (risk
    #: ``R-I``: after the S1.5 handler flip the very same number injects ~90x
    #: more relative noise on ``dt``).
    #:
    #: - ``"absolute"`` (**default**, kept for one release so no existing run
    #:   changes behaviour): the value IS the noise std in normalizer-output
    #:   units. Requires re-calibration whenever the contract changes.
    #: - ``"relative_to_normalized_std"``: the value is a DIMENSIONLESS fraction
    #:   of the normalizer's own per-dim output std; the effective noise is
    #:   ``sigma_eff[d] = per_feature_scale[d] * sigma_norm[d]``, which makes DR
    #:   invariant to ``normalizer_type`` by construction (measure ``M7``).
    per_feature_scale_mode: str = "absolute"
    #: RLRP-761 S2.8 -- the per-dim normalization strategy list
    #: (``per_dim_strategy``) that was in effect when :attr:`per_feature_scale`
    #: was calibrated, recorded so a STALE calibration can be DETECTED instead
    #: of silently applied.
    #:
    #: Only meaningful in ``"absolute"`` mode (in
    #: ``"relative_to_normalized_std"`` mode the vector is dimensionless and
    #: cannot go stale). When ``None`` the training-start diagnostic falls back
    #: to flagging every dimension whose resolved strategy is ``"zscore"``,
    #: since those are exactly the dimensions whose scale the S1.5 handler flip
    #: changed (``dt``: ~x90).
    per_feature_scale_calibrated_strategy: Optional[List[str]] = None
    default_scale: float = 0.0
    correlation: float = 0.0
    randomize_target: bool = False
    scheduler: Optional[NoiseScaleSchedulerSpec] = None


@dataclass
class DomainRandomizationDictKeySpec:
    """Randomization specification of F110 parameters. One instance represent a param key,
    its mean value and the randomization standard deviation.

    F110 vehicle parameter keys:

          mu: surface friction coefficient
          C_Sf: Cornering stiffness coefficient, front
          C_Sr: Cornering stiffness coefficient, rear
          lf: Distance from center of gravity to front axle
          lr: Distance from center of gravity to rear axle
          h: Height of center of gravity
          m: Total mass of the vehicle
          I: Moment of inertial of the entire vehicle about the z axis
          s_min: Minimum steering angle constraint
          s_max: Maximum steering angle constraint
          sv_min: Minimum steering velocity constraint
          sv_max: Maximum steering velocity constraint
          v_switch: Switching velocity (velocity at which the acceleration is no longer able to
          create wheel spin)
          a_max: Maximum longitudinal acceleration
          v_min: Minimum longitudinal velocity
          v_max: Maximum longitudinal velocity
          width: width of the vehicle in meters
          length: length of the vehicle in meters

    Note: Param 'value' will be set to environment default if empty
    """

    key: str
    stdev: float
    value: Optional[float] = None


@dataclass
class EnvironmentSpec:
    """Specification of an environment."""

    simulator_cfg_path: Optional[str] = None
    map_path: Optional[str] = None
    wpt_path: Optional[str] = None
    init_x: Optional[float] = None
    init_y: Optional[float] = None
    init_theta: Optional[float] = None

    def from_cfg(self, cfg: omegaconf.DictConfig):
        """Setup dataclass variable via the `simulator` field in a hydra configuration object

        Usage example:

        >>> env_spec = EnvironmentSpec()
        >>> env_spec.from_cfg(cfg)

        :param cfg: an hydra configuration object
        :return: None
        """
        self.map_path = cfg.map_path
        self.wpt_path = cfg.wpt_path
        self.init_x = cfg.init_x
        self.init_y = cfg.init_y
        self.init_theta = cfg.init_theta
        return None

    def from_yaml(self, cfg: omegaconf.DictConfig, simulator_cfg_path: str) -> None:
        """Setup dataclass variable via a `my_simulator_cfg.yaml` file

        Usage example:

        >>> env_spec = EnvironmentSpec()
        >>> sim_cfg_path = 'launcher/configs/simulator/f1tenth_sao_paulo.yaml'
        >>> env_spec.from_yaml(cfg, simulator_cfg_path=sim_cfg_path)

        :param cfg: a hydra configuration object (used for fetching the hydra base cwd directory)
        :param simulator_cfg_path: the local path from `src/` to the `my_simulator_cfg.yaml` file
        :return: None
        """
        self.simulator_cfg_path = simulator_cfg_path
        # hydra_orginal_cwd = get_hydra_original_cwd(cfg)
        simulator_cfg_path = os.path.join(cfg.project_src_root_path, simulator_cfg_path)
        with open(simulator_cfg_path) as file:
            loaded_sim_cfg = omegaconf.OmegaConf.load(file)
            cfg_ = omegaconf.OmegaConf.create(
                f"""
                project_src_root_path: {cfg.project_src_root_path}
                sim_cfg: {loaded_sim_cfg}
                """
            )
            resolved_cfg = omegaconf.OmegaConf.to_yaml(cfg_, resolve=True)
            cfg_ = omegaconf.OmegaConf.create(resolved_cfg)
            self.from_cfg(cfg_.sim_cfg)
        return None


@dataclass
class EnvironmentalConfigRandomizationSpec(EnvironmentSpec):
    """Specification of a randomized environment variation.
    An instance defines one map and its corresponding initial poses and randomization spec.
    """

    init_x_stdev: float = 0.0
    init_y_stdev: float = 0.0
    init_theta_stdev: float = 0.0


@dataclass
class TargetEnvironmentConfigSpec(EnvironmentSpec):
    """Specification of the target environment, the one aimed for deployment."""

    params: Optional[dict] = None
