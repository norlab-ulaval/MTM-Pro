# coding=utf-8
import getpass
import os
import shutil
import sys
from datetime import datetime
from typing import Any, AnyStr, Optional, Tuple

import hydra
import omegaconf
import optuna
import torch
from hydra import TaskFunction
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback
from omegaconf import DictConfig

from tools.dna_dev_tools.dn_pytest_tools import is_pytest_run
from tools.dna_dev_tools.dn_container_tools import (
    fetch_dna_container_root_path_via_env_var,
    is_dockerized_AnonLab_env_var_set, is_run_in_DN_arm64_jetson_architecture,
)
from tools.hydra_apps_tools.custom_omegaconf_resolver import (
    dataset_size_summation_resolver_quickhack,
    num_steps_summation_resolver_quickhack,
)

from tools.hydra_apps_tools.hydra_utils import (
    fetch_project_root_path_via_hydra,
    get_hydra_experiment_cwd,
    is_hydra_multirun,
    is_hydra_optuna_run,
    is_hydra_sweeper_multiprocess_run,
)
from tools.hydra_apps_tools.optuna_pruning import (
    OptunaTrialHandle,
    acquire_optuna_trial,
    mark_trial_pruned,
)
# Side-effect import: registers `R2S2RTPESamplerConfig` (X.5) into
# Hydra's ConfigStore at module import time so launcher YAMLs can select
# `override hydra/sweeper/sampler: tpe_r2s2r` and use the
# `R2S2RJournalAwareOptunaSweeper` (X.6).
from tools.hydra_apps_tools import r2s2r_optuna_sweeper as _r2s2r_optuna_sweeper  # noqa: F401
from tools.console_tools.console_recording import ConsoleRecorder
from tools.console_tools.format import ConsoleFormat
from tools.console_tools.message import (
    consol_msg_draw_terminal_wide_line,
    consol_msg_universal,
    consol_msg_universal_one_liner,
)
from tools.console_tools.terminal_splash import show_splash_screen_in_console


def configure_cuda_float32_high_precision(enable: bool = True) -> None:
    """Force maximum-precision float32 computation on CUDA (RLRC precision hardening).
    Set `enable=False` to allow TF32 operation (i.e., when speed mather more than precision).

    Relevant for the supported ``cuda`` backends (Valeria server GPUs and the
    Jetson-AGX-Orin, both Ampere). When a run falls back to ``float32`` (i.e. not
    ``model_use_double_precision``), TF32 silently truncates matmul mantissas to
    ~10 bits; disabling it (and selecting the ``"highest"`` matmul precision) keeps
    full float32 precision. No effect under ``float64`` and a harmless no-op on
    ``cpu`` (the local DNA container).

    Robust host-capability gate: the TF32 knobs are CUDA-only, so this is a
    silent no-op when ``torch.cuda.is_available()`` is False (the CPU-only DNA
    container).

    Note: ``torch.backends.cuda.matmul.allow_tf32`` and ``torch.set_float32_matmul_precision``
    drive the SAME underlying switch, so the two are always written together here; this is the
    ONLY writer of the TF32 state in the codebase (RLRP-824: the short-lived
    ``training_common.cuda_backend.allow_tf32`` duplicate was removed in favour of
    ``torch_backend.cuda_high_precision_float32``).
    """
    if not torch.cuda.is_available():
        # No-op on CPU-only hosts (local M3 DNA Docker container). TF32 is a
        # CUDA-only concern; nothing to harden here.
        consol_msg_universal_one_liner(
            "High numerical precision requested but no CUDA device available "
            "(TF32 is CUDA-only); nothing to do on this host.",
            caller_name="apply_torch_backend_cfg",
        )
        return None

    if enable:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False # Note: Will be deprecated. Use `torch.backends.cudnn.conv.fp32_precision = "ieee"` to disable in future torch release
        # "highest" => true float32 matmul (no TF32 / bf16 internal rounding).
        torch.set_float32_matmul_precision("highest")

        consol_msg_universal_one_liner(
            "Configured torch for high numerical precision (TF32 disabled, "
            "float32 matmul precision='highest') on CUDA device "
            f"'{torch.cuda.get_device_name(0)}'",
            caller_name="apply_torch_backend_cfg",
        )
    else:
        torch.backends.cuda.matmul.allow_tf32 = True # Note: Will be deprecated. Use `torch.backends.cuda.matmul.fp32_precision = "tf32"` to enable in future torch release
        torch.backends.cudnn.allow_tf32 = True
        # "medium" => TF32 matmul (TF32 / bf16 internal rounding).
        torch.set_float32_matmul_precision("medium")

        consol_msg_universal_one_liner(
            "Configured torch for faster float32 computation at the expense of numerical "
            "precision (TF32 enabled, float32 matmul precision='medium') on CUDA device "
            f"'{torch.cuda.get_device_name(0)}'",
            caller_name="apply_torch_backend_cfg",
        )
    return None


def apply_torch_backend_cfg(cfg: DictConfig) -> None:
    """Apply opt-in torch backend flags from a RLRC Hydra cfg.

    Permanent torch-backend bridge and the SINGLE writer of the process-global
    ``torch.backends`` state (RLRP-824 consolidation, 2026-09-16): every knob lives under
    ``cfg.torch_backend.*`` and is applied exactly once, in ``R2S2RPipelineHydraApp.setup()``,
    before any CUDA kernel runs. Pipeline / trainer setup code MUST NOT write
    ``torch.backends`` (a second writer at trainer-setup time silently reverted
    ``cudnn.benchmark`` and fought over TF32 -- see ``A2G_CHANGELOG.md``).

    Introduced by action ``A3a`` (stage 2, Batch S2-2) of the RLRC Training Speed &
    Efficiency `.junie` plan (``performance_training_speed_efficiency_plan_20260421.md``).
    Silently no-op on non-CUDA devices and when the whole cfg block is absent (legacy /
    unit-test behaviour).

    Exposes (applied in this order, precision first):
      - ``torch_backend.cuda_high_precision_float32`` (default ``true``) →
        ``configure_cuda_float32_high_precision(enable)``: TF32 off + ``"highest"`` matmul
        precision when true; TF32 on + ``"medium"`` when false (the A100 speed opt-in).
      - ``torch_backend.cudnn_benchmark`` (default ``false``) → ``torch.backends.cudnn.benchmark``
        (CUDA-gated; only written when true, so an absent / false key leaves the state untouched).
    """
    # R3 — Observability: log torch intra-op thread count so oversubscription
    # relative to `OMP_NUM_THREADS` / container CPU budget is visible at startup.
    # Introduced by action F-new.3 of the RLRC Training Speed & Efficiency `.junie` plan
    # (performance_training_speed_efficiency_plan_20260421.md). Observation-only —
    # no `torch.set_num_threads(...)` call here, so behaviour is bit-exact.
    _num_threads = torch.get_num_threads()
    _omp_env = os.environ.get("OMP_NUM_THREADS", "unset")
    consol_msg_universal_one_liner(
        f"torch.get_num_threads()={_num_threads} "
        f"(OMP_NUM_THREADS={_omp_env}); "
        f"if torch threads > OMP cap, oversubscription is likely "
        f"(see report_container_capabilities_m3_orin_20260422.md R3).",
        caller_name="apply_torch_backend_cfg",
    )

    torch_backend_cfg = cfg.get("torch_backend", None)
    if torch_backend_cfg is None:
        return None

    # RLRC precision hardening (Gate D). Always applied (both branches write) so the TF32 state
    # is fully determined by this key; CUDA-gated inside the helper.
    configure_cuda_float32_high_precision(
        enable=bool(torch_backend_cfg.get("cuda_high_precision_float32", True))
    )

    if bool(torch_backend_cfg.get("cudnn_benchmark", False)):
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            consol_msg_universal_one_liner(
                "A3a: torch.backends.cudnn.benchmark=True (CUDA detected)",
                caller_name="apply_torch_backend_cfg",
            )
        else:
            consol_msg_universal_one_liner(
                "A3a: cfg.torch_backend.cudnn_benchmark=true ignored "
                "(no CUDA device available)",
                caller_name="apply_torch_backend_cfg",
            )

    if torch.cuda.is_available():
        # One consolidated line with the FINAL state so a run log never shows two contradictory
        # "configured torch ..." messages.
        consol_msg_universal_one_liner(
            "torch.backends final state: "
            f"cuda.matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}, "
            f"float32_matmul_precision='{torch.get_float32_matmul_precision()}', "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}",
            caller_name="apply_torch_backend_cfg",
        )
    return None


class R2S2RHydraStartupInfoCallback(Callback):
    """Custom Hydra callback for managing startup info

    Note: instanciated via hydra cfg files `global_config.yaml`
    """

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        if config.startup_info:
            self._show_user_info(config)

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        # self._set_user_info(config)
        if config.startup_info:
            self._show_user_info(config)

    @staticmethod
    def _show_user_info(config: DictConfig) -> None:

        set_core_hydra_missing_keys(config)

        if not omegaconf.OmegaConf.is_missing(config, key="orig_cwd"):
            _orig_cwd = config.orig_cwd
        else:
            _orig_cwd = "not set"

        consol_msg_universal_one_liner(
            "Config:\n"
            f"   - python_version: {config.python_version}\n"
            "   - host_user\n"
            f"       - name: {config.host_user.name}\n"
            f"       - home: {config.host_user.home}\n"
            f"   - device: {config.device}\n"
            "   - IDE\n"
            f"       - pycharm: {config.IDE.pycharm}\n"
            f"       - ide_remote_run: {config.IDE.ide_remote_run}\n"
            f"   - orig_cwd: {_orig_cwd}\n"
            f"\n",
            caller_name='R2S2RHydraStartupInfoCallback'
        )
        return None


class R2S2RConsoleHydraCallback(Callback):
    """Custom Hydra callback for managing run/multirun console feedback

    Note: instanciated via hydra cfg files `global_config.yaml`
    """

    multirun_experiment_path: str = None
    experiment_path: str = None
    multirun_experiment_dir: str
    experiment_date_dir: str
    experiment_time: str
    _is_multirun: bool = False

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        show_splash_screen_in_console()
        return None

    def on_multirun_start(self, config: DictConfig, **kwargs: Any) -> None:
        self._is_multirun = True
        show_splash_screen_in_console()
        msg = consol_msg_universal_one_liner(
            "Starting multirun experiment "
            f"at {datetime.now().isoformat(sep=' ', timespec='minutes')}\n",
            print_it=False,
        )
        msg += self._set_optuna_dashboard_command_msg(config)
        print(msg)

        return None

    def on_job_start(
        self, config: DictConfig, *, task_function: TaskFunction, **kwargs: Any
    ) -> None:
        if config.get("orig_cwd", None) is None:
            config.orig_cwd = hydra.utils.get_original_cwd()

        if self._is_multirun:
            self._fetch_information_for_multirun_console_print(config)
        else:
            self._fetch_information_for_singlerun_console_print(config)

        consol_msg_universal(
            "Starting experiment "
            f"{ConsoleFormat.MSG_EMPH_FORMAT}"
            f"{self._get_experiment_id()}"
            f"{ConsoleFormat.MSG_END_FORMAT} "
            f"at {datetime.now().isoformat(sep=' ', timespec='minutes')}\n",
            space_before=False,
            space_after=False,
        )

        # .... Report hydra experiment overrides ..................................................
        hydra_conf = HydraConfig.get()
        hydra_job_task_overrides = omegaconf.OmegaConf.to_yaml(
            hydra_conf.overrides.task
        )
        consol_msg_universal_one_liner(
            f"Hydra overrides.task:\n{hydra_job_task_overrides}"
        )

        super().on_job_start(config, task_function=task_function, **kwargs)
        return None

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        consol_msg_universal(
            "Experiment "
            f"{self._get_experiment_id()}"
            f"{ConsoleFormat.MSG_EMPH_FORMAT}"
            f"{ConsoleFormat.MSG_DONE_FORMAT}"
            f" DONE "
            f"{ConsoleFormat.MSG_END_FORMAT}"
            f"at {datetime.now().isoformat(sep=' ', timespec='minutes')}",
            space_before=True,
            space_after=False,
        )

        if not self._is_multirun:
            consol_msg_draw_terminal_wide_line(
                char="\\", space_before=False, space_after=False
            )
            consol_msg_draw_terminal_wide_line(
                char="/", space_before=False, space_after=False
            )
        return None

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        show_splash_screen_in_console()
        consol_msg_universal_one_liner(
            "Experimental data for run "
            f"{ConsoleFormat.MSG_EMPH_FORMAT}"
            f"{os.path.join(self.experiment_date_dir, self.experiment_time)}"
            f"{ConsoleFormat.MSG_END_FORMAT} are recorded at:\n\n"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
            f"  {self.experiment_path}"
            f"{ConsoleFormat.MSG_END_FORMAT}\n"
        )
        return None

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        show_splash_screen_in_console()

        # Fetch again for multiprocess launcher case e.g. launcher=joblib
        self._fetch_information_for_multirun_console_print(config)

        msg = consol_msg_universal_one_liner(
            "Experimental data for multirun "
            f"{ConsoleFormat.MSG_EMPH_FORMAT}"
            f"{self.multirun_experiment_dir}"
            f"{ConsoleFormat.MSG_END_FORMAT} are recorded at:\n\n"
            f"{ConsoleFormat.MSG_DIMMED_FORMAT}"
            f"  {self.multirun_experiment_path}"
            f"{ConsoleFormat.MSG_END_FORMAT}\n\n",
            print_it=False,
        )

        # .... Copy optuna storage to experiment directory ........................................
        if is_hydra_optuna_run(config, debug_mode=False):
            shutil.copy(config.hparam_optimizer.db_path, self.multirun_experiment_path)
            new_db_path = os.path.join(
                self.multirun_experiment_path, config.hparam_optimizer.db_name
            )

            msg += (
                f"The experiment Optuna study database has been copied to the experiment "
                f"directory:\n\n{ConsoleFormat.MSG_DIMMED_FORMAT}"
                f"  {new_db_path}"
                f"{ConsoleFormat.MSG_END_FORMAT}\n\n"
            )

            msg += self._set_optuna_dashboard_command_msg(config, db_path=new_db_path)

        print(msg)
        return None

    def _get_experiment_id(self) -> str:
        experiement_id = os.path.join(self.experiment_date_dir, self.experiment_time)
        if self._is_multirun:
            job_num = hydra.utils.HydraConfig.get().job.num
            experiement_id += f" job #{job_num}"
        return experiement_id

    def _fetch_information_for_singlerun_console_print(
        self, config: DictConfig
    ) -> None:
        (
            self.experiment_path,
            self.experiment_date_dir,
            self.experiment_time,
        ) = get_experiment_path_date_time(config)
        return None

    def _fetch_information_for_multirun_console_print(
        self, config: omegaconf.DictConfig
    ) -> None:
        if self.multirun_experiment_path is None:
            experiment_path = config.hydra.sweep.dir
            assert experiment_path is not None

            self.multirun_experiment_path = experiment_path
            self.experiment_date_dir = os.path.basename(
                os.path.dirname(self.multirun_experiment_path)
            )
            self.experiment_time = os.path.basename(self.multirun_experiment_path)
            self.multirun_experiment_dir = os.path.join(
                self.experiment_date_dir, self.experiment_time
            )

        return None

    def _set_optuna_dashboard_command_msg(
        self, config: DictConfig, db_path: Optional[str] = None
    ) -> str:
        msg = ""
        if is_hydra_optuna_run(config, debug_mode=False):
            # (Priority) ToDo: implement path exist check to flag the cause of error
            # such as "sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open
            # database file" which is caused by missing directory e.g. `optuna_storage/`
            # `db_path_pre` is empty for non-sqlite backends (journal,
            # postgres); rebuild a dashboard-compatible URL based on
            # `hparam_optimizer.storage_kind` so the printed command works
            # regardless of the active storage backend.
            storage_kind = str(
                config.hparam_optimizer.get("storage_kind", "sqlite")
            ).lower()
            db_path_pre = config.hparam_optimizer.get("db_path_pre", "") or ""
            effective_db_path = db_path or config.hparam_optimizer.db_path
            if not db_path:
                hydra_sweeper_conf = config.hydra.sweeper
                storage = hydra_sweeper_conf.storage
            else:
                storage = f"{db_path_pre}{db_path}"

            # `optuna-dashboard` does NOT accept the `journal://` URL form
            # we feed our internal `R2S2RJournalAwareOptunaSweeper`; it
            # parses URLs as SQLAlchemy DSNs and rejects unknown dialects.
            # For the journal backend it requires `--storage-class
            # JournalFileStorage <path>` with a bare filesystem path.
            # Always rebuild the dashboard command for journal regardless
            # of whether `db_path_pre` is set.
            if storage_kind == "journal":
                storage = (
                    f"--storage-class JournalFileStorage "
                    f"{effective_db_path}"
                )
            elif storage_kind == "postgres" and not db_path_pre:
                storage = (
                    "<set OPTUNA_DB_PATH_PRE to your full "
                    "postgresql+psycopg2://user:pass@host:port/db DSN>"
                )

            msg += (
                "From a DN container termninal, open optuna-dashboard using the following "
                "command:\n\n"
                f"  {ConsoleFormat.MSG_DIMMED_FORMAT}$ optuna-dashboard --port "
                f"{config.hparam_optimizer.optuna_dashboard.port} --host "
                f"{config.hparam_optimizer.optuna_dashboard.host} "
                f"{storage}{ConsoleFormat.MSG_END_FORMAT}\n\n"
                "and then open a web browser using the explicit host ip adress "
                f"e.g.: (<removed-wifi-ssid> wlan) http://"
                f"{config.hparam_optimizer.optuna_dashboard.host}:"
                f"{config.hparam_optimizer.optuna_dashboard.port}"
            )
        return msg


def fetch_r2s2r_project_root_path(cfg: omegaconf.DictConfig, lvl_up: int) -> AnyStr:
    """Returns the absolute path to the MTM-Pro project root directory (aka
    the R2S2R research project).

    1. Use DN-project environment variable if set
    2. Fallback to hydra config otherwise

    :param cfg: the hydra config dict
    :param lvl_up: the number of level up from the hydra original cwd
    :return: the absolute path to the project root directory
    """
    expected_root_dir = "MTM-Pro"

    if is_dockerized_AnonLab_env_var_set():
        project_root_path = fetch_dna_container_root_path_via_env_var(
            expected_root_dir
        )
    else:
        project_root_path = fetch_project_root_path_via_hydra(
            cfg, lvl_up, expected_root_dir
        )

    assert os.path.basename(project_root_path) == expected_root_dir
    return project_root_path


class R2S2RPipelineHydraApp:
    """A class that sets up and manages a Hydra-based experiment including:
     - custom logger setup/teardown,
     - set headless vs rendered logic,
     - register custom omegaconf resolver


    Public attributes
    .................
    - `headless` : Indicates if the experiment is running in headless mode.
    - `wrap_it` : Indicates if the experiment is intended to be executed in gym F110 environment
      wrapped in a 'F1tenthGymSingleAgentWrapper' gym wrapper.
    """
    # (Priority) ToDo: migrate logic to Hydra callback object

    headless: bool
    wrap_it: bool
    _cfg: omegaconf.DictConfig
    optuna_trial_handle: Optional[OptunaTrialHandle]

    def __init__(self, cfg):
        self._cfg = cfg
        self.optuna_trial_handle = None
        self._setup()

    def _setup(self):
        """Initialize and configure the experiment setup.

        This method sets up the console output recorder, shows splash screen,
        draws terminal lines, determines the experiment path, displays startup
        information, and manages Hydra configuration. It performs runtime
        configuration modifications and adjusts setup execution based on
        simulation mode rendering.

        :return: self
        """
        is_multiprocess_hydra_job = is_hydra_optuna_run(
            self._cfg
        ) and is_hydra_sweeper_multiprocess_run(self._cfg)
        sys.stdout = ConsoleRecorder(print_to_console=not is_multiprocess_hydra_job)

        # .... Report experiment directory names ..................................................
        self._cfg.orig_cwd = hydra.utils.get_original_cwd()

        # .... Register omegaconf resolver ........................................................
        if omegaconf.__version__ >= "2.1.0dev20":
            if self._cfg.startup_info:
                consol_msg_universal_one_liner(
                    f"{ConsoleFormat.MSG_WARNING_FORMAT}"
                    "You can use nested interpolation with resolver in hydra now that you "
                    "have "
                    f"omegaconf.__version__ {omegaconf.__version__} >= 2.1.0dev20.\n\n"
                    "   (NICE TO HAVE) ToDo ref task RLRP-192:\n"
                    "      Use resolver to compute aritmetic with nested interpolation\n\n"
                    "      hydra_config.yaml\n"
                    "      >>> num_steps: ${eval:'${"
                    "overrides.sampler_rollout.nb_sampling_trials}*${"
                    "overrides.sampler_rollout.trajectory_max_length}'}\n\n"
                    "      Quickhack until then hydra_config.yaml\n"
                    "      >>> num_steps: ???\n"
                    f"{ConsoleFormat.MSG_END_FORMAT}"
                )
            omegaconf.OmegaConf.register_new_resolver(
                "sum", lambda x, y: x + y, replace=True
            )
            omegaconf.OmegaConf.register_new_resolver(
                "product", lambda x, y: x * y, replace=True
            )

        # ... Runtime config modification .........................................................
        set_hydra_config_missing_keys(self._cfg)

        # ... Setup execution .....................................................................
        self.headless = False
        if self._cfg.simulation_mode.rendering in ["headless_fast", "headless"]:
            self.headless = True

        self.wrap_it = False

        if self._cfg.simulation_mode.rendering:
            # Note: Meaning config for gym rendering is define (e.g. headless, human ...)
            #       and the run is intended to be executed in gym F110 environment wrapped in a
            #       'F1tenthGymSingleAgentWrapper' gym wrapper.
            self.wrap_it = True

        if self._cfg.debug_mode_force_early_exit:
            consol_msg_universal_one_liner(
                f"{ConsoleFormat.MSG_WARNING_FORMAT}Debug mode debug_mode_force_early_exit "
                f"activated{ConsoleFormat.MSG_END_FORMAT}"
            )
            os.system('/bin/bash -c "printenv"')
            exit()

        # Torch backend flags -- the ONE place that writes the process-global `torch.backends`
        # state, from `cfg.torch_backend.*` (RLRP-824 consolidation):
        #   - RLRC precision hardening (Gate D): `cuda_high_precision_float32` (default true)
        #     disables TF32 / forces "highest" float32 matmul precision so a float32 fallback
        #     keeps its full mantissa; `false` is the A100 TF32 speed opt-in. Gated on
        #     `torch.cuda.is_available()` (covers Valeria GPUs + Jetson, no-op on the CPU DNA
        #     container).
        #   - Training Speed & Efficiency plan A3a: `cudnn_benchmark` (cuDNN autotune).
        apply_torch_backend_cfg(self._cfg)

        # if is_run_in_DN_arm64_jetson_architecture() and bool(_torch_backend_cfg.get("high_precision_jetson", True)):
        #     self.configure_optimal_torch_settings_for_jetson()

        # RLRP-624 Phase A.3.4 — assert direction cardinality matches
        # the registered objectives. Best-effort, never blocks single-runs.
        self._assert_optuna_direction_cardinality()

        # RLRP-624 Phase B — Optuna pruning hook. Only active in MULTIRUN
        # Optuna jobs with a configured pruner; otherwise silently no-op.
        if is_hydra_optuna_run(self._cfg):
            self.optuna_trial_handle = acquire_optuna_trial(self._cfg)
            if self.optuna_trial_handle is not None:
                consol_msg_universal_one_liner(
                    f"Optuna trial #{self.optuna_trial_handle.trial.number} "
                    f"acquired by R2S2RPipelineHydraApp"
                )

        return self

    def _assert_optuna_direction_cardinality(self) -> None:
        """RLRP-624 A.3.4: ``hydra.sweeper.direction`` cardinality must match
        ``hparam_optimizer.objectives_name`` cardinality.

        Best-effort guard — only emits a warning if the cfg keys are not both
        present (e.g. non-HPO single runs).
        """
        try:
            hparam = self._cfg.get("hparam_optimizer", None)
            if hparam is None:
                return
            objectives = hparam.get("objectives_name", None)
            if objectives is None:
                return
            sweeper = omegaconf.OmegaConf.select(
                self._cfg, "hydra.sweeper", throw_on_missing=False
            )
            if sweeper is None:
                return
            direction = sweeper.get("direction", None)
            if direction is None:
                return
            n_obj = 1 if isinstance(objectives, str) else len(objectives)
            n_dir = 1 if isinstance(direction, str) else len(direction)
            if n_obj != n_dir:
                consol_msg_universal_one_liner(
                    f"{ConsoleFormat.MSG_WARNING_FORMAT}"
                    f"RLRP-624 A.3.4: hydra.sweeper.direction cardinality "
                    f"({n_dir}) != hparam_optimizer.objectives_name cardinality "
                    f"({n_obj}). Optuna will likely error at study creation."
                    f"{ConsoleFormat.MSG_END_FORMAT}"
                )
        except Exception:
            # Never fail the run on a config-shape introspection.
            return

    # ---- RLRP-624 Phase B — Optuna pruning API ------------------------------

    def report_to_optuna(self, value: float, step: int) -> None:
        """Forward an intermediate metric to the active Optuna trial (no-op
        when pruning is disabled or unavailable)."""
        h = self.optuna_trial_handle
        if h is None:
            return
        try:
            h.trial.report(float(value), step=int(step))
        except Exception as e:
            consol_msg_universal_one_liner(
                f"[OptunaPruning] report failed at step={step}: {e}"
            )

    def should_prune(self) -> bool:
        h = self.optuna_trial_handle
        if h is None:
            return False
        try:
            return bool(h.trial.should_prune())
        except Exception:
            return False

    def raise_if_should_prune(self, value: float, step: int) -> None:
        """One-call helper for trainer epoch callbacks: report and raise
        ``optuna.TrialPruned`` when the pruner asks to stop the trial."""
        self.report_to_optuna(value, step)
        if self.should_prune():
            consol_msg_universal_one_liner(
                f"[OptunaPruning] trial pruned at step={step} (value={value})"
            )
            raise optuna.TrialPruned()

    def finalize_pruned(self) -> None:
        """Mark the trial as PRUNED in storage (called by
        ``select_*_pipeline_and_execute`` when ``optuna.TrialPruned`` bubbles
        up from the pipeline)."""
        if self.optuna_trial_handle is not None:
            mark_trial_pruned(self.optuna_trial_handle)

    @staticmethod
    def configure_optimal_torch_settings_for_jetson() -> None:
        # (CRITICAL) ToDo: validate

        import torch

        # Optimize memory allocation
        # torch.cuda.empty_cache() # Q: Doest it work with multi-processing? ⚠️

        # torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available memory
        # Note: need to be set considering the number of process running

        consol_msg_universal_one_liner("Configured torch optimal settings for jetson", caller_name="R2S2RPipelineHydraApp")
        return None

    @staticmethod
    def configure_cuda_float32_high_precision(enable: bool = True) -> None:
        """Backward-compatible alias of the module-level
        :func:`configure_cuda_float32_high_precision`. Prefer ``apply_torch_backend_cfg(cfg)``,
        which is the single cfg-driven writer of the ``torch.backends`` state."""
        configure_cuda_float32_high_precision(enable=enable)
        return None

    def teardown(self):
        """Tears down the experimental setup by performing the following tasks:
        1. Outputs a message indicating pipeline exit.
        2. Closes the ConsoleRecorder object (i.e. teardown sys.stdout to original value)

        :return: None
        """
        if not is_hydra_multirun():
            consol_msg_universal_one_liner("Pipeline exit › see you.")
        else:
            consol_msg_universal_one_liner(
                f"(single run) Pipeline {ConsoleFormat.MSG_DONE_FORMAT}done"
                f"{ConsoleFormat.MSG_END_FORMAT}."
            )

        sys.stdout.close()
        return self


def get_experiment_path_date_time(cfg: DictConfig) -> Tuple[str, str, str]:
    experiment_path = get_hydra_experiment_cwd(cfg)
    # (NICE TO HAVE) ToDo: refactor >> see `src/tools/math_gym_env_tools/plot_3d_utils.py:213`
    #  at the "Show experiment relevant information" bloc.
    if is_hydra_multirun():
        experiment_path = os.path.dirname(experiment_path)
    experiment_date_dir = os.path.basename(os.path.dirname(experiment_path))
    experiment_time = os.path.basename(experiment_path)
    return experiment_path, experiment_date_dir, experiment_time


def set_hydra_config_missing_keys(cfg) -> None:
    """Fill missing hydra configuration dictionary keys
    Note: Update hydra cfg in place

    :param cfg: hydra config dict
    """
    set_core_hydra_missing_keys(cfg)

    if cfg.debug_mode:
        consol_msg_universal(
            f"{ConsoleFormat.MSG_WARNING_FORMAT}Debug mode activated"
            f"{ConsoleFormat.MSG_END_FORMAT}"
        )
        # Note: Set the following environment variable in your shell or run config
        # to show debug information
        #   - PYCHARM_DEBUG = True
        #   - HYDRA_FULL_ERROR = 1
        #   - OC_CAUSE = 1  # Set omegaconf full error backtrace
        #
        consol_msg_universal(
            f"Show hydra cfg:\n\n{omegaconf.OmegaConf.to_yaml(cfg, resolve=False)}"
        )
        consol_msg_universal(f"Show environment variables\n\n{str(os.environ)}")

    # .... Sampling app related setup .............................................................
    if omegaconf.OmegaConf.is_missing(cfg.overrides, key="num_steps"):
        num_steps_summation_resolver_quickhack(cfg)

    if omegaconf.OmegaConf.is_missing(cfg.algorithm, key="dataset_size"):
        dataset_size_summation_resolver_quickhack(cfg)

    return None


def set_core_hydra_missing_keys(cfg) -> None:
    if is_pytest_run():
        cfg.host_user.name = "pytest_runner"

    if os.getenv("JETBRAINS_REMOTE_RUN"):
        cfg.IDE.ide_remote_run = bool(os.getenv("JETBRAINS_REMOTE_RUN"))
    else:
        cfg.IDE.ide_remote_run = False

    if os.getenv("PYCHARM_HOSTED"):
        cfg.IDE.pycharm = bool(os.getenv("PYCHARM_HOSTED"))
    else:
        cfg.IDE.pycharm = False

    if not omegaconf.OmegaConf.select(cfg, "device", throw_on_missing=False):
        # Note: Mac M1 use `mps` instead of `cpu`
        cfg.device = (
                "cuda:0"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        consol_msg_universal_one_liner(f"Use {cfg.device} device",
                                       caller_name="R2S2RPipelineHydraApp")
    return None
