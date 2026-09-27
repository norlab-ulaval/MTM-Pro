# coding=utf-8
import inspect
from typing import Iterable, List, Optional, Tuple

import mbrl.models
import omegaconf
import torch

from tools.console_tools.message import consol_msg_universal_one_liner
from tools.hydra_apps_tools.omegaconf_utils import is_cfg_key_exist

# RLRP-783 (A8): torch ``2.8.0`` restricts ``capturable=True`` Adam to these device types
# (``torch/optim/adam.py`` asserts at ``.step()`` time). A plain ``cpu`` device is NOT in the
# set, so ``capturable`` is only a valid fallback on CUDA-family accelerators.
_CAPTURABLE_SUPPORTED_DEVICES = frozenset(
    {"cuda", "xpu", "hpu", "privateuseone", "xla"}
)


def build_mtm_pro_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    base_wd: float,
    body_lr_mult: float,
    head_lr_mult: float,
    body_wd_mult: float,
    head_wd_mult: float,
    head_prefixes: Tuple[str, ...] = ("projection_head", "deploy_head_"),
) -> List[dict]:
    """Split an MTM-Pro model's trainable params into a slow body and a fast deploy-head group.

    Deploy-path stabilization feature (2), RLRP-722: returns two ``torch.optim`` parameter groups
    so a single optimizer can give the forecast body (``hidden_layers`` / ``mean_and_logvar`` /
    ``mean_layer`` / ``logvar_layer`` / ``temporal_mixture_weights`` / auto-weighting / ...) a
    slower LR (and its own weight decay) while the deploy heads (``projection_head`` /
    ``deploy_head_*``) get a faster LR. The split is name-prefix based and robust: any parameter
    not matched by ``head_prefixes`` falls into the body/default group. A hard assert guarantees
    the body+head union equals all trainable params (no silently dropped/duplicated params).

    :param model: the (possibly wrapped) model; the trainable network is reached via
        ``getattr(model, "model", model)`` to unwrap ``OneDTransitionRewardModelV2``.
    :param base_lr: the base learning rate (the body/head LR multipliers scale this).
    :param base_wd: the base weight decay (the body/head WD multipliers scale this).
    :param body_lr_mult: body LR = ``base_lr * body_lr_mult``.
    :param head_lr_mult: deploy-head LR = ``base_lr * head_lr_mult``.
    :param body_wd_mult: body weight decay = ``base_wd * body_wd_mult``.
    :param head_wd_mult: deploy-head weight decay = ``base_wd * head_wd_mult``.
    :param head_prefixes: parameter-name prefixes routed to the fast (deploy-head) group.
    """
    net = getattr(model, "model", model)  # unwrap OneDTransitionRewardModelV2
    head: List[torch.nn.Parameter] = []
    body: List[torch.nn.Parameter] = []
    for name, p in net.named_parameters():
        if not p.requires_grad:
            # The EMA forecast clone (feature 3) has requires_grad=False params; skip them so
            # they never enter the optimizer.
            continue
        (head if name.startswith(head_prefixes) else body).append(p)
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert len(head) + len(body) == len(trainable), (
        "build_mtm_pro_param_groups: param-group split dropped or duplicated params "
        f"(head={len(head)} + body={len(body)} != trainable={len(trainable)})"
    )
    return [
        {
            "params": body,
            "lr": base_lr * body_lr_mult,
            "weight_decay": base_wd * body_wd_mult,
        },
        {
            "params": head,
            "lr": base_lr * head_lr_mult,
            "weight_decay": base_wd * head_wd_mult,
        },
    ]


def _resolve_adam_fused_kwargs(
    cfg_training: omegaconf.DictConfig,
    trainer: mbrl.models.ModelTrainer,
    show_consol_msg: bool = True,
) -> dict:
    """Resolve the RLRP-783 ``A8`` fused/capturable Adam opt-in flags into ctor kwargs.

    RLRP-783 follow-up action ``A8`` (see
    ``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``): torch ``2.8.0``'s
    single-tensor Adam kernel issues one ``torch.optim.optimizer._get_value(step).item()``
    device->host sync per parameter, every step — the dominant residual after ``A1``-``A6``.
    ``fused=True`` (CUDA-only) collapses the step into one kernel with no per-parameter
    ``.item()``; ``capturable=True`` keeps ``step`` on device (also removes the sync) and is the
    fallback on CUDA-family accelerators where fused is unavailable — but NOT on plain CPU
    (torch ``2.8.0`` rejects capturable there).

    Reads the two new keys off the ``optimizer`` sub-group (relative to ``cfg_training``, i.e.
    ``optimizer.adam_fused`` / ``optimizer.adam_capturable``) via ``OmegaConf.select`` with
    ``default=False`` so an ABSENT key ⇒ legacy single-tensor Adam (backward compatible).
    ``fused=True`` is only emitted on CUDA; ``capturable=True`` only on the torch-supported
    capturable devices (``cuda`` / ``xpu`` / ``hpu`` / ``privateuseone`` / ``xla`` — on torch
    ``2.8.0`` ``capturable=True`` on a plain CPU device raises at ``.step()`` time). When the
    requested variant is unavailable on the current device the resolver falls back (fused →
    capturable → legacy) and finally to the legacy single-tensor Adam, emitting a
    ``consol_msg_universal_one_liner`` notice. Returns at most one of ``{fused, capturable}``
    EXCEPT when BOTH are requested on CUDA: RLRP-786 (CUDA-graph captured training step,
    ``perf_RLRP-786_cuda_graph_train_step_plan_20260919.md``) needs the fused kernel AND
    ``capturable=True`` (the Adam ``step`` counter stays on device so ``optimizer.step()`` can be
    recorded in a ``torch.cuda.CUDAGraph``); torch >= 2.1 accepts the pair. An empty dict ⇒ keep
    the legacy single-tensor Adam.
    """
    want_fused = bool(
        omegaconf.OmegaConf.select(cfg_training, "optimizer.adam_fused", default=False)
    )
    want_capturable = bool(
        omegaconf.OmegaConf.select(
            cfg_training, "optimizer.adam_capturable", default=False
        )
    )
    if not want_fused and not want_capturable:
        return {}

    _device = getattr(trainer.model, "device", None)
    _device_type = getattr(_device, "type", None) if _device is not None else None
    is_cuda = _device_type == "cuda"
    # torch ``2.8.0``: ``capturable=True`` asserts params live on one of these device types.
    capturable_ok = _device_type in _CAPTURABLE_SUPPORTED_DEVICES

    if want_fused:
        if is_cuda:
            if want_capturable:
                if show_consol_msg:
                    consol_msg_universal_one_liner(
                        "A8 (RLRP-783) + RLRP-786: fused AND capturable Adam ON (CUDA) — "
                        "step counter on device, CUDA-graph capturable"
                    )
                return {"fused": True, "capturable": True}
            if show_consol_msg:
                consol_msg_universal_one_liner(
                    "A8 (RLRP-783): fused Adam ON (CUDA) — per-parameter _get_value sync removed"
                )
            return {"fused": True}
        if want_capturable and capturable_ok:
            if show_consol_msg:
                consol_msg_universal_one_liner(
                    "A8 (RLRP-783): fused Adam requested but device is not CUDA — "
                    "falling back to capturable Adam"
                )
            return {"capturable": True}
        if show_consol_msg:
            consol_msg_universal_one_liner(
                "A8 (RLRP-783): fused/capturable Adam requested but unavailable on device "
                f"'{_device_type}' — using legacy single-tensor Adam"
            )
        return {}

    # want_capturable only
    if capturable_ok:
        if show_consol_msg:
            consol_msg_universal_one_liner(
                "A8 (RLRP-783): capturable Adam ON — per-parameter _get_value sync removed"
            )
        return {"capturable": True}
    if show_consol_msg:
        consol_msg_universal_one_liner(
            "A8 (RLRP-783): capturable Adam requested but unavailable on device "
            f"'{_device_type}' — using legacy single-tensor Adam"
        )
    return {}


def _adapt_fused_kwargs_for(
    optim_cls: type,
    kwargs: dict,
    trainer: mbrl.models.ModelTrainer,
    show_consol_msg: bool = True,
) -> dict:
    """Filter/adapt the resolved fused/capturable kwargs to an optimizer's real signature.

    RLRP-783 ``A8`` follow-up bugfix: the fused/capturable opt-in was resolved once and passed
    verbatim to whichever optimizer branch matched, but the ``torch.optim`` optimizers do NOT
    share the same signature — e.g. ``NAdam`` accepts ``capturable`` but NOT ``fused`` (raising
    ``TypeError: NAdam.__init__() got an unexpected keyword argument 'fused'``), while ``SGD``
    accepts ``fused`` but NOT ``capturable``. This helper introspects the target optimizer's
    ``__init__`` signature and:

    - drops any kwarg the optimizer does not accept;
    - when ``fused=True`` was requested but the optimizer has no ``fused`` parameter, downgrades
      to ``capturable=True`` if the optimizer accepts it AND the device is a torch-supported
      capturable device (otherwise falls back to the legacy single-tensor path).

    An empty input dict (legacy path) is returned untouched.
    """
    if not kwargs:
        return {}
    accepted = inspect.signature(optim_cls).parameters
    out = {k: v for k, v in kwargs.items() if k in accepted}
    if kwargs.get("fused") and "fused" not in accepted:
        _device = getattr(trainer.model, "device", None)
        _device_type = getattr(_device, "type", None) if _device is not None else None
        if "capturable" in accepted and _device_type in _CAPTURABLE_SUPPORTED_DEVICES:
            out["capturable"] = True
            if show_consol_msg:
                consol_msg_universal_one_liner(
                    f"A8 (RLRP-783): {optim_cls.__name__} has no `fused` kwarg — "
                    "falling back to capturable"
                )
        elif show_consol_msg:
            consol_msg_universal_one_liner(
                f"A8 (RLRP-783): {optim_cls.__name__} has no `fused` kwarg and capturable is "
                f"unavailable on device '{_device_type}' — using legacy single-tensor path"
            )
    return out


def change_optimizer(
    cfg_training: omegaconf.DictConfig,
    trainer: mbrl.models.ModelTrainer,
    lr: Optional[float] = None,
    show_consol_msg: bool = True,
) -> mbrl.models.ModelTrainer:
    if lr:
        model_lr = lr
    else:
        model_lr = cfg_training.model_lr

    # .... Deploy-path training stabilization feature (2): two-timescale LR (RLRP-722) ...........
    # The two-timescale knobs live on the MODEL (not ``cfg_training``), so reach them off the
    # already-constructed model object (smaller blast radius than threading a new config arg).
    # ``params`` is passed to whichever optimizer branch is selected below; when the feature is
    # OFF it stays the flat ``trainer.model.parameters()`` (byte-for-byte unchanged behaviour).
    _mtm = getattr(trainer.model, "model", None)
    if _mtm is not None and getattr(_mtm, "_two_timescale_enable", False):
        params: Iterable = build_mtm_pro_param_groups(
            trainer.model,
            model_lr,
            cfg_training.model_wd,
            _mtm._two_timescale_body_lr_mult,
            _mtm._two_timescale_head_lr_mult,
            _mtm._two_timescale_body_wd_mult,
            _mtm._two_timescale_head_wd_mult,
        )
        if show_consol_msg:
            consol_msg_universal_one_liner(
                "Two-timescale LR ON (body/head param groups)"
            )
    else:
        params = trainer.model.parameters()

    # (NICE TO HAVE) ToDo: RLRP-312 investigate uder 1 iteration as a global step epoch
    # RLRP-727 (B6): track whether a non-Adam branch actually matched so we can fail loudly on a
    # misconfiguration instead of silently leaving the trainer's existing (Adam) optimizer in place.
    _use_adam = bool(omegaconf.OmegaConf.select(cfg_training, "optimizer.use_adam", default=True))
    _non_adam_branch_matched = False

    # RLRP-783 (A8): resolve the opt-in fused/capturable Adam kwargs once (CUDA-guarded, with a
    # capturable/legacy fallback + console notice). Absent flags ⇒ empty dict ⇒ legacy path.
    # The ``torch.optim`` optimizers do NOT share one signature: ``Adam`` / ``AdamW`` accept both
    # ``fused`` and ``capturable``; ``NAdam`` accepts ``capturable`` but NOT ``fused``; ``SGD``
    # accepts ``fused`` but NOT ``capturable``. So the resolved kwargs are adapted per optimizer
    # via ``_adapt_fused_kwargs_for`` (drop unsupported keys, downgrade ``fused``→``capturable``
    # where possible). See ``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``.
    _adam_fused_kwargs = _resolve_adam_fused_kwargs(cfg_training, trainer, show_consol_msg)

    if is_cfg_key_exist(cfg_training, "optimizer.SGD") and not cfg_training.optimizer.use_adam:
        if show_consol_msg:
            consol_msg_universal_one_liner("Setting optimizer to SGD")
        is_cfg_key_exist(cfg_training, "optimizer.SGD.momentum", raise_error=True)
        is_cfg_key_exist(cfg_training, "optimizer.SGD.dampening", raise_error=True)
        is_cfg_key_exist(cfg_training, "optimizer.SGD.nesterov", raise_error=True)
        trainer.optimizer = torch.optim.SGD(
            params,
            lr=model_lr,
            momentum=cfg_training.optimizer.SGD.momentum,
            dampening=cfg_training.optimizer.SGD.dampening,
            weight_decay=cfg_training.model_wd,
            nesterov=cfg_training.optimizer.SGD.nesterov,
            **_adapt_fused_kwargs_for(
                torch.optim.SGD, _adam_fused_kwargs, trainer, show_consol_msg
            ),
        )
        _non_adam_branch_matched = True
    elif is_cfg_key_exist(cfg_training, "optimizer.NAdam") and not cfg_training.optimizer.use_adam:
        if show_consol_msg:
            consol_msg_universal_one_liner("Setting optimizer to NAdam")
        is_cfg_key_exist(cfg_training, "optimizer.NAdam.momentum_decay", raise_error=True)
        trainer.optimizer = torch.optim.NAdam(
            params,
            lr=model_lr,
            weight_decay=cfg_training.model_wd,
            momentum_decay=cfg_training.optimizer.NAdam.momentum_decay,
            decoupled_weight_decay=cfg_training.optimizer.NAdam.get('decoupled_weight_decay', False),
            **_adapt_fused_kwargs_for(
                torch.optim.NAdam, _adam_fused_kwargs, trainer, show_consol_msg
            ),
        )
        _non_adam_branch_matched = True
    elif is_cfg_key_exist(cfg_training, "optimizer.AdamW") and not cfg_training.optimizer.use_adam:
        if show_consol_msg:
            consol_msg_universal_one_liner("Setting optimizer to AdamW")
        is_cfg_key_exist(cfg_training, "optimizer.AdamW.amsgrad", raise_error=True)
        trainer.optimizer = torch.optim.AdamW(
            params,
            lr=model_lr,
            weight_decay=cfg_training.model_wd,
            amsgrad=cfg_training.optimizer.AdamW.amsgrad,
            **_adapt_fused_kwargs_for(
                torch.optim.AdamW, _adam_fused_kwargs, trainer, show_consol_msg
            ),
        )
        _non_adam_branch_matched = True

    # RLRP-783 (A8): explicit Adam-rebuild branch. When Adam is in use (``optimizer.use_adam=true``
    # — the default and the HPO configs) the trainer keeps the single-tensor Adam built in
    # ``ModelTrainer.__init__``; rebuild it here with the resolved fused/capturable kwargs so the
    # per-parameter ``_get_value(step).item()`` device->host sync is eliminated. Preserves
    # lr/weight_decay/eps. Absent flags ⇒ ``_adam_fused_kwargs`` empty ⇒ this branch is skipped
    # and the legacy single-tensor Adam is kept byte-for-byte unchanged.
    if _use_adam and _adam_fused_kwargs:
        trainer.optimizer = torch.optim.Adam(
            params,
            lr=model_lr,
            weight_decay=cfg_training.model_wd,
            eps=getattr(trainer, "optim_eps", 1e-8),
            **_adam_fused_kwargs,
        )

    # RLRP-727 (B6): guard against silent passthrough. If a non-Adam optimizer was requested
    # (`optimizer.use_adam=false`) but none of the SGD/NAdam/AdamW branches matched (missing or
    # misspelled `optimizer.<KIND>` block), the trainer would silently keep its existing (Adam)
    # optimizer — a hard-to-diagnose misconfiguration. Fail loudly instead.
    if not _use_adam and not _non_adam_branch_matched:
        raise ValueError(
            "change_optimizer: `optimizer.use_adam=false` but no matching optimizer block was "
            "found (expected one of `optimizer.SGD`, `optimizer.NAdam`, `optimizer.AdamW`). "
            "The trainer optimizer was NOT changed. Check the `optimizer.<KIND>` config key."
        )
    return trainer
