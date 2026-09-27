# coding=utf-8
from typing import Callable, Dict, List, Optional, Sequence, Union

from deprecated import deprecated
import hydra
import omegaconf
import torch
from mbrl.models import EnsembleLinearLayer
from torch import distributions as dist, nn as nn
from torch.nn import functional as F
import math
import numpy as np

from tools.model_adapter_tools.ms_to_ss_observation_adapter import (
    MultistepObservationToSinglestepObservationAdapter,
)


def fast_distribution_mode_approximation(samples: torch.Tensor, ensemble_size: int) -> torch.Tensor:
    """
    Compute an approximation of the dominant mode in a sample distribution using pairwise
    distances and identifying the geometric median.

    This function computes pairwise distances between all samples and identifies the sample that
    minimizes the sum of distances to all other samples. This sample serves as a proxy for the
    dominant mode of the distribution.

    :param ensemble_size:
    :param samples: A tensor containing the samples with shape [num_samples, batch_size, feature_dim].
    :return: A tensor representing the sample in the input that corresponds to the
        geometric median of the distribution.
    """
    # Calculate pairwise distances
    if ensemble_size > 1:
        dists = torch.cdist(samples.mean(dim=1), samples.mean(dim=1))
    else:
        dists = torch.cdist(samples, samples)

    # The sample with the minimum sum of distances to others is the "geometric median"
    # (a proxy for the dominant mode)
    idx = torch.argmin(dists.sum(dim=-1), dim=0)
    return samples[idx]


@deprecated(
    reason=(
        "EnsembleResidualLinearLayer is deprecated and will be removed in a future version. "
        "Please use `EnsembleResidualBlock`."
    )
)
class EnsembleResidualLinearLayer(EnsembleLinearLayer):
    """LEGACY pre-activation-wrapped residual layer: ``act(x + Wx + b)``.

    Kept only for ``residual_form="legacy_preact_wrapped"`` ablations and to
    reproduce checkpoints trained before the switch to the canonical
    identity-preserving block (:class:`EnsembleResidualBlock`). New runs use
    ``residual_form="identity_v2"`` (the default). Requires ``in_size == out_size``
    so that the residual skip connection is dimensionally valid.
    """

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        assert in_size == out_size, (
            f"{type(self).__name__} requires in_size == out_size for the residual "
            f"skip connection, got in_size={in_size} != out_size={out_size}."
        )
        super().__init__(num_members, in_size, out_size, bias=bias)

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                xw = xw + self.bias[self.elite_models, ...]
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                xw = xw + self.bias
        return x + xw


class EnsembleResidualBlock(nn.Module):
    """Canonical identity-preserving residual block for ensemble models.

    Pre-activation v2 form (He et al. 2016, "Identity Mappings in Deep Residual
    Networks")::

        F(x) = Linear2( act( Linear1( dropout(x) ) ) )
        y    = x + F(x)                       # PURE identity skip connection

    Unlike :class:`EnsembleResidualLinearLayer` (which computes the
    activation-wrapped ``act(x + Wx + b)``), the skip connection here is an
    exact identity: the ``+ x`` is the LAST operation, so the Jacobian always
    contains an identity term ``I``. This is the standard choice for deep
    residual stacks (e.g. the MS-head stacks driven by ``ms_head_num_layers``).

    When ``zero_init_last=True`` the second linear is zero-initialized so the
    block is an EXACT identity at init (``y == x``), which gives the cleanest
    gradient flow for deep stacks.

    The block OWNS its dropout: pass the encoder ``dropout`` for hidden stacks
    and ``ms_head_dropout`` for MS-head stacks so each region can be toggled
    independently. Requires ``in_size == out_size``.
    """

    def __init__(
        self,
        num_members: int,
        size: int,
        create_activation: Callable[[], nn.Module],
        hidden_size: Optional[int] = None,
        dropout: float = 0.0,
        zero_init_last: bool = True,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.num_members = num_members
        self.size = size
        hidden_size = hidden_size or size
        self.hidden_size = hidden_size
        self.zero_init_last = zero_init_last
        self.use_layer_norm = bool(layer_norm)
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = EnsembleLinearLayer(num_members, size, hidden_size)
        # Optional pre-norm on the RESIDUAL BRANCH ONLY (never on the skip path),
        # so the exact-identity-at-init guarantee is preserved (RLRP-768).
        self.norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
        self.act = create_activation()
        self.lin2 = EnsembleLinearLayer(num_members, hidden_size, size)
        self.reset_last_layer_to_zero()

    def reset_last_layer_to_zero(self):
        """Zero-initialize the last linear so the block starts as an exact identity.

        Must be re-applied AFTER a model-level ``self.apply(truncated_normal_init)``
        (which would otherwise overwrite ``lin2`` with random weights); see
        :func:`zero_init_residual_blocks_`.
        """
        if self.zero_init_last:
            nn.init.zeros_(self.lin2.weight)
            if self.lin2.use_bias:
                nn.init.zeros_(self.lin2.bias)

    def forward(self, x):
        return x + self.lin2(self.act(self.norm(self.lin1(self.dropout(x)))))

    # CRITICAL: this block is NOT itself an EnsembleLinearLayer, so the elite
    # selection set by the parent model's `_maybe_toggle_layers_use_only_elite`
    # (which calls `layer[0].set_elite(...)` / `layer[0].toggle_use_only_elite()`
    # on the residual layer) must be forwarded to BOTH inner ensemble layers.
    def set_elite(self, elite_models: Sequence[int]):
        self.lin1.set_elite(elite_models)
        self.lin2.set_elite(elite_models)

    def toggle_use_only_elite(self):
        self.lin1.toggle_use_only_elite()
        self.lin2.toggle_use_only_elite()

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, size={self.size}, "
            f"hidden_size={self.hidden_size}, layer_norm={self.use_layer_norm}"
        )


def create_residual_linear_layer_(ensemble_size: int, l_in: int, l_out: int) -> nn.Module:
    assert l_in == l_out, (
        f"(Legacy) residual layer requires l_in == l_out, got l_in={l_in} != l_out={l_out}."
    )
    return EnsembleResidualLinearLayer(ensemble_size, l_in, l_out)


# Supported residual block variants (back-compat switch). The default for new
# runs is `identity_v2`; `legacy_preact_wrapped` reproduces the historical
# `act(x + Wx + b)` behavior and is kept for ablation / old checkpoints.
RESIDUAL_FORMS = ("identity_v2", "legacy_preact_wrapped")

# --- Layer-bloc taxonomy (RLRP-768) ------------------------------------------
# Permanent constants. Introduced by action A1 of the RLRC dense layer bloc
# re-introduction `.junie` plan
# (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
#
# Phase 1 ships four types; Phase 2 appends "gru" to BOTH tuples below (the only
# edit needed, the factory guard is already generic over GATED_LAYER_BLOC_TYPES).
LAYER_BLOC_TYPES = ("residual_v2", "residual_legacy", "dense", "gated_mlp")

# Blocs whose use is restricted to the encoder/hidden regions (RLRP-768).
# Rationale: gated blocs re-scale their output through a learned sigmoid/GLU
# gate, which distorts the scale information a logvar / mixture-logit head must
# preserve (same argument as the LayerNorm placement rule).
GATED_LAYER_BLOC_TYPES = ("gated_mlp",)

# Back-compat alias map: the historical `residual_form` values.
RESIDUAL_FORM_TO_LAYER_BLOC = {
    "identity_v2": "residual_v2",
    "legacy_preact_wrapped": "residual_legacy",
}

# `encoder` super-region: representation stacks ONLY, i.e. layers explicitly
# named `encoder` or `hidden`. LayerNorm + gated blocs are legal here.
LAYER_NORM_ALLOWED_REGIONS = ("encoder", "hidden")
# `decoder` super-region: EVERYTHING not named `encoder`/`hidden`. LayerNorm +
# gated blocs are FORBIDDEN here. `decoder` == the explicit p(y|z) decoder
# stack; `ms_head` == the `mean_and_logvar` / `mean_layer` / `logvar_layer`
# head; `projection` == the `_ProjectionNetworkHead` q_theta (trunk +
# mean/logvar split heads); `probabilistic_head` == any other density head.
LAYER_NORM_FORBIDDEN_REGIONS = (
    "decoder",
    "ms_head",
    "ss_head",
    "mixture",
    "projection",
    "probabilistic_head",
)
LAYER_BLOC_REGIONS = LAYER_NORM_ALLOWED_REGIONS + LAYER_NORM_FORBIDDEN_REGIONS

# The gated/recurrent blocs reuse the SAME allowed-region tuple.
GATED_BLOC_ALLOWED_REGIONS = LAYER_NORM_ALLOWED_REGIONS

# Default layer bloc per region (RLRP-768; encoder/hidden default flipped to
# `dense` in the RLRP-768 follow-up): every region now defaults to the plain
# dense bloc. NOTE: consequently the `residual_form` back-compat alias (which
# only re-selects a residual variant for regions whose default is `residual_v2`)
# is now inert; pin `layer_bloc: {<region>: {type: residual_v2}}` for the
# identity-preserving residual bloc.
DEFAULT_LAYER_BLOC_TYPE_BY_REGION = {
    "encoder": "dense",
    "hidden": "dense",
    "decoder": "dense",
    "ms_head": "dense",
    "ss_head": "dense",
    "mixture": "dense",
    "projection": "dense",
    "probabilistic_head": "dense",
}
assert set(DEFAULT_LAYER_BLOC_TYPE_BY_REGION) == set(LAYER_BLOC_REGIONS), (
    "DEFAULT_LAYER_BLOC_TYPE_BY_REGION must cover exactly LAYER_BLOC_REGIONS "
    f"({LAYER_BLOC_REGIONS}), got {tuple(DEFAULT_LAYER_BLOC_TYPE_BY_REGION)}"
)


class EnsembleDenseBlock(nn.Module):
    """Canonical dense (feed-forward) layer bloc for ensemble models.

    ``y = act( norm( Linear( dropout(x) ) ) )``

    Sibling of :class:`EnsembleResidualBlock` with the SAME ownership contract:
    the bloc owns its dropout, its optional ``LayerNorm`` and its activation, so
    a caller building a stack must NOT append a trailing activation.

    ``layer_norm`` is only legal in encoder/hidden regions; the placement rule is
    enforced upstream by :func:`create_layer_bloc_factory` (LayerNorm is
    forbidden on any component that explicitly learns a probability).

    Permanent public class. Introduced by action A2 of the RLRC dense layer bloc
    re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """

    def __init__(
        self,
        num_members: int,
        size: int,
        create_activation: Callable[[], nn.Module],
        out_size: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm: bool = False,
    ):
        super().__init__()
        out_size = out_size or size
        self.num_members = num_members
        self.size = size
        self.out_size = out_size
        self.use_layer_norm = bool(layer_norm)
        self.dropout = nn.Dropout(p=dropout)
        self.lin = EnsembleLinearLayer(num_members, size, out_size)
        self.norm = nn.LayerNorm(out_size) if layer_norm else nn.Identity()
        self.act = create_activation()

    def forward(self, x):
        return self.act(self.norm(self.lin(self.dropout(x))))

    # Same elite-forwarding contract as `EnsembleResidualBlock`: the bloc is not
    # itself an `EnsembleLinearLayer`, so forward the toggle to the inner layer.
    def set_elite(self, elite_models: Sequence[int]):
        self.lin.set_elite(elite_models)

    def toggle_use_only_elite(self):
        self.lin.toggle_use_only_elite()

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, size={self.size}, "
            f"out_size={self.out_size}, layer_norm={self.use_layer_norm}"
        )


class EnsembleGatedMLPBlock(nn.Module):
    """GLU/SwiGLU-style gated MLP layer bloc for ensemble models (RLRP-768).

    ``y = x + W_o( act(W_g u) * (W_v u) )`` with ``u = norm(dropout(x))``.

    LEGAL REGIONS: ``encoder`` / ``hidden`` only (multiplicative gating rescales
    the features, which is illegal on a probability-learning component). The
    placement rule is enforced upstream by :func:`create_layer_bloc_factory`.

    Permanent public class. Introduced by action A2c of the RLRC dense layer bloc
    re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """

    def __init__(
        self,
        num_members: int,
        size: int,
        create_activation: Callable[[], nn.Module],
        dropout: float = 0.0,
        layer_norm: bool = False,
        expansion: float = 2.0,
        residual: bool = True,
        zero_init_last: bool = True,
    ):
        super().__init__()
        hidden = max(1, int(round(expansion * size)))
        self.num_members = num_members
        self.size = size
        self.hidden = hidden
        self.use_layer_norm = bool(layer_norm)
        self.use_residual = bool(residual)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(size) if layer_norm else nn.Identity()
        self.gate_proj = EnsembleLinearLayer(num_members, size, hidden)
        self.value_proj = EnsembleLinearLayer(num_members, size, hidden)
        self.out_proj = EnsembleLinearLayer(num_members, hidden, size)
        self.act = create_activation()
        self.zero_init_last = bool(zero_init_last)
        self.reset_last_layer_to_zero()

    def reset_last_layer_to_zero(self):
        """Zero ``out_proj`` -> exact identity at init (residual variant only).

        MUST be re-applied AFTER the model-level ``self.apply(truncated_normal_init)``,
        exactly like :meth:`EnsembleResidualBlock.reset_last_layer_to_zero`; see
        :func:`reset_layer_bloc_init_`.
        """
        if self.zero_init_last and self.use_residual:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.use_bias:
                nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        u = self.norm(self.dropout(x))
        y = self.out_proj(self.act(self.gate_proj(u)) * self.value_proj(u))
        return x + y if self.use_residual else y

    def _inner_layers(self):
        return (self.gate_proj, self.value_proj, self.out_proj)

    def set_elite(self, elite_models: Sequence[int]):
        for layer in self._inner_layers():
            layer.set_elite(elite_models)

    def toggle_use_only_elite(self):
        for layer in self._inner_layers():
            layer.toggle_use_only_elite()

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, size={self.size}, hidden={self.hidden}, "
            f"layer_norm={self.use_layer_norm}, residual={self.use_residual}"
        )


def create_residual_layer_factory(
    residual_form: str,
    ensemble_size: int,
    create_activation: Callable[[], nn.Module],
    dropout: float = 0.0,
    zero_init_last: bool = True,
):
    """Resolve a residual-layer builder for the requested ``residual_form``.

    Thin back-compat wrapper over :func:`create_layer_bloc_factory` (RLRP-768,
    action A6): the returned ``(kind, make_res)`` tuple contract is unchanged.
    It still raises ``AssertionError`` on an unknown ``residual_form`` to remain
    byte-for-byte back-compatible with the historical callers (the mixer heads
    and the existing tests).

    - ``"block"``  (``identity_v2``): ``make_res(size)`` returns an
      :class:`EnsembleResidualBlock`.
    - ``"legacy"`` (``legacy_preact_wrapped``): ``make_res(size)`` returns a
      single :class:`EnsembleResidualLinearLayer`; the caller MUST still wrap it
      with an external ``create_activation()``.
    """
    assert residual_form in RESIDUAL_FORMS, (
        f"residual_form must be one of {RESIDUAL_FORMS}, got {residual_form!r}"
    )
    return create_layer_bloc_factory(
        RESIDUAL_FORM_TO_LAYER_BLOC[residual_form],
        ensemble_size,
        create_activation,
        dropout=dropout,
        zero_init_last=zero_init_last,
        layer_norm=False,
        region="encoder",
    )


def create_dense_layer_factory(
    ensemble_size: int,
    create_activation: Callable[[], nn.Module],
    dropout: float = 0.0,
    layer_norm: bool = False,
):
    """Resolve a dense-bloc builder. Returns ``("block", make_bloc)``.

    Permanent public function. Introduced by action A4 of the RLRC dense layer
    bloc re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """
    return (
        "block",
        lambda size: EnsembleDenseBlock(
            ensemble_size, size, create_activation,
            dropout=dropout, layer_norm=layer_norm,
        ),
    )


def create_layer_bloc_factory(
    layer_bloc_type: Optional[str],
    ensemble_size: int,
    create_activation: Callable[[], nn.Module],
    dropout: float = 0.0,
    zero_init_last: bool = True,
    layer_norm: bool = False,
    region: str = "hidden",
):
    """Single dispatch point for every stacked layer bloc.

    ``layer_bloc_type=None`` resolves to the documented per-region default
    (:data:`DEFAULT_LAYER_BLOC_TYPE_BY_REGION`, §3.5): ``dense`` for the
    encoder/hidden stack, ``dense`` for every other region.

    :returns: ``(kind, make_bloc)`` with ``kind in {"block", "legacy"}`` -- the
        SAME contract as the historical :func:`create_residual_layer_factory`,
        so the call sites' ``nn.Sequential`` / trailing-activation logic is
        unchanged.
    :raises ValueError: on an unknown ``layer_bloc_type`` / ``region``, when
        ``layer_norm`` is requested on a probability-learning region, or when a
        gated bloc is requested outside encoder/hidden.

    Permanent public function. Introduced by action A5 of the RLRC dense layer
    bloc re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """
    if region not in LAYER_BLOC_REGIONS:
        raise ValueError(f"region must be one of {LAYER_BLOC_REGIONS}, got {region!r}")
    if layer_bloc_type is None:  # per-region default (RLRP-768 §3.5)
        layer_bloc_type = DEFAULT_LAYER_BLOC_TYPE_BY_REGION[region]
    if layer_bloc_type not in LAYER_BLOC_TYPES:
        raise ValueError(
            f"layer_bloc type must be one of {LAYER_BLOC_TYPES}, got {layer_bloc_type!r}"
        )
    if layer_norm and region not in LAYER_NORM_ALLOWED_REGIONS:
        raise ValueError(
            f"LayerNorm is forbidden in region {region!r}: it may only be enabled in "
            f"{LAYER_NORM_ALLOWED_REGIONS} (the `encoder` super-region: layers named "
            f"encoder/hidden ONLY; never in a decoder-layer-group component "
            f"(decoder stack, head, mixture, projection, any probabilistic head), "
            f"RLRP-768)."
        )
    if (
        layer_bloc_type in GATED_LAYER_BLOC_TYPES
        and region not in GATED_BLOC_ALLOWED_REGIONS
    ):
        raise ValueError(
            f"layer bloc {layer_bloc_type!r} is only legal in {GATED_BLOC_ALLOWED_REGIONS} "
            f"(encoder / hidden layers), got region={region!r}. Gated blocs "
            f"apply a learned multiplicative rescaling that corrupts the scale semantics "
            f"of a logvar / mixture-logit head (RLRP-768)."
        )
    if layer_bloc_type == "dense":
        return create_dense_layer_factory(
            ensemble_size, create_activation, dropout=dropout, layer_norm=layer_norm
        )
    # Phase 2 inserts the `gru` branch here (see the Phase-2 plan §4.2).
    if layer_bloc_type == "gated_mlp":
        return (
            "block",
            lambda size: EnsembleGatedMLPBlock(
                ensemble_size, size, create_activation,
                dropout=dropout, layer_norm=layer_norm, zero_init_last=zero_init_last,
            ),
        )
    if layer_bloc_type == "residual_legacy":
        if layer_norm:
            raise ValueError(
                "layer_norm is not supported by the deprecated 'residual_legacy' bloc "
                "(act(x + Wx + b) has no branch to normalise); use 'residual_v2' or 'dense'."
            )
        return ("legacy", lambda size: create_residual_linear_layer_(ensemble_size, size, size))
    return (
        "block",
        lambda size: EnsembleResidualBlock(
            ensemble_size, size, create_activation,
            dropout=dropout, zero_init_last=zero_init_last, layer_norm=layer_norm,
        ),
    )


def make_layer_bloc_seq(
    kind: str,
    make_bloc: Callable[[int], nn.Module],
    size: int,
    create_activation: Callable[[], nn.Module],
    dropout: float = 0.0,
) -> nn.Sequential:
    """Wrap a bloc into the ``nn.Sequential`` shape expected by the elite toggler.

    The toggler addresses ``layer[0]``, so index 0 MUST be the member-aware
    module. ``kind == "legacy"`` still needs the external dropout + trailing
    activation; ``kind == "block"`` (residual_v2, dense, gated_mlp) owns both.

    Permanent public function. Introduced by action A7 of the RLRC dense layer
    bloc re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """
    if kind == "legacy":
        return nn.Sequential(nn.Dropout(p=dropout), make_bloc(size), create_activation())
    return nn.Sequential(make_bloc(size))


def reset_layer_bloc_init_(module: nn.Module) -> None:
    """Re-apply the special init of every layer bloc AFTER ``truncated_normal_init``.

    ``self.apply(truncated_normal_init)`` re-draws every ``EnsembleLinearLayer``
    weight and unconditionally zeroes its bias, destroying any special init done
    in a bloc's ``__init__``. This pass restores it (residual + gated-MLP
    identity-at-init). ``EnsembleDenseBlock`` is intentionally NOT reset (a
    zero-init dense bloc would kill the signal instead of preserving identity).

    Permanent public function. Introduced by action A8 of the RLRC dense layer
    bloc re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """
    for sub_module in module.modules():
        if isinstance(sub_module, (EnsembleResidualBlock, EnsembleGatedMLPBlock)):
            sub_module.reset_last_layer_to_zero()
        # Phase 2 appends an `elif EnsembleGRUBloc: reset_gate_bias_()` clause here.


def zero_init_residual_blocks_(module: nn.Module) -> None:
    """Deprecated alias for :func:`reset_layer_bloc_init_` (kept for the existing
    5 call sites). Behaviour is identical for a model that only contains
    :class:`EnsembleResidualBlock`s, so the legacy-parity path stays bit-for-bit
    identical (RLRP-768 action A8).
    """
    reset_layer_bloc_init_(module)


def _as_plain_layer_bloc_dict(layer_bloc) -> Dict:
    """Normalise a ``layer_bloc`` config (dict / OmegaConf / None) to a plain dict."""
    if layer_bloc is None:
        return {}
    if isinstance(layer_bloc, (omegaconf.DictConfig, omegaconf.ListConfig)):
        layer_bloc = omegaconf.OmegaConf.to_container(layer_bloc, resolve=True)
    if not isinstance(layer_bloc, dict):
        raise ValueError(
            f"layer_bloc must be a mapping region->settings or None, got {type(layer_bloc)}"
        )
    return layer_bloc


def resolve_layer_bloc_settings(
    layer_bloc=None,
    residual_form: str = "identity_v2",
) -> Dict[str, Dict]:
    """Resolve the per-region layer-bloc settings (RLRP-768 §3.5 / §3.5.b).

    Precedence per region:
      1. ``layer_bloc[region].type`` wins if given (non-null);
      2. else the deprecated ``residual_form`` alias selects the residual VARIANT,
         but ONLY for regions whose per-region default is a residual bloc
         (``encoder`` / ``hidden``); it never re-residualises a decoder-group
         region (otherwise the shipped ``residual_form: identity_v2`` configs
         would silently cancel the new per-region ``dense`` defaults);
      3. else ``type`` stays ``None`` -> the factory resolves the per-region
         default (:data:`DEFAULT_LAYER_BLOC_TYPE_BY_REGION`).

    :returns: ``{region: {"type": Optional[str], "layer_norm": bool,
        "dropout": Optional[float]}}`` for every region in
        :data:`LAYER_BLOC_REGIONS`.

    Permanent public function. Introduced by action B2 of the RLRC dense layer
    bloc re-introduction `.junie` plan
    (feat_reintroduce_dense_layer_bloc_plan_RLRP-768_20260809.md).
    """
    lb = _as_plain_layer_bloc_dict(layer_bloc)
    unknown = set(lb) - set(LAYER_BLOC_REGIONS)
    if unknown:
        raise ValueError(
            f"layer_bloc has unknown region(s) {sorted(unknown)}; "
            f"valid regions are {LAYER_BLOC_REGIONS} (RLRP-768)"
        )
    alias_type = RESIDUAL_FORM_TO_LAYER_BLOC.get(residual_form)
    resolved: Dict[str, Dict] = {}
    for region in LAYER_BLOC_REGIONS:
        entry = lb.get(region) or {}
        if not isinstance(entry, dict):
            raise ValueError(
                f"layer_bloc[{region!r}] must be a mapping, got {type(entry)}"
            )
        unknown_keys = set(entry) - {"type", "layer_norm", "dropout"}
        if unknown_keys:
            raise ValueError(
                f"layer_bloc[{region!r}] has unknown key(s) {sorted(unknown_keys)}; "
                f"valid keys are {{'type', 'layer_norm', 'dropout'}} (RLRP-768)"
            )
        bloc_type = entry.get("type", None)
        layer_norm = bool(entry.get("layer_norm", False))
        dropout = entry.get("dropout", None)
        if (
            bloc_type is None
            and alias_type is not None
            and DEFAULT_LAYER_BLOC_TYPE_BY_REGION[region] == "residual_v2"
        ):
            bloc_type = alias_type
        resolved[region] = {
            "type": bloc_type,
            "layer_norm": layer_norm,
            "dropout": dropout,
        }
    return resolved


def layer_bloc_signature_from_resolved(resolved: Dict[str, Dict]) -> str:
    """Build a stable checkpoint-topology signature from resolved settings.

    e.g. ``"encoder=residual_v2|hidden=residual_v2|decoder=dense|ms_head=dense|..."``.
    Used by the checkpoint guard so a bloc-topology mismatch fails loud with a
    descriptive error instead of a cryptic ``load_state_dict`` key error
    (RLRP-768 action B5).
    """
    parts = []
    for region in LAYER_BLOC_REGIONS:
        settings = resolved[region]
        bloc_type = settings["type"] or DEFAULT_LAYER_BLOC_TYPE_BY_REGION[region]
        ln = "+ln" if settings["layer_norm"] else ""
        parts.append(f"{region}={bloc_type}{ln}")
    return "|".join(parts)

def create_linear_layer_(ensemble_size: int, l_in: int, l_out: int) -> nn.Module:
    return EnsembleLinearLayer(ensemble_size, l_in, l_out)

def create_activation_(
    activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]],
) -> nn.Module:
    if activation_fn_cfg is None:
        activation_func = nn.ReLU()
    else:
        # Handle the case where activation_fn_cfg is a dict
        cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
        activation_func = hydra.utils.instantiate(cfg, _recursive_=False)
    return activation_func


class LogvarBoundLayer(nn.Module):
    def __init__(
        self,
        out_size: int,
        learn_logvar_bounds: bool,
        bound_min_init: torch.Tensor = torch.tensor(-10),
        bound_max_init: torch.Tensor = torch.tensor(0.5),
        grad_clip: Optional[float] = None,
        bound_loss_coeff: float = 0.01,
    ):
        """Soft (PETS/Chua) logvar bound layer.

        ``bound_loss_coeff`` (default ``0.01``) is the coefficient of the soft logvar-bound penalty
        ``coeff * (logvar_max.sum() - logvar_min.sum())`` added by ``bound_losses``. The value
        ``0.01`` is inherited VERBATIM from PETS / Chua et al. (2018) via the upstream ``mbrl-lib``
        ``GaussianMLP._nll_loss`` (``+ 0.01 * (max_logvar.sum() - min_logvar.sum())``). It is a
        small EMPIRICAL regularisation weight (NOT a likelihood-derived constant): it gently pulls
        the learnable soft bounds inward (``max`` down / ``min`` up) so they stay tight around the
        data instead of drifting to their init extremes on dimensions whose data does not push
        back. Being a free coefficient on the bound PARAMETERS (not on any density normaliser), it
        is distribution-agnostic and applies unchanged to gaussian and laplace heads; only its
        empirically-optimal magnitude is loosely coupled to the (mean-reduced) NLL scale, so it is
        exposed here as a tunable param rather than a hard-coded literal.

        ``grad_clip`` (RLRP-718 follow-up, robustness lever): when set to a positive float and the
        bounds are learnable, the gradient flowing into ``logvar_min`` / ``logvar_max`` is clamped
        element-wise to ``[-grad_clip, +grad_clip]``. This is a defensive guard against the large
        bound-parameter gradients observed when the bounds sit in a probabilistic multi-step NLL
        graph (e.g. the MS+SS true-mixture path), where the per-feature mixture NLL can push the
        bound parameters with magnitudes ~1000x the MS-only baseline. It DEFAULTS to ``None``
        (disabled) so existing training dynamics and checkpoints are byte-for-byte unchanged; it is
        an opt-in lever to be enabled per experiment when bound-gradient spikes need taming, NOT a
        substitute for the *penalise-iff-applied* anchoring that fixes the actual runaway.
        """
        super().__init__()

        self.logvar_min = nn.Parameter(
            bound_min_init
            * torch.ones(
                (1, out_size), device=bound_min_init.device, dtype=bound_min_init.dtype
            ),
            requires_grad=learn_logvar_bounds,
        )
        self.logvar_max = nn.Parameter(
            bound_max_init
            * torch.ones(
                (1, out_size), device=bound_max_init.device, dtype=bound_max_init.dtype
            ),
            requires_grad=learn_logvar_bounds,
        )

        self.bound_loss_coeff = bound_loss_coeff
        self.grad_clip = grad_clip
        if learn_logvar_bounds and grad_clip is not None:
            assert grad_clip > 0.0, f"grad_clip must be > 0, got {grad_clip}"
            _c = float(grad_clip)
            self.logvar_min.register_hook(lambda g: g.clamp(min=-_c, max=_c))
            self.logvar_max.register_hook(lambda g: g.clamp(min=-_c, max=_c))

    def forward(self, x):
        logvar = self.logvar_max - F.softplus(self.logvar_max - x)
        logvar = self.logvar_min + F.softplus(logvar - self.logvar_min)
        return logvar

    def bound_losses(
        self,
        nll_losses: torch.Tensor,
        bound_ms_adapter: Optional[Union[
            Callable,
            MultistepObservationToSinglestepObservationAdapter,
        ]] = None, # (NICE TO HAVE) ToDo: param name is missleading right now. Rename to 'bound_ms_mask' or 'bound_ms_adapter'
    ) -> torch.Tensor:
        if isinstance(
            bound_ms_adapter,
            (Callable, MultistepObservationToSinglestepObservationAdapter),
        ):
            nll_losses = nll_losses + self.bound_loss_coeff * (
                    bound_ms_adapter(self.logvar_max).sum()
                    - bound_ms_adapter(self.logvar_min).sum()
            )
        else:
            nll_losses = nll_losses + self.bound_loss_coeff * (
                self.logvar_max.sum() - self.logvar_min.sum()
            )

        return nll_losses


def create_logvar_bound_layer(
    l_out: int,
    learn_logvar_bounds: bool,
    bound_min_init: torch.Tensor = torch.tensor(-10),
    bound_max_init: torch.Tensor = torch.tensor(0.5),
    grad_clip: Optional[float] = None,
    bound_loss_coeff: float = 0.01,
) -> nn.Module:
    return LogvarBoundLayer(
        l_out,
        learn_logvar_bounds,
        bound_min_init,
        bound_max_init,
        grad_clip=grad_clip,
        bound_loss_coeff=bound_loss_coeff,
    )


def logistic_distribution(a: torch.Tensor, b: torch.Tensor):
    # Source: https://pytorch.org/docs/stable/distributions.html#transformeddistribution
    base_distribution = dist.Uniform(torch.zeros_like(a), torch.ones_like(b))
    transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=a, scale=b)]
    logistic = dist.TransformedDistribution(base_distribution, transforms)
    return logistic


class EnsembleLSTMLayer(nn.Module):
    # (CRITICAL) ToDo: RLRP-238 feat: implement baseline system-dynamic model architecture
    """Efficient gru layer for ensemble models."""

    def __init__(
        self,
        num_members: int,
        in_size: int,
        out_size: int,
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size

        self.lstm = nn.LSTM(in_size, out_size, 1, batch_first=True)

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.out_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.out_size)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


#: A6 (RLRP-783): NLL constants hoisted out of the hot path. Introduced by action ``A6`` of the
#: RLRC MTM-Pro models code optimization `.junie` plan
#: (``perf_RLRP-783_mtm_pro_models_code_optimization_plan_20260827.md``). ``_GAUSSIAN_NLL_CONSTANT``
#: is the identical Python float previously computed inline in the ``gaussian_manual`` branch
#: (bit-exact). ``_LOGISTIC_SCALE_FACTOR`` replaces the per-call ``np.sqrt(3) / torch.pi``.
_GAUSSIAN_NLL_CONSTANT = 0.5 * math.log(2 * math.pi)
_LOGISTIC_SCALE_FACTOR = math.sqrt(3) / math.pi


def negative_loglikelihood_loss__explicit(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    distribution_name: str = "gaussian_manual",
) -> torch.Tensor:
    # A6 (RLRP-783): ``pred_std = sqrt(exp(logvar))`` used to be computed UNCONDITIONALLY here but
    # is consumed ONLY by the ``logistic_manual`` branch, so every ``gaussian_manual`` /
    # ``gaussian_wo_cons`` / ``laplace`` NLL call (the hot path) paid a full-tensor ``exp`` AND
    # ``sqrt`` for nothing. It is now computed inside the ``logistic_manual`` branch only. The
    # Gaussian / Laplace paths are BIT-EXACT (only a dead computation is removed).
    if distribution_name == "laplace":
        l1 = F.l1_loss(pred_mean, target, reduction="none")
        inv_var = (-pred_logvar).exp()
        nll_losses = 2 * l1 * torch.sqrt(inv_var) + 0.5 * pred_logvar

    elif distribution_name == "gaussian_manual" or distribution_name == "gaussian_wo_cons":
        l2 = F.mse_loss(pred_mean, target, reduction="none")
        inv_var = (-pred_logvar).exp()
        nll_losses = 0.5 * (pred_logvar + l2 * inv_var)

        if distribution_name == "gaussian_manual":
            nll_losses += _GAUSSIAN_NLL_CONSTANT

    elif distribution_name == "logistic_manual":
        # ``exp(0.5 * logvar) == sqrt(exp(logvar))`` with one transcendental instead of two.
        scale = torch.exp(0.5 * pred_logvar) * _LOGISTIC_SCALE_FACTOR
        U = (target - pred_mean) / scale
        # ``2 * softplus(-U) == 2 * log(1 + exp(-U))``, without the ``exp`` overflow for U << 0.
        nll_losses = torch.log(scale) + U + 2 * F.softplus(-U)

    else:
        raise NotImplementedError(
            f"Distribution {distribution_name} not implemented yet! Choose between: "
            "gaussian, student, laplace, gaussian_manual or logistic_manual"
        )
    return nll_losses
