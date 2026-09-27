# coding=utf-8
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from mbrl.models import EnsembleLinearLayer
from torch import Tensor, nn as nn
from torch import distributions as dist

from tools.multistep_tools.models.exponential_family_mlp_utils import (
    create_residual_layer_factory,
)


def temporal_mixture_head_leaf_parameter_names(
    mixture_module: nn.Module,
) -> list:
    """Auto-discover the softmax-logit-producing parameters of a temporal-mixture module.

    Returns the parameter names (RELATIVE to ``mixture_module``, i.e. as returned by
    ``mixture_module.named_parameters()``) that should be logged under the tensorboard
    ``Model parameters head leaf`` tag.

    Rationale (RLRP-735): the mixer family now includes COMPOSED variants
    (:class:`AdditiveIndexAndHistoryDependentTemporalMixtureWeighs`,
    :class:`AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs`,
    :class:`InputAndIndexDependentTemporalMixtureWeighs`) whose logit-producing leaves are NESTED
    (e.g. ``index.mixture_logits``, ``input_head.mixture_logits.weight``) or additive
    (``index_base``). A single hard-coded name can no longer select them. Instead we rely on the
    naming CONVENTION shared by every variant — the logit leaf is always named ``mixture_logits``
    (an ``nn.Parameter``, or an ``(Ensemble)Linear``'s ``.weight``) and any additive index base is
    named ``index_base`` — so new composed variants are picked up automatically with no callback
    edit, as long as they keep the convention.

    Bias terms are intentionally excluded (only distribution-shaping weights are monitored).

    :param mixture_module: The temporal-mixture-weights module (``model.temporal_mixture_weights``).
    :return: A list of relative parameter names to monitor (possibly empty).
    """
    monitored: list = []
    for name, _ in mixture_module.named_parameters():
        if (
            name == "mixture_logits"  # bare nn.Parameter (pure index mixer)
            or name.endswith(".mixture_logits")  # nested nn.Parameter (e.g. index.mixture_logits)
            or name.endswith("mixture_logits.weight")  # (Ensemble)Linear logit head weight
            or name.endswith("index_base")  # additive zero-init index base
        ):
            monitored.append(name)
    return monitored


class _MixerTensorboardParamsMixin:
    """Mixin exposing a MERGED tensorboard view of a mixer's learnable mixing parameters.

    Every mixer builds its per-``(i, k)`` logits by (additively) COMPOSING one or more learnable
    terms in :meth:`forward` / ``logits_at_i`` — e.g. ``φ = γ + ψ + ρ`` for
    :class:`AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs`. This mixin mirrors that
    composition pattern for visualisation: :meth:`merged_mixer_parameter_vector` concatenates the
    flattened composing parameters into a SINGLE vector so the tensorboard callback can render one
    combined histogram per mixer (instead of one histogram per leaf), regardless of how many terms
    the concrete implementation composes.

    The set of composing parameters is auto-discovered from the module subtree via
    :func:`temporal_mixture_head_leaf_parameter_names` (naming convention: ``mixture_logits`` /
    ``index_base``), so composed variants are covered automatically. Toggle the whole feature with
    the class attribute :attr:`show_mixer_composed_params`.
    """

    #: Class-level toggle for the merged learnable-parameter histogram (see the callback).
    show_mixer_composed_params: bool = True

    @property
    def tensorboard_merged_mixer_params(self) -> list:
        """Relative names of the learnable parameters composing the mixing logits (per forward)."""
        return temporal_mixture_head_leaf_parameter_names(self)

    def merged_mixer_parameter_vector(self) -> Optional[Tensor]:
        """Return a single 1-D tensor concatenating the flattened composing parameters.

        The concatenation order follows :attr:`tensorboard_merged_mixer_params` (i.e. the additive
        composition terms of the concrete mixer). Returns ``None`` when the mixer exposes no
        composing parameter (defensive; should not happen for the shipped variants).
        """
        params = dict(self.named_parameters())
        chunks = [
            params[name].detach().reshape(-1)
            for name in self.tensorboard_merged_mixer_params
            if name in params
        ]
        if not chunks:
            return None
        return torch.cat(chunks)


class TemporalMixtureWeighs(_MixerTensorboardParamsMixin, nn.Module):
    """Learnable temporal mixture weights for autoregressive horizon blending.

    Stores unconstrained logits as the learnable parameter. The logits are passed
    directly to ``dist.Categorical(logits=...)`` which internally applies
    log-softmax, so they remain numerically safe regardless of gradient updates.
    """

    def __init__(
        self,
        unrol_len: int,
        horizon_len: int,
        ensemble_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.U = unrol_len
        self.F = horizon_len
        self.E = ensemble_size

        # Store as unconstrained logits (zeros → uniform initial distribution)
        self.mixture_logits = nn.Parameter(
            torch.zeros((self.E, self.F, self.F), dtype=dtype),
            requires_grad=True,
        )

    def _normalize(self, i: int) -> torch.Tensor:
        logits_i = self.logits_at_i(i)
        normalized_weights_i = torch.softmax(logits_i, dim=-1)
        return normalized_weights_i  # (E x min(i,U))

    def logits_at_i(self, i: int) -> Tensor:
        logits_i = self.mixture_logits[:, max(i - self.U + 1, 0) : i + 1, i]
        return logits_i

    def forward(self, horizon_index: int, log_prob: bool = True) -> torch.Tensor:
        """
        Return the unnormalized logits or the normalized probabilities for a specific horizon index.

        Output shape: (E x min(i,U))

        :param horizon_index: Index corresponding to the horizon for which weights are computed.
        :param log_prob: Flag indicating whether to return unnormalized logits (for use with
            ``dist.Categorical(logits=...)``). When False, returns softmax-normalized probabilities.
        :return: A tensor containing either the logits or normalized weights.
        """
        if log_prob:
            return self.logits_at_i(horizon_index)
        else:
            normalized_weights_i = self._normalize(horizon_index)
            return normalized_weights_i


class InputDependentTemporalMixtureWeighs(_MixerTensorboardParamsMixin, nn.Module):
    """Input-dependent learnable temporal mixture weights for autoregressive horizon blending.

    Unlike :class:`TemporalMixtureWeighs` (pure step-indexed ``nn.Parameter`` gammas), this
    variant derives the per-step mixing logits from the per-step component statistics tokens
    ``cat([pred_mean_i, pred_logvar_i])`` via an MLP. This gives an input-dependent degree of
    freedom so the ``mixing_only`` backprop scope has a live (data-dependent) gradient path into
    the mixing function, mirroring the transformer mixers but without attention.

    HEAD ARCHITECTURE (RLRP-735 refactor) — mirrors
    :class:`IndexAndHistoryDependentTemporalMixtureWeighs`: an ENSEMBLE-AWARE ``in_proj``
    (:class:`EnsembleLinearLayer`) + activation, a trunk of ``num_layers`` identity-preserving
    :class:`EnsembleResidualBlock`\\ s (built via
    ``create_residual_layer_factory("identity_v2", ...)``, block-owned ``dropout``), and a
    ZERO-initialised ``out_proj`` (named ``mixture_logits``) → ``R^{E×B'×1}``. The zero-init
    ``out_proj`` (plus the identity residual blocks) makes the per-step logit ``ρ = 0`` at init
    ⇒ the softmax over the valid window is UNIFORM (a clean warm start, matching the P3 additive
    design). This REPLACES the historical member-shared two-``nn.Linear`` MLP (which was NOT
    ensemble-aware, NOT zero-init, and had no ``num_layers`` / ``dropout``); as a consequence
    old ``input_dependent`` checkpoints no longer load (accepted, RLRP-735 P-plan A.6) and the
    init distribution is now uniform.

    Per-member weights + elite forwarding (``_elite_layers`` list + :meth:`set_elite` /
    :meth:`toggle_use_only_elite`) match the sibling mixer. NOTE: the model currently never
    applies elite selection to the mixer (family-wide elite==all), so these hooks are INERT and
    kept only for API symmetry.

    Call convention matches the transformer mixers: ``forward(tokens, log_prob=True)`` where
    ``tokens`` has shape ``(E, B, S, inputdim)`` and the output logits have shape ``(E, B, S)``.
    """

    def __init__(
        self,
        inputdim: int,
        ensemble_size: int,
        head_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation_factory: Optional[Callable[[], nn.Module]] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        # `hidden_dim` is a DEPRECATED alias for `head_size` (kept so old callers/configs that
        # still pass `hidden_dim` do not raise, RLRP-735 P-plan A.6).
        if hidden_dim is not None:
            head_size = hidden_dim
        self.dtype = dtype
        self.device = device
        self.E = ensemble_size
        self.inputdim = inputdim
        self.head_size = head_size
        self.num_layers = max(int(num_layers), 0)
        self.dropout = float(dropout)

        if activation_factory is None:
            activation_factory = lambda: nn.ReLU()

        # Ensemble-aware residual MLP head (mirrors `IndexAndHistoryDependentTemporalMixtureWeighs`).
        # The `EnsembleLinearLayer` / `EnsembleResidualBlock` constructors take NO `device` arg
        # (unlike `nn.Linear`); the caller moves the module with `.to(self.device)` after build.
        self.in_proj = EnsembleLinearLayer(ensemble_size, inputdim, head_size)
        self.in_act = activation_factory()
        _, make_res = create_residual_layer_factory(
            "identity_v2",
            ensemble_size,
            activation_factory,
            dropout=self.dropout,
            zero_init_last=True,
        )
        self.trunk = nn.Sequential(
            *[make_res(head_size) for _ in range(self.num_layers)]
        )
        # Named `mixture_logits` to enable downstream tensorboard logic.
        self.mixture_logits = EnsembleLinearLayer(ensemble_size, head_size, 1)
        # Zero-init the head so ρ = 0 at init ⇒ uniform mixture warm start.
        self._zero_init_out_proj()

        # Honour the requested dtype: `EnsembleLinearLayer` / `EnsembleResidualBlock` take NO
        # `dtype` arg (unlike `nn.Linear`), so cast the whole head here (parity with the sibling
        # index+history mixer, so a `model_use_double_precision=True` model keeps a double head).
        self.to(dtype=dtype)

        # Flat list of the top-level ensemble layers/blocks for elite forwarding (each entry is
        # an `EnsembleLinearLayer` OR an `EnsembleResidualBlock` that forwards the toggle to its
        # own two inner layers); iterating this list toggles every member-aware layer EXACTLY
        # once (iterating `self.modules()` would double-toggle the blocks' inner layers).
        self._elite_layers: list = [self.in_proj, *self.trunk, self.mixture_logits]

    def _zero_init_out_proj(self) -> None:
        nn.init.zeros_(self.mixture_logits.weight)
        if getattr(self.mixture_logits, "use_bias", False):
            nn.init.zeros_(self.mixture_logits.bias)

    def _normalize(self, logits_i) -> torch.Tensor:
        normalized_weights_i = torch.softmax(logits_i, dim=-1)
        return normalized_weights_i  # (E x B x S)

    def forward(self, tokens, log_prob: bool = True):
        """Map per-step component tokens to per-step mixing logits.

        :param tokens: A 4-D tensor of shape (E, B, S, inputdim).
        :param log_prob: If True return unnormalized logits, else softmax-normalized weights.
        :return: A tensor of shape (E, B, S).
        """
        tokens = tokens.to(dtype=self.mixture_logits.weight.dtype)
        lead = tokens.shape[:-1]  # (E, B, S)
        # Flatten the batch/step dims into the per-member batch axis (ensemble dim kept as axis
        # 0) so the `EnsembleLinearLayer` `x.matmul(weight (E, in, out))` contract holds.
        x = tokens.reshape(lead[0], -1, tokens.shape[-1])  # (E, B*S, inputdim)
        h = self.trunk(self.in_act(self.in_proj(x)))
        alpha_logit = self.mixture_logits(h).reshape(*lead)  # (E, B, S)
        if log_prob:
            return alpha_logit
        else:
            return self._normalize(alpha_logit)

    def set_elite(self, elite_models) -> None:
        for layer in self._elite_layers:
            layer.set_elite(elite_models)

    def toggle_use_only_elite(self) -> None:
        for layer in self._elite_layers:
            layer.toggle_use_only_elite()


class IndexAndHistoryDependentTemporalMixtureWeighs(_MixerTensorboardParamsMixin, nn.Module):
    """Index + input-history conditioned temporal mixture weights (RLRP-735).

    A THIRD mixer category alongside :class:`TemporalMixtureWeighs` (pure step-indexed
    ``nn.Parameter`` gammas, depending on ``(i, k)`` only) and
    :class:`InputDependentTemporalMixtureWeighs` (logits derived from the per-step *component
    prediction* tokens). Here the per-``(i, k)`` logits are produced by an ensemble-aware
    residual MLP conditioned on the model's *input history* realisation ``s_τ^H`` (ticket
    eq. 6.2 / 8.2)::

        α_{k,φ}(s_τ^H, i) := softmax_k( φ_i^k(s_τ^H) )   over k ∈ [max(i-U+1, 1), i]
                 with     φ_i^k(s_τ^H) = ψ_i^k(s_τ^H)      # PURE history head (no γ base, D3-b)

    ``s_τ^H`` is the raw ``model_input`` realisation (means-only, flat width
    ``D_h = (O+A)·history_len``) captured ONCE at forecast entry and held CONSTANT across the
    ``i``..``F`` autoregressive steps (operator decision D2). Consequently the head output
    ``ψ ∈ R^{E×B×F×F}`` is also constant across ``i``; :meth:`compute_psi` builds it once per
    forward and :meth:`forward` / :meth:`logits_at_i` only slice the valid ``k``-window (an
    optional precomputed ``psi`` avoids re-running the residual head ``F`` times in the hot AR
    loop, see D3 ψ-caching).

    HEAD ARCHITECTURE — mirrors :class:`_ProjectionNetworkHead`: an ensemble-aware
    ``in_proj`` (:class:`EnsembleLinearLayer`) + activation, a trunk of ``num_layers``
    identity-preserving :class:`EnsembleResidualBlock`\\ s (built via
    ``create_residual_layer_factory("identity_v2", ...)``, block-owned ``dropout``), and a
    zero-initialised ``out_proj`` → ``R^{E×B×F·F}``. The zero-init ``out_proj`` (plus the
    identity residual blocks) makes ``ψ = 0`` at init ⇒ the softmax over the valid window is
    UNIFORM (a clean warm start even without an index prior).

    Per-member weights + elite forwarding (``_elite_layers`` list + :meth:`set_elite` /
    :meth:`toggle_use_only_elite`) match :class:`_ProjectionNetworkHead` exactly.

    Call convention: per-batch, like :class:`InputDependentTemporalMixtureWeighs`.
    ``forward(history_token, horizon_index, log_prob=True, psi=None)`` returns the ``k``-window
    slice of shape ``(E, B, min(i+1, U))``. NOTE: unlike the index base which stores logits as
    ``[E, k, i]``, ``ψ`` is laid out ``[E, B, i, k]`` (same window bounds/length, different
    axis order — the class is self-consistent and matched to the ``MixtureSameFamily``
    component count, NOT a literal reuse of the index slice).
    """

    def __init__(
        self,
        unrol_len: int,
        horizon_len: int,
        ensemble_size: int,
        history_token_dim: int,
        head_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation_factory: Optional[Callable[[], nn.Module]] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.U = unrol_len
        self.F = horizon_len
        self.E = ensemble_size
        self.D_h = history_token_dim
        self.head_size = head_size
        self.num_layers = max(int(num_layers), 0)
        self.dropout = float(dropout)

        if activation_factory is None:
            activation_factory = lambda: nn.ReLU()

        # Ensemble-aware residual MLP history head (mirrors `_ProjectionNetworkHead`). The
        # `EnsembleLinearLayer` / `EnsembleResidualBlock` constructors take NO `device` arg
        # (unlike `nn.Linear`); the caller moves the module with `.to(self.device)` after build.
        self.in_proj = EnsembleLinearLayer(ensemble_size, history_token_dim, head_size)
        self.in_act = activation_factory()
        _, make_res = create_residual_layer_factory(
            "identity_v2",
            ensemble_size,
            activation_factory,
            dropout=self.dropout,
            zero_init_last=True,
        )
        self.trunk = nn.Sequential(
            *[make_res(head_size) for _ in range(self.num_layers)]
        )
        # Named `mixture_logits` to align with the tensorboard logic of the other mixers.
        self.mixture_logits = EnsembleLinearLayer(
            ensemble_size, head_size, self.F * self.F
        )
        # Zero-init the head so ψ = 0 at init ⇒ uniform mixture warm start.
        self._zero_init_out_proj()

        # Honour the requested dtype: `EnsembleLinearLayer` / `EnsembleResidualBlock` take NO
        # `dtype` arg (unlike `nn.Linear`), so cast the whole head here. Without this, a
        # `model_use_double_precision=True` model (cast to double INSIDE `super().__init__`,
        # BEFORE this mixer is built) would silently keep a float32 mixer head — the sibling
        # mixers (`TemporalMixtureWeighs`, `InputDependentTemporalMixtureWeighs`) pass `dtype`
        # into their parameters directly, so this keeps the three kinds consistent.
        self.to(dtype=dtype)

        # Flat list of the top-level ensemble layers/blocks for elite forwarding (each entry is
        # an `EnsembleLinearLayer` OR an `EnsembleResidualBlock` that forwards the toggle to its
        # own two inner layers); iterating this list toggles every member-aware layer EXACTLY
        # once (iterating `self.modules()` would double-toggle the blocks' inner layers).
        self._elite_layers: list = [self.in_proj, *self.trunk, self.mixture_logits]

    def _zero_init_out_proj(self) -> None:
        nn.init.zeros_(self.mixture_logits.weight)
        if getattr(self.mixture_logits, "use_bias", False):
            nn.init.zeros_(self.mixture_logits.bias)

    def _normalize_token(self, history_token: Tensor) -> Tensor:
        """Normalise the history token to the ``(E, *, D_h)`` `EnsembleLinearLayer` contract.

        Training passes ``(E, B, D_h)`` (ensemble axis already present); eval/``deploy`` passes
        a 2-D ``(B, D_h)`` realisation (no ensemble axis), which is broadcast over the ensemble
        dim so the per-member head output aligns with the ``(E, B, ...)`` component tensors.
        """
        if history_token.dim() == 2:
            history_token = (
                history_token.unsqueeze(0).expand(self.E, -1, -1).contiguous()
            )
        return history_token

    def compute_psi(self, history_token: Tensor) -> Tensor:
        """Run the residual head once, returning ψ of shape ``(E, B, F, F)`` (``[E, B, i, k]``).

        ψ is constant across the ``i``..``F`` AR steps (``s_τ^H`` is frozen, D2), so this is
        computed ONCE per forward and sliced per step by :meth:`logits_at_i`.
        """
        token = self._normalize_token(history_token)
        token = token.to(dtype=self.mixture_logits.weight.dtype)
        lead = token.shape[:-1]  # (E, B) or (E, B', ...)
        # Flatten any extra leading dims into the per-member batch axis (ensemble dim kept as
        # axis 0) so the `EnsembleLinearLayer` `x.matmul(weight (E, in, out))` contract holds.
        x = token.reshape(lead[0], -1, token.shape[-1])  # (E, B', D_h)
        h = self.trunk(self.in_act(self.in_proj(x)))
        logits = self.mixture_logits(h)  # (E, B', F*F)
        psi = logits.reshape(*lead, self.F, self.F)  # (E, *, i, k)
        return psi

    def logits_at_i(
        self, history_token: Tensor, i: int, psi: Optional[Tensor] = None
    ) -> Tensor:
        if psi is None:
            psi = self.compute_psi(history_token)
        logits_i = psi[..., i, max(i - self.U + 1, 0) : i + 1]  # (E, B, min(i+1, U))
        return logits_i

    def _normalize(self, logits_i: Tensor) -> Tensor:
        normalized_weights_i = torch.softmax(logits_i, dim=-1)
        return normalized_weights_i  # (E x B x min(i+1, U))

    def forward(
        self,
        history_token: Tensor,
        horizon_index: int,
        log_prob: bool = True,
        psi: Optional[Tensor] = None,
        input_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the per-``(i, k)`` logits (or softmax-normalised weights) at horizon ``i``.

        Output shape: ``(E, B, min(i+1, U))`` (per-batch, like the input-dependent mixer).

        :param history_token: The frozen ``s_τ^H`` realisation, ``(E, B, D_h)`` (train) or
            ``(B, D_h)`` (eval/deploy — broadcast over the ensemble axis).
        :param horizon_index: The horizon step ``i``.
        :param log_prob: When True return unnormalised logits (for ``dist.Categorical(logits=)``);
            when False return softmax-normalised weights.
        :param psi: Optional precomputed head output (see :meth:`compute_psi`); avoids re-running
            the residual head at every AR step.
        :param input_tokens: Ignored here (no-op). Accepted so the model's single history-mixer
            routing branch can pass the per-step ``cat([mean, logvar])`` tokens uniformly across
            all history kinds; only the combined
            :class:`AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs` consumes them.
        """
        logits_i = self.logits_at_i(history_token, horizon_index, psi=psi)
        if log_prob:
            return logits_i
        else:
            return self._normalize(logits_i)

    def set_elite(self, elite_models) -> None:
        for layer in self._elite_layers:
            layer.set_elite(elite_models)

    def toggle_use_only_elite(self) -> None:
        for layer in self._elite_layers:
            layer.toggle_use_only_elite()


class AdditiveIndexAndHistoryDependentTemporalMixtureWeighs(
    IndexAndHistoryDependentTemporalMixtureWeighs
):
    """Additive index-base + input-history conditioned mixture weights (RLRP-735 P3).

    A variant of :class:`IndexAndHistoryDependentTemporalMixtureWeighs` that revisits the
    operator decision D3-b (PURE history head) by adding an ADDITIVE per-``(i, k)`` index base
    ``γ`` on top of the history residual (ticket eq. 6.2 / 8.2 with the ``γ`` term restored)::

        α_{k,φ}(s_τ^H, i) := softmax_k( φ_i^k(s_τ^H) )   over k ∈ [max(i-U+1, 1), i]
                 with     φ_i^k(s_τ^H) = γ[i, k] + ψ_i^k(s_τ^H)   # index base + history residual

    ``γ`` is a batch-agnostic step-indexed ``nn.Parameter`` of shape ``(E, F, F)`` (laid out
    ``[E, i, k]`` to match ψ's ``[E, B, i, k]`` window slice), mirroring the pure-index
    :class:`TemporalMixtureWeighs` gammas. It is ZERO-initialised, so — together with the
    zero-init history head ψ — ``φ = 0`` at init ⇒ the softmax over the valid window is UNIFORM,
    keeping BYTE-FOR-BYTE init parity with the pure-history mixer. The index base gives the
    mixer a data-independent per-step prior it can fall back on if the history conditioning is
    hard to train, at zero extra risk at init.

    Everything else (frozen ``s_τ^H`` token, once-per-forward ψ caching, per-batch call
    convention, elite forwarding of the head layers) is inherited unchanged; only
    :meth:`logits_at_i` is overridden to add the index-base window slice. NOTE: ``γ`` is a plain
    per-member ``nn.Parameter`` (batch-agnostic), so it is NOT part of ``_elite_layers`` — it is
    broadcast over the batch axis and always uses its full ``E`` rows, exactly like the pure-index
    mixer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-(i, k) index base gamma (batch-agnostic), zero-init ⇒ φ = γ + ψ = 0 at init ⇒
        # uniform window softmax (parity with the pure-history mixer). Created AFTER
        # `super().__init__` (which already cast the head to `self.dtype`), so build it directly
        # in the requested dtype; the model moves the whole module to its device afterwards.
        self.index_base = nn.Parameter(
            torch.zeros((self.E, self.F, self.F), dtype=self.dtype),
            requires_grad=True,
        )  # layout [E, i, k]

    def logits_at_i(
        self, history_token: Tensor, i: int, psi: Optional[Tensor] = None
    ) -> Tensor:
        # History residual window slice, (E, B, min(i+1, U)).
        logits_i = super().logits_at_i(history_token, i, psi=psi)
        # Additive index-base window slice, (E, min(i+1, U)); unsqueeze the batch axis so it
        # broadcasts over B against the per-batch history logits.
        base_i = self.index_base[:, i, max(i - self.U + 1, 0) : i + 1]
        return logits_i + base_i.unsqueeze(1)


class AdditiveIndexInputAndHistoryDependentTemporalMixtureWeighs(
    AdditiveIndexAndHistoryDependentTemporalMixtureWeighs
):
    """Index + input + input-history conditioned mixture weights (RLRP-735 P3 combined).

    Combines :class:`InputDependentTemporalMixtureWeighs` (per-step component tokens ``ρ``) with
    :class:`AdditiveIndexAndHistoryDependentTemporalMixtureWeighs` (zero-init index base ``γ`` +
    frozen-history residual ``ψ``), a THREE-term additive logit model::

        α_{k}(·, i) := softmax_k( φ_i^k )     over k ∈ [max(i-U+1, 1), i]
             with     φ_i^k = γ[i, k] + ψ_i^k(s_τ^H) + ρ_i^k(token_i^k)

    - ``γ[i, k]``       : zero-init batch-agnostic ``(E, F, F)`` index base (inherited).
    - ``ψ_i^k(s_τ^H)``  : ensemble-aware residual head over the frozen input-history realisation
                          ``s_τ^H`` (inherited; ``compute_psi`` cached once per forward).
    - ``ρ_i^k(token_i)``: the ENSEMBLE-AWARE input head (post-RLRP-735 refactor) over the per-step
                          component tokens ``cat([mean, logvar])`` (width ``2·(O+A)``).

    The input head is zero-init (refactored :class:`InputDependentTemporalMixtureWeighs` zeros its
    ``out_proj``), so ``ρ = 0`` at init; together with ``γ = 0`` and ``ψ = 0`` this gives
    ``φ = 0`` at init ⇒ UNIFORM window softmax (byte-for-byte init parity with the pure-history /
    P3 mixers). All three terms share the same ``k``-window layout, so the addition is a plain
    broadcast of the window slice ``(E, B, min(i+1, U))``.

    Motivation: ``ρ`` conditions on the EVOLVING per-step forecast, whereas ``γ + ψ`` only see the
    step index and the FROZEN initial history — the added value over pure P3 is the dependence on
    the unrolled prediction rather than only the initial ``s_τ^H``.

    The input head's per-member layers are appended to ``_elite_layers`` for API symmetry (INERT
    today — the model never applies elite selection to the mixer).
    """

    def __init__(
        self,
        *args,
        input_token_dim: int,
        input_head_size: int = 64,
        input_num_layers: int = 2,
        input_dropout: float = 0.0,
        activation_factory: Optional[Callable[[], nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, activation_factory=activation_factory, **kwargs)
        # Compose the (refactored, ensemble-aware, zero-init) input head. Built directly in the
        # requested dtype; the model moves the whole module to its device afterwards.
        self.input_head = InputDependentTemporalMixtureWeighs(
            inputdim=input_token_dim,
            ensemble_size=self.E,
            head_size=input_head_size,
            num_layers=input_num_layers,
            dropout=input_dropout,
            activation_factory=activation_factory,
            dtype=self.dtype,
        )
        # API-symmetry only (INERT): extend elite forwarding with the input head's layers.
        self._elite_layers += self.input_head._elite_layers

    def forward(
        self,
        history_token: Tensor,
        horizon_index: int,
        log_prob: bool = True,
        psi: Optional[Tensor] = None,
        input_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        """Return the per-``(i, k)`` logits (or weights) with the additive ρ input term.

        :param input_tokens: The per-step tokens ``cat([mean, logvar])`` of shape
            ``(E, B, min(i+1, U), input_token_dim)``. When ``None`` this reduces to the P3
            ``γ + ψ`` mixer (so the same routing branch serves every history kind).
        """
        # γ + ψ window slice, (E, B, min(i+1, U)).
        logits_i = self.logits_at_i(history_token, horizon_index, psi=psi)
        if input_tokens is not None:
            # ρ input head, (E, B, min(i+1, U)); shares the window layout ⇒ plain addition.
            logits_i = logits_i + self.input_head(input_tokens, log_prob=True)
        if log_prob:
            return logits_i
        else:
            return self._normalize(logits_i)


class InputAndIndexDependentTemporalMixtureWeighs(_MixerTensorboardParamsMixin, nn.Module):
    """Index + input conditioned mixture weights (no input-history).

    Combines :class:`InputDependentTemporalMixtureWeighs` (per-step component tokens ``ρ``) with
    :class:`TemporalMixtureWeighs` (pure step-indexed learnable ``γ``), a TWO-term additive logit
    model WITHOUT the frozen-history ``ψ`` (so it needs none of the ``s_τ^H`` capture machinery)::

        α_{k}(token, i) := softmax_k( φ_i^k )     over k ∈ [max(i-U+1, 1), i]
             with     φ_i^k = γ[i, k] + ρ_i^k(token_i^k)

    Both terms are zero-init (``TemporalMixtureWeighs`` gammas are ``torch.zeros`` and the
    refactored input head zeros its ``out_proj``), so ``φ = 0`` at init ⇒ UNIFORM window softmax
    (warm-start parity with the other mixers).

    Call convention (per-batch, like the input-dependent mixer):
    ``forward(input_tokens, horizon_index, log_prob=True)`` where ``input_tokens`` has shape
    ``(E, B, min(i+1, U), input_token_dim)`` and the output is ``(E, B, min(i+1, U))``.

    The input head's per-member layers are exposed via ``_elite_layers`` for API symmetry (INERT
    today); ``γ`` is a batch-agnostic ``nn.Parameter`` (not elite), exactly like the pure-index
    mixer.
    """

    def __init__(
        self,
        unrol_len: int,
        horizon_len: int,
        ensemble_size: int,
        input_token_dim: int,
        head_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation_factory: Optional[Callable[[], nn.Module]] = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.U = unrol_len
        self.F = horizon_len
        self.E = ensemble_size
        # Pure step-indexed gammas (zero-init) — reused verbatim from the index mixer.
        self.index = TemporalMixtureWeighs(
            unrol_len=unrol_len,
            horizon_len=horizon_len,
            ensemble_size=ensemble_size,
            dtype=dtype,
        )
        # Ensemble-aware, zero-init input head (post-RLRP-735 refactor).
        self.input_head = InputDependentTemporalMixtureWeighs(
            inputdim=input_token_dim,
            ensemble_size=ensemble_size,
            head_size=head_size,
            num_layers=num_layers,
            dropout=dropout,
            activation_factory=activation_factory,
            dtype=dtype,
        )
        self.to(dtype=dtype)
        # API-symmetry only (INERT); γ is a batch-agnostic parameter, so NOT elite.
        self._elite_layers: list = list(self.input_head._elite_layers)

    def _normalize(self, logits_i: Tensor) -> Tensor:
        return torch.softmax(logits_i, dim=-1)  # (E x B x min(i+1, U))

    def logits_at_i(self, input_tokens: Tensor, i: int) -> Tensor:
        # Index γ window slice, (E, min(i+1, U)); unsqueeze the batch axis to broadcast over B.
        gamma_i = self.index.logits_at_i(i).unsqueeze(1)  # (E, 1, min(i+1, U))
        # ρ input head, (E, B, min(i+1, U)); shares the window layout ⇒ plain addition.
        rho_i = self.input_head(input_tokens, log_prob=True)
        return gamma_i + rho_i

    def forward(
        self, input_tokens: Tensor, horizon_index: int, log_prob: bool = True
    ) -> Tensor:
        """Return the per-``(i, k)`` logits (or softmax-normalised weights) at horizon ``i``.

        :param input_tokens: Per-step tokens ``cat([mean, logvar])`` of shape
            ``(E, B, min(i+1, U), input_token_dim)``.
        :param horizon_index: The horizon step ``i``.
        :param log_prob: When True return unnormalised logits; else softmax-normalised weights.
        """
        logits_i = self.logits_at_i(input_tokens, horizon_index)
        if log_prob:
            return logits_i
        else:
            return self._normalize(logits_i)

    def set_elite(self, elite_models) -> None:
        for layer in self._elite_layers:
            layer.set_elite(elite_models)

    def toggle_use_only_elite(self) -> None:
        for layer in self._elite_layers:
            layer.toggle_use_only_elite()


class MixtureWeightsTransformer(nn.Module):
    """
    Learnable mixture weights using a transformer-based model.

    This class is a PyTorch module designed to encode input tokens and compute mixture weights for
     them. It uses a transformer-based architecture for token transformation and applies a fully
     connected layer to generate logits or normalized weights.
    """

    def __init__(
        self,
        inputdim,
        tokendim=512,
        n_layers=2,
        n_heads=8,
        big_dim=2048,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        device=None,
    ):
        """
        Initializes a transformer-based encoder model to process input data into encoded token representations.
        The model supports multi-layer transformer encoder architecture with customizable dimensions and parameters.

        :param inputdim: Dimension of the input data to be processed.
        :param tokendim: Dimension of the token embeddings. Defaults to 512.
        :param n_layers: Number of layers in the transformer encoder. Defaults to 2.
        :param n_heads: Number of attention heads in the transformer encoder layer. Defaults to 8.
        :param big_dim: Dimension of the feedforward layers in the transformer encoder. Defaults to 2048.
        :param activation: the activation function to use in the transformer encoder. Defaults to nn.ReLU().
        :param dropout: the dropout rate to use in the transformer encoder. Defaults to 0.0.
        :param device: The computing device to execute the model (e.g., 'cpu' or 'cuda'). Defaults to None.
        """
        super().__init__()
        self.tokendim = tokendim
        self.device = device

        # ToDo(encode_token): assess >> Potentially use Random Fourier Feature instead of linear (Isabeau)
        self.encode_token = nn.Linear(inputdim, tokendim, device=device)

        self.base_layer = nn.TransformerEncoderLayer(
            d_model=tokendim,
            nhead=n_heads,
            dim_feedforward=big_dim,
            device=device,
            activation=activation,
            dropout=dropout,
            batch_first=True,
        )
        # ``enable_nested_tensor=False``: the activation is passed as a module instance
        # (``nn.ReLU()``), so PyTorch cannot set the ``activation_relu_or_gelu`` fast-path
        # flag and would otherwise warn ("enable_nested_tensor is True, but
        # self.use_nested_tensor is False ..."). Nested-tensor padding brings no benefit
        # for these fixed-length token sequences, so we disable it explicitly.
        self.tranformer = nn.TransformerEncoder(
            self.base_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Original log_alpha_fct. Changed to mixture_logits to enable downstream tensorboard logic
        self.mixture_logits = nn.Linear(tokendim, 1, device=device)

    def _normalize(self, logits_i) -> torch.Tensor:
        normalized_weights_i = torch.softmax(logits_i, dim=-1)
        return normalized_weights_i  # (E x B x S)

    def forward(self, tokens, log_prob: bool = True):
        """
        Processes input tokens through transformations and computes log-probabilities or
        normalized weights.

        The function accepts an input tensor of tokens, performs encoding and transformation
        operations, and calculates either log-probabilities or normalized weights depending
        on the provided flag.

        :param tokens: A 4-dimensional tensor of shape (E, B, S, input_dim), where E is the number
         of ensembles, B is the batch size, S is the sequence length, and input_dim is
         the dimensionality of the input tokens.
        :param log_prob: A boolean flag to determine the output type.
            If True, log-probabilities are returned.
            If False, normalized weights are returned.
        :return: A tensor representing either alpha logits (log-probabilities) of shape
            (E, B, S) or normalized weights of the same shape, depending on the `log_prob` flag.
        """
        E, B, S, input_dim = tokens.shape

        tokens = tokens.reshape(E * B * S, input_dim)
        encoded_tokens = self.encode_token(tokens).reshape(E * B, S, self.tokendim)
        transformed_tokens = self.tranformer(encoded_tokens)
        final_tokens = transformed_tokens.reshape(E * B * S, self.tokendim)
        alpha_logit = self.mixture_logits(final_tokens).reshape(E, B, S)
        if log_prob:
            return alpha_logit
        else:
            normalized_weights_i = self._normalize(alpha_logit)
            return normalized_weights_i


class TemporalMixtureWeightsTransformer(nn.Module):
    """
    Transformer-based learnable temporal mixture weights for autoregressive horizon blending.
    remark: Order-sensitive.

    This class is a PyTorch module designed to encode input tokens and compute temporal mixture
    weights for them. It uses a transformer-based architecture for token transformation and applies
    a fully connected layer to generate logits or normalized weights.
    """

    def __init__(
        self,
        inputdim: int,
        unroll_len: int,
        horizon_len: int,
        tokendim=512,
        n_layers=2,
        n_heads=8,
        big_dim=2048,
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        device=None,
    ):
        """
        Initializes a transformer-based encoder model to process input data into encoded token representations.
        The model supports multi-layer transformer encoder architecture with customizable dimensions and parameters.

        :param inputdim: Dimension of the input data to be processed.
        :param unroll_len: Maximum unroll length ``U`` (upper bound for the sequence dim ``S = min(i+1, U)``).
        :param horizon_len: Forecast horizon length ``F`` (number of distinct ``horizon_index`` values).
        :param tokendim: Dimension of the token embeddings. Defaults to 512.
        :param n_layers: Number of layers in the transformer encoder. Defaults to 2.
        :param n_heads: Number of attention heads in the transformer encoder layer. Defaults to 8.
        :param big_dim: Dimension of the feedforward layers in the transformer encoder. Defaults to 2048.
        :param activation: the activation function to use in the transformer encoder. Defaults to nn.ReLU().
        :param dropout: the dropout rate to use in the transformer encoder. Defaults to 0.0.
        :param device: The computing device to execute the model (e.g., 'cpu' or 'cuda'). Defaults to None.
        """
        super().__init__()
        self.tokendim = tokendim
        self.U = unroll_len
        self.F = horizon_len
        self.device = device

        # ToDo(encode_token): assess >> Potentially use Random Fourier Feature instead of linear (Isabeau)
        self.encode_token = nn.Linear(inputdim, tokendim, device=device)

        # learned positional embedding over the unroll-origin axis (S bounded by unrol_len)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.U, tokendim, device=device),
            requires_grad=True,
        )

        # learned horizon-index embedding (one entry per possible i in [0, F))
        self.horizon_embedding = nn.Embedding(self.F, tokendim, device=device)

        self.base_layer = nn.TransformerEncoderLayer(
            d_model=tokendim,
            nhead=n_heads,
            dim_feedforward=big_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            device=device,
        )
        # ``enable_nested_tensor=False``: the activation is passed as a module instance
        # (``nn.ReLU()``), so PyTorch cannot set the ``activation_relu_or_gelu`` fast-path
        # flag and would otherwise warn ("enable_nested_tensor is True, but
        # self.use_nested_tensor is False ..."). Nested-tensor padding brings no benefit
        # for these fixed-length token sequences, so we disable it explicitly.
        self.tranformer = nn.TransformerEncoder(
            self.base_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Original log_alpha_fct. Changed to mixture_logits to enable downstream tensorboard logic
        self.mixture_logits = nn.Linear(tokendim, 1, device=device)

    def _normalize(self, logits_i) -> torch.Tensor:
        normalized_weights_i = torch.softmax(logits_i, dim=-1)
        return normalized_weights_i  # (E x min(i,U))

    def forward(self, tokens, horizon_index: int, log_prob: bool = True):
        """
        Processes input tokens through transformations and computes log-probabilities or
        normalized weights for a specific horizon index ``i``.

        :param tokens: A 4-dimensional tensor of shape (E, B, S, input_dim), where E is the number
         of ensembles, B is the batch size, S is the sequence length (``min(i+1, U)``), and input_dim is
         the dimensionality of the input tokens.
        :param horizon_index: Current horizon index ``i`` (used to fetch the horizon embedding so the
         shared transformer can specialize its mixture weights per horizon).
        :param log_prob: A boolean flag to determine the output type.
            If True, log-probabilities are returned.
            If False, normalized weights are returned.
        :return: A tensor representing either alpha logits (log-probabilities) of shape
            (E, B, S) or normalized weights of the same shape, depending on the `log_prob` flag.
        """

        # Remark: That's the row's in the sketch
        E, B, S, input_dim = tokens.shape

        tokens = tokens.reshape(E * B * S, input_dim)
        encoded_tokens = self.encode_token(tokens).reshape(E * B, S, self.tokendim)

        # Positional (over unroll-origin axis) + horizon (per i) embeddings.
        encoded_tokens = encoded_tokens + self.pos_embedding[:, :S, :]
        # Remark: That's the column's in the sketch
        h_idx = torch.as_tensor(
            horizon_index, dtype=torch.long, device=encoded_tokens.device
        )
        h_emb = self.horizon_embedding(h_idx).view(1, 1, self.tokendim)
        encoded_tokens = encoded_tokens + h_emb

        transformed_tokens = self.tranformer(encoded_tokens)
        final_tokens = transformed_tokens.reshape(E * B * S, self.tokendim)
        alpha_logit = self.mixture_logits(final_tokens).reshape(E, B, S)
        if log_prob:
            return alpha_logit
        else:
            normalized_weights_i = self._normalize(alpha_logit)
            return normalized_weights_i
