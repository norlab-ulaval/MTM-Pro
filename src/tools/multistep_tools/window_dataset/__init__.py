# coding=utf-8
"""Lazy multistep window dataset tooling (RLRP-824).

Permanent package. Introduced by the Ultra-long-horizon MS->MS training via a lazy window
DataLoader ``.junie`` plan (``ultra_long_horizon_dataloader_plan_RLRP-824_20260915.md``).
Single-step trajectories are stored once and the asymmetric ``(H, F)`` composed windows are
gathered lazily per batch, so host memory scales with the number of single-step samples, not
with the forecast depth ``F``.
"""
