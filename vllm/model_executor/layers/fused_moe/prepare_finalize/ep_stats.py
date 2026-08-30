# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EP all2all stats for expert-parallel all2all backends.

Records dispatch/combine latency, token counts, active ranks, and buffer
sizes directly as Ray metrics from within the worker process (Ray actor).
This is necessary because the model forward pass (where dispatch/combine
happens) runs in a separate Ray actor process from the EngineCore/scheduler
— a module-level accumulator drained by the scheduler would be empty
because the two processes don't share Python globals.

Ray metrics created here are exported by Ray's metrics agent from the
worker actor process, appearing as ray_vllm_ep_all2all_* in VictoriaMetrics
with WorkerId and other auto-added Ray labels.

Module-level placement ensures survival across prepare_finalize instance
recreation during elastic EP scale-up/down. Gracefully degrades (no-op)
when Ray metrics wrappers are unavailable.
"""

_ep_lat_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]

try:
    from vllm.v1.metrics.ray_wrappers import (
        RayCounterWrapper,
        RayGaugeWrapper,
        RayHistogramWrapper,
    )

    _ep_dispatch_latency = RayHistogramWrapper(
        name="vllm:ep_all2all_dispatch_latency_seconds",
        documentation="Latency of EP all2all dispatch calls in seconds.",
        buckets=_ep_lat_buckets,
        labelnames=["backend"],
    )
    _ep_combine_latency = RayHistogramWrapper(
        name="vllm:ep_all2all_combine_latency_seconds",
        documentation="Latency of EP all2all combine calls in seconds.",
        buckets=_ep_lat_buckets,
        labelnames=["backend"],
    )
    _ep_dispatch_tokens = RayCounterWrapper(
        name="vllm:ep_all2all_dispatch_tokens_total",
        documentation="Total tokens dispatched via EP all2all.",
        labelnames=["backend"],
    )
    _ep_combine_tokens = RayCounterWrapper(
        name="vllm:ep_all2all_combine_tokens_total",
        documentation="Total tokens combined via EP all2all.",
        labelnames=["backend"],
    )
    _ep_active_ranks = RayGaugeWrapper(
        name="vllm:ep_all2all_active_ranks",
        documentation="Number of active EP all2all ranks in the current config.",
        labelnames=["backend"],
    )
    _ep_buffer_rdma_bytes = RayGaugeWrapper(
        name="vllm:ep_all2all_buffer_rdma_bytes",
        documentation="RDMA buffer size in bytes allocated by the EP all2all backend.",
        labelnames=["backend"],
    )
    _ep_buffer_nvl_bytes = RayGaugeWrapper(
        name="vllm:ep_all2all_buffer_nvl_bytes",
        documentation="NVL buffer size in bytes allocated by the EP all2all backend.",
        labelnames=["backend"],
    )
except ImportError:
    _ep_dispatch_latency = None
    _ep_combine_latency = None
    _ep_dispatch_tokens = None
    _ep_combine_tokens = None
    _ep_active_ranks = None
    _ep_buffer_rdma_bytes = None
    _ep_buffer_nvl_bytes = None

_ep_backend: str = "unknown"


def init_ep_all2all_stats(
    backend: str, all2all_manager: object | None = None
) -> None:
    """Register the active EP all2all backend and record buffer sizes.

    Called by prepare_finalize constructors in the worker process.
    """
    global _ep_backend
    _ep_backend = backend
    if all2all_manager is not None and _ep_buffer_rdma_bytes is not None:
        rdma = float(getattr(all2all_manager, "num_rdma_bytes", 0.0))
        nvl = float(getattr(all2all_manager, "num_nvl_bytes", 0.0))
        _ep_buffer_rdma_bytes.labels(backend).set(rdma)
        _ep_buffer_nvl_bytes.labels(backend).set(nvl)


def record_ep_all2all_stats(key: str, value: float) -> None:
    """Record an EP all2all stat directly to Ray metrics.

    Called from prepare_finalize dispatch/combine methods in the worker
    process. The backend label is bound from the value set by
    init_ep_all2all_stats.
    """
    backend = _ep_backend
    if key == "dispatch_latency" and _ep_dispatch_latency is not None:
        _ep_dispatch_latency.labels(backend).observe(value)
    elif key == "combine_latency" and _ep_combine_latency is not None:
        _ep_combine_latency.labels(backend).observe(value)
    elif key == "dispatch_tokens" and _ep_dispatch_tokens is not None:
        _ep_dispatch_tokens.labels(backend).inc(value)
    elif key == "combine_tokens" and _ep_combine_tokens is not None:
        _ep_combine_tokens.labels(backend).inc(value)
    elif key == "active_ranks" and _ep_active_ranks is not None:
        _ep_active_ranks.labels(backend).set(value)