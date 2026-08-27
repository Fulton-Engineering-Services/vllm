# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared EP all2all stats accumulator for expert-parallel all2all backends.

Collects dispatch/combine latency, token counts, active ranks, and buffer
sizes from any EP all2all prepare_finalize implementation (nixl_ep,
deepep_ll, deepep_ht). Drained by the scheduler's make_stats() and sent
to the frontend via SchedulerStats.ep_all2all_stats, where
PrometheusStatLogger records them as Ray metrics.
"""

from collections import defaultdict

# Module-level so they survive prepare_finalize instance recreation during
# elastic EP scale-up/down.
_ep_all2all_stats: dict[str, list[float]] = defaultdict(list)
_ep_all2all_backend: str = "unknown"
_ep_all2all_buffer_rdma_bytes: float = 0.0
_ep_all2all_buffer_nvl_bytes: float = 0.0


def set_ep_all2all_backend(backend: str) -> None:
    global _ep_all2all_backend
    _ep_all2all_backend = backend


def set_ep_all2all_buffer_sizes(rdma_bytes: float, nvl_bytes: float) -> None:
    global _ep_all2all_buffer_rdma_bytes, _ep_all2all_buffer_nvl_bytes
    _ep_all2all_buffer_rdma_bytes = rdma_bytes
    _ep_all2all_buffer_nvl_bytes = nvl_bytes


def record_ep_all2all_stats(key: str, value: float) -> None:
    _ep_all2all_stats[key].append(value)


def init_ep_all2all_stats(
    backend: str, all2all_manager: object | None = None
) -> None:
    """Register the active EP all2all backend and read buffer sizes from
    the all2all manager. Called by prepare_finalize constructors."""
    set_ep_all2all_backend(backend)
    if all2all_manager is not None:
        set_ep_all2all_buffer_sizes(
            float(getattr(all2all_manager, "num_rdma_bytes", 0.0)),
            float(getattr(all2all_manager, "num_nvl_bytes", 0.0)),
        )


def drain_ep_all2all_stats() -> dict[str, list[float] | float] | None:
    """Drain accumulated EP all2all stats. Called by the scheduler."""
    if not _ep_all2all_stats:
        return None
    stats: dict[str, list[float] | float] = dict(_ep_all2all_stats)
    stats["backend"] = _ep_all2all_backend
    stats["buffer_rdma_bytes"] = _ep_all2all_buffer_rdma_bytes
    stats["buffer_nvl_bytes"] = _ep_all2all_buffer_nvl_bytes
    _ep_all2all_stats.clear()
    return stats