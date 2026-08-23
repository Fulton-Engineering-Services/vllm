# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the NIXL host-staging adapter scaffolding."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.host_staging_adapter import (
    NixlHostStagingAdapter,
)


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


def test_adapter_follows_probe():
    """The adapter's enabled state follows the host-staging platform probe."""
    worker = MagicMock()
    worker.kv_transfer_config.kv_connector_extra_config = {}
    worker.device_id = 0

    adapter = NixlHostStagingAdapter(worker)

    # Arenas are allocated lazily inside register_kv_caches.
    assert isinstance(adapter.enabled, bool)
    assert adapter.send_arena is None
    assert adapter.recv_arena is None


def _make_worker_mock() -> MagicMock:
    worker = MagicMock()
    worker.kv_transfer_config.kv_connector_extra_config = {"host_staging": "on"}
    worker.kv_transfer_config.kv_role = "kv_both"
    worker.device_id = 0
    worker.engine_id = "engine_0"
    worker.tp_rank = 0
    worker.block_len_per_layer = [256]
    worker.kv_caches_base_addr = {"engine_0": [[0x1000_0000]]}
    worker._physical_blocks_per_logical_kv_block = 1
    worker.kv_cache_config.kv_cache_groups = [MagicMock()]
    worker._logical_to_kernel_block_ids = (
        lambda padded, expansion: [list(g) for g in padded]
    )
    worker.nixl_wrapper = MagicMock()
    worker.nixl_backends = ["UCX"]
    worker.nixl_wrapper.get_reg_descs.return_value = MagicMock()
    worker.nixl_wrapper.get_xfer_descs.return_value = MagicMock()
    worker.nixl_wrapper.prep_xfer_dlist.return_value = 123
    return worker


def test_adapter_forced_on():
    """Forcing host_staging='on' enables the adapter and allocates arenas."""
    pytest.importorskip("cupy", reason="cupy not available")

    worker = _make_worker_mock()
    adapter = NixlHostStagingAdapter(worker)
    assert adapter.enabled is True

    adapter.register_kv_caches({})

    assert adapter.send_arena is not None
    assert adapter.recv_arena is not None
    assert adapter.recv_window is not None


def test_adapter_forced_on_missing_cupy_raises():
    """When cupy is missing, forced-on register_kv_caches raises a clear error."""
    if _cupy_available():
        pytest.skip("cupy is available")

    worker = _make_worker_mock()
    adapter = NixlHostStagingAdapter(worker)
    assert adapter.enabled is True

    with pytest.raises(RuntimeError, match="cupy"):
        adapter.register_kv_caches({})


def test_staging_window_base_addrs_when_disabled():
    worker = MagicMock()
    worker.kv_transfer_config.kv_connector_extra_config = {"host_staging": "off"}
    worker.device_id = 0

    adapter = NixlHostStagingAdapter(worker)
    assert adapter.enabled is False
    assert adapter.get_staging_window_base_addrs() is None


def test_staging_window_base_addrs_when_enabled():
    pytest.importorskip("cupy", reason="cupy not available")

    worker = _make_worker_mock()
    adapter = NixlHostStagingAdapter(worker)
    adapter.register_kv_caches({})

    addrs = adapter.get_staging_window_base_addrs()
    assert addrs is not None
    assert len(addrs) == len(worker.block_len_per_layer)
