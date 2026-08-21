# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin re-export of the decomposed mooncake connector.

The implementation now lives in sibling private modules:

  base.py            - MooncakeConnector (scheduler-facing facade)
  scheduler.py       - MooncakeConnectorScheduler (scheduler-side state)
  worker.py          - MooncakeConnectorWorker (worker-side orchestrator)
  _protocol.py       - data classes + utility functions
  _transfer_planning.py - pure transfer-geometry functions
  _worker_send.py    - send-side protocol (mixin)
  _worker_receive.py - receive-side protocol (mixin)
  _worker_setup.py   - register_kv_caches + lifecycle + stats (mixin)
  staging.py         - host-staging (GB10) arena + mixin

This file exists solely to keep the dynamic connector-registry path
``vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector``
working.  Once all callers switch to the package __init__.py this can be
removed.
"""
# flake8: noqa: F401

from ._protocol import (
    MooncakeConnectorMetadata,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    ReqId,
    SendBlockMeta,
    TransferId,
    TransferRegion,
    _async_loop,
    _get_tensor_dense_flag,
    get_mooncake_bootstrap_addr,
    get_mooncake_side_channel_port,
    group_concurrent_contiguous,
    should_launch_bootstrap_server,
)
from ._transfer_planning import (
    _align_transfer_regions,
    _can_coalesce_block_transfers,
    _compute_sender_transfer_plan,
    _expand_transfer_regions,
    _get_tp_ratio,
    _validate_asymmetric_region_lengths,
)
from .base import MooncakeConnector
from .scheduler import MooncakeConnectorScheduler
from .worker import MooncakeConnectorWorker

# Re-export _PinnedStagingArena under its original name for pickling / type refs.
from .staging import _PinnedStagingArena, _StagingMixin

# Make the public symbols available at the legacy dot-path.
__all__ = [
    "MooncakeConnector",
    "MooncakeConnectorMetadata",
    "MooncakeConnectorScheduler",
    "MooncakeConnectorWorker",
    "MooncakeXferMetadata",
    "MooncakeXferResponse",
    "MooncakeXferResponseStatus",
    "PullReqMeta",
    "SendBlockMeta",
    "TransferRegion",
    "MooncakeKVConnectorStats",
    "TransferId",
    "ReqId",
    "_PinnedStagingArena",
    "_StagingMixin",
]

# Counter batch stats (lives in stats.py — exposed here for convenience).
from .stats import MooncakeKVConnectorStats