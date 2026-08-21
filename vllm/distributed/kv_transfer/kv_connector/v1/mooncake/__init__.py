# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake KV-transfer connector.

Public API surface:

  MooncakeConnector          — scheduler-facing facade (base.py)
  MooncakeConnectorScheduler — scheduler-side state machine (scheduler.py)
  MooncakeConnectorWorker    — worker-side orchestrator (worker.py)
  MooncakeConnectorMetadata  — per-step metadata container (_protocol.py)

The old mooncake_connector.py is now a thin re-export that preserves the
module path for dynamic loading via the connector registry.
"""
from .base import MooncakeConnector
from .scheduler import MooncakeConnectorScheduler
from .worker import MooncakeConnectorWorker
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
    get_mooncake_bootstrap_addr,
    get_mooncake_side_channel_port,
    should_launch_bootstrap_server,
)

__all__ = [
    "MooncakeConnector",
    "MooncakeConnectorMetadata",
    "MooncakeConnectorScheduler",
    "MooncakeConnectorWorker",
    "MooncakeXferMetadata",
    "MooncakeXferResponse",
    "MooncakeXferResponseStatus",
    "PullReqMeta",
    "ReqId",
    "SendBlockMeta",
    "TransferId",
    "TransferRegion",
    "get_mooncake_bootstrap_addr",
    "get_mooncake_side_channel_port",
    "should_launch_bootstrap_server",
]