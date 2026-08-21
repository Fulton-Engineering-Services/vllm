# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeConnector — the scheduler-facing KV connector facade.

This is the class that vLLM's MultiConnector and KV cache manager interact
with.  It delegates to MooncakeConnectorScheduler (scheduler-side) and
MooncakeConnectorWorker (worker-side).
"""
import asyncio
import logging
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorStats,
    SupportsHMA,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger

from ._protocol import (
    MooncakeConnectorMetadata,
    PullReqMeta,
)
from .scheduler import MooncakeConnectorScheduler

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    """Mooncake KV connector facade.

    The connector itself stores no mutable state — it is re-created each step
    by the scheduler.  Stateful components are MooncakeConnectorScheduler
    (per-scheduler lifetime) and MooncakeConnectorWorker (per-engine lifetime).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        conn_scheduler: MooncakeConnectorScheduler,
    ):
        self.vllm_config = vllm_config
        self.conn_scheduler = conn_scheduler
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._connector_metadata: MooncakeConnectorMetadata | None = None

    # ── KV cache layout ────────────────────────────────────────────────

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        kv_role = kv_transfer_config.kv_role
        if kv_role == "kv_producer":
            return "HMA"
        if kv_role == "kv_consumer":
            return kv_transfer_config.kv_connector_extra_config.get(
                "consumer_kvcache_layout", None
            )
        return None

    # ── lifecycle ──────────────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> int:
        return self.conn_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks"
    ):
        self.conn_scheduler.update_state_after_alloc(request, blocks)

    def build_connector_meta(
        self, requests: list["Request"]
    ) -> MooncakeConnectorMetadata:
        return self.conn_scheduler.build_connector_meta(requests)

    def request_finished(
        self, request: "Request", block_ids: list[list[int]]
    ) -> tuple[int, dict | None]:
        return self.conn_scheduler.request_finished(request, block_ids)

    def request_finished_all_groups(
        self, request: "Request", block_ids: list[list[int]]
    ) -> tuple[int, dict | None]:
        return self.conn_scheduler.request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        pass

    def get_finished(
        self, scheduler_output
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata,
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        pass

    # ── stats ──────────────────────────────────────────────────────────

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        return None

    @classmethod
    def build_kv_connector_stats(cls) -> KVConnectorStats | None:
        return None