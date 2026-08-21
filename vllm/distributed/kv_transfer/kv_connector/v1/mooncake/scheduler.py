# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeConnectorScheduler — scheduler-side logic for Mooncake KV transfer.

Extracted from mooncake_connector.py to keep the module tractable (~250 lines
instead of 2,685).
"""
import logging
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from ._protocol import (
    MooncakeConnectorMetadata,
    ReqId,
    TransferId,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        SlidingWindowSpec,
    )
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class MooncakeConnectorScheduler:
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self._has_mamba = kv_cache_config.has_mamba_layers
        self._reqs_need_recv: dict[ReqId, tuple["Request", list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, tuple["Request", list[list[int]]]] = {}
        self._reqs_not_processed: set[TransferId] = set()
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        params = request.kv_transfer_params
        if (
            params is not None
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return
            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )
        if not params:
            return 0, False
        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            count = self._get_remote_prefill_token_count(len(token_ids)) - (
                num_computed_tokens
            )
            if count > 0:
                return count, True
        if params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )
        if not params:
            return
        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                local_block_ids = self.get_sw_clipped_blocks(unhashed_block_ids)
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            params["do_remote_prefill"] = False
        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()
        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        from vllm.v1.request import RequestStatus
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None
        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None
        if not params.get("do_remote_decode"):
            return False, None
        assert not self.is_kv_consumer
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None
        delay_free_blocks = any(len(group) > 0 for group in block_ids)
        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (
                request,
                self.get_sw_clipped_blocks(block_ids),
            )
        return delay_free_blocks, None