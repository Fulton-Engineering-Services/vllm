# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin: register, lifecycle, stats, block arithmetic for MooncakeConnectorWorker.

Imports are deferred to TYPE_CHECKING where possible to keep the import graph
flat — this file is mixed into worker.py, not imported by the send/receive
mixin files.
"""
import asyncio
import logging
import threading
import time
from collections import defaultdict

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

from ._protocol import (
    ReqId,
    TransferId,
    _get_tensor_dense_flag,
)
from ._transfer_planning import _expand_transfer_regions
from .stats import MooncakeKVConnectorStats

logger = init_logger(__name__)


class _WorkerSetupMixin:
    """register_kv_caches + lifecycle + stats for MooncakeConnectorWorker."""

    # ── block arithmetic ───────────────────────────────────────────────

    def _sync_block_size_with_kernel(self) -> None:
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backends,
        )
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size

    def _logical_to_kernel_block_ids(
        self, block_ids: list[list[int]]
    ) -> list[list[int]]:
        if self._physical_blocks_per_logical_kv_block == 1:
            return block_ids
        from vllm.v1.kv_cache_interface import MambaSpec
        block_arange = np.arange(
            self._physical_blocks_per_logical_kv_block
        ).reshape(1, -1)
        group_specs = self.kv_cache_config.kv_cache_groups
        return [
            BlockTable.map_to_kernel_blocks(
                np.array(group),
                self._physical_blocks_per_logical_kv_block,
                block_arange,
            ).tolist()
            if not isinstance(group_specs[i].kv_cache_spec, MambaSpec)
            else group
            for i, group in enumerate(block_ids)
        ]

    # ── lifecycle ──────────────────────────────────────────────────────

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if hasattr(self, "bootstrap_server"):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    # ── KV cache registration ──────────────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.v1.kv_cache_interface import (
            MambaSpec,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        region_base_addresses: list[int] = []
        seen_storage_ptrs: set[int] = set()
        self.block_len_per_layer = []
        self.kv_block_len_per_layer = []
        self.registered_layer_names = []
        self.registered_layer_indices = []
        self.registered_group_indices = []

        for layer_name, cache_or_caches in kv_caches.items():
            layer_index = extract_layer_index(layer_name)
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s because no KV cache spec is present.",
                    layer_name,
                )
                continue
            if isinstance(layer_spec, MambaSpec):
                if isinstance(cache_or_caches, (list, tuple)) and len(cache_or_caches) > 1:
                    cache_list = list(cache_or_caches[:-1])
                else:
                    cache_list = [cache_or_caches]
            else:
                cache_list = [cache_or_caches]

            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                block_len = cache.stride(0) * cache.element_size()
                region_base_addresses.append(base_addr)

                if isinstance(layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                    kv_block_len = layer_spec.page_size_bytes
                elif self.transfer_topo.virtually_split_kv_in_blocks and not isinstance(
                    layer_spec, MambaSpec
                ):
                    kv_block_len = block_len // 2
                else:
                    kv_block_len = block_len
                self.block_len_per_layer.append(block_len)
                self.kv_block_len_per_layer.append(kv_block_len)
                self.registered_layer_names.append(layer_name)
                self.registered_layer_indices.append(layer_index)
                self.registered_group_indices.append(
                    self._layer_group_indices[layer_name]
                )
                storage = cache.untyped_storage()
                storage_addr = storage.data_ptr()
                if storage_addr not in seen_storage_ptrs:
                    seen_storage_ptrs.add(storage_addr)
                    kv_data_ptrs.append(storage_addr)
                    kv_data_lens.append(storage.nbytes())

        self.kv_caches_base_addr = region_base_addresses
        self.seen_base_addresses = kv_data_ptrs

        if not kv_data_ptrs:
            raise RuntimeError("No KV cache tensors were registered with Mooncake.")

        if self.host_staging:
            self._setup_host_staging()
        else:
            ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
            if ret_value != 0:
                raise RuntimeError("Mooncake batch memory registration failed.")

        self.device_kv_caches = kv_caches
        logger.debug(
            "registered block_lens=%s kv_block_lens=%s",
            self.block_len_per_layer,
            self.kv_block_len_per_layer,
        )

        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )

    # ── stats / finished-req reporting ─────────────────────────────────

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        import asyncio
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )
        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )
        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()
        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )
        return finished_sending_reqs or None, finished_recving_reqs or None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    async def fetch_finished_sending_reqs(self):
        from vllm import envs
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()
        now = time.perf_counter()
        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)
        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]
        return finished_sending_reqs

    async def fetch_finished_recving_reqs(self):
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    @staticmethod
    def _drain_finished_set(finished: set[ReqId]) -> set[ReqId]:
        taken = finished.copy()
        finished.clear()
        return taken

    # ── transfer planning helpers (used by both send and staging) ──────

    def _get_transfer_regions(
        self,
        base_addrs: list[int],
        block_lens: list[int],
        kv_block_lens: list[int],
        layer_names: list[str],
        layer_indices: list[int],
        group_indices: list[int] | None = None,
    ):
        from vllm.v1.kv_cache_interface import (
            MambaSpec,
            MLAAttentionSpec,
            SlidingWindowMLASpec,
        )
        if not group_indices:
            group_indices = [
                self._layer_group_indices.get(layer_name, 0)
                for layer_name in layer_names
            ]
        split_kv_regions = None
        if self.transfer_topo.virtually_split_kv_in_blocks:
            split_kv_regions = [
                not isinstance(
                    self._layer_specs[layer_name],
                    (MambaSpec, MLAAttentionSpec, SlidingWindowMLASpec),
                )
                for layer_name in layer_names
            ]
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            kv_block_lens=kv_block_lens,
            layer_names=layer_names,
            layer_indices=layer_indices,
            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks,
            group_indices=group_indices,
            split_kv_regions=split_kv_regions,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ):
        from ._transfer_planning import _compute_sender_transfer_plan
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache