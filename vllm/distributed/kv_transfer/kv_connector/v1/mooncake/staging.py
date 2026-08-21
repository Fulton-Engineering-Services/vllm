# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any
import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size


try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None


logger = init_logger(__name__)

from ._protocol import PullReqMeta, TransferRegion

from ._transfer_planning import _expand_transfer_regions

from vllm.platforms import current_platform

class _PinnedStagingArena:
    """Pinned host buffer registered with the Mooncake transfer engine.

    Why this exists (GB10 / DGX Spark): on unified-memory platforms GPUDirect
    is architecturally unavailable — nvidia-peermem does not load,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED == 0, and ibv_reg_mr() on CUDA VAs
    fails with EFAULT (all verified empirically on driver 580.173.02). The
    NVIDIA-sanctioned fallback (DGX Spark Porting Guide) is to transfer
    through cudaHostAlloc-style pinned host buffers, which register fine.

    Copy mechanism: torch's pinned-tensor path (dgx-spark-wheels torch
    2.13.0+cu13.3, verified 2026-08-21) allocates the backing store cheaply
    and holds it stable; cupy-cuda13x does the raw-pointer copies via
    UnownedMemory+MemoryPointer over BOTH the pinned buffer and the device
    buffers — PinnedMemoryPointer itself has no copy API, and
    cupy.cuda.runtime.memcpyAsync rejects explicit kind codes on this
    platform (both falsified empirically before choosing this).
    """

    def __init__(self, engine: "TransferEngine", num_bytes: int, label: str):
        try:
            import cupy as cp

            # torch owns the backing store (cheap allocator bookkeeping)
            self.buf = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
            self.cupy = cp
        except ImportError as e:
            raise RuntimeError(
                "Mooncake host_staging requires cupy (cupy-cuda13x) and torch"
            ) from e
        self.base_ptr = int(self.buf.data_ptr())
        self.num_bytes = num_bytes
        # cupy UnownedMemory wrapper over the torch-pinned storage -> the
        # copy API (PinnedMemoryPointer has none).
        self._unowned = self.cupy.cuda.UnownedMemory(self.base_ptr, num_bytes, None)
        self.stream = self.cupy.cuda.Stream(non_blocking=True)
        ret = engine.register_memory(self.base_ptr, num_bytes)
        if ret != 0:
            raise RuntimeError(
                f"Mooncake host_staging: failed to register {label} staging "
                f"arena ({num_bytes} bytes) with the transfer engine "
                f"(ret={ret})."
            )
        logger.info(
            "Mooncake host_staging: registered %s pinned arena at %#x (%d MiB)",
            label,
            self.base_ptr,
            num_bytes // 2**20,
        )

    def _device_mp(self, gpu_ptr: int, nbytes: int):
        """Copy-capable pointer over an external device buffer address."""
        return self.cupy.cuda.MemoryPointer(
            self.cupy.cuda.UnownedMemory(gpu_ptr, nbytes, None), 0
        )

    def d2h(self, gpu_ptr: int, offset: int, nbytes: int) -> None:
        """Async-copy nbytes from a device buffer into the arena at offset."""
        pinned_mp = self.cupy.cuda.MemoryPointer(self._unowned, offset)
        pinned_mp.copy_from_device_async(
            self._device_mp(gpu_ptr, nbytes), nbytes, self.stream
        )

    def h2d(self, offset: int, gpu_ptr: int, nbytes: int) -> None:
        """Async-copy nbytes from the arena at offset to a device buffer."""
        dev_dst_mp = self._device_mp(gpu_ptr, nbytes)
        dev_dst_mp.copy_from_host_async(
            self.base_ptr + offset, nbytes, self.stream
        )

    def synchronize(self) -> None:
        self.stream.synchronize()

class _StagingMixin:
    def _send_blocks_staged(
            self,
            remote_session: str,
            src_ptrs: list[int],
            dst_ptrs: list[int],
            lengths: list[int],
        ) -> int:
            """Staged variant of _send_blocks for host_staging (GB10).
    
            Chunks the descriptor list through the pinned send arena: D2H copy
            into the arena, sync write from the arena, reuse. The destination
            pointers already refer to the consumer's pinned receive window (it
            advertises window addresses in place of its GPU geometry), so only
            the source side needs rewriting here.
            """
            arena = self._staging_send_arena
            assert arena is not None
            start_time = time.perf_counter()
            stage_time = 0.0
            ret_value = 0
            total_bytes = sum(lengths)
            total_descs = len(src_ptrs)
            i = 0
            n = len(src_ptrs)
            while i < n and ret_value == 0:
                # Pack as many descriptors as fit in the arena. Oversized single
                # descriptors are split across arena-sized pieces.
                chunk_src: list[int] = []
                chunk_dst: list[int] = []
                chunk_len: list[int] = []
                pos = 0
                while i < n and pos < arena.num_bytes:
                    length = lengths[i]
                    remaining = arena.num_bytes - pos
                    take = min(length, remaining)
                    t0 = time.perf_counter()
                    arena.d2h(src_ptrs[i] + (lengths[i] - length), pos, take)
                    stage_time += time.perf_counter() - t0
                    chunk_src.append(arena.base_ptr + pos)
                    chunk_dst.append(dst_ptrs[i] + (lengths[i] - length))
                    chunk_len.append(take)
                    pos += take
                    if take == length:
                        i += 1
                    else:
                        # Partial: continue this descriptor in the next chunk.
                        lengths[i] = length - take
                        src_ptrs[i] += take
                        dst_ptrs[i] += take
                arena.synchronize()
                if chunk_src:
                    ret_value = self.engine.batch_transfer_sync_write(
                        remote_session, chunk_src, chunk_dst, chunk_len
                    )
            duration = time.perf_counter() - start_time
            if ret_value == 0:
                self.xfer_stats.record_transfer(
                    duration_s=duration,
                    total_bytes=total_bytes,
                    num_descs=total_descs,
                )
                logger.debug(
                    "host_staging: sent to %s in %.1f ms (stage %.1f ms)",
                    remote_session,
                    duration * 1000,
                    stage_time * 1000,
                )
            else:
                self.xfer_stats.record_failed_transfer()
                logger.warning(
                    "host_staging: sending to %s failed (ret=%s) after %.1f ms",
                    remote_session,
                    ret_value,
                    duration * 1000,
                )
            return ret_value
    def _setup_host_staging(self) -> None:
            """Allocate and register the pinned staging buffers for this role.
    
            Producer/kv_both: a flat send arena; _send_blocks chunks transfers
            through it. Consumer/kv_both: a per-region receive window advertised
            to the producer in place of the real GPU cache geometry.
            """
            if not self.is_kv_consumer:
                self._staging_send_arena = _PinnedStagingArena(
                    self.engine, self.staging_send_mib * 2**20, "send"
                )
            if not self.is_kv_producer:
                self._staging_recv_arena = _PinnedStagingArena(
                    self.engine, self.staging_recv_mib * 2**20, "recv"
                )
                # Carve the arena into per-(unexpanded-)region windows, each with
                # the same kernel-block capacity. block_len varies across regions
                # (full-attention layers vs GDN conv slices), so the capacity is
                # bounded by the total per-block-column bytes.
                col_bytes = sum(self.block_len_per_layer)
                if col_bytes <= 0:
                    raise RuntimeError(
                        "Mooncake host_staging: empty KV geometry at registration."
                    )
                self._staging_window_blocks = (
                    self.staging_recv_mib * 2**20 // col_bytes
                )
                if self._staging_window_blocks == 0:
                    raise RuntimeError(
                        "Mooncake host_staging: recv window "
                        f"({self.staging_recv_mib} MiB) smaller than one block "
                        f"column ({col_bytes} bytes)."
                    )
                bases: list[int] = []
                offset = 0
                for block_len in self.block_len_per_layer:
                    bases.append(self._staging_recv_arena.base_ptr + offset)
                    offset += block_len * self._staging_window_blocks
                if offset > self._staging_recv_arena.num_bytes:
                    raise RuntimeError(
                        "Mooncake host_staging: recv window geometry overflow "
                        f"({offset} > {self._staging_recv_arena.num_bytes})."
                    )
                self._staging_window_bases = bases
                # Expanded-region views for the completion H2D copy plan. The
                # window mirrors the real per-layer block layout exactly, so the
                # same _get_transfer_regions expansion applies to both.
                self._staging_window_regions = self._get_transfer_regions(
                    self._staging_window_bases,
                    self.block_len_per_layer,
                    self.kv_block_len_per_layer,
                    self.registered_layer_names,
                    self.registered_layer_indices,
                    self.registered_group_indices,
                )
                self._staging_real_regions = self._get_transfer_regions(
                    self.kv_caches_base_addr,
                    self.block_len_per_layer,
                    self.kv_block_len_per_layer,
                    self.registered_layer_names,
                    self.registered_layer_indices,
                    self.registered_group_indices,
                )
                logger.info(
                    "Mooncake host_staging: recv window holds %d kernel blocks "
                    "per region across %d regions (%d MiB).",
                    self._staging_window_blocks,
                    len(self._staging_window_bases),
                    offset // 2**20,
                )
    def _staging_assign_slots(
            self, pull_metas: dict[ReqId, PullReqMeta]
        ) -> list[ReqId]:
            """Assign receive-window slots to each request of a pull batch.
    
            Returns the req_ids that did NOT fit (callers drop them loudly).
            Slot ids are logical block ids; the producer converts them with the
            same _logical_to_kernel_block_ids expansion it applies to real ids,
            so its destination arithmetic lands in the window unchanged.
            """
            kernel_capacity = self._staging_window_blocks
            cursors: dict[int, int] = defaultdict(int)
            overflow: list[ReqId] = []
            for req_id, pull_meta in pull_metas.items():
                staging_slots: dict[int, tuple[int, list[int]]] = {}
                fits = True
                for group_index, group in enumerate(pull_meta.local_block_ids):
                    # Mirror the producer's NULL-block filtering (mamba/GDN
                    # align placeholders carry no transferable state) so slot
                    # ids and P's kernel indices stay 1:1.
                    filtered = [b for b in group if b != NULL_BLOCK_ID]
                    if not filtered:
                        continue
                    off = cursors[group_index]
                    # Capacity in this group's own (logical) id space: attention
                    # groups expand c-fold into kernel blocks, Mamba/GDN do not.
                    if (
                        off + len(filtered)
                    ) * self._staging_group_expansion(group_index) > kernel_capacity:
                        fits = False
                        break
                    staging_slots[group_index] = (off, filtered)
                    cursors[group_index] = off + len(filtered)
                if fits:
                    pull_meta.staging_slots = staging_slots
                else:
                    pull_meta.staging_slots = None
                    overflow.append(req_id)
            return overflow
    def _staging_slot_block_ids(self, pull_meta: PullReqMeta) -> list[list[int]]:
            """Per-group slot ids to advertise in place of real block ids."""
            slots = pull_meta.staging_slots
            assert slots is not None
            out: list[list[int]] = []
            for group_index in range(len(pull_meta.local_block_ids)):
                if group_index in slots:
                    off, filtered = slots[group_index]
                    out.append(list(range(off, off + len(filtered))))
                else:
                    out.append([])
            return out
    def _staging_group_expansion(self, group_index: int) -> int:
            """Physical kernel blocks per logical block for one KV group.
    
            Mirrors _logical_to_kernel_block_ids: only attention groups expand;
            Mamba/GDN state blocks stay in logical/page-id space.
            """
            spec = self.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec
            if isinstance(spec, MambaSpec):
                return 1
            return self._physical_blocks_per_logical_kv_block
    def _staging_h2d_copy_request(self, pull_meta: PullReqMeta) -> None:
            """Copy one request's staged KV from the recv window to the GPU cache.
    
            Runs on the receiver thread after the producer's sync writes for the
            request completed. Replays the expanded-region geometry: window slot
            column <-> real kernel block id, per region of each KV group.
            Only homogeneous-TP pairings are supported (TP=1<->1 here): the plan
            offsets are 0 and each window slot maps 1:1 to a real block.
            """
            assert pull_meta.staging_slots is not None
            arena = self._staging_recv_arena
            assert arena is not None
            current_platform.set_device(self.device_id)
            n_copies = 0
            total_bytes = 0
            start = time.perf_counter()
            for group_index, (slot_off, filtered) in pull_meta.staging_slots.items():
                # Align the group index so Mamba/GDN groups are not expanded
                # (matching the producer's per-group handling of the slot ids
                # we advertised).
                padded: list[list[int]] = [
                    [] for _ in self.kv_cache_config.kv_cache_groups
                ]
                padded[group_index] = filtered
                kernel_ids = self._logical_to_kernel_block_ids(padded)[group_index]
                kernel_slot_base = slot_off * self._staging_group_expansion(
                    group_index
                )
                for w_region, r_region in zip(
                    self._staging_window_regions, self._staging_real_regions
                ):
                    if w_region.group_index != group_index:
                        continue
                    nbytes = w_region.kv_block_len
                    for idx, kernel_block_id in enumerate(kernel_ids):
                        slot = kernel_slot_base + idx
                        arena.h2d(
                            slot * w_region.block_len,
                            r_region.base_addr + kernel_block_id * r_region.block_len,
                            nbytes,
                        )
                        n_copies += 1
                        total_bytes += nbytes
            arena.synchronize()
            duration = time.perf_counter() - start
            logger.debug(
                "host_staging: H2D copy for req %s done (%d copies, %.2f MiB, "
                "%.1f ms, %.1f GiB/s)",
                pull_meta.d_req_id,
                n_copies,
                total_bytes / 2**20,
                duration * 1000,
                total_bytes / max(duration, 1e-9) / 2**30,
            )
