# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-staging (GB10 / unified memory) support for Mooncake KV transfer.

GPUDirect is architecturally unavailable on unified-memory platforms
(GB10 / DGX Spark) — nvidia-peermem does not load, dma-buf export returns
CUDA_ERROR_INVALID_VALUE, and ibv_reg_mr on CUDA VAs fails with EFAULT.
The NVIDIA-sanctioned fallback (DGX Spark Porting Guide) is cudaHostAlloc
pinned buffers registered with ib_reg_mr, which the transfer engine handles
cleanly.

This module is the entire host-staging implementation: the pinned-arena
abstraction plus the Worker mixin that plugs into every relevant Worker
method.  Zero send/receive protocol logic lives here — the mixin overrides
only the three choke points: register_kv_caches, _send_blocks, and
receive_kv / process_pulling_result entry.
"""
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

from ._protocol import PullReqMeta, TransferRegion
from ._transfer_planning import _expand_transfer_regions
from .stats import MooncakeKVConnectorStats

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# PinnedStagingArena — transfer-engine-registered pinned host buffer
# ---------------------------------------------------------------------------


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

            self.buf = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
            self.cupy = cp
        except ImportError as e:
            raise RuntimeError(
                "Mooncake host_staging requires cupy (cupy-cuda13x) and torch"
            ) from e
        self.base_ptr = int(self.buf.data_ptr())
        self.num_bytes = num_bytes
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
        return self.cupy.cuda.MemoryPointer(
            self.cupy.cuda.UnownedMemory(gpu_ptr, nbytes, None), 0
        )

    def d2h(self, gpu_ptr: int, offset: int, nbytes: int) -> None:
        pinned_mp = self.cupy.cuda.MemoryPointer(self._unowned, offset)
        pinned_mp.copy_from_device_async(
            self._device_mp(gpu_ptr, nbytes), nbytes, self.stream
        )

    def h2d(self, offset: int, gpu_ptr: int, nbytes: int) -> None:
        dev_dst_mp = self._device_mp(gpu_ptr, nbytes)
        dev_dst_mp.copy_from_host_async(
            self.base_ptr + offset, nbytes, self.stream
        )

    def synchronize(self) -> None:
        self.stream.synchronize()


# ---------------------------------------------------------------------------
# StagingMixin — plugged into MooncakeConnectorWorker
# ---------------------------------------------------------------------------


class _StagingMixin:
    """All host-staging logic for MooncakeConnectorWorker.

    Intended as a mixin in the Worker class.  It expects the following
    attributes to be set by __init__ before any method is called:

      is_kv_producer / is_kv_consumer  (bool)
      host_staging  (bool)
      staging_send_mib / staging_recv_mib  (int)
      _staging_send_arena / _staging_recv_arena  (_PinnedStagingArena | None)
      _staging_window_bases  (list[int])
      _staging_window_blocks  (int)
      _staging_window_regions / _staging_real_regions  (list[TransferRegion])
      _logical_to_kernel_block_ids  (method)  — from the Worker main class
      _get_transfer_regions  (method)          — from _WorkerSendMixin
      kv_caches_base_addr  (list[int])
      block_len_per_layer / kv_block_len_per_layer  (list[int])
      registered_layer_names / registered_layer_indices  (list[int])
      registered_group_indices  (list[int])
      _physical_blocks_per_logical_kv_block  (int)
      kv_cache_config  (KVCacheConfig)
      _layer_specs  (dict[str, KVCacheSpec])
      _layer_group_indices  (dict[str, int])
    """

    # ── send-side ──────────────────────────────────────────────────────

    def _send_blocks_staged(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
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

    # ── registration-time setup ────────────────────────────────────────

    def _setup_host_staging(self) -> None:
        if not self.is_kv_consumer:
            self._staging_send_arena = _PinnedStagingArena(
                self.engine, self.staging_send_mib * 2**20, "send"
            )
        if not self.is_kv_producer:
            self._staging_recv_arena = _PinnedStagingArena(
                self.engine, self.staging_recv_mib * 2**20, "recv"
            )
            col_bytes = sum(self.block_len_per_layer)
            if col_bytes <= 0:
                raise RuntimeError(
                    "Mooncake host_staging: empty KV geometry at registration."
                )
            required_blocks = self._physical_blocks_per_logical_kv_block + 1
            raw_blocks = (self.staging_recv_mib * 2**20) // col_bytes
            if raw_blocks < required_blocks:
                raise RuntimeError(
                    "Mooncake host_staging: staging_recv_mib="
                    f"{self.staging_recv_mib} holds {raw_blocks} kernel "
                    f"blocks, but one attention logical block needs "
                    f"{required_blocks} (c="
                    f"{self._physical_blocks_per_logical_kv_block} + 1 "
                    "margin). Raise staging_recv_mib."
                )
            self._staging_window_blocks = raw_blocks
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

    # ── receive side: slot management / completion ─────────────────────

    def _staging_assign_slots(
        self, pull_metas: dict[str, PullReqMeta]
    ) -> list[str]:
        kernel_capacity = self._staging_window_blocks
        cursors: dict[int, int] = defaultdict(int)
        overflow: list[str] = []
        for req_id, pull_meta in pull_metas.items():
            staging_slots: dict[int, tuple[int, list[int]]] = {}
            fits = True
            for group_index, group in enumerate(pull_meta.local_block_ids):
                from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
                filtered = [b for b in group if b != NULL_BLOCK_ID]
                if not filtered:
                    continue
                expansion = self._staging_group_expansion(group_index)
                logical_capacity = kernel_capacity // expansion
                off = cursors[group_index]
                if off + len(filtered) > logical_capacity:
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
        from vllm.v1.kv_cache_interface import MambaSpec
        spec = self.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec
        if isinstance(spec, MambaSpec):
            return 1
        return self._physical_blocks_per_logical_kv_block

    def _staging_h2d_copy_request(self, pull_meta: PullReqMeta) -> None:
        assert pull_meta.staging_slots is not None
        arena = self._staging_recv_arena
        assert arena is not None
        current_platform.set_device(self.device_id)
        n_copies = 0
        total_bytes = 0
        import time
        start = time.perf_counter()
        for group_index, (slot_off, filtered) in pull_meta.staging_slots.items():
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