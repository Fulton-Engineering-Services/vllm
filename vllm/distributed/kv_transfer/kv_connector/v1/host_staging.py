# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable host staging for NIXL/Mooncake KV transfer on unified-memory GPUs.

On platforms such as GB10 / DGX Spark, GPUDirect RDMA is unavailable because
unified memory cannot be exported via DMA-buf.  The sanctioned fallback is to
stage transfers through cudaHostAlloc-style pinned host buffers.  This module
provides an engine-agnostic arena/window abstraction that can be wired into
Mooncake, NIXL KV, EPLB, and (when supported) NIXL EP.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

logger = init_logger(__name__)

# CUdevice_attribute enum value for DMA-buf support.  The symbolic constant is
# not exposed by all cupy builds, so we carry the numeric value inline.
_CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124


class PinnedHostStagingArena:
    """Pinned host buffer that can be registered with a transfer engine.

    The backing store is owned by ``torch`` (cheap allocator bookkeeping) and
    exposed to ``cupy`` through ``UnownedMemory`` so that we get a real copy
    API.  All copies run on a private non-blocking cupy stream.
    """

    def __init__(
        self,
        num_bytes: int,
        label: str,
        register_fn: Callable[[int, int], None] | None = None,
    ):
        try:
            import cupy as cp
        except ImportError as e:
            raise RuntimeError(
                "Host staging requires cupy (e.g. cupy-cuda13x). "
                "Install cupy or set host_staging='off'."
            ) from e

        if num_bytes <= 0:
            raise ValueError(f"Staging arena size must be positive, got {num_bytes}")

        self.cupy = cp
        self.label = label
        self.num_bytes = num_bytes

        # ``pin_memory=True`` is a no-op on CPU-only torch builds but is required
        # on CUDA builds to allocate page-locked memory.
        self.buf = torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)
        self.base_ptr = int(self.buf.data_ptr())

        self._unowned = self.cupy.cuda.UnownedMemory(
            self.base_ptr, num_bytes, owner=None
        )
        self.stream = self.cupy.cuda.Stream(non_blocking=True)

        if register_fn is not None:
            register_fn(self.base_ptr, num_bytes)

        logger.info(
            "Host staging: allocated %s pinned arena at %#x (%d MiB)",
            label,
            self.base_ptr,
            num_bytes // 2**20,
        )

    def register_with(self, registrar: Callable[[int, int], None]) -> None:
        """Register the arena with a transfer-engine specific callback."""
        registrar(self.base_ptr, self.num_bytes)

    def _device_mp(self, gpu_ptr: int, nbytes: int):
        """Copy-capable pointer over an external device buffer address."""
        return self.cupy.cuda.MemoryPointer(
            self.cupy.cuda.UnownedMemory(gpu_ptr, nbytes, owner=None), 0
        )

    def d2h(self, gpu_ptr: int, offset: int, nbytes: int) -> None:
        """Async-copy ``nbytes`` from a device buffer into the arena."""
        if offset < 0 or offset + nbytes > self.num_bytes:
            raise ValueError(
                f"Arena d2h out of bounds: offset={offset}, "
                f"nbytes={nbytes}, arena={self.num_bytes}"
            )
        pinned_mp = self.cupy.cuda.MemoryPointer(self._unowned, offset)
        pinned_mp.copy_from_device_async(
            self._device_mp(gpu_ptr, nbytes), nbytes, self.stream
        )

    def h2d(self, offset: int, gpu_ptr: int, nbytes: int) -> None:
        """Async-copy ``nbytes`` from the arena to a device buffer."""
        if offset < 0 or offset + nbytes > self.num_bytes:
            raise ValueError(
                f"Arena h2d out of bounds: offset={offset}, "
                f"nbytes={nbytes}, arena={self.num_bytes}"
            )
        dev_dst_mp = self._device_mp(gpu_ptr, nbytes)
        dev_dst_mp.copy_from_host_async(
            self.base_ptr + offset, nbytes, self.stream
        )

    def synchronize(self) -> None:
        self.stream.synchronize()


@dataclass(frozen=True)
class HostStagingRegion:
    """Minimal region description used by ``HostStagingWindow``.

    Engine-specific adapters can return their own region-like objects as long
    as they expose the same attributes.
    """

    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0


class HostStagingWindow:
    """Carve a receive arena into slots and replay H2D copies after a transfer.

    The window mirrors the real per-layer block layout in pinned host memory.
    Producers write into window slots (using the same logical/kernel block
    arithmetic as real blocks); consumers then replay H2D copies from those
    slots back to the GPU cache.
    """

    def __init__(
        self,
        arena: PinnedHostStagingArena,
        block_len_per_layer: Sequence[int],
        kv_block_len_per_layer: Sequence[int],
        region_builder_fn: Callable[
            [Sequence[int], Sequence[int], Sequence[int]],
            Sequence[Any],
        ],
        real_region_builder_fn: Callable[[], Sequence[Any]],
        group_expansion_fn: Callable[[int], int],
        logical_to_kernel_block_ids_fn: Callable[[int, Sequence[int]], list[int]],
        device_id: int,
        label: str = "recv",
        num_blocks: int | None = None,
    ):
        if len(block_len_per_layer) != len(kv_block_len_per_layer):
            raise ValueError(
                "block_len_per_layer and kv_block_len_per_layer must match"
            )

        self.arena = arena
        self.block_len_per_layer = list(block_len_per_layer)
        self.kv_block_len_per_layer = list(kv_block_len_per_layer)
        self.region_builder_fn = region_builder_fn
        self.real_region_builder_fn = real_region_builder_fn
        self.group_expansion_fn = group_expansion_fn
        self.logical_to_kernel_block_ids_fn = logical_to_kernel_block_ids_fn
        self.device_id = device_id
        self.label = label
        self._num_blocks = num_blocks

        self._init_window()

    def _init_window(self) -> None:
        col_bytes = sum(self.block_len_per_layer)
        if col_bytes <= 0:
            raise RuntimeError(
                f"Host staging window '{self.label}': empty KV geometry."
            )

        self.blocks_per_region = self.arena.num_bytes // col_bytes
        if self.blocks_per_region == 0:
            raise RuntimeError(
                f"Host staging window '{self.label}': arena "
                f"({self.arena.num_bytes} bytes) smaller than one block "
                f"column ({col_bytes} bytes)."
            )

        bases: list[int] = []
        offset = 0
        for block_len in self.block_len_per_layer:
            bases.append(self.arena.base_ptr + offset)
            offset += block_len * self.blocks_per_region

        if offset > self.arena.num_bytes:
            raise RuntimeError(
                f"Host staging window '{self.label}': geometry overflow "
                f"({offset} > {self.arena.num_bytes})."
            )

        self.window_bases = bases
        self.window_regions = list(
            self.region_builder_fn(
                bases, self.block_len_per_layer, self.kv_block_len_per_layer
            )
        )
        self.real_regions = list(self.real_region_builder_fn())

        # Logical slot capacity advertised to remote producers.  It is capped by
        # the physical arena size so that descriptor IDs always map to valid
        # memory.
        self.num_blocks = (
            min(self._num_blocks, self.blocks_per_region)
            if self._num_blocks is not None
            else self.blocks_per_region
        )

        # Persistent per-group free-list allocator.  Each group owns a set of
        # currently free logical slot offsets.  init/release are protected by
        # _slot_lock because the worker main thread and the NIXL notification
        # thread both touch the allocator.
        self._slot_lock = threading.Lock()
        max_group_index = max(
            (getattr(r, "group_index", 0) for r in self.window_regions),
            default=0,
        )
        self._num_groups = max_group_index + 1
        self._free_slots: list[set[int]] = [
            set(range(self._logical_capacity(g)))
            for g in range(self._num_groups)
        ]

        # Map a group index to the list of window region indices that belong to
        # it.  Used to expand logical slot offsets into per-region descriptor
        # IDs in engine-specific adapters.
        self._regions_by_group: list[list[int]] = [
            [] for _ in range(self._num_groups)
        ]
        for idx, region in enumerate(self.window_regions):
            gidx = getattr(region, "group_index", 0)
            self._regions_by_group[gidx].append(idx)

        logger.info(
            "Host staging window '%s': %d kernel blocks per region across "
            "%d regions (%d logical slots, %d MiB).",
            self.label,
            self.blocks_per_region,
            len(self.window_bases),
            self.num_blocks,
            offset // 2**20,
        )

    def _logical_capacity(self, group_index: int) -> int:
        expansion = max(1, self.group_expansion_fn(group_index))
        return min(self.blocks_per_region // expansion, self.num_blocks)

    def _find_contiguous_free_slots(
        self, group_index: int, num_slots: int
    ) -> tuple[int, bool]:
        """Find ``num_slots`` contiguous free slot offsets in a group.

        Returns ``(start_offset, found)``.  The free set is left untouched;
        callers must remove the range after a successful assignment.
        """
        free = self._free_slots[group_index]
        if num_slots > len(free):
            return (0, False)
        if num_slots == 0:
            return (0, True)
        sorted_free = sorted(free)
        # Look for a run of ``num_slots`` consecutive integers.
        run_start = sorted_free[0]
        run_len = 1
        for slot in sorted_free[1:]:
            if slot == run_start + run_len:
                run_len += 1
                if run_len == num_slots:
                    return (run_start, True)
            else:
                run_start = slot
                run_len = 1
        return (0, False)

    def assign_slots(
        self, pull_metas: Mapping[str, Any]
    ) -> list[str]:
        """Assign receive-window slots to each request of a pull batch.

        ``pull_metas`` is a mapping from request id to an object with a
        ``local_block_ids`` attribute (list of per-KV-group block id lists).
        On success the object's ``staging_slots`` attribute is set to
        ``{group_index: (slot_offset, filtered_block_ids)}``; on overflow it is
        set to ``None``.

        The allocator is persistent across calls: slots stay reserved until
        ``release_slots`` returns them.  This is required for asynchronous
        transfers where the consumer must keep the receive window valid until
        the completion notification arrives and H2D replay finishes.

        Returns the request ids that did NOT fit in the window.
        """
        overflow: list[str] = []

        with self._slot_lock:
            for req_id, pull_meta in pull_metas.items():
                local_block_ids = pull_meta.local_block_ids
                staging_slots: dict[int, tuple[int, list[int]]] = {}
                freed_on_failure: list[tuple[int, int, int]] = []
                fits = True

                for group_index, group in enumerate(local_block_ids):
                    # Filter NULL-block placeholders (mamba/GDN align-mode) so
                    # slot ids stay 1:1 with the producer's kernel indices.
                    filtered = [b for b in group if b != NULL_BLOCK_ID]
                    if not filtered:
                        continue

                    off, ok = self._find_contiguous_free_slots(
                        group_index, len(filtered)
                    )
                    if not ok:
                        fits = False
                        break

                    staging_slots[group_index] = (off, filtered)
                    # Remove the allocated range from the free set.
                    for s in range(off, off + len(filtered)):
                        self._free_slots[group_index].discard(s)
                    freed_on_failure.append(
                        (group_index, off, len(filtered))
                    )

                if fits:
                    pull_meta.staging_slots = staging_slots
                else:
                    pull_meta.staging_slots = None
                    overflow.append(req_id)
                    # Roll back any partial allocation for this request.
                    for group_index, off, length in freed_on_failure:
                        for s in range(off, off + length):
                            self._free_slots[group_index].add(s)

        return overflow

    def release_slots(self, pull_meta: Any) -> None:
        """Return the slots allocated to ``pull_meta`` back to the free pool.

        It is safe to call this for a request that was never assigned slots
        (it is a no-op).  The lock makes the call safe from the notification
        thread that runs H2D replay.
        """
        slots = getattr(pull_meta, "staging_slots", None) or {}
        if not slots:
            return

        with self._slot_lock:
            for group_index, (off, filtered) in slots.items():
                for s in range(off, off + len(filtered)):
                    self._free_slots[group_index].add(s)

    def slot_block_ids(self, pull_meta: Any) -> list[list[int]]:
        """Per-group slot ids to advertise in place of real block ids."""
        slots = pull_meta.staging_slots
        if slots is None:
            raise RuntimeError("slot_block_ids called with no assigned slots")
        num_groups = len(pull_meta.local_block_ids)
        out: list[list[int]] = [[] for _ in range(num_groups)]
        for group_index, (off, filtered) in slots.items():
            out[group_index] = list(range(off, off + len(filtered)))
        return out

    def replay_h2d(self, pull_meta: Any) -> None:
        """Copy one request's staged KV from the recv window to the GPU cache."""
        slots = pull_meta.staging_slots
        if slots is None:
            raise RuntimeError("replay_h2d called with no assigned slots")

        # Make sure the copy targets the right CUDA device on multi-GPU nodes.
        try:
            import cupy as cp
        except ImportError:
            cp = None

        if cp is not None:
            with cp.cuda.Device(self.device_id):
                pass

        for group_index, (slot_off, filtered) in slots.items():
            kernel_ids = self.logical_to_kernel_block_ids_fn(
                group_index, filtered
            )
            kernel_slot_base = slot_off * self.group_expansion_fn(group_index)

            for w_region, r_region in zip(self.window_regions, self.real_regions):
                if getattr(w_region, "group_index", 0) != group_index:
                    continue

                nbytes = getattr(w_region, "kv_block_len", w_region.block_len)
                for idx, kernel_block_id in enumerate(kernel_ids):
                    if kernel_block_id < 0 or kernel_block_id >= self.blocks_per_region:
                        raise ValueError(
                            f"replay_h2d: kernel_block_id {kernel_block_id} "
                            f"is out of range [0, {self.blocks_per_region})."
                        )
                    slot = kernel_slot_base + idx
                    self.arena.h2d(
                        slot * w_region.block_len,
                        r_region.base_addr + kernel_block_id * r_region.block_len,
                        nbytes,
                    )

        self.arena.synchronize()


class HostStagingSendPlanner:
    """Chunk a list of GPU source pointers through a pinned send arena."""

    def __init__(self, arena: PinnedHostStagingArena):
        self.arena = arena

    def send_blocks(
        self,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        transfer_fn: Callable[[list[int], list[int], list[int]], int],
    ) -> int:
        """Stage D2H copies and call ``transfer_fn`` for each arena-sized chunk.

        ``transfer_fn(chunk_src, chunk_dst, chunk_len)`` is synchronous and
        returns an engine-specific status code (0 == success).  On first
        non-zero status the loop stops and that status is returned.
        """
        if not (len(src_ptrs) == len(dst_ptrs) == len(lengths)):
            raise ValueError("src_ptrs, dst_ptrs, and lengths must have same length")

        # Copy the input lists so partial-descriptor chunking does not mutate
        # the caller's data — callers may retry or inspect the originals.
        src_ptrs = list(src_ptrs)
        dst_ptrs = list(dst_ptrs)
        lengths = list(lengths)

        i = 0
        n = len(src_ptrs)
        while i < n:
            chunk_src: list[int] = []
            chunk_dst: list[int] = []
            chunk_len: list[int] = []
            pos = 0

            while i < n and pos < self.arena.num_bytes:
                length = lengths[i]
                remaining = self.arena.num_bytes - pos
                take = min(length, remaining)

                self.arena.d2h(
                    src_ptrs[i] + (lengths[i] - length),
                    pos,
                    take,
                )
                chunk_src.append(self.arena.base_ptr + pos)
                chunk_dst.append(dst_ptrs[i] + (lengths[i] - length))
                chunk_len.append(take)
                pos += take

                if take == length:
                    i += 1
                else:
                    # Partial descriptor: continue in the next chunk.
                    lengths[i] = length - take
                    src_ptrs[i] += take
                    dst_ptrs[i] += take

            self.arena.synchronize()
            if chunk_src:
                ret = transfer_fn(chunk_src, chunk_dst, chunk_len)
                if ret != 0:
                    return ret

        return 0


def chunk_descriptor_ranges(
    desc_lengths: Sequence[int],
    chunk_size: int,
) -> Iterable[list[tuple[int, int, int]]]:
    """Split descriptor byte ranges into arena-sized contiguous chunks.

    Each input descriptor has a byte length.  The function yields one chunk
    at a time, where each chunk is a list of ``(desc_index,
    byte_offset_inside_descriptor, bytes_to_take)`` tuples and the sum of
    ``bytes_to_take`` in a chunk is at most ``chunk_size``.

    This is used by engine-specific adapters (NIXL) that register a pinned
    arena as one big contiguous region but must issue transfers in
    arena-sized slices.  A single descriptor whose length exceeds
    ``chunk_size`` cannot be staged and causes ``ValueError``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunk: list[tuple[int, int, int]] = []
    pos = 0

    for desc_idx, length in enumerate(desc_lengths):
        if length < 0:
            raise ValueError(f"Descriptor length must be non-negative, got {length}")
        if length == 0:
            continue
        if length > chunk_size:
            raise ValueError(
                f"Descriptor {desc_idx} length {length} exceeds chunk size "
                f"{chunk_size}. Increase the staging arena size or reduce "
                f"the transfer element size."
            )

        offset = 0
        while offset < length:
            remaining_in_chunk = chunk_size - pos
            if remaining_in_chunk == 0:
                yield chunk
                chunk = []
                pos = 0
                remaining_in_chunk = chunk_size

            take = min(length - offset, remaining_in_chunk)
            chunk.append((desc_idx, offset, take))
            offset += take
            pos += take

    if chunk:
        yield chunk


class HostStagingPlatformProbe:
    """Probe whether host staging should be activated.

    Activation is controlled by ``kv_connector_extra_config["host_staging"]``:
      * ``"off"`` (default): disabled until all staged transfer paths are
        fully implemented.  Switch to ``"auto"`` once NIXL push/pull/EPLB
        staged paths are complete.
      * ``"auto"``: enable when DMA-buf export is unavailable.
      * ``"on"`` / ``"off"``: force enable/disable.

    Legacy boolean values in config are accepted for backward compatibility.
    """

    def __init__(self, extra_config: Mapping[str, Any] | None = None):
        self.extra_config = dict(extra_config or {})
        raw = self.extra_config.get("host_staging", "off")
        if isinstance(raw, bool):
            self._override = "on" if raw else "off"
        else:
            self._override = str(raw).lower().strip()
        if self._override not in ("auto", "on", "off"):
            logger.warning(
                "Unknown host_staging value '%s'; defaulting to 'auto'.", raw
            )
            self._override = "auto"

    def enabled(self, device_id: int = 0) -> bool:
        if self._override == "on":
            return True
        if self._override == "off":
            return False
        return not self._dma_buf_supported(device_id)

    @staticmethod
    def _dma_buf_supported(device_id: int) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            import cupy as cp
        except ImportError:
            return False

        attr = getattr(
            cp.cuda.runtime,
            "CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED",
            _CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
        )
        try:
            return bool(cp.cuda.runtime.deviceGetAttribute(attr, device_id))
        except Exception as exc:
            logger.debug(
                "Host staging DMA-buf probe failed for device %s: %s",
                device_id,
                exc,
            )
            return False
