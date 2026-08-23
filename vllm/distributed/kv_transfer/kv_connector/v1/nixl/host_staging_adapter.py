# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-specific adapter around the reusable host-staging module.

This is intentionally a skeleton: Phase 3-7 of the reusable host-staging plan
still need to wire push/pull descriptor mapping and EPLB/EP all2all support.
The adapter already allocates and registers pinned host arenas with NIXL so
that follow-up work only has to integrate the transfer calls.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.host_staging import (
    HostStagingPlatformProbe,
    HostStagingRegion,
    HostStagingSendPlanner,
    HostStagingWindow,
    PinnedHostStagingArena,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

logger = init_logger(__name__)


class NixlHostStagingAdapter:
    """Adapter that stages NIXL KV transfers through pinned host memory.

    On unified-memory platforms (GB10 / DGX Spark) NIXL cannot register GPU
    memory for RDMA.  This adapter mirrors the Mooncake staging approach:
    producers copy into a pinned send arena, NIXL WRITEs into the consumer's
    pinned receive window, and consumers replay H2D into the real GPU cache.
    """

    def __init__(self, worker: "NixlBaseConnectorWorker"):
        self.worker = worker
        extra_config = worker.kv_transfer_config.kv_connector_extra_config or {}
        self.probe = HostStagingPlatformProbe(extra_config)
        self.enabled = self.probe.enabled(worker.device_id)

        self.send_arena: PinnedHostStagingArena | None = None
        self.recv_arena: PinnedHostStagingArena | None = None
        self.recv_window: HostStagingWindow | None = None
        self.send_planner: HostStagingSendPlanner | None = None
        self.send_mib = int(extra_config.get("staging_send_mib", 512))
        self.recv_mib = int(extra_config.get("staging_recv_mib", 6144))

        # NIXL handles built over the staging arenas/windows.
        self.send_arena_handle: int | None = None
        self.recv_window_handle: int | None = None
        # Per-region list of descriptor IDs inside the recv window handle.
        # window_desc_ids[region_index][slot_offset] -> NIXL descriptor id
        # relative to recv_window_handle.
        self.window_desc_ids: list[list[int]] = []

        self._registered_descs: list[Any] = []

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Allocate pinned staging arenas and register them with NIXL.

        The receive-window geometry is built from the same per-layer block
        metadata used for direct GPU registration.  Producers will advertise
        ``staging_window_base_addrs`` in ``NixlAgentMetadata`` instead of
        ``kv_caches_base_addr`` when this adapter is enabled.
        """
        if not self.enabled:
            return

        logger.info(
            "NIXL host staging enabled on device %s (send=%d MiB, recv=%d MiB)",
            self.worker.device_id,
            self.send_mib,
            self.recv_mib,
        )

        # Allocate the send arena on producer/kv_both roles.
        if self.worker.kv_transfer_config.kv_role != "kv_consumer":
            self.send_arena = PinnedHostStagingArena(
                self.send_mib * 2**20,
                "nixl-send",
                register_fn=self._register_arena_with_nixl,
            )
            self.send_planner = HostStagingSendPlanner(self.send_arena)
            self.send_arena_handle = self._build_send_arena_handle()

        # Allocate the receive window on consumer/kv_both roles.
        if self.worker.kv_transfer_config.kv_role != "kv_producer":
            self.recv_arena = PinnedHostStagingArena(
                self.recv_mib * 2**20,
                "nixl-recv",
                register_fn=self._register_arena_with_nixl,
            )
            self.recv_window = self._build_recv_window(kv_caches)
            self.recv_window_handle, self.window_desc_ids = (
                self._build_recv_window_handle()
            )

    def _register_arena_with_nixl(self, base_ptr: int, size: int) -> None:
        """Register a pinned host arena as DRAM with the NIXL agent."""
        descs = self.worker.nixl_wrapper.get_reg_descs(
            [(base_ptr, size, 0, "")], "DRAM"
        )
        self.worker.nixl_wrapper.register_memory(descs, backends=self.worker.nixl_backends)
        self._registered_descs.append(descs)
        logger.info(
            "NIXL host staging: registered pinned arena at %#x (%d MiB) as DRAM",
            base_ptr,
            size // 2**20,
        )

    def _build_send_arena_handle(self) -> int:
        """Prepare a NIXL dlist handle over the contiguous send arena.

        The handle is registered as one big DRAM descriptor.  Individual
        transfers select sub-ranges via descriptor-id offsets computed from
        the arena base pointer.
        """
        assert self.send_arena is not None
        blocks_data = np.array(
            [[self.send_arena.base_ptr, self.send_arena.num_bytes, 0]],
            dtype=np.uint64,
        )
        descs = self.worker.nixl_wrapper.get_xfer_descs(blocks_data, "DRAM")
        return self.worker.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)

    def _build_recv_window(self, kv_caches: dict[str, Any]) -> HostStagingWindow:
        """Build the host-staging receive window from KV cache geometry."""
        assert self.recv_arena is not None

        def region_builder(
            bases: list[int],
            block_lens: list[int],
            kv_block_lens: list[int],
        ) -> list[Any]:
            return [
                HostStagingRegion(
                    base_addr=base,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                    group_index=idx,
                )
                for idx, (base, block_len, kv_block_len) in enumerate(
                    zip(bases, block_lens, kv_block_lens)
                )
            ]

        def real_region_builder() -> list[Any]:
            return [
                HostStagingRegion(
                    base_addr=addr,
                    block_len=block_len,
                    kv_block_len=block_len,
                )
                for addr, block_len in zip(
                    self.worker.kv_caches_base_addr[self.worker.engine_id][
                        self.worker.tp_rank
                    ],
                    self.worker.block_len_per_layer,
                )
            ]

        def group_expansion(group_index: int) -> int:
            return self.worker._physical_blocks_per_logical_kv_block

        def logical_to_kernel(
            group_index: int, logical_ids: list[int]
        ) -> list[int]:
            padded: list[list[int]] = [
                [] for _ in self.worker.kv_cache_config.kv_cache_groups
            ]
            padded[group_index] = list(logical_ids)
            return self.worker._logical_to_kernel_block_ids(
                padded, self.worker._physical_blocks_per_logical_kv_block
            )[group_index]

        return HostStagingWindow(
            arena=self.recv_arena,
            block_len_per_layer=self.worker.block_len_per_layer,
            # NIXL packs K and V into the content dim, so the full block
            # stride is also the KV copy length.
            kv_block_len_per_layer=self.worker.block_len_per_layer,
            region_builder_fn=region_builder,
            real_region_builder_fn=real_region_builder,
            group_expansion_fn=group_expansion,
            logical_to_kernel_block_ids_fn=logical_to_kernel,
            device_id=self.worker.device_id,
            label="nixl-recv",
        )

    def _build_recv_window_handle(self) -> tuple[int, list[list[int]]]:
        """Prepare a NIXL dlist handle over the receive window slots.

        Returns the NIXL handle and a per-region list of descriptor IDs that
        map one-to-one with window slots.  Producers issue WRITEs using
        these descriptor IDs as the remote side.
        """
        assert self.recv_window is not None
        blocks_data: list[tuple[int, int, int]] = []
        desc_ids: list[list[int]] = []
        cursor = 0
        for region in self.recv_window.window_regions:
            region_desc_ids: list[int] = []
            for slot in range(self.recv_window.blocks_per_region):
                blocks_data.append(
                    (
                        region.base_addr + slot * region.block_len,
                        region.block_len,
                        0,
                    )
                )
                region_desc_ids.append(cursor)
                cursor += 1
            desc_ids.append(region_desc_ids)
        arr = np.array(blocks_data, dtype=np.uint64)
        descs = self.worker.nixl_wrapper.get_xfer_descs(arr, "DRAM")
        handle = self.worker.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        return handle, desc_ids

    def get_staging_window_base_addrs(self) -> list[int] | None:
        if self.recv_window is None:
            return None
        return self.recv_window.window_bases

    def stage_push_xfer(
        self,
        src_ptrs: list[int],
        dst_window_desc_ids: list[int],
        lengths: list[int],
        transfer_fn: Callable[[list[int], list[int], list[int]], int],
    ) -> int:
        """Stage a push-mode transfer through the send arena.

        Copies source GPU blocks into the send arena and invokes
        ``transfer_fn(arena_desc_ids, dst_desc_ids, lengths)`` for each
        arena-sized chunk.  This is a stub: it currently raises
        ``NotImplementedError``; Phase 3 will implement the full D2H+WRITE
        path.
        """
        raise NotImplementedError(
            "NIXL staged push transfer is not fully implemented yet."
        )

    def stage_pull_request(
        self,
        pull_meta: Any,
        remote_block_ids: list[list[int]],
    ) -> dict[str, Any]:
        """Build a staged-pull registration payload for a consumer request.

        Assigns receive-window slots and returns the registration data that
        the consumer sends to the producer.  This is a stub for Phase 4.
        """
        raise NotImplementedError(
            "NIXL staged pull request is not fully implemented yet."
        )

    def replay_recv_h2d(self, pull_meta: Any) -> None:
        """Replay H2D copies from the receive window to the GPU cache."""
        if self.recv_window is None:
            raise RuntimeError("replay_recv_h2d called without a receive window")
        self.recv_window.replay_h2d(pull_meta)

    def shutdown(self) -> None:
        """Release NIXL staging resources."""
        if self.send_arena_handle is not None:
            with contextlib.suppress(Exception):
                self.worker.nixl_wrapper.release_dlist_handle(self.send_arena_handle)
            self.send_arena_handle = None
        if self.recv_window_handle is not None:
            with contextlib.suppress(Exception):
                self.worker.nixl_wrapper.release_dlist_handle(self.recv_window_handle)
            self.recv_window_handle = None
        for descs in self._registered_descs:
            with contextlib.suppress(Exception):
                self.worker.nixl_wrapper.deregister_memory(descs)
        self._registered_descs.clear()
