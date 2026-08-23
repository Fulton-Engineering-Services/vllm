# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake-specific wrappers around the reusable host-staging module.

This file is intentionally thin: the arena/window logic lives in
``vllm.distributed.kv_transfer.kv_connector.v1.host_staging`` and is reused
by Mooncake, NIXL KV, EPLB, and (eventually) NIXL EP.
"""

import time
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.host_staging import (
    HostStagingPlatformProbe,
    HostStagingSendPlanner,
    HostStagingWindow,
    PinnedHostStagingArena,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import MambaSpec

from ._protocol import PullReqMeta, ReqId

logger = init_logger(__name__)

__all__ = [
    "HostStagingPlatformProbe",
    "_PinnedStagingArena",
    "_StagingMixin",
]


class _PinnedStagingArena(PinnedHostStagingArena):
    """Pinned arena registered with the Mooncake TransferEngine."""

    def __init__(self, engine: Any, num_bytes: int, label: str):
        def _register(base_ptr: int, size: int) -> None:
            ret = engine.register_memory(base_ptr, size)
            if ret != 0:
                raise RuntimeError(
                    f"Mooncake host_staging: failed to register {label} staging "
                    f"arena ({size} bytes) with the transfer engine "
                    f"(ret={ret})."
                )
            logger.info(
                "Mooncake host_staging: registered %s pinned arena at %#x (%d MiB)",
                label,
                base_ptr,
                size // 2**20,
            )

        super().__init__(num_bytes, label, register_fn=_register)


class _StagingMixin:
    """Mooncake host-staging methods delegated to the shared window/planner."""

    _staging_send_arena: _PinnedStagingArena | None = None
    _staging_recv_arena: _PinnedStagingArena | None = None
    _staging_send_planner: HostStagingSendPlanner | None = None
    _staging_window: HostStagingWindow | None = None

    def _setup_host_staging(self) -> None:
        """Allocate and register pinned staging buffers for this role."""
        assert self.host_staging

        if not self.is_kv_consumer:
            self._staging_send_arena = _PinnedStagingArena(
                self.engine, self.staging_send_mib * 2**20, "send"
            )
            self._staging_send_planner = HostStagingSendPlanner(
                self._staging_send_arena
            )

        if not self.is_kv_producer:
            self._staging_recv_arena = _PinnedStagingArena(
                self.engine, self.staging_recv_mib * 2**20, "recv"
            )

            def _build_window_regions(
                bases: list[int],
                block_lens: list[int],
                kv_block_lens: list[int],
            ) -> list[Any]:
                return self._get_transfer_regions(
                    bases,
                    block_lens,
                    kv_block_lens,
                    self.registered_layer_names,
                    self.registered_layer_indices,
                    self.registered_group_indices,
                )

            def _build_real_regions() -> list[Any]:
                return self._get_transfer_regions(
                    self.kv_caches_base_addr,
                    self.block_len_per_layer,
                    self.kv_block_len_per_layer,
                    self.registered_layer_names,
                    self.registered_layer_indices,
                    self.registered_group_indices,
                )

            def _group_expansion(group_index: int) -> int:
                spec = self.kv_cache_config.kv_cache_groups[
                    group_index
                ].kv_cache_spec
                if isinstance(spec, MambaSpec):
                    return 1
                return self._physical_blocks_per_logical_kv_block

            def _logical_to_kernel(
                group_index: int, logical_ids: list[int]
            ) -> list[int]:
                padded: list[list[int]] = [
                    [] for _ in self.kv_cache_config.kv_cache_groups
                ]
                padded[group_index] = list(logical_ids)
                return self._logical_to_kernel_block_ids(padded)[group_index]

            self._staging_window = HostStagingWindow(
                arena=self._staging_recv_arena,
                block_len_per_layer=self.block_len_per_layer,
                kv_block_len_per_layer=self.kv_block_len_per_layer,
                region_builder_fn=_build_window_regions,
                real_region_builder_fn=_build_real_regions,
                group_expansion_fn=_group_expansion,
                logical_to_kernel_block_ids_fn=_logical_to_kernel,
                device_id=self.device_id,
            )
            self._staging_window_bases = self._staging_window.window_bases
            self._staging_window_blocks = self._staging_window.blocks_per_region
            self._staging_window_regions = self._staging_window.window_regions
            self._staging_real_regions = self._staging_window.real_regions

    def _send_blocks_staged(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        assert self._staging_send_planner is not None

        total_bytes = sum(lengths)
        total_descs = len(src_ptrs)

        def _transfer(
            chunk_src: list[int], chunk_dst: list[int], chunk_len: list[int]
        ) -> int:
            return self.engine.batch_transfer_sync_write(
                remote_session, chunk_src, chunk_dst, chunk_len
            )

        start_time = time.perf_counter()
        ret = self._staging_send_planner.send_blocks(
            src_ptrs, dst_ptrs, lengths, _transfer
        )
        duration = time.perf_counter() - start_time
        if ret == 0:
            self.xfer_stats.record_transfer(
                duration_s=duration,
                total_bytes=total_bytes,
                num_descs=total_descs,
            )
        else:
            self.xfer_stats.record_failed_transfer()
        return ret

    def _staging_assign_slots(
        self, pull_metas: dict[ReqId, PullReqMeta]
    ) -> list[ReqId]:
        assert self._staging_window is not None
        return self._staging_window.assign_slots(pull_metas)

    def _staging_slot_block_ids(self, pull_meta: PullReqMeta) -> list[list[int]]:
        assert self._staging_window is not None
        return self._staging_window.slot_block_ids(pull_meta)

    def _staging_h2d_copy_request(self, pull_meta: PullReqMeta) -> None:
        assert self._staging_window is not None
        self._staging_window.replay_h2d(pull_meta)
