# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the reusable host-staging module."""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.host_staging import (
    HostStagingPlatformProbe,
    HostStagingRegion,
    HostStagingSendPlanner,
    HostStagingWindow,
    PinnedHostStagingArena,
    chunk_descriptor_ranges,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


class _FakePullMeta:
    def __init__(self, local_block_ids: list[list[int]]):
        self.local_block_ids = local_block_ids
        self.staging_slots: dict[int, tuple[int, list[int]]] | None = None


class _MockArena:
    """Arena that records copies without touching real GPU memory."""

    def __init__(self, num_bytes: int, base_ptr: int = 0x1000_0000):
        self.num_bytes = num_bytes
        self.base_ptr = base_ptr
        self.d2h_calls: list[tuple[int, int, int]] = []
        self.h2d_calls: list[tuple[int, int, int]] = []
        self.sync_count = 0

    def d2h(self, gpu_ptr: int, offset: int, nbytes: int) -> None:
        self.d2h_calls.append((gpu_ptr, offset, nbytes))

    def h2d(self, offset: int, gpu_ptr: int, nbytes: int) -> None:
        self.h2d_calls.append((offset, gpu_ptr, nbytes))

    def synchronize(self) -> None:
        self.sync_count += 1


def _make_window(arena: _MockArena):
    block_len = 256
    kv_block_len = 128

    def region_builder(bases, block_lens, kv_block_lens):
        return [
            HostStagingRegion(
                base_addr=bases[0],
                block_len=block_lens[0],
                kv_block_len=kv_block_lens[0],
                group_index=0,
            )
        ]

    def real_region_builder():
        return [
            HostStagingRegion(
                base_addr=0x2000_0000,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=0,
            )
        ]

    return HostStagingWindow(
        arena=arena,
        block_len_per_layer=[block_len],
        kv_block_len_per_layer=[kv_block_len],
        region_builder_fn=region_builder,
        real_region_builder_fn=real_region_builder,
        group_expansion_fn=lambda _group_index: 1,
        logical_to_kernel_block_ids_fn=lambda _group_index, ids: list(ids),
        device_id=0,
    )


def test_window_capacity_and_bases():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)
    # 2560 bytes / 256 byte column = 10 kernel blocks per region.
    assert window.blocks_per_region == 10
    assert window.window_bases == [arena.base_ptr]
    assert len(window.window_regions) == 1
    assert len(window.real_regions) == 1


def test_window_assign_slots_basic():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)

    # Use non-zero block ids because NULL_BLOCK_ID == 0 in this branch.
    metas = {
        "req_a": _FakePullMeta([[1, 2, 3]]),
        "req_b": _FakePullMeta([[4, 5]]),
    }
    overflow = window.assign_slots(metas)
    assert overflow == []
    assert metas["req_a"].staging_slots == {0: (0, [1, 2, 3])}
    assert metas["req_b"].staging_slots == {0: (3, [4, 5])}


def test_window_assign_slots_overflow():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)

    metas = {
        "fits": _FakePullMeta([[1, 2, 3]]),
        "overflow": _FakePullMeta([[i for i in range(1, 12)]]),
    }
    overflow = window.assign_slots(metas)
    assert overflow == ["overflow"]
    assert metas["fits"].staging_slots is not None
    assert metas["overflow"].staging_slots is None


def test_window_assign_slots_null_block_filtering():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)

    metas = {
        "req": _FakePullMeta([[NULL_BLOCK_ID, 1, 2, NULL_BLOCK_ID, 3]]),
    }
    overflow = window.assign_slots(metas)
    assert overflow == []
    # NULL blocks are filtered out before slot assignment.
    assert metas["req"].staging_slots == {0: (0, [1, 2, 3])}


def test_window_slot_block_ids():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)

    meta = _FakePullMeta([[1, 2, 3, 4]])
    overflow = window.assign_slots({"req": meta})
    assert overflow == []
    assert window.slot_block_ids(meta) == [[0, 1, 2, 3]]


def test_window_replay_h2d():
    arena = _MockArena(num_bytes=2560)
    window = _make_window(arena)

    meta = _FakePullMeta([[1, 2]])
    window.assign_slots({"req": meta})
    window.replay_h2d(meta)

    assert arena.sync_count == 1
    assert len(arena.h2d_calls) == 2
    # h2d is called with (arena_offset, gpu_dst_ptr, nbytes).
    # Slot offsets are 0-based within the receive window.
    assert arena.h2d_calls[0][0] == 0 * 256
    assert arena.h2d_calls[1][0] == 1 * 256
    # Real base + kernel block_id * block_len.
    assert arena.h2d_calls[0][1] == 0x2000_0000 + 1 * 256
    assert arena.h2d_calls[1][1] == 0x2000_0000 + 2 * 256
    # Copy length equals kv_block_len.
    assert arena.h2d_calls[0][2] == 128


def test_send_planner_basic():
    arena = _MockArena(num_bytes=512)
    planner = HostStagingSendPlanner(arena)

    transfers: list[tuple[list[int], list[int], list[int]]] = []

    def transfer_fn(src, dst, lengths):
        transfers.append((src, dst, lengths))
        return 0

    ret = planner.send_blocks(
        src_ptrs=[0x100, 0x200],
        dst_ptrs=[0x1000, 0x2000],
        lengths=[300, 300],
        transfer_fn=transfer_fn,
    )
    assert ret == 0
    assert len(transfers) == 2
    assert arena.sync_count == 2

    # First chunk is limited by arena capacity.
    src0, dst0, len0 = transfers[0]
    assert src0 == [arena.base_ptr + 0, arena.base_ptr + 300]
    assert dst0 == [0x1000, 0x2000]
    assert len0 == [300, 212]

    # Second chunk drains the tail of the second descriptor.
    src1, dst1, len1 = transfers[1]
    assert src1 == [arena.base_ptr + 0]
    assert dst1 == [0x2000 + 212]
    assert len1 == [88]


def test_send_planner_failure_stop():
    arena = _MockArena(num_bytes=512)
    planner = HostStagingSendPlanner(arena)

    def transfer_fn(src, dst, lengths):
        return 42

    ret = planner.send_blocks(
        src_ptrs=[0x100],
        dst_ptrs=[0x1000],
        lengths=[100],
        transfer_fn=transfer_fn,
    )
    assert ret == 42
    assert arena.sync_count == 1


def test_probe_override_values():
    assert HostStagingPlatformProbe({"host_staging": "on"}).enabled(0) is True
    assert HostStagingPlatformProbe({"host_staging": "OFF"}).enabled(0) is False
    assert HostStagingPlatformProbe({"host_staging": True}).enabled(0) is True
    assert HostStagingPlatformProbe({"host_staging": False}).enabled(0) is False


def test_probe_unknown_value_defaults_to_auto():
    # Unknown values should fall back to the same behavior as "auto".
    auto_probe = HostStagingPlatformProbe({"host_staging": "auto"})
    unknown_probe = HostStagingPlatformProbe({"host_staging": "maybe"})
    assert unknown_probe.enabled(0) == auto_probe.enabled(0)


def test_probe_default_is_off():
    # When no host_staging key is present, the probe should default to
    # disabled ("off") until all staged transfer paths are implemented.
    probe = HostStagingPlatformProbe({})
    assert probe.enabled(0) is False


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Needs CUDA",
)
def test_pinned_arena_round_trip():
    cupy = pytest.importorskip("cupy", reason="cupy not available")

    arena = PinnedHostStagingArena(4096, "test")
    src = cupy.arange(256, dtype=cupy.uint8)
    dst = cupy.zeros(256, dtype=cupy.uint8)

    arena.d2h(src.data.ptr, 0, 256)
    arena.synchronize()
    arena.h2d(0, dst.data.ptr, 256)
    arena.synchronize()

    cupy.testing.assert_array_equal(src, dst)


def test_chunk_descriptor_ranges_basic():
    chunks = list(chunk_descriptor_ranges([100, 100, 100], 250))
    assert len(chunks) == 2
    assert chunks[0] == [(0, 0, 100), (1, 0, 100), (2, 0, 50)]
    assert chunks[1] == [(2, 50, 50)]


def test_chunk_descriptor_ranges_single_descriptor_larger_than_chunk():
    with pytest.raises(ValueError):
        list(chunk_descriptor_ranges([100], 50))


def test_chunk_descriptor_ranges_empty():
    assert list(chunk_descriptor_ranges([], 100)) == []


def test_chunk_descriptor_ranges_exact_fit():
    chunks = list(chunk_descriptor_ranges([100, 100], 200))
    assert len(chunks) == 1
    assert chunks[0] == [(0, 0, 100), (1, 0, 100)]
