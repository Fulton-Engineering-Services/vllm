# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3 kpool tail ring-buffer manager (CPU-only)."""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.glm5next.tail_manager import KpoolTailManager

pytestmark = pytest.mark.cpu_test

# glm53-flash-tp4 tail parameters: index_kpool=4 raw K+gate slots per
# request, tail block_size 4 (2 slots x 4 tokens x 128 half elems).
BLOCK_SIZE = 4
SLIDING_WINDOW = 4


def get_tail_manager(block_pool: BlockPool) -> KpoolTailManager:
    from vllm.v1.glm5next.spec_math import build_tail_spec

    spec = build_tail_spec(head_dim=128, index_kpool=SLIDING_WINDOW)
    assert spec.block_size == BLOCK_SIZE
    return KpoolTailManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=spec.max_admission_blocks_per_request(
            max_in_flight_tokens=10**9, max_model_len=10**9
        ),
    )


def test_fresh_chunk_admission_is_constant():
    """A 16K-token first prefill chunk must demand one real tail block, not
    cdiv(chunk_tokens, 4) == 4096 (the generic SWA policy that stalled the
    prefill->decode path beyond ~7.4K cold tokens)."""
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)

    demand = manager.get_num_blocks_to_allocate(
        request_id="req",
        num_tokens=16384,
        new_computed_blocks=[],
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=16384,
    )
    assert demand == 1


def test_allocate_null_pads_below_live_edge():
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)

    blocks = manager.allocate_new_blocks(
        "req", num_tokens=16384, num_tokens_main_model=16384
    )

    req_blocks = manager.req_to_blocks["req"]
    assert len(req_blocks) == 16384 // BLOCK_SIZE
    assert len(blocks) == 1
    real_blocks = [b for b in req_blocks if not b.is_null]
    assert [b.block_id for b in real_blocks] == [blocks[0].block_id]
    # live edge: (16384 - 4 + 1) // 4 == 4095 -> the real block sits at the end
    assert req_blocks[4095].block_id == blocks[0].block_id


def test_decode_advances_ring_one_block():
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)
    manager.allocate_new_blocks("req", 16384, 16384)

    new_blocks = manager.allocate_new_blocks("req", 16388, 16384)

    assert len(new_blocks) == 1
    req_blocks = manager.req_to_blocks["req"]
    assert len(req_blocks) == 4097
    assert req_blocks[4096].block_id == new_blocks[0].block_id
    assert not new_blocks[0].is_null


def test_decode_demand_counts_only_new_tail_block():
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)
    manager.allocate_new_blocks("req", 16384, 16384)

    demand = manager.get_num_blocks_to_allocate(
        request_id="req",
        num_tokens=16388,
        new_computed_blocks=[],
        total_computed_tokens=16384,
        num_local_computed_tokens=16384,
        num_tokens_main_model=16384,
    )
    assert demand == 1


def test_remove_skipped_blocks_retires_dead_reals():
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)
    manager.allocate_new_blocks("req", 16384, 16384)
    manager.allocate_new_blocks("req", 16388, 16384)
    free_before = block_pool.get_num_free_blocks()

    # processed_computed_tokens=16388 -> skipped = (16388 - 4 + 1) // 4 = 4096
    # blocks: positions [0, 4096) retire; the real block at 4095 is freed,
    # the one at 4096 stays live.
    manager.remove_skipped_blocks("req", 16388)

    req_blocks = manager.req_to_blocks["req"]
    assert all(b.is_null for b in req_blocks[:4096])
    assert not req_blocks[4096].is_null
    assert block_pool.get_num_free_blocks() == free_before + 1
    assert manager._null_frontier["req"] == 4096


def test_pop_blocks_for_free_resets_frontier():
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=False, hash_block_size=BLOCK_SIZE
    )
    manager = get_tail_manager(block_pool)
    manager.allocate_new_blocks("req", 16384, 16384)

    popped = manager.pop_blocks_for_free("req")

    assert len(popped) == 4096
    assert sum(not b.is_null for b in popped) == 1
    assert "req" not in manager._null_frontier
    assert "req" not in manager.req_to_blocks
