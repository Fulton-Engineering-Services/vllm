# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring-buffer manager for the GLM-5.3-Flash kpool indexer tail cache."""

from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.glm5next.kv_specs import KpoolTailSpec


class KpoolTailManager(SlidingWindowManager):
    """Ring-buffer manager for the GLM-5.3 kpool tail cache (KpoolTailSpec).

    The tail holds only the in-progress kpool pool: the trailing
    ``sliding_window`` (== index_kpool) raw K+gate slots per request,
    rewritten as positions advance; all older content is dead. The generic
    SlidingWindowManager policy -- one real pool block per 4-token span,
    recycled a step later -- transiently demands ``cdiv(chunk_tokens, 4)``
    real blocks for a fresh request's first prefill chunk (~4096 blocks for
    a 16K-token chunk against the ~1865-block glm53-flash-tp4 pool), so a
    cold prompt beyond ~7.4K tokens never left the waiting queue (the
    prefill->decode stall).

    This manager materializes the tail as ``[null ... null, real ...
    real]``: positions below the live edge (trailing window plus in-flight
    speculative lookahead) are null-padded at allocation time, keeping the
    per-request pool footprint at ~3 blocks for any context length. The
    block table keeps the same full-width layout the worker produces for
    the SWA fallback (the kpool op resolves writes through the generic
    ``block_table[pos // block_size]`` slot mapping), so no runner-side
    change is needed.
    """

    def __init__(self, kv_cache_spec: KpoolTailSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        # Per-request count of leading req_to_blocks entries known to be
        # null. Entries past the frontier may hold dead real blocks until
        # remove_skipped_blocks retires them.
        self._null_frontier: dict[str, int] = {}

    def _live_span(
        self, num_tokens: int, num_tokens_main_model: int
    ) -> tuple[int, int]:
        """Block-table ``[start, end)`` that must be backed by real blocks to
        hold the live content when covering ``num_tokens`` slots.
        ``num_tokens`` includes the speculative lookahead, while the window
        trails the committed main-model length."""
        required = cdiv(num_tokens, self.block_size)
        live_start = (
            max(0, num_tokens_main_model - self.sliding_window + 1) // self.block_size
        )
        return live_start, required

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        live_start, required = self._live_span(num_tokens, num_tokens_main_model)
        held = len(self.req_to_blocks.get(request_id, ())) + len(new_computed_blocks)
        new_real = required - max(held, live_start)
        if new_real <= 0:
            return 0
        # Hit blocks consumed out of the free queue count against it, same
        # as the base class.
        return new_real + self._get_num_evictable_blocks(new_computed_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block,
            # as in the base class.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        live_start, required = self._live_span(num_tokens, num_tokens_main_model)
        held = len(req_blocks)
        if held >= required:
            return cow_blocks
        pad_end = min(max(held, live_start), required)
        if pad_end > held:
            req_blocks.extend([self._null_block] * (pad_end - held))
        num_real = required - pad_end
        if num_real <= 0:
            return cow_blocks
        new_blocks = self.block_pool.get_new_blocks(num_real)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return cow_blocks + new_blocks

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        # The base implementation's backward scan breaks on the first null,
        # which strands dead real blocks behind the chunk-boundary null gaps
        # this manager creates (~3 blocks per prefill chunk). Scan forward
        # from the known-null frontier instead; each position is visited at
        # most once over the request's lifetime.
        del num_prompt_tokens
        blocks = self.req_to_blocks.get(request_id)
        if not blocks:
            return
        skipped = (
            self.get_num_skipped_tokens(processed_computed_tokens) // self.block_size
        )
        first = min(self._null_frontier.get(request_id, 0), len(blocks))
        last = min(skipped, len(blocks))
        if first < last:
            freed = [b for b in blocks[first:last] if not b.is_null]
            blocks[first:last] = [self._null_block] * (last - first)
            if freed:
                self.block_pool.free_blocks(freed)
        self._null_frontier[request_id] = max(first, last)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self._null_frontier.pop(request_id, None)
        return super().pop_blocks_for_free(request_id)
