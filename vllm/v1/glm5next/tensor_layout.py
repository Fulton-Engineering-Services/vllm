# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash hybrid KV tensor layout: detection, emission, accounting.

Extracted from ``vllm.v1.core.kv_cache_utils``. ``detect_layout`` recognizes
the group configuration produced by ``vllm.v1.glm5next.kv_groups`` (possibly
PP-projected) so tensor emission and the accounting paths can never disagree.
"""

from dataclasses import dataclass
from typing import cast

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import may_override_num_blocks
from vllm.v1.glm5next.kv_specs import KpoolTailSpec
from vllm.v1.kv_cache_interface import (
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


@dataclass(frozen=True)
class Glm5NextLayout:
    """The GLM-5.3-Flash slot-sharing KV layout.

    One tensor per MLA layer, co-owned by that MLA layer and one mamba layer
    per mamba group (plus exact-fit drafter and Eagle3 hidden-state layers);
    per-layer indexer tensors co-owned by sibling tail layers; a standalone
    drafter gets compact per-layer tensors. ``idx_page`` is the scaled
    per-scheduler-block indexer page (real indexer page ×
    mla_block / idx_block); ``tail_page`` is the logical (unpadded) tail page.
    """

    attn_group: KVCacheGroupSpec
    mamba_groups: list[KVCacheGroupSpec]
    mla_names: list[str]
    idx_names: list[str]
    mla_page: int
    idx_page: int
    tail_names: list[str]
    tail_page: int
    draft_group: KVCacheGroupSpec | None
    hidden_names: list[str]

    @property
    def draft_names(self) -> list[str]:
        return list(self.draft_group.layer_names) if self.draft_group else []

    @property
    def draft_page(self) -> int:
        if self.draft_group is None:
            return 0
        return next(
            iter(
                cast(
                    UniformTypeKVCacheSpecs, self.draft_group.kv_cache_spec
                ).kv_cache_specs.values()
            )
        ).page_size_bytes

    @property
    def draft_shared(self) -> bool:
        """True when an exact-fit drafter slot-shares the MLA tensors."""
        return self.draft_page == self.mla_page

    @property
    def per_block_bytes(self) -> int:
        """Bytes each block id costs in the shared pool: one MLA page per MLA
        slot plus one scaled indexer page per indexer slot. Mamba and tail
        layers co-own those tensors (no added bytes); an exact-fit drafter and
        the Eagle3 hidden-state layers ride the MLA tensors; a standalone
        drafter adds one page per drafter layer."""
        per_block = len(self.mla_names) * self.mla_page + len(self.idx_names) * (
            self.idx_page
        )
        if self.draft_group is not None and not self.draft_shared:
            per_block += len(self.draft_names) * self.draft_page
        return per_block


def detect_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> Glm5NextLayout | None:
    """Detect the glm5_next layout; ``None`` if the group config is invalid."""
    uniform_groups = [
        g
        for g in kv_cache_groups
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
    ]
    mamba_groups = [
        g for g in kv_cache_groups if isinstance(g.kv_cache_spec, MambaSpec)
    ]
    # The MLA target group, the kpool indexer group, the kpool tail group, and
    # the drafter group are all UniformTypeKVCacheSpecs; distinguish by the
    # inner spec type and (for MLA) the compress_ratio.
    attn_group: KVCacheGroupSpec | None = None
    idx_group: KVCacheGroupSpec | None = None
    tail_group: KVCacheGroupSpec | None = None
    draft_group: KVCacheGroupSpec | None = None
    for g in uniform_groups:
        group_inner = cast(UniformTypeKVCacheSpecs, g.kv_cache_spec).kv_cache_specs
        if all(type(s) is MLAAttentionSpec for s in group_inner.values()):
            if all(s.compress_ratio == 1 for s in group_inner.values()):
                attn_group = g
            elif all(s.compress_ratio > 1 for s in group_inner.values()):
                idx_group = g
            else:
                return None
        elif all(isinstance(s, KpoolTailSpec) for s in group_inner.values()):
            tail_group = g
        elif group_inner and all(
            type(s) is SlidingWindowSpec for s in group_inner.values()
        ):
            draft_group = g
    if attn_group is None or idx_group is None or not mamba_groups:
        return None
    # HiddenStateCacheSpec groups (Eagle3 aux) are neither uniform-type nor
    # mamba; exclude them from the group-count check.
    non_hidden_groups = [
        g
        for g in kv_cache_groups
        if not isinstance(g.kv_cache_spec, HiddenStateCacheSpec)
    ]
    if len(uniform_groups) + len(mamba_groups) != len(non_hidden_groups):
        return None
    attn_uniform = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)
    if not all(
        type(s) is MLAAttentionSpec and s.page_size_padded is None
        for s in attn_uniform.kv_cache_specs.values()
    ):
        return None
    inner = cast(dict[str, MLAAttentionSpec], attn_uniform.kv_cache_specs)
    idx_inner = cast(
        dict[str, MLAAttentionSpec],
        cast(UniformTypeKVCacheSpecs, idx_group.kv_cache_spec).kv_cache_specs,
    )
    mla_names = list(attn_group.layer_names)
    idx_names = list(idx_group.layer_names)
    mla_pages = {inner[n].page_size_bytes for n in mla_names}
    idx_pages = {idx_inner[n].page_size_bytes for n in idx_names}
    if len(mla_pages) != 1 or len(idx_pages) != 1:
        return None
    mla_page = mla_pages.pop()
    if any(g.kv_cache_spec.page_size_bytes != mla_page for g in mamba_groups):
        return None
    if draft_group is not None:
        # The drafter must have one uniform page and never be page_size_padded.
        draft_inner = cast(
            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec
        ).kv_cache_specs
        draft_pages = {s.page_size_bytes for s in draft_inner.values()}
        if len(draft_pages) != 1:
            return None
        if any(s.page_size_padded is not None for s in draft_inner.values()):
            return None
        if draft_pages.pop() == mla_page and len(draft_group.layer_names) > len(
            mla_names
        ):
            return None
    tail_names: list[str] = []
    tail_page = 0
    if tail_group is not None:
        tail_names = list(tail_group.layer_names)
        # The tail spec is padded to idx_page (co-owns the indexer tensor), so
        # page_size_bytes returns idx_page. Callers need the *logical* tail page
        # (2048 B) for transfer sizing and accounting; use unpadded.
        tail_pages = {
            cast(KpoolTailSpec, s).unpadded_page_size_bytes
            for s in cast(
                UniformTypeKVCacheSpecs, tail_group.kv_cache_spec
            ).kv_cache_specs.values()
        }
        if len(tail_pages) != 1:
            return None
        tail_page = tail_pages.pop()
    # Eagle3 aux hidden-state layers (padded to mla_page) slot-share the first
    # len(hidden) MLA tensors exactly like the mamba layers do.
    hidden_names = [
        name
        for g in kv_cache_groups
        if isinstance(g.kv_cache_spec, HiddenStateCacheSpec)
        for name in g.layer_names
    ]
    # The kpool indexer runs at a finer block_size than the MLA/scheduler
    # block, so each indexer layer needs (mla_block / idx_block) real pages per
    # scheduler block. Surface the scaled per-scheduler-block indexer page so
    # tensor emission and accounting never under-allocate it (the generic path
    # hides this by unifying block sizes; the lane keeps them real).
    idx_page = idx_pages.pop()
    idx_block_size = cast(UniformTypeKVCacheSpecs, idx_group.kv_cache_spec).block_size
    mla_block_size = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec).block_size
    if mla_block_size % idx_block_size != 0:
        return None
    idx_page_per_sched = idx_page * (mla_block_size // idx_block_size)
    return Glm5NextLayout(
        attn_group=attn_group,
        mamba_groups=mamba_groups,
        mla_names=mla_names,
        idx_names=idx_names,
        mla_page=mla_page,
        idx_page=idx_page_per_sched,
        tail_names=tail_names,
        tail_page=tail_page,
        draft_group=draft_group,
        hidden_names=hidden_names,
    )


def build_kv_cache_config(
    layout: Glm5NextLayout,
    vllm_config,
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Emit the slot-sharing KV cache tensors and the pool block count."""
    draft_names = layout.draft_names
    draft_page = layout.draft_page
    draft_shared = layout.draft_shared
    if layout.tail_names:
        assert len(layout.idx_names) == len(layout.tail_names), (
            "indexer/tail layer count mismatch: cannot pair for slot-sharing"
        )
    num_blocks = available_memory // layout.per_block_bytes
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)
    return num_blocks, [
        KVCacheTensor(
            size=layout.mla_page * num_blocks,
            shared_by=[mla_name]
            + [g.layer_names[i] for g in layout.mamba_groups if i < len(g.layer_names)]
            + ([draft_names[i]] if draft_shared and i < len(draft_names) else [])
            + ([layout.hidden_names[i]] if i < len(layout.hidden_names) else []),
        )
        for i, mla_name in enumerate(layout.mla_names)
    ] + [
        # Each indexer tensor is co-owned by its sibling tail layer (paired
        # by model-layer order). idx_page here is the per-scheduler-block
        # indexer page (already scaled by mla_block/idx_block in the layout),
        # so the tensor spans the full 2304-token scheduler block.
        KVCacheTensor(
            size=layout.idx_page * num_blocks,
            shared_by=(
                [layout.idx_names[i], layout.tail_names[i]]
                if layout.tail_names
                else [layout.idx_names[i]]
            ),
        )
        for i in range(len(layout.idx_names))
    ] + [
        # Standalone drafter: compact per-layer tensors.
        KVCacheTensor(size=draft_page * num_blocks, shared_by=[name])
        for name in ([] if draft_shared else draft_names)
    ]


def max_memory_usage_bytes(layout: Glm5NextLayout, vllm_config) -> int:
    """Peak bytes the pool must hold.

    Every block id — attention-, mamba-, tail-, or drafter-owned — is charged
    the full per-block byte sum. The tail co-owns the indexer tensor
    (1 block/req), so it adds to the shared block-id demand like mamba, not as
    a separate tensor. Eagle3 aux hidden-state layers slot-share the MLA
    tensors AND read the same positions the MLA layer processes (they are aux
    capture taps, not independent attention), so they add neither bytes nor
    block-id demand.
    """
    uniform_spec = cast(UniformTypeKVCacheSpecs, layout.attn_group.kv_cache_spec)
    blocks_needed = uniform_spec.max_memory_usage_pages(vllm_config)
    for group in layout.mamba_groups:
        spec = group.kv_cache_spec
        blocks_needed += cdiv(
            spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes
        )
    if layout.tail_names:
        # Tail: 1 block/req, drawn from the shared pool.
        blocks_needed += 1
    if layout.draft_group is not None:
        # Charge the drafter's window-bounded block-id demand; a standalone
        # drafter also adds its pages to every block's byte cost.
        draft_uniform = cast(UniformTypeKVCacheSpecs, layout.draft_group.kv_cache_spec)
        blocks_needed += draft_uniform.max_memory_usage_pages(vllm_config)
    return blocks_needed * layout.per_block_bytes
