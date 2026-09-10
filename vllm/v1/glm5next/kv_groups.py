# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KV cache group building.

Extracted from ``vllm.v1.core.kv_cache_utils`` so the engine keeps only a
thin hook call site in ``get_kv_cache_groups``. Dispatches on spec shape
(self-guarding; returns ``None`` when the spec set is not a GLM-5.3 hybrid),
never on ``vllm.models.*`` imports.
"""

import math
from dataclasses import replace
from typing import cast

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import (
    _largest_divisor_at_most,
    create_kv_cache_group_specs,
)
from vllm.v1.glm5next.kv_specs import KpoolTailSpec
from vllm.v1.kv_cache_interface import (
    HiddenStateCacheSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)


def pp_balanced_mamba_group_count(
    vllm_config: VllmConfig,
    mamba_layer_names: list[str],
    mla_layer_names: list[str],
) -> int | None:
    """Mamba group count for `try_build_kv_cache_groups`.

    Under PP, each stage's largest projected Mamba group must fit its local MLA
    count. Returns the smallest group count that satisfies every stage, or
    ``None`` when a stage has Mamba layers but no MLA layer.
    """
    num_groups = cdiv(len(mamba_layer_names), len(mla_layer_names))
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    if pp_size == 1:
        return num_groups

    from vllm.distributed.utils import get_pp_indices
    from vllm.model_executor.models.utils import extract_layer_index

    total_layers = vllm_config.model_config.get_total_num_hidden_layers()
    mamba_indices = [extract_layer_index(name) for name in mamba_layer_names]
    mla_indices = [extract_layer_index(name) for name in mla_layer_names]
    for rank in range(pp_size):
        start, end = get_pp_indices(total_layers, rank, pp_size)
        num_mamba = sum(start <= i < end for i in mamba_indices)
        num_mla = sum(start <= i < end for i in mla_indices)
        if not num_mamba:
            continue
        if not num_mla:
            return None
        # A stage's mamba layers are contiguous in round-robin order, so its
        # largest slice under `count` groups is exactly cdiv(num_mamba, count),
        # which fits num_mla slots iff count >= cdiv(num_mamba, num_mla).
        num_groups = max(num_groups, cdiv(num_mamba, num_mla))
    return num_groups


def try_build_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Groups for GLM-5.3-Flash hybrids: MLA(+kpool indexer) + mamba.

    Mamba layers share the MLA layers' KV cache tensors. Indexer layers get
    their own KV cache tensors as indexer blocks are small. A spec-decode
    (DFlash2) drafter's plain SlidingWindowSpec layers are partitioned into
    their own group appended last (existing group ids stay stable); when the
    drafter's real page exactly fills the MLA page it slot-shares the MLA
    tensors (exact fit, no added per-block bytes), otherwise it gets compact
    standalone tensors.

    Returns ``None`` when ``kv_cache_spec`` is not a GLM-5.3-Flash hybrid so
    the caller falls through to the generic paths.
    """
    mamba_specs = {k: v for k, v in kv_cache_spec.items() if isinstance(v, MambaSpec)}
    tail_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, KpoolTailSpec)
    }
    # Partition out the drafter's plain SlidingWindowSpec layers (exact type:
    # KpoolTailSpec subclasses SlidingWindowSpec, so test type identity) so they
    # do not disqualify the model from this fast path.
    draft_specs = {
        k: v for k, v in kv_cache_spec.items() if type(v) is SlidingWindowSpec
    }
    attn_specs = {
        k: v
        for k, v in kv_cache_spec.items()
        if not isinstance(v, (MambaSpec, KpoolTailSpec, HiddenStateCacheSpec))
        and type(v) is not SlidingWindowSpec
    }
    if not mamba_specs or not all(
        type(s) is MLAAttentionSpec for s in attn_specs.values()
    ):
        return None
    mla_specs = cast(dict[str, MLAAttentionSpec], attn_specs)
    if not any(s.compress_ratio > 1 for s in mla_specs.values()):
        return None

    assert all(s.page_size_padded is None for s in mla_specs.values())
    # The MLA target (block_size = model block, compress_ratio == 1) and the
    # kpool indexer (block_size = kernel_tile * kpool, compress_ratio == kpool)
    # have different block sizes and real page sizes. They CANNOT share a
    # UniformTypeKVCacheSpecs (which requires one block_size), and the generic
    # path's page unification would pad the indexer's 8.4 KiB page to the MLA
    # page (138x). Keep them in SEPARATE groups at their real sizes; the
    # indexer tensor holds its own small pages and the mamba/drafter/tail
    # slot-share the MLA / indexer tensors.
    mla_names = [n for n, s in mla_specs.items() if s.compress_ratio == 1]
    idx_names = [n for n, s in mla_specs.items() if s.compress_ratio > 1]
    mla_pages = {mla_specs[n].page_size_bytes for n in mla_names}
    idx_pages = {mla_specs[n].page_size_bytes for n in idx_names}
    assert len(mla_pages) == 1 and len(idx_pages) == 1
    mla_page = mla_pages.pop()
    idx_page = idx_pages.pop()
    idx_block_size = mla_specs[idx_names[0]].block_size
    assert all(mla_specs[n].block_size == idx_block_size for n in idx_names)

    mla_uniform = UniformTypeKVCacheSpecs.from_specs(
        {n: mla_specs[n] for n in mla_names}
    )
    assert mla_uniform is not None
    attn_group = KVCacheGroupSpec(list(mla_names), mla_uniform)
    idx_uniform = UniformTypeKVCacheSpecs.from_specs(
        {n: mla_specs[n] for n in idx_names}
    )
    assert idx_uniform is not None
    idx_group = KVCacheGroupSpec(list(idx_names), idx_uniform)

    # Keep all indexer tails in one group and pad their pages to the indexer
    # page size so each tail can share its sibling indexer's storage.
    tail_group = None
    if tail_specs:
        padded_tail_specs: dict[str, KVCacheSpec] = {
            name: replace(s, page_size_padded=idx_page)
            for name, s in tail_specs.items()
        }
        tail_uniform = UniformTypeKVCacheSpecs.from_specs(padded_tail_specs)
        assert tail_uniform is not None
        tail_group = KVCacheGroupSpec(list(padded_tail_specs), tail_uniform)

    any_mamba = next(iter(mamba_specs.values()))
    assert all(spec == any_mamba for spec in mamba_specs.values())
    if any_mamba.page_size_bytes > mla_page:
        raise ValueError(
            f"the mamba state page ({any_mamba.page_size_bytes} bytes) "
            f"does not fit the MLA page ({mla_page} bytes); increase tensor "
            "parallelism or use a wider KV cache dtype"
        )
    padded_specs: dict[str, KVCacheSpec] = {
        name: replace(any_mamba, page_size_padded=mla_page) for name in mamba_specs
    }
    num_groups = pp_balanced_mamba_group_count(
        vllm_config, list(mamba_specs), mla_names
    )
    if num_groups is None:
        raise ValueError(
            "a pipeline stage has mamba layers but no MLA layer to share "
            "slots with; realign the stage boundaries (VLLM_PP_LAYER_PARTITION)"
        )
    mamba_grouped_names: list[list[str]] = [[] for _ in range(num_groups)]
    for k, name in enumerate(mamba_specs):
        mamba_grouped_names[k % num_groups].append(name)

    # Drafter group: appended LAST so existing group ids stay stable. NEVER
    # page_size_padded: a padded spec routes the runner into the strided-view
    # reshape, which is invalid when the backend virtually splits the manager
    # block into smaller kernel blocks. Both modes below reshape contiguously.
    draft_group = None
    if draft_specs:
        any_draft = next(iter(draft_specs.values()))
        assert all(spec == any_draft for spec in draft_specs.values()), (
            "drafter SlidingWindowSpec layers must share one spec"
        )
        draft_bytes_per_token = any_draft.page_size_bytes // any_draft.block_size
        mla_block = mla_specs[mla_names[0]].block_size
        fit_block = (
            mla_page // draft_bytes_per_token
            if mla_page % draft_bytes_per_token == 0
            else 0
        )
        # The manager block is lcm(mla_block, fit_block); the SWA backends
        # (MultipleOf(64)/MultipleOf(16)) split that manager block into kernel
        # blocks. A TP4 drafter (2 KV heads) gives fit_block=1152, a TP1
        # drafter (8 KV heads) gives fit_block=288 — both have lcm=2304, which
        # is 64-divisible, so the common split is clean. Gating on fit_block
        # itself being 64-divisible wrongly rejects the TP1 288 case and drops
        # the drafter to standalone tensors (page unification collapses the
        # pool ~3.3x). Gate on the common block instead.
        common_block = (
            mla_block * fit_block // math.gcd(mla_block, fit_block) if fit_block else 0
        )
        if (
            fit_block
            # select_common_block_size splits the manager block into 64-token
            # (or 16-token) kernel blocks; require the common block to be
            # 64-divisible so every SWA kernel block size divides it cleanly.
            and common_block % 64 == 0
            # The common block must be an exact multiple of both spans so the
            # scheduler LCM stays at lcm(mla_block, fit_block) (no explosion).
            and common_block % fit_block == 0
            and common_block % mla_block == 0
            and len(draft_specs) <= len(mla_names)
        ):
            # EXACT FIT: drafter layer i co-owns MLA tensor i at disjoint block
            # ids (like mamba); per-block pool cost unchanged.
            new_draft_specs: dict[str, KVCacheSpec] = {
                name: replace(s, block_size=fit_block)
                for name, s in draft_specs.items()
            }
        else:
            # STANDALONE: compact per-layer drafter tensors, charged per block.
            new_draft_specs = dict(draft_specs)
        draft_uniform = UniformTypeKVCacheSpecs.from_specs(new_draft_specs)
        assert draft_uniform is not None
        draft_group = KVCacheGroupSpec(list(new_draft_specs), draft_uniform)

    # HiddenStateCacheSpec layers (Eagle3 aux-capture for the drafter) were
    # excluded from attn_specs above; re-add each as its own group aligned to
    # the MLA page so they don't affect the slot-sharing layout.
    hidden_groups: list[KVCacheGroupSpec] = []
    hidden_specs = {
        k: v for k, v in kv_cache_spec.items() if isinstance(v, HiddenStateCacheSpec)
    }
    if hidden_specs:
        group_block_size = mla_specs[mla_names[0]].block_size
        for name, spec in hidden_specs.items():
            per_token = spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
            max_block_size = max(mla_page // per_token, 1)
            new_bs = _largest_divisor_at_most(group_block_size, max_block_size)
            aligned = replace(spec, block_size=new_bs, page_size_padded=mla_page)
            hidden_groups.append(KVCacheGroupSpec([name], aligned))

    return (
        [attn_group, idx_group]
        + ([tail_group] if tail_group is not None else [])
        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)
        + ([draft_group] if draft_group is not None else [])
        + hidden_groups
    )
