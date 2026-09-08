# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3 KV-cache group lane (CPU-only).

Covers the reject matrix and the DFlash2 drafter partition modes. The accept
path (group structure, tensor emission, admission pins) is covered by
``tests/v1/core/test_glm5next_kv_cache_sizing.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.glm5next.kv_groups import (
    pp_balanced_mamba_group_count,
    try_build_kv_cache_groups,
)
from vllm.v1.glm5next.kv_specs import KpoolTailSpec
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

# glm53-flash-tp4 numbers (TP4, block 2304, fp8 KV): MLA target page 1,179,648 B.
BLOCK_SIZE = 2304


def _vllm_config(pp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
    )


def _mla_target() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="fp8_e4m3",
        head_size_v=0,
        compress_ratio=1,
    )


def _indexer() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        cache_dtype_str="fp8_e4m3",
        head_size_v=0,
        compress_ratio=4,
    )


def _mamba() -> MambaSpec:
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((6144, 3), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        mamba_cache_mode="align",
    )


def _tail() -> KpoolTailSpec:
    return KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=4,
    )


def _full_attn() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.uint8,
    )


def test_lane_rejects_when_no_mamba_layers():
    specs = {"a.self_attn": _mla_target(), "a.indexer": _indexer()}
    assert try_build_kv_cache_groups(_vllm_config(), specs) is None


def test_lane_rejects_non_mla_attn_specs():
    specs = {
        "a.self_attn": _mla_target(),
        "a.indexer": _indexer(),
        "a.linear_attn": _mamba(),
        "b.full_attn": _full_attn(),
    }
    assert try_build_kv_cache_groups(_vllm_config(), specs) is None


def test_lane_rejects_without_compressed_indexer():
    specs = {
        "a.self_attn": _mla_target(),
        "a.linear_attn": _mamba(),
    }
    assert try_build_kv_cache_groups(_vllm_config(), specs) is None


def test_lane_rejects_when_only_tail_specs():
    specs = {
        "a.indexer.tail": _tail(),
        "a.linear_attn": _mamba(),
    }
    assert try_build_kv_cache_groups(_vllm_config(), specs) is None


def _glm5_specs(with_tail: bool = False) -> dict[str, object]:
    specs = {
        "a.self_attn": _mla_target(),
        "b.self_attn": _mla_target(),
        "a.indexer": _indexer(),
        "b.indexer": _indexer(),
        "a.linear_attn": _mamba(),
        "b.linear_attn": _mamba(),
    }
    if with_tail:
        specs["a.indexer.tail"] = _tail()
        specs["b.indexer.tail"] = _tail()
    return specs


def test_accept_path_group_order_and_types():
    groups = try_build_kv_cache_groups(_vllm_config(), _glm5_specs(with_tail=True))
    assert groups is not None
    # 2 mamba layers / 2 MLA layers -> cdiv == 1 mamba group of both layers.
    assert len(groups) == 4  # attn, indexer, tail, mamba
    assert all(isinstance(g, KVCacheGroupSpec) for g in groups)
    assert [len(g.layer_names) for g in groups] == [2, 2, 2, 2]
    attn_spec = cast_spec(groups[0])
    assert all(s.compress_ratio == 1 for s in attn_spec.kv_cache_specs.values())
    idx_spec = cast_spec(groups[1])
    assert all(s.compress_ratio > 1 for s in idx_spec.kv_cache_specs.values())
    assert isinstance(groups[2].kv_cache_spec, UniformTypeKVCacheSpecs)


def cast_spec(group: KVCacheGroupSpec) -> UniformTypeKVCacheSpecs:
    assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    return group.kv_cache_spec


def test_tail_group_pages_padded_to_indexer_page():
    groups = try_build_kv_cache_groups(_vllm_config(), _glm5_specs(with_tail=True))
    assert groups is not None
    idx_pages = {
        s.page_size_bytes for s in cast_spec(groups[1]).kv_cache_specs.values()
    }
    tail_inner = cast_spec(groups[2]).kv_cache_specs
    assert idx_pages == {s.page_size_bytes for s in tail_inner.values()}
    assert all(s.page_size_padded is not None for s in tail_inner.values())


def test_mamba_pages_padded_to_mla_page():
    groups = try_build_kv_cache_groups(_vllm_config(), _glm5_specs())
    assert groups is not None
    mla_page = next(iter(cast_spec(groups[0]).kv_cache_specs.values())).page_size_bytes
    mamba_group = groups[-1]
    assert mamba_group.kv_cache_spec.page_size_bytes == mla_page


def test_drafter_exact_fit_slot_shares_mla_page():
    """A drafter whose per-token bytes exactly fill the MLA page at a
    64-divisible block slot-shares the MLA tensors (block_size replaced)."""
    mla_page = next(
        iter(
            cast_spec(
                try_build_kv_cache_groups(_vllm_config(), _glm5_specs())[0]
            ).kv_cache_specs.values()
        )
    ).page_size_bytes
    # head_size 288 x 2 (K+V) x bf16 -> 1152 B/token -> fit_block 1152
    # (1327104 // 1152): % 64 == 0 and 2304 % 1152 == 0 -> EXACT FIT.
    drafter = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=288,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    per_token = drafter.page_size_bytes // drafter.block_size
    expected_fit_block = mla_page // per_token

    specs = _glm5_specs()
    specs["d.draft"] = drafter

    groups = try_build_kv_cache_groups(_vllm_config(), specs)
    assert groups is not None
    draft_group = groups[-1]
    assert isinstance(draft_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    inner = draft_group.kv_cache_spec.kv_cache_specs
    assert all(s.block_size == expected_fit_block for s in inner.values())
    assert expected_fit_block != BLOCK_SIZE
    assert all(s.page_size_padded is None for s in inner.values())
    # EXACT FIT: drafter page == MLA page (slot-shares, no added per-block bytes)
    mla_page_built = next(
        iter(cast_spec(groups[0]).kv_cache_specs.values())
    ).page_size_bytes
    assert all(s.page_size_bytes == mla_page_built for s in inner.values())


def test_drafter_standalone_when_page_does_not_fit():
    """A drafter whose page cannot cleanly fill the MLA page keeps its real
    spec and gets compact per-layer tensors charged per block."""
    # head_size 1152 x 2 (K+V) x bf16 -> 4608 B/token -> fit_block 288, whose
    # % 64 != 0 fails the exact-fit conditions -> STANDALONE.
    drafter = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1152,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    specs = _glm5_specs()
    specs["d.draft"] = drafter

    groups = try_build_kv_cache_groups(_vllm_config(), specs)
    assert groups is not None
    draft_group = groups[-1]
    assert isinstance(draft_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    inner = draft_group.kv_cache_spec.kv_cache_specs
    assert all(s.block_size == BLOCK_SIZE for s in inner.values())
    assert all(s.page_size_padded is None for s in inner.values())
    mla_page_built = next(
        iter(cast_spec(groups[0]).kv_cache_specs.values())
    ).page_size_bytes
    assert all(s.page_size_bytes != mla_page_built for s in inner.values())


def test_drafter_group_appended_last_keeps_group_ids_stable():
    drafter = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=288,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    base = try_build_kv_cache_groups(_vllm_config(), _glm5_specs())
    with_draft = try_build_kv_cache_groups(
        _vllm_config(), {**_glm5_specs(), "d.draft": drafter}
    )
    assert base is not None and with_draft is not None
    assert len(with_draft) == len(base) + 1
    # The first len(base) groups are identical in type and layer names.
    for base_group, draft_group in zip(base, with_draft[:-1]):
        assert base_group.layer_names == draft_group.layer_names
        assert type(base_group.kv_cache_spec) is type(draft_group.kv_cache_spec)


def test_pp_balanced_group_count_pp1():
    mamba = [f"model.layers.{i}.linear_attn" for i in range(7)]
    mla = [f"model.layers.{i}.self_attn" for i in range(3)]
    assert pp_balanced_mamba_group_count(_vllm_config(pp_size=1), mamba, mla) == 3


def test_pp_balanced_group_count_stage_without_mla_is_none():
    # 4 layers, 2 contiguous stages: mamba on layers 0-1 (stage 0 only),
    # MLA on layers 2-3 (stage 1 only) -> stage 0 has mamba but no MLA.
    mamba = [f"model.layers.{i}.linear_attn" for i in (0, 1)]
    mla = [f"model.layers.{i}.self_attn" for i in (2, 3)]
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(get_total_num_hidden_layers=lambda: 4),
    )
    assert pp_balanced_mamba_group_count(cfg, mamba, mla) is None
