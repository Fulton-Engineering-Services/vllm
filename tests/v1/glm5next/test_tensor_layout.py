# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM-5.3 hybrid tensor-layout detection and accounting."""

import os
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.glm5next.kv_groups import try_build_kv_cache_groups
from vllm.v1.glm5next.tensor_layout import (
    Glm5NextLayout,
    build_kv_cache_config,
    detect_layout,
    max_memory_usage_bytes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheGroupSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

# The synthetic 1M-context config exceeds Qwen3-derived max_position_embeddings.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

BLOCK_SIZE = 2304
NUM_MLA = 11
NUM_IDX = 11
NUM_MAMBA = 34


def _pp_config(pp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
    )


def _vllm_config(max_model_len: int = 16):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    return VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        cache_config=CacheConfig(
            block_size=BLOCK_SIZE,
            gpu_memory_utilization=0.75,
            cache_dtype="fp8_e4m3",
        ),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=16384,
            max_num_seqs=6,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
        compilation_config=CompilationConfig(mode=0),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=None,
        kv_transfer_config=None,
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


def _glm5_specs() -> dict[str, object]:
    specs: dict[str, object] = {}
    for i in range(NUM_MLA):
        specs[f"model.layers.{3 + 4 * i}.self_attn"] = _mla_target()
        specs[f"model.layers.{3 + 4 * i}.self_attn.indexer"] = _indexer()
    for i in range(NUM_MAMBA):
        specs[f"model.layers.{i}.linear_attn"] = _mamba()
    return specs


@pytest.fixture(scope="module")
def glm5_layout():
    cfg = _vllm_config(max_model_len=1_048_576)
    groups = try_build_kv_cache_groups(cfg, _glm5_specs())
    layout = detect_layout(groups) if groups else None
    assert layout is not None
    return cfg, groups, layout


def test_detect_layout_fields(glm5_layout):
    _, groups, layout = glm5_layout
    assert isinstance(layout, Glm5NextLayout)
    assert len(layout.mla_names) == NUM_MLA
    assert len(layout.idx_names) == NUM_IDX
    # cdiv(34 mamba, 11 MLA) = 4 round-robin groups slot-sharing MLA slots.
    assert len(layout.mamba_groups) == 4
    assert layout.tail_names == []
    assert layout.draft_group is None
    assert layout.hidden_names == []
    # Real MLA page: 2304 tokens x 576 latent x 1 byte (fp8) = 1,327,104 B.
    assert layout.mla_page == BLOCK_SIZE * 576
    # Scaled indexer page: storage page 576 x 132 x 1 B = 76,032, scale 1
    # (idx block 2304 == mla block 2304 in this synthetic set).
    assert layout.idx_page == (BLOCK_SIZE // 4) * 132
    assert layout.tail_page == 0


def test_detect_layout_rejects_single_uniform_group():
    # Two identical MLA layers -> one uniform group: no indexer group, no
    # mamba group -> not a glm5 layout.
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    groups = get_kv_cache_groups(
        _vllm_config(), {"a": _mla_target(), "b": _mla_target()}
    )
    assert detect_layout(groups) is None


def test_detect_layout_rejects_padded_drafter(glm5_layout):
    _, _, layout = glm5_layout
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    padded_draft_group = KVCacheGroupSpec(
        ["d.draft"],
        UniformTypeKVCacheSpecs.from_specs(
            {"d.draft": replace(spec, page_size_padded=layout.mla_page)}
        ),
    )
    groups = [
        layout.attn_group,
        *layout.mamba_groups,
        padded_draft_group,
    ]
    assert detect_layout(groups) is None


def test_per_block_bytes_slot_sharing(glm5_layout):
    _, _, layout = glm5_layout
    # 11 MLA pages (2304 x 576) + 11 storage indexer pages (576 x 132).
    assert layout.per_block_bytes == NUM_MLA * (BLOCK_SIZE * 576) + NUM_IDX * (
        (BLOCK_SIZE // 4) * 132
    )


def test_per_block_bytes_standalone_drafter_adds_pages(glm5_layout):
    _, _, layout = glm5_layout
    drafter = SimpleNamespace(
        layer_names=["d.draft"],
        kv_cache_spec=None,
    )
    # Build a real uniform group for the drafter: 1 layer, page != mla page.
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    drafter.kv_cache_spec = UniformTypeKVCacheSpecs.from_specs({"d.draft": spec})
    standalone = Glm5NextLayout(
        attn_group=layout.attn_group,
        mamba_groups=layout.mamba_groups,
        mla_names=layout.mla_names,
        idx_names=layout.idx_names,
        mla_page=layout.mla_page,
        idx_page=layout.idx_page,
        tail_names=[],
        tail_page=0,
        draft_group=drafter,
        hidden_names=[],
    )
    expected = layout.per_block_bytes + 1 * spec.page_size_bytes
    assert standalone.per_block_bytes == expected
    assert not standalone.draft_shared


def test_build_kv_cache_config_shape_and_sharing(glm5_layout):
    cfg, _, layout = glm5_layout
    available = 64 * layout.per_block_bytes
    num_blocks, tensors = build_kv_cache_config(layout, cfg, available)
    assert num_blocks == 64
    # One tensor per MLA layer + one per indexer layer.
    assert len(tensors) == NUM_MLA + NUM_IDX
    mla_tensors = [t for t in tensors if t.shared_by[0].endswith("self_attn")]
    idx_tensors = [t for t in tensors if t.shared_by[0].endswith("indexer")]
    assert len(mla_tensors) == NUM_MLA and len(idx_tensors) == NUM_IDX
    for tensor in mla_tensors:
        assert tensor.size == layout.mla_page * num_blocks
        # Co-owned by one mamba layer from each group that has a slot i
        # (round-robin groups may be uneven: 34 layers / 4 groups).
        assert 1 <= len(tensor.shared_by) <= 1 + len(layout.mamba_groups)
        assert all(n.endswith("linear_attn") for n in tensor.shared_by[1:])
    for tensor in idx_tensors:
        assert tensor.size == layout.idx_page * num_blocks
        assert len(tensor.shared_by) == 1


def test_max_memory_usage_bytes_formula(glm5_layout):
    cfg, groups, layout = glm5_layout
    from vllm.v1.core.kv_cache_utils import _max_memory_usage_bytes_from_groups

    assert max_memory_usage_bytes(layout, cfg) == (
        _max_memory_usage_bytes_from_groups(cfg, groups)
    )
