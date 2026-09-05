# SPDX-License-Identifier: Apache-2.0
"""Ground-truth tests for GLM-5.3-Flash (Glm5Next) KV-cache sizing on SM121.

Reproduces the in-container KV cache spec construction for the
glm53-flash-tp4 deployment (TP4, block-size 2304, fp8 KV) and checks the
resulting `_max_memory_usage_bytes_from_groups` estimate against the
reference (day-0 tonyd2wild image) budget: ~24 GiB/rank serves a 1M-token
context; ~22.6 GiB should serve well over 115K tokens.

Fork vllm@866b344005 fails this: it bills ~22.8 GiB for a single 115,200-token
request (~207.8 KB/token/rank, ~9x the correct per-token cost). The 9x factor
is exactly block_size 2304 / 256 -- the indexer's page-size override in
`Glm5NextIndexerCache.get_kv_cache_spec` shrinks the spec block_size to
PAGED_MQA_PAGE_SIZES(64) * kpool(4) * compress_ratio(1) = 256, then the
hybrid-manager page unification pads every layer's page up to the MLA page
(1,179,648 B) *without* restoring its block span, so each layer bills
1,179,648 B per 256 tokens instead of per 2304 tokens.

Run inside the deployment image:

    docker run --rm -v <repo>/vllm:/src:ro -w /src \
        --entrypoint python3 10.100.170.3:5000/vllm-gb10:0.28.0-sm121-cu133 \
        -m pytest tests/v1/core/test_glm5next_kv_cache_sizing.py -v -s
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.models.glm5next.nvidia.attention import (
    Glm5NextIndexerCache,
    Glm5NextTailCache,
)
from vllm.v1.core.kv_cache_utils import (
    _check_enough_kv_cache_memory,
    _max_memory_usage_bytes_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
)

# glm53-flash-tp4 deployment parameters (templates/vllm-glm53-flash-tp4.env.tmpl)
BLOCK_SIZE = 2304
MAX_MODEL_LEN = 115_200
NUM_HIDDEN_LAYERS = 45
NUM_FULL_ATTN_LAYERS = 11  # full_attention_interval = 4 over 45 layers
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
INDEXER_DTYPE = torch.uint8  # fp8 indexer cache
KV_LORA_RANK = 512
KV_CACHE_DTYPE = torch.uint8  # fp8_e4m3

# Observed on gx10-node1 boot (2026-09-05 09:27): "Available KV cache memory: 22.62 GiB"
EXPECTED_AVAILABLE_GIB = 22.62


def _vllm_config() -> VllmConfig:
    # ModelConfig() with no model id skips HF loading (mirrors the pattern in
    # tests/v1/core/test_kv_cache_utils.py). Only the fields the sizing code
    # reads are overridden.
    model_config = ModelConfig(max_model_len=MAX_MODEL_LEN)
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.75,
        cache_dtype="fp8_e4m3",
    )
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=16384,
        max_num_seqs=6,
        max_model_len=MAX_MODEL_LEN,
        is_encoder_decoder=False,
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        compilation_config=CompilationConfig(mode=0),
        speculative_config=None,
        kv_transfer_config=None,
    )


def _build_specs(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Mirror the model runner's per-layer spec construction."""
    cache_config = vllm_config.cache_config
    specs: dict[str, KVCacheSpec] = {}
    full_layer_ids = [i for i in range(NUM_HIDDEN_LAYERS) if i % 4 == 0]

    for i in range(NUM_HIDDEN_LAYERS):
        if i in full_layer_ids:
            specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=KV_LORA_RANK,
                dtype=KV_CACHE_DTYPE,
                cache_dtype_str="fp8_e4m3",
                head_size_v=0,
            )
            idx = Glm5NextIndexerCache(
                head_dim=INDEX_HEAD_DIM,
                dtype=INDEXER_DTYPE,
                prefix=f"model.layers.{i}.self_attn.indexer",
                cache_config=cache_config,
                index_kpool=INDEX_KPOOL,
            )
            specs[idx.prefix] = idx.get_kv_cache_spec(vllm_config)
            tail = Glm5NextTailCache(
                head_dim=INDEX_HEAD_DIM,
                dtype=torch.bfloat16,
                prefix=f"model.layers.{i}.self_attn.indexer.tail",
                cache_config=cache_config,
                index_kpool=INDEX_KPOOL,
            )
            specs[tail.prefix] = tail.get_kv_cache_spec(vllm_config)
        else:
            specs[f"model.layers.{i}.linear_attn"] = MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=((4096, 128),),
                dtypes=(torch.bfloat16,),
                mamba_type=MambaAttentionBackendEnum.MAMBA2,
                mamba_cache_mode="align",
            )
    return specs


@pytest.fixture(scope="module")
def cfg_and_specs():
    vllm_config = _vllm_config()
    with set_current_vllm_config(vllm_config):
        specs = _build_specs(vllm_config)
        groups = get_kv_cache_groups(vllm_config, specs)
        yield vllm_config, specs, groups


def test_indexer_spec_block_size_preserved(cfg_and_specs):
    """Indexer spec must stay on the model-wide block_size (2304); DeepGEMM
    page-tiling is a virtual split and must not change allocation accounting."""
    _, specs, _ = cfg_and_specs
    idx_spec = specs["model.layers.0.self_attn.indexer"]
    assert isinstance(idx_spec, MLAAttentionSpec)
    print(
        f"\nindexer spec: block_size={idx_spec.block_size} "
        f"compress_ratio={idx_spec.compress_ratio} "
        f"tokens_per_state={idx_spec.tokens_per_state} "
        f"storage_block_size={idx_spec.storage_block_size} "
        f"page={idx_spec.page_size_bytes}"
    )
    assert idx_spec.block_size == BLOCK_SIZE, (
        f"indexer spec block_size={idx_spec.block_size}, expected {BLOCK_SIZE}; "
        "shrinking it to the DeepGEMM page tile inflates per-token billing 9x"
    )
    assert idx_spec.storage_block_size == BLOCK_SIZE // INDEX_KPOOL


def test_needed_memory_fits_reference_budget(cfg_and_specs):
    """A 115,200-token request must fit the profiled 22.62 GiB/rank budget."""
    vllm_config, _, groups = cfg_and_specs
    needed = _max_memory_usage_bytes_from_groups(vllm_config, groups)
    available = int(EXPECTED_AVAILABLE_GIB * (1 << 30))
    group_size = max(len(g.layer_names) for g in groups)
    page_sizes = sorted({g.kv_cache_spec.page_size_bytes for g in groups})
    print(f"\nneeded={needed / (1 << 30):.2f} GiB  available={available / (1 << 30):.2f} GiB")
    print(f"groups={len(groups)} group_size={group_size} page_sizes={page_sizes}")
    for g in groups:
        print(
            f"  group: layers={len(g.layer_names)} type={type(g.kv_cache_spec).__name__} "
            f"page={g.kv_cache_spec.page_size_bytes} block_size={g.kv_cache_spec.block_size}"
        )
    print(f"per-token billed: {needed / MAX_MODEL_LEN:.0f} B/token/rank")
    assert needed < available, (
        f"needed {needed / (1 << 30):.2f} GiB exceeds available "
        f"{available / (1 << 30):.2f} GiB for max_model_len={MAX_MODEL_LEN}"
    )


def test_max_memory_usage_per_token_reasonable(cfg_and_specs):
    vllm_config, _, groups = cfg_and_specs
    needed = _max_memory_usage_bytes_from_groups(vllm_config, groups)
    per_token = needed / MAX_MODEL_LEN
    print(f"\nper-token: {per_token:.0f} B/token/rank ({per_token / 1024:.1f} KB/token/rank)")
    assert per_token < 100 * 1024, (
        f"{per_token / 1024:.1f} KB/token/rank; reference bills ~25 KB/token/rank"
    )


def test_check_enough_memory_does_not_raise(cfg_and_specs):
    from functools import partial

    vllm_config, _, groups = cfg_and_specs
    available = int(EXPECTED_AVAILABLE_GIB * (1 << 30))
    _check_enough_kv_cache_memory(
        available,
        partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
        MAX_MODEL_LEN,
        lambda _mem: 0,
    )
