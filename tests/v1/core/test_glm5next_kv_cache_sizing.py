# SPDX-License-Identifier: Apache-2.0
"""Ground-truth tests for GLM-5.3-Flash (Glm5Next) KV-cache sizing on SM121.

Reproduces the in-container KV cache spec construction for the
glm53-flash-tp4 deployment (TP4, block-size 2304, fp8 KV) via the model-free
engine builders in ``vllm.v1.glm5next.spec_math`` and checks the
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

Runs on CPU-only hosts (no ``vllm.models.*`` imports, no GPU kernels).
"""

import os
from dataclasses import replace
from math import gcd, lcm

import pytest
import torch

# The deployment's 115,200-token pin exceeds Qwen3-derived
# max_position_embeddings; the real deployment sets this env var too.
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
    FlashInferMLASparseSM90Backend,
)
from vllm.v1.attention.backends.mla.glm5next_indexer import (
    Glm5NextKpoolIndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.core.kv_cache_utils import (
    _check_enough_kv_cache_memory,
    _max_memory_usage_bytes_from_groups,
    get_kv_cache_groups,
)
from vllm.v1.glm5next.spec_math import (
    build_indexer_spec,
    build_tail_spec,
    indexer_head_dim,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaAttentionBackendEnum,
    MambaSpec,
    MLAAttentionSpec,
)

# glm53-flash-tp4 deployment parameters (templates/vllm-glm53-flash-tp4.env.tmpl
# and text_config of the LibertAIDAI/GLM-5.3-Flash-NVFP4 checkpoint).
BLOCK_SIZE = 2304
MAX_MODEL_LEN = 115_200
NUM_HIDDEN_LAYERS = 45
TP_SIZE = 4
# text_config.linear_attn_config
LINEAR_NUM_HEADS = 64
LINEAR_HEAD_DIM = 128
LINEAR_CONV_KERNEL = 4
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
INDEXER_DTYPE = torch.uint8  # fp8 indexer cache
KV_LORA_RANK = 512
KV_CACHE_DTYPE = torch.uint8  # fp8_e4m3

# KDA state shapes (mamba_utils.kda_state_shape, conv dim-first):
#   conv = (conv_dim/tp, kernel-1+num_spec) = ((3*64*128)/4, 3) = (6144, 3) bf16
#   recurrent = (heads/tp, head_dim, head_dim) = (16, 128, 128) fp32
KDA_CONV_SHAPE = (
    LINEAR_NUM_HEADS * LINEAR_HEAD_DIM * 3 // TP_SIZE,
    LINEAR_CONV_KERNEL - 1,
)
KDA_REC_SHAPE = (LINEAR_NUM_HEADS // TP_SIZE, LINEAR_HEAD_DIM, LINEAR_HEAD_DIM)

# Observed on gx10-node1 boot (2026-09-05 09:27): "Available KV cache memory: 22.62 GiB"
EXPECTED_AVAILABLE_GIB = 22.62

# Live deployment pin (--kv-cache-memory 25769803776 = 24 GiB).
KV_MEMORY_BYTES = 25769803776


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
        distributed_executor_backend="mp",
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        compilation_config=CompilationConfig(mode=0),
        device_config=DeviceConfig(device="cpu"),
        speculative_config=None,
        kv_transfer_config=None,
    )


def _build_specs(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Mirror the model runner's per-layer spec construction."""
    cache_config = vllm_config.cache_config
    specs: dict[str, KVCacheSpec] = {}

    for i in range(NUM_HIDDEN_LAYERS):
        if i in FULL_ATTN_LAYERS:
            specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=KV_LORA_RANK,
                dtype=KV_CACHE_DTYPE,
                cache_dtype_str="fp8_e4m3",
                head_size_v=0,
            )
            # Live indexer folds the fp8 per-128 scale into head_dim:
            # head_dim = 128 + 128 // 128 * 4 = 132 (spec_math.indexer_head_dim).
            specs[f"model.layers.{i}.self_attn.indexer"] = build_indexer_spec(
                cache_block_size=cache_config.block_size,
                head_dim=indexer_head_dim(INDEX_HEAD_DIM),
                dtype=INDEXER_DTYPE,
                index_kpool=INDEX_KPOOL,
            )
            specs[f"model.layers.{i}.self_attn.indexer.tail"] = build_tail_spec(
                head_dim=INDEX_HEAD_DIM, index_kpool=INDEX_KPOOL
            )
        else:
            specs[f"model.layers.{i}.linear_attn"] = MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=(KDA_CONV_SHAPE, KDA_REC_SHAPE),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.MAMBA2,
                mamba_cache_mode="align",
            )
    # Mirror gpu_model_runner: annotate each attention spec with its backend's
    # indexes_kv_by_block_stride and apply backend.customize_spec.
    backend_by_name = {}
    for i in range(NUM_HIDDEN_LAYERS):
        if i in FULL_ATTN_LAYERS:
            backend_by_name[f"model.layers.{i}.self_attn"] = (
                FlashInferMLASparseSM90Backend
            )
            backend_by_name[f"model.layers.{i}.self_attn.indexer"] = (
                Glm5NextKpoolIndexerBackend
            )
            backend_by_name[f"model.layers.{i}.self_attn.indexer.tail"] = (
                KpoolTailBackend
            )
    out: dict[str, KVCacheSpec] = {}
    for name, spec in specs.items():
        backend = backend_by_name.get(name)
        if backend is not None and isinstance(spec, AttentionSpec):
            spec = replace(
                spec, indexes_kv_by_block_stride=backend.indexes_kv_by_block_stride()
            )
            spec = backend.customize_spec(spec)
        out[name] = spec
    return out


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
    idx_spec = specs[f"model.layers.{FULL_ATTN_LAYERS[0]}.self_attn.indexer"]
    assert isinstance(idx_spec, MLAAttentionSpec)
    print(
        f"\nindexer spec: block_size={idx_spec.block_size} "
        f"compress_ratio={idx_spec.compress_ratio} "
        f"tokens_per_state={idx_spec.tokens_per_state} "
        f"storage_block_size={idx_spec.storage_block_size} "
        f"page={idx_spec.page_size_bytes}"
    )
    # The indexer presents the model-wide scheduler block (2304) to the KV
    # manager so its pool accounting is uniform with the co-located MLA.
    # compress_ratio = index_kpool (4); the runtime hybrid block-table splits
    # each 2304 manager block into 256-token kernel blocks, and the metadata
    # builder reads the kernel block -> storage_block_size = 256 // 4 = 64
    # (DeepGEMM-legal). At construction the spec's storage_block_size is 576
    # (2304 // 4); the DeepGEMM block_kv is derived from the kernel block.
    assert idx_spec.block_size == BLOCK_SIZE
    assert idx_spec.compress_ratio == INDEX_KPOOL
    assert idx_spec.tokens_per_state == INDEX_KPOOL
    # Page must cover the full 2304-token block at the kpool-compressed density
    # (576 storage slots x 132 B = 76032), NOT padded to the MLA page.
    assert idx_spec.page_size_bytes == 76032


def test_needed_memory_fits_reference_budget(cfg_and_specs):
    """A 115,200-token request must fit the profiled 22.62 GiB/rank budget."""
    vllm_config, specs, groups = cfg_and_specs
    needed = _max_memory_usage_bytes_from_groups(vllm_config, groups)
    available = int(EXPECTED_AVAILABLE_GIB * (1 << 30))
    group_size = max(len(g.layer_names) for g in groups)
    page_sizes = sorted({g.kv_cache_spec.page_size_bytes for g in groups})
    print(
        f"\nneeded={needed / (1 << 30):.2f} GiB  "
        f"available={available / (1 << 30):.2f} GiB"
    )
    print(f"groups={len(groups)} group_size={group_size} page_sizes={page_sizes}")
    for name, spec in specs.items():
        if "layers.3." in name or "layers.0." in name:
            print(
                f"  {name}: type={type(spec).__name__} block_size={spec.block_size} "
                f"page={spec.page_size_bytes} "
                f"unpadded={getattr(spec, 'unpadded_page_size_bytes', 'n/a')}"
            )
    for g in groups:
        print(
            f"  group: layers={len(g.layer_names)} "
            f"type={type(g.kv_cache_spec).__name__} "
            f"page={g.kv_cache_spec.page_size_bytes} "
            f"block_size={g.kv_cache_spec.block_size} "
            f"max_mem={g.kv_cache_spec.max_memory_usage_bytes(vllm_config)}"
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
    print(
        f"\nper-token: {per_token:.0f} B/token/rank "
        f"({per_token / 1024:.1f} KB/token/rank)"
    )
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


# --- GLM-5.3-Flash KV fast-path lane (slot-sharing, real pages) -------------
#
# Regression guard for the lane dropped in the v0.28.0 rebase and restored on
# the glm5next-lane-restore branch. The generic hybrid path pads the kpool
# indexer's 8.4 KiB page to the 1.125 MiB MLA page (138x); the lane keeps each
# cache kind at its real page size via slot-sharing.


def _build_glm5_specs(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    """Raw model-runner specs (no backend customize_spec promotion): the lane
    sees KpoolTailSpec / SlidingWindowSpec / MLAAttentionSpec / MambaSpec."""
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    cache_config = vllm_config.cache_config
    specs: dict[str, KVCacheSpec] = {}
    for i in range(NUM_HIDDEN_LAYERS):
        if i in FULL_ATTN_LAYERS:
            specs[f"model.layers.{i}.self_attn"] = MLAAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=KV_LORA_RANK,
                dtype=KV_CACHE_DTYPE,
                cache_dtype_str="fp8_e4m3",
                head_size_v=0,
            )
            specs[f"model.layers.{i}.self_attn.indexer"] = build_indexer_spec(
                cache_block_size=cache_config.block_size,
                head_dim=indexer_head_dim(INDEX_HEAD_DIM),
                dtype=INDEXER_DTYPE,
                index_kpool=INDEX_KPOOL,
            )
            specs[f"model.layers.{i}.self_attn.indexer.tail"] = build_tail_spec(
                head_dim=INDEX_HEAD_DIM, index_kpool=INDEX_KPOOL
            )
        else:
            specs[f"model.layers.{i}.linear_attn"] = MambaSpec(
                block_size=BLOCK_SIZE,
                shapes=(KDA_CONV_SHAPE, KDA_REC_SHAPE),
                dtypes=(torch.bfloat16, torch.float32),
                mamba_type=MambaAttentionBackendEnum.MAMBA2,
                mamba_cache_mode="align",
            )
    # DFlash2 drafter: 5 plain sliding-window attention layers.
    for j in range(5):
        specs[f"drafter.model.layers.{j}.self_attn"] = SlidingWindowSpec(
            block_size=8,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=2048,
        )
    return specs


@pytest.fixture(scope="module")
def glm5_lane():
    from vllm.v1.glm5next.kv_groups import (
        try_build_kv_cache_groups as _get_kv_cache_groups_glm5_next,
    )
    from vllm.v1.glm5next.tensor_layout import (
        detect_layout as _glm5_next_tensor_layout,
    )

    vllm_config = _vllm_config()
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    with set_current_vllm_config(vllm_config):
        specs = _build_glm5_specs(vllm_config)
        groups = _get_kv_cache_groups_glm5_next(vllm_config, specs)
        layout = _glm5_next_tensor_layout(groups) if groups else None
        yield vllm_config, specs, groups, layout


def test_lane_grouping_structure(glm5_lane):
    """The lane must produce MLA + indexer + tail + mamba + drafter groups."""
    _, _, groups, layout = glm5_lane
    assert groups is not None, "glm5 lane returned None"
    assert layout is not None, "layout detection failed"
    assert len(layout.mla_names) == 11 and len(layout.idx_names) == 11
    assert len(layout.tail_names) == 11 and len(layout.mamba_groups) == 4
    assert layout.draft_group is not None
    assert layout.hidden_names == []  # no Eagle3 aux layers in this spec set
    # The indexer is uniform with the MLA at block_size=2304, so the layout's
    # idx_page is its real per-block page (576 storage slots x 132 B = 76032)
    # with NO padding to the 1.125 MiB MLA page and no 9x scaling.
    assert layout.idx_page == 76032
    assert layout.idx_page < layout.mla_page


def test_lane_indexer_not_padded(glm5_lane):
    """The indexer's real page must survive (no 138x MLA-page padding)."""
    _, specs, _, _ = glm5_lane
    idx_spec = specs["model.layers.3.self_attn.indexer"]
    assert idx_spec.page_size_bytes == 76032
    assert idx_spec.page_size_padded is None


def test_lane_tensor_emission_and_pool(glm5_lane):
    """Tensor emission must slot-share mamba into MLA and tail into indexer,
    and the pool must serve well over 1M tokens on the 24 GiB pin."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups

    vllm_config, _, groups, layout = glm5_lane
    assert layout is not None
    cfg = get_kv_cache_config_from_groups(vllm_config, groups, KV_MEMORY_BYTES)
    tokens = cfg.num_blocks * BLOCK_SIZE
    print(f"\nlane: num_blocks={cfg.num_blocks} tokens={tokens} (live generic: 333K)")
    assert tokens > 1_000_000, (
        f"lane serves {tokens} tokens; must exceed the 1M max_model_len"
    )
    # Every indexer tensor is co-owned by its sibling tail layer.
    idx_tensors = [t for t in cfg.kv_cache_tensors if "indexer" in t.shared_by[0]]
    assert len(idx_tensors) == 11
    for t in idx_tensors:
        assert any("tail" in n for n in t.shared_by)
    # Every MLA tensor is co-owned by a mamba layer (slot-sharing).
    mla_tensors = [
        t
        for t in cfg.kv_cache_tensors
        if t.shared_by and t.shared_by[0].endswith("self_attn")
    ]
    assert any(len(t.shared_by) > 1 for t in mla_tensors)


def test_lane_admission_fits_pin(glm5_lane):
    """One 1M-token request must fit the 24 GiB pin (day-0: ~1.12x headroom)."""
    vllm_config, _, groups, layout = glm5_lane
    assert layout is not None
    needed = _max_memory_usage_bytes_from_groups(vllm_config, groups)
    print(
        f"\nadmission demand: {needed / (1 << 30):.2f} GiB vs pin "
        f"{KV_MEMORY_BYTES / (1 << 30):.1f} GiB = "
        f"{KV_MEMORY_BYTES / needed:.2f}x"
    )
    assert needed < KV_MEMORY_BYTES, (
        f"1M request needs {needed / (1 << 30):.2f} GiB > "
        f"{KV_MEMORY_BYTES / (1 << 30):.1f} GiB pin"
    )


def test_lane_tp1_drafter_serves_1m_tokens():
    """A draft_tensor_parallel_size=1 drafter (8 KV heads, not the TP4-sharded
    2) must still slot-share the MLA page and serve >1M tokens.

    Regression guard for the live 1.74M->534K pool collapse: the TP1 drafter's
    fit_block is not 64-divisible on its own, so the pre-fix exact-fit gate
    rejected it and the drafter fell to standalone tensors whose page
    unification shrank the pool ~3.3x. The common-block LCM (2304) is
    64-divisible, so it must EXACT-FIT."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.glm5next.kv_groups import (
        try_build_kv_cache_groups as _get_kv_cache_groups_glm5_next,
    )
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    vllm_config = _vllm_config()
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    with set_current_vllm_config(vllm_config):
        specs = _build_glm5_specs(vllm_config)
        # Replace the TP4-sharded drafter (num_kv_heads=2) with the TP1
        # replicated drafter (num_kv_heads=8, model-wide block_size).
        specs = {k: v for k, v in specs.items() if not k.startswith("drafter.")}
        for j in range(5):
            specs[f"drafter.model.layers.{j}.self_attn"] = SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.bfloat16,
                sliding_window=2048,
            )
        groups = _get_kv_cache_groups_glm5_next(vllm_config, specs)
        assert groups is not None
        cfg = get_kv_cache_config_from_groups(vllm_config, groups, KV_MEMORY_BYTES)
        tokens = cfg.num_blocks * BLOCK_SIZE
        print(f"\nTP1 drafter: num_blocks={cfg.num_blocks} tokens={tokens}")
        assert tokens > 1_000_000, (
            f"TP1 drafter serves only {tokens} tokens; the drafter must "
            "slot-share (exact-fit), not collapse the pool"
        )


def test_lane_survives_eagle3_hidden_layers():
    """Regression: the DFlash2 drafter's Eagle3 aux-capture layers are
    HiddenStateCacheSpec. On the first restored lane they fell into attn_specs,
    made the gate return None, and the live boot dropped to the generic path
    (448K tokens, mamba page padding 0.70%). The lane must exclude them from
    attn_specs, slot-share them into the MLA tensors, and still serve 1M."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
    from vllm.v1.glm5next.kv_groups import (
        try_build_kv_cache_groups as _get_kv_cache_groups_glm5_next,
    )
    from vllm.v1.glm5next.tensor_layout import (
        detect_layout as _glm5_next_tensor_layout,
    )
    from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

    vllm_config = _vllm_config()
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    with set_current_vllm_config(vllm_config):
        specs = _build_glm5_specs(vllm_config)
        # Add the 5 Eagle3 aux hidden-state layers the live boot injects.
        for a in [6, 15, 25, 34, 43]:
            specs[f"model.layers.{a}.aux_hidden_state"] = HiddenStateCacheSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=16,
                head_size=2048,
                dtype=torch.bfloat16,
            )
        groups = _get_kv_cache_groups_glm5_next(vllm_config, specs)
        assert groups is not None, "lane returned None with Eagle3 hidden layers"
        layout = _glm5_next_tensor_layout(groups)
        assert layout is not None
        assert len(layout.hidden_names) == 5
        cfg = get_kv_cache_config_from_groups(vllm_config, groups, KV_MEMORY_BYTES)
        tokens = cfg.num_blocks * BLOCK_SIZE
        print(f"\nwith Eagle3 hidden: num_blocks={cfg.num_blocks} tokens={tokens}")
        assert tokens > 1_000_000
        # Hidden layers slot-share the first 5 MLA tensors (no extra tensors).
        mla_tensors = [
            t
            for t in cfg.kv_cache_tensors
            if t.shared_by and t.shared_by[0].endswith("self_attn")
        ]
        assert any(
            any("aux_hidden_state" in n for n in t.shared_by) for t in mla_tensors
        )


def test_kpool_tail_admission_bounded_working_set(glm5_lane):
    """A long chunked prefill must be admittable, and the kpool tail must
    hold only a handful of real blocks per request at every step.

    Regression guard for the glm53-flash-tp4 prefill->decode stall: with the
    SlidingWindowManager fallback, the tail's first-chunk allocation demanded
    cdiv(chunk_tokens, 4) real blocks (~4096 for a 16K chunk) against the
    ~1865-block pool, so a cold prompt beyond ~7.4K tokens never left the
    waiting queue. KpoolTailManager null-pads dead positions and keeps only
    the live window + lookahead span real (~3 blocks).
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
    )
    from vllm.v1.glm5next.kv_specs import KpoolTailSpec
    from vllm.v1.glm5next.tail_manager import KpoolTailManager
    from vllm.v1.request import Request

    vllm_config, _, groups, _ = glm5_lane
    assert groups is not None
    worker_cfg = get_kv_cache_config_from_groups(vllm_config, groups, 4 << 30)
    sched_cfg = generate_scheduler_kv_cache_config([worker_cfg])

    tail_gid = next(
        i
        for i, g in enumerate(sched_cfg.kv_cache_groups)
        if isinstance(g.kv_cache_spec, KpoolTailSpec)
    )
    group_bs = [g.kv_cache_spec.block_size for g in sched_cfg.kv_cache_groups]
    mgr = KVCacheManager(
        kv_cache_config=sched_cfg,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=lcm(*group_bs),
        hash_block_size=gcd(*group_bs),
        max_in_flight_tokens=vllm_config.max_in_flight_tokens,
        enable_caching=False,
        use_eagle=True,
        num_prefill_lookahead=7,
    )
    tail_mgr = mgr.coordinator.single_type_managers[tail_gid]
    assert isinstance(tail_mgr, KpoolTailManager)

    req = Request(
        request_id="tail-40k",
        prompt_token_ids=[0] * 40_000,
        sampling_params=SamplingParams(max_tokens=4, temperature=0),
        pooling_params=None,
    )
    free0 = mgr.block_pool.get_num_free_blocks()
    # Chunked prefill (16K-token chunks) + a few decode steps.
    while req.num_computed_tokens < 40_000:
        num_new = min(16_384, 40_000 - req.num_computed_tokens)
        blocks = mgr.allocate_slots(
            req,
            num_new,
            num_lookahead_tokens=7,
            full_sequence_must_fit=(req.num_computed_tokens == 0),
            has_scheduled_reqs=False,
        )
        assert blocks is not None, (
            f"allocation stalled at computed={req.num_computed_tokens}"
        )
        req.num_computed_tokens += num_new
        real = sum(not b.is_null for b in tail_mgr.req_to_blocks[req.request_id])
        assert real <= 8, f"tail holds {real} real blocks at {req.num_computed_tokens}"
    for _ in range(3):
        blocks = mgr.allocate_slots(
            req, 1, num_lookahead_tokens=7, has_scheduled_reqs=False
        )
        assert blocks is not None, "decode allocation stalled"
        req.num_computed_tokens += 1
        real = sum(not b.is_null for b in tail_mgr.req_to_blocks[req.request_id])
        assert real <= 8, f"tail holds {real} real blocks during decode"
    used = free0 - mgr.block_pool.get_num_free_blocks()
    assert used < 300, f"40K prompt consumed {used} blocks"
