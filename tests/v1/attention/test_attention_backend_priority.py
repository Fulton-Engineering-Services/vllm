# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MLA sparse attention backend priority ordering on SM12x.

Guards against the production OOM where SM120 was selected instead of SM90
on GB10 (SM121) devices. The priority list in cuda.py puts SM90 before
SM120 for SM12x, but if the SM90 backend's ``supports_combination`` rejects
the model config (e.g. because ``has_flashinfer_sm90_nope_mla()`` returns
False), SM120 silently takes over with an incompatible KV layout.

These tests verify:
1. The priority list for SM12x has SM90 before SM120.
2. SM90 ``supports_combination`` accepts GLM-5.3 Flash config (NoPE,
   kv_lora_rank=512, qk_rope_head_dim=0) with fp8 KV cache.
3. SM90 ``supports_combination`` rejects fp8 KV when
   ``has_flashinfer_sm90_nope_mla()`` returns False (the gate that
   caused SM120 to win in production).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability


def test_sm12x_priority_has_sm90_before_sm120():
    """SM90 must come before SM120 in the SM12x backend priority list.

    If this ordering is reversed, SM120 (with its incompatible fp8_ds_mla
    KV layout) would be tried first and selected for GLM-5.3 Flash,
    causing the production OOM.
    """
    from vllm.platforms.cuda import _get_backend_priorities
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    sm12 = DeviceCapability(major=12, minor=1)
    priorities = _get_backend_priorities(
        use_mla=True,
        device_capability=sm12,
    )

    sm90_idx = None
    sm120_idx = None
    for i, backend in enumerate(priorities):
        if backend is AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90:
            sm90_idx = i
        elif backend is AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120:
            sm120_idx = i

    assert sm90_idx is not None, "SM90 backend not in SM12x priority list"
    assert sm120_idx is not None, "SM120 backend not in SM12x priority list"
    assert sm90_idx < sm120_idx, (
        f"SM90 (index {sm90_idx}) must come before SM120 (index {sm120_idx}) "
        f"in the SM12x priority list. Full list: {[b.name for b in priorities]}"
    )


def test_sm90_supports_glm53_flash_nope_config():
    """SM90 backend must accept the GLM-5.3 Flash NoPE configuration.

    GLM-5.3 Flash uses kv_lora_rank=512, qk_rope_head_dim=0 (NoPE MLA),
    and fp8 KV cache. This is the exact combination that was rejected in
    production, falling back to SM120.
    """
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )

    # Build a mock vllm_config with GLM-5.3 Flash's hf_text_config signature.
    hf_text_config = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_topk=2048,
    )
    mock_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
    )

    sm12 = DeviceCapability(major=12, minor=1)

    with mock.patch(
        "vllm.utils.flashinfer.has_flashinfer_sm90_nope_mla",
        return_value=True,
    ), mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=mock_config,
    ):
        result = FlashInferMLASparseSM90Backend.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=sm12,
        )
    assert result is None, (
        f"SM90 should accept GLM-5.3 Flash NoPE config with fp8 KV, "
        f"but rejected with: {result}"
    )


def test_sm90_rejects_fp8_kv_when_flashinfer_lacks_sm90_mla():
    """SM90 must reject fp8 KV when has_flashinfer_sm90_nope_mla() is False.

    This is the production failure mode: the deployed FlashInfer build
    lacked the ckv_scale_arr parameter in BatchMLAPagedAttentionWrapper.run,
    so SM90 rejected fp8 KV and SM120 (with incompatible KV layout) took
    over, causing OOM.
    """
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )

    hf_text_config = SimpleNamespace(
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        index_topk=2048,
    )
    mock_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_text_config),
    )

    sm12 = DeviceCapability(major=12, minor=1)

    with mock.patch(
        "vllm.utils.flashinfer.has_flashinfer_sm90_nope_mla",
        return_value=False,
    ), mock.patch(
        "vllm.config.get_current_vllm_config",
        return_value=mock_config,
    ):
        result = FlashInferMLASparseSM90Backend.supports_combination(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            device_capability=sm12,
        )
    assert result is not None, (
        "SM90 should reject fp8 KV when has_flashinfer_sm90_nope_mla() "
        "is False, but it accepted it"
    )
    assert "SM90" in result or "FlashInfer" in result, (
        f"Rejection message should mention SM90/FlashInfer, got: {result}"
    )


def test_sm90_supports_compute_capability_for_sm12x():
    """SM90 backend must claim support for SM12x (GB10 is SM121)."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )

    sm12 = DeviceCapability(major=12, minor=1)
    assert FlashInferMLASparseSM90Backend.supports_compute_capability(sm12), (
        "SM90 backend must support SM12x (GB10/SM121)"
    )


def test_sm90_supported_head_sizes_include_512():
    """SM90 must support head_size=512 (kv_lora_rank=512 + qk_rope_head_dim=0).

    This is the NoPE MLA shape used by GLM-5.3 Flash.
    """
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )

    head_sizes = FlashInferMLASparseSM90Backend.get_supported_head_sizes()
    assert 512 in head_sizes, (
        f"SM90 must support head_size=512 for GLM-5.3 Flash NoPE MLA, "
        f"got: {head_sizes}"
    )


def test_sm90_supports_fp8_kv_cache_dtype():
    """SM90 must list fp8 and fp8_e4m3 as supported KV cache dtypes."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm90 import (
        FlashInferMLASparseSM90Backend,
    )

    supported = FlashInferMLASparseSM90Backend.supported_kv_cache_dtypes
    assert "fp8" in supported, f"fp8 must be in supported KV dtypes: {supported}"
    assert "fp8_e4m3" in supported, (
        f"fp8_e4m3 must be in supported KV dtypes: {supported}"
    )
