# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility wrapper for DeepGEMM API changes.

Users of vLLM should always import **only** these wrappers.
"""

import contextlib
import functools
import importlib
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, NoReturn

import torch

import vllm.envs as envs
from vllm.logger import logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv

_DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES: set[str] = {
    "qwen3_5_text",
    "qwen3_5_moe_text",
}

PAGED_MQA_PAGE_SIZES: tuple[int, ...] = (32, 64)


def should_auto_disable_deep_gemm(model_type: str | None) -> bool:
    """Check if DeepGemm should be auto-disabled for this model on Blackwell.

    Returns True if the model is known to have accuracy degradation with
    DeepGemm's E8M0 scale format on Blackwell GPUs (SM100+).
    """
    if model_type is None:
        return False
    if not (
        current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    ):
        return False
    return model_type in _DEEPGEMM_BLACKWELL_EXCLUDED_MODEL_TYPES


class DeepGemmQuantScaleFMT(Enum):
    # Float32 scales in Float32 tensor
    FLOAT32 = 0
    # Compute float32 scales and ceil the scales to UE8M0.
    # Keep the scales in Float32 tensor.
    FLOAT32_CEIL_UE8M0 = 1
    # Compute float32 scales and ceil the scales to UE8M0.
    # Pack the scales into a int32 tensor where each int32
    # element contains 4 scale values.
    UE8M0 = 2

    @classmethod
    def init_oracle_cache(cls) -> None:
        """Initialize the oracle decision and store it in the class cache"""
        cached = getattr(cls, "_oracle_cache", None)
        if cached is not None:
            return

        use_e8m0 = (
            envs.VLLM_USE_DEEP_GEMM_E8M0
            and is_deep_gemm_supported()
            and (_fp8_gemm_nt_impl is not None)
        )
        if not use_e8m0:
            cls._oracle_cache = cls.FLOAT32  # type: ignore
            return

        cls._oracle_cache = (  # type: ignore
            cls.UE8M0
            if (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
            else cls.FLOAT32_CEIL_UE8M0
        )

    @classmethod
    def from_oracle(cls) -> "DeepGemmQuantScaleFMT":
        """Return the oracle decision, initializing it on first use.

        The cache is normally populated by ``_lazy_init()`` (e.g. during
        engine startup), but standalone consumers such as ``QuantFP8`` with an
        explicit ``use_ue8m0=True`` can reach this before any DeepGEMM kernel
        wrapper has run. Resolve the DeepGEMM symbols and initialize the
        decision here instead of asserting; without DeepGEMM this yields
        FLOAT32, matching ``is_deep_gemm_e8m0_used()``.
        """
        cached = getattr(cls, "_oracle_cache", None)
        if cached is None:
            _lazy_init()
            cls.init_oracle_cache()
            cached = cls._oracle_cache  # type: ignore[attr-defined]
        return cached


@functools.cache
def is_deep_gemm_supported() -> bool:
    """Return `True` if DeepGEMM is supported on the current platform.
    Currently, only Hopper and Blackwell GPUs are supported.
    """
    is_supported_arch = current_platform.support_deep_gemm()
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch


@functools.cache
def is_deep_gemm_e8m0_used() -> bool:
    """Return `True` if vLLM is configured to use DeepGEMM "
    "E8M0 scale on a Hopper or Blackwell-class GPU.
    """
    if not is_deep_gemm_supported():
        logger.debug_once(
            "DeepGEMM E8M0 disabled: DeepGEMM not supported on this system."
        )
        return False

    _lazy_init()

    if _fp8_gemm_nt_impl is None:
        logger.info_once("DeepGEMM E8M0 disabled: _fp8_gemm_nt_impl not found")
        return False

    if envs.VLLM_USE_DEEP_GEMM_E8M0:
        logger.info_once("DeepGEMM E8M0 enabled on current platform.")
        return True

    logger.info_once("DeepGEMM E8M0 disabled on current configuration.")
    return False


def _missing(*_: Any, **__: Any) -> NoReturn:
    """Placeholder for unavailable DeepGEMM backend."""
    raise RuntimeError(
        "DeepGEMM backend is unavailable in the current vLLM environment, "
        "or the available DeepGEMM package does not provide the required APIs "
        "for these kernels."
    )


_cublaslt_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_gemm_nt_impl: Callable[..., Any] | None = None
_fp8_einsum_impl: Callable[..., Any] | None = None
_grouped_impl: Callable[..., Any] | None = None
_grouped_masked_impl: Callable[..., Any] | None = None
_grouped_fp4_impl: Callable[..., Any] | None = None
_fp8_fp4_mqa_logits_impl: Callable[..., Any] | None = None
_fp8_fp4_paged_mqa_logits_impl: Callable[..., Any] | None = None
_get_paged_mqa_logits_metadata_impl: Callable[..., Any] | None = None
_tf32_hc_prenorm_gemm_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_tensor_impl: Callable[..., Any] | None = None
_get_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = None
_get_theoretical_mk_alignment_for_contiguous_layout_impl: Callable[..., Any] | None = (
    None
)
_transform_sf_into_required_layout_impl: Callable[..., Any] | None = None
_pack_ue8m0_to_int_impl: Callable[..., Any] | None = None
_get_mn_major_tma_aligned_packed_ue8m0_tensor_impl: Callable[..., Any] | None = None
_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl: (
    Callable[..., Any] | None
) = None


@functools.cache
def _import_deep_gemm():
    """Import the deep_gemm module.

    Prefers an externally installed ``deep_gemm`` package (so users can
    pin a specific version), then falls back to the vendored copy bundled
    in the vLLM wheel.

    Returns ``None`` when neither source is usable.
    """
    # 1. Try the external (pip-installed) package first.
    try:
        module = importlib.import_module("deep_gemm")
        logger.debug_once("Imported deep_gemm module from site-packages")
        return module
    except ImportError:
        logger.info_once(
            "deep_gemm not found in site-packages, "
            "trying vendored vllm.third_party.deep_gemm"
        )

    # 2. Fall back to the vendored copy bundled in the vLLM wheel.
    try:
        module = importlib.import_module("vllm.third_party.deep_gemm")
        logger.debug_once("Imported deep_gemm module from vllm.third_party.deep_gemm")
        return module
    except ImportError:
        logger.info_once("Vendored deep_gemm not found either")
    except Exception as e:
        # The vendored module may raise RuntimeError during _C.init()
        # if JIT include files are missing (e.g. incomplete wheel).
        logger.warning_once("Failed to import vendored deep_gemm: %s", e)

    return None


def _apply_pdl(mod, enable: bool = True) -> None:
    mod_name = getattr(mod, "__name__", str(mod))
    try:
        set_pdl_fn = getattr(mod, "set_pdl", None)
        if set_pdl_fn is None:
            return
        set_pdl_fn(enable)
        logger.info_once(
            "DeepGEMM PDL %s on %s.",
            "enabled" if enable else "disabled",
            mod_name,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning_once("Failed to set DeepGEMM PDL on %s: %s", mod_name, e)


def _lazy_init() -> None:
    """Import deep_gemm and resolve symbols on first use."""
    global _cublaslt_gemm_nt_impl
    global _fp8_gemm_nt_impl, _fp8_einsum_impl
    global _grouped_impl, _grouped_masked_impl, _grouped_fp4_impl
    global _fp8_fp4_mqa_logits_impl, _fp8_fp4_paged_mqa_logits_impl
    global _get_paged_mqa_logits_metadata_impl
    global _tf32_hc_prenorm_gemm_impl
    global _get_mn_major_tma_aligned_tensor_impl
    global _get_mk_alignment_for_contiguous_layout_impl
    global _get_theoretical_mk_alignment_for_contiguous_layout_impl
    global _transform_sf_into_required_layout_impl
    global _pack_ue8m0_to_int_impl
    global _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    global _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl
    # fast path
    if (
        _cublaslt_gemm_nt_impl is not None
        or _fp8_gemm_nt_impl is not None
        or _fp8_einsum_impl is not None
        or _grouped_impl is not None
        or _grouped_masked_impl is not None
        or _grouped_fp4_impl is not None
        or _fp8_fp4_mqa_logits_impl is not None
        or _fp8_fp4_paged_mqa_logits_impl is not None
        or _get_paged_mqa_logits_metadata_impl is not None
        or _tf32_hc_prenorm_gemm_impl is not None
        or _get_mk_alignment_for_contiguous_layout_impl is not None
        or _transform_sf_into_required_layout_impl is not None
        or _pack_ue8m0_to_int_impl is not None
        or _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
        or _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is not None
    ):
        return

    if not has_deep_gemm():
        return

    # Set up deep_gemm cache path
    DEEP_GEMM_JIT_CACHE_ENV_NAME = "DG_JIT_CACHE_DIR"
    if not os.environ.get(DEEP_GEMM_JIT_CACHE_ENV_NAME, None):
        os.environ[DEEP_GEMM_JIT_CACHE_ENV_NAME] = os.path.join(
            envs.VLLM_CACHE_ROOT, "deep_gemm"
        )

    _dg = _import_deep_gemm()
    if _dg is None:
        return

    # Enable PDL for DeepGEMM on architectures that support it (SM90+).
    if current_platform.is_arch_support_pdl():
        _apply_pdl(_dg, True)
    _cublaslt_gemm_nt_impl = getattr(_dg, "cublaslt_gemm_nt", None)
    _fp8_gemm_nt_impl = getattr(_dg, "fp8_gemm_nt", None)
    _fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)
    _grouped_impl = getattr(_dg, "m_grouped_fp8_gemm_nt_contiguous", None)
    _grouped_masked_impl = getattr(_dg, "fp8_m_grouped_gemm_nt_masked", None)
    _grouped_fp4_impl = getattr(_dg, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    # DeepGEMM exposes fp8_fp4_*_mqa_logits as the canonical symbols that
    # handle both the FP8 and FP4 Q/K paths via a tuple-typed `q`.
    _fp8_fp4_mqa_logits_impl = getattr(_dg, "fp8_fp4_mqa_logits", None)
    _fp8_fp4_paged_mqa_logits_impl = getattr(_dg, "fp8_fp4_paged_mqa_logits", None)
    _get_paged_mqa_logits_metadata_impl = getattr(
        _dg, "get_paged_mqa_logits_metadata", None
    )
    _tf32_hc_prenorm_gemm_impl = getattr(_dg, "tf32_hc_prenorm_gemm", None)
    _get_mn_major_tma_aligned_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_tensor", None
    )
    _get_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_mk_alignment_for_contiguous_layout", None
    )
    _get_theoretical_mk_alignment_for_contiguous_layout_impl = getattr(
        _dg, "get_theoretical_mk_alignment_for_contiguous_layout", None
    )
    _transform_sf_into_required_layout_impl = getattr(
        _dg, "transform_sf_into_required_layout", None
    )
    _pack_ue8m0_to_int_impl = getattr(_dg, "pack_ue8m0_to_int", None)
    _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl = getattr(
        _dg, "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", None
    )
    DeepGemmQuantScaleFMT.init_oracle_cache()


def get_num_sms() -> int:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    return int(dg.get_num_sms())


def set_num_sms(num_sms: int) -> None:
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_num_sms(num_sms)


def get_mk_alignment_for_contiguous_layout() -> list[int]:
    _lazy_init()
    if _get_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()
    return [mk_align_size, mk_align_size]


def get_theoretical_mk_alignment_for_contiguous_layout(
    expected_m: int | None = None,
    num_groups: int | None = None,
) -> int:
    """Per-call optimal M alignment for grouped contiguous GEMMs.

    `expected_m` is the TOTAL routed tokens (sum across experts, typically
    M × num_topk). `num_groups` is the number of experts on this rank.
    The helper divides to recover per-expert em and picks an alignment based
    on data-driven thresholds (see deep_gemm runtime.hpp comments).

    Older callers that omit `num_groups` are interpreted as passing already
    per-expert em (legacy behaviour preserved for backward compat).
    """
    _lazy_init()
    if _get_theoretical_mk_alignment_for_contiguous_layout_impl is None:
        return _missing()
    if num_groups is None:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(expected_m)
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    try:
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(
            expected_m, num_groups
        )
    except TypeError:
        per_group_m = None if expected_m is None else cdiv(expected_m, num_groups)
        return _get_theoretical_mk_alignment_for_contiguous_layout_impl(per_group_m)


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    """Set DeepGEMM's BLOCK_M cap for grouped contiguous GEMMs.

    The DG heuristic constrains BLOCK_M ≤ this value when picking a kernel
    layout. Use this in concert with `compute_aligned_M_and_alignment`'s
    per-call alignment so the workspace's per-expert padding matches the
    kernel's BLOCK_M; a mismatch leads to the scheduler reading the wrong
    expert_id from `m_indices` at `m_block_idx * BLOCK_M` stride and
    OOB-indexing the B-weights tensor (manifests as IMA under CUDA-graph
    replay).
    """
    _lazy_init()
    dg = _import_deep_gemm()
    if dg is None:
        raise RuntimeError("DeepGEMM is not available")
    dg.set_mk_alignment_for_contiguous_layout(value)


@contextlib.contextmanager
def mk_alignment_scope(value: int):
    """Temporarily set DeepGEMM's BLOCK_M cap, restoring on exit.

    Use around a sequence of grouped-contiguous GEMM calls whose workspace
    is padded to `value` (typically the per_call_align returned by
    `compute_aligned_M_and_alignment`).
    """
    prev = get_mk_alignment_for_contiguous_layout()[0]
    set_mk_alignment_for_contiguous_layout(value)
    try:
        yield
    finally:
        set_mk_alignment_for_contiguous_layout(prev)


def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for DeepGEMM's get_mn_major_tma_aligned_tensor"""
    _lazy_init()
    if _get_mn_major_tma_aligned_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_tensor_impl(x)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack 4 UE8M0 (uint8) scales into one int32.

    DeepGEMM's SM100/SM120 FP8/FP4 kernels accept either ``float32`` scales
    (legacy format, 4 B/scale) or ``int32`` packed UE8M0 scales (1 B/scale
    after 4:1 packing — 4× smaller than the legacy fp32 representation).
    """
    _lazy_init()
    if _pack_ue8m0_to_int_impl is None:
        return _missing()
    return _pack_ue8m0_to_int_impl(x)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(x: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 (uint8) → int32 with the MN-major TMA-aligned layout the
    DeepGEMM kernels consume directly. 16× smaller than the fp32 legacy SF
    format. Use for non-grouped 2D scale tensors.
    """
    _lazy_init()
    if _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_mn_major_tma_aligned_packed_ue8m0_tensor_impl(x)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf: torch.Tensor,
    ks_tensor: torch.Tensor,
    ks: list[int],
    gran_k: int,
) -> torch.Tensor:
    """Grouped (3D, expert-batched) variant of
    ``get_mn_major_tma_aligned_packed_ue8m0_tensor``. Use for MoE weight
    scale tensors of shape ``(num_experts, mn, k_scale)``.
    """
    _lazy_init()
    if _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl is None:
        return _missing()
    return _get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor_impl(
        sf, ks_tensor, ks, gran_k
    )


def cublaslt_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _cublaslt_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    return _cublaslt_gemm_nt_impl(*args, **kwargs)


def fp8_gemm_nt(*args, **kwargs):
    _lazy_init()
    if _fp8_gemm_nt_impl is None:
        return _missing(*args, **kwargs)
    if "is_deep_gemm_e8m0_used" in kwargs:
        use_ue8m0 = kwargs["is_deep_gemm_e8m0_used"]
        del kwargs["is_deep_gemm_e8m0_used"]
    else:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    return _fp8_gemm_nt_impl(*args, disable_ue8m0_cast=not use_ue8m0, **kwargs)


def fp8_einsum(*args, **kwargs):
    _lazy_init()
    if _fp8_einsum_impl is None:
        return _missing(*args, **kwargs)
    return _fp8_einsum_impl(*args, **kwargs)


def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def m_grouped_fp8_fp4_gemm_nt_contiguous(*args, **kwargs):
    _lazy_init()
    if _grouped_fp4_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_fp4_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
    _lazy_init()
    if _grouped_masked_impl is None:
        return _missing(*args, **kwargs)
    return _grouped_masked_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def transform_sf_into_required_layout(*args, **kwargs):
    _lazy_init()
    if _transform_sf_into_required_layout_impl is None:
        return _missing(*args, **kwargs)
    return _transform_sf_into_required_layout_impl(
        *args, disable_ue8m0_cast=not is_deep_gemm_e8m0_used(), **kwargs
    )


def _dequant_fp4_q_for_sm121(
    q: tuple[torch.Tensor, torch.Tensor | None],
    head_dim: int,
) -> tuple[torch.Tensor, None]:
    """Dequantize FP4 Q to FP8 for SM121 SM90-kernel compatibility.

    Returns (q_fp8, None) suitable for the FP8 dispatch path.
    Only dequantizes Q — KV cache conversion is handled separately for the
    paged decode path.
    """
    q_vals, q_scale = q
    if q_scale is None:
        return q

    assert q_vals.dtype in (torch.uint8, torch.int8), (
        f"expected uint8/int8 FP4 Q, got {q_vals.dtype}"
    )
    assert q_scale.dtype == torch.int32, (
        f"expected int32 UE8M0 Q scale, got {q_scale.dtype}"
    )

    batch_size, next_n, num_heads, packed_dim = q_vals.shape
    assert packed_dim * 2 == head_dim, (
        f"packed dim {packed_dim} * 2 != head_dim {head_dim}"
    )

    vals_f32 = q_vals.view(torch.uint8).to(torch.float32)
    lo = vals_f32 % 16
    hi = vals_f32 // 16
    fp4_unpacked = torch.stack([lo, hi], dim=-1).reshape(
        batch_size, next_n, num_heads, head_dim
    )

    num_scale_groups = head_dim // 32
    shifts = torch.tensor(
        [0, 8, 16, 24],
        dtype=torch.int32,
        device=q_scale.device,
    )[:num_scale_groups]
    scale_bytes = (
        (q_scale.unsqueeze(-1) >> shifts) & 0xFF
    ).to(torch.float32)
    scale_f32 = torch.where(
        scale_bytes > 0,
        torch.pow(2.0, scale_bytes - 127.0),
        torch.tensor(0.0, device=scale_bytes.device, dtype=torch.float32),
    )

    scale_f32 = scale_f32.reshape(batch_size, next_n, num_heads, num_scale_groups)
    scale_bc = scale_f32.repeat_interleave(32, dim=-1)

    fp32_vals = fp4_unpacked * scale_bc

    fp32_flat = fp32_vals.reshape(-1, head_dim)
    max_abs = fp32_flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    fp8_scale_fold = max_abs / 448.0
    fp8_vals = (fp32_flat / fp8_scale_fold).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    fp8_vals = fp8_vals.reshape(batch_size, next_n, num_heads, head_dim)

    return (fp8_vals, None)


def _convert_fp4_fused_kv_cache_to_fp8(
    fused_kv_cache: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Convert FP4-format fused KV cache to FP8 format for SM121 SM90 compat.

    FP4 physical layout per block:
      [block_kv * head_dim//2 bytes: packed FP4 values (positions contiguous)]
      [block_kv * 4 bytes: int32 UE8M0 scales (one per position)]

    FP8 physical layout per block:
      [block_kv * head_dim bytes: float8_e4m3fn values (positions contiguous)]
      [block_kv * 4 bytes: float32 scales (one per position)]

    The tensor view stride(1)=head_dim//2+4 is a naive row-major view over the
    flat byte buffer.  We create as_strided views that correctly skip the
    scale section, matching the from_blob logic in the C++ dispatch.
    """
    num_kv_blocks, block_kv, _, fp4_row_bytes = fused_kv_cache.shape
    fp4_val_bytes = fp4_row_bytes - 4
    assert fp4_val_bytes * 2 == head_dim, (
        f"fp4_val_bytes {fp4_val_bytes} * 2 != head_dim {head_dim}"
    )

    device = fused_kv_cache.device
    kv_block_stride = fused_kv_cache.stride(0)

    fp4_vals = torch.as_strided(
        fused_kv_cache,
        size=(num_kv_blocks, block_kv, fp4_val_bytes),
        stride=(kv_block_stride, fp4_val_bytes, 1),
    )

    fp4_sf = torch.as_strided(
        fused_kv_cache,
        size=(num_kv_blocks, block_kv, 4),
        stride=(kv_block_stride, 4, 1),
        storage_offset=fused_kv_cache.storage_offset()
        + block_kv * fp4_val_bytes,
    )

    fp4_lo = fp4_vals.bitwise_and(15).to(torch.float32)
    fp4_hi = fp4_vals.bitwise_right_shift(4).to(torch.float32)
    fp4_unpacked = torch.stack([fp4_lo, fp4_hi], dim=-1).reshape(
        num_kv_blocks, block_kv, head_dim
    )

    sf_int32 = fp4_sf.contiguous().view(torch.int32).squeeze(-1)

    num_scale_groups = head_dim // 32
    shifts = torch.tensor(
        [0, 8, 16, 24],
        dtype=torch.int32,
        device=device,
    )[:num_scale_groups]
    sf_ue8m0 = ((sf_int32.unsqueeze(-1) >> shifts) & 0xFF).to(torch.float32)
    sf_f32 = torch.where(
        sf_ue8m0 > 0,
        torch.pow(2.0, sf_ue8m0 - 127.0),
        torch.tensor(0.0, device=device, dtype=torch.float32),
    )

    sf_bc = sf_f32.repeat_interleave(32, dim=-1)

    fp32_vals = fp4_unpacked * sf_bc

    fp32_flat = fp32_vals.reshape(-1, head_dim)
    max_abs = fp32_flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    fp8_scale = max_abs / 448.0
    fp8_vals = (fp32_flat / fp8_scale).clamp(-448.0, 448.0).to(
        torch.float8_e4m3fn
    )
    fp8_vals = fp8_vals.reshape(num_kv_blocks, block_kv, head_dim)

    out = torch.zeros(
        num_kv_blocks,
        block_kv,
        1,
        head_dim + 4,
        dtype=torch.uint8,
        device=device,
    )

    out_vals = torch.as_strided(
        out,
        size=(num_kv_blocks, block_kv, head_dim),
        stride=(out.stride(0), head_dim, 1),
    )
    out_vals.copy_(fp8_vals.view(torch.uint8))

    out_sf = torch.as_strided(
        out,
        size=(num_kv_blocks, block_kv, 4),
        stride=(out.stride(0), 4, 1),
        storage_offset=out.storage_offset() + block_kv * head_dim,
    )
    out_sf.copy_(fp8_scale.reshape(num_kv_blocks, block_kv, 1).view(torch.uint8))

    return out


def _dequant_fp4_qkv_for_sm121(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, None], tuple[torch.Tensor, torch.Tensor]]:
    """Dequantize FP4 Q and KV to FP8 for SM121 SM90-kernel compatibility.

    SM121 routes through SM90 (wgmma) kernels which only accept FP8 inputs.
    Converts packed-FP4 (uint8) + UE8M0 (int32) scales to FP8 (float8_e4m3fn)
    values + FP32 scales.
    """
    q_vals, q_scale = q
    kv_vals, kv_scale = kv
    if q_scale is None:
        return q, kv

    assert q_vals.dtype == torch.uint8
    assert q_scale.dtype == torch.int32
    assert kv_vals.dtype == torch.uint8
    assert kv_scale.dtype == torch.int32

    def _fp4_packed_to_fp8(
        packed: torch.Tensor, ue8m0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_f32 = packed.to(torch.float32)
        lo = (packed_f32 % 16).to(torch.float32)
        hi = (packed_f32 // 16).to(torch.float32)
        vals = torch.stack([lo, hi], dim=-1).flatten(-2)

        scales_s32 = ue8m0.to(torch.int32)
        shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device=scales_s32.device)
        bytes_t = ((scales_s32.unsqueeze(-1) >> shifts) & 0xFF).to(torch.float32)
        exp2 = torch.where(bytes_t > 0, torch.exp2(bytes_t - 127.0), torch.tensor(0.0, device=bytes_t.device))

        D = vals.shape[-1]
        block_scale = exp2.flatten(-2)
        block_scale = block_scale.repeat_interleave(32, dim=-1)
        pad = (D + 127) // 128 * 128 - D
        if pad > 0:
            block_scale = block_scale[..., :D]

        fp32 = vals * block_scale
        max_abs = fp32.abs().max()
        scale = max_abs / 448.0 if max_abs > 0 else 1.0
        fp8 = (fp32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        fp8_scale = torch.tensor([scale], dtype=torch.float32, device=fp32.device)
        return fp8, fp8_scale

    q_vals_fp8, q_scale_fp8 = _fp4_packed_to_fp8(q_vals, q_scale)
    kv_vals_fp8, kv_scale_fp8 = _fp4_packed_to_fp8(kv_vals, kv_scale)
    return (q_vals_fp8, None), (kv_vals_fp8, kv_scale_fp8)


def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    _lazy_init()
    if _fp8_fp4_mqa_logits_impl is None:
        return _missing()

    if q[1] is not None and current_platform.is_device_capability_family(120):
        q, kv = _dequant_fp4_qkv_for_sm121(q, kv)

    return _fp8_fp4_mqa_logits_impl(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def native_next_n_supported(next_n: int) -> bool:
    """Whether the paged MQA logits kernel takes `next_n` Q rows per request.

    SM90 implements only {1, 2, 4}; SM100 and SM120 schedule any `next_n` via
    multi-atom tiles. Unsupported values must be flattened to one row per query.
    """
    if current_platform.is_device_capability_family(90):
        return next_n in (1, 2, 4)
    return True


def _paged_mqa_logits_schedule_slots(num_sms: int, next_n: int) -> int:
    """Scheduler tasks the paged MQA logits kernel launches.

    SM90 `next_n=4` runs one task per 2-CTA multicast cluster rather than per
    SM, and `fp8_fp4_paged_mqa_logits` asserts its metadata is sized to match.
    """
    num_kv_multicast = (
        2 if next_n == 4 and current_platform.is_device_capability_family(90) else 1
    )
    return num_sms // num_kv_multicast


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_size: int,
    num_sms: int,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B, next_n], dtype int32; effective
            context length per Q row.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper
        indices: Optional request index for each varlen row.

    Returns:
        Tensor of shape [slots + 1, 2] consumed by `fp8_fp4_paged_mqa_logits`
        to schedule work across SMs.
    """
    _lazy_init()
    if _get_paged_mqa_logits_metadata_impl is None:
        return _missing()
    next_n = context_lens.shape[1] if context_lens.dim() == 2 else 1
    num_slots = _paged_mqa_logits_schedule_slots(num_sms, next_n)
    kwargs = {} if indices is None else {"indices": indices}
    return _get_paged_mqa_logits_metadata_impl(
        context_lens, block_size, num_slots, **kwargs
    )


def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, 1,
            D+4], dtype ``torch.uint8``, with the last 4 bytes per (block, pos)
            storing the float dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype ``torch.float32``.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by ``get_paged_mqa_logits_metadata``;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.
        clean_logits: Whether to clean the unfilled logits into ``-inf``.
        indices: Optional request index for each varlen row.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        ``torch.float32``.
    """
    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()

    if q[1] is not None and current_platform.is_device_capability_family(120):
        fp4_val_bytes = kv_cache.shape[-1] - 4
        head_dim = fp4_val_bytes * 2
        q = _dequant_fp4_q_for_sm121(q, head_dim)
        kv_cache = _convert_fp4_fused_kv_cache_to_fp8(kv_cache, head_dim)
        # FP8 Q has no per-token scale tuple — fold into weights.
        # The C++ SM90 path handles q_scale=None natively.

    kwargs = {} if indices is None else {"indices": indices}
    return _fp8_fp4_paged_mqa_logits_impl(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
        **kwargs,
    )


def tf32_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """
    Perform the following computation:
        out = x.float() @ fn.T
        sqrsum = x.float().square().sum(-1)

    See the caller function for shape requirement
    """
    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _align(x: int, y: int) -> int:
    return cdiv(x, y) * y


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/csrc/utils/math.hpp#L19
def get_tma_aligned_size(x: int, element_size: int) -> int:
    return _align(x, 16 // element_size)


DEFAULT_BLOCK_SIZE = [128, 128]


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/dd6ed14acbc7445dcef224248a77ab4d22b5f240/deep_gemm/utils/math.py#L38
@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = DEFAULT_BLOCK_SIZE, use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = current_platform.fp8_dtype()
    assert x.dim() == 2
    m, n = x.shape
    block_m, block_n = block_size
    x_padded = torch.zeros(
        (_align(m, block_m), _align(n, block_n)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, block_m, x_padded.size(1) // block_n, block_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    _, fp8_max = get_fp8_min_max()
    sf = x_amax / fp8_max
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(fp8_dtype)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight_shape: tuple[int, int],
    supports_deep_gemm: bool | None = None,
):
    if supports_deep_gemm is None:
        supports_deep_gemm = is_deep_gemm_supported()

    # Verify DeepGEMM N/K dims requirements
    # NOTE: Also synchronized with test_w8a8_block_fp8_deep_gemm_matmul
    # test inside kernels/quantization/test_block_fp8.py
    N_MULTIPLE = 64
    K_MULTIPLE = 128

    return (
        supports_deep_gemm
        and output_dtype == torch.bfloat16
        and weight_shape[0] % N_MULTIPLE == 0
        and weight_shape[1] % K_MULTIPLE == 0
    )


__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "fp8_einsum",
    "m_grouped_fp8_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "native_next_n_supported",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "set_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
    "get_theoretical_mk_alignment_for_contiguous_layout",
    "pack_ue8m0_to_int",
    "get_mn_major_tma_aligned_packed_ue8m0_tensor",
    "get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor",
]
