# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for VLLM_B12X_MOE_FP4_AUTO env-knob mode resolution."""

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
)


def _dummy_moe_config():
    return FusedMoEConfig(
        num_experts=72,
        experts_per_token=8,
        hidden_dim=4096,
        intermediate_size=4096,
        num_local_experts=72,
        num_logical_experts=72,
        activation=MoEActivation.SILU,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        in_dtype=torch.bfloat16,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
    )


def _nvfp4_quant_config(activation_dtype="nvfp4"):
    scale = torch.ones(1, dtype=torch.float32)
    return FusedMoEQuantConfig.make(
        quant_dtype=activation_dtype,
        weight_dtype="nvfp4",
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
    )


def test_b12x_auto_disabled_by_default(monkeypatch):
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        B12xExperts, "process_weights_after_loading", lambda self, layer: None
    )
    monkeypatch.delenv("VLLM_B12X_MOE_FP4_AUTO", raising=False)

    experts = B12xExperts(_dummy_moe_config(), _nvfp4_quant_config())
    assert experts._quant_mode == "nvfp4"
    assert experts._source_format == "modelopt_nvfp4"


def test_b12x_auto_engaged(monkeypatch):
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        B12xExperts, "process_weights_after_loading", lambda self, layer: None
    )
    monkeypatch.setenv("VLLM_B12X_MOE_FP4_AUTO", "1")

    experts = B12xExperts(_dummy_moe_config(), _nvfp4_quant_config())
    assert experts._quant_mode == "nvfp4"
    assert experts._effective_quant_mode == "nvfp4_auto"
    assert experts._source_format == "modelopt_nvfp4"


def test_b12x_auto_does_not_affect_w4a16(monkeypatch):
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        B12xExperts, "process_weights_after_loading", lambda self, layer: None
    )
    monkeypatch.setenv("VLLM_B12X_MOE_FP4_AUTO", "1")

    experts = B12xExperts(_dummy_moe_config(), _nvfp4_quant_config(None))
    assert experts._quant_mode == "w4a16"


def test_b12x_auto_does_not_affect_w4a8_nvfp4(monkeypatch):
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        B12xExperts, "process_weights_after_loading", lambda self, layer: None
    )
    monkeypatch.setenv("VLLM_B12X_MOE_FP4_AUTO", "1")

    experts = B12xExperts(_dummy_moe_config(), _nvfp4_quant_config("mxfp8"))
    assert experts._quant_mode == "w4a8_nvfp4"


def test_b12x_auto_does_not_affect_mxfp4(monkeypatch):
    monkeypatch.setattr(B12xExperts, "_supports_current_device", lambda: True)
    monkeypatch.setattr(
        B12xExperts, "process_weights_after_loading", lambda self, layer: None
    )
    monkeypatch.setenv("VLLM_B12X_MOE_FP4_AUTO", "1")

    scale = torch.ones(1, dtype=torch.float32)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=None,
        weight_dtype="mxfp4",
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
    )
    experts = B12xExperts(_dummy_moe_config(), quant_config)
    assert experts._quant_mode == "w4a16"