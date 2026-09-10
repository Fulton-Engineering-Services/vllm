# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the GLM-5.3-Flash vision tower.

Covers ``Glm5NextVisionTransformer`` (rotary geometry, forward shapes,
encoder-metadata parity) plus the GLM-5.3-specific SwiGLU clamping and
merger op order. Runs on CPU (TORCH_SDPA ViT backend, gloo world-size-1)
with tiny configs; no checkpoint weights needed.
"""

import contextlib
import os
import tempfile

import pytest
import torch

import vllm.models.glm5next.nvidia.multimodal as glm5_mm
from tests.utils import ensure_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)

PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
IN_CHANNELS = 3
PATCH_DIM = IN_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 1176


def _qk_rmsnorm_ref(q, k, q_weight, k_weight, eps):
    """Eager fp32 reference for the fused triton q/k RMSNorm kernel
    (``vllm/models/common/ops/fused_qk_rmsnorm.py``), which has no CPU
    build. Bit-for-bit contract: fp32 accumulate, single cast at store."""

    def rms(x, w):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + eps) * w.float()).to(x.dtype)

    return rms(q, q_weight), rms(k, k_weight)


@pytest.fixture
def dist_env():
    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


@pytest.fixture
def vision_tower(dist_env, monkeypatch):
    """Tiny 2-block tower with deterministic weights (the conv/linear layers
    use ``torch.empty`` — uninitialized memory NaNs the forward on CPU)."""
    monkeypatch.setattr(glm5_mm, "fused_q_kv_rmsnorm", _qk_rmsnorm_ref)
    text_config = Glm5NextTextConfig(swiglu_limit=10.0)
    vision_config = Glm5NextVisionConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        out_hidden_size=128,
        projection_intermediate_size=256,
        patch_size=PATCH_SIZE,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        spatial_merge_size=MERGE_SIZE,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        tower = glm5_mm.Glm5NextVisionTransformer(
            text_config, vision_config, norm_eps=1e-6, prefix="visual"
        )
    finally:
        torch.set_default_dtype(prev_dtype)
    with torch.no_grad():
        for p in tower.parameters():
            if p.numel():
                p.copy_(torch.randn_like(p) * 0.02)
    for mod in tower.modules():
        if isinstance(mod, LinearBase) and not mod.weight.is_meta:
            dispatch_cpu_unquantized_gemm(mod, remove_weight=False)
    return tower


def _expected_pos_ids(t, h, w, merge):
    """Hand-computed merged h/w position ids matching rot_pos_emb's
    reshape/permute layout (grid-major over merge blocks)."""
    hpos = (
        torch.arange(h)
        .unsqueeze(1)
        .expand(-1, w)
        .reshape(h // merge, merge, w // merge, merge)
        .permute(0, 2, 1, 3)
        .flatten()
    )
    wpos = (
        torch.arange(w)
        .unsqueeze(0)
        .expand(h, -1)
        .reshape(h // merge, merge, w // merge, merge)
        .permute(0, 2, 1, 3)
        .flatten()
    )
    return torch.stack([hpos, wpos], dim=-1).repeat(t, 1)


class TestRotPosEmb:
    def test_position_ids_match_hand_computed(self, vision_tower):
        grid = [[1, 4, 6], [2, 2, 4]]
        _, _, pos_ids = vision_tower.rot_pos_emb(grid)
        expected = torch.cat(
            [_expected_pos_ids(t, h, w, MERGE_SIZE) for t, h, w in grid]
        )
        torch.testing.assert_close(pos_ids.cpu(), expected)

    def test_partial_rotary_width(self, vision_tower):
        # partial_rotary_factor=0.5 -> cos/sin carry half the head dim.
        head_dim = 64 // 4
        cos, sin, _ = vision_tower.rot_pos_emb([[1, 4, 4]])
        assert cos.shape == sin.shape == (4 * 4, head_dim // 2)

    def test_cos_sin_indexed_by_pos_ids(self, vision_tower):
        cos, sin, pos_ids = vision_tower.rot_pos_emb([[1, 2, 2]])
        base_cos, base_sin = vision_tower.rotary_pos_emb.get_cos_sin(2)
        torch.testing.assert_close(
            cos, base_cos[pos_ids.to(base_cos.device)].flatten(1)
        )
        torch.testing.assert_close(
            sin, base_sin[pos_ids.to(base_sin.device)].flatten(1)
        )


class TestVisionTowerForward:
    def test_forward_output_shape_and_finite(self, vision_tower):
        torch.manual_seed(0)
        grid = [[1, 4, 8]]  # 32 patches -> 32/4 = 8 merged tokens
        x = torch.randn(4 * 8, PATCH_DIM)
        out = vision_tower(x, grid)
        assert out.shape == (4 * 8 // MERGE_SIZE**2, 128)
        assert out.dtype == vision_tower.dtype
        assert torch.isfinite(out).all()

    def test_multi_item_cu_seqlens_split(self, vision_tower):
        """Two items (image grid_t=1, video grid_t=2) run as separate
        sequences: cu_seqlens boundaries follow per-frame patch counts."""
        torch.manual_seed(0)
        grid = [[1, 4, 4], [2, 4, 4]]
        total = 16 + 2 * 16
        x = torch.randn(total, PATCH_DIM)
        out = vision_tower(x, grid)
        assert out.shape == (total // MERGE_SIZE**2, 128)
        assert torch.isfinite(out).all()

    def test_encoder_metadata_matches_eager(self, vision_tower):
        """prepare_encoder_metadata (CUDA-graph path) must agree with the
        eager in-forward rebuild — the fleet is eager, but a drift here is
        exactly the class of bug that corrupts image features silently."""
        grid = [[1, 4, 8], [2, 2, 4]]
        md = vision_tower.prepare_encoder_metadata(grid, device=torch.device("cpu"))

        cos, sin, _ = vision_tower.rot_pos_emb(grid)
        torch.testing.assert_close(md["rotary_pos_emb_cos"], cos)
        torch.testing.assert_close(md["rotary_pos_emb_sin"], sin)

        # eager: per-frame patch counts repeated by grid_t, prefixed with 0.
        # [1,4,8] -> 32 patches; [2,2,4] -> 8 patches per frame x 2 frames.
        cu = [0, 32, 40, 48]
        torch.testing.assert_close(
            md["cu_seqlens"].cpu(), torch.tensor(cu, dtype=torch.int32)
        )
        # TORCH_SDPA needs no max_seqlen or sequence_lengths.
        assert md["max_seqlen"].item() == 0
        assert md["sequence_lengths"] is None

        torch.manual_seed(0)
        x = torch.randn(48, PATCH_DIM)
        torch.manual_seed(0)
        out_eager = vision_tower(x, grid)
        out_meta = vision_tower(x, grid, encoder_metadata=md)
        torch.testing.assert_close(out_meta, out_eager)


class TestSwigluClamp:
    def test_inputs_clamped_to_limit(self, dist_env):
        """Gate clamps at +limit (no lower bound), up clamps at ±limit, so
        the product can exceed the limit — what matters is inputs are
        clamped (GLM-5.3 delta vs GLM-4V's unclamped SwiGLU)."""
        act = SiluAndMulWithClamp(swiglu_limit=7.0)
        x = torch.randn(64, 32) * 100
        out = act(x)
        assert torch.isfinite(out).all()
        gate = torch.clamp(x[:, :16], max=7.0)
        up = torch.clamp(x[:, 16:], min=-7.0, max=7.0)
        expected = gate * torch.sigmoid(gate) * up
        torch.testing.assert_close(out, expected)

    def test_small_values_pass_through(self, dist_env):
        act = SiluAndMulWithClamp(swiglu_limit=10.0)
        x = torch.randn(16, 8) * 0.01
        gate, up = x.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate) * up
        torch.testing.assert_close(act(x), expected)


class TestMergerOpOrder:
    def test_merger_op_sequence(self, vision_tower):
        """GLM-5.3 merger: proj -> LayerNorm -> GELU -> gate_up ->
        clamp-SwiGLU -> down. Pins the GELU-on-projection delta vs GLM-4V."""
        merger = vision_tower.merger
        x = torch.randn(8, merger.hidden_size)

        def gelu_branch(inp):
            return merger.extra_activation_func(merger.post_projection_norm(inp))

        proj_out, _ = merger.proj(x)
        h = gelu_branch(proj_out)
        gate_up, _ = merger.gate_up_proj(h)
        expected, _ = merger.down_proj(merger.act_fn(gate_up))
        torch.testing.assert_close(merger(x), expected)

        # Swapped order (GELU before norm) must NOT match.
        wrong_h = merger.post_projection_norm(merger.extra_activation_func(proj_out))
        wrong_gate_up, _ = merger.gate_up_proj(wrong_h)
        wrong, _ = merger.down_proj(merger.act_fn(wrong_gate_up))
        assert not torch.allclose(merger(x), wrong)


class TestWeightInitSanity:
    def test_conv_weights_initialized_after_load_pattern(self, vision_tower):
        """Guard the test fixture itself: production towers get weights from
        the checkpoint, but a CPU-constructed tower must never silently
        forward torch.empty() memory (it NaNs — see WI-2 bring-up)."""
        for name, p in vision_tower.named_parameters():
            assert torch.isfinite(p).all(), f"uninitialized param: {name}"
