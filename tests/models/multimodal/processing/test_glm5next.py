# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the GLM-5.3-Flash vLLM-native multimodal processor.

Covers the geometry helpers and frame sampler in
``vllm/transformers_utils/processors/glm5next.py`` and the
``Glm5NextMultiModalProcessor`` contract in
``vllm/models/glm5next/nvidia/multimodal.py``. CPU-only; no checkpoint
weights are needed (the Mock-based ``ProcessingInfo`` pattern from
``test_glm4_1v.py``).
"""

from unittest.mock import Mock

import numpy as np
import pytest

from vllm.models.glm5next.nvidia.multimodal import (
    Glm5NextMultiModalProcessor,
    Glm5NextProcessingInfo,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.transformers_utils.processors.glm5next import (
    _MAX_VIDEO_TOKENS,
    GLM_VIDEO_DEFAULT_FPS,
    GLM_VIDEO_DEFAULT_MAX_FRAMES,
    _pixel_budget,
    glm_sample_frame_indices,
    smart_resize,
)

PATCH_SIZE = 14
MERGE_SIZE = 2
TEMPORAL_PATCH_SIZE = 2
# pixels covered by one vision token: temporal * (patch * merge)^2
PIXELS_PER_TOKEN = TEMPORAL_PATCH_SIZE * (PATCH_SIZE * MERGE_SIZE) ** 2  # 1568


def _make_processing_info(min_tokens=16, max_tokens=8000, max_video_tokens=None):
    """Mock Glm5NextProcessingInfo wired like test_glm4_1v.py."""
    max_video_tokens = (
        _MAX_VIDEO_TOKENS if max_video_tokens is None else max_video_tokens
    )
    info = Mock(spec=Glm5NextProcessingInfo)
    vision_config = info.get_hf_config.return_value.vision_config
    vision_config.patch_size = PATCH_SIZE
    vision_config.spatial_merge_size = MERGE_SIZE
    vision_config.temporal_patch_size = TEMPORAL_PATCH_SIZE

    image_processor = info.get_hf_processor.return_value.image_processor
    image_processor.min_image_tokens = min_tokens
    image_processor.max_image_tokens = max_tokens
    image_processor.patch_size = PATCH_SIZE
    image_processor.merge_size = MERGE_SIZE
    image_processor.temporal_patch_size = TEMPORAL_PATCH_SIZE
    image_processor.patch_expand_factor = 1
    video_processor = info.get_hf_processor.return_value.video_processor
    video_processor.min_image_tokens = min_tokens
    video_processor.max_image_tokens = max_video_tokens
    video_processor.patch_size = PATCH_SIZE
    video_processor.merge_size = MERGE_SIZE
    video_processor.temporal_patch_size = TEMPORAL_PATCH_SIZE

    info._processor_pixel_budget.side_effect = lambda proc: (
        Glm5NextProcessingInfo._processor_pixel_budget(info, proc)
    )
    info._get_image_max_pixels.side_effect = lambda: (
        Glm5NextProcessingInfo._get_image_max_pixels(info)
    )
    info._get_video_max_pixels.side_effect = lambda: (
        Glm5NextProcessingInfo._get_video_max_pixels(info)
    )
    info._get_vision_info.side_effect = lambda **kwargs: (
        Glm5NextProcessingInfo._get_vision_info(info, **kwargs)
    )
    # `ctx` is an instance property on BaseProcessingInfo, so Mock(spec=...)
    # does not expose it; attach it explicitly.
    info.attach_mock(Mock(), "ctx")
    info.ctx.get_merged_mm_kwargs.return_value = {}
    return info


class TestPixelBudget:
    def test_token_bounds_to_pixels(self):
        min_px, max_px = _pixel_budget(
            min_image_tokens=16,
            max_image_tokens=8000,
            patch_size=PATCH_SIZE,
            merge_size=MERGE_SIZE,
            temporal_patch_size=TEMPORAL_PATCH_SIZE,
        )
        assert min_px == 16 * PIXELS_PER_TOKEN
        assert max_px == 8000 * PIXELS_PER_TOKEN

    def test_missing_token_bounds_raise(self):
        with pytest.raises(ValueError, match="min_image_tokens and max_image_tokens"):
            _pixel_budget(None, 8000, PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE)
        with pytest.raises(ValueError, match="min_image_tokens and max_image_tokens"):
            _pixel_budget(16, None, PATCH_SIZE, MERGE_SIZE, TEMPORAL_PATCH_SIZE)


class TestSmartResize:
    def test_alignment_rounds_up_to_factor(self):
        factor = PATCH_SIZE * MERGE_SIZE  # 28
        h, w = smart_resize(
            t=TEMPORAL_PATCH_SIZE,
            h=100,
            w=100,
            t_factor=TEMPORAL_PATCH_SIZE,
            h_factor=factor,
            w_factor=factor,
            min_pixels=1,
            max_pixels=10**9,
        )
        assert h % factor == 0 and w % factor == 0
        assert (h, w) == (112, 112)  # ceil(100/28)*28

    def test_expand_factor_scales_alignment(self):
        # patch_expand_factor folds into the spatial factor.
        h, w = smart_resize(
            t=TEMPORAL_PATCH_SIZE,
            h=29,
            w=29,
            t_factor=TEMPORAL_PATCH_SIZE,
            h_factor=PATCH_SIZE * MERGE_SIZE * 2,  # expand factor 2
            w_factor=PATCH_SIZE * MERGE_SIZE * 2,
            min_pixels=1,
            max_pixels=10**9,
        )
        assert (h, w) == (56, 56)  # ceil(29/56)*56

    def test_over_budget_canvas_is_refit(self):
        factor = PATCH_SIZE * MERGE_SIZE
        max_pixels = TEMPORAL_PATCH_SIZE * factor * factor * 4  # 4 aligned patches
        h, w = smart_resize(
            t=TEMPORAL_PATCH_SIZE,
            h=2000,
            w=2000,
            t_factor=TEMPORAL_PATCH_SIZE,
            h_factor=factor,
            w_factor=factor,
            min_pixels=1,
            max_pixels=max_pixels,
        )
        assert TEMPORAL_PATCH_SIZE * h * w <= max_pixels
        # The refit keeps the largest aligned canvas that fits (up to 4
        # aligned patches for a square input: 2x2).
        assert (h, w) == (factor * 2, factor * 2)

    def test_below_budget_budget_raises(self):
        factor = PATCH_SIZE * MERGE_SIZE
        with pytest.raises(ValueError, match="too small"):
            smart_resize(
                t=TEMPORAL_PATCH_SIZE,
                h=10,
                w=10,
                t_factor=TEMPORAL_PATCH_SIZE,
                h_factor=factor,
                w_factor=factor,
                min_pixels=1,
                max_pixels=TEMPORAL_PATCH_SIZE * factor * factor - 1,
            )

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="must be positive"):
            smart_resize(t=0, h=10, w=10)
        with pytest.raises(ValueError, match="less than or equal to max_pixels"):
            smart_resize(t=1, h=10, w=10, min_pixels=100, max_pixels=50)


class TestVisionInfo:
    def test_grid_math_matches_token_formula(self):
        info = _make_processing_info()
        # 112x112 image -> 8x8 grid of 14px patches -> 16 merged tokens.
        size, num_tokens = Glm5NextProcessingInfo._get_vision_info(
            info,
            image_width=112,
            image_height=112,
            num_frames=1,
            max_image_pixels=8000 * PIXELS_PER_TOKEN,
        )
        assert size.width == 112 and size.height == 112
        grid_t = 1  # padded 2 frames / temporal 2
        assert num_tokens == (grid_t * 8 * 8) // MERGE_SIZE**2 == 16

    def test_temporal_frames_round_up(self):
        info = _make_processing_info()
        _, tokens_16 = Glm5NextProcessingInfo._get_vision_info(
            info,
            image_width=112,
            image_height=112,
            num_frames=16,
            max_image_pixels=8000 * PIXELS_PER_TOKEN,
        )
        _, tokens_17 = Glm5NextProcessingInfo._get_vision_info(
            info,
            image_width=112,
            image_height=112,
            num_frames=17,
            max_image_pixels=8000 * PIXELS_PER_TOKEN,
        )
        # 16 frames -> grid_t 8; 17 pads to 18 -> grid_t 9.
        assert tokens_17 - tokens_16 == (8 * 8) // MERGE_SIZE**2

    def test_max_pixels_floor_keeps_one_aligned_canvas(self):
        info = _make_processing_info()
        factor = PATCH_SIZE * MERGE_SIZE
        # Budget below one aligned canvas is floored up, not erroring.
        _, num_tokens = Glm5NextProcessingInfo._get_vision_info(
            info,
            image_width=56,
            image_height=56,
            num_frames=1,
            max_image_pixels=1,
        )
        expected = ((factor // PATCH_SIZE) ** 2) // MERGE_SIZE**2
        assert num_tokens == expected == 1


class TestFrameSampling:
    def test_even_count_and_valid_indices(self):
        indices = glm_sample_frame_indices(
            total_frames=100, fps=30.0, duration=3.4, target_fps=2.0
        )
        assert len(indices) % 2 == 0
        assert all(0 <= i < 100 for i in indices)
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))

    def test_max_frame_count_caps_extraction(self):
        capped = glm_sample_frame_indices(
            total_frames=10000,
            fps=30.0,
            duration=333.0,
            target_fps=60.0,
            max_frame_count=8,
        )
        assert len(capped) <= 8

    def test_defaults_match_checkpoint_knobs(self):
        assert GLM_VIDEO_DEFAULT_FPS == 2.0
        assert GLM_VIDEO_DEFAULT_MAX_FRAMES == 2048

    def test_short_clip_spreads_frames(self):
        # Fewer frames than extract_t: floor-sampled spread, duplicated last
        # frame to keep the count even.
        indices = glm_sample_frame_indices(
            total_frames=3, fps=30.0, duration=0.1, target_fps=2.0
        )
        assert len(indices) % 2 == 0
        assert all(0 <= i < 3 for i in indices)


class TestMultiModalProcessorContract:
    def test_hf_processor_does_not_apply_updates(self):
        """vLLM owns prompt expansion (image token repeat / video frame and
        timestamp structure); the HF processor leaves the prompt unchanged."""
        processor = Glm5NextMultiModalProcessor.__new__(Glm5NextMultiModalProcessor)
        assert (
            processor._hf_processor_applies_updates(
                prompt_text="x",
                mm_items=Mock(spec=MultiModalDataItems),
                hf_processor_mm_kwargs={},
                tokenization_kwargs={},
            )
            is False
        )

    def test_video_token_cap_constant(self):
        """from_pretrained caps only the video budget at _MAX_VIDEO_TOKENS;
        the image budget follows the checkpoint verbatim (8000)."""
        assert _MAX_VIDEO_TOKENS == 30000
        info = _make_processing_info(max_video_tokens=240000)
        _, max_video_px = Glm5NextProcessingInfo._processor_pixel_budget(
            info, info.get_hf_processor().video_processor
        )
        _, max_image_px = Glm5NextProcessingInfo._processor_pixel_budget(
            info, info.get_hf_processor().image_processor
        )
        assert max_video_px % PIXELS_PER_TOKEN == 0
        assert max_image_px % PIXELS_PER_TOKEN == 0


class TestPromptExpansionFormula:
    """The inherited Glm4v prompt updates replace each ``<|image|>`` with
    ``prod(image_grid_thw) // merge_size**2`` copies of the image token id.
    Pin that the processor's expansion equals what
    ``Glm5NextProcessingInfo._get_vision_info`` budgets — they must agree or
    placeholder/scatter lengths diverge at runtime."""

    def test_image_budget_matches_expansion(self):
        info = _make_processing_info()
        _, budget_tokens = Glm5NextProcessingInfo._get_vision_info(
            info,
            image_width=140,
            image_height=196,
            num_frames=1,
            max_image_pixels=8000 * PIXELS_PER_TOKEN,
        )
        # Expansion for the same grid: prod(grid_thw) // merge^2 where
        # grid_thw = [1, 196//14, 140//14] = [1, 14, 10].
        grid_thw = np.array([1, 14, 10])
        expanded = int(grid_thw.prod()) // MERGE_SIZE**2
        assert budget_tokens == expanded == 35
