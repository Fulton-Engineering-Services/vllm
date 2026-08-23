# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Analytic flops/memory estimation module for transformer components,
to help derive MFU (Model Flops Utilization) stats for a running model.
"""

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import prometheus_client
import torch
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing_extensions import Self

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_dtype_size,
    get_kv_cache_torch_dtype,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.utils import create_metric_per_engine

logger = init_logger(__name__)


class InvalidComponent(Exception):
    """
    Custom exception to indicate that a certain ComponentMetric is not
    applicable to the given VllmConfig.
    """

    pass


# Mapping from quantization method name to effective weight byte size.
# Used by both AttentionQuantizationConfigParser and
# FfnQuantizationConfigParser to determine the weight_byte_size for
# flops/memory estimation.
#
# NOTE: Methods like GPTQ and BitsAndBytes support variable bit-widths
# (e.g., 4-bit and 8-bit). We default to 4-bit (0.5 bytes) since this
# is by far the most common configuration.
_QUANT_WEIGHT_BYTE_SIZE: dict[str, float] = {
    # FP8 methods (1 byte per weight)
    "fp8": 1,
    "fbgemm_fp8": 1,
    "ptpc_fp8": 1,
    "fp_quant": 1,
    "modelopt": 1,
    "modelopt_mxfp8": 1,
    # FP4 / INT4 methods (0.5 bytes per weight)
    "mxfp4": 0.5,
    "awq": 0.5,
    "awq_marlin": 0.5,
    "gptq": 0.5,
    "gptq_marlin": 0.5,
    "bitsandbytes": 0.5,
    "modelopt_fp4": 0.5,
    "petit_nvfp4": 0.5,
    "compressed-tensors": 0.5,
    "torchao": 0.5,
    "quark": 0.5,
    "moe_wna16": 0.5,
    "inc": 0.5,
    "experts_int8": 1,
    # Additional methods (verified via get_name())
    "auto_awq": 0.5,
    "auto_gptq": 0.5,
    "deepseek_v4_fp8": 1,
    "gpt_oss_mxfp4": 0.5,
}


# Mapping from modelopt_mixed quant_algo names to effective weight byte size.
# Used by _compute_modelopt_mixed_component_byte_sizes.
_MODELOPT_MIXED_ALGO_BYTE_SIZE: dict[str, float] = {
    "FP8": 1,
    "MXFP8": 1,
    "NVFP4": 0.5,
    "W4A16_NVFP4": 0.5,
    "W4A16": 0.5,
    "AWQ": 0.5,
    "GPTQ": 0.5,
    "FP4": 0.5,
    "INT4": 0.5,
}


def _get_modelopt_mixed_algo_byte_size(layer_info: object, default_byte_size: float) -> float:
    """Return the effective byte size for a single quantized_layers entry."""
    algo = getattr(layer_info, "quant_algo", None)
    if algo is None:
        algo = getattr(layer_info, "algo", None)

    if algo is None:
        logger.warning_once(
            "modelopt_mixed parser: layer missing quant_algo, using dtype size"
        )
        return default_byte_size

    byte_size = _MODELOPT_MIXED_ALGO_BYTE_SIZE.get(algo)
    if byte_size is None:
        logger.warning_once(
            "modelopt_mixed parser: unknown quant_algo '%s', using dtype size", algo
        )
        return default_byte_size

    return byte_size


def _classify_quantized_layer(layer_name: str) -> str | None:
    """Classify a quantized_layers key into a component type."""
    layer_lower = layer_name.lower()

    # Attention projections
    if any(layer_lower.endswith(suffix) for suffix in ("q", "k", "v", "qkv", "o", "o_proj")):
        return "attn"

    # Unembed
    if layer_lower == "lm_head":
        return "unembed"

    # Mamba projections / conv / ssm
    if layer_lower in ("in_proj", "out_proj", "conv1d", "x_proj", "dt_proj", "ssm") \
            or any(layer_lower.startswith(prefix) for prefix in ("in_proj.", "out_proj.", "conv1d.", "x_proj.", "dt_proj.", "ssm.")):
        return "mamba"

    # FFN / MoE projections
    if any(keyword in layer_lower for keyword in ("up_proj", "down_proj", "gate_proj", "experts", "shared_experts")):
        return "ffn"

    return None


def _compute_modelopt_mixed_component_byte_sizes(
    quant_config: object, default_byte_size: float
) -> dict[str, float]:
    """
    Parse a modelopt_mixed quantized_layers map and return the average
    effective weight byte size for each component (attn, ffn, mamba, unembed).

    Missing or unknown quant_algo values fall back to ``default_byte_size``.
    If ``quantized_layers`` is empty, falls back to the coarse
    ``_QUANT_WEIGHT_BYTE_SIZE["modelopt_mixed"]`` entry for every component.
    """
    quantized_layers = getattr(quant_config, "quantized_layers", None) or {}

    if not quantized_layers:
        coarse = _QUANT_WEIGHT_BYTE_SIZE.get("modelopt_mixed", default_byte_size)
        return {"attn": coarse, "ffn": coarse, "mamba": coarse, "unembed": coarse}

    component_entries: dict[str, list[float]] = {
        "attn": [],
        "ffn": [],
        "mamba": [],
        "unembed": [],
    }

    for layer_name, layer_info in quantized_layers.items():
        component = _classify_quantized_layer(layer_name)
        if component is None:
            logger.warning_once(
                "modelopt_mixed parser: unknown layer '%s', categorizing as ffn", layer_name
            )
            component = "ffn"

        byte_size = _get_modelopt_mixed_algo_byte_size(layer_info, default_byte_size)
        component_entries[component].append(byte_size)

    component_sizes: dict[str, float] = {}
    for component, entries in component_entries.items():
        if entries:
            component_sizes[component] = sum(entries) / len(entries)
        else:
            component_sizes[component] = default_byte_size

    return component_sizes


#### Basic Data Types ####


@dataclass
class DebugPerfStats:
    ## Stats for debugging the metrics calculation
    calc_duration: float = 0.0  # time spent calculating these stats
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    context_breakdown: dict[str, int] | None = None
    num_flops_per_gpu_breakdown: dict[str, int] | None = None
    num_read_bytes_per_gpu_breakdown: dict[str, int] | None = None
    num_write_bytes_per_gpu_breakdown: dict[str, int] | None = None


@dataclass
class PerfStats:
    num_flops_per_gpu: int = 0
    num_read_bytes_per_gpu: int = 0
    num_write_bytes_per_gpu: int = 0
    debug_stats: DebugPerfStats | None = None


@dataclass
class ExecutionContext:
    """
    Represents an execution context for a batch of requests.

    This class aggregates statistics across multiple requests in a batch,
    separately tracking prefill and decode phases.

    Example)
    - Batch with one full prefill (2048 tokens) and one decode (1 token, 8192 context):
      ctx = ExecutionContext()
      ctx.add(2048, 2048, is_prefill=True)
      ctx.add(1, 8192, is_prefill=False)
    """

    # Prefill phase statistics
    num_prefill_requests: int = 0
    prefill_num_tokens: int = 0  # sum of num_tokens for prefill requests
    prefill_context_len: int = 0  # sum of context_len for prefill requests
    prefill_token_context_product: int = 0  # sum of (num_tokens * context_len)

    # Decode phase statistics
    num_decode_requests: int = 0
    decode_num_tokens: int = 0  # sum of num_tokens for decode requests
    decode_context_len: int = 0  # sum of context_len for decode requests
    decode_token_context_product: int = 0  # sum of (num_tokens * context_len)

    def add(self, num_tokens: int, context_len: int, is_prefill: bool) -> None:
        """Add a single request's statistics to this batch context."""
        if is_prefill:
            self.num_prefill_requests += 1
            self.prefill_num_tokens += num_tokens
            self.prefill_context_len += context_len
            self.prefill_token_context_product += num_tokens * context_len
        else:
            self.num_decode_requests += 1
            self.decode_num_tokens += num_tokens
            self.decode_context_len += context_len
            self.decode_token_context_product += num_tokens * context_len

    def total_num_tokens(self) -> int:
        """Total number of tokens across all requests in the batch."""
        return self.prefill_num_tokens + self.decode_num_tokens

    def total_token_context_product(self) -> int:
        """Total sum of (num_tokens * context_len) across all requests."""
        return self.prefill_token_context_product + self.decode_token_context_product

    def num_logits_tokens(self) -> int:
        """Number of tokens that require logits computation (unembedding).

        For prefill, only the last token per request needs logits.
        For decode, all tokens need logits.
        """
        return self.num_prefill_requests + self.decode_num_tokens

    @classmethod
    def from_single_request(
        cls, num_tokens: int, context_len: int, is_prefill: bool
    ) -> "ExecutionContext":
        """Create an ExecutionContext from a single request.

        This is a convenience method primarily for testing.
        """
        ctx = cls()
        ctx.add(num_tokens, context_len, is_prefill)
        return ctx


class ParsedArgs:
    """
    Syntactic sugar so that Parsers can use dot notations
    to access/update the parsed arguments.

    e.g.)
        args = ParsedArgs()
        args.x = 3
        args.y = args.x + 1
    """

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)

    def model_dump(self) -> dict[str, Any]:
        return vars(self).copy()


#### Abstract ####


class Parser(Protocol):
    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        """
        Parse the vllm config and update the current ParsedArgs and pass it on.
        If the parser isn't applicable to the vllm_config, it will do nothing.
        """
        ...


class ParserChain:
    """
    Applies chain of parser in a sequential order.
    Later parsers might overwrite results from previous parsers,
    so parsers should be chained in the appropriate order if they
    are not mutually exclusive.
    """

    def __init__(self, *parsers: Parser) -> None:
        self.parsers = list(parsers)

    def add_parser(self, parser: Parser) -> None:
        self.parsers.append(parser)

    def parse(self, vllm_config: VllmConfig) -> ParsedArgs:
        args = ParsedArgs()
        for parser in self.parsers:
            args = parser.parse(args, vllm_config)
        return args


_COMPONENT_METRICS_REGISTRY: dict[str, type["ComponentMetrics"]] = {}


class ComponentMetrics(BaseModel, ABC):
    """
    Each concrete ComponentMetrics class is associated with:
    - fields that are required for metric derivation
      (fields are specified/validated through pydantic model)
    - parser to parse VllmConfig into fields
    - metric methods that derive flops/bytes for a given execution context
    """

    @classmethod
    @abstractmethod
    def component_type(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_parser(cls) -> ParserChain:
        """
        Return a ParserChain that provides values for all required fields.
        The returned parser chain must populate ParsedArgs with values for every
        field defined on this ComponentMetrics class. Missing fields will cause
        a ValidationError when from_vllm_config() is called.
        See individual Parser docstrings for which args they provide, and field
        comments on ComponentMetrics subclasses for which parser provides each field.
        """
        ...

    def __init_subclass__(cls):
        _COMPONENT_METRICS_REGISTRY[cls.component_type()] = cls

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig) -> Self:
        """
        Instantiate this class from VllmConfig.
        Raises ValidationError if parsing fails.
        """

        parser = cls.get_parser()
        parsed_args = parser.parse(vllm_config)
        try:
            return cls.model_validate(parsed_args.model_dump())
        except ValidationError as e:
            raise InvalidComponent(f"Invalid {cls.component_type()} config: {e}") from e

    @classmethod
    def registered_metrics(cls) -> Iterable[type["ComponentMetrics"]]:
        return iter(_COMPONENT_METRICS_REGISTRY.values())

    @abstractmethod
    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    @abstractmethod
    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]: ...

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_num_flops_breakdown(ctx, per_gpu).values())

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_read_bytes_breakdown(ctx, per_gpu).values())

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(self.get_write_bytes_breakdown(ctx, per_gpu).values())


#### parsers ####

class BaseConfigParser(Parser):
    """
    Parses base model configuration.
    Provides: vocab_size, hidden_size, num_attention_heads, num_hidden_layers,
    weight_byte_size, activation_byte_size, dp_size, tp_size, pp_size, enable_ep,
    and optionally num_attention_layers from layers_block_type.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.vocab_size = model_config.get_vocab_size()
        args.hidden_size = model_config.get_hidden_size()
        # NOTE: model_config.get_attention_heads() divide by TP
        # so we access field manually here to get total num_heads
        args.num_attention_heads = get_required(
            model_config.hf_text_config, "num_attention_heads"
        )

        # Determine number of layers from layers_block_type if available,
        # otherwise fall back to num_hidden_layers
        hf_config = model_config.hf_text_config
        if hasattr(hf_config, "layers_block_type"):
            block_types = hf_config.layers_block_type
            if isinstance(block_types, str):
                block_types = [block_types]
            # Count entries by type
            attn_count = sum(1 for bt in block_types if bt == "attention")
            moe_count = sum(1 for bt in block_types if bt == "moe")
            mamba_count = sum(1 for bt in block_types if bt == "mamba")
            mlp_count = sum(1 for bt in block_types if bt == "mlp")

            args.num_hidden_layers = len(block_types)  # total
            args.num_attention_layers = attn_count
            # Layer-type-aware counts for FfnMetrics (-1 sentinel means "not from block type")
            args.num_moe_layers_from_block_type = moe_count
            args.num_dense_ffn_layers_from_block_type = mlp_count
        else:
            args.num_hidden_layers = get_required(
                model_config.hf_text_config, "num_hidden_layers"
            )
            args.num_attention_layers = None  # will use num_hidden_layers fallback
            args.num_moe_layers_from_block_type = -1
            args.num_dense_ffn_layers_from_block_type = -1

        model_dtype = vllm_config.model_config.dtype

        if isinstance(model_dtype, torch.dtype):
            torch_dtype = model_dtype
        elif isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
        else:
            # FIXME: handle this better
            logger.warning(
                "Unknown model_dtype %s, defaulting to bfloat16",
                model_dtype,
            )
            torch_dtype = torch.bfloat16

        args.weight_byte_size = get_dtype_size(torch_dtype)

        # FIXME: handle this better by parsing whether activations use
        # bf16, fp32, etc...
        args.activation_byte_size = 2

        args.dp_size = vllm_config.parallel_config.data_parallel_size
        args.tp_size = vllm_config.parallel_config.tensor_parallel_size
        args.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        args.enable_ep = vllm_config.parallel_config.enable_expert_parallel

        return args


#### Attention ####


class BaseAttentionConfigParser(Parser):
    """
    Parses attention-specific configuration.
    Provides: num_key_value_heads, head_dim, cache_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config

        args.num_key_value_heads = model_config.get_total_num_kv_heads()
        args.head_dim = model_config.get_head_size()

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype

        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class AttentionQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for attention layers.
    Overrides: weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()

        if quant_method == "modelopt_mixed":
            args = self._parse_modelopt_mixed_attn(cfg, args)
        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for attention metrics: {quant_method}"
            )

        return args

    def _parse_modelopt_mixed_attn(self, cfg, args: ParsedArgs) -> ParsedArgs:
        """Parse per-component weight byte size from quantized_layers for modelopt_mixed."""
        component_sizes = _compute_modelopt_mixed_component_byte_sizes(
            cfg, args.weight_byte_size
        )
        args.weight_byte_size = component_sizes.get("attn", args.weight_byte_size)
        return args


class UnembedQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for unembed (lm_head) layers.
    Overrides: weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()
        if quant_method == "modelopt_mixed":
            component_sizes = _compute_modelopt_mixed_component_byte_sizes(
                cfg, args.weight_byte_size
            )
            args.weight_byte_size = component_sizes.get("unembed", args.weight_byte_size)
        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            # Fall back to dtype-based byte size
            model_dtype = vllm_config.model_config.dtype
            if isinstance(model_dtype, torch.dtype):
                args.weight_byte_size = get_dtype_size(model_dtype)
            else:
                args.weight_byte_size = 2  # default bfloat16
        return args


class AttentionDetectionParser(Parser):
    """
    Prevents standard AttentionMetrics from being instantiated for MLA models.
    MLA models should use MLAAttentionMetrics instead.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        if vllm_config.model_config.is_deepseek_mla:
            raise InvalidComponent(
                "Model uses MLA attention; use MLAAttentionMetrics instead"
            )
        return args


class AttentionMetrics(ComponentMetrics):
    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From BaseAttentionConfigParser
    num_key_value_heads: int = Field(..., gt=0)
    head_dim: int = Field(..., gt=0)
    cache_byte_size: int = Field(..., gt=0)

    # From BaseConfig Parser, overridden by AttentionQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    # TODO: discern cases where we have mixture of different attention layer types
    # such as SWA, MLA, etc.

    @classmethod
    def component_type(cls) -> str:
        return "attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            AttentionDetectionParser(),
            BaseConfigParser(),
            BaseAttentionConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        L, D, q, kv, d = (
            self.num_attention_layers if self.num_attention_layers is not None else self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_proj": 2 * T * D * (q + 2 * kv) * d * L,
            "attn_qk": 2 * q * TC * d * L,
            "attn_av": 2 * q * TC * d * L,
            "out_proj": 2 * T * D * q * d * L,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        L, D, q, kv, d = (
            self.num_attention_layers if self.num_attention_layers is not None else self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        read_bytes = {}

        read_bytes["qkv_input"] = T * D * self.activation_byte_size * L
        read_bytes["qkv_weight"] = int(D * (q + 2 * kv) * d * self.weight_byte_size * L)

        # Attention input reads differ between prefill and decode
        # Prefill: read Q, K, V activations (all in activation_byte_size)
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                (ctx.prefill_num_tokens * q + 2 * ctx.prefill_context_len * kv)
                * d
                * self.activation_byte_size
                * L
            )

        # Decode: read Q activations + read K, V from cache (in cache_byte_size)
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * d * self.activation_byte_size * L
                + 2 * ctx.decode_context_len * kv * d * self.cache_byte_size * L
            )

        read_bytes["out_input"] = T * q * d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for attention layers."""
        L, D, q, kv, d = (
            self.num_attention_layers if self.num_attention_layers is not None else self.num_hidden_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size
            # tensor parallel along heads
            q = max(1, q // self.tp_size)
            kv = max(1, kv // self.tp_size)

        return {
            "qkv_output": T * (q + 2 * kv) * d * self.activation_byte_size * L,
            "kv_cache": 2 * T * kv * d * self.cache_byte_size * L,
            "out_output": T * D * self.activation_byte_size * L,
        }


#### MLA Attention ####


class MLADetectionParser(Parser):
    """
    Validates that the model uses MLA attention.
    Raises InvalidComponent if the model does not use MLA,
    so MLAAttentionMetrics is silently skipped for non-MLA models.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        if not vllm_config.model_config.is_deepseek_mla:
            raise InvalidComponent("Model does not use MLA attention")
        return args


class MLAConfigParser(Parser):
    """
    Parses MLA-specific configuration fields.
    Provides: kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, q_lora_rank
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        model_config = vllm_config.model_config
        cfg = model_config.hf_text_config

        args.kv_lora_rank = get_required(cfg, "kv_lora_rank")
        args.qk_nope_head_dim = get_required(cfg, "qk_nope_head_dim")
        args.qk_rope_head_dim = get_required(cfg, "qk_rope_head_dim")
        args.v_head_dim = get_required(cfg, "v_head_dim")
        args.q_lora_rank = getattr(cfg, "q_lora_rank", None)

        model_dtype = vllm_config.model_config.dtype
        cache_dtype = vllm_config.cache_config.cache_dtype
        kv_cache_torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
        args.cache_byte_size = get_dtype_size(kv_cache_torch_dtype)

        return args


class MLAAttentionMetrics(ComponentMetrics):
    """
    Performance metrics for Multi-Latent Attention (MLA) layers.

    MLA uses a compressed latent representation for KV cache:
    - KV cache stores a single compressed vector of size
      (kv_lora_rank + qk_rope_head_dim) per token per layer,
      instead of 2 * num_kv_heads * head_dim as in standard MHA/GQA.
    - Q path uses optional low-rank compression:
      h -> q_lora_rank -> num_heads * qk_head_dim
    - KV path: h -> (kv_lora_rank + qk_rope_head_dim),
      then kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)

    Used by DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, and similar models.
    """

    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    num_attention_heads: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From BaseConfigParser, can be overridden by AttentionQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    # From MLAConfigParser
    kv_lora_rank: int = Field(..., gt=0)
    qk_nope_head_dim: int = Field(..., gt=0)
    qk_rope_head_dim: int = Field(..., gt=0)
    v_head_dim: int = Field(..., gt=0)
    q_lora_rank: int | None = Field(None)
    cache_byte_size: int = Field(..., gt=0)

    @classmethod
    def component_type(cls) -> str:
        return "mla_attn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            MLADetectionParser(),
            BaseConfigParser(),
            MLAConfigParser(),
            AttentionQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for MLA attention layers.

        MLA projection structure:
        - Q path: h -> q_lora_rank -> num_heads * qk_head_dim
          (or h -> num_heads * qk_head_dim if q_lora_rank is None)
        - KV path: h -> (kv_lora_rank + qk_rope_head_dim),
          then kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        - Attention: Q @ K^T and attn @ V
        - Output: num_heads * v_head_dim -> h
        """
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        TC = ctx.total_token_context_product()

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        flops: dict[str, int] = {}

        # Q projection
        if q_rank is not None:
            # Two-stage: h -> q_lora_rank -> num_heads * qk_head_dim
            flops["q_a_proj"] = 2 * T * D * q_rank * L
            flops["q_b_proj"] = 2 * T * q_rank * q * qk_head_dim * L
        else:
            # Direct: h -> num_heads * qk_head_dim
            flops["q_proj"] = 2 * T * D * q * qk_head_dim * L

        # KV projection (always compressed, shared across heads)
        # kv_a: h -> (kv_lora_rank + qk_rope_head_dim)  [replicated]
        flops["kv_a_proj"] = 2 * T * D * (c + r) * L
        # kv_b: kv_lora_rank -> num_heads * (qk_nope + v_head_dim)
        flops["kv_b_proj"] = 2 * T * c * q * (self.qk_nope_head_dim + v_d) * L

        # Attention core
        flops["attn_qk"] = 2 * q * TC * qk_head_dim * L
        flops["attn_av"] = 2 * q * TC * v_d * L

        # Output projection: num_heads * v_head_dim -> h
        flops["out_proj"] = 2 * T * q * v_d * D * L

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for MLA attention layers."""
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        # Compressed KV cache size per token
        kv_compressed_dim = c + r

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        read_bytes: dict[str, int] = {}

        # Q projection weight + input reads
        if q_rank is not None:
            read_bytes["q_a_input"] = T * D * self.activation_byte_size * L
            read_bytes["q_a_weight"] = int(D * q_rank * self.weight_byte_size * L)
            read_bytes["q_b_input"] = T * q_rank * self.activation_byte_size * L
            read_bytes["q_b_weight"] = int(
                q_rank * q * qk_head_dim * self.weight_byte_size * L
            )
        else:
            read_bytes["q_input"] = T * D * self.activation_byte_size * L
            read_bytes["q_weight"] = int(
                D * q * qk_head_dim * self.weight_byte_size * L
            )

        # KV projection weight + input reads
        # kv_a is replicated (not TP-sharded)
        read_bytes["kv_a_input"] = T * D * self.activation_byte_size * L
        read_bytes["kv_a_weight"] = int(
            D * kv_compressed_dim * self.weight_byte_size * L
        )
        # kv_b is TP-sharded along heads
        read_bytes["kv_b_input"] = T * c * self.activation_byte_size * L
        read_bytes["kv_b_weight"] = int(
            c * q * (self.qk_nope_head_dim + v_d) * self.weight_byte_size * L
        )

        # Attention input reads
        # Prefill: read Q activations + K,V from kv_b_proj output
        if ctx.prefill_num_tokens > 0:
            read_bytes["attn_input"] = (
                ctx.prefill_num_tokens * q * qk_head_dim * self.activation_byte_size * L
                + ctx.prefill_context_len
                * q
                * (qk_head_dim + v_d)
                * self.activation_byte_size
                * L
            )

        # Decode: read Q activations + read compressed KV from cache
        if ctx.decode_num_tokens > 0:
            read_bytes["attn_input"] = read_bytes.get("attn_input", 0) + (
                ctx.decode_num_tokens * q * qk_head_dim * self.activation_byte_size * L
                + ctx.decode_context_len * kv_compressed_dim * self.cache_byte_size * L
            )

        # Output projection reads
        read_bytes["out_input"] = T * q * v_d * self.activation_byte_size * L
        read_bytes["out_weight"] = int(q * v_d * D * self.weight_byte_size * L)

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for MLA attention layers."""
        L = self.num_hidden_layers
        D = self.hidden_size
        q = self.num_attention_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        v_d = self.v_head_dim
        c = self.kv_lora_rank
        r = self.qk_rope_head_dim
        q_rank = self.q_lora_rank

        T = ctx.total_num_tokens()
        kv_compressed_dim = c + r

        if per_gpu:
            L //= self.pp_size
            q = max(1, q // self.tp_size)

        write_bytes: dict[str, int] = {}

        # Q projection outputs
        if q_rank is not None:
            write_bytes["q_a_output"] = T * q_rank * self.activation_byte_size * L
            write_bytes["q_b_output"] = (
                T * q * qk_head_dim * self.activation_byte_size * L
            )
        else:
            write_bytes["q_output"] = (
                T * q * qk_head_dim * self.activation_byte_size * L
            )

        # KV projection outputs
        write_bytes["kv_a_output"] = (
            T * kv_compressed_dim * self.activation_byte_size * L
        )
        write_bytes["kv_b_output"] = (
            T * q * (self.qk_nope_head_dim + v_d) * self.activation_byte_size * L
        )

        # KV cache write: one compressed vector per token
        # (kv_lora_rank + qk_rope_head_dim) instead of
        # 2 * num_kv_heads * head_dim in standard MHA
        write_bytes["kv_cache"] = T * kv_compressed_dim * self.cache_byte_size * L

        # Output projection
        write_bytes["out_output"] = T * D * self.activation_byte_size * L

        return write_bytes


#### Ffn ####


class BaseFfnConfigParser(Parser):
    """
    Parses FFN and MoE configuration.
    Provides: intermediate_size, num_experts, num_experts_per_tok,
    moe_intermediate_size, num_shared_experts, num_moe_layers, mlp_hidden_act
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        args.intermediate_size = getattr(cfg, "intermediate_size", args.hidden_size * 4)

        # Try different naming conventions.
        args.num_experts = vllm_config.model_config.get_num_experts()
        args.num_experts_per_tok = getattr_from_list(
            cfg, ["num_experts_per_tok", "moe_topk"], 0
        )
        args.moe_intermediate_size = getattr_from_list(
            cfg, ["moe_intermediate_size", "intermediate_size"], 0
        )
        args.num_shared_experts = getattr_from_list(
            cfg, ["n_shared_experts", "num_shared_experts"], 0
        )

        # NemotronH non-gated MoE signal: mlp_hidden_act == "relu2"
        args.mlp_hidden_act = getattr(cfg, "mlp_hidden_act", None)

        is_moe = args.num_experts != 0
        # Assume all MoE layers by default
        args.num_moe_layers = args.num_hidden_layers if is_moe else 0

        return args


class FfnParallelParser(Parser):
    """
    Parses FFN parallelism configuration.

    Provides: ffn_tp_size, ffn_ep_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        # NOTE: ffn tp_size does not equal the tp_size parameter directly.
        # e.g.) If we use DP2TP4, ffn will use TP8 (or EP8 if EP is enabled.)
        if args.enable_ep:
            ffn_tp_size, ffn_ep_size = 1, args.dp_size * args.tp_size
        else:
            ffn_tp_size, ffn_ep_size = args.dp_size * args.tp_size, 1

        args.ffn_tp_size = ffn_tp_size
        args.ffn_ep_size = ffn_ep_size

        return args


class InterleaveMoeLayerStepParser(Parser):
    """
    Parses interleave_moe_layer_step field for models like Llama4.

    Overrides: num_moe_layers
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if (
            hasattr(cfg, "interleave_moe_layer_step")
            and cfg.interleave_moe_layer_step > 0
        ):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if (layer + 1) % cfg.interleave_moe_layer_step == 0
                ]
            )

        return args


class MoeLayerFreqParser(Parser):
    """
    Parses moe_layer_freq and first_k_dense_replace fields for models like Deepseek.

    Overrides: num_moe_layers
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        if hasattr(cfg, "moe_layer_freq") and hasattr(cfg, "first_k_dense_replace"):
            args.num_moe_layers = len(
                [
                    layer
                    for layer in range(args.num_hidden_layers)
                    if layer >= cfg.first_k_dense_replace
                    and layer % cfg.moe_layer_freq == 0
                ]
            )

        return args


class FfnQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for FFN layers.

    Overrides: weight_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()

        if quant_method == "modelopt_mixed":
            args = self._parse_modelopt_mixed_ffn(cfg, args)
        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]
        else:
            raise InvalidComponent(
                f"Unsupported quantization method for FFN metrics: {quant_method}"
            )

        return args

    def _parse_modelopt_mixed_ffn(self, cfg, args: ParsedArgs) -> ParsedArgs:
        """Parse per-component weight byte size from quantized_layers for modelopt_mixed."""
        component_sizes = _compute_modelopt_mixed_component_byte_sizes(
            cfg, args.weight_byte_size
        )
        args.weight_byte_size = component_sizes.get("ffn", args.weight_byte_size)
        return args


class FfnMetrics(ComponentMetrics):
    # From BaseConfigParser
    num_hidden_layers: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)

    # From FfnParallelParser
    ffn_tp_size: int = Field(..., gt=0)
    ffn_ep_size: int = Field(..., gt=0)

    # From BaseFfnConfigParser
    intermediate_size: int = Field(..., gt=0)
    num_experts: int = Field(0)
    num_experts_per_tok: int = Field(1)
    moe_intermediate_size: int = Field(0)
    num_shared_experts: int = Field(0)

    # From BaseConfigParser, can be overridden InterleaveMoeLayerStep or MoeLayerFreq
    num_moe_layers: int = Field(..., ge=0)

    # From layers_block_type (set by BaseConfigParser when available; -1 means absent)
    num_moe_layers_from_block_type: int = Field(-1, ge=-1)  # count of "moe" entries
    num_dense_ffn_layers_from_block_type: int = Field(-1, ge=-1)  # count of "mlp" entries

    # From BaseFfnConfigParser
    mlp_hidden_act: str | None = Field(None)

    # FIXME: might have to make this more granular
    # (i.e. dense_weight_byte_size, moe_routed_weight_byte_size,
    # moe_shared_weight_byte_size)
    # since it can differ from byte size of other components (e.g. attn)
    # and can differ even from each other.

    # From BaseConfigParser, can be overridden by FfnQuantizationConfigParser
    weight_byte_size: int | float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_moe_fields(self) -> Self:
        """Validate that MoE-related fields are properly set when num_moe_layers > 0."""
        if self.num_moe_layers > 0:
            assert self.num_experts, f"{self.num_experts=}"
            assert self.num_experts_per_tok, f"{self.num_experts_per_tok=}"
            assert self.moe_intermediate_size, f"{self.moe_intermediate_size=}"
        return self

    @classmethod
    def component_type(cls) -> str:
        return "ffn"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            FfnParallelParser(),
            BaseFfnConfigParser(),
            InterleaveMoeLayerStepParser(),
            MoeLayerFreqParser(),
            FfnQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for FFN layers."""
        # Use layer-type counts from blocks if available, otherwise fall back
        if self.num_moe_layers_from_block_type >= 0:
            Lm = self.num_moe_layers_from_block_type
            Ld = self.num_dense_ffn_layers_from_block_type
        else:
            Lm = self.num_moe_layers
            Ld = self.num_hidden_layers - self.num_moe_layers

        D, DI = self.hidden_size, self.intermediate_size
        E, MI, S = (
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        flops = {}

        # Dense FFN layers (SwiGLU: up, gate, down projections)
        # For non-gated MoE (mlp_hidden_act == "relu2"), only 2 GEMMs: up + down
        num_gemms = 2 if self.mlp_hidden_act == "relu2" else 3

        if Ld:
            flops["dense_ffn"] = 2 * num_gemms * D * DI * T * Ld

        # MoE routed experts (each token activates E experts)
        if Lm and E:
            flops["routed_ffn"] = 2 * num_gemms * D * MI * num_activated_tokens * Lm

        # MoE shared experts (all S shared experts run for every token)
        if Lm and S:
            flops["shared_ffn"] = 2 * num_gemms * D * MI * S * T * Lm

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for FFN layers."""
        # Use layer-type counts from blocks if available, otherwise fall back
        if self.num_moe_layers_from_block_type >= 0:
            Lm = self.num_moe_layers_from_block_type
            Ld = self.num_dense_ffn_layers_from_block_type
        else:
            Lm = self.num_moe_layers
            Ld = self.num_hidden_layers - self.num_moe_layers

        D, DI = self.hidden_size, self.intermediate_size
        E, MI, S = (
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()
        num_experts = self.num_experts

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size
            if num_experts is not None:
                num_experts //= self.ffn_ep_size

        read_bytes = {}

        # Dense FFN layers (SwiGLU: up, gate, down projections + SiLU activation)
        # For non-gated MoE (mlp_hidden_act == "relu2"), only 2 GEMMs: up + down
        num_gemms = 2 if self.mlp_hidden_act == "relu2" else 3

        if Ld:
            read_bytes["dense_up_gate_input"] = int(
                T * D * self.activation_byte_size * Ld
            )
            read_bytes["dense_up_gate_weights"] = int(
                num_gemms * D * DI * self.weight_byte_size * Ld
            )
            read_bytes["dense_silu_input"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_input"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            read_bytes["dense_down_weights"] = int(
                num_gemms * D * DI * self.weight_byte_size * Ld
            )

        if Lm:
            # MoE routed expert reads
            if E:
                # FIXME: Assume perfect load balancing for now.
                num_activated_experts = min(num_activated_tokens, num_experts)

                read_bytes["routed_up_gate_input"] = int(
                    num_activated_tokens * D * self.activation_byte_size * Lm
                )
                read_bytes["routed_up_gate_weights"] = int(
                    num_gemms * D * MI * num_activated_experts * self.weight_byte_size * Lm
                )
                read_bytes["routed_silu_input"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_input"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                read_bytes["routed_down_weights"] = int(
                    num_gemms * D * MI * num_activated_experts * self.weight_byte_size * Lm
                )

            # MoE shared expert reads
            if S:
                read_bytes["shared_up_gate_input"] = int(
                    T * D * self.activation_byte_size * Lm
                )
                read_bytes["shared_up_gate_weights"] = int(
                    num_gemms * D * MI * S * self.weight_byte_size * Lm
                )
                read_bytes["shared_silu_input"] = int(
                    2 * T * MI * S * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_input"] = int(
                    T * MI * self.activation_byte_size * Lm
                )
                read_bytes["shared_down_weights"] = int(
                    num_gemms * D * MI * S * self.weight_byte_size * Lm
                )

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for FFN layers."""
        # Use layer-type counts from blocks if available, otherwise fall back
        if self.num_moe_layers_from_block_type >= 0:
            Lm = self.num_moe_layers_from_block_type
            Ld = self.num_dense_ffn_layers_from_block_type
        else:
            Lm = self.num_moe_layers
            Ld = self.num_hidden_layers - self.num_moe_layers

        D, DI = self.hidden_size, self.intermediate_size
        E, MI, S = (
            self.num_experts_per_tok,
            self.moe_intermediate_size,
            self.num_shared_experts,
        )
        T = ctx.total_num_tokens()

        num_activated_tokens = T * E if E else 0

        if per_gpu:
            Ld //= self.pp_size
            Lm //= self.pp_size

            DI //= self.ffn_tp_size
            if MI is not None:
                MI //= self.ffn_tp_size
            if E:
                num_activated_tokens //= self.ffn_ep_size

        write_bytes = {}

        # Dense FFN layers
        num_gemms = 2 if self.mlp_hidden_act == "relu2" else 3

        if Ld:
            write_bytes["dense_up_gate_output"] = int(
                2 * T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_silu_output"] = int(
                T * DI * self.activation_byte_size * Ld
            )
            write_bytes["dense_down_output"] = int(
                num_gemms * T * D * self.activation_byte_size * Ld
            )

        if Lm:
            # MoE outputs
            if E:
                write_bytes["routed_up_gate_output"] = int(
                    2 * num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_silu_output"] = int(
                    num_activated_tokens * MI * self.activation_byte_size * Lm
                )
                write_bytes["routed_down_output"] = int(
                    num_gemms * num_activated_tokens * D * self.activation_byte_size * Lm
                )
            if S:
                write_bytes["shared_up_gate_output"] = int(
                    2 * T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_silu_output"] = int(
                    T * S * MI * self.activation_byte_size * Lm
                )
                write_bytes["shared_down_output"] = int(
                    num_gemms * T * S * D * self.activation_byte_size * Lm
                )

        return write_bytes


#### Unembed ####


class UnembedMetrics(ComponentMetrics):
    # From BaseConfigParser
    hidden_size: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    weight_byte_size: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)

    tp_size: int

    @classmethod
    def component_type(cls) -> str:
        return "unembed"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            UnembedQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for unembedding layer."""
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "unembed": 2 * T * D * V,
        }

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for unembedding layer."""
        D, V = self.hidden_size, self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "input": T * D * self.activation_byte_size,
            "weight": D * V * self.weight_byte_size,
        }

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for unembedding layer."""
        V = self.vocab_size
        T = ctx.num_logits_tokens()

        if per_gpu:
            V //= self.tp_size

        return {
            "output": T * V * self.activation_byte_size,
        }


#### MambaMetrics ####


class MambaMetrics(ComponentMetrics):
    """
    Performance metrics for Mamba/GDN layers.

    Mamba (Gated Linear Activation) layer FLOPs and memory:
    - in_proj: projects input to [intermediate_size + conv_dim + num_heads]
    - conv1d: 1D convolution with kernel_size=conv_kernel
    - ssm: state update (A*dA + B⊗x + C·state_read) per token

    Uses per-component weight_byte_size from modelopt_mixed quantization,
    or dtype-based fallback.
    """

    # From BaseConfigParser
    hidden_size: int = Field(..., gt=0)
    intermediate_size: int = Field(..., gt=0)
    ssm_state_size: int = Field(..., gt=0)
    ssm_state_byte_size: int | float = Field(..., gt=0)
    n_groups: int = Field(..., gt=0)
    conv_kernel: int = Field(..., gt=0)
    num_mamba_layers: int = Field(..., gt=0)
    activation_byte_size: int = Field(..., gt=0)
    tp_size: int = Field(..., gt=0)
    pp_size: int = Field(..., gt=0)
    weight_byte_size: int | float = Field(..., gt=0)

    # From Mamba mixer configuration
    num_heads: int = Field(..., gt=0)  # mamba head count (for in_proj_out calc)

    @classmethod
    def component_type(cls) -> str:
        return "mamba"

    @classmethod
    def get_parser(cls) -> ParserChain:
        return ParserChain(
            BaseConfigParser(),
            MambaDetectionParser(),
            MambaConfigParser(),
            MambaQuantizationConfigParser(),
        )

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate flops breakdown for Mamba layers."""
        L, D, I, N = (
            self.num_mamba_layers,
            self.hidden_size,
            self.intermediate_size,
            self.n_groups,
        )
        H = self.num_heads
        K = self.conv_kernel
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size

        # in_proj = 2 * T * D * in_proj_out
        # in_proj_out = I + conv_dim + H where conv_dim = I + 2*N*ssm_state_size
        conv_dim = I + 2 * N * self.ssm_state_size
        in_proj_out = I + conv_dim + H

        flops = {}

        # in_proj: input projection
        flops["mamba_in_proj"] = 2 * T * D * in_proj_out * L

        # out_proj: output projection
        flops["mamba_out_proj"] = 2 * T * I * D * L

        # conv1d
        flops["mamba_conv1d"] = 2 * T * conv_dim * K * L

        # ssm: first-order approximation (A*dA + B⊗x + C·state_read)
        # 6 * T * I * ssm_state_size per layer
        flops["mamba_ssm"] = 6 * T * I * self.ssm_state_size * L

        return flops

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate read memory traffic for Mamba layers."""
        L, D, I, N = (
            self.num_mamba_layers,
            self.hidden_size,
            self.intermediate_size,
            self.n_groups,
        )
        H = self.num_heads
        K = self.conv_kernel
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size

        conv_dim = I + 2 * N * self.ssm_state_size
        in_proj_out = I + conv_dim + H

        read_bytes = {}

        # in_proj weight + input reads
        read_bytes["mamba_in_proj_weight"] = int(
            D * in_proj_out * self.weight_byte_size * L
        )
        read_bytes["mamba_in_proj_input"] = T * D * self.activation_byte_size * L

        # out_proj weight + input reads
        read_bytes["mamba_out_proj_weight"] = int(I * D * self.weight_byte_size * L)
        read_bytes["mamba_out_proj_input"] = T * I * self.activation_byte_size * L

        # conv1d weight + input reads
        read_bytes["mamba_conv1d_weight"] = int(
            conv_dim * K * self.weight_byte_size * L
        )
        read_bytes["mamba_conv1d_input"] = T * conv_dim * self.activation_byte_size * L

        # ssm state read (per token)
        read_bytes["mamba_ssm_state_read"] = (
            T * I * self.ssm_state_size * self.ssm_state_byte_size * L
        )

        return read_bytes

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        """Calculate write memory traffic for Mamba layers."""
        L, D, I, N = (
            self.num_mamba_layers,
            self.hidden_size,
            self.intermediate_size,
            self.n_groups,
        )
        H = self.num_heads
        K = self.conv_kernel
        T = ctx.total_num_tokens()

        if per_gpu:
            L //= self.pp_size

        conv_dim = I + 2 * N * self.ssm_state_size
        in_proj_out = I + conv_dim + H

        write_bytes = {}

        # in_proj output writes
        write_bytes["mamba_in_proj_output"] = T * in_proj_out * self.activation_byte_size * L

        # out_proj output writes
        write_bytes["mamba_out_proj_output"] = T * I * self.activation_byte_size * L

        # conv1d output writes
        write_bytes["mamba_conv1d_output"] = T * conv_dim * self.activation_byte_size * L

        # ssm state write (per token)
        write_bytes["mamba_ssm_state_write"] = (
            T * I * self.ssm_state_size * self.ssm_state_byte_size * L
        )

        return write_bytes


class MambaDetectionParser(Parser):
    """
    Prevents MambaMetrics from being instantiated when there are no
    "mamba" entries in layers_block_type. Dense models skip it.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        hf_config = vllm_config.model_config.hf_text_config
        if not hasattr(hf_config, "layers_block_type"):
            raise InvalidComponent("Model has no layers_block_type; assuming dense model")

        block_types = hf_config.layers_block_type
        if isinstance(block_types, str):
            block_types = [block_types]

        if not any(bt == "mamba" for bt in block_types):
            raise InvalidComponent("Model has no mamba layers")

        return args


class MambaConfigParser(Parser):
    """
    Parses Mamba-specific configuration fields from hf_config.
    Provides: hidden_size, intermediate_size, ssm_state_size, n_groups,
    conv_kernel, num_mamba_layers, num_heads, ssm_state_byte_size
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.model_config.hf_text_config
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg = cfg.text_config

        # hidden_size from BaseConfigParser

        # intermediate_size = mamba_num_heads * mamba_head_dim
        mamba_num_heads = getattr(cfg, "mamba_num_heads", None)
        mamba_head_dim = getattr(cfg, "mamba_head_dim", None)
        if mamba_num_heads is not None and mamba_head_dim is not None:
            args.intermediate_size = mamba_num_heads * mamba_head_dim
        else:
            # Fallback to an explicit intermediate_size if available
            args.intermediate_size = getattr(
                cfg, "intermediate_size", args.hidden_size * 4
            )

        # ssm_state_size
        args.ssm_state_size = getattr(cfg, "ssm_state_size", 16)

        # ssm_state_byte_size from cache config if exposed, else fp32 default
        cache_config = vllm_config.cache_config
        if (
            cache_config is not None
            and hasattr(cache_config, "mamba_ssm_cache_dtype")
            and cache_config.mamba_ssm_cache_dtype is not None
        ):
            mamba_dtype = cache_config.mamba_ssm_cache_dtype
            if isinstance(mamba_dtype, torch.dtype):
                args.ssm_state_byte_size = get_dtype_size(mamba_dtype)
            elif isinstance(mamba_dtype, str) and mamba_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                args.ssm_state_byte_size = get_dtype_size(STR_DTYPE_TO_TORCH_DTYPE[mamba_dtype])
            else:
                args.ssm_state_byte_size = 4  # fp32 fallback
        else:
            args.ssm_state_byte_size = 4  # fp32 default

        # n_groups
        args.n_groups = getattr(cfg, "n_groups", 1)

        # conv_kernel
        args.conv_kernel = getattr(cfg, "conv_kernel", 4)

        # num_mamba_layers
        args.num_mamba_layers = getattr(cfg, "num_mamba_layers", None)
        if args.num_mamba_layers is None:
            # Fall back to num_hidden_layers if not specified
            args.num_mamba_layers = getattr(cfg, "num_hidden_layers", 0)

        # num_heads (mamba head count for in_proj_out calculation)
        args.num_heads = getattr(cfg, "num_heads", None)
        if args.num_heads is None:
            # Infer from mamba_num_heads if available
            args.num_heads = getattr(cfg, "mamba_num_heads", 1)

        return args


class MambaQuantizationConfigParser(Parser):
    """
    Parses quantization configuration for Mamba layers when using modelopt_mixed.
    Classifies quantized_layers keys by suffix and maps quant_algo → bytes.
    """

    def parse(self, args: ParsedArgs, vllm_config: VllmConfig) -> ParsedArgs:
        cfg = vllm_config.quant_config

        if cfg is None:
            return args

        quant_method = cfg.get_name()

        if quant_method == "modelopt_mixed":
            component_sizes = _compute_modelopt_mixed_component_byte_sizes(
                cfg, args.weight_byte_size
            )
            args.weight_byte_size = component_sizes.get("mamba", args.weight_byte_size)
        elif quant_method in _QUANT_WEIGHT_BYTE_SIZE:
            args.weight_byte_size = _QUANT_WEIGHT_BYTE_SIZE[quant_method]

        return args


#### ModelMetrics ####


class ModelMetrics:
    def __init__(self, vllm_config: VllmConfig) -> None:
        """
        Parse vllm_config to instantiate metrics for each component.
        is_enabled() will return False if no component metrics could be instantiated.
        """

        self.vllm_config = vllm_config

        self.metrics: list[ComponentMetrics] = []
        for metric_cls in ComponentMetrics.registered_metrics():
            try:
                metric = metric_cls.from_vllm_config(vllm_config)
                self.metrics.append(metric)
                logger.info(
                    "Instantiated ComponentMetrics [%s] with (%s)",
                    metric.component_type(),
                    str(metric),
                )
            except InvalidComponent as e:
                logger.debug(
                    "Failed to instantiate %s from %s",
                    metric_cls.component_type(),
                    str(e),
                )

    def is_enabled(self) -> bool:
        return len(self.metrics) > 0

    def get_num_flops(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_num_flops(ctx, per_gpu) for metric in self.metrics)

    def get_read_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_read_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_write_bytes(self, ctx: ExecutionContext, per_gpu: bool = True) -> int:
        return sum(metric.get_write_bytes(ctx, per_gpu) for metric in self.metrics)

    def get_num_flops_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_num_flops_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_read_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_read_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_write_bytes_breakdown(
        self, ctx: ExecutionContext, per_gpu: bool = True
    ) -> dict[str, int]:
        total = {}
        for metric in self.metrics:
            breakdown = metric.get_write_bytes_breakdown(ctx, per_gpu)
            component = metric.component_type()
            prefixed = {f"{component}.{key}": val for key, val in breakdown.items()}
            total.update(prefixed)
        return total

    def get_step_perf_stats_per_gpu(
        self, scheduler_output: SchedulerOutput
    ) -> PerfStats:
        """
        Calculate perf stats for the current step based on scheduled tokens.
        """

        t0 = time.monotonic()

        # Build a single batch context
        ctx = ExecutionContext()

        # Process new requests (these are in prefill phase)
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For new requests, context_len = num_computed_tokens + num_tokens
            # num_computed_tokens represents previously computed tokens in the sequence
            context_len = new_req.num_computed_tokens + num_tokens
            ctx.add(num_tokens, context_len, is_prefill=True)

        # Process cached requests (continuing requests)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_tokens == 0:
                continue

            # For cached requests, we have the current num_computed_tokens
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            context_len = num_computed_tokens + num_tokens

            # Cached requests are typically in decode phase (num_tokens == 1)
            # unless they're doing chunked prefill (num_tokens > 1)
            is_prefill = num_tokens > 1
            ctx.add(num_tokens, context_len, is_prefill)

        num_flops_breakdown = self.get_num_flops_breakdown(ctx, True)
        read_bytes_breakdown = self.get_read_bytes_breakdown(ctx, True)
        write_bytes_breakdown = self.get_write_bytes_breakdown(ctx, True)
        perf_stats = PerfStats(
            sum(num_flops_breakdown.values()),
            sum(read_bytes_breakdown.values()),
            sum(write_bytes_breakdown.values()),
        )

        if envs.VLLM_DEBUG_MFU_METRICS:
            perf_stats.debug_stats = DebugPerfStats(
                time.monotonic() - t0,
                ctx.num_prefill_requests,
                ctx.num_decode_requests,
                asdict(ctx),
                num_flops_breakdown,
                read_bytes_breakdown,
                write_bytes_breakdown,
            )

        return perf_stats


#### Logging ####


class PerfMetricsDebugLogging:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_calc_duration: float = 0.0
        self.total_num_prefill_requests: int = 0
        self.total_num_decode_requests: int = 0
        self.total_num_batches: int = 0
        self.total_context_breakdown: dict[str, int] = {}
        self.total_num_flops_per_gpu_breakdown: dict[str, int] = {}
        self.total_read_bytes_per_gpu_breakdown: dict[str, int] = {}
        self.total_write_bytes_per_gpu_breakdown: dict[str, int] = {}

    def observe(self, debug_stats: DebugPerfStats) -> None:
        self.total_calc_duration += debug_stats.calc_duration
        self.total_num_prefill_requests += debug_stats.num_prefill_requests
        self.total_num_decode_requests += debug_stats.num_decode_requests
        self.total_num_batches += 1

        for dst, src in zip(
            [
                self.total_context_breakdown,
                self.total_num_flops_per_gpu_breakdown,
                self.total_read_bytes_per_gpu_breakdown,
                self.total_write_bytes_per_gpu_breakdown,
            ],
            [
                debug_stats.context_breakdown,
                debug_stats.num_flops_per_gpu_breakdown,
                debug_stats.num_read_bytes_per_gpu_breakdown,
                debug_stats.num_write_bytes_per_gpu_breakdown,
            ],
        ):
            assert isinstance(src, dict)
            for key, val in src.items():
                dst[key] = dst.get(key, 0) + val

    def log(self, log_fn, log_prefix: str, delta_time: float):
        # pretty print breakdowns
        total_num_flops_per_gpu_breakdown = {
            k: f"{v / 1e12:.1f}TF"
            for k, v in self.total_num_flops_per_gpu_breakdown.items()
        }
        total_read_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_read_bytes_per_gpu_breakdown.items()
        }
        total_write_bytes_per_gpu_breakdown = {
            k: f"{v / 1e9:.1f}GB"
            for k, v in self.total_write_bytes_per_gpu_breakdown.items()
        }

        logger.debug(
            "%sMFU details: %s",
            log_prefix,
            json.dumps(
                {
                    "prefill_reqs": self.total_num_prefill_requests,
                    "decode_reqs": self.total_num_decode_requests,
                    "num_batches": self.total_num_batches,
                    "context_breakdown": self.total_context_breakdown,
                    "flops_breakdown": total_num_flops_per_gpu_breakdown,
                    "num_read_bytes_breakdown": total_read_bytes_per_gpu_breakdown,
                    "num_write_bytes_breakdown": (total_write_bytes_per_gpu_breakdown),
                    "duration": f"{delta_time:.1f}s",
                    "mfu_calc_overhead": (
                        f"{self.total_calc_duration / delta_time:.1%}"
                    ),
                },
                indent=2,
            ),
        )


class PerfMetricsLogging:
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size

        self.debug_logging: PerfMetricsDebugLogging | None = None
        if envs.VLLM_DEBUG_MFU_METRICS:
            self.debug_logging = PerfMetricsDebugLogging()

        self.reset()

    def reset(self):
        self.last_log_time = time.monotonic()

        self.total_num_flops_per_gpu: int = 0
        self.total_read_bytes_per_gpu: int = 0
        self.total_write_bytes_per_gpu: int = 0

        if self.debug_logging:
            self.debug_logging.reset()

    def observe(self, perf_stats: PerfStats) -> None:
        self.total_num_flops_per_gpu += perf_stats.num_flops_per_gpu
        self.total_read_bytes_per_gpu += perf_stats.num_read_bytes_per_gpu
        self.total_write_bytes_per_gpu += perf_stats.num_write_bytes_per_gpu

        if self.debug_logging:
            assert perf_stats.debug_stats is not None
            self.debug_logging.observe(perf_stats.debug_stats)

    def log(self, log_fn=logger.info, log_prefix: str = "") -> None:
        if not (
            self.total_num_flops_per_gpu
            or self.total_read_bytes_per_gpu
            or self.total_write_bytes_per_gpu
        ):
            return

        now = time.monotonic()
        delta_time = now - self.last_log_time

        if delta_time <= 0.0:
            avg_tflops_per_gpu = 0.0
            avg_gbps_per_gpu = 0.0
        else:
            avg_tflops_per_gpu = self.total_num_flops_per_gpu / delta_time / 1e12
            avg_gbps_per_gpu = (
                (self.total_read_bytes_per_gpu + self.total_write_bytes_per_gpu)
                / delta_time
                / 1e9
            )

        log_fn(
            "%sMFU: %.1f TF/s/GPU %.1f GB/s/GPU",
            log_prefix,
            avg_tflops_per_gpu,
            avg_gbps_per_gpu,
        )

        if self.debug_logging:
            self.debug_logging.log(log_fn, log_prefix, delta_time)

        self.reset()


#### Prometheus Integration ####


class PerfMetricsProm:
    """Record performance metrics in Prometheus.

    Average TFLOPS (tera floating-point operations per second) can be
    calculated using a PromQL query:

      rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

    Average memory bandwidth in GB/s can be calculated using:

      (rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
       rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
    """

    _counter_cls = prometheus_client.Counter

    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        counter_flops = self._counter_cls(
            name="vllm:estimated_flops_per_gpu_total",
            documentation=(
                "Estimated number of floating point operations per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_flops = create_metric_per_engine(
            counter_flops, per_engine_labelvalues
        )

        counter_read_bytes = self._counter_cls(
            name="vllm:estimated_read_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes read from memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_read_bytes = create_metric_per_engine(
            counter_read_bytes, per_engine_labelvalues
        )

        counter_write_bytes = self._counter_cls(
            name="vllm:estimated_write_bytes_per_gpu_total",
            documentation=(
                "Estimated number of bytes written to memory per GPU "
                "(for Model Flops Utilization calculations)."
            ),
            labelnames=labelnames,
        )
        self.counter_write_bytes = create_metric_per_engine(
            counter_write_bytes, per_engine_labelvalues
        )

    def observe(self, perf_stats: PerfStats, engine_idx: int = 0):
        if not (
            perf_stats.num_flops_per_gpu
            or perf_stats.num_read_bytes_per_gpu
            or perf_stats.num_write_bytes_per_gpu
        ):
            return
        self.counter_flops[engine_idx].inc(perf_stats.num_flops_per_gpu)
        self.counter_read_bytes[engine_idx].inc(perf_stats.num_read_bytes_per_gpu)
        self.counter_write_bytes[engine_idx].inc(perf_stats.num_write_bytes_per_gpu)


## util functions


def get_required(obj: object, attr: str):
    """Get an attr from an object, or throw a InvalidComponentError if it's not set."""
    if not hasattr(obj, attr):
        raise InvalidComponent(f"Missing required attr {attr} in config")
    return getattr(obj, attr)


def getattr_from_list(obj: object, attrs: list[str], default: object = None):
    """Try to get the first attr that exists in the object
    from a list of attrs. Otherwise return None."""
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default
