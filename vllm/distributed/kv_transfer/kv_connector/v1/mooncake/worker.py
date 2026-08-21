# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any
import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size


try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None


logger = init_logger(__name__)

from ._protocol import (
    _async_loop, get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server, EngineId, MooncakeXferMetadata,
    MooncakeXferResponse, SendBlockMeta, TransferId, ReqId,
)

from ._worker_send import _WorkerSendMixin

from ._worker_receive import _WorkerReceiveMixin

from ._worker_setup import _WorkerSetupMixin

from .staging import _PinnedStagingArena, _StagingMixin

from .mooncake_utils import MooncakeBootstrapServer

from .stats import MooncakeKVConnectorStats

class MooncakeConnectorWorker(
    _WorkerSendMixin,
    _WorkerReceiveMixin,
    _WorkerSetupMixin,
    _StagingMixin,
):
    def __init__(
            self,
            vllm_config: VllmConfig,
            engine_id: str,
            kv_cache_config: "KVCacheConfig",
        ):
            if TransferEngine is None:
                logger.error("Mooncake is not available")
                raise RuntimeError("Mooncake is not available")
            logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)
    
            self.vllm_config = vllm_config
            # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
            # may change the current CUDA device during engine.initialize().
            self.device_id = torch.accelerator.current_device_index()
            current_platform.set_device(self.device_id)
    
            self.engine = TransferEngine()
            self.hostname = get_ip()
    
            assert (kv_transfer_config := vllm_config.kv_transfer_config)
            self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
            self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
            self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
                "num_workers", 10
            )
            # Create more tasks than workers to keep the thread pool saturated.
            # Tasks can await async events, so a surplus (2x is a robust heuristic)
            # prevents workers from idling.
            self.num_sender_tasks = self.num_sender_workers * 2
            protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
                "mooncake_protocol", "rdma"
            )
            device_name = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
                "device_name", ""
            )
            # Host-staging (GB10 / unified memory): transfer KV through pinned
            # host buffers because GPUDirect is architecturally unavailable (see
            # _PinnedStagingArena). Both ends must run this build; the D side
            # advertises staging-window addresses in place of its GPU cache
            # geometry, so the wire protocol itself is unchanged.
            self.host_staging = bool(
                kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
                    "host_staging", False
                )
            )
            self.staging_send_mib = int(
                kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
                    "staging_send_mib", 512
                )
            )
            self.staging_recv_mib = int(
                kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
                    "staging_recv_mib", 6144
                )
            )
            self._staging_send_arena: _PinnedStagingArena | None = None
            self._staging_recv_arena: _PinnedStagingArena | None = None
            # D-side receive window: per-unexpanded-region base addresses
            # (advertised to P in place of kv_caches_base_addr), kernel-block
            # capacity per region, and the expanded-region views used to replay
            # the transfer geometry for the completion H2D copy.
            self._staging_window_bases: list[int] = []
            self._staging_window_blocks = 0
            self._staging_window_regions: list[TransferRegion] = []
            self._staging_real_regions: list[TransferRegion] = []
            self._recv_staging_lock: asyncio.Lock | None = None
            if self.host_staging:
                logger.info(
                    "Mooncake host_staging ENABLED (send arena %d MiB, recv "
                    "window %d MiB) — KV crosses via pinned host buffers.",
                    self.staging_send_mib,
                    self.staging_recv_mib,
                )
            logger.info(
                "The Mooncake Transfer Engine is using %s as its protocol.", protocol
            )
            ret_value = self.engine.initialize(
                self.hostname, "P2PHANDSHAKE", protocol, device_name
            )
            if ret_value != 0:
                raise RuntimeError("Mooncake Transfer Engine initialization failed.")
    
            self.rpc_port = self.engine.get_rpc_port()
    
            logger.debug(
                "Mooncake Transfer Engine initialized at %s:%d",
                self.hostname,
                self.rpc_port,
            )
    
            self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
            self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
            self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
            self.engine_id: EngineId = engine_id
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.block_len_per_layer: list[int] = []
            self.kv_block_len_per_layer: list[int] = []
            self.registered_layer_names: list[str] = []
            self.registered_layer_indices: list[int] = []
            self.registered_group_indices: list[int] = []
            self.seen_base_addresses: list[int] = []
    
            assert (parallel_config := vllm_config.parallel_config)
            dp_rank = parallel_config.data_parallel_index
            dp_local_rank = parallel_config.data_parallel_rank_local
            self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
            self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
            self.pp_rank = get_pp_group().rank_in_group
    
            self.kv_caches_base_addr: list[int] = []
            self.device_kv_caches: dict[str, torch.Tensor] = {}
            self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}
    
            # For kv_both, we will act both prefiller and decoder.
            if not self.is_kv_consumer:
                # Background threads for sending kvcaches to D.
                # Each pool thread must be bound to the correct CUDA device
                # because CUDA device selection is thread-local.
                self._sender_executor = ThreadPoolExecutor(
                    max_workers=self.num_sender_workers,
                    thread_name_prefix="vllm-mooncake-sender",
                    initializer=self._bind_sender_thread_device,
                )
                logger.debug(
                    "Mooncake Prefiller: use %d workers to send kvcaches",
                    self.num_sender_workers,
                )
                # An asyncio queue to buffer incoming requests for the sender
                self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
                self.sender_loop = asyncio.new_event_loop()
                # Background thread for processing new sending requests.
                self._sender_listener_t = threading.Thread(
                    target=_async_loop, args=(self.sender_loop,), daemon=True
                )
                self._sender_listener_t.start()
    
                # Start bootstrap server on global rank 0.
                if should_launch_bootstrap_server(vllm_config):
                    _, port = get_mooncake_bootstrap_addr(vllm_config)
                    self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                    self.bootstrap_server.start()
    
            if not self.is_kv_producer:
                self.receiver_loop = asyncio.new_event_loop()
                self._mooncake_receiver_t = threading.Thread(
                    target=_async_loop, args=(self.receiver_loop,), daemon=True
                )
                self._mooncake_receiver_t.start()
                logger.debug("Mooncake Decoder: start receiver thread")
    
            self.finished_sending_reqs: set[ReqId] = set()
            self.finished_recving_reqs: set[ReqId] = set()
    
            self.xfer_stats = MooncakeKVConnectorStats()
    
            self.block_size = vllm_config.cache_config.block_size
            self.model_config = vllm_config.model_config
            self.cache_config = vllm_config.cache_config
            self.kv_cache_config = kv_cache_config
            self.use_mla = self.model_config.use_mla
            self._physical_blocks_per_logical_kv_block = 1
            self._sync_block_size_with_kernel()
    
            self.attn_backends = get_current_attn_backends(vllm_config)
            self.kv_cache_layout = get_kv_cache_layout()
            logger.debug(
                "Detected attention backends %s",
                [backend.get_name() for backend in self.attn_backends],
            )
            logger.debug("Detected kv cache layout %s", self.kv_cache_layout)
    
            self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
            self._layer_specs: dict[str, KVCacheSpec] = {}
            for group in kv_cache_config.kv_cache_groups:
                group_spec = group.kv_cache_spec
                specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
                for layer_name in group.layer_names:
                    self._layer_specs[layer_name] = specs_by_layer.get(
                        layer_name, group_spec
                    )
            self._layer_group_indices: dict[str, int] = {
                layer: group_index
                for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
                for layer in group.layer_names
            }
            self.transfer_topo = TransferTopology(
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                block_size=self.block_size,
                engine_id=self.engine_id,
                is_mla=self.use_mla,
                is_mamba=kv_cache_config.has_mamba_layers,
                total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
                attn_backends=self.attn_backends,
            )
    
            self.async_zmq_ctx = zmq.asyncio.Context()
            self._encoder = msgspec.msgpack.Encoder()
            self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
            self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)
