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


logger = init_logger(__name__)

from ._protocol import (
    EngineId, MooncakeConnectorMetadata, MooncakeXferMetadata,
    MooncakeXferResponse, MooncakeXferResponseStatus, PullReqMeta, ReqId,
)

class _WorkerReceiveMixin:
    async def receive_kv_from_single_worker(
            self,
            worker_addr: str,
            pull_metas: dict[ReqId, PullReqMeta],
        ):
            req_ids = set(pull_metas)
            if self.host_staging:
                # Advertise the pinned receive window's geometry in place of the
                # real GPU cache: the producer's destination arithmetic then lands
                # in the window, where it registers with the engine. Per-request
                # block ids are replaced with window slot ids assigned by
                # _staging_assign_slots; the producer's logical->kernel expansion
                # maps them onto window slots exactly as it would real blocks.
                advertised_bases = self._staging_window_bases
                req_blocks = {
                    req_id: (
                        pull_meta.transfer_id,
                        self._staging_slot_block_ids(pull_meta),
                    )
                    for req_id, pull_meta in pull_metas.items()
                }
            else:
                advertised_bases = self.kv_caches_base_addr
                req_blocks = {
                    req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                    for req_id, pull_meta in pull_metas.items()
                }
            metadata = MooncakeXferMetadata(
                remote_hostname=self.hostname,
                remote_port=self.rpc_port,
                remote_tp_size=self.tp_size,
                remote_tp_rank=self.tp_rank,
                req_blocks=req_blocks,
                kv_caches_base_addr=advertised_bases,
                block_lens=self.block_len_per_layer,
                kv_block_lens=self.kv_block_len_per_layer,
                registered_layer_names=self.registered_layer_names,
                registered_layer_indices=self.registered_layer_indices,
                registered_group_indices=self.registered_group_indices,
            )
    
            encoded_data = self._encoder.encode(metadata)
            logger.debug(
                "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
            )
            logger.debug(
                "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
            )
    
            # Send query for the request.
            try:
                with make_zmq_socket(
                    self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
                ) as sock:
                    # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                    sock.setsockopt(
                        zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                    )
                    await sock.send(encoded_data)
                    while True:
                        ret_msg = await sock.recv()
                        response = self._xfer_resp_decoder.decode(ret_msg)
                        if response.status == MooncakeXferResponseStatus.ERROR:
                            logger.error(
                                "Error happens during transferring kvcache for %s: %s",
                                req_ids,
                                response.err_msg,
                            )
                            self.xfer_stats.record_failed_recv()
                            return
                        self.process_pulling_result(response, pull_metas)
                        if response.status == MooncakeXferResponseStatus.FINISH:
                            break
            except zmq.ContextTerminated:
                logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
            except Exception as e:
                logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
                self.xfer_stats.record_failed_recv()
                return
    def process_pulling_result(
            self,
            response: MooncakeXferResponse,
            pull_metas: dict[ReqId, PullReqMeta],
        ):
            ok_reqs: list[ReqId] = response.ok_reqs or []
    
            for req_id in ok_reqs:
                pull_meta = pull_metas[req_id]
                # No race because we are in async loop.
                pull_meta.pull_tasks_count -= 1
                if pull_meta.pull_tasks_count == 0:
                    if self.host_staging and pull_meta.staging_slots:
                        # The producer's sync writes landed in the pinned recv
                        # window; move KV into the real GPU blocks before the
                        # scheduler is told the request is ready.
                        try:
                            self._staging_h2d_copy_request(pull_meta)
                        except Exception:
                            logger.exception(
                                "host_staging: H2D completion copy failed for "
                                "req %s; NOT marking it received.",
                                req_id,
                            )
                            self.xfer_stats.record_failed_recv()
                            continue
                    self.finished_recving_reqs.add(pull_meta.d_req_id)
    
            if ok_reqs:
                logger.debug("pulling kv_caches for %s finished", ok_reqs)
    
            if response.err_reqs:
                logger.error(
                    "pulling kv_caches for %s failed: %s",
                    response.err_reqs,
                    response.err_msg,
                )
    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
            url = remote_bootstrap_addr + "/query"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data: dict = response.json()
                    for _, dp_entry in data.items():
                        remote_engine_id = dp_entry["engine_id"]
                        self._remote_agents[remote_engine_id] = {
                            int(tp_rank): {
                                int(pp_rank): worker_addr
                                for pp_rank, worker_addr in tp_entry.items()
                            }
                            for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                        }
                        self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
            except Exception as e:
                logger.error(
                    "Failed to connect to bootstrap server %s: %s",
                    remote_bootstrap_addr,
                    e,
                )
    
            # Always notify others regardless of connection success or failure.
            self._pending_bootstrap_queries[remote_bootstrap_addr].set()
            del self._pending_bootstrap_queries[remote_bootstrap_addr]
    def receive_kv(
            self,
            remote_engine_id: EngineId,
            pull_metas: dict[ReqId, PullReqMeta],
        ) -> list[asyncio.Task] | None:
            if self.host_staging:
                # Serialized in _start_load_kv: only one pull batch occupies the
                # receive window at a time. Assign window slots before any
                # metadata goes out.
                overflow = self._staging_assign_slots(pull_metas)
                for req_id in overflow:
                    logger.error(
                        "host_staging: req %s exceeds the recv window capacity "
                        "(%d blocks); it will fail on the producer's timeout.",
                        req_id,
                        self._staging_window_blocks,
                    )
                pull_metas = {
                    req_id: meta
                    for req_id, meta in pull_metas.items()
                    if meta.staging_slots is not None
                }
                if not pull_metas:
                    return []
            remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
                self._tp_size[remote_engine_id]
            )
            worker_addrs: list[str] = []
            selected_remote_pp: dict[int, list[int]] = {}
            for remote_tp_rank in remote_tp_ranks:
                pp_to_addr = self._remote_agents[remote_engine_id][remote_tp_rank]
                if self.pp_size == len(pp_to_addr) and self.pp_rank in pp_to_addr:
                    pp_ranks = [self.pp_rank]
                else:
                    pp_ranks = sorted(pp_to_addr)
                selected_remote_pp[remote_tp_rank] = pp_ranks
                worker_addrs.extend(pp_to_addr[pp_rank] for pp_rank in pp_ranks)
    
            count = len(worker_addrs)
            logger.debug(
                "Receiving Mooncake KV for engine %s from producer TP ranks %s "
                "and PP ranks %s",
                remote_engine_id,
                remote_tp_ranks,
                selected_remote_pp,
            )
            for pull_meta in pull_metas.values():
                pull_meta.pull_tasks_count = count
            tasks = [
                asyncio.create_task(
                    self.receive_kv_from_single_worker(worker_addr, pull_metas)
                )
                for worker_addr in worker_addrs
            ]
            return tasks if self.host_staging else None
    async def handle_new_engine_id(
            self,
            remote_engine_id: EngineId,
            pull_metas: dict[ReqId, PullReqMeta],
        ) -> list[asyncio.Task] | None:
            remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
            if remote_bootstrap_addr not in self._pending_bootstrap_queries:
                self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
                await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
            else:
                await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()
    
            if remote_engine_id not in self._remote_agents:
                logger.error(
                    "Failed to find remote engine_id %s from bootstrap server %s",
                    remote_engine_id,
                    remote_bootstrap_addr,
                )
                return None
    
            return self.receive_kv(remote_engine_id, pull_metas)
    async def _start_load_kv(
            self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
        ):
            if self.host_staging:
                # Serialize pull batches so the receive window's slot space is
                # owned by one batch at a time (slots reset per batch). The lock
                # is held until the batch's transfers complete because
                # receive_kv only spawns the per-worker tasks.
                if self._recv_staging_lock is None:
                    self._recv_staging_lock = asyncio.Lock()
                async with self._recv_staging_lock:
                    await self._start_load_kv_inner(reqs_to_recv, await_tasks=True)
                return
            await self._start_load_kv_inner(reqs_to_recv, await_tasks=False)
    async def _start_load_kv_inner(
            self,
            reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]],
            await_tasks: bool,
        ):
            for remote_engine_id, pull_metas in reqs_to_recv.items():
                if remote_engine_id not in self._remote_agents:
                    if await_tasks:
                        # host_staging: keep the window lock held across bootstrap
                        # discovery AND the ensuing transfer.
                        tasks = await self.handle_new_engine_id(
                            remote_engine_id, pull_metas
                        )
                        if tasks:
                            await asyncio.gather(*tasks, return_exceptions=True)
                    else:
                        asyncio.create_task(
                            self.handle_new_engine_id(remote_engine_id, pull_metas)
                        )
                else:
                    tasks = self.receive_kv(remote_engine_id, pull_metas)
                    if await_tasks and tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
            if not self.is_kv_producer and metadata.reqs_to_recv:
                asyncio.run_coroutine_threadsafe(
                    self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
                )
    
            if not self.is_kv_consumer and (
                metadata.reqs_to_send or metadata.reqs_not_processed
            ):
                asyncio.run_coroutine_threadsafe(
                    self.record_send_reqs(metadata), self.sender_loop
                )
