# Vendored `nixl_ep` with host-staging support

This directory contains a vendored copy of the NIXL expert-parallel (EP)
communication example from `ai-dynamo/nixl` (`examples/device/ep/`), with a
small, self-contained delta that adds a **host-staging mode** for unified-memory
platforms such as GB10 / DGX Spark where GPUDirect RDMA / DMA-buf export is
unavailable.

## Upstream provenance

See `UPSTREAM.txt` for the exact upstream tag and commit.

The upstream `LICENSE` (Apache-2.0) and `LICENSE-DeepEP` (MIT) files are kept
intact. All file headers from the upstream source are preserved.

## Delta summary

- `csrc/vmm.hpp` / `csrc/vmm.cpp`: add `pinned_host_region` for
  `cudaMallocHost`-allocated page-locked host memory.
- `csrc/nixl_ep.hpp` / `csrc/nixl_ep.cpp`: add a `host_staging` constructor flag.
  When enabled:
  - NIXL-registered buffers (`rdma_buffer_ptr`, `sync_buffer_ptr`,
    `sync_count_ptr`, `mask_buffer_ptr`, and the HT barrier counter if used) are
    allocated in pinned host memory and registered as `DRAM_SEG`.
  - `num_nvl_bytes` must be `0` (NVLink disabled).
  - `get_local_buffer_tensor(use_rdma_buffer=True)` returns a CPU tensor over
    the pinned host buffer.
- `nixl_ep/buffer.py`: exposes `host_staging` and a `host_staging_enabled`
  property.
- vLLM wiring: `VLLM_NIXL_EP_HOST_STAGING` controls activation; see
  `vllm/distributed/device_communicators/all2all.py`.

The high-throughput (`ht_*`) and NVLink paths are intentionally unchanged and
out of scope for this fork.

## Build

The extension is built by `setup.py` using `torch.utils.cpp_extension`:

```bash
cd third_party/nixl_ep
python setup.py build_ext --inplace
```

vLLM's top-level packaging builds this extension automatically when CUDA is
enabled and the installed `nixl-cu*` wheel's `libnixl.so` can be located.

## Runtime note

Host-staging mode is homogeneous: all ranks in an EP group must use the same
`VLLM_NIXL_EP_HOST_STAGING` setting. Mixed staging/non-staging groups fail loudly
at `connect_ranks`.
