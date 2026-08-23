# SPDX-License-Identifier: Apache-2.0
"""Setuptools build for the vendored nixl_ep extension.

This extension links against the installed ``libnixl.so`` from the
``nixl-cu*`` wheel and takes NIXL C++ headers from the vendored
``nixl_headers/`` directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
NIXL_HEADERS = ROOT / "nixl_headers"


def _find_nixl_lib() -> tuple[str, str] | None:
    """Return (library_directory, include_directory) for the installed NIXL core."""
    candidate_modules = ["nixl", "nixl_cu13", "nixl_cu12"]
    for mod_name in candidate_modules:
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        pkg_dir = Path(mod.__file__).resolve().parent
        # The wheel places libnixl.so in a sibling .<variant>.mesonpy.libs dir.
        search_roots = [pkg_dir.parent, pkg_dir]
        for root in search_roots:
            if not root.is_dir():
                continue
            for so_path in root.rglob("libnixl.so*"):
                lib_dir = str(so_path.resolve().parent)
                return lib_dir, str(NIXL_HEADERS)
    return None


def _nvcc_version() -> tuple[int, int] | None:
    """Return (major, minor) of the system nvcc, or None."""
    nvcc = os.environ.get("NVCC", "nvcc")
    try:
        output = subprocess.check_output(
            [nvcc, "--version"], universal_newlines=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    # Parse "release 12.9, V12.9.99"
    for line in output.splitlines():
        if "release" in line:
            try:
                version_part = line.split("release")[-1].split(",")[0].strip()
                major, minor = map(int, version_part.split(".")[:2])
                return major, minor
            except Exception:
                return None
    return None


def _build_extension() -> CUDAExtension | None:
    nixl_info = _find_nixl_lib()
    if nixl_info is None:
        print(
            "WARNING: nixl-cu* wheel not found; skipping vendored nixl_ep build.",
            file=sys.stderr,
        )
        return None

    lib_dir, include_dir = nixl_info
    print(f"Building vendored nixl_ep against NIXL lib dir: {lib_dir}", file=sys.stderr)

    sources = [
        str(CSRC / "nixl_ep.cpp"),
        str(CSRC / "vmm.cpp"),
        str(CSRC / "kernels" / "nixl_ep_ll.cu"),
        str(CSRC / "kernels" / "nixl_ep_ht.cu"),
        str(CSRC / "kernels" / "layout.cu"),
        str(CSRC / "kernels" / "runtime.cu"),
    ]

    cpp_args = [
        "-DHAVE_CUDA",
        "-DTORCH_EXTENSION_NAME=nixl_ep_cpp",
        "-DTOPK_IDX_BITS=32",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]

    cuda_args = [
        "-DHAVE_CUDA",
        "-DTORCH_EXTENSION_NAME=nixl_ep_cpp",
        "-DTOPK_IDX_BITS=32",
        "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
        "--expt-relaxed-constexpr",
        "--ptxas-options=--register-usage-level=10",
        "-Xcompiler",
        "-Wno-deprecated-declarations",
        "-Xcompiler",
        "-Wno-unused-variable",
        "-Xcompiler",
        "-Wno-sign-compare",
        "-Xcompiler",
        "-Wno-reorder",
        "-Xcompiler",
        "-Wno-attributes",
    ]

    nvcc_version = _nvcc_version()
    if nvcc_version is not None and nvcc_version == (12, 9):
        cpp_args.append("-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE")
        cuda_args.append("-D_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE")

    include_dirs = [
        str(CSRC),
        str(CSRC / "kernels"),
        str(NIXL_HEADERS),
    ]

    library_dirs = [lib_dir]
    libraries = ["nixl", "cuda"]

    return CUDAExtension(
        name="nixl_ep.nixl_ep_cpp",
        sources=sources,
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args={"cxx": cpp_args, "nvcc": cuda_args},
        extra_link_args=[f"-Wl,-rpath,{lib_dir}"],
    )


ext_modules: list[CUDAExtension] = []
_ext = _build_extension()
if _ext is not None:
    ext_modules.append(_ext)

setup(
    name="nixl_ep",
    version="1.4.0+vllm",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
