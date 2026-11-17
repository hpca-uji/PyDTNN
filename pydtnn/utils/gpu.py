"""Utilities for GPU management and hardware information retrieval."""

import enum
import logging
import re
import subprocess

import numpy as np

__all__ = (
    "nccl_dtype_from_numpy",
    "cudnn_dtype_from_numpy",
    "get_cuda_memory_used",
    "get_cuda_device_count",
)

logger = logging.getLogger(__name__)


def nccl_dtype_from_numpy(dtype: np.dtype) -> enum.Enum:
    """Convert a Numpy DType to NCCL DType"""
    from pydtnn.libs import nccl
    return {
        np.dtype(np.float64): nccl.DataType.Float64,
        np.dtype(np.float32): nccl.DataType.Float32,
        np.dtype(np.int8): nccl.DataType.Int8,
        np.dtype(np.int32): nccl.DataType.Int32,
    }[dtype]


def cudnn_dtype_from_numpy(dtype: np.dtype) -> int:
    """Convert a Numpy DType to cuDNN DType"""
    from pydtnn.libs import cudnn
    return {
        np.dtype(np.float64): cudnn.cudnnDataType["CUDNN_DATA_DOUBLE"],
        np.dtype(np.float32): cudnn.cudnnDataType["CUDNN_DATA_FLOAT"],
        np.dtype(np.int8): cudnn.cudnnDataType["CUDNN_DATA_INT8"],
        np.dtype(np.int32): cudnn.cudnnDataType["CUDNN_DATA_INT32"],
    }[dtype]


def get_cuda_memory_used() -> str:
    """Retrieves the current CUDA memory usage from nvidia-smi.

    Returns:
        A string representing the used memory, or 'None' if retrieval fails.
    """
    pattern = r"Used *: .*"
    try:
        memory = subprocess.check_output(["nvidia-smi", "-q", "-d", "MEMORY"]).decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        memory = str(None)
    else:
        if match := re.search(pattern, memory):
            memory = match.group().split(":")[-1].strip()
        else:
            memory = str(None)
    return memory


def get_cuda_device_count() -> int:
    """Counts the number of available CUDA devices on the current node.

    Returns:
        The number of detected CUDA devices, or 0 if retrieval fails.
    """
    try:
        gpus_per_node = subprocess.check_output(["nvidia-smi", "-L"]).count(b"UUID")
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus_per_node = 0
    return gpus_per_node
