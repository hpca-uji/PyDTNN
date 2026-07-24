"""Utilities for GPU management and hardware information retrieval."""

import enum
import logging
import re
import subprocess

__all__ = (
    "CudnnDataType",
    "get_gpu_memory_used",
    "get_gpus_per_node",
)

logger = logging.getLogger(__name__)


class CudnnDataType(enum.StrEnum):
    """Enumeration of supported cuDNN data types."""

    FLOAT64 = "CUDNN_DATA_DOUBLE"
    FLOAT32 = "CUDNN_DATA_FLOAT"
    INT8 = "CUDNN_DATA_INT8"
    INT32 = "CUDNN_DATA_INT32"


def get_gpu_memory_used() -> str:
    """Retrieves the current GPU memory usage from nvidia-smi.

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


def get_gpus_per_node() -> int:
    """Counts the number of available GPUs on the current node.

    Returns:
        The number of detected GPUs, or 0 if nvidia-smi is unavailable.
    """
    try:
        gpus_per_node = subprocess.check_output(["nvidia-smi", "-L"]).count(b"UUID")
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus_per_node = 0
    return gpus_per_node
