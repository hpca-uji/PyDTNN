import enum
import logging
import re
import subprocess

logger = logging.getLogger(__name__)


class CudnnDataType(enum.StrEnum):
    FLOAT64 = "CUDNN_DATA_DOUBLE"
    FLOAT32 = "CUDNN_DATA_FLOAT"
    INT8 = "CUDNN_DATA_INT8"
    INT32 = "CUDNN_DATA_INT32"


def get_gpu_memory_used() -> str:
    pattern = r"Used *: .*"
    try:
        memory = subprocess.check_output(["nvidia-smi", "-q", "-d", "MEMORY"]).decode()
    except (FileNotFoundError, subprocess.CalledProcessError):
        memory = str(None)
    else:
        memory = re.search(pattern, memory).group().split(":")[-1].strip()  # type: ignore
    return memory


def get_gpus_per_node() -> int:
    try:
        gpus_per_node = subprocess.check_output(["nvidia-smi", "-L"]).count(b'UUID')
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpus_per_node = 0
    return gpus_per_node
