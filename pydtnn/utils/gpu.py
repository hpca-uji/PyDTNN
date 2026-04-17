import subprocess
import re
import logging
logger = logging.getLogger(__name__)


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
