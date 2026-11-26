"""PyDTNN framework package"""

import logging
import os
import platform
import uuid
from datetime import datetime
from importlib import metadata as libdata

__all__ = (
    "host_uuid",
    "metadata",
    "package_name",
    "state_path",
    "timestamp"
)

logger = logging.getLogger(__name__)


# CONSTANTS
host_uuid = uuid.getnode()
package_name = __name__
state_path = os.environ.get(f"{package_name.upper()}_STATE", ".")
timestamp = (
    datetime.now()
    .isoformat(timespec="seconds")
    .replace(" ", "-")
    .replace(":", "-")
    .replace(".", "-")
)

cpu_affinity = tuple(range(os.process_cpu_count()))
try:
    _sched_getaffinity = os.sched_getaffinity
except AttributeError:
    pass  # unsupported
else:
    cpu_affinity = tuple(_sched_getaffinity(0))
try:
    from psutil import Process
except ModuleNotFoundError:
    pass  # unavailable
else:
    cpu_affinity = tuple(Process().cpu_affinity())

# ENVIRONMENT
metadata = {
    "time": timestamp,
    "node": host_uuid,
    "host": " ".join(platform.uname()).strip(),
    "cpu": str(cpu_affinity),
    "packages": {
        platform.python_implementation().lower(): platform.python_version(),
        **{
            dist.name: dist.version
            for dist in libdata.distributions()
        },
    },
    "environ": dict(os.environ),
}
