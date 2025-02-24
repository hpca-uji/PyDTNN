"""Message Passing Interface"""

import sys as _sys

from pydtnn import comms as _comms


# Select implementation
match _comms.PROTOCOL:
    case _comms.Protocol():
        from pydtnn.libs.mpi import client as _module
        from pydtnn.libs.mpi import comm as _comm

        # If requested, start a local server
        if _comm.get_init():
            if _comm.get_rank() == 0:
                from pydtnn.libs.mpi.server import start_local_server as _start_local_server
                _start_local_server()
            else:
                # Allow some time for server startup
                from time import sleep as _sleep
                _sleep(0.5)

    case _:
        from mpi4py import MPI as _module


# Replace module
_sys.modules[__name__] = _module


def __getattr__(key):
    """Proxy all attributes to module"""
    return getattr(_module, key)
