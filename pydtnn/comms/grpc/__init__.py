"""gRPC communications"""

# NOTE: Module considerations
#
# gRPC does not conform well to a async send & async receive model,
# it expexts remote procedure calls to be recived, processed and responded.
# To simulate this model we created a send procedure and a receive procedure.
# Sent data is queued at the server, recived data is polled until available.
#
# It is important to not hold the prodedures for long, since a receive might
# be waiting on a send, but the send can not be processed if all threads are
# are blocked on receive prodedures.
#
# Polling is implemented with a exponential backoff time and a limit provided
# by the server. The gRPC library implementation queues requests, so requests
# would be replyed in a timely maner, but we do not want to hogh the CPU or
# network with usesless requests.
#
# Low level comunications are handled single-threaded and are limited to pushing
# or pulling data to queues without blocking, so all operations are minimal
# and fast.
#
# Expensive operations, such as serialization and blocking, are done at at the
# public's API callers thread.

import sys

from pydtnn import comms

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import grpc  # noqa: F401
finally:
    sys.path.insert(0, _pkg)


__all__ = (
    "Protocol",
)


class Protocol(comms.Communication):
    """Shared base gRPC implementation"""
    _compression = grpc.Compression.NoCompression
