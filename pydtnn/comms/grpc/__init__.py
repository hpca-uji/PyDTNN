"""gRPC communications"""

# NOTE: gRPC does not conform well to a async send & async receive model,
# it expexts remote procedure calls to be called, processed and responded.
# To simulate this model we created a send procedure and a receive procedure.
# Sent data is queued at the server, recived data is polled until available.

# NOTE: Polling is implemented with a exponential backoff time and a limit
# provided by the server. The gRPC library queues requests, so requests would
# always be replyed in a timely maner, but we do not want to hogh the CPU or
# network with usesless requests.

# NOTE: It is important to not hold the prodedures for long, since a receive might
# be waiting on a send, but the send can not be processed if all threads are
# are blocked on receive prodedures.

# NOTE: The server can not detect when clients disconnect ungratefully, since
# procedure connections are ephemeral. If a fin message is not sent, clients
# data and queues are held indefinitely.

# TODO: Implement procedure buffering and use streaming procedures. This is
# specially useful when multiple threads use the same communication object.


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
