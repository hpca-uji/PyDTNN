"""gRPC communications"""

# NOTE: gRPC does not conform well to a async send & async receive model,
# it expects remote procedure calls to be called, processed and responded.
# To simulate this model we created a bidirectional streaming procedure.
# Sent data is queued at the server, recived data is polled until available.

# NOTE: Polling is implemented with a exponential backoff time and a limit.
# The gRPC library queues requests, so requests would always be replyed in
# a timely maner, but we do not want to hogh the CPU or network with
# usesless requests.

# NOTE: It is important to not hold the prodedures indefinitely, since this
# could starve the server of threads. Additionaly, if a streaming direction
# was already closed, messages could end up queued forever if not restarted.

import sys
import uuid
from collections import abc

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import grpc
finally:
    sys.path.insert(0, _pkg)

from pydtnn.libs import net_queue as comms  # noqa: E402

__all__ = (
    "Protocol",
)


class Protocol[T](comms.Communicator[T]):
    """Shared base gRPC implementation"""
    _compression = grpc.Compression.NoCompression

    def __init__(self, options: comms.CommunicatorOptions = comms.CommunicatorOptions()) -> None:
        """Initialize protocol"""
        super().__init__(options)
        self._grpc_options = {"grpc.max_receive_message_length": self.options.connection.max_size, "grpc.max_send_message_length": self.options.connection.max_size}

    def _put_flush(self, peer: uuid.UUID) -> abc.Generator[abc.Buffer]:
        """Transforms state to message"""
        state = self._states[peer]

        size = 0
        state.put_flush_queue()
        for view in state.put_flush_buffer():
            with view:
                yield bytes(view)
                size += len(view)
        self._put_commit(peer, size)
