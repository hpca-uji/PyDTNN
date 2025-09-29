"""gRPC communications"""

import sys
import uuid
from collections import abc

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import grpc
finally:
    sys.path.insert(0, _pkg)

import net_queue as nq  # noqa: E402

__all__ = (
    "Protocol",
)


class Protocol[T](nq.Communicator[T]):
    """Shared base gRPC implementation"""
    _compression = grpc.Compression.NoCompression

    def __init__(self, options: nq.CommunicatorOptions = nq.CommunicatorOptions()) -> None:
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
