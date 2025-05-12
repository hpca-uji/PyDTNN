"""gRPC communications"""

# NOTE: gRPC does not conform well to a async send & async receive model,
# it expects remote procedure calls to be called, processed and responded.
# To simulate this model we created a bidirectional streaming procedure.
# Sent data is queued at the server, recived data is polled until available.

# NOTE: Polling is implemented with a exponential backoff time and a limit
# provided by the server. The gRPC library queues requests, so requests would
# always be replyed in a timely maner, but we do not want to hogh the CPU or
# network with usesless requests.

# NOTE: It is important to not hold the prodedures indefinitely, since this
# could starve the server of threads. Additionaly, if a streaming direction
# was already closed, messages could end up queued forever if not restarted.

import sys
import typing
from collections import abc

from pydtnn import comms
from pydtnn.comms.grpc import grpc_pb2

# Make sure global package is not confused with current package
_pkg = sys.path.pop(0)
try:
    import grpc  # noqa: F401
finally:
    sys.path.insert(0, _pkg)


__all__ = (
    "Protocol",
)


class Protocol(comms.Communicator):
    """Shared base gRPC implementation"""
    _compression = grpc.Compression.NoCompression
    _max_message_size = 16 * 1024 ** 1 - 1

    def __init__(self, addr: str, port: int) -> None:
        """Initialize protocol"""
        super().__init__(addr, port)

        # Calculate maximun data size (reduced for protobuf overhead)
        data = bytearray(self._max_message_size)
        size = grpc_pb2.Message(data=bytes(data)).ByteSize()
        headers = size - len(data)
        self._max_data_size = self._max_message_size - headers

    @property
    def _options(self) -> abc.Iterable[tuple[str, typing.Any]]:
        """gRPC channel options"""
        return (
            ("grpc.max_receive_message_length", self._max_message_size),
            ("grpc.max_send_message_length", self._max_message_size)
        )

    @staticmethod
    def _m2d(messages: abc.Iterable[grpc_pb2.Message]) -> abc.Generator[bytes]:
        """Transforms gRPC messages to bytes"""
        for message in messages:
            yield message.data

    @staticmethod
    def _s2m(state: comms.ConnectionData) -> abc.Generator[grpc_pb2.Message]:
        """Transforms state to message"""
        state.put_flush()
        while not state.put_buffer.empty():
            with state.put_read() as view:
                yield grpc_pb2.Message(data=bytes(view))
