"""gRPC client"""

import uuid
import grpc
import copy
import threading
from collections import abc
from queue import SimpleQueue, Empty
from concurrent.futures import Future

from pydtnn import comms
from pydtnn.comms import client
from pydtnn.comms.grpc import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions, ResourceClosed


__all__ = (
    "Client",
)


# Sentinel objects
ARG_MISSING = object()


class Client(Protocol[grpc.StreamStreamMultiCallable], client.Client[grpc.StreamStreamMultiCallable]):
    """gRPC client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(copy.replace(options, workers=1))

        # State
        self._get_grpc = SimpleQueue()
        self._put_grpc = threading.Event()

        # gRPC
        config: abc.MutableMapping = {
            "target": str(self._options.netloc),
            "options": list(self._grpc_options.items()),
            "compression": self._compression
        }

        if comms.SSL:
            config["credentials"] = grpc.ssl_channel_credentials(root_certificates=comms.SSL_CERT.read_bytes() if comms.SSL_CERT else None)
            self._channel = grpc.secure_channel(**config)
        else:
            self._channel = grpc.insecure_channel(**config)

        self._client = True
        self._com = self._channel.stream_stream(method="/grpc/com", request_serializer=bytes, response_deserializer=lambda x: x)

        self._ini(self._com)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        sock = self._peers[peer]
        self._pool.submit(self._c2s, sock).add_done_callback(lambda future: future.result())
        return future

    def _get(self, *peers: uuid.UUID) -> uuid.UUID:
        try:
            peer = self._get_event.get_nowait()
        except Empty:
            if not hasattr(self, "_client"):
                raise ResourceClosed()
            sock = self._com
            self._pool.submit(self._s2c, sock)
            peer = self._get_event.get()
            self._get_grpc.put(None)
        return peer

    def _handle_connection(self, sock: grpc.StreamStreamMultiCallable) -> None:
        """Communication round"""
        peer = self._get_peer(sock)
        state = self._state[peer]

        for data in self._m2d(sock(self._s2m(state))):
            state.get_write(data)
            self._process_session(peer)

        if not state.state and state.put_empty():
            self._fin(sock)

    @staticmethod
    def _new_backoff(start=-10, end=0) -> abc.Generator[float]:
        """Exponential backoff generator"""
        if start >= end:
            raise ValueError(f"Null backoff range ({start} to {end})")

        # Exponential growth
        for exponent in range(start, end):
            backoff = 2 ** exponent
            yield backoff

        # Plateau backoff
        while True:
            yield backoff

    def _c2s(self, sock: grpc.StreamStreamMultiCallable) -> None:
        """Communication client to server"""
        # Check if already handled
        peer = self._get_peer(sock)
        state = self._state[peer]
        if state.put_empty():
            return
        self._handle_connection(sock)

    def _s2c(self, sock: grpc.StreamStreamMultiCallable) -> None:
        """Communication server to client"""
        # Check if already handled
        try:
            self._get_grpc.get_nowait()
        except Empty:
            pass
        else:
            return

        # Handle recive loop
        backoff = None
        while self._get_grpc.empty():
            if not hasattr(self, "_client"):
                return
            elif backoff:
                self._put_grpc.wait(next(backoff))
                self._put_grpc.clear()
            else:
                backoff = self._new_backoff()
            self._handle_connection(sock)
        self._get_grpc.get_nowait()

    def _extra_fin(self, peer: uuid.UUID) -> None:
        """Communication finalization"""
        del self._client
        super()._extra_fin(peer)

    def _close(self) -> None:
        """Close the client"""
        sock = self._com
        peer = self._get_peer(sock)
        state = self._state[peer]
        self._put(self._session_fin(state), peer)

        while state.state:
            self._pool.submit(self._s2c, sock).result()

        with self._lock:
            while hasattr(self, "_client"):
                self._lock.wait()

        self._pool.shutdown()
        self._channel.close()

        super()._close()
