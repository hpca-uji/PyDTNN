"""gRPC client"""

import uuid
import grpc
import copy
import threading
from collections import abc
from queue import SimpleQueue, Empty
from concurrent.futures import Future

from net_queue import client
from net_queue import asynctools
from net_queue.grpc import Protocol
from net_queue.io_stream import Stream
from net_queue import CommunicatorOptions, ResourceClosed


__all__ = (
    "Communicator",
)


# Sentinel objects
ARG_MISSING = object()


class Communicator(Protocol[grpc.StreamStreamMultiCallable], client.Client[grpc.StreamStreamMultiCallable]):
    """gRPC client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(copy.replace(options, workers=1))

        # State
        self._get_grpc = SimpleQueue()
        self._put_grpc = threading.Event()

        # gRPC
        config: abc.MutableMapping = {
            "target": str(self.options.netloc),
            "options": list(self._grpc_options.items()),
            "compression": self._compression
        }

        if self.options.security:
            config["credentials"] = grpc.ssl_channel_credentials(root_certificates=self.options.security.certificate.read_bytes() if self.options.security.certificate else None)
            self._channel = grpc.secure_channel(**config)
        else:
            self._channel = grpc.insecure_channel(**config)

        self._conntected = True
        self._comm = self._channel.stream_stream(
            method="/grpc/comm",
            request_serializer=lambda x: x,  # type: ignore (handled externally)
            response_deserializer=lambda x: x
        )

        self._connection_ini(self._comm)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        future = super()._put(stream, peer)
        comm = self._comms[peer]
        self._pool.submit(self._c2s, comm).add_done_callback(asynctools.future_warn_exception)
        return future

    def _get(self, *peers: uuid.UUID) -> uuid.UUID:
        try:
            peer = self._get_events.get_nowait()
        except Empty:
            if not hasattr(self, "_conntected"):
                raise ResourceClosed()
            comm = self._comm
            self._pool.submit(self._s2c, comm)
            peer = self._get_events.get()
            self._get_grpc.put(None)
        return peer

    def _handle_connection(self, comm: grpc.StreamStreamMultiCallable) -> None:
        """Communication round"""
        peer = self._set_default_peer(comm)
        state = self._states[peer]

        for data in comm(self._put_flush(peer)):
            state.get_write(data)
            self._process_gets(peer)
            peer = state.peer

        if not state.state and state.put_empty():
            self._connection_fin(comm)

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

    def _c2s(self, comm: grpc.StreamStreamMultiCallable) -> None:
        """Communication client to server"""
        # Check if already handled
        peer = self._set_default_peer(comm)
        state = self._states[peer]
        if state.put_empty():
            return
        self._handle_connection(comm)

    def _s2c(self, comm: grpc.StreamStreamMultiCallable) -> None:
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
            if not hasattr(self, "_conntected"):
                return
            elif backoff:
                self._put_grpc.wait(next(backoff))
                self._put_grpc.clear()
            else:
                backoff = self._new_backoff()
            self._handle_connection(comm)
        self._get_grpc.get_nowait()

    def _connection_pre_fin(self, peer: uuid.UUID) -> None:
        """Communication finalization"""
        del self._conntected
        super()._connection_pre_fin(peer)

    def _close(self) -> None:
        """Close the client"""
        comm = self._comm
        peer = self._comms.inverse[comm]
        state = self._states[peer]
        self._session_fin(peer)

        while state.state:
            self._pool.submit(self._s2c, comm).result()

        with self._lock:
            while hasattr(self, "_conntected"):
                self._lock.wait()

        self._pool.shutdown()
        self._channel.close()

        super()._close()
