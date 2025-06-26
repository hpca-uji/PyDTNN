"""gRPC client"""

import uuid
import grpc
import threading
from collections import abc
from queue import SimpleQueue, Empty
from concurrent.futures import Future, ThreadPoolExecutor

from pydtnn import comms
from pydtnn.comms.grpc import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_MAX, UUID_NIL
from pydtnn.comms import ConnectionData, ConnectionState, ResourceClosed, Message


__all__ = (
    "Client",
)


# Sentinel objects
ARG_MISSING = object()


class Client(Protocol):
    """gRPC client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._state = ConnectionData()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")

        self._get_grpc = SimpleQueue()
        self._put_grpc = threading.Event()

        # gRPC
        config: abc.MutableMapping = {
            "target": f"{self._addr}:{self._port}",
            "options": self._options,
            "compression": self._compression
        }

        if comms.SSL:
            config["credentials"] = grpc.ssl_channel_credentials(root_certificates=comms.SSL_CERT.read_bytes() if comms.SSL_CERT else None)
            self._channel = grpc.secure_channel(**config)
        else:
            self._channel = grpc.insecure_channel(**config)

        self._client = True
        self._com = self._channel.stream_stream(method="/grpc/com", request_serializer=bytes, response_deserializer=lambda x: x)

        self._session_ini()

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _put(self, stream: Stream) -> Future[None]:
        """Put stream into queue and notify"""
        future = self._state.put(stream)
        self._submit(self._c2s)
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to server"""
        assert len(peers) == 0, "Client can not publish to another client"
        stream = self._serializer.dump(obj)
        return self._put(stream)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get from the server"""
        assert len(peers) == 0, "Client can not get from another client"

        try:
            peer = self._get_event.get_nowait()
        except Empty:
            if not hasattr(self, "_client"):
                raise ResourceClosed()
            self._submit(self._s2c)
            peer = self._get_event.get()
            self._get_grpc.put(None)

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=state.peer, obj=obj)

    def _get_flush(self) -> None:
        state = self._state
        peer = state.peer

        while True:
            try:
                stream = state.get()
            except BlockingIOError:
                break

            if stream.empty():
                self._handle_session_fin(stream)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _handle_connection(self) -> None:
        """Communication round"""
        state = self._state

        for data in self._m2d(self._com(self._s2m(state))):
            state.get_buffer.write(data)
            self._get_flush()

        if not state.state and state.put_empty():
            self._fin()

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

    def _c2s(self) -> None:
        """Communication client to server"""
        # Check if already handled
        state = self._state
        if state.put_empty():
            return
        self._handle_connection()

    def _s2c(self) -> None:
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
            self._handle_connection()
        self._get_grpc.get_nowait()

    def _fin(self) -> None:
        """Communication finalization"""
        with self._lock:
            del self._client
            self._lock.notify_all()

    def _session_ini(self) -> None:
        """Send session ini message"""
        state = self._state
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        self.put(self._id)

    def _session_fin(self) -> None:
        """Send session fin message"""
        state = self._state
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        self._put(Stream())

    def _handle_session_ini(self, stream: Stream) -> None:
        """Handle session initialize message"""
        state = self._state
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

    def _handle_session_fin(self, stream: Stream) -> None:
        """Handle session finalize message"""
        state = self._state
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"

        state.state &= ~ConnectionState.READABLE
        with self._lock:
            self._lock.notify_all()

    def _close(self) -> None:
        """Close the client"""
        self._session_fin()

        while self._state.state:
            self._submit(self._s2c).result()

        with self._lock:
            while hasattr(self, "_client"):
                self._lock.wait()

        self._pool.shutdown()
        self._channel.close()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        super()._close()
