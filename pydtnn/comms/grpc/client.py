"""gRPC client"""

import uuid
import grpc
import threading
from collections import abc
from queue import SimpleQueue, Empty
from concurrent.futures import ThreadPoolExecutor

from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.grpc import Protocol, grpc_pb2_grpc


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
        self._get_queue = SimpleQueue()
        self._put_queue = SimpleQueue()
        self._get_event = SimpleQueue()
        self._put_event = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")

        # gRPC
        self._channel = grpc.insecure_channel(
            target=f"{self._addr}:{self._port}",
            compression=self._compression,
            options=self._options
        )
        self._client = grpc_pb2_grpc.gRPCStub(self._channel)
        self._submit(self._ini).result()

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        self._put_queue.put(obj)
        self._put_event.set()
        self._submit(self._c2s)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from the server"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"

        try:
            obj = self._get_queue.get_nowait()
        except Empty:
            if hasattr(self, "_client"):
                self._submit(self._s2c)
                obj = self._get_queue.get()
                self._get_event.put(None)
            else:
                raise ResourceClosed()

        return Message(peer=self._server, obj=obj)

    @staticmethod
    def _new_backoff(start=-10, end=1) -> abc.Generator[float]:
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

    def _com(self, method: str = "_com") -> None:
        """Communication round"""
        func = getattr(self._client, method)
        put_queue = self._consume_queue(self._put_queue)
        for obj in self._m2o(func(self._o2m(put_queue))):
            self._get_queue.put(obj)

    def _ini(self) -> None:
        """Communication inizialization"""
        self._put_queue.put(self._id)
        self._com(method="_ini")
        self._server = self._get_queue.get_nowait()

    def _fin(self) -> None:
        """Communication finalization"""
        self._com(method="_fin")
        self._channel.close()
        del self._client

    def _c2s(self) -> None:
        """Communication client to server"""
        # Check if already handled
        if self._put_queue.empty():
            return
        self._com()

    def _s2c(self) -> None:
        """Communication server to client"""
        # Check if already handled
        try:
            self._get_event.get_nowait()
        except Empty:
            pass
        else:
            return

        # Handle recive loop
        backoff = None
        while self._get_event.empty():
            if not hasattr(self, "_client"):
                raise ResourceClosed()
            elif backoff:
                self._put_event.wait(next(backoff))
                self._put_event.clear()
            else:
                backoff = self._new_backoff()
            self._com()
        self._get_event.get_nowait()

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        super().close()

        self._submit(self._fin).result()
        self._pool.shutdown()
