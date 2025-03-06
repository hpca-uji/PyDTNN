"""gRPC server"""

import uuid
import grpc
import threading
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = b""


class Server(Protocol):
    """gRPC server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Lock()
        self._peers = bidict[uuid.UUID, str]()
        self._request_queue = SimpleQueue[uuid.UUID]()
        self._requests = dict[uuid.UUID, SimpleQueue[bytes]]()
        self._responses = dict[uuid.UUID, SimpleQueue[bytes]]()

        # gRPC
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")
        self._server = grpc.server(
            thread_pool=self._pool,
            compression=self._compression
        )
        grpc_pb2_grpc.add_gRPCServicer_to_server(servicer=self, server=self._server)
        self._server.add_insecure_port(address=f"{self._addr}:{self._port}")
        self._server.start()

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    @property
    def _size(self) -> int:
        """Get the approximate number of clients"""
        return len(self._peers)

    def _peer(self, context: grpc.ServicerContext) -> uuid.UUID:
        """Get peer from a context"""
        grpc_peer = context.peer()
        return self._peers.inverse[grpc_peer]

    def _syc(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client connection startup"""
        # NOTE: communication thead
        if self.closed:
            context.set_code(grpc.StatusCode.ABORTED)
            return grpc_pb2.Message()

        grpc_peer = context.peer()
        peer = self._deserialize(message.data)

        # Thread-safe client setup
        with self._lock:
            self._peers[peer] = grpc_peer
            self._requests[peer] = SimpleQueue()
            self._responses[peer] = SimpleQueue()

        # Send server identification
        data = self._serialize(self.id)
        return grpc_pb2.Message(data=data)

    def _fin(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client connection finalizer"""
        # NOTE: communication thead
        peer = self._peer(context)

        # Thread-safe client taredown
        with self._lock:
            del self._peers[peer]
            requests = self._requests.pop(peer)
            responses = self._responses.pop(peer)

        # Drain queue
        while requests:
            try:
                request_peer = self._request_queue.get_nowait()
            except Empty:
                break
            if request_peer == peer:
                requests.get_nowait()
            else:
                self._request_queue.put(request_peer)

        return grpc_pb2.Message()

    def _c2s(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client to server communication"""
        # NOTE: communication thead
        peer = self._peer(context)
        data = message.data

        self._requests[peer].put(data)
        self._request_queue.put(peer)

        return grpc_pb2.Message()

    def _s2c(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Server to client communication"""
        # NOTE: communication thead
        peer = self._peer(context)

        # Try reduce responses
        try:
            data = self._responses[peer].get_nowait()

        # Response not found, abort
        except Empty:
            pass

        # Response found, respond
        else:
            return grpc_pb2.Message(data=data)

        # Signal "no response, retry later"
        max_backoff = self._size
        context.set_code(grpc.StatusCode.UNAVAILABLE)
        context.set_details(str(max_backoff))
        return grpc_pb2.Message()

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        super().get(*peers)
        assert len(peers) == 0, "Server can not get from specific client"

        while True:
            # Wait for a request
            peer = self._request_queue.get()

            # Get request
            try:
                data = self._requests[peer].get_nowait()

            # Request not found, revert notification and retry
            except (KeyError, Empty):
                self._request_queue.put(peer)
                continue

            # Request found, continue
            else:
                break

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=peer, obj=obj)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to clients"""
        super().put(obj, *peers)

        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        data = self._serialize(obj)

        for peer in peers:
            self._responses[peer].put(data)

    def close(self) -> None:
        """Close the server"""
        if self.closed:
            return
        super().close()

        # Unlock inflight external API
        with self._lock:
            for queue in self._requests.values():
                queue.put(END_COMM)

        # Bootstrap backoff generator
        backoff = self._new_backoff()
        next(backoff)

        # Wait peers to drain
        while self._peers:
            backoff.send(1.0)

        # Close resources
        self._server.stop(grace=None)
        self._pool.shutdown()
