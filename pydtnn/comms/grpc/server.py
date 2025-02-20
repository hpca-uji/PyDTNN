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
    _shutdown_grace = 15.0

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Lock()
        self._peers = bidict[uuid.UUID, str]()
        self._request_count = threading.Semaphore(value=0)
        self._response_count = threading.Semaphore(value=0)
        self._requests = dict[uuid.UUID, SimpleQueue[bytes]]()
        self._responses = dict[uuid.UUID, SimpleQueue[bytes]]()

        # gRPC
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._server = grpc.server(
            thread_pool=self._pool,
            compression=self._compression
        )
        grpc_pb2_grpc.add_gRPCServicer_to_server(servicer=self, server=self._server)
        self._server.add_insecure_port(address=f"{self._addr}:{self._port}")
        self._server.start()

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

        # Drain queues and update counts
        for _ in range(requests.qsize()):
            self._request_count.acquire()
        for _ in range(responses.qsize()):
            self._response_count.acquire()

        return grpc_pb2.Message()

    def _c2s(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Client to server communication"""
        # NOTE: communication thead
        peer = self._peer(context)
        data = message.data

        self._requests[peer].put(data)
        self._request_count.release()

        return grpc_pb2.Message()

    def _s2c(self, message: grpc_pb2.Message, context: grpc.ServicerContext) -> grpc_pb2.Message:
        """Server to client communication"""
        # NOTE: communication thead
        peer = self._peer(context)

        # Try reduce notifications
        if self._response_count.acquire(blocking=False):

            # Try reduce responses
            try:
                data = self._responses[peer].get_nowait()

            # Response not found, revet notification and abort
            except Empty:
                self._response_count.release()

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

        while True:
            # Wait for a request
            # FIXME: if peers is defined and other peers have messages this is a busy-wait
            self._request_count.acquire()

            # Acquire peers
            if peers:
                _peers = peers
            else:
                with self._lock:
                    _peers = tuple(self._requests)

            # Search for a request
            for peer in _peers:
                try:
                    data = self._requests[peer].get_nowait()
                except (KeyError, Empty):
                    continue
                else:
                    break

            # Request not found, revert notification and retry
            else:
                self._request_count.release()
                continue

            # Request found, continue
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
            self._response_count.release()

    def close(self) -> None:
        """Close the server"""
        if self.closed:
            return
        super().close()
        self._server.stop(grace=self._shutdown_grace)
        with self._lock:
            for queue in self._requests.values():
                queue.put(END_COMM)
        self._pool.shutdown()
