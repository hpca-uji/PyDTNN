"""gRPC communication"""

# NOTE: Module considerations
#
# gRPC does not conform well to a async send & async receive model,
# it expexts remote procedure calls to be recived, processed and responded.
# To simulate this model we created a send procedure and a receive procedure.
# Sent data is queued at the server, recived data is polled until available.
#
# It is important to not hold the prodedures for long, since a receive might
# be waiting on a send, but the send can not be processed if all threads are
# are blocked on receive prodedures.
#
# Polling is implemented with a exponential backoff time and a limit provided
# by the server. The gRPC library implementation queues requests, so requests
# would be replyed in a timely maner, but we do not want to hogh the CPU or
# network with usesless requests.
#
# Low level comunications are handled single-threaded and are limited to pushing
# or pulling data to queues without blocking, so all operations are minimal
# and fast.
#
# Expensive operations, such as serialization and blocking, are done at at the
# public's API callers thread.

import time
import math
import uuid
import grpc
import typing
import threading
from queue import Empty, SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc


__all__ = (
    "Server",
    "Client"
)


# Sentinel objects
ARG_MISSING = object()
END_COMM = b""


class Server(Protocol):
    """gRPC server"""
    _shutdown_grace = 15.0

    def __init__(self) -> None:
        """Server initialization"""
        super().__init__()

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
        self._server.add_insecure_port(address=self._netloc)
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
        super().close()
        self._server.stop(grace=self._shutdown_grace)
        for queue in self._requests.values():
            queue.put(END_COMM)
        self._pool.shutdown()


class Client(Protocol):
    """gRPC client"""
    _backoff_initial_exponent = -10

    def __init__(self) -> None:
        """Client initialization"""
        super().__init__()

        self._channel = grpc.insecure_channel(
            target=self._netloc,
            compression=self._compression
        )
        self._client = grpc_pb2_grpc.gRPCStub(self._channel)
        self._server: uuid.UUID = self._call("_syc", obj=self.id)

    def _call(self, method: str, obj=ARG_MISSING):
        """Generic gRPC call"""
        handler = getattr(self._client, method)
        data = None if obj is ARG_MISSING else self._serialize(obj)
        request = grpc_pb2.Message(data=data)
        response: grpc_pb2.Message = handler(request)
        obj = None if not response.data else self._deserialize(response.data)
        return typing.cast(typing.Any, obj)  # not inferred my typecheker

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        self._call("_c2s", obj=obj)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get server data"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"
        backoff_exponet = self._backoff_initial_exponent
        while True:
            try:
                obj = self._call("_s2c")

            except grpc.RpcError as exc:
                # No response, retry later
                if exc.code() is grpc.StatusCode.UNAVAILABLE:  # type: ignore (incorrect 3-party typing)
                    max_backoff = int(exc.details())  # type: ignore (incorrect 3-party typing)
                    backoff = 2 ** backoff_exponet
                    if backoff >= max_backoff:
                        backoff = max_backoff
                        backoff_exponet = math.ceil(math.log2(max_backoff))
                    else:
                        backoff_exponet += 1
                    time.sleep(backoff)
                    continue

            except Exception:
                # Communication closed
                if self.closed:
                    raise ResourceClosed() from None

                # Communication error
                else:
                    raise
            else:
                break

        return Message(obj=obj, peer=self._server)

    def close(self) -> None:
        """Close the client"""
        super().close()
        self._call("_fin")
        self._channel.close()
