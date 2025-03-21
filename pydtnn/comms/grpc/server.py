"""gRPC server"""

import uuid
import grpc
import threading
from collections import abc
from queue import SimpleQueue
from concurrent.futures import ThreadPoolExecutor

from bidict import bidict

from pydtnn.utils.io_stream import StreamSerializer
from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc


__all__ = (
    "Server",
)


# Sentinel objects
END_COMM = None


class Server(Protocol):
    """gRPC server"""

    def __init__(self, addr: str, port: int) -> None:
        """Server initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._peers = bidict[uuid.UUID, str]()
        self._queue = SimpleQueue[uuid.UUID]()
        self._put_queue = bidict[uuid.UUID, SimpleQueue]()
        self._get_queue = bidict[uuid.UUID, SimpleQueue]()

        # gRPC
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")
        self._server = grpc.server(
            thread_pool=self._pool,
            compression=self._compression,
            options=self._options
        )
        grpc_pb2_grpc.add_gRPCServicer_to_server(servicer=self, server=self._server)
        self._server.add_insecure_port(address=f"{self._addr}:{self._port}")
        self._server.start()

    def _peer(self, context: grpc.ServicerContext) -> uuid.UUID:
        """Get peer from a context"""
        grpc_peer = context.peer()
        return self._peers.inverse[grpc_peer]

    def _ini(self, messages: abc.Iterable[grpc_pb2.Message], context: grpc.ServicerContext) -> abc.Iterable[grpc_pb2.Message]:
        """Client connection startup"""
        # NOTE: communication thread
        if self.closed:
            context.set_code(grpc.StatusCode.ABORTED)
            return

        serializer = StreamSerializer()

        # Get peer
        grpc_peer = context.peer()
        message, = messages
        serializer.write(message.data)
        peer = serializer.load()

        # Thread-safe client setup
        with self._lock:
            if peer not in self._peers:
                self._put_queue[peer] = SimpleQueue()
                self._get_queue[peer] = SimpleQueue()
            self._peers[peer] = grpc_peer
            self._lock.notify_all()

        # Send server identification
        size = serializer.dump(self._id)
        with serializer.read(size) as view:
            yield grpc_pb2.Message(data=view.tobytes())

    def _fin(self, messages: abc.Iterable[grpc_pb2.Message], context: grpc.ServicerContext) -> abc.Iterable[grpc_pb2.Message]:
        """Client connection finalizer"""
        # NOTE: communication thread
        peer = self._peer(context)

        # Drain queues
        yield from self._com(messages, context)

        # Thread-safe client taredown
        with self._lock:
            del self._peers[peer]
            del self._put_queue[peer]
            if self._get_queue[peer].empty():
                del self._get_queue[peer]
            self._lock.notify_all()

    def _com(self, messages: abc.Iterable[grpc_pb2.Message], context: grpc.ServicerContext) -> abc.Iterable[grpc_pb2.Message]:
        """Client to server communication"""
        # NOTE: communication thread
        peer = self._peer(context)
        put_queue = self._put_queue[peer]
        get_queue = self._get_queue[peer]

        # Message generators
        put_queue = self._consume_queue(put_queue)
        balance = 0

        def get_generator():
            nonlocal balance
            for message in messages:
                yield message
                balance += 1

        def put_generator():
            nonlocal balance
            for message in self._o2m(put_queue):
                yield message
                balance -= 1

        # Message streaming
        put_messages = put_generator()
        for obj in self._m2o(get_generator()):

            # Get messages
            get_queue.put(obj)
            self._queue.put(peer)

            # Publish messages
            if balance <= 0:
                continue
            for message in put_messages:
                yield message
                if balance <= 0:
                    break

        # Drain remainder queue
        yield from put_messages

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        super().get(*peers)
        assert len(peers) == 0, "Server can not get from specific client"

        peer = self._queue.get()

        # Exit signaled
        if peer == self._id:
            raise ResourceClosed()

        # Get response
        get_queue = self._get_queue[peer]
        obj = get_queue.get_nowait()

        # Cleanup dead queues
        if get_queue.empty():
            with self._lock:
                if peer not in self._peers and get_queue.empty():
                    del self._get_queue[peer]

        return Message(peer=peer, obj=obj)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to clients"""
        super().put(obj, *peers)

        # Get peers if not given
        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        # Queue to as many peers as plausible
        errors = list[uuid.UUID]()
        with self._lock:
            for peer in peers:
                if queue := self._put_queue.get(peer):
                    queue.put(obj)
                else:
                    errors.append(peer)

        # Check for errors
        if errors:
            raise ResourceClosed(errors)

    def close(self) -> None:
        """Close the server"""
        if self.closed:
            return
        super().close()

        # Unlock inflight external API
        for _ in range(threading.active_count()):
            self._queue.put(self._id)

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()
                break

        # Close resources
        # Allow some time for RPC taredown
        self._server.stop(grace=0.5)
        self._pool.shutdown()
