"""gRPC server"""

import uuid
import grpc
import threading
import traceback
from collections import abc
from queue import SimpleQueue
from concurrent.futures import Future, ThreadPoolExecutor

from bidict import bidict

from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_MAX, UUID_NIL
from pydtnn.utils.asynctools import merge_futures
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc
from pydtnn.comms import ConnectionData, ConnectionState, ResourceClosed, Message


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
        self._get_event = SimpleQueue[uuid.UUID]()
        self._peers = bidict[uuid.UUID, str]()
        self._state = dict[uuid.UUID, ConnectionData]()

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

    def _com(self, messages: abc.Iterable[grpc_pb2.Message], context: grpc.ServicerContext) -> abc.Iterable[grpc_pb2.Message]:
        try:
            yield from self._handle_connection(messages, context)
        except Exception as exc:
            traceback.print_exception(exc)
            context.set_code(grpc.StatusCode.INTERNAL)

    def _new_connection(self, sock: str) -> uuid.UUID:
        """Handle new incomming connections"""
        # NOTE: communication thead
        peer = uuid.uuid4()  # temporary ID

        with self._lock:
            self._peers[peer] = sock
            self._state[peer] = ConnectionData(buffer_size=self._max_data_size)
            self._lock.notify_all()

        # ACK
        self._session_ini(peer)

        return peer

    def _get_flush(self, peer: uuid.UUID):
        state = self._state[peer]

        while True:
            try:
                stream = state.get()
            except BlockingIOError:
                break

            if stream.empty():
                self._handle_session_fin(peer, stream)
                self._session_fin(peer)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(peer, stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _handle_connection(self, messages: abc.Iterable[grpc_pb2.Message], context: grpc.ServicerContext) -> abc.Iterable[grpc_pb2.Message]:
        """Client to server communication"""
        # NOTE: communication thread
        sock = context.peer()
        try:
            peer = self._peers.inverse[sock]
        except KeyError:
            peer = self._new_connection(sock)
        state = self._state[peer]

        balance = 0

        def get_generator():
            nonlocal balance
            for data in self._m2d(messages):
                yield data
                balance += 1

        def put_generator():
            nonlocal balance
            for message in self._s2m(state):
                yield message
                balance -= 1

        # Message streaming
        put_messages = put_generator()
        for data in get_generator():

            # Get messages
            state.get_buffer.write(data)
            self._get_flush(peer)
            peer = state.peer

            # Publish messages
            if balance <= 0:
                continue
            for message in put_messages:
                yield message
                if balance <= 0:
                    break

        # Drain remainder queue
        yield from put_messages

        if not state.state and state.put_empty():
            self._fin(peer)

    def _fin(self, peer: uuid.UUID) -> None:
        """Close connection"""

        # Remove peer
        with self._lock:
            del self._peers[peer]

            # TODO: reuse peer_cleanup
            if self._state[peer].get_empty():
                del self._state[peer]

            self._lock.notify_all()

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        """Remove finalized drained peer"""
        state = self._state[peer]

        if peer not in self._peers and state.get_empty():
            with self._lock:
                if peer not in self._peers and state.get_empty():
                    del self._state[peer]

    def _session_ini(self, peer: uuid.UUID) -> None:
        """Send session ini message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        self.put(self._id, peer)

    def _session_fin(self, peer: uuid.UUID) -> None:
        """Send session fin message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        self._put(Stream(), peer)

    def _handle_session_ini(self, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session initialize message"""
        sock = self._peers[peer]
        state = self._state[peer]
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

        # New ID, move state from tmp ID
        if id not in self._peers:
            with self._lock:
                self._state[id] = state = self._state.pop(peer)

        # Change socket ID association
        with self._lock:
            self._peers.inverse[sock] = id

    def _handle_session_fin(self, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session finalize message"""
        state = self._state[peer]
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"
        stream.close()
        state.state &= ~ConnectionState.READABLE

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from a client"""
        # NOTE: peers could be missing or disconnect creating infinite wait, which is an expected state during startup
        assert len(peers) == 0, "Server can not get from specific client"
        peer = self._get_event.get()

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state[peer]
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        self._peer_cleanup(peer)

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=peer, obj=obj)

    def _put(self, stream: Stream, peer: uuid.UUID) -> Future[None]:
        """Put stream into queue and notify"""
        try:
            state = self._state[peer]
            future = state.put(stream)
        except (KeyError, ResourceClosed):
            raise ResourceClosed(peer)
        return future

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to clients"""
        if not peers:
            with self._lock:
                peers = tuple(self._peers)

        futures = list[Future[None]]()
        errors = list[ResourceClosed]()
        with self._serializer.dump(obj) as stream:
            for peer in peers:
                try:
                    future = self._put(stream.copy(), peer)
                except ResourceClosed as exc:
                    errors.append(exc)
                    continue
                else:
                    futures.append(future)

        if errors:
            raise ExceptionGroup("Peer does not exist", errors)

        return merge_futures(futures)

    def _close(self) -> None:
        """Close the server"""

        # Wait peers to drain
        with self._lock:
            while self._peers:
                self._lock.wait()

        # Unlock inflight external API
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        # Close resources
        # Allow some time for RPC taredown
        self._server.stop(grace=0.5)
        self._pool.shutdown()
        super()._close()
