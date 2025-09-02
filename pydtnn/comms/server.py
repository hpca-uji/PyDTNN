
import uuid
from concurrent.futures import Future

from bidict import bidict
from pydtnn.utils import UUID_NIL
from pydtnn.utils.asynctools import merge_futures
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import Communicator, CommunicatorOptions, ConnectionData, ResourceClosed


class Server[T](Communicator):
    """Base server implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # State
        self._peers = bidict[uuid.UUID, T]()
        self._state = dict[uuid.UUID, ConnectionData]()

    def _get_peer(self, sock: T) -> uuid.UUID:
        try:
            return self._peers.inverse[sock]
        except KeyError:
            return self._new_connection(sock)

    def _get_state(self, peer: uuid.UUID) -> ConnectionData:
        return self._state[peer]

    def _extra_ini(self, peer: uuid.UUID) -> None:
        pass

    def _new_connection(self, sock: T) -> uuid.UUID:
        """Handle new incomming connections"""
        # NOTE: communication thead
        peer = uuid.uuid4()  # temporary ID

        with self._lock:
            self._peers[peer] = sock
            self._state[peer] = state = self._new_state()
            self._extra_ini(peer)

            # ACK
            peer = self._peers.inverse[sock]
            state = self._state[peer]
            self._put(self._session_ini(state), peer)
            self._lock.notify_all()

        return peer

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

    def _get_flush(self, peer: uuid.UUID) -> uuid.UUID:
        state = self._get_state(peer)

        for stream in state.get_flush():
            if stream.empty():
                self._handle_session_fin(state, stream)
                self._put(self._session_fin(state), peer)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(state, peer, stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

        return peer

    def _handle_session_ini(self, state: ConnectionData, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session initialize message"""
        sock = self._peers[peer]

        super()._handle_session_ini(state, stream)
        id = state.peer

        # New ID, move state from tmp ID
        if id not in self._peers:
            with self._lock:
                self._state[id] = state = self._state.pop(peer)

        # Change socket ID association
        with self._lock:
            self._peers.inverse[sock] = id

    def _extra_fin(self, peer: uuid.UUID) -> None:
        del self._peers[peer]

        # TODO: reuse peer_cleanup
        if self._state[peer].get_empty():
            del self._state[peer]

    def _fin(self, peer: uuid.UUID) -> None:
        """Close connection"""
        with self._lock:
            self._extra_fin(peer)
            self._lock.notify_all()

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        """Remove finalized drained peer"""
        state = self._state[peer]

        if peer not in self._peers and state.get_empty():
            with self._lock:
                if peer not in self._peers and state.get_empty():
                    del self._state[peer]
