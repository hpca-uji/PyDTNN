
from concurrent.futures import Future
import uuid

from pydtnn.utils.asynctools import merge_futures
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_NIL
from pydtnn.comms import Communicator, CommunicatorOptions, ConnectionData, ResourceClosed


class Client[T](Communicator):
    """Base client implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # State
        self._peers[self._id] = None  # type: ignore
        self._state[self._id] = self._new_state()

    def _get_flush(self) -> None:
        state = self._get_state(self._id)
        peer = state.peer

        for stream in state.get_flush():
            if stream.empty():
                self._handle_session_fin(state, stream)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(state, stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _get_state(self, peer: uuid.UUID) -> ConnectionData:
        return self._state[self._id]

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        pass

    def _extra_fin(self) -> None:
        pass

    def _fin(self) -> None:
        """Communication finalization"""
        with self._lock:
            self._extra_fin()
            self._lock.notify_all()

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

    def _handle_session_fin(self, state: ConnectionData, stream: Stream) -> None:
        """Handle session finalize message"""
        super()._handle_session_fin(state, stream)

        with self._lock:
            self._lock.notify_all()
