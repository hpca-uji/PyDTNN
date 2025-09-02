
from concurrent.futures import Future
import uuid

from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_NIL
from pydtnn.comms import Communicator, CommunicatorOptions, ConnectionData


class Client[T](Communicator):
    """Base client implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # State
        self._state = self._new_state()

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
        return self._state

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        pass

    def _extra_fin(self) -> None:
        pass

    def _fin(self) -> None:
        """Communication finalization"""
        with self._lock:
            self._extra_fin()
            self._lock.notify_all()

    def _put(self, stream: Stream) -> Future[None]:
        """Put stream into queue and notify"""
        return self._state.put(stream)

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to server"""
        assert len(peers) == 0, "Client can not publish to another client"
        stream = self._serializer.dump(obj)
        return self._put(stream)

    def _handle_session_fin(self, state: ConnectionData, stream: Stream) -> None:
        """Handle session finalize message"""
        super()._handle_session_fin(state, stream)

        with self._lock:
            self._lock.notify_all()
