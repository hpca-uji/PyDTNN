
import uuid
import threading
from queue import SimpleQueue

from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_NIL
from pydtnn.comms import Communicator, CommunicatorOptions, ConnectionState


class Client[T](Communicator):
    """Base client implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # State
        self._state = self._new_state()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._lock = threading.Condition()

    def _get_flush(self) -> None:
        state = self._state
        peer = state.peer

        for stream in state.get_flush():
            if stream.empty():
                self._handle_session_fin(stream)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _session_ini(self) -> Stream:
        """Send session ini message"""
        state = self._state
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        return self._serializer.dump(self._id)

    def _session_fin(self) -> Stream:
        """Send session fin message"""
        state = self._state
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        return Stream()

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
