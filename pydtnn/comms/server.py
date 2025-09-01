
import uuid
import threading
from queue import SimpleQueue

from bidict import bidict
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import Communicator, CommunicatorOptions, ConnectionData, ConnectionState


class Server[T](Communicator):
    """Base server implementation"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Server initialization"""
        super().__init__(options)

        # State
        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._peers = bidict[uuid.UUID, T]()
        self._state = dict[uuid.UUID, ConnectionData]()

    def _session_ini(self, peer: uuid.UUID) -> Stream:
        """Send session ini message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        return self._serializer.dump(self._id)

    def _session_fin(self, peer: uuid.UUID) -> Stream:
        """Send session fin message"""
        state = self._state[peer]
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        return Stream()

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

    def _peer_cleanup(self, peer: uuid.UUID) -> None:
        """Remove finalized drained peer"""
        state = self._state[peer]

        if peer not in self._peers and state.get_empty():
            with self._lock:
                if peer not in self._peers and state.get_empty():
                    del self._state[peer]
