
import uuid

from pydtnn.utils.io_stream import Stream
from pydtnn.comms import Communicator, ConnectionData


class Server[T](Communicator[T]):
    """Base server implementation"""

    def _handle_session_fin(self, state: ConnectionData, peer: uuid.UUID, stream: Stream) -> None:
        super()._handle_session_fin(state, peer, stream)
        self._put(self._session_fin(state), peer)
