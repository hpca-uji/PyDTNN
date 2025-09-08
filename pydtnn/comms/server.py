
import uuid

from pydtnn.comms import Communicator
from pydtnn.utils.io_stream import Stream


class Server[T](Communicator[T]):
    """Base server implementation"""

    def _handle_session_fin(self, peer: uuid.UUID, stream: Stream) -> None:
        super()._handle_session_fin(peer, stream)
        self._session_fin(peer)
