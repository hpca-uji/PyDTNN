"""Communications server package"""

import uuid

from net_queue import Communicator
from net_queue.io_stream import Stream


class Server[T](Communicator[T]):
    """Base server implementation"""

    def _handle_session_fin(self, peer: uuid.UUID, stream: Stream) -> None:
        """Handle session finalize message"""
        super()._handle_session_fin(peer, stream)
        self._session_fin(peer)
