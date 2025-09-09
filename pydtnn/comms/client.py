
from uuid import UUID
from concurrent.futures import Future

from pydtnn.comms import Communicator


class Client[T](Communicator[T]):
    """Base client implementation"""

    def put(self, obj, *peers: UUID) -> Future[None]:
        """Publish data to peers"""
        assert len(peers) == 0, "Client can not put to specific peer"
        return super().put(obj, *peers)
