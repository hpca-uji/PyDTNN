"""gRPC client"""

import uuid
import grpc
import typing

from pydtnn.comms import ResourceClosed, Message
from pydtnn.comms.grpc import Protocol, grpc_pb2, grpc_pb2_grpc


__all__ = (
    "Client",
)


# Sentinel objects
ARG_MISSING = object()


class Client(Protocol):
    """gRPC client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        self._channel = grpc.insecure_channel(
            target=f"{self._addr}:{self._port}",
            compression=self._compression
        )
        self._client = grpc_pb2_grpc.gRPCStub(self._channel)
        self.server: uuid.UUID = self._call("_syc", obj=self.id)

    def _call(self, method: str, obj=ARG_MISSING):
        """Generic gRPC call"""
        handler = getattr(self._client, method)
        data = None if obj is ARG_MISSING else self._serialize(obj)
        request = grpc_pb2.Message(data=data)
        response: grpc_pb2.Message = handler(request)
        obj = None if not response.data else self._deserialize(response.data)
        return typing.cast(typing.Any, obj)  # not inferred my typecheker

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        self._call("_c2s", obj=obj)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from the server"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"

        # Bootstrap backoff generator
        backoff = self._new_backoff()
        next(backoff)

        try:
            while True:
                try:
                    obj = self._call("_s2c")

                except grpc.RpcError as exc:
                    # No response, retry later
                    if exc.code() is grpc.StatusCode.UNAVAILABLE:  # type: ignore (incorrect 3-party typing)
                        exc_details = exc.details()  # type: ignore (incorrect 3-party typing)
                        try:
                            max_backoff = int(exc_details)
                        except ValueError:
                            pass
                        else:
                            backoff.send(max_backoff)
                            continue
                    raise

                else:
                    break

        except Exception:
            # Communication closed
            if self.closed:
                raise ResourceClosed() from None

            # Communication error
            else:
                raise

        return Message(peer=self.server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        super().close()
        self._call("_fin")

        # Close resources
        self._channel.close()
