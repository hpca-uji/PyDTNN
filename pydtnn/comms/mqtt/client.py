"""MQTT client"""

import uuid
from queue import SimpleQueue

import paho.mqtt.client as mqtt_client

from pydtnn.comms.mqtt import Protocol
from pydtnn.comms import ResourceClosed, Message


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = b""


class Client(Protocol):
    """MQTT client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._responses = SimpleQueue[bytes]()

        # MQTT
        self._register_handler(topic="s2c", handler=self._s2c)
        self._register_handler(topic=f"s2c/{self.id}", handler=self._s2c)
        self._syc()

    def _syc(self) -> None:
        self._publish(topic=f"syc/{self.id}")
        data = self._responses.get()
        self.server = self._deserialize(data)

    def _s2c(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        """Server message handler"""
        data = message.payload
        self._responses.put(data)

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        data = self._serialize(obj)
        self._publish(topic=f"c2s/{self.id}", data=data)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get server data"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"
        data = self._responses.get()

        # Exit signaled
        if data == END_COMM:
            raise ResourceClosed()

        obj = self._deserialize(data)
        return Message(peer=self.server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        self._publish(topic=f"fin/{self.id}")
        super().close()
        self._responses.put(END_COMM)
