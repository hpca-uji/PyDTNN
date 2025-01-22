"""MQTT communication"""

from queue import SimpleQueue
from pydtnn.comms.mqtt import Base
import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client


__all__ = (
    "Server",
    "Client"
)


class Server(Base):
    """MQTT server"""

    def __init__(self) -> None:
        """Server initialization"""
        super().__init__()

        # State
        self._queue = SimpleQueue[mqtt_client.MQTTMessage]()

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )
        self._client.on_message = self._handle
        self._client.connect(host=self._addr, port=self._port)
        self._client.subscribe(topic="server", qos=self._qos)
        self._client.loop_start()

    def _handle(self, client: mqtt_client.Client, userdata, msg: mqtt_client.MQTTMessage) -> None:
        """Client message handler"""
        self._queue.put(msg)

    def get(self):
        """Get data from a client"""
        msg = self._queue.get()
        obj = self._deserialize(msg.payload)
        return obj

    def put(self, obj) -> None:
        """Publish data to clients"""
        data = self._serialize(obj)
        self._client.publish(topic="client", payload=data, qos=self._qos)

    def close(self) -> None:
        """Close the server"""
        self._client.loop_stop()
        super().close()


class Client(Base):
    """MQTT client"""

    def __init__(self) -> None:
        """Client initialization"""
        super().__init__()

        # State
        self._queue = SimpleQueue[mqtt_client.MQTTMessage]()

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )
        self._client.on_message = self.handle
        self._client.connect(host=self._addr, port=self._port)
        self._client.subscribe(topic="client", qos=self._qos)
        self._client.loop_start()

    def handle(self, client: mqtt_client.Client, userdata, msg: mqtt_client.MQTTMessage) -> None:
        """Server message handler"""
        self._queue.put(msg)

    def put(self, obj) -> None:
        """Publish data to server"""
        data = self._serialize(obj)
        self._client.publish(topic="server", payload=data, qos=self._qos)

    def get(self):
        """Get server data"""
        msg = self._queue.get()
        obj = self._deserialize(msg.payload)
        return obj

    def close(self) -> None:
        """Close the client"""
        self._client.loop_stop()
        super().close()
