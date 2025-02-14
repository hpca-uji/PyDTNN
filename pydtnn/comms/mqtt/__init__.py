"""MQTT communications"""

# NOTE: Module considerations
#
# MQTT broker implementations are not common, so the server provided here is
# actually another client. Therefore the address and port provided to both,
# the client and server, should be the one of the actual broker, not where this
# server is running.
#
# Low level comunications are handled single-threaded and are limited to pushing
# or pulling data to queues without blocking, so all operations are minimal
# and fast.
#
# Expensive operations, such as serialization and blocking, are done at at the
# public's API callers thread.

# TODO: Peer-specific and global comunications are topic optimized, but peer groups
# are not. This could be implemented using grouping requests that generate new
# UUID per group. This would reduce also reduce load on the broker.

import uuid

import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client

from pydtnn import comms


__all__ = (
    "Protocol",
)


# Sentinel object
ARG_MISSING = object()


class Protocol(comms.Communication):
    """Shared base MQTT implementation"""

    _qos = 0
    _transport = "tcp"
    _protocol = mqtt_client.MQTTv5

    def __init__(self, addr: str, port: int) -> None:
        """Communication initialization"""
        super().__init__(addr, port)

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )
        self._client.connect(host=self._addr, port=self._port)
        self._client.loop_start()

    def _register_handler(self, topic: str, handler: mqtt_client.CallbackOnMessage) -> None:
        """Setup a topic handler"""
        self._client.message_callback_add(sub=topic, callback=handler)
        self._client.subscribe(topic=topic, qos=self._qos)

    def _peer(self, msg: mqtt_client.MQTTMessage) -> uuid.UUID:
        """Get peer from a mesage"""
        return uuid.UUID(msg.topic.split("/", 1)[1])

    def _publish(self, topic: str, data=None) -> None:
        """Generic MQTT publish"""
        self._client.publish(topic=topic, payload=data, qos=self._qos)

    def close(self) -> None:
        """Close the connection"""
        if self.closed:
            return  # Ignore multiple closes
        self._client.loop_stop()
        super().close()
