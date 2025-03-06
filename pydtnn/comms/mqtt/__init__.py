"""MQTT communications"""

# NOTE: MQTT broker implementations are not common, so the server provided here
# is actually another client. Therefore the address and port provided to both,
# the client and server, should be the one of the actual broker, not where the
# server is running.

# NOTE: The server can not detect when persistent clients clear their session.
# If a fin message is not sent, clients data and queues are held indefinitely.
#
# NOTE: The MQTT library handles comunications single-threaded, therefore
# operations on related callbacks are limited to pushing or pulling data from
# queues without blocking, so all operations are minimal and fast.

# TODO: Peer-specific and global comunications are topic optimized, but peer groups
# are not. This could be implemented using grouping requests that generate new
# UUID per group. This would reduce also reduce load on the broker.

from concurrent.futures import ThreadPoolExecutor
import uuid
import threading

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

        # State
        self._inflight = threading.Semaphore(value=0)

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")
        self._client.connect(host=self._addr, port=self._port)

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _start_loop(self) -> None:
        """Start connection handling loop"""
        self._submit(self._client.loop_forever)

    def _register_handler(self, topic: str, handler: mqtt_client.CallbackOnMessage) -> None:
        """Setup a topic handler"""
        self._client.message_callback_add(sub=topic, callback=handler)
        self._client.subscribe(topic=topic, qos=self._qos)

    def _peer(self, message: mqtt_client.MQTTMessage) -> uuid.UUID:
        """Get peer from a mesage"""
        return uuid.UUID(message.topic.split("/", 1)[1])

    def _publish(self, topic: str, data=None) -> None:
        """Generic MQTT publish"""
        self._inflight.release()
        message = self._client.publish(topic=topic, payload=data, qos=self._qos)
        self._submit(message.wait_for_publish)
