"""MQTT communications"""

import copy

import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client

import net_queue as nq
from net_queue import asynctools
from net_queue.asynctools import thread_queue


__all__ = (
    "Protocol",
)


# Sentinel object
ARG_MISSING = object()


class Protocol[T](nq.Communicator[T]):
    """Shared base MQTT implementation"""

    _qos = 0
    _transport = "tcp"
    _protocol = mqtt_client.MQTTv311

    def __init__(self, options: nq.CommunicatorOptions = nq.CommunicatorOptions()) -> None:
        """Communication initialization"""
        super().__init__(copy.replace(options, workers=1))

        # State
        self._publish_queue = thread_queue(f"{__name__}.{self.__class__.__qualname__}:{id(self)}.publish")

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            client_id=self.id.hex,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )

        if self.options.security:
            self._client.tls_set(ca_certs=str(self.options.security.certificate) if self.options.security.certificate else None)

        self._client.connect(host=self.options.netloc.host, port=self.options.netloc.port)
        self._client.loop_start()

    def _start_loop(self) -> None:
        """Start connection handling loop"""
        self._client.loop_start()

    def _stop_loop(self) -> None:
        self._publish_queue.shutdown()
        self._client.loop_stop()

    def _register_handler(self, topic: str, handler: mqtt_client.CallbackOnMessage) -> None:
        """Setup a topic handler"""
        self._client.message_callback_add(sub=topic, callback=handler)
        self._client.subscribe(topic=topic, qos=self._qos)

    def _peer(self, message: mqtt_client.MQTTMessage) -> str:
        """Get peer from a mesage"""
        return message.topic.split("/", 1)[1]

    def _publish(self, topic: str, data=None) -> None:
        """Generic MQTT publish"""
        message = self._client.publish(topic=topic, payload=data, qos=self._qos)
        self._publish_queue.submit(message.wait_for_publish).add_done_callback(asynctools.future_warn_exception)
