"""
PyDTNN tracer module for monitoring model execution and memory usage.
"""
from __future__ import annotations

import abc
import logging
import resource
import sys
from typing import TYPE_CHECKING

from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.utils import find_component

__all__ = (
    "EventType",
    "PostInitCaller",
    "Tracer",
    "select",
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.abstract.layerable import Layerable
    from pydtnn.model.layers import Layers as Model


class EventType:
    """
    Container for managing and retrieving event descriptions by code.
    """

    def __init__(self, name):
        """
        Initialize the EventType container.

        Args:
            name: The name of the event category.
        """
        self.name = name
        self._events = {}

    def __getitem__(self, item):
        """
        Retrieve the description for a given event code.

        Args:
            item: The event code to look up.

        Returns:
            The description string or an error message if not found.
        """
        try:
            description = self._events[item]
        except KeyError:
            sys.stderr.write(f"SimpleTracer warning: No event with code '{item}' in the '{self.name}' type of events.\n")
            return f"Unknown code {self.name}"
        return description

    def __setitem__(self, value, description):
        """
        Register an event description.

        Args:
            value: The event code.
            description: The description string.
        """
        self._events[value] = description

    def __len__(self):
        """
        Return the number of registered events.
        """
        return len(self._events)

    def items(self):
        """
        Return the items in the event dictionary.
        """
        return self._events.items()


class PostInitCaller(type):
    """
    Metaclass that triggers __post_init__ after object instantiation.
    """
    def __call__(cls, *args, **kwargs):
        """
        Create an instance and call its __post_init__ method.
        """
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Tracer(metaclass=PostInitCaller):
    """
    Base class for implementing model execution and memory tracers.
    """

    def __init__(self, tracing: bool):
        """
        Initialize the tracer.

        Args:
            tracing: Whether tracing is enabled.
        """
        self.event_types = {
            PYDTNN_MDL_EVENT: EventType("Model"),
            PYDTNN_OPS_EVENT: EventType("Operations"),
        }
        self.tracing = tracing

    def __post_init__(self):
        """
        Perform post-initialization setup to enable or disable tracing and memory monitoring.
        """
        # NOTE: This method will be called AFTER all the derived classes __init__ methods are completed.
        # By proceeding in this way, when the derived classes enable/disable methods are called, all the attributes
        # they require will already have been defined on their corresponding __init__ methods.

        if self.tracing:
            self.enable_tracing()
            self.enable_print_memory_usage()
        else:
            self.disable_tracing()
            self.disable_print_memory_usage()

    def enable_tracing(self):
        """Enable tracing methods by binding them to the instance."""
        setattr(self, "define_event_types", self._define_event_types)
        setattr(self, "emit_event", self._emit_event)
        setattr(self, "emit_nevent", self._emit_nevent)

    def disable_tracing(self):
        """Disable tracing methods by binding them to no-op lambdas."""
        setattr(self, "define_event_types", lambda *args, **kwargs: None)
        setattr(self, "emit_event", lambda *args, **kwargs: None)
        setattr(self, "emit_nevent", lambda *args, **kwargs: None)

    def enable_print_memory_usage(self):
        """Enable memory usage printing by binding the method."""
        setattr(self, "print_memory_usage", self._print_memory_usage)

    def disable_print_memory_usage(self):
        """Disable memory usage printing by binding to a no-op lambda."""
        setattr(self, "print_memory_usage", lambda *args, **kwargs: None)

    def define_event_types(self, model: Model):
        """Placeholder for defining event types, replaced at runtime."""
        pass

    def emit_event(self, evt_type: int, evt_val: int, stream=None):
        """Placeholder for emitting a single event, replaced at runtime."""
        pass

    def emit_nevent(self, evt_evt: list[int], evt_val: list[int], stream=None):
        """Placeholder for emitting multiple events, replaced at runtime."""
        pass

    def print_memory_usage(self, text: str):
        """Placeholder for printing memory usage, replaced at runtime."""
        pass

    def _get_layers_recursively(self, layers: list["Layerable"]) -> list["Layerable"]:
        """
        Recursively collect all layers from a list of layerable objects.

        Args:
            layers: A list of layerable objects.

        Returns:
            A flat list of all nested layers.
        """
        all_layers = []
        for layer in layers:
            all_layers.append(layer)
            all_layers += self._get_layers_recursively(layer.children)
        return all_layers

    def _define_event_types(self, model: Model):
        """
        Populate event types based on the provided model structure.

        Args:
            model: The model instance to extract layers from.
        """
        mdl_event = self.event_types[PYDTNN_MDL_EVENT]
        ops_event = self.event_types[PYDTNN_OPS_EVENT]
        mdl_event[0] = "End"
        ops_event[0] = "End"
        mdl_constants = [(event._name_, event._value_) for event in PYDTNN_MDL_EVENT_enum]
        ops_constants = [(event._name_, event._value_) for event in PYDTNN_OPS_EVENT_enum]
        for layer in model.get_all_layers():
            for name, val in mdl_constants:
                mdl_event[layer.id * PYDTNN_MDL_EVENTS + val] = f"{layer.name_with_id}_{name.lower()}"
            for name, val in ops_constants:
                ops_event[layer.id * PYDTNN_OPS_EVENTS + val] = f"{layer.id:03}_{layer.name}_{name.lower()}"

    @abc.abstractmethod
    def _emit_event(self, evt_type: int, evt_val: int, stream=None):
        """Abstract method to emit a single event."""
        pass

    @abc.abstractmethod
    def _emit_nevent(self, evt_evt: list[int], evt_val: list[int], stream=None):
        """Abstract method to emit multiple events."""
        pass

    @staticmethod
    def _print_memory_usage(text=""):
        """
        Log the current process memory usage.

        Args:
            text: Optional prefix text for the log message.
        """
        u = resource.getrusage(resource.RUSAGE_SELF)
        if text != "":
            text = f" {text}:"
        logger.info(f">>>{text} user time={u[0]:.2f}, sys time={u[1]:.2f}, mem={u[2] / 1024:.2f} MiB")

    def set_stream(self, stream):
        """
        Set the output stream for the tracer.

        Args:
            stream: The stream object to use for output.
        """
        pass


def select(name: str) -> type[Tracer]:
    """
    Select and return a tracer class by name.

    Args:
        name: The name of the tracer class to retrieve.

    Returns:
        The tracer class.
    """
    assert __package__, "Package not found!"
    return find_component(__package__, name)