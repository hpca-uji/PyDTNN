import resource
import sys
import abc

from typing import TYPE_CHECKING

from pydtnn.tracers.events import PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS, PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, PYDTNN_MDL_EVENT_enum, PYDTNN_OPS_EVENT_enum
from pydtnn.utils import find_component

if TYPE_CHECKING:
    from pydtnn.model import Model
    from pydtnn.layers.layer import Layer


class EventType:
    """
    EventType container
    """

    def __init__(self, name):
        self.name = name
        self._events = {}

    def __getitem__(self, item):
        try:
            description = self._events[item]
        except KeyError:
            sys.stderr.write(f"SimpleTracer warning: No event with code '{item}' "
                             f"in the '{self.name}' type of events.\n")
            return f"Unknown code {self.name}"
        return description

    def __setitem__(self, value, description):
        self._events[value] = description

    def __len__(self):
        return len(self._events)

    def items(self):
        return self._events.items()


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Tracer(metaclass=PostInitCaller):
    """
    Tracer base class
    """

    def __init__(self, tracing: bool):
        self.event_types = {
            PYDTNN_MDL_EVENT: EventType("Model"),
            PYDTNN_OPS_EVENT: EventType("Operations"),
        }
        self.tracing = tracing

    def __post_init__(self):
        """
        This method will be called AFTER all the derived classes __init__ methods are completed.
        By proceeding in this way, when the derived classes enable/disable methods are called, all the attributes
        they require will already have been defined on their corresponding __init__ methods.
        """
        if self.tracing:
            self.enable_tracing()
            self.enable_print_memory_usage()
        else:
            self.disable_tracing()
            self.disable_print_memory_usage()

    def enable_tracing(self):
        """Actions that must be done if tracing is enabled"""
        setattr(self, "define_event_types", self._define_event_types)
        setattr(self, "emit_event", self._emit_event)
        setattr(self, "emit_nevent", self._emit_nevent)

    def disable_tracing(self):
        """Actions that must be done if tracing is disabled"""
        setattr(self, "define_event_types", lambda *args, **kwargs: None)
        setattr(self, "emit_event", lambda *args, **kwargs: None)
        setattr(self, "emit_nevent", lambda *args, **kwargs: None)

    def enable_print_memory_usage(self):
        """Actions that must be done if print memory usage is enabled"""
        setattr(self, "print_memory_usage", self._print_memory_usage)

    def disable_print_memory_usage(self):
        """Actions that must be done if print memory usage is disabled"""
        setattr(self, "print_memory_usage", lambda *args, **kwargs: None)

    def define_event_types(self, model: "Model"):
        """Fake method, will be replaced by lambda: None or _define_event_types()"""
        pass

    def emit_event(self, evt_type: int, evt_val: int, stream=None):
        """Fake method, will be replaced by lambda: None or _emit_event()"""
        pass

    def emit_nevent(self, evt_evt: list[int, int], evt_val: list[int, int], stream=None):
        """Fake method, will be replaced by lambda: None or _emit_nevent()"""
        pass

    def print_memory_usage(self, text: str):
        """Fake method, will be replaced by lambda: None or _print_memory_usage()"""
        pass

    def _get_layers_recursively(self, layers: list["Layer"]) -> list["Layer"]:
        all_layers = []
        for layer in layers:
            all_layers.append(layer)
            all_layers += self._get_layers_recursively(layer.children)
        return all_layers

    def _define_event_types(self, model: "Model"):
        """This method will be called only if tracing is enabled"""
        mdl_event = self.event_types[PYDTNN_MDL_EVENT]
        ops_event = self.event_types[PYDTNN_OPS_EVENT]
        mdl_event[0] = "End"
        ops_event[0] = "End"
        mdl_constants = [(event._name_, event._value_) for event in PYDTNN_MDL_EVENT_enum]
        ops_constants = [(event._name_, event._value_) for event in PYDTNN_OPS_EVENT_enum]
        for layer in model.get_all_layers():
            for (name, val) in mdl_constants:
                mdl_event[layer.id * PYDTNN_MDL_EVENTS + val] = f"{layer.name_with_id}_{name[11:].lower()}"
            for (name, val) in ops_constants:
                ops_event[
                    layer.id * PYDTNN_OPS_EVENTS + val] = f"{layer.id:03}_{layer.name}_{name[11:].lower()}"

    @abc.abstractmethod
    def _emit_event(self, evt_type: int, evt_val: int, stream=None):
        """This method will be called only if tracing is enabled"""
        pass

    @abc.abstractmethod
    def _emit_nevent(self, evt_evt: list[int], evt_val: list[int], stream=None):
        """This method will be called only if tracing is enabled"""
        pass

    @staticmethod
    def _print_memory_usage(text=""):
        """This method will be called only if print memory usage is enabled"""
        u = resource.getrusage(resource.RUSAGE_SELF)
        if text != "":
            text = f" {text}:"
        print(f">>>{text} user time={u[0]:.2f}, sys time={u[1]:.2f}, mem={u[2] / 1024:.2f} MiB")

    def set_stream(self, stream):
        pass


def select(name: str) -> type[Tracer]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
