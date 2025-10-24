import enum
import importlib
import typing

from pydtnn import model as model_module


class BackendType(enum.StrEnum):
    CPU = enum.auto()
    GPU = enum.auto()


class PromoteToBackend:
    _backend: typing.Self

    def __new__(cls, *args, **kwds):
        # Save top-level constructor arguments
        self = super().__new__(cls)
        self._backend_new = (args, kwds)  # type: ignore
        return self

    def __getattribute__(self, name: str):
        ref = "_backend"

        # Get backend
        try:
            backend = super().__getattribute__(ref)
        except AttributeError:
            backend = None

        # Skip backend if internal
        if backend is None or ref in name:
            return super().__getattribute__(name)
        else:
            return getattr(backend, name)

    def set_backend(self, backend: BackendType) -> None:
        # Clear backend
        try:
            del self._backend
        except AttributeError:
            pass

        # Get backend class
        cls = self.__class__
        module_name = cls.__module__.split(".", 1)[1]
        backend_module_name = f"pydtnn.backends.{backend}.{module_name}_{backend}"
        backend_module = importlib.import_module(backend_module_name)
        cls_name = f"{cls.__name__}{backend.upper()}"
        cls = getattr(backend_module, cls_name)

        # Create backend instance
        args, kwds = self._backend_new
        self._backend = cls(*args, **kwds)

    # Base class
    model: "model_module.Model"

    def set_model(self, model: "model_module.Model") -> None:
        self.model = model
