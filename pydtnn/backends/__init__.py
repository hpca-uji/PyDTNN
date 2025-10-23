import enum
import importlib
from contextlib import suppress
import typing

from pydtnn.backends import cpu
from pydtnn.backends import gpu
from pydtnn import model as model_module


class BackendType(enum.StrEnum):
    CPU = enum.auto()
    GPU = enum.auto()


class PromoteToBackend:
    model: "model_module.Model"
    backend: typing.Self

    def __new__(cls, *args, **kwds):
        # Save top-level constructor arguments
        self = super().__new__(cls)
        self._new_ = (args, kwds)  # type: ignore
        return self

    def __getattribute__(self, name: str):
        # Get backend
        try:
            backend = super().__getattribute__("backend")
        except AttributeError:
            backend = None

        if backend is self or backend is None:
            # We are the backend
            return super().__getattribute__(name)
        else:
            # We are the abstract
            return getattr(backend, name)

    def set_backend(self, backend: BackendType) -> None:
        # Clear backend
        try:
            del self.backend
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
        args, kwds = self._new_
        self.backend = cls(*args, **kwds)

    def set_model(self, model: "model_module.Model") -> None:
        self.model = model

    @property
    def canonical_name(self) -> str:
        suffix = ""
        module_submodules = self.__module__.split(".")
        canonical_name = self.__class__.__name__
        for i, submodule in enumerate(module_submodules):
            if submodule == "backends":
                with suppress(IndexError):
                    suffix = module_submodules[i + 1].upper()
                break
        if suffix != "":
            suffix_len = len(suffix)
            if canonical_name[-suffix_len:] == suffix:
                canonical_name = canonical_name[:-suffix_len]
        return canonical_name
