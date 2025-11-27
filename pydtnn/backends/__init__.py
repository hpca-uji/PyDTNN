import enum
import importlib
import typing

if typing.TYPE_CHECKING:
    from pydtnn import model as model_module


class BackendType(enum.StrEnum):
    CPU = enum.auto()
    GPU = enum.auto()


class PromoteToBackend:
    _backend: typing.Self

    def __new__(cls, *args, **kwds):
        # Save top-level constructor arguments
        self = super().__new__(cls)
        self._new_backend = (args, kwds)  # type: ignore
        return self

    def __getattribute__(self, name: str):
        ref = "_backend"

        # Get backend
        try:
            backend = super().__getattribute__(ref)
        except AttributeError:
            backend = None

        # Skip backend if internal
        if backend is None or name.endswith(ref):
            return super().__getattribute__(name)
        else:
            return getattr(backend, name)

    def _get_backend_cls(self) -> typing.Any:
        cls = self.__class__
        backend = self.model._backend
        module_name = cls.__module__.split(".", 1)[1]
        backend_module_name = f"pydtnn.backends.{backend}.{module_name}"
        backend_module = importlib.import_module(backend_module_name)
        cls_name = f"{cls.__name__}{backend.upper()}"
        cls = getattr(backend_module, cls_name)
        return cls

    def __setattr__(self, name: str, value) -> None:
        ref = "_backend"

        # Get backend
        backend = getattr(self, ref, None)

        # Skip backend if internal
        if backend is None or name.endswith(ref):
            super().__setattr__(name, value)
        else:
            setattr(backend, name, value)

    def __delattr__(self, name: str) -> None:
        ref = "_backend"

        # Get backend
        backend = getattr(self, ref, None)

        # Skip backend if internal
        if backend is None or name.endswith(ref):
            super().__delattr__(name)
        else:
            delattr(backend, name)

    def init_backend(self) -> None:
        """
        Initialize the backend implementation used

        **Notice**: All object attributes are cleared when called.
        So, if used, this method should be the first called.
        """
        # Clear backend
        try:
            del self._backend
        except AttributeError:
            pass

        # Get backend class
        cls = self._get_backend_cls()
        if cls is None:
            return

        # Create backend instance
        args, kwds = self._new_backend
        self._backend = cls(*args, **kwds)
        self._frontend = self

    # Base class
    model: "model_module.Model"

    def set_model(self, model: "model_module.Model") -> None:
        """Link a to a new model instance"""
        self.model = model

    def init_backend_from_model(self, model: "model_module.Model") -> None:
        """Initialize backend and link a new model instance"""
        #  NOTE: This one is to have access to "model" in the "get_backend_cls"
        self.set_model(model)
        #  NOTE: This function masks all previous attribute data.
        self.init_backend()
        #  NOTE: This one is to set model in the backend class.
        self.set_model(model)
    # ---

    @classmethod
    def from_model[C](cls: type[C], model: "model_module.Model") -> C:
        raise NotImplementedError("Use a concrete optimizer!")
