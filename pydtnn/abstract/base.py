import re
import typing
import importlib

from pydtnn import utils
from pydtnn.utils.constants import Array

if typing.TYPE_CHECKING:
    from pydtnn._model import model_base as model_module


class Base[T: Array]:
    _backend: typing.Self
    _frontend: typing.Self

    _map_backend = {
        "all": "pydtnn",
        "cpu": "numpy,cython",
        "gpu": "pycuda"
    }

    def __new__(cls, *args, **kwds):
        # Save top-level constructor arguments
        self = super().__new__(cls)
        self._new_backend = (args, kwds)  # type: ignore
        return self

    def __init__(self) -> None:
        self.memory_used: int = 0
        self.tmp_memory_used: int = 0

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

    def _parse_backend(self, spec: str) -> dict[str, list[str]]:
        """
        Parse a backend spec string.
        Format: [module[,module[,...]]:]backend[,backend[,...]][;...]
        Example: all:numpy;conv_2d:gemm;layers,optimizers:numpy,cython
        Selection: More specific modules are attempted first, backend order goes from least to most priority.
        """
        groups = {}

        spec = re.sub(
            fr"\b({r"|".join(self._map_backend)})\b",
            lambda match: self._map_backend[match.group()],
            spec
        )

        for group in spec.split(";"):
            kv = group.split(":", 1)

            values = kv.pop()
            try:
                keys = kv.pop()
            except IndexError:
                keys = "pydtnn"

            for key in keys.split(","):
                groups.setdefault(key, []).extend(reversed(values.split(",")))

        return dict(sorted(
            groups.items(),
            key=lambda item: (-item[0].count("."), item[0]),
        ))

    def _get_backend(self) -> typing.Any:
        """Get relevant backend class"""
        cls = self.__class__
        module_name = cls.__module__
        submodule_name = module_name.split(".", 1)[1]
        spec = self._parse_backend(self.model.backend)

        if module_name.startswith("pydtnn.backends."):
            return None  # We are a backend

        for group, backends in spec.items():
            if f".{group}." not in f".{module_name}.":
                continue  # Spec not relevant to class
            for backend in backends:
                backend_module_name = f"pydtnn.backends.{backend}.{submodule_name}"
                try:
                    backend_module = importlib.import_module(backend_module_name)
                except ModuleNotFoundError as exc:
                    if backend_module_name.startswith(exc.name):
                        continue  # Backend not found
                    raise  # Backend raised exception
                cls_name = f"{cls.__name__}{backend.title()}"
                cls = getattr(backend_module, cls_name)
                return cls

        raise ValueError(f"Backend not found for {self} with {spec}")

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

    def _init_backend(self) -> None:
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
        cls = self._get_backend()
        if cls is None:
            return

        # Create backend instance
        args, kwds = self._new_backend
        self._backend = cls(*args, **kwds)
        self._frontend = self

    # Base class
    model: "model_module.Model"

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def canonical_name(self) -> str:
        self = getattr(self, "_frontend", self)
        return type(self).__name__

    def _show_props(self) -> dict:
        props = {}

        props["name"] = self.canonical_name

        if self.name != self.canonical_name:
            props["backend"] = self.name.removeprefix(self.canonical_name).lower()

        if self.memory_used > 0:
            memory = utils.convert_size_bytes(self.memory_used)
            if self.tmp_memory_used > 0:
                tmp_memory = utils.convert_size_bytes(self.tmp_memory_used)
                memory = f"{memory} ({tmp_memory} tmp)"
            props["memory"] = memory

        return props

    def __repr__(self) -> str:
        props = self._show_props()
        name = props.pop("name")

        props = " ".join(
            f"{key}={value!r}"
            for key, value in props.items()
        )

        return f"<{name} {props}>" if props else f"<{name}>"

    def _model_init(self) -> None:
        pass

    def _post_init(self) -> None:
        pass

    def _init_backend_with_model(self, model: "model_module.Model_Base[T]") -> None:
        """Initialize backend and link a new model instance"""
        self.model = model  # Set on frontend
        self._init_backend()
        self.model = model  # Set on backend

    @classmethod
    def from_model[S](cls: type[S], model: "model_module.Model_Base[T]") -> S:
        """Create object from a given model"""
        raise NotImplementedError("Use a concrete optimizer!")
