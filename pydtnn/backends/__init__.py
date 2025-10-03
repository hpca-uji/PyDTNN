import importlib
from contextlib import suppress

from . import cpu
from . import gpu
from .. import model as model_module


class PromoteToBackendMixin:

    def __new__(cls, *args, **kwargs):
        if not model_module.enable_cudnn:
            backend = "cpu"
        else:
            backend = "gpu"
        new_cls_name: str = f"{cls.__name__}{backend.upper()}"
        # cls.__module__ should be something like 'pydtnn.activations.arctanh'
        submodule_name: str = cls.__module__.split(".")[1]
        if submodule_name == "backends":
            new_cls = cls
        else:
            backend_module_name: str = f"pydtnn.backends.{backend}.{submodule_name}"
            backend_module = importlib.import_module(backend_module_name)
            try:
                new_cls = getattr(backend_module, new_cls_name)
            except AttributeError:
                new_cls = cls
        instance = super().__new__(new_cls)
        if new_cls != cls:
            # noinspection PyArgumentList
            instance.__init__(*args, **kwargs)
        return instance

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
