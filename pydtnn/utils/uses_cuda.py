"""Utilities for managing and loading CUDA kernel code within the PyDTNN framework."""

import functools
from typing import TYPE_CHECKING, Any

from pydtnn.utils import read_file

__all__ = ("UsesCudaCode",)

if TYPE_CHECKING:
    import cupy as cp  # type: ignore
    from pycuda.driver import Function, Module  # type: ignore

type Abs_Module = "Module | cp.RawModule"
type Abs_Function = "Function | cp.RawKernel"


class UsesCudaCode[M: Abs_Module, F: Abs_Function]:
    """Mixin class providing functionality to load, compile, and retrieve CUDA kernels."""

    def __init__(self, *args: Any, **kwds: Any) -> None:
        """Initializes the base path and defines replacement dictionary."""
        super().__init__(*args, **kwds)
        self.base_path_code = "/".join([*self.__module__.split(".")[1:3], "utils"])
        self.defines_replaces = dict[str, str]()

    def _get_kernel_function(
        self, kernel: M, func_name: str | None, func_name_subfix: str = ""
    ) -> F:
        """Retrieves a specific function from a compiled CUDA module."""
        if func_name is None:
            # NOTE: self.__module__ must be something like "pydtnn.backends.cython.activations.relu"
            func_name = self.__module__.split(".")[-1]
        func_name = f"{func_name}{func_name_subfix}"

        return kernel.get_function(func_name)

    def _get_code(
        self,
        path_code: str | None,
        code_file_name: str | None,
        defines_replaces: dict[str, str] | None,
        file_extension: str | None = ".cu",
    ) -> str:
        """Reads and processes CUDA source code from a file."""

        if defines_replaces is None:
            defines_replaces = self.defines_replaces

        if path_code is None:
            path_code = self.base_path_code

        if code_file_name is None:
            # NOTE: self.__module__ must be something like "pydtnn.backends.cython.activations.relu"
            code_file_name = self.__module__.split(".")[-1]

        if file_extension is not None:
            code_file_name = f"{code_file_name}{file_extension}"

        path_code = f"{path_code}/{code_file_name}"

        return read_file(path_code, defines_replaces)

    @staticmethod
    @functools.cache
    def _get_module(module: M, code: str) -> M:
        """Compiles CUDA code into a module, cached by code content."""
        return module(code)

    def _cuda_kernel(self, code: str) -> M:
        """Abstract method to compile CUDA code into a module."""
        raise NotImplementedError(
            "This is a fake function that must be implemented by a child class."
        )

    def _get_kernel(
        self,
        path_code: str | None = None,
        code_file_name: str | None = None,
        func_name: str | None = None,
        defines_replaces: dict[str, str] | None = None,
        func_name_subfix: str = "",
        file_extension: str | None = ".cu",
    ) -> F:
        """Loads, compiles, and returns a CUDA kernel function."""
        # NOTE: If you are searching for the source files, go to "{self.base_path_code}/{file_name_without_exception}.cu"
        # NOTE (cont.) e.g.: if you are searching for the leaky relu source files, go
        # to "/pydtnn/backends/{backend}/leaky_relu.cu"

        code = self._get_code(
            path_code=path_code,
            code_file_name=code_file_name,
            defines_replaces=defines_replaces,
            file_extension=file_extension,
        )
        kernel = self._get_module(self._cuda_kernel, code)

        return self._get_kernel_function(
            kernel=kernel, func_name=func_name, func_name_subfix=func_name_subfix
        )

    def _fwd_kernel(
        self,
        path_code: str | None = None,
        code_file_name: str | None = None,
        func_name: str | None = None,
        defines_replaces: dict[str, str] | None = None,
    ) -> F:
        """Retrieves the forward pass CUDA kernel."""
        return self._get_kernel(
            path_code, code_file_name, func_name, defines_replaces, func_name_subfix="_fwd"
        )

    def _bwd_kernel(
        self,
        path_code: str | None = None,
        code_file_name: str | None = None,
        func_name: str | None = None,
        defines_replaces: dict[str, str] | None = None,
    ) -> F:
        """Retrieves the backward pass CUDA kernel."""
        return self._get_kernel(
            path_code, code_file_name, func_name, defines_replaces, func_name_subfix="_bwd"
        )
