"""PyDTNN's setup"""

from itertools import chain
from pathlib import Path
from os import environ, process_cpu_count
from typing import Any

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# configuration
PACKAGE = Path("pydtnn")

EXTENSION_ARGS: dict[str, Any] = {
    "extra_compile_args": ["-fopenmp", "-O2", "-g0"],
    "extra_link_args": ["-fopenmp", "-s"],
    "include_dirs": [numpy.get_include()],
}

CYTHON_ARGS: dict[str, Any] = {
    "language_level": 3,
    "compiler_directives": {
        "cdivision": True,
        "overflowcheck": False,
        "wraparound": False,
        "boundscheck": False,
        "initializedcheck": False
    }
}

CYTHON_UTILITY = PACKAGE.joinpath("utils/_cyutility.c")


# backend
class BuildExt(build_ext):
    """Extension builder"""

    @staticmethod
    def path_module(path: Path) -> str:
        """Convert a path to a module name"""
        return ".".join(path.with_suffix("").parts)

    def initialize_options(self) -> None:
        """Default extensions configuration"""
        super().initialize_options()

        if self.parallel is None:
            self.parallel = environ.get(f"{PACKAGE.name}_parallel".upper()) or process_cpu_count()

    def finalize_options(self) -> None:
        """Finish extensions configuration"""
        try:
            self.parallel = int(self.parallel)
        except ValueError:
            self.parallel = 1

        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules,
            **CYTHON_ARGS,
            nthreads=self.parallel,
            shared_utility_qualified_name=self.path_module(CYTHON_UTILITY)
        )

        if self.parallel <= 1:
            self.parallel = None

        super().finalize_options()


# entrypoint
setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension(
            BuildExt.path_module(path),
            [str(path)],
            **EXTENSION_ARGS
        )
        for path in chain([CYTHON_UTILITY], PACKAGE.rglob("*.pyx"))
    ]
)
