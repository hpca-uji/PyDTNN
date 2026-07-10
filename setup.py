"""PyDTNN's setup"""

from pathlib import Path
from os import process_cpu_count

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# configuration
PACKAGE = Path("pydtnn")

EXTENSION_ARGS = {
    "extra_compile_args": ["-fopenmp", "-O3", "-g0"],
    "extra_link_args": ["-fopenmp", "-s"],
    "include_dirs": [numpy.get_include()],
}

CYTHON_ARGS = {
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
            self.parallel = process_cpu_count()

    def finalize_options(self) -> None:
        """Finish extensions configuration"""
        self.parallel = int(self.parallel)

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
        Extension(BuildExt.path_module(CYTHON_UTILITY), sources=[str(CYTHON_UTILITY)]),
        *(Extension(BuildExt.path_module(pyx), [str(pyx)], **EXTENSION_ARGS) for pyx in PACKAGE.rglob("*.pyx"))  # type: ignore
    ]
)
