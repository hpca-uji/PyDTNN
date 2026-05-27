from pathlib import Path
from os import process_cpu_count

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        if self.parallel is None:
            self.parallel = process_cpu_count()


ext_modules = [
    Extension("pydtnn.utils._cyutility", sources=["pydtnn/utils/_cyutility.c"])
]

for pyx in Path("pydtnn").rglob("*.pyx"):
    ext_modules.append(Extension(
        ".".join(pyx.with_suffix("").parts),
        [str(pyx)],
        extra_compile_args=["-fopenmp", "-flto", "-O3", "-fvisibility=hidden", "-march=native", "-g0"],
        extra_link_args=["-fopenmp", "-flto", "-s"],
        include_dirs=[numpy.get_include()],
    ))

setup(
    cmdclass={
        "build_ext": BuildExt
    },
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives={
            "cdivision": True,
            "overflowcheck": False,
            "wraparound": False,
            'boundscheck': False,
            "initializedcheck": False
        },
        cache=True,
        nthreads=process_cpu_count(),
        shared_utility_qualified_name="pydtnn.utils._cyutility"
    )
)
