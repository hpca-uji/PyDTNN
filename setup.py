from pathlib import Path

from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize


setup(
    ext_modules=cythonize([
        Extension(
            ".".join(pyx.with_suffix("").parts),
            [str(pyx)],
            extra_compile_args=["-fopenmp", "-O3", "-march=native", "-g0"],
            extra_link_args=["-fopenmp"],
            include_dirs=[numpy.get_include()],
        ) for pyx in Path("pydtnn").rglob("*.pyx")
    ], language_level=3),
)
