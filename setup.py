from pathlib import Path

from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize


ext_modules = [
    Extension("pydtnn.utils._cyutility", sources=["pydtnn/utils/_cyutility.c"])
]

for pyx in Path("pydtnn").rglob("*.pyx"):
    ext_modules.append(Extension(
        ".".join(pyx.with_suffix("").parts),
        [str(pyx)],
        extra_compile_args=["-fopenmp", "-O3", "-march=native", "-g0"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
    ))

setup(
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        shared_utility_qualified_name="pydtnn.utils._cyutility"
    ),
)
