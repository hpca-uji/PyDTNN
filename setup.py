from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 

ext_modules = [
    Extension(
        "NN_im2col_cython",
        ["NN_im2col_cython.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='NN_im2col_cython',
    ext_modules=cythonize(ext_modules),
)
