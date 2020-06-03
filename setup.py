from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 

ext_modules = [
    Extension(
        module,
        ["%s.pyx" % module],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()],
        #libraries=["python3.7m"],
        #library_dirs=["/usr/local/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin"]
    ) for module in ["NN_im2col_cython", "NN_argmax_cython", \
                     "NN_relu_cython", "NN_add_cython"]
]

setup(
    ext_modules=cythonize(ext_modules),
)
