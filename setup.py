from distutils.core import setup, Extension
from Cython.Build import cythonize, build_ext
import numpy
import os

module_name = 'boxcounting'

e1 = Extension('fastload', ['src/fastload.pyx'], )
e2 = Extension('boxcounting', ['src/boxcounting.pyx'], )
e3 = Extension('tree', ['src/boxcounting.pyx'], )

ext_modules = [e1, e2, e3]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"} #all are Python-3
    e.define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    e.include_dirs = [numpy.get_include(), '.']

setup(
    name = module_name,
    cmdclass = {'build_ext': build_ext},
    ext_modules=ext_modules
    )
