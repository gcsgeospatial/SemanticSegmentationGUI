# Replaces upstream's setup.py (hardcodes flags that break modern toolchains).
# Copied over utils/nearest_neighbors/setup.py by the recipe's path source.
from setuptools import setup, Extension
import sys
import numpy

if sys.platform == "win32":
    # MSVC: C++14 is the default (no -std flag exists); /openmp covers -fopenmp
    cflags, lflags = ["/openmp"], []
else:
    cflags = ["-std=c++11", "-fopenmp"]
    lflags = ["-std=c++11", "-fopenmp"]

setup(
    name="nearest_neighbors",
    ext_modules=[
        Extension(
            "nearest_neighbors",
            sources=["knn.cpp", "knn_.cxx"],
            include_dirs=["./", numpy.get_include()],
            language="c++",
            extra_compile_args=cflags,
            extra_link_args=lflags,
        )
    ],
)
