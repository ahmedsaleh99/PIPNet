from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "cpu_nms",
        sources=["cpu_nms.pyx"],  # Use .pyx if that's the file
        include_dirs=[np.get_include()],
        extra_compile_args=["-O2"],
    )
]

setup(
    name="cpu_nms",
    ext_modules=cythonize(ext_modules, language_level="3"),
)
