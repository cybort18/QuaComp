import os
import sys
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    has_pybind11 = True
except ImportError:
    has_pybind11 = False

class BuildExtOptional(build_ext if has_pybind11 else object):
    """
    Subclass build_ext to allow optional C++ extension compilation.
    If C++ compiler is absent or build fails, fallback safely to pure Python.
    """
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\n[QuaComp] WARNING: Optional C++ extension '{ext.name}' failed to build: {e}")
            print("[QuaComp] NOTE: QuaComp will automatically fallback to high-performance CPython/NumPy engine.\n")

ext_modules = []
cmdclass = {}

if has_pybind11 and os.path.exists(os.path.join("src", "engine", "cpp", "fusion.cpp")):
    ext_modules = [
        Pybind11Extension(
            "quacomp_cpp",
            [os.path.join("src", "engine", "cpp", "fusion.cpp")],
            include_dirs=[os.path.join("src", "engine", "cpp")],
            cxx_std=14,
            optional=True,
        )
    ]
    cmdclass = {"build_ext": BuildExtOptional}

if __name__ == "__main__":
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
