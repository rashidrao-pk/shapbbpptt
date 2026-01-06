from setuptools import setup, Extension, find_packages
import numpy as np
import platform
import sys
from pathlib import Path

is_windows = (platform.system() == "Windows")
is_mingw = is_windows and ("--compiler=mingw32" in sys.argv)

extra_compile_args = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-Wunreachable-code",
]
if is_mingw:
    extra_compile_args.append("-DMS_WIN64")

def get_ext_modules():
    pyx = Path("shap_bpt/bpt.pyx")
    c_file = Path("shap_bpt/bpt.c")

    # Prefer Cython if available; otherwise fall back to pre-generated C
    try:
        from Cython.Build import cythonize
        sources = [str(pyx)]
        extensions = [
            Extension(
                "shap_bpt.bpt",
                sources,
                include_dirs=[np.get_include()],
                extra_compile_args=extra_compile_args,
            )
        ]
        return cythonize(extensions, language_level=3)
    except Exception:
        if not c_file.exists():
            raise RuntimeError(
                "Cython is not available and shap_bpt/bpt.c is missing. "
                "Either install Cython or generate bpt.c from bpt.pyx."
            )
        return [
            Extension(
                "shap_bpt.bpt",
                [str(c_file)],
                include_dirs=[np.get_include()],
                extra_compile_args=extra_compile_args,
            )
        ]

setup(
    name="shap_bpt",
    version="1.0",
    packages=find_packages(),
    ext_modules=get_ext_modules(),
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        "scikit-image",
    ],
)
