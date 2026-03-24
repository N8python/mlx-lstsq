from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import setup

from mlx import extension


ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8")


def read_version() -> str:
    init_py = (ROOT / "mlx_lstsq" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', init_py, re.MULTILINE)
    if not match:
        raise RuntimeError("Unable to find package version.")
    return match.group(1)


class CMakeBuild(extension.CMakeBuild):
    def build_extension(self, ext: extension.CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_SHARED_LIBS=ON",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_ROOT_DIR={sys.prefix}",
            "-DPython_FIND_STRATEGY=LOCATION",
        ]
        build_args = []

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += [f"-j{os.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        os.environ["MLX_DIR"] = extension._MLX_PATH

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True)


if __name__ == "__main__":
    setup(
        name="mlx-lstsq",
        version=read_version(),
        description="Least-squares solvers for MLX with Apple MPS native extensions.",
        long_description=README,
        long_description_content_type="text/markdown",
        license="CC0-1.0",
        license_files=("LICENSE",),
        python_requires=">=3.10",
        install_requires=["mlx>=0.31.1"],
        packages=["mlx_lstsq"],
        package_data={"mlx_lstsq": ["*.dylib", "*.metallib"]},
        ext_modules=[extension.CMakeExtension("mlx_lstsq._ext")],
        cmdclass={"build_ext": CMakeBuild},
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Operating System :: MacOS :: MacOS X",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: 3.14",
            "Programming Language :: C++",
            "Topic :: Scientific/Engineering :: Mathematics",
        ],
        keywords=["mlx", "least-squares", "linear-algebra", "metal", "mps"],
        zip_safe=False,
    )
