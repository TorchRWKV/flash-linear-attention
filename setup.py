# -*- coding: utf-8 -*-

import ast
import os
import re
import subprocess
import warnings
from pathlib import Path

from datetime import datetime
from packaging.version import Version, parse
from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import shutil
import subprocess
import sys


with open('README.md') as f:
    long_description = f.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = 'rwkv-fla'

# FORCE_BUILD: force a fresh build locally, instead of attempting to find prebuilt wheels
FORCE_BUILD = os.getenv('FLA_FORCE_BUILD', "FALSE") == 'TRUE'
# SKIP_CUDA_BUILD: allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
SKIP_CUDA_BUILD = os.getenv('FLA_SKIP_CUDA_BUILD', "TRUE") == 'TRUE'
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv('FLA_FORCE_CXX11_ABI', "FALSE") == 'TRUE'


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


ext_modules = []
if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]

    check_if_cuda_home_none('fla')
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                "FLA is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if CUDA_HOME is not None:
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")

    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"] + generator_flag,
        "nvcc": append_nvcc_threads(
            [
                "-O3",
                "-std=c++17",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ]
            + generator_flag
            + cc_flag
        ),
    }


def get_package_version():
    with open(Path(this_dir) / 'fla' / '__init__.py') as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    version = ast.literal_eval(version_match.group(1))
    # Check for .git directory
    git_dir = os.path.join(this_dir, '.git')
    if os.path.exists(git_dir):
        # We're in a Git repository, try to get the branch name
        try:
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                                 universal_newlines=True,
                                                 stderr=subprocess.DEVNULL,
                                                 cwd=this_dir).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_branch = None
    else:
        git_branch = None

    build_date = datetime.now().strftime("%Y%m%d%H%M")
    if git_branch and git_branch in ('stable'):
        return f"{version}.{build_date}"
    else:
        return f"{version}.dev{build_date}"


def check_conflicts():
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if 'fla' in result.stdout and 'rwkv-fla' not in result.stdout:
            print("Error: fla package is already installed. Please uninstall it first with 'pip uninstall fla'")
            sys.exit(1)
    except Exception:
        pass

def rename2rwkvfla():
    packages = find_packages()

    import os
    import fileinput

    if os.path.exists('fla'):
        shutil.rmtree('rwkvfla') if os.path.exists('rwkvfla') else None
        shutil.copytree('fla', 'rwkvfla')

        # 递归遍历 rwkvfla 目录下的所有 .py 文件
        for root, dirs, files in os.walk('rwkvfla'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    # 读取文件内容
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # 替换 import 语句
                    content = content.replace('from fla', 'from rwkvfla')
                    content = content.replace('import fla', 'import rwkvfla')
                    
                    # 写回文件
                    with open(filepath, 'w') as f:
                        f.write(content)

    package_map = {'fla': 'rwkvfla'}
    new_packages = [package_map.get(p, p) for p in packages]
    return new_packages

check_conflicts()

setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    description='Fast Triton-based implementations for RWKV',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zhiyuan Li, Songlin Yang, Yu Zhang',
    author_email='uniartisan2017@gmail.com',
    url='https://github.com/TorchRWKV/flash-linear-attention',
    packages=rename2rwkvfla(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
    install_requires=[
        'transformers',
        'datasets',
        'einops',
        'ninja'
    ],
    dependency_links=['https://download.pytorch.org/whl/nightly/'],
    extras_require={
        'conv1d': ['causal-conv1d>=1.4.0'],
        'cuda': ['triton'],
        'xpu': ['pytorch-triton-xpu'],
        'rocm': ['pytorch-triton-rocm'],
    }
)
