# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform

import setuptools

try:
    import torch
    from torch.utils import cpp_extension
except ModuleNotFoundError:
    raise ModuleNotFoundError("Unable to import torch. Please install torch>=1.6.0 at https://pytorch.org.")

extra_compile_args = []
extra_link_args = []

# This is a problem of macOS: https://github.com/pytorch/pytorch/issues/16805.
if platform.system() == "Darwin":
    extra_compile_args += ["-stdlib=libc++"]
    extra_link_args += ["-stdlib=libc++"]

brownian_lib_prefix = os.path.join(".", "csrc")
sources = os.listdir(brownian_lib_prefix)
sources = filter(lambda x: x.endswith('.cpp'), sources)
# Don't include C++ tests for now.
sources = filter(lambda x: 'test' not in x, sources)
sources = map(lambda x: os.path.join(brownian_lib_prefix, x), sources)
sources = list(sources)

USE_CUDA = torch.cuda.is_available() and cpp_extension.CUDA_HOME is not None
if os.getenv('FORCE_CPU', '0') == '1':
    USE_CUDA = False

if USE_CUDA:
    define_macros = [('USE_CUDA', None)]
    extension_func = cpp_extension.CUDAExtension
else:
    define_macros = []
    extension_func = cpp_extension.CppExtension

setuptools.setup(
    name="torchsde",
    version="0.1.2",
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    description="SDE solvers and stochastic adjoint sensitivity analysis in PyTorch.",
    url="https://github.com/google-research/torchsde",
    packages=setuptools.find_packages(exclude=['diagnostics', 'tests', 'benchmarks']),
    ext_modules=[
        extension_func(name='torchsde._brownian_lib',
                       sources=sources,
                       extra_compile_args=extra_compile_args,
                       extra_link_args=extra_link_args,
                       define_macros=define_macros,
                       optional=True)
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch>=1.6.0', 'blist', 'numpy>=1.19.1', 'boltons>=20.2.1'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
