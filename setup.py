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
import re

import setuptools

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'torchsde', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

setuptools.setup(
    name="torchsde",
    version=version,
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    description="SDE solvers and stochastic adjoint sensitivity analysis in PyTorch.",
    url="https://github.com/google-research/torchsde",
    packages=setuptools.find_packages(exclude=['benchmarks', 'diagnostics', 'examples', 'tests']),
    # TODO(lxuechen): Tested on my local machine, adjoint breaks for torch==1.8.0 (w/ cu111 and above) due to segfault
    #  occurring in mvp for ExAdditive and Stratonovich at the following line:
    #  https://github.com/google-research/torchsde/blob/f965bc9a716a86bce45c3d410bc9eaf22283037e/torchsde/_core/misc.py#L62
    #  Since this problem doesn't occur for lower versions, it's likely a problem with the C extension.
    install_requires=['torch>=1.6.0, <1.8.0', 'numpy==1.19.*', 'boltons>=20.2.1', 'trampoline>=0.1.2', 'scipy==1.5.*'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
