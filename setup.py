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

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="torchsde",
    version=version,
    author="Xuechen Li, Patrick Kidger",
    author_email="lxuechen@cs.stanford.edu, hello@kidger.site",
    description="SDE solvers and stochastic adjoint sensitivity analysis in PyTorch.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/torchsde",
    packages=setuptools.find_packages(exclude=['benchmarks', 'diagnostics', 'examples', 'tests']),
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.5",
        "torch>=1.6.0",
        "trampoline>=0.1.2",
    ],
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
