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

import functools

from . import brownian_interval


class ReverseBrownian:
    def __init__(self, base_brownian):
        super(ReverseBrownian, self).__init__()
        self.base_brownian = base_brownian

    def __call__(self, ta, tb, return_U=False, return_A=False):
        # Whether or not to negate the statistics depends on the return value of the adjoint SDE. Currently, the adjoint
        # returns negated drift and diffusion, so we don't negate here.
        return self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)

    @property
    def shape(self):
        return self.base_brownian.shape

    @property
    def levy_area_approximation(self):
        return self.base_brownian.levy_area_approximation


BrownianPath = functools.partial(brownian_interval.BrownianInterval, cache_size=None)
BrownianPath.__doc__ = \
"""Brownian path, storing every computed value.

Useful for speed, when memory isn't a concern.

See BrownianInterval for its arguments.

To use:
>>> bm = BrownianPath(t0=0.0, t1=1.0, shape=(4, 1), device='cuda')
>>> bm(0., 0.5)
tensor([[ 0.0733],
        [-0.5692],
        [ 0.1872],
        [-0.3889]], device='cuda:0')
"""


# Set dt to None so that it can't also be set by the user.
BrownianTree = functools.partial(brownian_interval.BrownianInterval, halfway_tree=True, dt=None, tol=1e-6)
BrownianTree.__doc__ = \
"""Brownian tree with fixed entropy.

Useful when there the map from entropy -> Brownian motion shouldn't depend on 
the locations and order of the query points.

To use:
>>> bm = BrownianTree(t0=0.0, t1=1.0, shape=(4, 1), device='cuda')
>>> bm(0., 0.5)
tensor([[ 0.0733],
        [-0.5692],
        [ 0.1872],
        [-0.3889]], device='cuda:0')
"""
