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

from . import brownian_base
from . import brownian_interval


class ReverseBrownian(brownian_base.BaseBrownian):
    def __init__(self, base_brownian):
        super(ReverseBrownian, self).__init__()
        self.base_brownian = base_brownian

    def __call__(self, ta, tb, return_U=False, return_A=False):
        # Whether or not to negate the statistics depends on the return value of the adjoint SDE. Currently, the adjoint
        # returns negated drift and diffusion, so we don't negate here.
        return self.base_brownian(-tb, -ta, return_U=return_U, return_A=return_A)

    def __repr__(self):
        return f"{self.__class__.__name__}(base_brownian={self.base_brownian})"

    @property
    def dtype(self):
        return self.base_brownian.dtype

    @property
    def device(self):
        return self.base_brownian.device

    @property
    def shape(self):
        return self.base_brownian.shape

    @property
    def levy_area_approximation(self):
        return self.base_brownian.levy_area_approximation


class BrownianPath(brownian_interval.BrownianInterval):
    """Brownian path, storing every computed value.

    Useful for speed, when memory isn't a concern.

    To use:
    >>> bm = BrownianPath(t0=0.0, t1=1.0, shape=(4, 1), device='cuda')
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    def __init__(self, *args, **kwargs):
        """Arguments as BrownianInterval."""
        super(BrownianPath, self).__init__(*args, **kwargs, cache_size=None)


class BrownianTree(brownian_interval.BrownianInterval):
    """Brownian tree with fixed entropy.

    Useful when the map from entropy -> Brownian motion shouldn't depend on the
    locations and order of the query points. (As the usual BrownianInterval
    does - note that BrownianTree is slower as a result though.)

    To use:
    >>> bm = BrownianTree(t0=0.0, t1=1.0, shape=(4, 1), device='cuda')
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    def __init__(self, *args, **kwargs):
        """Arguments as BrownianInterval."""
        if 'tol' not in kwargs:
            kwargs['tol'] = 1e-6
        super(BrownianTree, self).__init__(*args, **kwargs, halfway_tree=True)
