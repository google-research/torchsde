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

import torch

from . import brownian_base
from . import brownian_interval
from ..types import Optional, Scalar, Tensor, Tuple, Union


class ReverseBrownian(brownian_base.BaseBrownian):
    def __init__(self, base_brownian):
        super(ReverseBrownian, self).__init__()
        self.base_brownian = base_brownian

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
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


class BrownianPath(brownian_base.BaseBrownian):
    """Brownian path, storing every computed value.

    Useful for speed, when memory isn't a concern.

    To use:
    >>> bm = BrownianPath(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]])
    """

    def __init__(self, t0: Scalar, w0: Tensor, window_size: int = 8):
        """Initialize Brownian path.
        Arguments:
            t0: Initial time.
            w0: Initial state.
            window_size: Unused; deprecated.
        """
        t1 = t0 + 1
        self._w0 = w0
        self._interval = brownian_interval.BrownianInterval(t0=t0, t1=t1, size=w0.shape, dtype=w0.dtype,
                                                            device=w0.device, cache_size=None)
        super(BrownianPath, self).__init__()

    def __call__(self, t, tb=None, return_U=False, return_A=False):
        # Deliberately called t rather than ta, for backward compatibility
        out = self._interval(t, tb, return_U=return_U, return_A=return_A)
        if tb is None and not return_U and not return_A:
            out = out + self._w0
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self._interval})"

    @property
    def dtype(self):
        return self._interval.dtype

    @property
    def device(self):
        return self._interval.device

    @property
    def shape(self):
        return self._interval.shape

    @property
    def levy_area_approximation(self):
        return self._interval.levy_area_approximation


class BrownianTree(brownian_base.BaseBrownian):
    """Brownian tree with fixed entropy.

    Useful when the map from entropy -> Brownian motion shouldn't depend on the
    locations and order of the query points. (As the usual BrownianInterval
    does - note that BrownianTree is slower as a result though.)

    To use:
    >>> bm = BrownianTree(t0=0.0, w0=torch.zeros(4, 1))
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    def __init__(self, t0: Scalar,
                 w0: Tensor,
                 t1: Optional[Scalar] = None,
                 w1: Optional[Tensor] = None,
                 entropy: Optional[int] = None,
                 tol: float = 1e-6,
                 pool_size: int = 24,
                 cache_depth: int = 9,
                 safety: Optional[float] = None):
        """Initialize the Brownian tree.

        The random value generation process exploits the parallel random number paradigm and uses
        `numpy.random.SeedSequence`. The default generator is PCG64 (used by `default_rng`).

        Arguments:
            t0: Initial time.
            w0: Initial state.
            t1: Terminal time.
            w1: Terminal state.
            entropy: Global seed, defaults to `None` for random entropy.
            tol: Error tolerance before the binary search is terminated; the search depth ~ log2(tol).
            pool_size: Size of the pooled entropy. This parameter affects the query speed significantly.
            cache_depth: Unused; deprecated.
            safety: Unused; deprecated.
        """

        if t1 is None:
            t1 = t0 + 1
        if w1 is None:
            W = None
        else:
            W = w1 - w0
        self._w0 = w0
        self._interval = brownian_interval.BrownianInterval(t0=t0,
                                                            t1=t1,
                                                            size=w0.shape,
                                                            dtype=w0.dtype,
                                                            device=w0.device,
                                                            entropy=entropy,
                                                            tol=tol,
                                                            pool_size=pool_size,
                                                            halfway_tree=True,
                                                            W=W)
        super(BrownianTree, self).__init__()

    def __call__(self, t, tb=None, return_U=False, return_A=False):
        # Deliberately called t rather than ta, for backward compatibility
        out = self._interval(t, tb, return_U=return_U, return_A=return_A)
        if tb is None and not return_U and not return_A:
            out = out + self._w0
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}(interval={self._interval})"

    @property
    def dtype(self):
        return self._interval.dtype

    @property
    def device(self):
        return self._interval.device

    @property
    def shape(self):
        return self._interval.shape

    @property
    def levy_area_approximation(self):
        return self._interval.levy_area_approximation


def brownian_interval_like(y: Tensor,
                           t0: Optional[Scalar] = 0.,
                           t1: Optional[Scalar] = 1.,
                           size: Optional[Tuple[int, ...]] = None,
                           dtype: Optional[torch.dtype] = None,
                           device: Optional[Union[str, torch.device]] = None,
                           **kwargs):
    """Returns a BrownianInterval object with the same size, device, and dtype as a given tensor."""
    size = y.shape if size is None else size
    dtype = y.dtype if dtype is None else dtype
    device = y.device if device is None else device
    return brownian_interval.BrownianInterval(t0=t0, t1=t1, size=size, dtype=dtype, device=device, **kwargs)
