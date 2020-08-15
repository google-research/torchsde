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

import functools as ft
import math
import operator
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar

from . import base_brownian
from . import utils


_rsqrt3 = 1 / math.sqrt(3)


def _randn(shape, dtype, device, seed):
    generator = torch.Generator(device).manual_seed(seed)
    return torch.randn(shape, dtype=dtype, device=device, generator=generator)


def _assert_floating_tensor(name, tensor):
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name}={tensor} should be a Tensor.")
    if not tensor.is_floating_point():
        raise ValueError(f"{name}={tensor} should be floating point.")


class _Interval:
    # Intervals correspond to some subinterval of the overall interval [t0, t1].
    # They are arranged as a binary tree: each node corresponds to an interval. If a node has children, they are left
    # and right subintervals, which partition the parent interval.

    __slots__ = ('_start', '_end', '_parent', '_is_left', '_top', '_W_generator', '_H_generator', '_a_generator',
                 '_midway', '_left_child', '_right_child')

    def __init__(self, start, end, parent, is_left, top, W_generator, H_generator, a_generator):
        # These are the things that every interval has
        self._start = start      # the left hand edge of the interval
        self._end = end          # the right hand edge of the interval
        self._parent = parent    # our parent interval
        self._is_left = is_left  # are we the left or right child of our parent
        self._top = top          # the top-level BrownianInterval, where we cache certain state
        self._W_generator = W_generator  # A generator for increments
        self._H_generator = H_generator  # A generator for space-time Levy areas
        self._a_generator = a_generator  # A generator for full Levy areas

        # These are the things that intervals which are parents also have
        self._midway = None       # The point at which we split between left and right subintervals
        self._left_child = None   # The left subinterval
        self._right_child = None  # The right subinterval

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    ########################################
    #  Calculate increments and levy area  #
    ########################################
    #
    # This is a little bit convoluted, so here's an explanation.
    #
    # The entry point is increment_and_levy_area, below. This immediately calls _increment_and_space_time_levy_area,
    # applies the space-time to full Levy area correction, and then returns.
    #
    # _increment_and_space_time_levy_area in turn calls a central LRU cache, as (later on) we'll need the increment and
    # space-time Levy area of the parent interval to compute our own increment and space-time Levy area, and it's likely
    # that our parent exists in the cache, as if we're being queried then our parent was probably queried recently as
    # well.
    # Note that we don't call the cache directly because the global BrownianInterval overrides
    # _increment_and_space_time_levy_area to return its own increment and space-time Levy area, effectively holding them
    # permanently in the cache.
    # If the request isn't found in the LRU cache then it heads into increment_and_space_time_levy_area_uncached.
    #
    # Now it turns out that the size of our increment and space-time Levy area is really most naturally thought of as a
    # property of our parent: it depends on our parent's increment, space-time Levy area, and whether we are the left or
    # right interval within our parent. So increment_and_space_time_levy_area_uncached in turn checks if we are on the
    # left or right of our parent and dispatches to the parent.
    #
    # left_increment_and_space_time_levy_area and right_increment_and_space_time_levy_area then really do the
    # calculation. As helper functions, they have _brownian_bridge and _common_levy_computation that factor out their
    # common code.

    def increment_and_levy_area(self):
        W, H = self._increment_and_space_time_levy_area()
        A = utils.davie_foster_approximation(W, H, self._end - self._start, self._top.levy_area_approximation,
                                             lambda: self._randn('a'))
        return W, H, A

    def _increment_and_space_time_levy_area(self):
        return self._top.increment_and_space_time_levy_area_cache(self)

    def increment_and_space_time_levy_area_uncached(self):
        if self._is_left:
            return self._parent.left_increment_and_space_time_levy_area()
        else:
            return self._parent.right_increment_and_space_time_levy_area()

    def left_increment_and_space_time_levy_area(self):
        if self._top.have_H:
            left_diff, right_diff, h_reciprocal, a, b, c, W, H, X1, X2, third_coeff = self._common_levy_computation()

            first_coeff = left_diff * h_reciprocal
            second_coeff = 6 * first_coeff * right_diff * h_reciprocal

            left_W = first_coeff * W + second_coeff * H + third_coeff * X1
            left_H = first_coeff**2 * H - a * X1 + c * right_diff * X2
        else:
            # Don't compute space-time Levy area unless we need to
            left_W = self._brownian_bridge(self._start, self._midway)
            left_H = None
        return left_W, left_H

    def right_increment_and_space_time_levy_area(self):
        if self._top.have_H:
            left_diff, right_diff, h_reciprocal, a, b, c, W, H, X1, X2, third_coeff = self._common_levy_computation()

            first_coeff = right_diff * h_reciprocal
            second_coeff = 6 * first_coeff * left_diff * h_reciprocal

            right_W = first_coeff * W - second_coeff * H - third_coeff * X1
            right_H = first_coeff**2 * H - b * X1 - c * left_diff * X2
        else:
            # Don't compute space-time Levy area unless we need to
            right_W = self._brownian_bridge(self._midway, self._end)
            right_H = None
        return right_W, right_H

    def _brownian_bridge(self, ta, tb):
        W, _ = self._increment_and_space_time_levy_area()
        h_reciprocal = 1 / (self._end - self._start)
        mean = (tb - ta) * W * h_reciprocal
        var = (self._end - self._midway) * (self._midway - self._start) * h_reciprocal
        noise = self._randn('W')
        return mean + math.sqrt(var) * noise

    def _common_levy_computation(self):
        W, H = self._increment_and_space_time_levy_area()

        left_diff = self._midway - self._start
        right_diff = self._end - self._midway
        left_diff_squared = left_diff ** 2
        right_diff_squared = right_diff ** 2
        left_diff_cubed = left_diff * left_diff_squared
        right_diff_cubed = right_diff * right_diff_squared

        h_reciprocal = 1 / (self._end - self._start)
        v = 0.5 * math.sqrt(left_diff * right_diff / (left_diff_cubed + right_diff_cubed))

        a = v * left_diff_squared * h_reciprocal
        b = v * right_diff_squared * h_reciprocal
        c = v * _rsqrt3

        X1 = self._randn('W')
        X2 = self._randn('H')

        third_coeff = 2 * (a * left_diff + b * right_diff) * h_reciprocal

        return left_diff, right_diff, h_reciprocal, a, b, c, W, H, X1, X2, third_coeff

    def _randn(self, key):  # key can be either 'W', 'H' or 'a'
        # We generate random noise deterministically wrt some seed; this seed is determined by the generator.
        # This means that if we drop out of the cache, then we'll create the same random noise next time, as we still
        # have the generator.
        if key == 'W':
            generator = self._W_generator
        elif key == 'H':
            generator = self._H_generator
        else:  # key == 'a'
            generator = self._a_generator
        return _randn(self._top.shape, self._top.dtype, self._top.device, generator.generate_state(1).item())

    ########################################
    # Locate an interval in the hierarchy  #
    ########################################
    #
    # The other important piece of this construction is a way to locate any given interval within the binary tree
    # hierarchy. (This is typically the slightly slower part, actually, so if you want to speed things up then this is
    # the bit to target.)
    #
    # loc finds the interval [ta, tb] - and creates it in the appropriate place (as a child of some larger interval) if
    # it doesn't already exist. As in principle we may request an interval that covers multiple existing intervals, then
    # in fact the interval [ta, tb] is returned as an ordered list of existing subintervals.
    #
    # It calls _loc, which operates recursively. See _loc for more details on how the search works.

    def loc(self, ta, tb):
        out = []
        self._loc(ta, tb, out)
        return out

    def _loc(self, ta, tb, out):
        # Expected to have ta < tb

        # First, we (this interval) only have jurisdiction over [self._start, self._end]. So if we're asked for
        # something outside of that then we pass the buck up to our parent, who is strictly larger.
        if ta < self._start or tb > self._end:
            self._parent._loc(ta, tb, out)
            return

        # If it's us that's being asked for, then we add ourselves on to out and return.
        if ta == self._start and tb == self._end:
            out.append(self)
            return

        # If we've got this far then we know that it's an interval that's within our jurisdiction, and that it's not us.
        # So next we check if it's up to us to figure out, or up to our children.
        if self._midway is None:
            # It's up to us. Create subintervals (_split) if appropriate.
            if ta == self._start:
                self._split(tb)
                out.append(self._left_child)
                return
            # implies ta > self._start
            self._split(ta)
            # Query our (newly created) right_child: if tb == self._end then our right child will be the result, and it
            # will tell us so. But if tb < self._end then our right_child will need to make another split of its own.
            self._right_child._loc(ta, tb, out)
            return

        # If we're here then we have children: self._midway is not None
        if tb <= self._midway:
            # Strictly our left_child's problem
            self._left_child._loc(ta, tb, out)
            return
        if ta >= self._midway:
            # Strictly our right_child's problem
            self._right_child._loc(ta, tb, out)
            return
        # It's a problem for both of our children: the requested interval overlaps our midpoint. Call the left_child
        # first (to append to out in the correct order), then call our right child.
        # (Implies ta < self._midway < tb)
        self._left_child._loc(ta, self._midway, out)
        self._right_child._loc(self._midway, tb, out)

    def _split(self, midway):  # Create two children
        self._midway = midway

        # Use splittable PRNGs to generate noise.
        # TODO: spawning is slow (about half the runtime), because of the tuple addition to create the child's
        #  spawn_key: our spawn_key + (index,). Find a clever solution?
        left_W_generator, right_W_generator = self._W_generator.spawn(2)
        left_H_generator = right_H_generator = left_a_generator = right_a_generator = None
        # creating generators actually has nontrivial overhead so we avoid it if possible
        if self._top.have_H:
            left_H_generator, right_H_generator = self._H_generator.spawn(2)
        if self._top.have_A:
            left_a_generator, right_a_generator = self._a_generator.spawn(2)
        self._left_child = _Interval(start=self._start,
                                     end=midway,
                                     parent=self,
                                     is_left=True,
                                     top=self._top,
                                     W_generator=left_W_generator,
                                     H_generator=left_H_generator,
                                     a_generator=left_a_generator)
        self._right_child = _Interval(start=midway,
                                      end=self._end,
                                      parent=self,
                                      is_left=False,
                                      top=self._top,
                                      W_generator=right_W_generator,
                                      H_generator=right_H_generator,
                                      a_generator=right_a_generator)


class BrownianInterval(_Interval, base_brownian.BaseBrownian):
    """Brownian interval with fixed entropy.

    Computes increments (and optionally Levy area).

    To use:
    >>> bm = BrownianInterval(t0=0.0, t1=1.0, shape=(4, 1), dtype=torch.float32, device='cuda')
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    __slots__ = ('shape', 'dtype', 'device', '_w_h', '_entropy', '_dt', '_cache_size', 'levy_area_approximation',
                 '_last_interval', 'increment_and_space_time_levy_area_cache')

    def __init__(self,
                 t0: Scalar,
                 t1: Scalar,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 entropy: Optional[int] = None,
                 dt: Optional[Scalar] = None,
                 cache_size: Optional[int] = 45,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 W: Optional[torch.Tensor] = None,
                 H: Optional[torch.Tensor] = None,
                 **kwargs):
        """Initialize the Brownian interval.

        Args:
            t0: Initial time.
            t1: Terminal time.
            shape: The shape of each Brownian sample. The last dimension is
                treated as the channel dimension and any/all preceding
                dimensions are treated as batch dimensions.
            dtype: The dtype of each Brownian sample.
            device: The device of each Brownian sample.
            entropy: Global seed, defaults to `None` for random entropy.
            dt: The expected average step size of the SDE solver. Set it if you
                know it (e.g. when using a fixed solver); else it will default
                to equal the first step this is evaluated with. This allows us
                to set up a structure that should be efficient to query at these
                intervals.
            cache_size: How big a cache of recent calculations to use. (As new
                calculations depend on old calculations, this speeds things up
                dramatically, rather than recomputing things.) The default is
                set to be pretty close to optimum: smaller values imply more
                recalculation, whilst larger values imply more time spent
                keeping track of the cache.
            levy_area_approximation: Whether to also approximate Levy area.
                Defaults to None. Valid options are either 'none', 'space-time',
                'davie' or 'foster', corresponding to approximation type. This
                is needed for some higher-order SDE solvers.
            W: The increment of the Brownian motion over the interval [t0, t1].
                Will be generated randomly if not provided.
            H: The space-time Levy area of the Brownian motion over the interval
                [t0, t1]. Will be generated randomly if not provided.
        """

        if not utils.is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')
        if not utils.is_scalar(t1):
            raise ValueError('Terminal time t1 should be a float or 0-d torch.Tensor.')
        if dt is not None and not utils.is_scalar(dt):
            raise ValueError('Expected average time step dt should be a float or 0-d torch.Tensor.')

        if t0 > t1:
            raise ValueError(f'Initial time {t0} should be less than terminal time {t1}.')
        t0 = float(t0)
        t1 = float(t1)
        if dt is not None:
            dt = float(dt)

        shapes = []
        dtypes = []
        devices = []
        if shape is not None:
            shapes.append(shape)
        if dtype is not None:
            dtypes.append(dtype)
        if device is not None:
            devices.append(device)
        if torch.is_tensor(W):
            shapes.append(W.shape)
            dtypes.append(W.dtype)
            devices.append(W.device)
        if torch.is_tensor(H):
            shapes.append(H.shape)
            dtypes.append(H.dtype)
            devices.append(H.device)
        if len(shapes) == 0:
            raise ValueError("Must either specify `shape` or pass in `W` or `H` to implicity define the shape.")
        if len(dtypes) == 0:
            raise ValueError("Must either specify `dtype` or pass in `W` or `H` to implicity define the dtype.")
        if len(devices) == 0:
            raise ValueError("Must either specify `device` or pass in `W` or `H` to implicity define the device.")
        # Make sure the reduce actually does a comparison, to get a bool datatype
        shapes.append(shapes[-1])
        dtypes.append(dtypes[-1])
        devices.append(devices[-1])
        if not ft.reduce(operator.eq, shapes):
            raise ValueError(f"Multiple shapes found. Make sure whichever of `shape`, `W`, `H` that are passed are "
                             f"consistent. {shapes}")
        if not ft.reduce(operator.eq, dtypes):
            raise ValueError("Multiple dtypes found. Make sure whichever of `dtype`, `W`, `H` that are passed are "
                             "consistent.")
        if not ft.reduce(operator.eq, devices):
            raise ValueError("Multiple devices found. Make sure whichever of `device`, `W`, `H` that are passed are "
                             "consistent.")

        if entropy is None:
            entropy = random.randint(0, 2 ** 31 - 1)

        self.shape = tuple(shapes[0])  # convert from torch.Size if necessary
        self.dtype = dtypes[0]
        self.device = devices[0]
        self._entropy = entropy

        # The central piece of our implementation: an LRU cache on the calculations for increments and space-time Levy
        # area.
        @ft.lru_cache(cache_size)
        def increment_and_space_time_levy_area_cache(interval):
            return interval.increment_and_space_time_levy_area_uncached()

        self.increment_and_space_time_levy_area_cache = increment_and_space_time_levy_area_cache

        generator = np.random.SeedSequence(entropy=entropy)
        W_generator, H_generator, a_generator = generator.spawn(3)

        super(BrownianInterval, self).__init__(start=t0,
                                               end=t1,
                                               parent=None,
                                               is_left=None,
                                               top=self,
                                               W_generator=W_generator,
                                               H_generator=H_generator,
                                               a_generator=a_generator,
                                               **kwargs)

        if W is None:
            W = self._randn('W') * math.sqrt(t1 - t0)
        else:
            _assert_floating_tensor('W', W)
        if H is None:
            H = self._randn('H') * math.sqrt((t1 - t0) / 12)
        else:
            _assert_floating_tensor('H', H)
        self._w_h = (W, H)

        if levy_area_approximation not in LEVY_AREA_APPROXIMATIONS:
            raise ValueError(f"`levy_area_approximation` must be one of {LEVY_AREA_APPROXIMATIONS}.")

        self._dt = None
        self._cache_size = cache_size
        self.levy_area_approximation = levy_area_approximation

        # Precompute these as we don't want to spend lots of time checking strings in hot loops.
        self.have_H = self.levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.spacetime,
                                                       LEVY_AREA_APPROXIMATIONS.davie,
                                                       LEVY_AREA_APPROXIMATIONS.foster)
        self.have_A = self.levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.davie,
                                                       LEVY_AREA_APPROXIMATIONS.foster)

        # This is another nice trick.
        # We keep track of the most recently queried interval, and start searching for the next interval from that
        # element of the binary tree.
        self._last_interval = self

        if dt is not None:  # We'll take dt = first step if it's not passed here
            # We pre-create a binary tree dependency between the points. If we don't do this then the forward pass is
            # still efficient at O(N), but we end up with a dependency chain stretching along the interval [t0, t1],
            # making the backward pass O(N^2). By setting up a dependency tree of depth relative to `dt` and
            # `cache_size` we can instead make both directions O(N log N).
            self._create_dependency_tree(dt)

    # Effectively permanently store our increment and space-time Levy area in the cache.
    def _increment_and_space_time_levy_area(self):
        return self._w_h

    def __call__(self, ta, tb):
        ta = float(ta)
        tb = float(tb)
        # Can get queries just inside and outside the specified region in SDE solvers; we just clamp.
        ta = min(self.start, max(ta, self.end))
        tb = min(self.start, max(tb, self.end))
        if ta > tb:
            raise RuntimeError(f"Query times ta={ta:.3f} and tb={tb:.3f} must respect ta <= tb.")

        if ta == tb:
            W = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
            H = None
            A = None
            if self.have_H:
                H = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
            if self.have_A:
                shape = (*self.shape, *self.shape[-1:])  # not self.shape[-1] as that may not exist
                A = torch.zeros(shape, dtype=self.dtype, device=self.device)
        else:
            if self._dt is None:
                # If 'dt' wasn't specified, then take the first step as an estimate of the expected average step size
                self._create_dependency_tree(tb - ta)

            # Find the intervals that correspond to the query. We start our search at the last interval we accessed in
            # the binary tree, as it's likely that the next query will come nearby.
            intervals = self._last_interval.loc(ta, tb)
            # Ideally we'd keep track of intervals[0] on the backward pass. Practically speaking len(intervals) tends to
            # be 1 or 2 almost always so this isn't a huge deal.
            self._last_interval = intervals[-1]

            W, H, A = intervals[0].increment_and_levy_area()
            if len(intervals) > 1:
                # If we have multiple intervals then add up their increments and Levy areas.

                # Clone to avoid modifying the W, H, A that may exist in the cache
                W = W.clone()
                if self.have_H:
                    H = H.clone()
                if self.have_A:
                    A = A.clone()

                for interval in intervals[1:]:
                    Wi, Hi, Ai = interval.increment_and_levy_area()
                    if self.have_H:
                        H += Hi + (interval.end - interval.start) * W
                    if self.have_A and self.shape != ():
                        A += Ai + 0.5 * (W.unsqueeze(-1) * Wi.unsqueeze(-2) - Wi.unsqueeze(-1) * W.unsqueeze(-2))
                    W += Wi

        U = None
        if self.have_H:
            U = (tb - ta) * (H + 0.5 * W)

        if self.levy_area_approximation == LEVY_AREA_APPROXIMATIONS.none:
            return W
        elif self.levy_area_approximation == LEVY_AREA_APPROXIMATIONS.spacetime:
            return W, U
        return W, U, A

    def _create_dependency_tree(self, dt):
        self._dt = dt

        if self._cache_size is not None:  # cache_size=None corresponds to infinite cache.

            # Rationale: We are prepared to hold `cache_size` many things in memory, so when making steps of size `dt`
            # then we can afford to have the intervals at the bottom of our binary tree be of size `dt * cache_size`.
            # For safety we then make this a bit smaller by multiplying by 0.8.
            piece_length = dt * self._cache_size * 0.8

            def _set_t_cache(interval):
                start = interval._start
                end = interval._end
                if end - start > piece_length:
                    midway = (end + start) / 2
                    interval.loc(start, midway)
                    _set_t_cache(interval._left_child)
                    _set_t_cache(interval._right_child)

            _set_t_cache(self)

    def __repr__(self):
        if self._dt is None:
            dt = None
        else:
            dt = f"{self._dt:.3f}"
        return (f"{self.__class__.__name__}("
                f"t0={self._start:.3f}, "
                f"t1={self._end:.3f}, "
                f"shape={self.shape}, "
                f"dtype={self.dtype}, "
                f"device={self.device}, "
                f"entropy={self._entropy}, "
                f"dt={dt}, "
                f"cache_size={self._cache_size}, "
                f"levy_area_approximation={self.levy_area_approximation}"
                f")")

    def to(self, *args, **kwargs):
        self._w_h = tuple(v.to(*args, **kwargs) for v in self._w_h)
