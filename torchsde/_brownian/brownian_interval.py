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

import math
import warnings

import boltons.cacheutils
import numpy as np
import torch

from . import base_brownian
from . import utils
from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar, Optional, Tuple, Union, Tensor

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

    __slots__ = ('_start', '_end', '_parent', '_is_left', '_top', '_generator', '_W_seed', '_H_seed', '_a_seed',
                 '_midway', '_left_child', '_right_child')

    def __init__(self, start, end, parent, is_left, top, generator):
        # These are the things that every interval has
        self._start = start  # the left hand edge of the interval
        self._end = end  # the right hand edge of the interval
        self._parent = parent  # our parent interval
        self._is_left = is_left  # are we the left or right child of our parent
        self._top = top  # the top-level BrownianInterval, where we cache certain state
        self._generator = generator
        self._W_seed, self._H_seed, self._a_seed = generator.generate_state(3)

        # These are the things that intervals which are parents also have
        self._midway = None  # The point at which we split between left and right subintervals
        self._left_child = None  # The left subinterval
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
    # (The top-level BrownianInterval overrides _increment_and_space_time_levy_area to return its own increment and
    # space-time Levy area, effectively holding them permanently in the cache.)
    #
    # If the request isn't found in the LRU cache then it computes it from its parent.
    # Now it turns out that the size of our increment and space-time Levy area is really most naturally thought of as a
    # property of our parent: it depends on our parent's increment, space-time Levy area, and whether we are the left or
    # right interval within our parent. So _increment_and_space_time_levy_area in turn checks if we are on the
    # left or right of our parent and does most of the computation using the parent's attributes.

    def increment_and_levy_area(self):
        W, H = self._increment_and_space_time_levy_area()
        A = utils.davie_foster_approximation(W, H, self._end - self._start, self._top.levy_area_approximation,
                                             self._randn_levy)
        return W, H, A

    def _increment_and_space_time_levy_area(self):
        # TODO: switch this over to a trampoline?

        # It's quite important that this whole block of code be inline, without any additional function calls between
        # it and calling parent._increment_and_space_time_levy_area().
        # The recursion can in normal usage grow quite large - and if there's any additional stack frames in the way,
        # large enough to violate the default recursion limit.
        # This is also the reason we have an inlined LRU cache rather than wrapping with functools.lru_cache.

        try:
            return self._top._increment_and_space_time_levy_area_cache[self]
        except KeyError:
            parent = self._parent

            W, H = parent._increment_and_space_time_levy_area()
            h_reciprocal = 1 / (parent._end - parent._start)
            left_diff = parent._midway - parent._start
            right_diff = parent._end - parent._midway

            if self._top.have_H:
                left_diff_squared = left_diff ** 2
                right_diff_squared = right_diff ** 2
                left_diff_cubed = left_diff * left_diff_squared
                right_diff_cubed = right_diff * right_diff_squared

                v = 0.5 * math.sqrt(left_diff * right_diff / (left_diff_cubed + right_diff_cubed))

                a = v * left_diff_squared * h_reciprocal
                b = v * right_diff_squared * h_reciprocal
                c = v * _rsqrt3

                X1 = parent._randn(parent._W_seed)
                X2 = parent._randn(parent._H_seed)

                third_coeff = 2 * (a * left_diff + b * right_diff) * h_reciprocal

                if self._is_left:
                    first_coeff = left_diff * h_reciprocal
                    second_coeff = 6 * first_coeff * right_diff * h_reciprocal
                    out_W = first_coeff * W + second_coeff * H + third_coeff * X1
                    out_H = first_coeff ** 2 * H - a * X1 + c * right_diff * X2
                else:
                    first_coeff = right_diff * h_reciprocal
                    second_coeff = 6 * first_coeff * left_diff * h_reciprocal
                    out_W = first_coeff * W - second_coeff * H - third_coeff * X1
                    out_H = first_coeff ** 2 * H - b * X1 - c * left_diff * X2
            else:
                # Don't compute space-time Levy area unless we need to

                mean = left_diff * W * h_reciprocal
                var = left_diff * right_diff * h_reciprocal
                noise = parent._randn(parent._W_seed)
                left_W = mean + math.sqrt(var) * noise

                if self._is_left:
                    out_W = left_W
                else:
                    out_W = W - left_W
                out_H = None

            self._top._increment_and_space_time_levy_area_cache[self] = (out_W, out_H)
            return out_W, out_H

    def _randn(self, seed):
        # We generate random noise deterministically wrt some seed; this seed is determined by the generator.
        # This means that if we drop out of the cache, then we'll create the same random noise next time, as we still
        # have the generator.
        shape = self._top.shape
        return _randn(shape, self._top.dtype, self._top.device, seed)

    def _randn_levy(self):
        shape = (*self._top.shape, *self._top.shape[-1:])
        return _randn(shape, self._top.dtype, self._top.device, self._a_seed)

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
        # Expect to have ta < tb

        # TODO: switch this over to a trampoline w/ tail recursion?

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

        # TODO: put generator creation for 'self' here, not for the children

        # Use splittable PRNGs to generate noise.
        # TODO: spawning is slow (about half the runtime), because of the tuple addition to create the child's
        #  spawn_key: our spawn_key + (index,). Find a clever solution?
        left_generator, right_generator = self._generator.spawn(2)
        self._left_child = _Interval(start=self._start,
                                     end=midway,
                                     parent=self,
                                     is_left=True,
                                     top=self._top,
                                     generator=left_generator)
        self._right_child = _Interval(start=midway,
                                      end=self._end,
                                      parent=self,
                                      is_left=False,
                                      top=self._top,
                                      generator=right_generator)


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
                 '_last_interval', '_increment_and_space_time_levy_area_cache')

    def __init__(self,
                 t0: Scalar,
                 t1: Scalar,
                 shape: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 entropy: Optional[int] = None,
                 dt: Optional[Scalar] = None,
                 pool_size: int = 8,
                 cache_size: Optional[int] = 45,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 W: Optional[Tensor] = None,
                 H: Optional[Tensor] = None,
                 **kwargs):
        """Initialize the Brownian interval.

        Args:
            t0 (float or Tensor): Initial time.
            t1 (float or Tensor): Terminal time.
            shape (tuple of int): The shape of each Brownian sample.
                If zero dimensional represents a scalar Brownian motion.
                If one dimensional represents a batch of scalar Brownian motions.
                If >two dimensional the last dimension represents the size of a
                a multidimensional Brownian motion, and all previous dimensions
                represent batch dimensions.
            dtype (torch.dtype): The dtype of each Brownian sample.
                Defaults to the PyTorch default.
            device (str or torch.device): The device of each Brownian sample.
                Defaults to the CPU.
            entropy (int): Global seed, defaults to `None` for random entropy.
            dt (float or Tensor): The expected average step size of the SDE
                solver. Set it if you know it (e.g. when using a fixed-step
                solver); else it will default to equal the first step this is
                evaluated with. This allows us to set up a structure that should
                be efficient to query at these intervals.
            pool_size (int): Size of the pooled entropy. If you care about
                statistical randomness then increasing this will help (but will
                slow things down).
            cache_size (int): How big a cache of recent calculations to use.
                (As new calculations depend on old calculations, this speeds
                things up dramatically, rather than recomputing things.)
                The default is set to be pretty close to the optimum: smaller
                values imply more recalculation, whilst larger values imply
                more time spent keeping track of the cache.
            levy_area_approximation (str): Whether to also approximate Levy
                area. Defaults to 'none'. Valid options are 'none',
                'space-time', 'davie' or 'foster', corresponding to different
                approximation types.
                This is needed for some higher-order SDE solvers.
            W (Tensor): The increment of the Brownian motion over the interval
                [t0, t1]. Will be generated randomly if not provided.
            H (Tensor): The space-time Levy area of the Brownian motion over the
                interval [t0, t1]. Will be generated randomly if not provided.
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

        shape, dtype, device = utils.check_tensor_info(W, H, shape=shape, dtype=dtype, device=device,
                                                       name='`W` or `H`')

        # Let numpy dictate randomness, so we have fewer seeds to set for reproducibility.
        if entropy is None:
            entropy = np.random.randint(0, 2 ** 31 - 1)

        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._entropy = entropy

        self._increment_and_space_time_levy_area_cache = boltons.cacheutils.LRU(max_size=cache_size)

        generator = np.random.SeedSequence(entropy=entropy, pool_size=pool_size)
        # First three states are reserved as in _Initial
        _, _, _, initial_W_seed, initial_H_seed = generator.generate_state(5)

        super(BrownianInterval, self).__init__(start=t0,
                                               end=t1,
                                               parent=None,
                                               is_left=None,
                                               top=self,
                                               generator=generator,
                                               **kwargs)

        if W is None:
            W = self._randn(initial_W_seed) * math.sqrt(t1 - t0)
        else:
            _assert_floating_tensor('W', W)
        if H is None:
            H = self._randn(initial_H_seed) * math.sqrt((t1 - t0) / 12)
        else:
            _assert_floating_tensor('H', H)
        self._w_h = (W, H)

        if levy_area_approximation not in LEVY_AREA_APPROXIMATIONS:
            raise ValueError(f"`levy_area_approximation` must be one of {LEVY_AREA_APPROXIMATIONS}, but got "
                             f"'{levy_area_approximation}'.")

        self._dt = dt
        self._cache_size = cache_size
        self.levy_area_approximation = levy_area_approximation

        # Precompute these as we don't want to spend lots of time checking strings in hot loops.
        self.have_H = self.levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.space_time,
                                                       LEVY_AREA_APPROXIMATIONS.davie,
                                                       LEVY_AREA_APPROXIMATIONS.foster)
        self.have_A = self.levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.davie,
                                                       LEVY_AREA_APPROXIMATIONS.foster)

        # This is another nice trick.
        # We keep track of the most recently queried interval, and start searching for the next interval from that
        # element of the binary tree.
        self._last_interval = self

        self._average_dt = 0
        self._tree_dt = t1 - t0
        self._num_evaluations = -100  # start off with a warmup period to get a decent estimate of the average
        if dt is not None:
            # We pre-create a binary tree dependency between the points. If we don't do this then the forward pass is
            # still efficient at O(N), but we end up with a dependency chain stretching along the interval [t0, t1],
            # making the backward pass O(N^2). By setting up a dependency tree of depth relative to `dt` and
            # `cache_size` we can instead make both directions O(N log N).
            self._create_dependency_tree(dt)

    # Effectively permanently store our increment and space-time Levy area in the cache.
    def _increment_and_space_time_levy_area(self):
        return self._w_h

    # TODO: pick better names for return_U, return_A. A should be called 'levy_area', but what about U?
    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        if tb is None:
            warnings.warn(f"{self.__class__.__name__} is optimised for interval-based queries, not point evaluation. "
                          f"Consider using BrownianPath or BrownianTree instead.")
            ta, tb = self.start, ta
            tb_name = 'ta'
        else:
            tb_name = 'tb'
        ta = float(ta)
        tb = float(tb)
        if ta < self.start:
            warnings.warn(f"Should have ta>=t0 but got ta={ta} and t0={self.start}.")
            ta = self.start
        if tb < self.start:
            warnings.warn(f"Should have {tb_name}>=t0 but got {tb_name}={tb} and t0={self.start}.")
            tb = self.start
        if ta > self.end:
            warnings.warn(f"Should have ta<=t1 but got ta={ta} and t1={self.end}.")
            ta = self.end
        if tb > self.end:
            warnings.warn(f"Should have {tb_name}<=t1 but got {tb_name}={tb} and t1={self.end}.")
            tb = self.end
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
                self._num_evaluations += 1
                # We start off with "negative" num evaluations, to give us a small warm-up period at the start.
                if self._num_evaluations > 0:
                    # Compute average step size so far
                    dt = tb - ta
                    self._average_dt = (dt + self._average_dt * (self._num_evaluations - 1)) / self._num_evaluations
                    if self._average_dt < 0.5 * self._tree_dt:
                        # If 'dt' wasn't specified, then check the average interval length against the size of the
                        # bottom of the dependency tree. If we're below halfway then refine the tree by splitting all
                        # the bottom pieces into two.
                        self._create_dependency_tree(dt)

            # Find the intervals that correspond to the query. We start our search at the last interval we accessed in
            # the binary tree, as it's likely that the next query will come nearby.
            intervals = self._last_interval.loc(ta, tb)
            # Ideally we'd keep track of intervals[0] on the backward pass. Practically speaking len(intervals) tends to
            # be 1 or 2 almost always so this isn't a huge deal.
            self._last_interval = intervals[-1]

            W, H, A = intervals[0].increment_and_levy_area()
            if len(intervals) > 1:
                # If we have multiple intervals then add up their increments and Levy areas.

                for interval in intervals[1:]:
                    Wi, Hi, Ai = interval.increment_and_levy_area()
                    if self.have_H:
                        # Aggregate H:
                        # Given s < u < t, then
                        # H_{s,t} = (term1 + term2) / (t - s)
                        # where
                        # term1 = (t - u) * (H_{u, t} + W_{s, u} / 2)
                        # term2 = (u - s) * (H_{s, u} - W_{u, t} / 2)
                        term1 = (interval.end - interval.start) * (Hi + 0.5 * W)
                        term2 = (interval.start - ta) * (H - 0.5 * Wi)
                        H = (term1 + term2) / (interval.end - ta)
                    if self.have_A and len(self.shape) not in (0, 1):
                        # If len(self.shape) in (0, 1) then we treat our scalar / single dimension as a batch
                        # dimension, so we have zero Levy area. (And these unsqueezes will result in a tensor of shape
                        # (batch, batch) which is wrong.)

                        # Let B_{x, y} = \int_x^y W^1_{s,u} dW^2_u.
                        # Then
                        # B_{s, t} = \int_s^t W^1_{s,u} dW^2_u
                        #          = \int_s^v W^1_{s,u} dW^2_u + \int_v^t W^1_{s,v} dW^2_u + \int_v^t W^1_{v,u} dW^2_u
                        #          = B_{s, v} + W^1_{s, v} W^2_{v, t} + B_{v, t}
                        #
                        # A is now the antisymmetric part of B, which gives the formula below.
                        A = A + Ai + 0.5 * (W.unsqueeze(-1) * Wi.unsqueeze(-2) - Wi.unsqueeze(-1) * W.unsqueeze(-2))
                    W = W + Wi

        U = None
        if self.have_H:
            U = utils.H_to_U(W, H, tb - ta)

        if return_U:
            if return_A:
                return W, U, A
            else:
                return W, U
        else:
            if return_A:
                return W, A
            else:
                return W

    def _create_dependency_tree(self, dt):
        # For safety we take a max with 100: if people take very large cache sizes then this would then break the
        # logarithmic into linear, which causes RecursionErrors.
        if self._cache_size is None:  # cache_size=None corresponds to infinite cache.
            cache_size = 100
        else:
            cache_size = min(self._cache_size, 100)

        self._tree_dt = min(self._tree_dt, dt)
        # Rationale: We are prepared to hold `cache_size` many things in memory, so when making steps of size `dt`
        # then we can afford to have the intervals at the bottom of our binary tree be of size `dt * cache_size`.
        # For safety we then make this a bit smaller by multiplying by 0.8.
        piece_length = self._tree_dt * cache_size * 0.8

        def _set_points(interval):
            start = interval._start
            end = interval._end
            if end - start > piece_length:
                midway = (end + start) / 2
                interval.loc(start, midway)
                _set_points(interval._left_child)
                _set_points(interval._right_child)

        _set_points(self)

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
                f"device={repr(self.device)}, "
                f"entropy={self._entropy}, "
                f"dt={dt}, "
                f"cache_size={self._cache_size}, "
                f"levy_area_approximation={repr(self.levy_area_approximation)}"
                f")")

    def to(self, *args, **kwargs):
        raise AttributeError(f"BrownianInterval does not support the method `to`.")
