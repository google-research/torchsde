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
import trampoline
import warnings

import boltons.cacheutils
import numpy as np
import torch

from . import brownian_base
from ..settings import LEVY_AREA_APPROXIMATIONS
from ..types import Scalar, Optional, Tuple, Union, Tensor

_rsqrt3 = 1 / math.sqrt(3)
_r12 = 1 / 12


def _randn(size, dtype, device, seed):
    generator = torch.Generator(device).manual_seed(int(seed))
    return torch.randn(size, dtype=dtype, device=device, generator=generator)


def _is_scalar(x):
    return isinstance(x, int) or isinstance(x, float) or (isinstance(x, torch.Tensor) and x.numel() == 1)


def _assert_floating_tensor(name, tensor):
    if not torch.is_tensor(tensor):
        raise ValueError(f"{name}={tensor} should be a Tensor.")
    if not tensor.is_floating_point():
        raise ValueError(f"{name}={tensor} should be floating point.")


def _check_tensor_info(*tensors, size, dtype, device):
    """Check if sizes, dtypes, and devices of input tensors all match prescribed values."""
    tensors = list(filter(torch.is_tensor, tensors))

    if dtype is None and len(tensors) == 0:
        dtype = torch.get_default_dtype()
    if device is None and len(tensors) == 0:
        device = torch.device("cpu")

    sizes = [] if size is None else [size]
    sizes += [t.shape for t in tensors]

    dtypes = [] if dtype is None else [dtype]
    dtypes += [t.dtype for t in tensors]

    devices = [] if device is None else [device]
    devices += [t.device for t in tensors]

    if len(sizes) == 0:
        raise ValueError(f"Must either specify `size` or pass in `W` or `H` to implicitly define the size.")

    if not all(i == sizes[0] for i in sizes):
        raise ValueError(f"Multiple sizes found. Make sure `size` and `W` or `H` are consistent.")
    if not all(i == dtypes[0] for i in dtypes):
        raise ValueError(f"Multiple dtypes found. Make sure `dtype` and `W` or `H` are consistent.")
    if not all(i == devices[0] for i in devices):
        raise ValueError(f"Multiple devices found. Make sure `device` and `W` or `H` are consistent.")

    # Make sure size is a tuple (not a torch.Size) for neat repr-printing purposes.
    return tuple(sizes[0]), dtypes[0], devices[0]


def _davie_foster_approximation(W, H, h, levy_area_approximation, get_noise):
    if levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.none, LEVY_AREA_APPROXIMATIONS.space_time):
        return None
    elif W.ndimension() in (0, 1):
        # If we have zero or one dimensions then treat the scalar / single dimension we have as batch, so that the
        # Brownian motion is one dimensional and the Levy area is zero.
        return torch.zeros_like(W)
    else:
        # Davie's approximation to the Levy area from space-time Levy area
        A = H.unsqueeze(-1) * W.unsqueeze(-2) - W.unsqueeze(-1) * H.unsqueeze(-2)
        noise = get_noise()
        noise = noise - noise.transpose(-1, -2)  # noise is skew symmetric of variance 2
        if levy_area_approximation == LEVY_AREA_APPROXIMATIONS.foster:
            # Foster's additional correction to Davie's approximation
            tenth_h = 0.1 * h
            H_squared = H ** 2
            std = (tenth_h * (tenth_h + H_squared.unsqueeze(-1) + H_squared.unsqueeze(-2))).sqrt()
        else:  # davie approximation
            std = math.sqrt(_r12 * h ** 2)
        a_tilde = std * noise
        A += a_tilde
        return A


def _H_to_U(W: torch.Tensor, H: torch.Tensor, h: float) -> torch.Tensor:
    return h * (.5 * W + H)


class _EmptyDict:
    def __setitem__(self, key, value):
        pass

    def __getitem__(self, item):
        raise KeyError


class _Interval:
    # Intervals correspond to some subinterval of the overall interval [t0, t1].
    # They are arranged as a binary tree: each node corresponds to an interval. If a node has children, they are left
    # and right subintervals, which partition the parent interval.

    __slots__ = (
                 # These are the things that every interval has
                 '_start',
                 '_end',
                 '_parent',
                 '_is_left',
                 '_top',
                 # These are the things that intervals which are parents also have
                 '_midway',
                 '_spawn_key',
                 '_depth',
                 '_W_seed',
                 '_H_seed',
                 '_left_a_seed',
                 '_right_a_seed',
                 '_left_child',
                 '_right_child')

    def __init__(self, start, end, parent, is_left, top):
        self._start = top._round(start)  # the left hand edge of the interval
        self._end = top._round(end)  # the right hand edge of the interval
        self._parent = parent  # our parent interval
        self._is_left = is_left  # are we the left or right child of our parent
        self._top = top  # the top-level BrownianInterval, where we cache certain state
        self._midway = None  # The point at which we split between left and right subintervals

    ########################################
    #  Calculate increments and levy area  #
    ########################################
    #
    # This is a little bit convoluted, so here's an explanation.
    #
    # The entry point is _increment_and_levy_area, below. This immediately calls _increment_and_space_time_levy_area,
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

    def _increment_and_levy_area(self):
        W, H = trampoline.trampoline(self._increment_and_space_time_levy_area())
        A = _davie_foster_approximation(W, H, self._end - self._start, self._top._levy_area_approximation,
                                        self._randn_levy)
        return W, H, A

    def _increment_and_space_time_levy_area(self):
        try:
            return self._top._increment_and_space_time_levy_area_cache[self]
        except KeyError:
            parent = self._parent

            W, H = yield parent._increment_and_space_time_levy_area()
            h_reciprocal = 1 / (parent._end - parent._start)
            left_diff = parent._midway - parent._start
            right_diff = parent._end - parent._midway

            if self._top._have_H:
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
        size = self._top._size
        return _randn(size, self._top._dtype, self._top._device, seed)

    def _a_seed(self):
        return self._parent._left_a_seed if self._is_left else self._parent._right_a_seed

    def _randn_levy(self):
        size = (*self._top._size, *self._top._size[-1:])
        return _randn(size, self._top._dtype, self._top._device, self._a_seed())

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

    def _loc(self, ta, tb):
        out = []
        ta = self._top._round(ta)
        tb = self._top._round(tb)
        trampoline.trampoline(self._loc_inner(ta, tb, out))
        return out

    def _loc_inner(self, ta, tb, out):
        # Expect to have ta < tb

        # First, we (this interval) only have jurisdiction over [self._start, self._end]. So if we're asked for
        # something outside of that then we pass the buck up to our parent, who is strictly larger.
        if ta < self._start or tb > self._end:
            raise trampoline.TailCall(self._parent._loc_inner(ta, tb, out))

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
                raise trampoline.TailCall(self._left_child._loc_inner(ta, tb, out))
            # implies ta > self._start
            self._split(ta)
            # Query our (newly created) right_child: if tb == self._end then our right child will be the result, and it
            # will tell us so. But if tb < self._end then our right_child will need to make another split of its own.
            raise trampoline.TailCall(self._right_child._loc_inner(ta, tb, out))

        # If we're here then we have children: self._midway is not None
        if tb <= self._midway:
            # Strictly our left_child's problem
            raise trampoline.TailCall(self._left_child._loc_inner(ta, tb, out))
        if ta >= self._midway:
            # Strictly our right_child's problem
            raise trampoline.TailCall(self._right_child._loc_inner(ta, tb, out))
        # It's a problem for both of our children: the requested interval overlaps our midpoint. Call the left_child
        # first (to append to out in the correct order), then call our right child.
        # (Implies ta < self._midway < tb)
        yield self._left_child._loc_inner(ta, self._midway, out)
        raise trampoline.TailCall(self._right_child._loc_inner(self._midway, tb, out))

    def _set_spawn_key_and_depth(self):
        self._spawn_key = 2 * self._parent._spawn_key + (0 if self._is_left else 1)
        self._depth = self._parent._depth + 1

    def _split(self, midway):
        if self._top._halfway_tree:
            self._split_exact(0.5 * (self._end + self._start))
            # self._midway is now the rounded halfway point.
            if midway > self._midway:
                self._right_child._split(midway)
            elif midway < self._midway:
                self._left_child._split(midway)
        else:
            self._split_exact(midway)

    def _split_exact(self, midway):  # Create two children
        self._midway = self._top._round(midway)
        # Use splittable PRNGs to generate noise.
        self._set_spawn_key_and_depth()
        generator = np.random.SeedSequence(entropy=self._top._entropy,
                                           spawn_key=(self._spawn_key, self._depth),
                                           pool_size=self._top._pool_size)
        self._W_seed, self._H_seed, self._left_a_seed, self._right_a_seed = generator.generate_state(4)

        self._left_child = _Interval(start=self._start,
                                     end=midway,
                                     parent=self,
                                     is_left=True,
                                     top=self._top)
        self._right_child = _Interval(start=midway,
                                      end=self._end,
                                      parent=self,
                                      is_left=False,
                                      top=self._top)


class BrownianInterval(brownian_base.BaseBrownian, _Interval):
    """Brownian interval with fixed entropy.

    Computes increments (and optionally Levy area).

    To use:
    >>> bm = BrownianInterval(t0=0.0, t1=1.0, size=(4, 1), device='cuda')
    >>> bm(0., 0.5)
    tensor([[ 0.0733],
            [-0.5692],
            [ 0.1872],
            [-0.3889]], device='cuda:0')
    """

    __slots__ = (
                 # Inputs
                 '_size',
                 '_dtype',
                 '_device',
                 '_entropy',
                 '_levy_area_approximation',
                 '_dt',
                 '_tol',
                 '_pool_size',
                 '_cache_size',
                 '_halfway_tree',
                 # Quantisation
                 '_round',
                 # Caching, searching and computing values
                 '_increment_and_space_time_levy_area_cache',
                 '_last_interval',
                 '_have_H',
                 '_have_A',
                 '_w_h',
                 '_top_a_seed',
                 # Dependency tree creation
                 '_average_dt',
                 '_tree_dt',
                 '_num_evaluations'
                 )

    def __init__(self,
                 t0: Optional[Scalar] = 0.,
                 t1: Optional[Scalar] = 1.,
                 size: Optional[Tuple[int, ...]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 entropy: Optional[int] = None,
                 dt: Optional[Scalar] = None,
                 tol: Scalar = 0.,
                 pool_size: int = 8,
                 cache_size: Optional[int] = 45,
                 halfway_tree: bool = False,
                 levy_area_approximation: str = LEVY_AREA_APPROXIMATIONS.none,
                 W: Optional[Tensor] = None,
                 H: Optional[Tensor] = None):
        """Initialize the Brownian interval.

        Args:
            t0 (float or Tensor): Initial time.
            t1 (float or Tensor): Terminal time.
            size (tuple of int): The shape of each Brownian sample.
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
            levy_area_approximation (str): Whether to also approximate Levy
                area. Defaults to 'none'. Valid options are 'none',
                'space-time', 'davie' or 'foster', corresponding to different
                approximation types.
                This is needed for some higher-order SDE solvers.
            dt (float or Tensor): The expected average step size of the SDE
                solver. Set it if you know it (e.g. when using a fixed-step
                solver); else it will be estimated from the first few queries.
                This is used to set up the data structure such that it is
                efficient to query at these intervals.
            tol (float or Tensor): What tolerance to resolve the Brownian motion
                to. Must be non-negative. Defaults to zero, i.e. floating point
                resolution. Usually worth setting in conjunction with
                `halfway_tree`, below.
            pool_size (int): Size of the pooled entropy. If you care about
                statistical randomness then increasing this will help (but will
                slow things down).
            cache_size (int): How big a cache of recent calculations to use.
                (As new calculations depend on old calculations, this speeds
                things up dramatically, rather than recomputing things.)
                Set this to `None` to use an infinite cache, which will be fast
                but memory inefficient.
            halfway_tree (bool): Whether the dependency tree (the internal data
                structure) should be the dyadic tree. Defaults to `False`.
                Normally, the sample path is determined by both `entropy`,
                _and_ the locations and order of the query points. Setting this
                 to `True` will make it deterministic with respect to just
                 `entropy`; however this is much slower.
            W (Tensor): The increment of the Brownian motion over the interval
                [t0, t1]. Will be generated randomly if not provided.
            H (Tensor): The space-time Levy area of the Brownian motion over the
                interval [t0, t1]. Will be generated randomly if not provided.
        """

        #####################################
        #    Check and normalise inputs     #
        #####################################

        if not _is_scalar(t0):
            raise ValueError('Initial time t0 should be a float or 0-d torch.Tensor.')
        if not _is_scalar(t1):
            raise ValueError('Terminal time t1 should be a float or 0-d torch.Tensor.')
        if dt is not None and not _is_scalar(dt):
            raise ValueError('Expected average time step dt should be a float or 0-d torch.Tensor.')

        if t0 > t1:
            raise ValueError(f'Initial time {t0} should be less than terminal time {t1}.')
        t0 = float(t0)
        t1 = float(t1)
        if dt is not None:
            dt = float(dt)

        if halfway_tree:
            if tol <= 0.:
                raise ValueError("`tol` should be positive.")
            if dt is not None:
                raise ValueError("`dt` is not used and should be set to `None` if `halfway_tree` is True.")
        else:
            if tol < 0.:
                raise ValueError("`tol` should be non-negative.")

        size, dtype, device = _check_tensor_info(W, H, size=size, dtype=dtype, device=device)

        # Let numpy dictate randomness, so we have fewer seeds to set for reproducibility.
        if entropy is None:
            entropy = np.random.randint(0, 2 ** 31 - 1)

        if levy_area_approximation not in LEVY_AREA_APPROXIMATIONS:
            raise ValueError(f"`levy_area_approximation` must be one of {LEVY_AREA_APPROXIMATIONS}, but got "
                             f"'{levy_area_approximation}'.")

        #####################################
        #          Record inputs            #
        #####################################

        self._size = size
        self._dtype = dtype
        self._device = device
        self._entropy = entropy
        self._levy_area_approximation = levy_area_approximation
        self._dt = dt
        self._tol = tol
        self._pool_size = pool_size
        self._cache_size = cache_size
        self._halfway_tree = halfway_tree

        #####################################
        #   A miscellany of other things    #
        #####################################

        # We keep a cache of recent queries, and their results. This is very important for speed, so that we don't
        # recurse all the way up to the top every time we have a query.
        if cache_size is None:
            self._increment_and_space_time_levy_area_cache = {}
        elif cache_size == 0:
            self._increment_and_space_time_levy_area_cache = _EmptyDict()
        else:
            self._increment_and_space_time_levy_area_cache = boltons.cacheutils.LRU(max_size=cache_size)

        # We keep track of the most recently queried interval, and start searching for the next interval from that
        # element of the binary tree. This is because subsequent queries are likely to be near the most recent query.
        self._last_interval = self

        # Precompute these as we don't want to spend lots of time checking strings in hot loops.
        self._have_H = self._levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.space_time,
                                                         LEVY_AREA_APPROXIMATIONS.davie,
                                                         LEVY_AREA_APPROXIMATIONS.foster)
        self._have_A = self._levy_area_approximation in (LEVY_AREA_APPROXIMATIONS.davie,
                                                         LEVY_AREA_APPROXIMATIONS.foster)

        # If we like we can quantise what level we want to compute the Brownian motion to.
        if tol == 0.:
            self._round = lambda x: x
        else:
            ndigits = -int(math.log10(tol))
            self._round = lambda x: round(x, ndigits)

        # Initalise as _Interval.
        # (Must come after _round but before _w_h)
        super(BrownianInterval, self).__init__(start=t0,
                                               end=t1,
                                               parent=None,
                                               is_left=None,
                                               top=self)

        # Set the global increment and space-time Levy area
        generator = np.random.SeedSequence(entropy=entropy, pool_size=pool_size)
        initial_W_seed, initial_H_seed, top_a_seed = generator.generate_state(3)
        if W is None:
            W = self._randn(initial_W_seed) * math.sqrt(t1 - t0)
        else:
            _assert_floating_tensor('W', W)
        if H is None:
            H = self._randn(initial_H_seed) * math.sqrt((t1 - t0) / 12)
        else:
            _assert_floating_tensor('H', H)
        self._w_h = (W, H)
        self._top_a_seed = top_a_seed

        if not self._halfway_tree:
            # We create a binary tree dependency between the points. If we don't do this then the forward pass is still
            # efficient at O(N), but we end up with a dependency chain stretching along the interval [t0, t1], making
            # the backward pass O(N^2). By setting up a dependency tree of depth relative to `dt` and `cache_size` we
            # can instead make both directions O(N log N).
            self._average_dt = 0
            self._tree_dt = t1 - t0
            self._num_evaluations = -100  # start off with a warmup period to get a decent estimate of the average
            if dt is not None:
                # Create the dependency tree based on the supplied hint `dt`.
                self._create_dependency_tree(dt)
            # If dt is None, then create the dependency tree based on observed statistics of query points. (In __call__)

    # Effectively permanently store our increment and space-time Levy area in the cache.
    def _increment_and_space_time_levy_area(self):
        return self._w_h
        yield  # make it a generator

    def _a_seed(self):
        return self._top_a_seed

    def _set_spawn_key_and_depth(self):
        self._spawn_key = 0
        self._depth = 0

    def __call__(self, ta, tb=None, return_U=False, return_A=False):
        if tb is None:
            warnings.warn(f"{self.__class__.__name__} is optimised for interval-based queries, not point evaluation.")
            ta, tb = self._start, ta
            tb_name = 'ta'
        else:
            tb_name = 'tb'
        ta = float(ta)
        tb = float(tb)
        if ta < self._start:
            warnings.warn(f"Should have ta>=t0 but got ta={ta} and t0={self._start}.")
            ta = self._start
        if tb < self._start:
            warnings.warn(f"Should have {tb_name}>=t0 but got {tb_name}={tb} and t0={self._start}.")
            tb = self._start
        if ta > self._end:
            warnings.warn(f"Should have ta<=t1 but got ta={ta} and t1={self._end}.")
            ta = self._end
        if tb > self._end:
            warnings.warn(f"Should have {tb_name}<=t1 but got {tb_name}={tb} and t1={self._end}.")
            tb = self._end
        if ta > tb:
            raise RuntimeError(f"Query times ta={ta:.3f} and tb={tb:.3f} must respect ta <= tb.")

        if ta == tb:
            W = torch.zeros(self._size, dtype=self._dtype, device=self._device)
            H = None
            A = None
            if self._have_H:
                H = torch.zeros(self._size, dtype=self._dtype, device=self._device)
            if self._have_A:
                size = (*self._size, *self._size[-1:])  # not self._size[-1] as that may not exist
                A = torch.zeros(size, dtype=self._dtype, device=self._device)
        else:
            if self._dt is None and not self._halfway_tree:
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
            intervals = self._last_interval._loc(ta, tb)
            # Ideally we'd keep track of intervals[0] on the backward pass. Practically speaking len(intervals) tends to
            # be 1 or 2 almost always so this isn't a huge deal.
            self._last_interval = intervals[-1]

            W, H, A = intervals[0]._increment_and_levy_area()
            if len(intervals) > 1:
                # If we have multiple intervals then add up their increments and Levy areas.

                for interval in intervals[1:]:
                    Wi, Hi, Ai = interval._increment_and_levy_area()
                    if self._have_H:
                        # Aggregate H:
                        # Given s < u < t, then
                        # H_{s,t} = (term1 + term2) / (t - s)
                        # where
                        # term1 = (t - u) * (H_{u, t} + W_{s, u} / 2)
                        # term2 = (u - s) * (H_{s, u} - W_{u, t} / 2)
                        term1 = (interval._end - interval._start) * (Hi + 0.5 * W)
                        term2 = (interval._start - ta) * (H - 0.5 * Wi)
                        H = (term1 + term2) / (interval._end - ta)
                    if self._have_A and len(self._size) not in (0, 1):
                        # If len(self._size) in (0, 1) then we treat our scalar / single dimension as a batch
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
        if self._have_H:
            U = _H_to_U(W, H, tb - ta)

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
        # For safety we take a min with 100: if people take very large cache sizes then this would then break the
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
                interval._loc(start, midway)
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
                f"size={self._size}, "
                f"dtype={self._dtype}, "
                f"device={repr(self._device)}, "
                f"entropy={self._entropy}, "
                f"dt={dt}, "
                f"tol={self._tol}, "
                f"pool_size={self._pool_size}, "
                f"cache_size={self._cache_size}, "
                f"levy_area_approximation={repr(self._levy_area_approximation)}"
                f")")

    def display_binary_tree(self):
        stack = [(self, 0)]
        out = []
        while len(stack):
            elem, depth = stack.pop()
            out.append(" " * depth + f"({elem._start}, {elem._end})")
            if elem._midway is not None:
                stack.append((elem._right_child, depth + 1))
                stack.append((elem._left_child, depth + 1))
        print("\n".join(out))

    @property
    def shape(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def entropy(self):
        return self._entropy

    @property
    def levy_area_approximation(self):
        return self._levy_area_approximation

    @property
    def dt(self):
        return self._dt

    @property
    def tol(self):
        return self._tol

    @property
    def pool_size(self):
        return self._pool_size

    @property
    def cache_size(self):
        return self._cache_size

    @property
    def halfway_tree(self):
        return self._halfway_tree

    def size(self):
        return self._size
