# Documentation
## Mathematics of SDEs
The library provides functionality to integrate the SDE
```
dy(t) = f(t, y(t)) dt + g(t, y(t)) dW(t)        y(t0) = y0 
```
where:
- `t` is the _time_, a scalar.
- `y(t)` is the _state_, a vector of size `n`,
- `f(t, y(t))` is the _drift_, a vector of size `n`,
- `g(t, y(t))` is the _diffusion_,  a matrix of size `n, m`,
- `W(t)` is _Brownian motion_, a vector of size `m`.

## The main functionality: `sdeint`

This SDE is solved using `sdeint`:
```python
from torchsde import sdeint
ys = sdeint(sde, y0, ts)
```
where
- `sde` is a `torch.nn.Module` with
    - an attribute `sde_type`, which should either be `"ito"` or `"stratonovich"`,
    - an attribute `noise_type`, which should either be `"scalar"`, `"additive"`, `"diagonal"`, `"general"`,
    - a method `f(t, y)` corresponding to the drift, producing a tensor of shape `(batch_size, state_size)`,
    - a method `g(t, y)` corresponding to the diffusion. The appropriate output shape depends on the noise type.
- `y0` is a tensor of shape `(batch_size, state_size)`, giving the initial value of the SDE at time `ts[0]`, 
- `ts` is a tensor of times to output `y` at, of shape `(t_size,)`. The SDE will be solved over the interval `[ts[0], ts[-1]]`.

Several keyword arguments are also accepted, see [below](#keyword-arguments-of-sdeint).

The output tensor `ys` will have shape `(t_size, batch_size, state_size)`, and corresponds to a sample from the SDE.

### Possible noise types
- `scalar`: The diffusion `g` has output size `(batch_size, state_size, 1)`. The Brownian motion is a batch of 1-dimensional Brownian motions.
- `additive`: The diffusion `g` is assumed to be constant w.r.t. `y` and has output size `(batch_size, state_size, brownian_size)`. The Brownian motion is a batch of `browian_size`-dimensional Brownian motions.
- `diagonal`: The diffusion `g` is element-wise and has output size `(batch_size, state_size)`. The Brownian motion is a batch of `state_size`-dimensional Brownian motions.
- `general`: The diffusion `g` has output size `(batch_size, state_size, brownian_size)`. The Brownian motion is a batch of `brownian_size`-dimensional Brownian motions.

### Keyword arguments for `sdeint`
- `bm`: A `BrownianInterval` object, see [below](#brownian-motion). Optionally include to control the Brownian motion.
- `method`: A string, corresponding to one of the solvers listed [below](#list-of-sde-solvers).
- `dt`: A float for the constant step size, or initial step size for adaptive time-stepping. Defaults to `1e-3`.
- `adaptive`: If True, use adaptive time-stepping. Defaults to False.
- `rtol`: Relative tolerance for adaptive time-stepping. Defaults to `1e-5`.
- `atol`: Absolute tolerance for adaptive time-stepping. Defaults to `1e-4`.
- `dt_min`: Minimum step size for adaptive time-stepping. Defaults to `1e-5`.
- `names`: A dictionary giving alternate names for the drift and diffusion methods. Acceptable keys are `"drift"` and `"diffusion"`. For example `{"drift": "foo"}` will use `foo` instead of `f`.

### List of SDE solvers
The available solvers depends on the SDE type and the noise type.

**Ito solvers**
- `"euler"`: [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)
- `"milstein"`: [Milstein method](https://en.wikipedia.org/wiki/Milstein_method)
- `"srk"`: [Stochastic Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_method_(SDE))

**Stratonovich solvers**
- `"euler_heun"`: [Euler-Heun method](https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf)
- `"heun"`: [Heun's method](https://arxiv.org/abs/1102.4401)
- `"midpoint"`: Midpoint method
- `"milstein"`: [Milstein method](https://en.wikipedia.org/wiki/Milstein_method)

Note that Milstein and SRK don't support general noise.

Additionally, [gradient-free Milstein](https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf) can be used by selecting Milstein, and then also passing in the keyword argument `sdeint(..., options=dict(grad_free=True))`.


### Providing specialised methods
If your drift/diffusion have special structure, for example the drift and diffusion share some computations, then it may be more efficient to evaluate them together rather than alone. As such, if the following methods are present on `sde`, then they will be used if possible: `g_prod(t, y, v)`, `f_and_g(t, y)`, `f_and_g_prod(t, y, v)`. Here `g_prod` is expected to compute the batch matrix-vector product between the diffusion and the vector `v`. `f_and_*` should return a 2-tuple of `f(t, y)` and `g(t, y)`/`g_prod(t, y, v)` as appropriate.

Depending on the integration method used it may suffice to provide only some of these methods (`f` and `g` are not mandatory).

## Adjoints

The SDE solve may be backpropagated through. This may be done either by backpropagating through the internal operations of the solver (when using `sdeint`), or via the _adjoint method_ [\[1\]](https://arxiv.org/pdf/2001.01328.pdf), which solves another "adjoint SDE" backwards in time to calculate the gradients (when using `sdeint_adjoint`).

Using the adjoint method will reduce memory usage, but takes (much) longer to compute. In addition, because the backward SDE is solved numerically, there will be a certain amount of numerical error, which can be nontrivial for large step sizes.

If using the adjoint method, then we recommend using Stratonovich SDEs if possible, as these are faster to calculate the adjoint SDE for. We also recommend using adaptive step sizes on both the forward and backward passes, as this will help to reduce the numerical error.

`sdeint_adjoint` supports all the arguments and keyword arguments that `sdeint` does, as well as the following additional keyword arguments: 
- `adjoint_method`, `adjoint_adaptive`, `adjoint_rtol`, `adjoint_atol`: as for the forward pass.  

- `adjoint_params`: the tensors to calculate gradients with respect to. If not passed, defaults to `sde.parameters()`.

Double backward is supported through the adjoint method, and is calculated using an adjoint-of-adjoint SDE, with the same method, tolerances, etc. as for the adjoint SDE. Note that the numerical errors can easily grow very large for adjoint-of-adjoint SDEs.


## Brownian motion

The `bm` argument to `sdeint` and `sdeint_adjoint` allows for tighter control on the Brownian motion. This should be a `torchsde.BrownianInterval` object. In particular, this allows for fixing its random seed (to produce the same Brownian motion every time), and for adjusting certain parameters that may affect its speed or memory usage.

`BrownianInterval` can also be used as a standalone object, if you just want to be able to sample Brownian motion for any other reason (including the ability to sample Levy area).

### Examples
**Quick example**

```python
from torchsde import BrownianInterval
bm = BrownianInterval(t0=0., t1=1., size=(4, 1))
dW = bm(0.2, 0.3)
```
Produces a tensor `dW` of shape `(4, 1)` corresponding to the increment `W(0.3) - W(0.2)` for a Brownian motion `W` defined over `[0, 1]`, taking values in `(4, 1)`-dimensional space.

(Mathematically: `W \in C([0, 1]; R^(4 x 1))`.)

**Example with `sdeint`**

```python
import torch
from torchsde import BrownianInterval, sdeint

batch_size, state_size, brownian_size = 32, 4, 3
sde = ...
y0 = torch.randn(batch_size, state_size, device='cuda')
ts = torch.tensor([0., 1.], device='cuda')
bm = BrownianInterval(t0=ts[0], 
                      t1=ts[-1], 
                      size=(batch_size, brownian_size),
                      device='cuda')
ys = sdeint(sde=sde, y0=y0, ts=ts, bm=bm)
```

### Arguments

- `t0` (float or Tensor): The initial time for the Brownian motion.
- `t1` (float or Tensor): The terminal time for the Brownian motion.
- `size` (tuple of int): The shape of each Brownian sample. If zero dimensional represents a scalar Brownian motion. If one dimensional represents a batch of scalar Brownian motions. If >two dimensional the last dimension represents the size of a a multidimensional Brownian motion, and all previous dimensions represent batch dimensions.
- `dtype` (torch.dtype): The dtype of each Brownian sample. Defaults to the PyTorch default.
- `device` (str or torch.device): The device of each Brownian sample. Defaults to the CPU.
- `entropy` (int): Global seed, defaults to `None` for random entropy.
- `levy_area_approximation` (str): Whether to also approximate Levy area. Defaults to `"none"`. Valid options are `"none"`, `"space-time"`, `"davie"` or `"foster"`, corresponding to different approximation types, see [below](#levy-area-approximation). This is needed for some higher-order SDE solvers.
- `dt` (float or Tensor): The expected average step size of the SDE solver. Set it if you know it (e.g. when using a fixed-step solver); else it will be estimated from the first few queries. This is used to set up the data structure such that it is efficient to query at these intervals.
- `tol` (float or Tensor): What tolerance to resolve the Brownian motion to. Must be non-negative. Defaults to zero, i.e. floating point resolution. Usually worth setting in conjunction with `halfway_tree`, below.
- `pool_size` (int): Size of the pooled entropy. If you care about
    statistical randomness then increasing this will help (but will
    slow things down).
- `cache_size` (int): How big a cache of recent calculations to use. (As new calculations depend on old calculations, this speeds things up dramatically, rather than recomputing things.) Set this to `None` to use an infinite cache, which will be fast but memory inefficient.
- `halfway_tree` (bool): Whether the dependency tree (the internal data structure) should be the dyadic tree. Defaults to `False`. Normally, the sample path is determined by both `entropy`, _and_ the locations and order of the query points. Setting this to `True` will make it deterministic with respect to just `entropy`; however this is much slower.
- `W` (Tensor): The increment of the Brownian motion over the interval
    `[t0, t1]`. Will be generated randomly if not provided.
- `H` (Tensor): The space-time Levy area of the Brownian motion over the
    interval `[t0, t1]`. Will be generated randomly if not provided.

### Important special cases

**Speed over memory**

If speed is important, and you're happy to use extra memory, then use
```python
BrownianInterval(..., cache_size=None)
```

**Fixed randomness**

If you want to use the same random seed to deterministically create the same Brownian motion, then use
```python
BrownianInterval(..., entropy=<integer>, tol=1e-5, halfway_tree=True)
```
If you're using a fixed SDE solver (or more precisely, if the locations and order of the queries to the Brownian interval are fixed), then just
```python
BrownianInterval(..., entropy=<integer>)
```
will suffice, and will be faster.

### BrownianPath and BrownianTree

`torchsde.BrownianPath` and `torchsde.BrownianTree` are the legacy ways of creating Brownian motions, corresponding to each of the two important special cases above (respectively).

These are still supported, but we encourage using the more flexible `BrownianInterval` for new projects.

### Levy area approximation
The `levy_area_approximation` argument may be either `"none"`, `"space-time"`, `"davie"` or `"foster"`. Levy area approximations are used in certain higher-order SDE solvers, and so this must be set to the appropriate value if using these higher-order solvers.

In particular, space-time Levy area is used in the stochastic Runge--Kutta solver.