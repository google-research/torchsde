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
    - Optionally: a method `h(t, y)` corresponding to a prior drift to calculate a KL divergence from. (See the discussion on `logqp` and the [notes on KL divergence](#calculating-kl-divergence) below.)
- `y0` is a tensor of shape `(batch_size, state_size)`, giving the initial value of the SDE at time `ts[0]`, 
- `ts` is a tensor of times to output `y` at, of shape `(t_size,)`. The SDE will be solved over the interval `[ts[0], ts[-1]]`.

Several keyword arguments are also accepted, see [below](#keyword-arguments-of-sdeint).

The output tensor `ys` will have shape `(t_size, batch_size, state_size)`, and corresponds to a sample from the SDE.

### Possible noise types
- `scalar`: The diffusion `g` has output size `(batch_size, state_size, 1)`. The Brownian motion is a batch of 1-dimensional Brownian motions.
- `additive`: The diffusion `g` is assumed to be constant w.r.t. `y` and has output size `(batch_size, state_size, brownian_size)`. The Brownian motion is a batch of `browian_size`-dimensional Brownian motions.
- `diagonal`: The diffusion `g` is element-wise and has output size `(batch_size, state_size)`. The Brownian motion is a batch of `state_size`-dimensional Brownian motions.
- `general`: The diffusion `g` has output size `(batch_size, state_size, brownian_size)`. The Brownian motion is a batch of `brownian_size`-dimensional Brownian motions.

Whilst `scalar`, `additive` and `diagonal` are special cases of `general` noise, they allow for additional solvers to be used and additional optimisations to take place.

### Keyword arguments for `sdeint`
- `bm`: A `BrownianInterval` object, see [below](#brownian-motion). Optionally include to control the Brownian motion.
- `method`: A string, corresponding to one of the solvers listed [below](#choice-of-solver). If not passed then a sensible default is used.
- `dt`: A float for the constant step size, or initial step size for adaptive time-stepping. Defaults to `1e-3`.
- `adaptive`: If True, use adaptive time-stepping. Defaults to `False`.
- `rtol`: Relative tolerance for adaptive time-stepping. Defaults to `1e-5`.
- `atol`: Absolute tolerance for adaptive time-stepping. Defaults to `1e-4`.
- `dt_min`: Minimum step size for adaptive time-stepping. Defaults to `1e-5`.
- `names`: A dictionary giving alternate names for the drift and diffusion methods. Acceptable keys are `"drift"`,`"diffusion"` and `"prior_drift"`. For example `{"drift": "foo"}` will use `foo` instead of `f`.
- `logqp`: Whether to calculate an estimate of the KL-divergence between two SDEs. See the [notes on KL divergence](#calculating-kl-divergence) below. This is returned as an additional tensor.
- `extra`: Whether to also return any additional variables tracked by the solver. This is returned as an additional tensor.
- `extra_solver_state`: Initial values for any additional variables tracked by the solver. (Rather than constructing sensible defaults automatically.) In particular, between this and the `extra` argument, it is possible to solve an SDE forwards in time, and then exactly reconstruct it backwards in time. (Without any additional numerical error.) This represents advanced functionality best used only if you know what you're doing, as it's not really documented yet.


### Providing specialised methods
If your drift/diffusion have special structure, for example the drift and diffusion share some computations, then it may be more efficient to evaluate them together rather than alone.

As such, if the following methods are present on `sde`, then they will be used if possible: `g_prod(t, y, v)`, `f_and_g(t, y)`, `f_and_g_prod(t, y, v)`. Here `g_prod` is expected to compute the batch matrix-vector product between the diffusion and the vector `v`. `f_and_*` should return a 2-tuple of `f(t, y)` and `g(t, y)`/`g_prod(t, y, v)` as appropriate.

(Although at present the `names` argument only works for renaming `f`, `g`, `h`, and not any of these.)

## Choice of solver

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
- `"reversible_heun"`: The reversible midpoint method, as introduced in [\[1\]](https://arxiv.org/abs/2105.13493).
- `"adjoint_reversible_heun"`: A special method: pass this as part of `sdeint_adjoint(..., method="reversible_heun", adjoint_method="adjoint_reversible_heun")` for more accurate gradient calculations with the adjoint method, see [the notes on adjoint methods](#adjoints).

Note that Milstein and SRK don't support general noise.

Additionally, [gradient-free Milstein](https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf) can be used by selecting Milstein, and then also passing in the keyword argument `sdeint(..., options=dict(grad_free=True))`.

### Which solver should I use?

If calculating an Ito SDE, then `srk` will generally produce a more accurate estimate of the solution than `milstein`, which will generally produce a more accurate estimate than `euler`. Meanwhile `srk` is the most expensive, followed by `milstein`, followed by `euler` as the computationally cheapest.

If calculating a Stratonovich SDE, then `midpoint` , `heun` and `milstein` are more computationally expensive. `euler_heun` and `reversible_heun` are the cheapest.

If training neural SDEs _without_ the adjoint method then accurate SDE solutions usually aren't super important (unless e.g. wanting to use a different discretisation between training and inference). So the vector fields can learn to work with the discretisation chosen, whilst computational cost often matters a lot. This makes `euler` (for Ito) or `reversible_heun` (for Stratonovich) good choices.

If training neural SDEs _with_ the adjoint method, then specifically Stratonovich SDEs and `reversible_heun` come strongly recommended; [see the notes on adjoints](#adjoints).

## Adjoints

The SDE solve may be backpropagated through. This may be done either by backpropagating through the internal operations of the solver (when using `sdeint`), or via the _adjoint method_ [\[2\]](https://arxiv.org/pdf/2001.01328.pdf), which solves another "adjoint SDE" backwards in time to calculate the gradients (when using `sdeint_adjoint`).

Using the adjoint method will reduce memory usage, but takes longer to compute.

`sdeint_adjoint` supports all the arguments and keyword arguments that `sdeint` does, as well as the following additional keyword arguments: 

- `adjoint_method`, `adjoint_adaptive`, `adjoint_rtol`, `adjoint_atol`: as for the forward pass.  
- `adjoint_params`: the tensors to calculate gradients with respect to. If not passed, defaults to `sde.parameters()`.

Double backward is supported through the adjoint method, and is calculated using an adjoint-of-adjoint SDE, with the same method, tolerances, etc. as for the adjoint SDE. Note that the numerical errors can easily grow large for adjoint-of-adjoint SDEs.

#### Advice on adjoint methods

Use Stratonovich SDEs if possible, as these have a computationally cheaper adjoint SDE than Ito SDEs.

Solving the adjoint SDE implies making some numerical error. If not managed carefully then this can make training more difficult, as the gradients calculated will be less accurate. Whilst the issue can just be ignored -- it's usually not a deal-breaker -- there are two main options available for managing it:

- The usual best approach is to use `method="reversible_heun"` and `adjoint_method="adjoint_reversible_heun"`, as introduced in [\[1\]](https://arxiv.org/abs/2105.13493). These are a special pair of solvers which when used in conjunction have almost zero numerical error, as the adjoint solver carefully reconstructs the numerical trajectory taken by the forward solver.
- Use adaptive step sizes, or small step sizes, on both the forward and backward pass. This usually implies additional computational cost, but can reduce numerical error.

## Calculating KL divergence

If `logqp=True` then an additional tensor of shape `(t_size - 1, batch_size)`, will be returned, corresponding to

```
\int_{ts[i-1]}^{ts[i]} g(t, y(t))^-1 (f(t, y(y)) - h(t, y(t))) dt
```

for all `i`, for each sample path `y(t)` that is generated by the solver. Taking an average over the `batch_size` dimension then represents an estimate of the KL divergence between

```
dy(t) = f(t, y(t)) dt + g(t, y(t)) dW(t)        y(ts[0]) = y0 
```

and

```
dz(t) = h(t, z(t)) dt + g(t, z(t)) dW(t)        y(ts[0]) = y0 
```

for each time interval `ts[i - 1]`, `ts[i]`.

Note that this implies calculating a matrix inverse of `g`. This can be quite expensive for `general` or `additive` noise, and `diagonal` noise is generally much computationally cheaper.


## Brownian motion

The `bm` argument to `sdeint` and `sdeint_adjoint` allows for tighter control on the Brownian motion. This should be a `torchsde.BrownianInterval` object. In particular, this allows for fixing its random seed (to produce the same Brownian motion every time), and for adjusting certain parameters that may affect its speed or memory usage.

`BrownianInterval` can also be used as a standalone object, if you just want to be able to sample Brownian motion for any other reason.

The time and memory efficient sampling provided by the Brownian Interval was introduced in [\[1\]](https://arxiv.org/abs/2105.13493).

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

# References

\[1\] Patrick Kidger, James Foster, Xuechen Li, Terry Lyons. "Efficient and Accurate Gradients for Neural SDEs". 2021. [[arXiv]](https://arxiv.org/abs/2105.13493)

\[2\] Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, David Duvenaud. "Scalable Gradients for Stochastic Differential Equations". *International Conference on Artificial Intelligence and Statistics.* 2020. [[arXiv]](https://arxiv.org/pdf/2001.01328.pdf)

