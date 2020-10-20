###################
# Let's have a look at how to train an SDE as a GAN.
# This follows the paper "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs"
###################

import torch
try:
    import torchcde
except ImportError as e:
    raise ImportError("`torchcde` is not installed: go to https://github.com/patrick-kidger/torchcde.") from e
import torchsde


###################
# First some standard helper objects.
###################

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super(MLP, self).__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 torch.nn.Softplus()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(torch.nn.Softplus())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

        ###################
        # Note the use of softplus activations: these are used for theoretical reasons regarding the smoothness of the
        # vector fields of our SDE. It's unclear how much it matters in practice, though.
        ###################

    def forward(self, x):
        return self._model(x)


def gp_penalty(generated, real, call):
    for fake_, real_ in zip(generated, real):
        assert fake_.shape == real_.shape  # including batch dimension
    batch_size = generated[0].size(0)
    for fake_ in generated:
        assert fake_.size(0) == batch_size

    alpha = torch.rand(batch_size, dtype=generated[0].dtype, device=generated[0].device)
    interpolated = []
    for fake_, real_ in zip(generated, real):
        alpha_ = alpha
        for _ in range(fake_.ndimension() - 1):
            alpha_ = alpha_.unsqueeze(-1)
        interpolated_ = alpha_ * real_.detach() + (1 - alpha_) * fake_.detach()
        interpolated_.requires_grad_(True)
        interpolated.append(interpolated_)

    with torch.enable_grad():
        score_interpolated = call(*interpolated)
        penalties = torch.autograd.grad(score_interpolated, tuple(interpolated),
                                        torch.ones_like(score_interpolated),
                                        create_graph=True, retain_graph=True)
    penalty = torch.cat([penalty.reshape(batch_size, -1) for penalty in penalties], dim=1)
    return penalty.norm(2, dim=-1).sub(1).pow(2).mean()


###################
# Now we define the SDEs.
###################

###################
# We begin by defining the generator SDE.
# The choice of Ito vs Stratonovich, and the choice of different noise types, isn't super important here. We happen to
# be using Stratonovich with general noise.
###################
class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super(GeneratorFunc, self).__init__()

        ###################
        # Drift and diffusion are MLPs. We happen to make them the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers,
                              tanh=True)

    def f(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx)

    def g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._diffusion(tx)


###################
# Now we wrap it up into something that computes the SDE
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super(Generator, self).__init__()

        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, t, batch_size):
        # t has shape (seq_len,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        x0 = self._initial(torch.randn(batch_size, self._initial_noise_size, device=t.device))
        xs = torchsde.sdeint(self._func, x0, t, method='midpoint', dt=1e-1)  # shape (seq_len, batch_size, data_size)
        xs = xs.transpose(0, 1)  # switch seq_len and batch_size
        return self._readout(xs)


###################
# Next the discriminator.
# There's a few different (roughly equivalent) ways of making the discriminator work. It's important to get this right.
# This requires a bit of reading! It's quite straightforward, though, and by the end of this you should have a really
# comprehensive knowledge of how these things fit together.
#
# Let Y be the real/generated sample, and let H be the hidden state of the discriminator.
# For real data, then Y is some interpolation of an (irregular) time series. (As with neural CDEs, if you're familiar -
# for a nice exposition on this see https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py.)
# In the case of generated data, then Y is _either_ the continuous-time sample produced by sdeint, _or_ it is an
# interpolation (probably linear interpolation) of the generated sample between particular evaluation points, We'll
# refer to these as cases (*) and (**) respectively.
#
# In terms of the mathematics, our options for the discriminator are:
# (a1) Solve dH(t) = f(t, H(t)) dt + g(t, H(t)) dY(t),
# (a2) Solve dH(t) = (f, g)(t, H(t)) d(t, Y(t))
# (b) Solve dH(t) = f(t, H(t), Y(t)) dt.
# Option (a1) is what is stated in the "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs" paper.
# Option (a2) is theoretically the same as (a1), but the drift and diffusion have been merged into a single function,
# and the sample Y has been augmented with time. This can sometimes be a more helpful way to think about things.
# Option (b) is a special case of the first two, by Appendix C of arXiv:2005.08926.
# [Note that just dH(t) = g(t, H(t)) dY(t) would _not_ be enough, by what's known as the tree-like equivalence property.
#  It's a bit technical, but the basic idea is that the discriminator wouldn't be able to tell how fast we traverse Y.
#  This is a really easy mistake to make; make sure you don't fall into it.]
#
# Whether we use (*) or (**), and (a1) or (a2) or (b), doesn't really affect the quality of the discriminator, as far as
# we know. However, these distinctions do affect how we solve them in terms of code. Depending on each combination, our
# options are to use a solver of the following types:
#
#      | (a1)   (a2)   (b)
# -----+----------------------
#  (*) | SDE           SDE
# (**) |        CDE    ODE
#
# So, (*) implies using an SDE solver: the continuous-time sample is only really available inside sdeint, so if we're
# going to use the continuous-time sample then we need to solve generator and discriminator together inside a single SDE
# solve. In this case, as our generator takes the form
# Y(t) = l(X(t)) with dX(t) = μ(t, X(t)) dt + σ(t, X(t)) dW(t),
# then
# dY(t) = l(X(t)) dX(t) = l(X(t))μ(t, X(t)) dt + l(X(t))σ(t, X(t)) dW(t).
# Then for (a1) we get
# dH(t) = ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) ) dt + g(t, H(t))l(X(t))σ(t, X(t)) dW(t),
# which we can now put together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t)                                )      ( σ(t, X(t))                  )
# d( Y(t) ) = ( l(X(t))μ(t, X(t)                         ) dt + ( l(X(t))σ(t, X(t))           ) dW(t)
#  ( H(t) )   ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) )      ( g(t, H(t))l(X(t))σ(t, X(t)) ),
# whilst for (b) we can put things together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t))       )      ( σ(t, X(t))        )
# d( Y(t) ) = ( l(X(t))μ(t, X(t) ) dt + ( l(X(t))σ(t, X(t)) ) dW(t)
#  ( H(t) )   ( f(t, H(t), Y(t)) )      ( 0                 )
#
# Phew, what a lot of stuff to write down. Don't be put off by this: there's no complicated algebra, it's literally just
# substituting one equation into another. Also, note that all of this is for the _generated_ data. If using real data,
# then Y(t) is as previously described always an interpolation of the data. If you're able to evaluate the derivative of
# the interpolation then you can then apply (a1) by rewriting it as dY(t) = (dY/dt)(t) dt and substituting in. If you're
# able to evaluate the interpolation itself then you can apply (b) directly.
#
# The benefit of using (*) is that everything can be done inside a single SDE solve, which is important if you're
# thinking about using adjoint methods and the like, for memory efficiency. The downside is that the code gets a bit
# more complicated: you need to be able to solve just the generator on its own (to produce samples at inference time),
# just the discriminator on its own (to evaluate the discriminator on the real data), and the combined
# generator-discriminator system (to evaluate the discriminator on the generated data).
#
# Right, let's move on to (**). In comparison, this is much simpler. We don't need to substitute in anything. We're just
# taking our generated data, sampling it at a bunch of points, and then doing some kind of interpolation (probably
# linear interpolation). Then we either solve (a2) directly with a CDE solver (regardless of whether we're using real or
# generated data), or solve (b) directly with an ODE solver (regardless of whether we're using real or generated data).
#
# The benefit of this is that it's much simpler to code: unlike (*) we can separate the generator and discriminator, and
# don't ever need to combine them. Also, real and generated data is treated the same in the discriminator. (Which is
# arguably a good thing anyway.) The downside is that we can't really take advantage of things like adjoint methods to
# backpropagate efficiently through the generator, because we need to produce (and thus store) our generated sample at
# lots of time points, which reduces the memory efficiency.
#
# Note that the use of ODE solvers for (**) is only valid because we're using _interpolated_ real or generated data,
# and we're assuming that we're using some kind of interpolation that is at least piecewise smooth. (For example, linear
# interpolation is piecewise smooth.) It wouldn't make sense to apply ODE solvers to some rough signal like Brownian
# motion - that's what case (*) and SDE solvers are about.
#
# Right, let's wrap up this wall of text. Here, we're going to use option (**), (a2). This is arguably the simplest
# option, and we'd like to keep the code readable in this example. To solve the CDEs we're going to the CDE solvers
# available through torchcde: https://github.com/patrick-kidger/torchcde.
# (Note that in the code for the "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs" paper, we actually used
# option (*), (b).)
###################

class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(DiscriminatorFunc, self).__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        self._module = MLP(1 + data_size, hidden_size * data_size, mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(Discriminator, self).__init__()
        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, y):
        # y has shape (batch_size, seq_len, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        y_coeffs = torchcde.linear_interplation_coeffs(y)
        Y = torchcde.LinearInterpolation(y_coeffs)
        Y0 = Y.evaluate(0)
        # This 0 is the start of the region of integration. This doesn't have to align with the times that the data was
        # measured at. (As per how neural CDEs work.)

        h0 = self._initial(Y0)
        # We happen to use midpoint with step_size=1e-1 for consistency with the SDE solve, but that isn't important.
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, adjoint=False, method='midpoint',
                             options=dict(step_size=1e-1))  # shape (batch_size, 2, hidden_size)
        score = self._readout(hs[:, -1])
        return score


def get_data():
    pass


def main():
    data_size
    initial_noise_size = 40  # How many noise dimensions to sample at the start of the SDE
    noise_size = 3           # How many dimensions the Brownian motion has
    hidden_size = 64         # How big the hidden size of the generator SDE and the discriminator CDE are. (Better
                             #     performance is generally obtained by making this larger than mlp_size.)
    mlp_size = 32            # How big the layers in the various MLPs are.
    num_layers = 2           # How many hidden layers to have in the various MLPs.

    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers)


###################
# And that's an SDE as a GAN. Now, exercise for the reader: turn all of this into a conditional GAN! :)
###################
