###################
# Let's have a look at how to train an SDE as a GAN.
# This follows the paper "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs".
###################

import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
try:
    import torchcde
except ImportError as e:
    raise ImportError("`torchcde` is not installed: go to https://github.com/patrick-kidger/torchcde.") from e
import torchsde
import tqdm


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
            ###################
            # Note the use of softplus activations: these are used for theoretical reasons regarding the smoothness of
            # the vector fields of our SDE. It's unclear how much it matters in practice, though.
            ###################
            model.append(torch.nn.Softplus())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


def gradient_penalty(generated, real, call):
    assert generated.shape == real.shape
    batch_size = generated.size(0)

    alpha = torch.rand(batch_size, *[1 for _ in range(real.ndimension() - 1)],
                       dtype=generated.dtype, device=generated.device)
    interpolated = alpha * real.detach() + (1 - alpha) * generated.detach()
    interpolated.requires_grad_(True)

    with torch.enable_grad():
        score_interpolated = call(interpolated)
        penalty, = torch.autograd.grad(score_interpolated, interpolated,
                                       torch.ones_like(score_interpolated),
                                       create_graph=True)
    penalty = penalty.reshape(batch_size, -1)
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

        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
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
        return self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


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

        # Final layer has an easier (convex) problem to solve, so increase its learning rate.
        self._readout.weight.register_hook(lambda grad: 100 * grad)
        self._readout.bias.register_hook(lambda grad: 100 * grad)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        x0 = self._initial(torch.randn(batch_size, self._initial_noise_size, device=ts.device))
        xs = torchsde.sdeint(self._func, x0, ts, method='midpoint', dt=1e-1)  # shape (t_size, batch_size, hidden_size)
        xs = xs.transpose(0, 1)  # switch t_size and batch_size
        vals = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        t_size = ts.size(0)
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, t_size, 1)
        return torch.cat([ts, vals], dim=2)


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs" paper.
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# strongly encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################

class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(DiscriminatorFunc, self).__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(Discriminator, self).__init__()
        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

        self._readout.weight.register_hook(lambda grad: 100 * grad)
        self._readout.bias.register_hook(lambda grad: 100 * grad)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])

        h0 = self._initial(Y0)
        # We happen to use midpoint with step_size=1e-1 for consistency with the SDE solve, but that isn't important.
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, adjoint=False, method='midpoint',
                             options=dict(step_size=1e-1))  # shape (batch_size, 2, hidden_size)
        score = self._readout(hs[:, -1])
        return score.mean()


###################
# Generate some data. For this example we generate some synthetic data based on the Heston SDE, from mathematical
# finance.
###################
def get_data():
    class HestonSDE(torch.nn.Module):
        sde_type = 'ito'
        noise_type = 'diagonal'

        def __init__(self, mu, theta, kappa, xi):
            assert 2 * kappa * theta > xi ** 2, "Feller condition not satisfied"
            super(HestonSDE, self).__init__()
            self.mu = mu
            self.theta = theta
            self.kappa = kappa
            self.xi = xi

        def f(self, t, y):
            S, nu = y[:, 0], y[:, 1]
            return torch.stack([self.mu * S, self.kappa * (self.theta - nu)], dim=1)

        def g(self, t, y):
            S, nu = y[:, 0], y[:, 1]
            return torch.stack([nu.sqrt() * S, self.xi * nu.sqrt()], dim=1)

    dataset_size = 16384
    t_size = 100

    heston_sde = HestonSDE(mu=0.05, theta=0.5, kappa=2., xi=0.2)
    S0 = 0.25 + torch.rand(dataset_size) * 0.05
    nu0 = 0.95 + torch.rand(dataset_size) * 0.1
    y0 = torch.stack([S0, nu0], dim=1)
    ts = torch.linspace(0, 10, t_size)
    ys = torchsde.sdeint(heston_sde, y0, ts, dt=1e-2)

    ###################
    # To demonstrate how to handle irregular data, then here we additionally drop some of the data (by setting it to
    # NaN.)
    ###################
    ys_num = ys.numel()
    to_drop = torch.randperm(ys_num)[:int(0.3 * ys_num)]
    ys.view(-1)[to_drop] = float('nan')

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
                    ys.transpose(0, 1)], dim=2)
    # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

    return ts, ys


###################
# Now do normal GAN training, and plot the results.
###################

def train_generator(ts, batch_size, generator, discriminator, generator_optimiser, discriminator_optimiser):
    generated_samples = generator(ts, batch_size)
    # Computing coeffs is actually a no-op because there's no missing data (because this is the generated
    # data), and the data itself is used as the coefficients for linear interpolation. Still we include it
    # here for completeness.
    generated_samples = torchcde.linear_interpolation_coeffs(generated_samples)
    generated_score = discriminator(generated_samples)

    generated_score.backward()
    generator_optimiser.step()
    generator_optimiser.zero_grad()
    discriminator_optimiser.zero_grad()


def train_discriminator(ts, batch_size, real_samples, generator, discriminator, discriminator_optimiser, device,
                        gp_coeff):
    with torch.no_grad():
        generated_samples = generator(ts, batch_size)
    generated_samples = torchcde.linear_interpolation_coeffs(generated_samples)
    generated_score = discriminator(generated_samples)

    real_samples = real_samples.to(device)
    real_score = discriminator(real_samples)

    penalty = gradient_penalty(generated_samples, real_samples, discriminator)
    loss = generated_score - real_score
    (gp_coeff * penalty - loss).backward()
    discriminator_optimiser.step()
    discriminator_optimiser.zero_grad()


def evaluate_loss(ts, batch_size, dataloader, generator, discriminator, device):
    with torch.no_grad():
        total_loss = 0
        for real_samples, in dataloader:
            real_samples = real_samples.to(device)
            generated_samples = generator(ts, batch_size)
            generated_samples = torchcde.linear_interpolation_coeffs(generated_samples)
            generated_score = discriminator(generated_samples)

            real_samples = real_samples.to(device)
            real_score = discriminator(real_samples)

            loss = generated_score - real_score
            total_loss += loss.item()
    return total_loss


def main():
    # Architectural hyperparameters. These are quite small for illustrative purposes.
    data_size = 2           # How many channels the data has (not including time).
    initial_noise_size = 8  # How many noise dimensions to sample at the start of the SDE.
    noise_size = 3          # How many dimensions the Brownian motion has.
    hidden_size = 32        # How big the hidden size of the generator SDE and the discriminator CDE are.
    mlp_size = 16           # How big the layers in the various MLPs are.
    num_layers = 2          # How many hidden layers to have in the various MLPs.

    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    ratio = 5               # How many discriminator training steps to take per generator training step.
    gp_coeff = 100          # How much to regularise with gradient penalty
    lr = 2e-5               # Learning rate often needs careful tuning to the problem.
    batch_size = 1024       # Batch size.
    pre_epochs = 200        # How many epochs to train just the discriminator for at the start.
    epochs = 2000           # How many epochs to train both generator and discriminator for.
    betas = (0.9, 0.99)     # What beta values to use with the generator's optimiser.
    weight_decay = 0.001    # How much weight decay to apply to the generator and discriminator.
    init_mult = 0.1         # Small initialisation sometimes seems to give better training.
    
    print_per_epoch = 50    # How often to print the loss

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")

    # Data
    print("Generating data...")
    ts, ys = get_data()
    # Important to normalise data
    means = []
    stds = []
    for ys_ in ys.unbind(dim=-1):
        ys_ = ys_.view(-1)
        ys_ = ys_.masked_select(~torch.isnan(ys_))
        mean = torch.mean(ys_)
        std = torch.std(ys_)
        means.append(mean)
        stds.append(std)
    mean = torch.stack(means)
    std = torch.stack(stds)
    ys = (ys - mean) / std
    print("Generated data.")
    ts = ts.to(device)
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=6, pin_memory=is_cuda)

    # Models
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    with torch.no_grad():
        for parameter in generator.parameters():
            parameter *= init_mult
        for parameter in discriminator.parameters():
            parameter *= init_mult

    # Optimisers.
    generator_optimiser = torch.optim.AdamW(generator.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.RMSprop(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    # Initially train just the discriminator
    print("Pretraining discriminator...")
    trange = tqdm.tqdm(range(pre_epochs))
    for epoch in trange:
        for real_samples, in dataloader:
            train_discriminator(ts, batch_size, real_samples, generator, discriminator, discriminator_optimiser, device,
                                gp_coeff)
        if (epoch % print_per_epoch) == 0 or epoch == pre_epochs - 1:
            total_loss = evaluate_loss(ts, batch_size, dataloader, generator, discriminator, device)
            trange.write(f"Epoch: {epoch:3} Loss: {total_loss:.4f}")
    print("Pretrained.")

    # Train both generator and discriminator
    print("Training...")
    i = 0
    trange = tqdm.tqdm(range(epochs))
    swa_epoch_cutoff = epochs // 5
    for epoch in trange:
        for real_samples, in dataloader:
            i += 1
            if (i % ratio) == 0:
                train_generator(ts, batch_size, generator, discriminator, generator_optimiser, discriminator_optimiser)
            else:
                train_discriminator(ts, batch_size, real_samples, generator, discriminator, discriminator_optimiser,
                                    device, gp_coeff)

        # Stochastic weight averaging typically improves performance quite a lot
        if epoch > swa_epoch_cutoff:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (epoch % print_per_epoch) == 0 or epoch == epochs - 1:
            total_unaveraged_loss = evaluate_loss(ts, batch_size, dataloader, generator, discriminator, device)
            if epoch > swa_epoch_cutoff:
                total_averaged_loss = evaluate_loss(ts, batch_size, dataloader, averaged_generator.module,
                                                    averaged_discriminator.module, device)
                trange.write(f"Epoch: {epoch:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Epoch: {epoch:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())
    print("Trained.")

    # Plot results
    print("Plotting...")
    plot_size = 10
    real_samples, = next(iter(dataloader))
    real_samples = real_samples[:plot_size]
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_samples = mean + std * real_samples
    # Each have shape (plot_size=10, t_size=100, 1 + data_size=3)
    # The three channels are time, S, nu. S is the one that's of most interest for this problem, so we're going to plot
    # that.
    real_samples = real_samples[..., 1]

    with torch.no_grad():
        generated_samples = generator(ts, plot_size).cpu()
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_samples = mean + std * generated_samples
    generated_samples = generated_samples[..., 1]

    first = True
    for real_sample in real_samples:
        kwargs = {'label': 'Real'} if first else {}
        first = False
        plt.plot(ts.cpu(), real_sample.cpu(), color='blue', **kwargs)
    first = True
    for generated_sample in generated_samples:
        kwargs = {'label': 'Generated'} if first else {}
        first = False
        plt.plot(ts.cpu(), generated_sample.cpu(), color='red', **kwargs)
    plt.legend()
    plt.show()
    print("Done.")


if __name__ == '__main__':
    main()

###################
# And that's an SDE as a GAN. Now, exercise for the reader: turn all of this into a conditional GAN.
# As a final warning, getting these working can be pretty finickity! GANs always are... :D
###################

###################
# Appendix: discriminators for a neural SDE
#
# This is a little long, but should all be quite straightforward. By the end of this you should have a really
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
# Right, let's wrap up this wall of text. Here, we use option (**), (a2). This is arguably the simplest option, and
# is chosen as we'd like to keep the code readable in this example. To solve the CDEs we use the CDE solvers available
# through torchcde: https://github.com/patrick-kidger/torchcde.
# (Note that in the code for the "Neural SDEs Made Easy: SDEs are Infinite-Dimensional GANs" paper, we actually used
# option (*), (b).)
###################
