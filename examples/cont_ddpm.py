# Copyright 2021 Google LLC
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

"""A min example for continuous-time Denoising Diffusion Probabilistic Models.

Trains the backward dynamics to be close to the reverse of a fixed forward
dynamics via a score-matching-type objective.

Trains a simple model on MNIST and samples from both the reverse ODE and
SDE formulation.

To run this file, first run the following to install extra requirements:
pip install kornia
pip install einops
pip install torchdiffeq
pip install fire

To run, execute:
python -m examples.cont_ddpm
"""
import abc
import logging
import math
import os

import fire
import torch
import torchdiffeq
import torchvision as tv
import tqdm
from torch import nn, optim
from torch.utils import data

import torchsde
from . import unet


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


class Module(abc.ABC, nn.Module):
    """A wrapper module that's more convenient to use."""

    def __init__(self):
        super(Module, self).__init__()
        self._checkpoint = False

    def zero_grad(self) -> None:
        for p in self.parameters(): p.grad = None

    @property
    def device(self):
        return next(self.parameters()).device


class ScoreMatchingSDE(Module):
    """Wraps score network with analytical sampling and cond. score computation.

    The variance preserving formulation in
        Score-Based Generative Modeling through Stochastic Differential Equations
        https://arxiv.org/abs/2011.13456
    """

    def __init__(self, denoiser, input_size=(1, 28, 28), t0=0., t1=1., beta_min=.1, beta_max=20.):
        super(ScoreMatchingSDE, self).__init__()
        if t0 > t1:
            raise ValueError(f"Expected t0 <= t1, but found t0={t0:.4f}, t1={t1:.4f}")

        self.input_size = input_size
        self.denoiser = denoiser

        self.t0 = t0
        self.t1 = t1

        self.beta_min = beta_min
        self.beta_max = beta_max

    def score(self, t, y):
        if isinstance(t, float):
            t = y.new_tensor(t)
        if t.dim() == 0:
            t = t.repeat(y.shape[0])
        return self.denoiser(t, y)

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _indefinite_int(self, t):
        """Indefinite integral of beta(t)."""
        return self.beta_min * t + .5 * t ** 2 * (self.beta_max - self.beta_min)

    def analytical_mean(self, t, x_t0):
        mean_coeff = (-.5 * (self._indefinite_int(t) - self._indefinite_int(self.t0))).exp()
        mean = x_t0 * fill_tail_dims(mean_coeff, x_t0)
        return mean

    def analytical_var(self, t, x_t0):
        analytical_var = 1 - (-self._indefinite_int(t) + self._indefinite_int(self.t0)).exp()
        return analytical_var

    @torch.no_grad()
    def analytical_sample(self, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return mean + torch.randn_like(mean) * fill_tail_dims(var.sqrt(), mean)

    @torch.no_grad()
    def analytical_score(self, x_t, t, x_t0):
        mean = self.analytical_mean(t, x_t0)
        var = self.analytical_var(t, x_t0)
        return - (x_t - mean) / fill_tail_dims(var, mean).clamp_min(1e-5)

    def f(self, t, y):
        return -0.5 * self._beta(t) * y

    def g(self, t, y):
        return fill_tail_dims(self._beta(t).sqrt(), y).expand_as(y)

    def sample_t1_marginal(self, batch_size, tau=1.):
        return torch.randn(size=(batch_size, *self.input_size), device=self.device) * math.sqrt(tau)

    def lambda_t(self, t):
        return self.analytical_var(t, None)

    def forward(self, x_t0, partitions=1):
        """Compute the score matching objective.
        Split [t0, t1] into partitions; sample uniformly on each partition to reduce gradient variance.
        """
        u = torch.rand(size=(x_t0.shape[0], partitions), dtype=x_t0.dtype, device=x_t0.device)
        u.mul_((self.t1 - self.t0) / partitions)
        shifts = torch.arange(0, partitions, device=x_t0.device, dtype=x_t0.dtype)[None, :]
        shifts.mul_((self.t1 - self.t0) / partitions).add_(self.t0)
        t = (u + shifts).reshape(-1)
        lambda_t = self.lambda_t(t)

        x_t0 = x_t0.repeat_interleave(partitions, dim=0)
        x_t = self.analytical_sample(t, x_t0)

        fake_score = self.score(t, x_t)
        true_score = self.analytical_score(x_t, t, x_t0)
        loss = (lambda_t * ((fake_score - true_score) ** 2).flatten(start_dim=1).sum(dim=1))
        return loss


class ReverseDiffeqWrapper(Module):
    """Wrapper of the score network for odeint/sdeint.

    We split this module out, so that `forward` of the score network is solely
    used for computing the score, and the `forward` here is used for odeint.
    Helps with data parallel.
    """
    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, module: ScoreMatchingSDE):
        super(ReverseDiffeqWrapper, self).__init__()
        self.module = module

    # --- odeint ---
    def forward(self, t, y):
        return -(self.module.f(-t, y) - .5 * self.module.g(-t, y) ** 2 * self.module.score(-t, y))

    # --- sdeint ---
    def f(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -(self.module.f(-t, y) - self.module.g(-t, y) ** 2 * self.module.score(-t, y))
        return out.flatten(start_dim=1)

    def g(self, t, y):
        y = y.view(-1, *self.module.input_size)
        out = -self.module.g(-t, y)
        return out.flatten(start_dim=1)

    # --- sample ---
    def sample_t1_marginal(self, batch_size, tau=1.):
        return self.module.sample_t1_marginal(batch_size, tau)

    @torch.no_grad()
    def ode_sample(self, batch_size=64, tau=1., t=None, y=None, dt=1e-2):
        self.module.eval()

        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y
        return torchdiffeq.odeint(self, y, t, method="rk4", options={"step_size": dt})

    @torch.no_grad()
    def ode_sample_final(self, batch_size=64, tau=1., t=None, y=None, dt=1e-2):
        return self.ode_sample(batch_size, tau, t, y, dt)[-1]

    @torch.no_grad()
    def sde_sample(self, batch_size=64, tau=1., t=None, y=None, dt=1e-2, tweedie_correction=True):
        self.module.eval()

        t = torch.tensor([-self.t1, -self.t0], device=self.device) if t is None else t
        y = self.sample_t1_marginal(batch_size, tau) if y is None else y

        ys = torchsde.sdeint(self, y.flatten(start_dim=1), t, dt=dt)
        ys = ys.view(len(t), *y.size())
        if tweedie_correction:
            ys[-1] = self.tweedie_correction(self.t0, ys[-1], dt)
        return ys

    @torch.no_grad()
    def sde_sample_final(self, batch_size=64, tau=1., t=None, y=None, dt=1e-2):
        return self.sde_sample(batch_size, tau, t, y, dt)[-1]

    def tweedie_correction(self, t, y, dt):
        return y + dt ** 2 * self.module.score(t, y)

    @property
    def t0(self):
        return self.module.t0

    @property
    def t1(self):
        return self.module.t1


def preprocess(x, logit_transform, alpha=0.95):
    if logit_transform:
        x = alpha + (1 - 2 * alpha) * x
        x = (x / (1 - x)).log()
    else:
        x = (x - 0.5) * 2
    return x


def postprocess(x, logit_transform, alpha=0.95, clamp=True):
    if logit_transform:
        x = (x.sigmoid() - alpha) / (1 - 2 * alpha)
    else:
        x = x * 0.5 + 0.5
    return x.clamp(min=0., max=1.) if clamp else x


def make_loader(
        root="./data/mnist",
        train_batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True
):
    """Make a simple loader for training images in MNIST."""

    def dequantize(x, nvals=256):
        """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
        return x

    train_transform = tv.transforms.Compose([tv.transforms.ToTensor(), dequantize])
    train_data = tv.datasets.MNIST(root, train=True, transform=train_transform, download=True)
    train_loader = data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    return train_loader


def main(
        train_dir="./dump/cont_ddpm/",
        epochs=100,
        lr=1e-4,
        batch_size=128,
        pause_every=1000,
        tau=1.,
        logit_transform=True,
):
    """Train and sample once in a while.

    Args:
        train_dir: Path to a folder to dump things.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        batch_size: Batch size for training.
        pause_every: Log and write figures once in this many iterations.
        tau: The temperature for sampling.
        logit_transform: Applies the typical logit transformation if True.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data.
    train_loader = make_loader(root=os.path.join(train_dir, 'data'), train_batch_size=batch_size)

    # Model + optimizer.
    denoiser = unet.Unet(
        input_size=(1, 28, 28),
        dim_mults=(1, 2, 4,),
        attention_cls=unet.LinearTimeSelfAttention,
    )
    forward = ScoreMatchingSDE(denoiser=denoiser).to(device)
    reverse = ReverseDiffeqWrapper(forward)
    optimizer = optim.Adam(params=forward.parameters(), lr=lr)

    def plot(imgs, path):
        assert not torch.any(torch.isnan(imgs)), "Found nans in images"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imgs = postprocess(imgs, logit_transform=logit_transform).detach().cpu()
        tv.utils.save_image(imgs, path)

    global_step = 0
    for epoch in range(epochs):
        for x, _ in tqdm.tqdm(train_loader):
            forward.train()
            forward.zero_grad()
            x = preprocess(x.to(device), logit_transform=logit_transform)
            loss = forward(x).mean(dim=0)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % pause_every == 0:
                logging.warning(f'global_step: {global_step:06d}, loss: {loss:.4f}')

                img_path = os.path.join(train_dir, 'ode_samples', f'global_step_{global_step:07d}.png')
                ode_samples = reverse.ode_sample_final(tau=tau)
                plot(ode_samples, img_path)

                img_path = os.path.join(train_dir, 'sde_samples', f'global_step_{global_step:07d}.png')
                sde_samples = reverse.sde_sample_final(tau=tau)
                plot(sde_samples, img_path)


if __name__ == "__main__":
    fire.Fire(main)
