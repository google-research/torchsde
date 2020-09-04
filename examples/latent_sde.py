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

import argparse
import logging
import math
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import tqdm
from torch import nn, optim
from torch.distributions import Normal, Laplace, kl_divergence

from examples import utils
from torchsde import sdeint, sdeint_adjoint, SDEIto, BrownianInterval

Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])


class LatentSDE(SDEIto):

    def __init__(self, theta=1.0, mu=0.0, sigma=0.5):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        # Prior drift.
        self.theta = nn.Parameter(torch.tensor([[theta]]), requires_grad=False)
        self.mu = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor([[sigma]]), requires_grad=False)

        # p(y0).
        logvar = math.log(sigma ** 2. / (2. * theta))
        self.py0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.py0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=False)

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(3, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 1)
        )
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)

    def h(self, t, y):  # Prior drift.
        return self.theta * (self.mu - y)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = float(t) * torch.ones_like(y)
        # Positional encoding in transformers; must use `t`, since the posterior is likely inhomogeneous.
        inp = torch.cat((torch.sin(t), torch.cos(t), y), dim=-1)
        return self.net(inp)

    def g(self, t, y):  # Shared diffusion.
        return self.sigma.repeat(y.size(0), 1)

    def forward(self, ts, batch_size, eps=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std

        qy0 = Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = kl_divergence(qy0, py0).sum(1).mean(0)  # KL(time=0).

        if args.adjoint:
            zs, logqp = sdeint_adjoint(self, y0, ts, logqp=True, method=args.method, dt=args.dt, adaptive=args.adaptive,
                                       rtol=args.rtol, atol=args.atol)
        else:
            zs, logqp = sdeint(self, y0, ts, logqp=True, method=args.method, dt=args.dt, adaptive=args.adaptive,
                               rtol=args.rtol, atol=args.atol)
        logqp = logqp.sum(0).mean(0)
        log_ratio = logqp0 + logqp  # KL(time=0) + KL(path).

        return zs, log_ratio

    def sample_p(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return sdeint(self, y0, ts, bm=bm, method='srk', dt=args.dt, names={'drift': 'h'})

    def sample_q(self, ts, batch_size, eps=None, bm=None):
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return sdeint(self, y0, ts, bm=bm, method='srk', dt=args.dt)

    @property
    def py0_std(self):
        return torch.exp(.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(.5 * self.qy0_logvar)


def make_segmented_cosine_data():
    with torch.no_grad():
        ts_ = np.concatenate((np.linspace(0.3, 0.8, 10), np.linspace(1.2, 1.5, 10)), axis=0)
        ts_ext_ = np.array([0.] + list(ts_) + [2.0])
        ts_vis_ = np.linspace(0., 2.0, 300)
        ys_ = np.cos(ts_ * (2. * math.pi))[:, None]

        ts = torch.tensor(ts_).float()
        ts_ext = torch.tensor(ts_ext_).float()
        ts_vis = torch.tensor(ts_vis_).float()
        ys = torch.tensor(ys_).float().to(device)
        return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


def make_irregular_sine_data():
    with torch.no_grad():
        ts_ = np.sort(npr.uniform(low=0.4, high=1.6, size=16))
        ts_ext_ = np.array([0.] + list(ts_) + [2.0])
        ts_vis_ = np.linspace(0., 2.0, 300)
        ys_ = np.sin(ts_ * (2. * math.pi))[:, None] * 0.8

        ts = torch.tensor(ts_).float()
        ts_ext = torch.tensor(ts_ext_).float()
        ts_vis = torch.tensor(ts_vis_).float()
        ys = torch.tensor(ys_).float().to(device)
        return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


def make_data():
    return {
        'segmented_cosine': make_segmented_cosine_data(),
        'irregular_sine': make_irregular_sine_data()
    }[args.data]


def main():
    # Dataset.
    ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = make_data()

    # Plotting parameters.
    vis_batch_size = 1024
    ylims = (-1.75, 1.75)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = npr.permutation(vis_batch_size)
    # From https://colorbrewer2.org/.
    if args.color == "blue":
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = len(sample_colors)
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)

    # Fix seed for the random draws used in the plots.
    eps = torch.randn(vis_batch_size, 1).to(device)
    bm = BrownianInterval(t0=ts_vis[0], t1=ts_vis[-1], shape=(vis_batch_size, 1), dtype=torch.float32, device=device,
                          levy_area_approximation='space-time')  # We need space-time Levy area to use the SRK solver

    # Model.
    model = LatentSDE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = utils.LinearScheduler(iters=args.kl_anneal_iters)

    logp_metric = utils.EMAMetric()
    log_ratio_metric = utils.EMAMetric()
    loss_metric = utils.EMAMetric()

    if args.show_prior:
        with torch.no_grad():
            zs = model.sample_p(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
            ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
            zs_ = np.sort(zs_, axis=1)

            img_dir = os.path.join(args.train_dir, 'prior.png')
            plt.subplot(frameon=False)
            for alpha, percentile in zip(alphas, percentiles):
                idx = int((1 - percentile) / 2. * vis_batch_size)
                zs_bot_ = zs_[:, idx]
                zs_top_ = zs_[:, -idx]
                plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

            # `zorder` determines who's on top; the larger the more at the top.
            plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)  # Data.
            plt.ylim(ylims)
            plt.xlabel('$t$')
            plt.ylabel('$Y_t$')
            plt.tight_layout()
            plt.savefig(img_dir, dpi=args.dpi)
            plt.close()
            logging.info(f'Saved prior figure at: {img_dir}')

    for global_step in tqdm.tqdm(range(args.train_iters)):
        # Plot and save.
        if global_step % args.pause_iters == 0:
            img_path = os.path.join(args.train_dir, f'global_step_{global_step}.png')

            with torch.no_grad():
                zs = model.sample_q(ts=ts_vis, batch_size=vis_batch_size, eps=eps, bm=bm).squeeze()
                samples = zs[:, vis_idx]
                ts_vis_, zs_, samples_ = ts_vis.cpu().numpy(), zs.cpu().numpy(), samples.cpu().numpy()
                zs_ = np.sort(zs_, axis=1)
                plt.subplot(frameon=False)

                if args.show_percentiles:
                    for alpha, percentile in zip(alphas, percentiles):
                        idx = int((1 - percentile) / 2. * vis_batch_size)
                        zs_bot_, zs_top_ = zs_[:, idx], zs_[:, -idx]
                        plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

                if args.show_mean:
                    plt.plot(ts_vis_, zs_.mean(axis=1), color=mean_color)

                if args.show_samples:
                    for j in range(num_samples):
                        plt.plot(ts_vis_, samples_[:, j], color=sample_colors[j], linewidth=1.0)

                if args.show_arrows:
                    num, dt = 12, 0.12
                    t, y = torch.meshgrid(
                        [torch.linspace(0.2, 1.8, num).to(device), torch.linspace(-1.5, 1.5, num).to(device)]
                    )
                    t, y = t.reshape(-1, 1), y.reshape(-1, 1)
                    fty = model.f(t=t, y=y).reshape(num, num)
                    dt = torch.zeros(num, num).fill_(dt).to(device)
                    dy = fty * dt
                    dt_, dy_, t_, y_ = dt.cpu().numpy(), dy.cpu().numpy(), t.cpu().numpy(), y.cpu().numpy()
                    plt.quiver(t_, y_, dt_, dy_, alpha=0.3, edgecolors='k', width=0.0035, scale=50)

                if args.hide_ticks:
                    plt.xticks([], [])
                    plt.yticks([], [])

                plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)  # Data.
                plt.ylim(ylims)
                plt.xlabel('$t$')
                plt.ylabel('$Y_t$')
                plt.tight_layout()
                plt.savefig(img_path, dpi=args.dpi)
                plt.close()
                logging.info(f'Saved figure at: {img_path}')

                if args.save_ckpt:
                    torch.save(
                        {'model': model.state_dict()}, os.path.join(ckpt_dir, f'global_step_{global_step}.ckpt')
                    )

        # Train.
        optimizer.zero_grad()
        zs, log_ratio = model(ts=ts_ext, batch_size=args.batch_size)
        zs = zs.squeeze()
        zs = zs[1:-1]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.

        likelihood = {
            "laplace": Laplace(loc=zs, scale=args.scale),
            "normal": Normal(loc=zs, scale=args.scale)
        }[args.likelihood]
        logp = likelihood.log_prob(ys).sum(dim=0).mean(dim=0)

        loss = -logp + log_ratio * kl_scheduler()
        loss.backward()
        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        logp_metric.step(logp)
        log_ratio_metric.step(log_ratio)
        loss_metric.step(loss)

        logging.info(
            f'global_step: {global_step}, '
            f'logp: {logp_metric.val():.3f}, log_ratio: {log_ratio_metric.val():.3f}, loss: {loss_metric.val():.3f}'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--save-ckpt', action='store_true')

    parser.add_argument('--data', type=str, default='segmented_cosine', choices=['segmented_cosine', 'irregular_sine'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--train-iters', type=int, default=5000, help='Number of iterations for training.')
    parser.add_argument('--pause-iters', type=int, default=50, help='Number of iterations before pausing.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='laplace')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal and Laplace.')

    parser.add_argument('--adjoint', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)

    parser.add_argument('--show-prior', type=utils.str2bool, default=True)
    parser.add_argument('--show-samples', type=utils.str2bool, default=True)
    parser.add_argument('--show-percentiles', type=utils.str2bool, default=True)
    parser.add_argument('--show-arrows', type=utils.str2bool, default=True)
    parser.add_argument('--show-mean', type=utils.str2bool, default=False)
    parser.add_argument('--hide-ticks', type=utils.str2bool, default=False)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    ckpt_dir = os.path.join(args.train_dir, 'ckpts')
    utils.makedirs(args.train_dir, ckpt_dir)

    main()
