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

import copy
import os

import torch
import tqdm

from torchsde import BaseBrownian, BaseSDE, sdeint
from torchsde.settings import SDE_TYPES
from torchsde.types import Tensor, Vector, Scalar, Tuple, Optional, Callable
from . import utils


def inspect_samples(y0: Tensor,
                    ts: Vector,
                    dt: Scalar,
                    sde: BaseSDE,
                    bm: BaseBrownian,
                    img_dir: str,
                    methods: Tuple[str, ...],
                    options: Optional[Tuple] = None,
                    vis_dim=0,
                    dt_true: Optional[float] = 2 ** -14,
                    labels: Optional[Tuple[str, ...]] = None):
    if options is None:
        options = (None,) * len(methods)
    if labels is None:
        labels = methods

    sde = copy.deepcopy(sde).requires_grad_(False)

    solns = [
        sdeint(sde, y0, ts, bm, method=method, dt=dt, options=options_)
        for method, options_ in zip(methods, options)
    ]

    method_for_true = 'euler' if sde.sde_type == SDE_TYPES.ito else 'midpoint'
    true = sdeint(sde, y0, ts, bm, method=method_for_true, dt=dt_true)

    labels += ('true',)
    solns += [true]

    # (T, batch_size, d) -> (T, batch_size) -> (batch_size, T).
    solns = [soln[..., vis_dim].t() for soln in solns]

    for i, samples in enumerate(zip(*solns)):
        utils.swiss_knife_plotter(
            img_path=os.path.join(img_dir, f'{i}'),
            plots=[
                {'x': ts, 'y': sample, 'label': label, 'marker': 'x'}
                for sample, label in zip(samples, labels)
            ]
        )


def inspect_orders(y0: Tensor,
                   t0: Scalar,
                   t1: Scalar,
                   dts: Vector,
                   sde: BaseSDE,
                   bm: BaseBrownian,
                   img_dir: str,
                   methods: Tuple[str, ...],
                   options: Optional[Tuple] = None,
                   dt_true: Optional[float] = 2 ** -14,
                   labels: Optional[Tuple[str, ...]] = None,
                   test_func: Optional[Callable] = lambda x: (x ** 2).flatten(start_dim=1).sum(dim=1)):
    if options is None:
        options = (None,) * len(methods)
    if labels is None:
        labels = methods

    sde = copy.deepcopy(sde).requires_grad_(False)
    ts = torch.tensor([t0, t1], device=y0.device)

    solns = [
        [
            sdeint(sde, y0, ts, bm, method=method, dt=dt, options=options_)[-1]
            for method, options_ in zip(methods, options)
        ]
        for dt in tqdm.tqdm(dts)
    ]

    if hasattr(sde, 'analytical_sample'):
        true = sde.analytical_sample(y0, ts, bm)[-1]
    else:
        method_for_true = 'euler' if sde.sde_type == SDE_TYPES.ito else 'midpoint'
        true = sdeint(sde, y0, ts, bm, method=method_for_true, dt=dt_true)[-1]

    mses = []
    maes = []
    for dt, solns_ in zip(dts, solns):
        mses_for_dt = [utils.mse(soln, true) for soln in solns_]
        mses.append(mses_for_dt)

        maes_for_dt = [utils.mae(soln, true, test_func) for soln in solns_]
        maes.append(maes_for_dt)

    strong_order_slopes = [
        utils.linregress_slope(utils.log(dts), .5 * utils.log(mses_for_method))
        for mses_for_method in zip(*mses)
    ]

    weak_order_slopes = [
        utils.linregress_slope(utils.log(dts), utils.log(maes_for_method))
        for maes_for_method in zip(*maes)
    ]

    utils.swiss_knife_plotter(
        img_path=os.path.join(img_dir, 'strong_order'),
        plots=[
            {'x': dts, 'y': mses_for_method, 'label': f'{label}(k={slope:.4f})', 'marker': 'x'}
            for mses_for_method, label, slope in zip(zip(*mses), labels, strong_order_slopes)
        ],
        options={'xscale': 'log', 'yscale': 'log', 'cycle_line_style': True}
    )

    utils.swiss_knife_plotter(
        img_path=os.path.join(img_dir, 'weak_order'),
        plots=[
            {'x': dts, 'y': mres_for_method, 'label': f'{label}(k={slope:.4f})', 'marker': 'x'}
            for mres_for_method, label, slope in zip(zip(*maes), labels, weak_order_slopes)
        ],
        options={'xscale': 'log', 'yscale': 'log', 'cycle_line_style': True}
    )
