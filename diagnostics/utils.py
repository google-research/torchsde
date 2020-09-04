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

import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from torchsde.types import Optional, Tensor, Sequence, Union, Callable


def to_numpy(*args):
    """Convert a sequence which might contain Tensors to numpy arrays."""
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, torch.Tensor):
            arg = _to_numpy_single(arg)
        return arg
    else:
        return tuple(_to_numpy_single(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)


def _to_numpy_single(arg: torch.Tensor) -> np.ndarray:
    return arg.detach().cpu().numpy()


def mse(x: Tensor, y: Tensor, norm_dim: Optional[int] = 1, mean_dim: Optional[int] = 0) -> np.ndarray:
    """Compute mean squared error."""
    return _to_numpy_single((torch.norm(x - y, dim=norm_dim) ** 2).mean(dim=mean_dim))


def mae(x: Tensor, y: Tensor, test_func: Callable, mean_dim: Optional[int] = 0) -> np.ndarray:
    return _to_numpy_single(
        abs(test_func(x).mean(mean_dim) - test_func(y).mean(mean_dim))
    )


def log(x: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Compute element-wise log of a sequence of floats."""
    return np.log(np.array(x))


def linregress_slope(x, y):
    """Return the slope of a least-squares regression for two sets of measurements."""
    return stats.linregress(x, y)[0]


def swiss_knife_plotter(img_path, plots=None, scatters=None, hists=None, options=None):
    """A multi-functional *standalone* wrapper; reduces boilerplate.

    Args:
        img_path (str): A path to the place where the image should be written.
        plots (list of dict, optional): A list of curves that needs `plt.plot`.
        scatters (list of dict, optional): A list of scatter plots that needs `plt.scatter`.
        hists (list of histograms, optional): A list of histograms that needs `plt.hist`.
        options (dict, optional): A dictionary of optional arguments. Possible entries include
            - xscale (str): Scale of xaxis.
            - yscale (str): Scale of yaxis.
            - xlabel (str): Label of xaxis.
            - ylabel (str): Label of yaxis.
            - title (str): Title of the plot.
            - cycle_linestyle (bool): Cycle through matplotlib's possible line styles if True.

    Returns:
        Nothing.
    """
    img_dir = os.path.dirname(img_path)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if plots is None: plots = ()
    if scatters is None: scatters = ()
    if hists is None: hists = ()
    if options is None: options = {}

    plt.figure(dpi=300)
    if 'xscale' in options: plt.xscale(options['xscale'])
    if 'yscale' in options: plt.yscale(options['yscale'])
    if 'xlabel' in options: plt.xlabel(options['xlabel'])
    if 'ylabel' in options: plt.ylabel(options['ylabel'])
    if 'title' in options: plt.title(options['title'])

    cycle_linestyle = options.get('cycle_linestyle', False)
    cycler = itertools.cycle(["-", "--", "-.", ":"]) if cycle_linestyle else None
    for entry in plots:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        entry['x'], entry['y'] = to_numpy(entry['x'], entry['y'])
        if cycle_linestyle:
            kwargs['linestyle'] = next(cycler)
        plt.plot(entry['x'], entry['y'], **kwargs)

    for entry in scatters:
        kwargs = {key: entry[key] for key in entry if key != 'x' and key != 'y'}
        entry['x'], entry['y'] = to_numpy(entry['x'], entry['y'])
        plt.scatter(entry['x'], entry['y'], **kwargs)

    for entry in hists:
        kwargs = {key: entry[key] for key in entry if key != 'x'}
        entry['x'] = to_numpy(entry['x'])
        plt.hist(entry['x'], **kwargs)

    if len(plots) > 0 or len(scatters) > 0: plt.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()


def manual_seed(seed: Optional[int] = 1147481649):
    """Set seeds for default generators of 1) torch, 2) numpy, and 3) Python's random library."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
