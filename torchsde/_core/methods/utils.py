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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch


def compute_trapezoidal_approx(bm, t0, y0, dt, sqrt_dt, dt1_div_dt=10, dt1_min=0.01):
    """Estimate int_{t0}^{t0+dt} int_{t0}^{s} dW(u) ds with trapezoidal rule.

    Slower compared to using the Gaussian with analytically derived mean and standard deviation, but ensures
    true determinism, since this rids the random number generation in the solver, i.e. all randomness comes from `bm`.

    The loop is from using the Trapezoidal rule to estimate int_0^1 v(s) ds with step size `dt1`.
    """
    dt1 = max(min(1.0, dt1_div_dt * dt), dt1_min)
    v = lambda s: [bmi / sqrt_dt for bmi in bm(s * dt + t0)]  # noqa

    # Estimate int_0^1 v(s) ds by Trapezoidal rule.
    # Based on Section 1.4 of Stochastic Numerics for Mathematical Physics.
    int_v_01 = [-v0 - v1 for v0, v1 in zip(v(0.0), v(1.0))]
    for t in torch.arange(0, 1 + 1e-7, dt1):
        int_v_01 = [a + 2. * b for a, b in zip(int_v_01, v(t))]
    int_v_01 = [a * dt1 / 2. for a in int_v_01]
    return [(dt ** (3 / 2) * a - dt * b).to(y0[0]) for a, b in zip(int_v_01, bm(t0))]
