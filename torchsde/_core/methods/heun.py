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

"""Stratonovich Heun method (strong order 1.0 scheme) from

Burrage K., Burrage P. M. and Tian T. 2004 "Numerical methods for strong solutions
of stochastic differential equations: an overview" Proc. R. Soc. Lond. A. 460: 373–402.
"""

from .. import base_solver
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS


class Heun(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.general, NOISE_TYPES.scalar)
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 0.5 if sde.noise_type == NOISE_TYPES.general else 1.0
        super(Heun, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y0):
        dt = t1 - t0
        I_k = self.bm(t0, t1)

        f = self.sde.f(t0, y0)
        g_prod = self.sde.g_prod(t0, y0, I_k)

        y0_prime = y0 + dt * f + g_prod

        f_prime = self.sde.f(t1, y0_prime)
        g_prod_prime = self.sde.g_prod(t1, y0_prime, I_k)

        y1 = y0 + (dt * (f + f_prime) + g_prod + g_prod_prime) * 0.5

        return y1
