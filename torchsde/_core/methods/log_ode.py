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

"""Log-ODE scheme constructed by combining Lie-Trotter splitting with the explicit midpoint method.

The scheme uses Levy area approximations.
"""

from .. import adjoint_sde
from .. import base_solver
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS


class LogODEMidpoint(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.scalar, NOISE_TYPES.general)
    levy_area_approximations = (LEVY_AREA_APPROXIMATIONS.davie, LEVY_AREA_APPROXIMATIONS.foster)

    def __init__(self, sde, **kwargs):
        if isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"Log-ODE schemes cannot be used for adjoint SDEs, because they require "
                             f"direct access to the diffusion, whilst adjoint SDEs rely on a more efficient "
                             f"diffusion-vector product. Use a different method instead.")
        self.strong_order = 0.5 if sde.noise_type == NOISE_TYPES.general else 1.0
        super(LogODEMidpoint, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y0):
        dt = t1 - t0
        I_k, A = self.bm(t0, t1, return_A=True)

        f = self.sde.f(t0, y0)
        g_prod = self.sde.g_prod(t0, y0, I_k)

        half_dt = 0.5 * dt

        t_prime = t0 + half_dt
        y_prime = y0 + half_dt * f + .5 * g_prod

        f_prime = self.sde.f(t_prime, y_prime)
        g_prod_prime = self.sde.g_prod(t_prime, y_prime, I_k)
        dg_ga_prime = self.sde.dg_ga_jvp_column_sum(t_prime, y_prime, A)

        y1 = y0 + dt * f_prime + g_prod_prime + dg_ga_prime

        return y1
