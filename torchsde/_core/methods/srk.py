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

"""Strong order 1.5 scheme from

Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions
of stochastic differential equations." SIAM Journal on Numerical Analysis 48,
no. 3 (2010): 922-952.
"""

from .tableaus import sra1, srid2
from .. import adjoint_sde
from .. import base_solver
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS

_r2 = 1 / 2
_r6 = 1 / 6


class SRK(base_solver.BaseSDESolver):
    strong_order = 1.5
    weak_order = 1.5
    sde_type = SDE_TYPES.ito
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.scalar)
    levy_area_approximations = (LEVY_AREA_APPROXIMATIONS.space_time,
                                LEVY_AREA_APPROXIMATIONS.davie,
                                LEVY_AREA_APPROXIMATIONS.foster)

    def __init__(self, sde, **kwargs):
        if sde.noise_type == NOISE_TYPES.additive:
            self.step = self.additive_step
        else:
            self.step = self.diagonal_or_scalar_step

        if isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"Stochastic Runge–Kutta methods cannot be used for adjoint SDEs, because it requires "
                             f"direct access to the diffusion, whilst adjoint SDEs rely on a more efficient "
                             f"diffusion-vector product. Use a different method instead.")

        super(SRK, self).__init__(sde=sde, **kwargs)

    def step(self, t0, t1, y):
        # Just to make @abstractmethod happy, as we assign during __init__.
        raise RuntimeError

    def diagonal_or_scalar_step(self, t0, t1, y0):
        dt = t1 - t0
        rdt = 1 / dt
        sqrt_dt = dt.sqrt()
        I_k, I_k0 = self.bm(t0, t1, return_U=True)
        I_kk = (I_k ** 2 - dt) * _r2
        I_kkk = (I_k ** 3 - 3 * dt * I_k) * _r6

        y1 = y0
        H0, H1 = [], []
        for s in range(srid2.STAGES):
            H0s, H1s = y0, y0  # Values at the current stage to be accumulated.
            for j in range(s):
                f = self.sde.f(t0 + srid2.C0[j] * dt, H0[j])
                g = self.sde.g(t0 + srid2.C1[j] * dt, H1[j])
                g = g.squeeze(2) if g.dim() == 3 else g
                H0s = H0s + srid2.A0[s][j] * f * dt + srid2.B0[s][j] * g * I_k0 * rdt
                H1s = H1s + srid2.A1[s][j] * f * dt + srid2.B1[s][j] * g * sqrt_dt
            H0.append(H0s)
            H1.append(H1s)

            f = self.sde.f(t0 + srid2.C0[s] * dt, H0s)
            g_weight = (
                    srid2.beta1[s] * I_k +
                    srid2.beta2[s] * I_kk / sqrt_dt +
                    srid2.beta3[s] * I_k0 * rdt +
                    srid2.beta4[s] * I_kkk * rdt
            )
            g_prod = self.sde.g_prod(t0 + srid2.C1[s] * dt, H1s, g_weight)
            y1 = y1 + srid2.alpha[s] * f * dt + g_prod
        return y1

    def additive_step(self, t0, t1, y0):
        dt = t1 - t0
        rdt = 1 / dt
        I_k, I_k0 = self.bm(t0, t1, return_U=True)

        y1 = y0
        H0 = []
        for i in range(sra1.STAGES):
            H0i = y0
            for j in range(i):
                f = self.sde.f(t0 + sra1.C0[j] * dt, H0[j])
                g_weight = sra1.B0[i][j] * I_k0 * rdt
                g_prod = self.sde.g_prod(t0 + sra1.C1[j] * dt, y0, g_weight)
                H0i = H0i + sra1.A0[i][j] * f * dt + g_prod
            H0.append(H0i)

            f = self.sde.f(t0 + sra1.C0[i] * dt, H0i)
            g_weight = sra1.beta1[i] * I_k + sra1.beta2[i] * I_k0 * rdt
            g_prod = self.sde.g_prod(t0 + sra1.C1[i] * dt, y0, g_weight)
            y1 = y1 + sra1.alpha[i] * f * dt + g_prod
        return y1
