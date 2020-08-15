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

Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of stochastic differential
equations." SIAM Journal on Numerical Analysis 48, no. 3 (2010): 922-952.
"""

import math

import torch

from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS

from .. import base_solver
from .. import misc

from .tableaus import sra1, srid2


class BaseSRK(base_solver.BaseSDESolver):
    strong_order = 1.5
    weak_order = 1.5
    sde_type = SDE_TYPES.ito
    levy_area_approximation = LEVY_AREA_APPROXIMATIONS.spacetime

    def __init__(self, sde, bm, y0, dt, adaptive, rtol, atol, dt_min, options, **kwargs):
        super(BaseSRK, self).__init__(sde=sde, bm=bm, y0=y0, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol,
                                      dt_min=dt_min, options=options, **kwargs)
        self.dt1_min = self.options.get('dt1_min', 0.01)
        self.dt1_div_dt = self.options.get('dt1_div_dt', 10)


class DiagonalSRK(BaseSRK):
    noise_types = (NOISE_TYPES.diagonal, NOISE_TYPES.scalar)

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        sqrt_dt = torch.sqrt(dt) if isinstance(dt, torch.Tensor) else math.sqrt(dt)
        I_k, H = self.bm(t0, t0 + dt)
        I_k0 = dt * (H + 0.5 * I_k)
        I_kk = [(delta_bm_ ** 2. - dt) / 2. for delta_bm_ in I_k]
        I_kkk = [(delta_bm_ ** 3. - 3. * dt * delta_bm_) / 6. for delta_bm_ in I_k]

        t1, y1 = t0 + dt, y0
        H0, H1 = [], []
        for s in range(srid2.STAGES):
            H0s, H1s = y0, y0  # Values at the current stage to be accumulated.
            for j in range(s):
                f_eval = self.sde.f(t0 + srid2.C0[j] * dt, H0[j])
                g_eval = self.sde.g(t0 + srid2.C1[j] * dt, H1[j])
                H0s = [
                    H0s_ + srid2.A0[s][j] * f_eval_ * dt + srid2.B0[s][j] * g_eval_ * I_k0_ / dt
                    for H0s_, f_eval_, g_eval_, I_k0_ in zip(H0s, f_eval, g_eval, I_k0)
                ]
                H1s = [
                    H1s_ + srid2.A1[s][j] * f_eval_ * dt + srid2.B1[s][j] * g_eval_ * sqrt_dt
                    for H1s_, f_eval_, g_eval_ in zip(H1s, f_eval, g_eval)
                ]
            H0.append(H0s)
            H1.append(H1s)

            f_eval = self.sde.f(t0 + srid2.C0[s] * dt, H0s)
            g_eval = self.sde.g(t0 + srid2.C1[s] * dt, H1s)
            g_weight = [
                srid2.beta1[s] * I_k_ + srid2.beta2[s] * I_kk_ / sqrt_dt +
                srid2.beta3[s] * I_k0_ / dt + srid2.beta4[s] * I_kkk_ / dt
                for I_k_, I_kk_, I_k0_, I_kkk_ in zip(I_k, I_kk, I_k0, I_kkk)
            ]
            y1 = [
                y1_ + srid2.alpha[s] * f_eval_ * dt + g_weight_ * g_eval_
                for y1_, f_eval_, g_eval_, g_weight_ in zip(y1, f_eval, g_eval, g_weight)
            ]
        return t1, y1


class AdditiveSRK(BaseSRK):
    noise_types = (NOISE_TYPES.additive,)

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        I_k, I_k0 = self.bm(t0, t0 + dt)

        t1, y1 = t0 + dt, y0
        H0 = []
        for i in range(sra1.STAGES):
            H0i = y0
            for j in range(i):
                f_eval = self.sde.f(t0 + sra1.C0[j] * dt, H0[j])
                g_eval = self.sde.g(t0 + sra1.C1[j] * dt, y0)  # The state should not affect the diffusion.
                H0i = [
                    H0i_ + sra1.A0[i][j] * f_eval_ * dt + sra1.B0[i][j] * misc.batch_mvp(g_eval_, I_k0_) / dt
                    for H0i_, f_eval_, g_eval_, I_k0_ in zip(H0i, f_eval, g_eval, I_k0)
                ]
            H0.append(H0i)

            f_eval = self.sde.f(t0 + sra1.C0[i] * dt, H0i)
            g_eval = self.sde.g(t0 + sra1.C1[i] * dt, y0)
            g_weight = [sra1.beta1[i] * I_k_ + sra1.beta2[i] * I_k0_ / dt for I_k_, I_k0_ in zip(I_k, I_k0)]
            y1 = [
                y1_ + sra1.alpha[i] * f_eval_ * dt + misc.batch_mvp(g_eval_, g_weight_)
                for y1_, f_eval_, g_eval_, g_weight_ in zip(y1, f_eval, g_eval, g_weight)
            ]
        return t1, y1
