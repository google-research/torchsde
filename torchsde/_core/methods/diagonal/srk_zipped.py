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

"""Strong order 1.5 scheme for diagonal noise SDEs from

Rößler, Andreas. "Runge–Kutta methods for the strong approximation of solutions of stochastic differential
equations." SIAM Journal on Numerical Analysis 48, no. 3 (2010): 922-952.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch

from torchsde._core import base_solver
from torchsde._core.methods import utils
from torchsde._core.methods.tableaus import srid2

STAGES, C0, C1, A0, A1, B0, B1, alpha, beta1, beta2, beta3, beta4 = (
    srid2.STAGES,
    srid2.C0, srid2.C1,
    srid2.A0, srid2.A1,
    srid2.B0, srid2.B1,
    srid2.alpha,
    srid2.beta1, srid2.beta2, srid2.beta3, srid2.beta4
)


class SRKDiagonal(base_solver.GenericSDESolver):

    def __init__(self, sde, bm, y0, dt, adaptive, rtol, atol, dt_min, options):
        super(SRKDiagonal, self).__init__(
            sde=sde, bm=bm, y0=y0, dt=dt, adaptive=adaptive, rtol=rtol, atol=atol, dt_min=dt_min, options=options)
        # Trapezoidal approximation of \int \int \dW_u \ds using only `bm` allows for truly deterministic behavior.
        self.trapezoidal_approx = self.options.get('trapezoidal_approx', True)
        self.dt1_min = self.options.get('dt1_min', 0.01)
        self.dt1_div_dt = self.options.get('dt1_div_dt', 10)

    def step(self, t0, y0, dt):
        assert dt > 0, 'Underflow in dt {}'.format(dt)

        sqrt_dt = torch.sqrt(dt) if isinstance(dt, torch.Tensor) else math.sqrt(dt)
        I_k = [(bm_next - bm_cur).to(y0[0]) for bm_next, bm_cur in zip(self.bm(t0 + dt), self.bm(t0))]
        I_kk = [(delta_bm_ ** 2. - dt) / 2. for delta_bm_ in I_k]
        I_k0 = (
            utils.compute_trapezoidal_approx(
                self.bm, t0, y0, dt, sqrt_dt, dt1_div_dt=self.dt1_div_dt, dt1_min=self.dt1_min
            ) if self.trapezoidal_approx else [
                dt / 2. * (delta_bm_ + torch.randn_like(delta_bm_) * sqrt_dt / math.sqrt(3))
                for delta_bm_ in I_k
            ]
        )
        I_kkk = [(delta_bm_ ** 3. - 3. * dt * delta_bm_) / 6. for delta_bm_ in I_k]

        t1, y1, H0, H1 = t0 + dt, y0, [], []
        for A0_s, A1_s, B0_s, B1_s, C0_s, C1_s, alpha_s, beta1_s, beta2_s, beta3_s, beta4_s in zip(
                A0, A1, B0, B1, C0, C1, alpha, beta1, beta2, beta3, beta4
        ):  # Suffix "_s" indexes stage.
            H0_s, H1_s = y0, y0  # Values at the current stage to be accumulated.
            for A0_s_j, A1_s_j, B0_s_j, B1_s_j, H0_j, H1_j, C0_j, C1_j in zip(
                    A0_s, A1_s, B0_s, B1_s, H0, H1, C0, C1
            ):
                f_eval = self.sde.f(t0 + C0_j * dt, H0_j)
                g_eval = self.sde.g(t0 + C1_j * dt, H1_j)
                H0_s = [
                    H0s_ + A0_s_j * f_eval_ * dt + B0_s_j * g_eval_ * I_k0_ / dt
                    for H0s_, f_eval_, g_eval_, I_k0_ in zip(H0_s, f_eval, g_eval, I_k0)
                ]
                H1_s = [
                    H1s_ + A1_s_j * f_eval_ * dt + B1_s_j * g_eval_ * sqrt_dt
                    for H1s_, f_eval_, g_eval_ in zip(H1_s, f_eval, g_eval)
                ]
            H0.append(H0_s)
            H1.append(H1_s)

            f_eval = self.sde.f(t0 + C0_s * dt, H0_s)
            g_eval = self.sde.g(t0 + C1_s * dt, H1_s)
            g_weight = [
                beta1_s * I_k_ + beta2_s * I_kk_ / sqrt_dt + beta3_s * I_k0_ / dt + beta4_s * I_kkk_ / dt
                for I_k_, I_kk_, I_k0_, I_kkk_ in zip(I_k, I_kk, I_k0, I_kkk)
            ]
            y1 = [
                y1_ + alpha_s * f_eval_ * dt + g_weight_ * g_eval_
                for y1_, f_eval_, g_eval_, g_weight_ in zip(y1, f_eval, g_eval, g_weight)
            ]
        return t1, y1
