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

import abc

from .. import adjoint_sde
from .. import base_solver
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHOD_OPTIONS


class BaseMilstein(base_solver.BaseSDESolver, metaclass=abc.ABCMeta):
    strong_order = 1.0
    weak_order = 1.0
    noise_types = (NOISE_TYPES.additive, NOISE_TYPES.diagonal, NOISE_TYPES.scalar)
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, options, **kwargs):
        if METHOD_OPTIONS.grad_free not in options:
            options[METHOD_OPTIONS.grad_free] = False
        if options[METHOD_OPTIONS.grad_free]:
            if sde.noise_type == NOISE_TYPES.additive:
                # dg=0 in this case, and gdg_prod is already setup to handle that, whilst the grad_free code path isn't.
                options[METHOD_OPTIONS.grad_free] = False
        if options[METHOD_OPTIONS.grad_free]:
            if isinstance(sde, adjoint_sde.AdjointSDE):
                # We need access to the diffusion to do things grad-free.
                raise ValueError(f"Derivative-free Milstein cannot be used for adjoint SDEs, because it requires "
                                 f"direct access to the diffusion, whilst adjoint SDEs rely on a more efficient "
                                 f"diffusion-vector product. Use derivative-using Milstein instead: "
                                 f"`adjoint_options=dict({METHOD_OPTIONS.grad_free}=False)`")
        super(BaseMilstein, self).__init__(sde=sde, options=options, **kwargs)

    @abc.abstractmethod
    def v_term(self, I_k, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def y_prime_f_factor(self, dt, f):
        raise NotImplementedError

    def step(self, t0, t1, y0):
        dt = t1 - t0
        I_k = self.bm(t0, t1)
        v = self.v_term(I_k, dt)

        f = self.sde.f(t0, y0)
        g_prod_I_k = self.sde.g_prod(t0, y0, I_k)

        if self.options[METHOD_OPTIONS.grad_free]:
            g = self.sde.g(t0, y0)
            g = g.squeeze(2) if g.dim() == 3 else g
            g_prod_v = self.sde.g_prod(t0, y0, v)
            sqrt_dt = dt.sqrt()
            y0_prime = y0 + self.y_prime_f_factor(dt, f) + g * sqrt_dt
            g_prod_v_prime = self.sde.g_prod(t0, y0_prime, v)
            gdg_prod = (g_prod_v_prime - g_prod_v) / sqrt_dt
        else:
            gdg_prod = self.sde.gdg_prod(t0, y0, v)

        y1 = y0 + f * dt + g_prod_I_k + .5 * gdg_prod

        return y1


class MilsteinIto(BaseMilstein):
    sde_type = SDE_TYPES.ito

    def v_term(self, I_k, dt):
        return I_k ** 2 - dt

    def y_prime_f_factor(self, dt, f):
        return dt * f


class MilsteinStratonovich(BaseMilstein):
    sde_type = SDE_TYPES.stratonovich

    def v_term(self, I_k, dt):
        return I_k ** 2

    def y_prime_f_factor(self, dt, f):
        return 0.
