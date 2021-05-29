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

import torch

from .. import adjoint_sde
from .. import base_solver
from .. import misc
from ...settings import SDE_TYPES, NOISE_TYPES, LEVY_AREA_APPROXIMATIONS, METHODS, METHOD_OPTIONS


class ReversibleHeun(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        super(ReversibleHeun, self).__init__(sde=sde, **kwargs)

    def init_extra_solver_state(self, t0, y0):
        return self.sde.f_and_g(t0, y0) + (y0,)

    def step(self, t0, t1, y0, extra0):
        f0, g0, z0 = extra0
        dt = t1 - t0
        dW = self.bm(t0, t1)

        z1 = 2 * y0 - z0 + f0 * dt + self.sde.prod(g0, dW)
        f1, g1 = self.sde.f_and_g(t1, z1)
        y1 = y0 + (f0 + f1) * (0.5 * dt) + self.sde.prod(g0 + g1, 0.5 * dW)

        return y1, (f1, g1, z1)


class AdjointReversibleHeun(base_solver.BaseSDESolver):
    weak_order = 1.0
    sde_type = SDE_TYPES.stratonovich
    noise_types = NOISE_TYPES.all()
    levy_area_approximations = LEVY_AREA_APPROXIMATIONS.all()

    def __init__(self, sde, **kwargs):
        if not isinstance(sde, adjoint_sde.AdjointSDE):
            raise ValueError(f"{METHODS.adjoint_reversible_heun} can only be used for adjoint_method.")
        self.strong_order = 1.0 if sde.noise_type == NOISE_TYPES.additive else 0.5
        super(AdjointReversibleHeun, self).__init__(sde=sde, **kwargs)
        self.forward_sde = sde.base_sde

        if self.forward_sde.noise_type == NOISE_TYPES.diagonal:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1 * tensor2
        else:
            self._adjoint_of_prod = lambda tensor1, tensor2: tensor1.unsqueeze(-1) * tensor2.unsqueeze(-2)

    def init_extra_solver_state(self, t0, y0):
        # We expect to always be given the extra state from the forward pass.
        raise RuntimeError("Please report a bug to torchsde.")

    def step(self, t0, t1, y0, extra0):
        forward_f0, forward_g0, forward_z0 = extra0
        dt = t1 - t0
        dW = self.bm(t0, t1)
        half_dt = 0.5 * dt
        half_dW = 0.5 * dW
        forward_y0, adj_y0, (adj_f0, adj_g0, adj_z0, *adj_params), requires_grad = self.sde.get_state(t0, y0,
                                                                                                      extra_states=True)
        adj_y0_half_dt = adj_y0 * half_dt
        adj_y0_half_dW = self._adjoint_of_prod(adj_y0, half_dW)

        forward_z1 = 2 * forward_y0 - forward_z0 - forward_f0 * dt - self.forward_sde.prod(forward_g0, dW)

        adj_y1 = adj_y0
        adj_f1 = adj_y0_half_dt
        adj_f0 = adj_f0 + adj_y0_half_dt
        adj_g1 = adj_y0_half_dW
        adj_g0 = adj_g0 + adj_y0_half_dW

        # TODO: efficiency. It should be possible to make one fewer forward call by re-using the forward computation
        #  in the previous step.
        with torch.enable_grad():
            if not forward_z0.requires_grad:
                forward_z0 = forward_z0.detach().requires_grad_()
            re_forward_f0, re_forward_g0 = self.forward_sde.f_and_g(-t0, forward_z0)

            vjp_z, *vjp_params = misc.vjp(outputs=(re_forward_f0, re_forward_g0),
                                          inputs=[forward_z0] + self.sde.params,
                                          grad_outputs=[adj_f0, adj_g0],
                                          allow_unused=True,
                                          retain_graph=True,
                                          create_graph=requires_grad)
        adj_z0 = adj_z0 + vjp_z
        assert len(adj_params) == len(vjp_params)
        adj_params = [adj_param_i + vjp_param_i for adj_param_i, vjp_param_i in zip(adj_params, vjp_params)]

        forward_f1, forward_g1 = self.forward_sde.f_and_g(-t1, forward_z1)
        forward_y1 = forward_y0 - (forward_f0 + forward_f1) * half_dt - self.forward_sde.prod(forward_g0 + forward_g1,
                                                                                              half_dW)

        adj_y1 = adj_y1 + 2 * adj_z0
        adj_z1 = -adj_z0
        adj_f1 = adj_f1 + adj_z0 * dt
        adj_g1 = adj_g1 + self._adjoint_of_prod(adj_z0, dW)

        y1 = misc.flatten([forward_y1, adj_y1, adj_f1, adj_g1, adj_z1] + adj_params).unsqueeze(0)

        return y1, (forward_f1, forward_g1, forward_z1)
