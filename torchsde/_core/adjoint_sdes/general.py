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

"""Define the class of the adjoint SDE when the original forward SDE has general noise."""

import torch

from .. import base_sde
from .. import misc
from ... import settings


class AdjointSDEGeneralStratonovich(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEGeneralStratonovich, self).__init__(base_sde=sde, noise_type=settings.NOISE_TYPES.general)
        self.params = params

    def f(self, t, y_aug):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            minus_adj_y = [-adj_y_.detach() for adj_y_ in adj_y]
            minus_f = [-f_ for f_ in sde.f(-t, y)]
            vjp_y_and_params = misc.grad(
                outputs=minus_f,
                inputs=y + params,
                grad_outputs=minus_adj_y,
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*minus_f, *vjp_y, vjp_params)

    def g_prod(self, t, y_aug, v):
        sde, params, n_tensors = self._base_sde, self.params, len(y_aug) // 2
        y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]

        with torch.enable_grad():
            y = [y_.detach().requires_grad_(True) for y_ in y]
            minus_adj_y = [-adj_y_.detach() for adj_y_ in adj_y]
            minus_g = [-g_ for g_ in sde.g(-t, y)]
            minus_g_prod = misc.seq_batch_mvp(minus_g, v)
            minus_g_weighted = [(minus_g_ * v_.unsqueeze(-2)).sum(-1) for minus_g_, v_ in zip(minus_g, v)]
            vjp_y_and_params = misc.grad(
                outputs=minus_g_weighted,
                inputs=y + params,
                grad_outputs=minus_adj_y,
                allow_unused=True,
            )
            vjp_y, vjp_params = vjp_y_and_params[:n_tensors], vjp_y_and_params[n_tensors:]
            vjp_params = misc.flatten(vjp_params)

        return (*minus_g_prod, *vjp_y, vjp_params)
