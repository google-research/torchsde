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

"""Define the class of the adjoint SDE when the original forward SDE has scalar noise."""

from .. import base_sde


class AdjointSDEScalar(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEScalar, self).__init__(sde, noise_type="scalar")
        self.params = params

    def f(self, t, y_aug):
        raise NotImplementedError('Adjoint mode for scalar noise SDEs not supported.')

    def g(self, t, y):
        raise NotImplementedError('Adjoint mode for scalar noise SDEs not supported.')

    def h(self, t, y):
        raise NotImplementedError('Adjoint mode for scalar noise SDEs not supported.')

    def g_prod(self, t, y, v):
        raise NotImplementedError('Adjoint mode for scalar noise SDEs not supported.')

    def gdg_prod(self, t, y, v):
        raise NotImplementedError("This method shouldn't be called.")


class AdjointSDEScalarLogqp(base_sde.AdjointSDE):

    def __init__(self, sde, params):
        super(AdjointSDEScalarLogqp, self).__init__(sde, noise_type="scalar")
        self.params = params

    def f(self, t, y_aug):
        raise NotImplementedError

    def g(self, t, y):
        raise NotImplementedError

    def h(self, t, y):
        raise NotImplementedError

    def g_prod(self, t, y, v):
        raise NotImplementedError

    def gdg_prod(self, t, y, v):
        raise NotImplementedError("This method shouldn't be called.")
