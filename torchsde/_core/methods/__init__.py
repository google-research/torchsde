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

from ..settings import METHODS, NOISE_TYPES

from .additive.adjoint_sde import AdjointSDEAdditive, AdjointSDEAdditiveLogqp
from .diagonal.adjoint_sde import AdjointSDEDiagonal, AdjointSDEDiagonalLogqp
from .scalar.adjoint_sde import AdjointSDEScalar, AdjointSDEScalarLogqp

from .euler import AdditiveEuler, GeneralEuler
from .milstein import Milstein
from .srk import AdditiveSRK, DiagonalSRK


def select(method, noise_type):
    if method == METHODS.euler:
        if noise_type == NOISE_TYPES.additive:
            return AdditiveEuler
        else:
            return GeneralEuler
    elif method == METHODS.milstein:
        return Milstein
    elif method == METHODS.srk:
        if noise_type == NOISE_TYPES.additive:
            return AdditiveSRK
        else:
            return DiagonalSRK
    else:
        raise ValueError
