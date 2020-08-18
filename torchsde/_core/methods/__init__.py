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

from ...settings import METHODS

from .euler import Euler
from .midpoint import Midpoint
from .milstein import Milstein
from .srk import SRK
from .heun import Heun
from .milstein_strat import MilsteinStrat


def select(method):
    if method == METHODS.euler:
        return Euler
    elif method == METHODS.milstein:
        return Milstein
    elif method == METHODS.srk:
        return SRK
    elif method == METHODS.midpoint:
        return Midpoint
    elif method == METHODS.heun:
        return Heun
    elif method == METHODS.milstein_strat:
        return MilsteinStrat
    else:
        raise ValueError(f"Method '{method}' does not match any known method.")
