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

# From "RUNGE-KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS".
# For additive noise structure.
# (ODE order, SDE strong order) = (2.0, 1.5).

STAGES = 2

C0 = (0, 3 / 4)
C1 = (1 / 3, 1)

A0 = (
    (),
    (3 / 4,),
)

B0 = (
    (),
    (3 / 2,),
)

alpha = (1 / 3, 2 / 3)
beta1 = (0, 1)
beta2 = (-3 / 2, 3 / 2)
