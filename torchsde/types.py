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

# We import from `typing` more than what's enough, so that other modules can import from this file and not `typing`.
from typing import Sequence, Union, Optional, Any, Dict, Tuple, Callable

import torch

Tensor = torch.Tensor
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]

Scalar = Union[float, Tensor]
Vector = Union[Sequence[float], Tensor]
