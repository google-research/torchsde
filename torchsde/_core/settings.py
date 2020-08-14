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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ContainerMeta(type):
    def __contains__(cls, item):
        return item in cls.__dict__.values()


class METHODS(metaclass=ContainerMeta):
    euler = 'euler'
    milstein = 'milstein'
    srk = 'srk'


class NOISE_TYPES(metaclass=ContainerMeta):
    general = 'general'
    diagonal = 'diagonal'
    scalar = 'scalar'
    additive = 'additive'


class SDE_TYPES(metaclass=ContainerMeta):
    ito = 'ito'
    stratonovich = 'stratonovich'


class LEVY_AREA_APPROXIMATIONS(metaclass=ContainerMeta):
    none = 'none'
    davie = 'davie'
    foster = 'foster'
