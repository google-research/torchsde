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


class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith('__'))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()


# TODO: consider moving all these enums into some appropriate section of the code, rather than having them be global
#  like this. (e.g. instead set METHODS = {'euler': Euler, ...} in methods/__init__.py)
class METHODS(metaclass=ContainerMeta):
    euler = 'euler'
    milstein = 'milstein'
    srk = 'srk'
    midpoint = 'midpoint'
    heun = 'heun'
    log_ode_midpoint = 'log_ode'
    euler_heun = 'euler_heun'


class NOISE_TYPES(metaclass=ContainerMeta):  # noqa
    general = 'general'
    diagonal = 'diagonal'
    scalar = 'scalar'
    additive = 'additive'


class SDE_TYPES(metaclass=ContainerMeta):  # noqa
    ito = 'ito'
    stratonovich = 'stratonovich'


class LEVY_AREA_APPROXIMATIONS(metaclass=ContainerMeta):  # noqa
    none = 'none'  # Don't compute any Levy area approximation
    space_time = 'space-time'  # Only compute an (exact) space-time Levy area
    davie = 'davie'  # Compute Davie's approximation to Levy area
    foster = 'foster'  # Compute Foster's correction to Davie's approximation to Levy area


class METHOD_OPTIONS(metaclass=ContainerMeta):  # noqa
    grad_free = 'grad_free'
