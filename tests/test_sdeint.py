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

import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import unittest

import torch

from tests import basic_sde
from tests.torch_test import TorchTestCase
from torchsde import BrownianInterval, sdeint

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 16
d = 3
m = 2
t0 = 0.0
t1 = 0.3
T = 5
dt = 1e-2
dtype = torch.get_default_dtype()
ts = torch.linspace(t0, t1, steps=T, device=device)
y0 = torch.ones(batch_size, d, device=device)

basic_sdes = (
    basic_sde.BasicSDE1(d=d).to(device),
    basic_sde.BasicSDE2(d=d).to(device),
    basic_sde.BasicSDE3(d=d).to(device),
    basic_sde.BasicSDE4(d=d).to(device),
)

# Make bms explicitly for testing.
bm_diagonal = BrownianInterval(
    t0=t0, t1=t1, shape=(batch_size, d), dtype=dtype, device=device, levy_area_approximation='space-time'
)
bm_general = BrownianInterval(
    t0=t0, t1=t1, shape=(batch_size, m), dtype=dtype, device=device, levy_area_approximation='space-time'
)
bm_scalar = BrownianInterval(
    t0=t0, t1=t1, shape=(batch_size, 1), dtype=dtype, device=device, levy_area_approximation='space-time'
)


class TestSdeint(TorchTestCase):

    def test_rename_methods(self):
        # Test renaming works with a subset of names when `logqp=False`.
        sde = basic_sde.CustomNamesSDE().to(device)
        ans = sdeint(sde, y0, ts, dt=dt, names={'drift': 'forward'})
        self.assertEqual(ans.shape, (T, batch_size, d))

        # Test renaming works with a subset of names when `logqp=True`.
        sde = basic_sde.CustomNamesSDELogqp().to(device)
        ans = sdeint(sde, y0, ts, dt=dt, names={'drift': 'forward', 'prior_drift': 'w'}, logqp=True)
        self.assertEqual(ans[0].shape, (T, batch_size, d))
        self.assertEqual(ans[1].shape, (T - 1, batch_size))

    def test_sdeint_general(self):
        sde = basic_sde.GeneralSDE(d=d, m=m).to(device)
        for method in ('euler',):
            self._test_sdeint(sde, bm=bm_general, adaptive=False, method=method, dt=dt)
            self._test_sdeint_logqp(sde, bm=bm_general, adaptive=False, method=method, dt=dt)

    def test_sdeint_additive(self):
        sde = basic_sde.AdditiveSDE(d=d, m=m).to(device)
        for method in ('euler', 'milstein', 'srk'):
            self._test_sdeint(sde, bm=bm_general, adaptive=False, method=method, dt=dt)
            self._test_sdeint_logqp(sde, bm=bm_general, adaptive=False, method=method, dt=dt)

    def test_sde_scalar(self):
        sde = basic_sde.ScalarSDE(d=d, m=m).to(device)
        for method in ('euler', 'milstein', 'srk'):
            self._test_sdeint(sde, bm=bm_scalar, adaptive=False, method=method, dt=dt)
            self._test_sdeint_logqp(sde, bm=bm_scalar, adaptive=False, method=method, dt=dt)

    def test_srk_determinism(self):
        # srk for additive.
        sde = basic_sde.AdditiveSDE(d=d, m=m).to(device)
        ys1 = sdeint(sde, y0, ts, bm=bm_general, adaptive=False, method='srk', dt=dt)
        ys2 = sdeint(sde, y0, ts, bm=bm_general, adaptive=False, method='srk', dt=dt)
        self.tensorAssertAllClose(ys1, ys2)

        # srk for diagonal.
        sde = basic_sde.BasicSDE1(d=d).to(device)
        ys1 = sdeint(sde, y0, ts, bm=bm_diagonal, adaptive=False, method='srk', dt=dt)
        ys2 = sdeint(sde, y0, ts, bm=bm_diagonal, adaptive=False, method='srk', dt=dt)
        self.tensorAssertAllClose(ys1, ys2)

    # All tests below for diagonal noise. These cases are to see if solvers still work when some of the functions don't
    # depend on the states/params and when some states/params don't require gradients.
    def test_sdeint_fixed(self):
        for sde in basic_sdes:
            for method in ('euler', 'milstein', 'srk'):
                self._test_sdeint(sde, bm_diagonal, adaptive=False, method=method, dt=dt)

    def test_sdeint_adaptive(self):
        for sde in basic_sdes:
            for method in ('milstein', 'srk'):
                self._test_sdeint(sde, bm_diagonal, adaptive=True, method=method, dt=dt)

    def test_sdeint_logqp_fixed(self):
        for sde in basic_sdes:
            for method in ('euler', 'milstein', 'srk'):
                self._test_sdeint_logqp(sde, bm_diagonal, adaptive=False, method=method, dt=dt)

    def test_sdeint_logqp_adaptive(self):
        for sde in basic_sdes:
            for method in ('milstein', 'srk'):
                self._test_sdeint_logqp(sde, bm_diagonal, adaptive=True, method=method, dt=dt)

    def test_sdeint_tuple_sde(self):
        y0_ = (y0,)  # Make tuple input.
        sde = basic_sde.TupleSDE(d=d).to(device)

        for method in ('euler', 'milstein', 'srk'):
            ans = sdeint(sde, y0_, ts, method=method, dt=dt)
            self.assertTrue(isinstance(ans, tuple))
            self.assertEqual(ans[0].size(), (T, batch_size, d))

    def _test_sdeint(self, sde, bm, adaptive, method, dt):
        # Using `f` as drift.
        with torch.no_grad():
            ans = sdeint(sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive)
        self.assertEqual(ans.shape, (T, batch_size, d))

        # Using `h` as drift.
        with torch.no_grad():
            ans = sdeint(sde, y0, ts, bm, method=method, dt=dt, adaptive=adaptive, names={'drift': 'h'})
        self.assertEqual(ans.shape, (T, batch_size, d))

    def _test_sdeint_logqp(self, sde, bm, adaptive, method, dt):
        # Using `f` as drift.
        with torch.no_grad():
            ans, logqp = sdeint(sde, y0, ts, bm, logqp=True, method=method, dt=dt, adaptive=adaptive)
        self.assertEqual(ans.shape, (T, batch_size, d))
        self.assertEqual(logqp.shape, (T - 1, batch_size))

        # Using `h` as drift.
        with torch.no_grad():
            ans, logqp = sdeint(
                sde, y0, ts, bm, logqp=True, method=method, dt=dt, adaptive=adaptive, names={'drift': 'h'}
            )
        self.assertEqual(ans.shape, (T, batch_size, d))
        self.assertEqual(logqp.shape, (T - 1, batch_size))


if __name__ == '__main__':
    unittest.main()
