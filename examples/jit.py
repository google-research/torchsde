import torch
import torchsde
from torchsde import sdeint
import timeit


class SDE(torchsde.SDEIto):

    def __init__(self, mu, sigma):
        super().__init__(noise_type="diagonal")

        self.mu = mu
        self.sigma = sigma

    @torch.jit.export
    def f(self, t, y):
        return self.mu * y

    @torch.jit.export
    def g(self, t, y):
        return self.sigma * y


batch_size, d, m = 4, 1, 1  # State dimension d, Brownian motion dimension m.
geometric_bm = SDE(mu=0.5, sigma=1)

# Works for torch==1.6.0.
geometric_bm = torch.jit.script(geometric_bm)

y0 = torch.zeros(batch_size, d).fill_(0.1)  # Initial state.
ts = torch.linspace(0, 1, 20)


def time_func():
    ys = sdeint(geometric_bm, y0, ts, adaptive=False, dt=ts[1], options={'trapezoidal_approx': True})


print(timeit.Timer(time_func).timeit(number=100))
