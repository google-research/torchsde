import argparse
import logging
import time

import numpy.random as npr
import torch

import torchsde
import torchsde.brownian_lib as brownian_lib


t0, t1 = 0.0, 1.0
reps, steps = 10, 1000
small_batch_size, small_d = 128, 5
large_batch_size, large_d = 512, 128


def _compare(w0, ts, msg=''):
    bm = brownian_lib.BrownianPath(t0=t0, w0=w0)
    now = time.time()
    for _ in range(reps):
        for t in ts:
            bm(t)
    logging.warning(f'{msg} (brownian_lib.BrownianPath): {time.time() - now:.4f}')

    bm = torchsde.BrownianPath(t0=t0, w0=w0)
    now = time.time()
    for _ in range(reps):
        for t in ts:
            bm(t)
    logging.warning(f'{msg} (torchsde.BrownianPath): {time.time() - now:.4f}')


def sequential_access():
    ts = torch.linspace(t0, t1, steps=steps)

    w0 = torch.randn(small_batch_size, small_d).to(device)
    _compare(w0, ts, msg='small sequential access')

    w0 = torch.randn(large_batch_size, large_d).to(device)
    _compare(w0, ts, msg='large sequential access')


def random_access():
    ts = torch.empty(steps).uniform_(t0, t1)

    w0 = torch.randn(small_batch_size, small_d).to(device)
    _compare(w0, ts, msg='small random access')

    w0 = torch.randn(large_batch_size, large_d).to(device)
    _compare(w0, ts, msg='large random access')


def main():
    sequential_access()
    random_access()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')

    npr.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    main()
