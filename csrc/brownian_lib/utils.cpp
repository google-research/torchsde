/* Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "utils.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <TH/TH.h>
#include <structmember.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/tensor_types.h>

#ifdef USE_CUDA
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCTensorRandom.h>
#endif

#include <math.h>
#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <sstream>

torch::Tensor brownian_bridge(float t, float t0, float t1,
                              torch::Tensor const &w0,
                              torch::Tensor const &w1) {
  auto mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0);
  auto std = std::sqrt((t1 - t) * (t - t0) / (t1 - t0));
  return mean + torch::randn_like(mean) * std;
}

std::string format_float(float t, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << t;
  return stream.str();
}

torch::Tensor brownian_bridge_with_seed(double t, double t0, double t1,
                                        torch::Tensor const &w0,
                                        torch::Tensor const &w1,
                                        std::uint64_t seed) {
  auto device = w0.device();
  at::Generator generator;

  // Adapted from:
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Generator.cpp.
#ifdef USE_CUDA
  if (device.type() == at::kCPU) {
    generator = torch::make_generator<at::CPUGeneratorImpl>();
  } else if (device.type() == at::kCUDA) {
    generator = torch::make_generator<at::CUDAGeneratorImpl>(device.index());
  } else {
    AT_ERROR("Device type ", c10::DeviceTypeName(device.type()),
             " is not supported for torch.Generator() api.");
  }
#else
  TORCH_CHECK(device.type() == at::kCPU, "Device type ",
              c10::DeviceTypeName(device.type()),
              " is not supported for torch.Generator() api.");
  generator = torch::make_generator<at::CPUGeneratorImpl>();
#endif
  generator.set_current_seed(seed);
  torch::Tensor mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0);
  double std = std::sqrt((t1 - t) * (t - t0) / (t1 - t0));
  torch::Tensor bridge_point = at::normal(mean, std, generator);
  return bridge_point;
}

torch::Tensor binary_search_with_seed(double t, double t0, double t1,
                                      torch::Tensor w0, torch::Tensor w1,
                                      std::uint64_t root, double tol) {
  std::seed_seq seq({root});
  std::vector<std::uint64_t> seeds(3);
  seq.generate(seeds.begin(), seeds.end());

  std::uint64_t seedv = seeds[0];
  std::uint64_t seedl = seeds[1];
  std::uint64_t seedr = seeds[2];

  auto t_mid = (t0 + t1) / 2;
  auto w_mid = brownian_bridge_with_seed(t_mid, t0, t1, w0, w1, seedv);

  while (std::abs(t - t_mid) > tol) {
    if (t < t_mid) {
      t1 = t_mid;
      w1 = w_mid;
      root = seedl;
    } else {
      t0 = t_mid;
      w0 = w_mid;
      root = seedr;
    }

    std::seed_seq seq({root});
    seq.generate(seeds.begin(), seeds.end());
    seedv = seeds[0];
    seedl = seeds[1];
    seedr = seeds[2];

    t_mid = (t0 + t1) / 2;
    w_mid = brownian_bridge_with_seed(t_mid, t0, t1, w0, w1, seedv);
  }
  return w_mid;
}

void populate_cache(double t0, torch::Tensor const &w0, double t1, int entropy,
                    int cache_depth, std::map<double, torch::Tensor> &cache,
                    std::vector<std::uint64_t> &seeds) {
  auto k = std::pow(2, cache_depth);
  double dt = (t1 - t0) / k;

  auto t = t0;
  auto w = w0;
  for (int i = 0; i <= k; i++) {
    cache.insert(std::pair<double, torch::Tensor>(t, w));
    t = t + dt;
    w = w + torch::randn_like(w) * sqrt(dt);
  }

  std::seed_seq seq({entropy});
  seq.generate(seeds.begin(), seeds.end());
}
