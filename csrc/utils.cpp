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

#include <math.h>
#include <torch/torch.h>

#include <iomanip>
#include <sstream>

torch::Tensor brownian_bridge(float t, float t0, float t1, torch::Tensor w0,
                              torch::Tensor w1) {
  auto mean = ((t1 - t) * w0 + (t - t0) * w1) / (t1 - t0);
  auto std = std::sqrt((t1 - t) * (t - t0) / (t1 - t0));
  return mean + torch::randn_like(mean) * std;
}

std::string format_float(float t, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << t;
  return stream.str();
}
