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

#include "brownian_path.hpp"

#include <math.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iterator>
#include <map>

#include "utils.hpp"

BrownianPath::BrownianPath(double t0, torch::Tensor w0) {
  cache.insert(std::pair<double, torch::Tensor>(t0, w0));
}

BrownianPath::BrownianPath(std::map<double, torch::Tensor> data) {
  cache = data;
}

torch::Tensor BrownianPath::call(double t) {
  auto head = cache.begin();
  auto tail = cache.rbegin();

  if (t > tail->first) {
    auto w =
        tail->second + torch::randn_like(tail->second) * sqrt(t - tail->first);
    cache.insert(std::pair<double, torch::Tensor>(t, w));
    return w;
  } else if (t < head->first) {
    auto w =
        head->second + torch::randn_like(head->second) * sqrt(head->first - t);
    cache.insert(std::pair<double, torch::Tensor>(t, w));
    return w;
  } else if (cache.find(t) != cache.end()) {
    return cache.at(t);
  } else {
    auto lo = cache.lower_bound(t);
    auto hi = lo--;
    auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
    cache.insert(std::pair<double, torch::Tensor>(t, w));
    return w;
  }
}

void BrownianPath::insert(double t, torch::Tensor w) {
  cache.insert(std::pair<double, torch::Tensor>(t, w));
}

std::string BrownianPath::repr() const {
  double t_head = cache.begin()->first;
  double t_tail = cache.rbegin()->first;
  return "BrownianPath(t0=" + format_float(t_head, 3) +
         ", t1=" + format_float(t_tail, 3) + ")";
}

std::map<double, torch::Tensor> BrownianPath::get_cache() const {
  return cache;
}

double BrownianPath::get_t_head() const { return cache.begin()->first; }

double BrownianPath::get_t_tail() const { return cache.rbegin()->first; }

torch::Tensor BrownianPath::get_w_head() const { return cache.begin()->second; }

torch::Tensor BrownianPath::get_w_tail() const {
  return cache.rbegin()->second;
}
