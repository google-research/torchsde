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

BrownianPath::BrownianPath(float t0, torch::Tensor w0) {
  cache.insert(std::pair<float, torch::Tensor>(t0, w0));
  t_head = t0;
  t_tail = t0;
  w_head = w0;
  w_tail = w0;
}

BrownianPath::BrownianPath(std::map<float, torch::Tensor> data) {
  for (auto const& x : data) {
    cache.insert(std::pair<float, torch::Tensor>(x.first, x.second));
  }

  auto head = data.begin();
  auto tail = data.rbegin();
  t_head = head->first;
  t_tail = tail->first;
  w_head = head->second;
  w_tail = tail->second;
}

torch::Tensor BrownianPath::call(float t) {
  if (t > t_tail) {
    auto w = w_tail + torch::randn_like(w_tail) * sqrt(t - t_tail);
    cache.insert(std::pair<float, torch::Tensor>(t, w));
    t_tail = t;
    w_tail = w;
    return w;
  } else if (t < t_head) {
    auto w = w_head + torch::randn_like(w_head) * sqrt(t_head - t);
    cache.insert(std::pair<float, torch::Tensor>(t, w));
    t_head = t;
    w_head = w;
    return w;
  } else if (cache.find(t) != cache.end()) {
    return cache.at(t);
  } else {
    auto lo = cache.lower_bound(t);
    auto hi = lo--;
    auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
    cache.insert(std::pair<float, torch::Tensor>(t, w));
    return w;
  }
}

void BrownianPath::insert(float t, torch::Tensor w) {
  cache.insert(std::pair<float, torch::Tensor>(t, w));

  if (t < t_head) {
    t_head = t;
    w_head = w;
  } else if (t > t_tail) {
    t_tail = t;
    w_tail = w;
  }
}

std::string BrownianPath::repr() const {
  return "BrownianPath(t0=" + format_float(t_head, 3) +
         ", t1=" + format_float(t_tail, 3) + ")";
}

std::map<float, torch::Tensor> BrownianPath::get_cache() const { return cache; }

float BrownianPath::get_t_head() const { return t_head; }

float BrownianPath::get_t_tail() const { return t_tail; }

torch::Tensor BrownianPath::get_w_head() const { return w_head; }

torch::Tensor BrownianPath::get_w_tail() const { return w_tail; }
