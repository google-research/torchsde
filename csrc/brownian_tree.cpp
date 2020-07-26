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

#include "brownian_tree.hpp"

#include <torch/torch.h>

#include "utils.hpp"

BrownianTree::BrownianTree(double t0, torch::Tensor w0, double t1,
                           torch::Tensor w1, int entropy, double tol,
                           int cache_depth, double safety) {
  this->t0 = t0;
  this->w0 = w0;
  this->t1 = t1;
  this->w1 = w1;
  this->entropy = entropy;
  this->tol = tol;
  this->cache_depth = cache_depth;
  this->safety = safety;

  auto t00 = t0 - safety;
  auto w00 = w0 + torch::randn_like(w0) * sqrt(safety);
  this->prev_cache.insert(std::pair<double, torch::Tensor>(t0, w0));
  this->prev_cache.insert(std::pair<double, torch::Tensor>(t00, w00));

  auto t11 = t1 + safety;
  auto w11 = w1 + torch::randn_like(w1) * sqrt(safety);
  this->post_cache.insert(std::pair<double, torch::Tensor>(t1, w1));
  this->post_cache.insert(std::pair<double, torch::Tensor>(t11, w11));
}

torch::Tensor BrownianTree::call(double t) {
  if (t <= t0) {  // Preceed boundary.
    auto begin = prev_cache.begin();
    if (t < begin->first) {
      auto w00 = begin->second;
      auto w = w00 + torch::randn_like(w00) * std::sqrt(begin->first - t);
      prev_cache.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    } else if (prev_cache.find(t) != prev_cache.end()) {
      return prev_cache.at(t);
    } else {
      auto lo = prev_cache.lower_bound(t);
      auto hi = lo--;
      auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
      prev_cache.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    }
  } else if (t >= t1) {  // Exceed boundary.
    auto end = post_cache.rbegin();
    if (t > end->first) {
      auto w11 = end->second;
      auto w = w11 + torch::randn_like(w11) * std::sqrt(t - end->first);
      post_cache.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    } else if (post_cache.find(t) != post_cache.end()) {
      return post_cache.at(t);
    } else {
      auto lo = post_cache.lower_bound(t);
      auto hi = lo--;
      auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
      post_cache.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    }
  } else {
    return binary_search_with_seed(t, t0, t1, w0, w1, entropy, tol);
  }
}

std::string BrownianTree::repr() const {
  return "BrownianTree(t0=" + format_float(t0, 3) + ", " +
         "t1=" + format_float(t1, 3) + ", " +
         "entropy=" + std::to_string(entropy) + ", " +
         "tol=" + std::to_string(tol) + ")";
}

double BrownianTree::get_t0() const { return t0; }

double BrownianTree::get_t1() const { return t1; }

torch::Tensor BrownianTree::get_w0() const { return w0; }

torch::Tensor BrownianTree::get_w1() const { return w1; }
