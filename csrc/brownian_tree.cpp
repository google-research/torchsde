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

#include <math.h>
#include <torch/torch.h>

#include "utils.hpp"

BrownianTree::BrownianTree(double t0, torch::Tensor w0, double t1, int entropy,
                           double tol, int cache_depth, double safety) {
  // Main cache using O(2^cache_depth) memory.
  seeds = std::vector<std::uint64_t>(std::pow(2, cache_depth));
  populate_cache(t0, w0, t1, entropy, cache_depth, cache, seeds);

  // Head cache.
  auto t00 = t0 - safety;
  auto w00 = w0 + torch::randn_like(w0) * sqrt(safety);
  cache_prev.insert(std::pair<double, torch::Tensor>(t0, w0));
  cache_prev.insert(std::pair<double, torch::Tensor>(t00, w00));

  // Tail cache.
  auto t11 = t1 + safety;
  auto w1 = cache.rbegin()->second;
  auto w11 = w1 + torch::randn_like(w1) * sqrt(safety);
  cache_post.insert(std::pair<double, torch::Tensor>(t1, w1));
  cache_post.insert(std::pair<double, torch::Tensor>(t11, w11));

  this->entropy = entropy;
  this->tol = tol;
  this->cache_depth = cache_depth;
  this->safety = safety;
}

BrownianTree::BrownianTree(double t0, torch::Tensor w0, double t1,
                           torch::Tensor w1, int entropy, double tol,
                           int cache_depth, double safety) {
  // Main cache.
  seeds = std::vector<std::uint64_t>(1, entropy);
  cache.insert(std::pair<double, torch::Tensor>(t0, w0));
  cache.insert(std::pair<double, torch::Tensor>(t1, w1));

  // Head cache.
  auto t00 = t0 - safety;
  auto w00 = w0 + torch::randn_like(w0) * sqrt(safety);
  cache_prev.insert(std::pair<double, torch::Tensor>(t0, w0));
  cache_prev.insert(std::pair<double, torch::Tensor>(t00, w00));

  // Tail cache.
  auto t11 = t1 + safety;
  auto w11 = w1 + torch::randn_like(w1) * sqrt(safety);
  cache_post.insert(std::pair<double, torch::Tensor>(t1, w1));
  cache_post.insert(std::pair<double, torch::Tensor>(t11, w11));

  this->entropy = entropy;
  this->tol = tol;
  this->cache_depth = cache_depth;
  this->safety = safety;
}

BrownianTree::BrownianTree(int entropy, double tol, double cache_depth,
                           double safety, std::map<double, torch::Tensor> cache,
                           std::map<double, torch::Tensor> cache_prev,
                           std::map<double, torch::Tensor> cache_post,
                           std::vector<std::uint64_t> seeds) {
  this->cache = cache;
  this->cache_prev = cache_prev;
  this->cache_post = cache_post;
  this->seeds = seeds;

  this->entropy = entropy;
  this->tol = tol;
  this->cache_depth = cache_depth;
  this->safety = safety;
}

torch::Tensor BrownianTree::call(double t) {
  if (t <= cache.begin()->first) {  // Preceed boundary.
    auto begin = cache_prev.begin();
    if (t < begin->first) {
      auto w00 = begin->second;
      auto w = w00 + torch::randn_like(w00) * std::sqrt(begin->first - t);
      cache_prev.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    } else if (cache_prev.find(t) != cache_prev.end()) {
      return cache_prev.at(t);
    } else {
      auto lo = cache_prev.lower_bound(t);
      auto hi = lo--;
      auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
      cache_prev.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    }
  } else if (t >= cache.rbegin()->first) {  // Exceed boundary.
    auto end = cache_post.rbegin();
    if (t > end->first) {
      auto w11 = end->second;
      auto w = w11 + torch::randn_like(w11) * std::sqrt(t - end->first);
      cache_post.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    } else if (cache_post.find(t) != cache_post.end()) {
      return cache_post.at(t);
    } else {
      auto lo = cache_post.lower_bound(t);
      auto hi = lo--;
      auto w = brownian_bridge(t, lo->first, hi->first, lo->second, hi->second);
      cache_post.insert(std::pair<double, torch::Tensor>(t, w));
      return w;
    }
  } else if (cache.find(t) != cache.end()) {  // t in main cache.
    return cache.at(t);
  } else {  // t in range of main cache, but not in it.
    auto lo = cache.lower_bound(t);
    auto hi = lo--;

    auto t0 = lo->first;
    auto t1 = hi->first;
    auto w0 = lo->second;
    auto w1 = hi->second;
    auto dt = (t1 - t0) / std::pow(2, cache_depth);
    auto seed = seeds[floor((t - t0) / dt)];
    return binary_search_with_seed(t, t0, t1, w0, w1, seed, tol);
  }
}

std::string BrownianTree::repr() const {
  auto t0 = cache.begin()->first;
  auto t1 = cache.rbegin()->first;
  return "BrownianTree(t0=" + format_float(t0, 3) + ", " +
         "t1=" + format_float(t1, 3) + ", " +
         "entropy=" + std::to_string(entropy) + ", " +
         "tol=" + std::to_string(tol) + ")";
}

std::vector<std::map<double, torch::Tensor>> BrownianTree::get_cache() const {
  return std::vector<std::map<double, torch::Tensor>>(
      {cache, cache_prev, cache_post});
}

std::vector<std::uint64_t> BrownianTree::get_seeds() const { return seeds; }

double BrownianTree::get_t0() const { return cache.begin()->first; }

double BrownianTree::get_t1() const { return cache.rbegin()->first; }

torch::Tensor BrownianTree::get_w0() const { return cache.begin()->second; }

torch::Tensor BrownianTree::get_w1() const { return cache.rbegin()->second; }
