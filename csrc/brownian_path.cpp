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

#include <math.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>
#include <iterator>
#include <map>

#include "utils.hpp"

namespace py = pybind11;

class BrownianPath {
 private:
  std::map<float, torch::Tensor> cache;
  float t_head;
  float t_tail;
  torch::Tensor w_head;
  torch::Tensor w_tail;
  BrownianPath() {}

 public:
  static BrownianPath construct_from_pair(float t0, torch::Tensor w0) {
    BrownianPath bp = BrownianPath();
    bp.cache.insert(std::pair<float, torch::Tensor>(t0, w0));

    bp.t_head = t0;
    bp.t_tail = t0;
    bp.w_head = w0;
    bp.w_tail = w0;
    return bp;
  }

  static BrownianPath construct_from_dict(std::map<float, torch::Tensor> data) {
    BrownianPath bp = BrownianPath();
    for (auto const& x : data) {
      bp.cache.insert(std::pair<float, torch::Tensor>(x.first, x.second));
    }

    auto head = data.begin();
    auto tail = data.rbegin();
    bp.t_head = head->first;
    bp.t_tail = tail->first;
    bp.w_head = head->second;
    bp.w_tail = tail->second;
    return bp;
  }

  torch::Tensor call(float t) {
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

  void insert(float t, torch::Tensor w) {
    cache.insert(std::pair<float, torch::Tensor>(t, w));

    if (t < t_head) {
      t_head = t;
      w_head = w;
    } else if (t > t_tail) {
      t_tail = t;
      w_tail = w;
    }
  }

  auto repr() const {
    return "BrownianPath(t0=" + format_float(t_head, 3) +
           ", t1=" + format_float(t_tail, 3) + ")";
  }

  std::map<float, torch::Tensor> get_cache() const { return cache; }

  float get_t_head() const { return t_head; }

  float get_t_tail() const { return t_tail; }

  torch::Tensor get_w_head() const { return w_head; }

  torch::Tensor get_w_tail() const { return w_tail; }
};

PYBIND11_MODULE(_brownian_lib, m) {
  m.doc() = "Fast Brownian motion based on PyTorch C++ API.";
  py::class_<BrownianPath>(m, "BrownianPath")
      .def_static("construct_from_pair", &BrownianPath::construct_from_pair,
                  py::arg("t0"), py::arg("w0"))
      .def_static("construct_from_dict", &BrownianPath::construct_from_dict,
                  py::arg("data"))
      .def("__call__", &BrownianPath::call, py::arg("t"))
      .def("__repr__", &BrownianPath::repr)
      .def("insert", &BrownianPath::insert)
      .def("get_cache", &BrownianPath::get_cache)
      .def("get_t_head", &BrownianPath::get_t_head)
      .def("get_t_tail", &BrownianPath::get_t_tail)
      .def("get_w_head", &BrownianPath::get_w_head)
      .def("get_w_tail", &BrownianPath::get_w_tail);
}
