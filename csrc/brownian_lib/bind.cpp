#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "brownian_path.hpp"
#include "brownian_tree.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_brownian_lib, m) {
  m.doc() = "Fast Brownian motion based on PyTorch C++ API.";

  // Binded constructors defaults to Python taking ownership of returned object.
  py::class_<BrownianPath>(m, "BrownianPath")
      // For initialization.
      .def(py::init([](double t0, torch::Tensor w0) {
             return new BrownianPath(t0, w0);
           }),
           py::arg("t0"), py::arg("w0"))
      // For device transfer.
      .def(py::init([](std::map<double, torch::Tensor> data) {
             return new BrownianPath(data);
           }),
           py::arg("data"))
      .def("__call__", &BrownianPath::call, py::arg("t"))
      .def("__repr__", &BrownianPath::repr)
      .def("insert", &BrownianPath::insert)
      .def("get_cache", &BrownianPath::get_cache)
      .def("get_t_head", &BrownianPath::get_t_head)
      .def("get_t_tail", &BrownianPath::get_t_tail)
      .def("get_w_head", &BrownianPath::get_w_head)
      .def("get_w_tail", &BrownianPath::get_w_tail);

  py::class_<BrownianTree>(m, "BrownianTree")
      // For initialization.
      .def(py::init([](double t0, torch::Tensor w0, double t1, int entropy,
                       double tol, int cache_depth, double safety) {
             return new BrownianTree(t0, w0, t1, entropy, tol, cache_depth,
                                     safety);
           }),
           py::arg("t0"), py::arg("w0"), py::arg("t1"), py::arg("entropy"),
           py::arg("tol"), py::arg("cache_depth"), py::arg("safety"))
      // For device transfer.
      .def(py::init([](int entropy, double tol, double cache_depth,
                       double safety, std::map<double, torch::Tensor> cache,
                       std::map<double, torch::Tensor> cache_prev,
                       std::map<double, torch::Tensor> cache_post,
                       std::vector<std::uint64_t> seeds) {
             return new BrownianTree(entropy, tol, cache_depth, safety, cache,
                                     cache_prev, cache_post, seeds);
           }),
           py::arg("entropy"), py::arg("tol"), py::arg("cache_depth"),
           py::arg("safety"), py::arg("cache"), py::arg("cache_prev"),
           py::arg("cache_post"), py::arg("seeds"))
      // For testing correctness.
      .def(
          py::init([](double t0, torch::Tensor w0, double t1, torch::Tensor w1,
                      int entropy, double tol, int cache_depth, double safety) {
            return new BrownianTree(t0, w0, t1, w1, entropy, tol, cache_depth,
                                    safety);
          }),
          py::arg("t0"), py::arg("w0"), py::arg("t1"), py::arg("w1"),
          py::arg("entropy"), py::arg("tol"), py::arg("cache_depth"),
          py::arg("safety"))
      .def("__call__", &BrownianTree::call, py::arg("t"))
      .def("__repr__", &BrownianTree::repr)
      .def("get_cache", &BrownianTree::get_cache)
      .def("get_seeds", &BrownianTree::get_seeds)
      .def("get_t0", &BrownianTree::get_t0)
      .def("get_t1", &BrownianTree::get_t1)
      .def("get_w0", &BrownianTree::get_w0)
      .def("get_w1", &BrownianTree::get_w1);
}
