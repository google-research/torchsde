#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "brownian_path.hpp"
#include "brownian_tree.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_brownian_lib, m) {
  m.doc() = "Fast Brownian motion based on PyTorch C++ API.";
  py::class_<BrownianPath>(m, "BrownianPath")
      .def(py::init([](float t0, torch::Tensor w0) {
             return new BrownianPath(t0, w0);
           }),
           py::arg("t0"), py::arg("w0"))
      .def(py::init([](std::map<float, torch::Tensor> data) {
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
      .def(
          py::init([](double t0, torch::Tensor w0, double t1, torch::Tensor w1,
                      int entropy, double tol, int cache_depth, double safety) {
            return new BrownianTree(t0, w0, t1, w1, entropy, tol, cache_depth,
                                    safety);
          }))
      .def("__call__", &BrownianTree::call, py::arg("t"))
      .def("__repr__", &BrownianTree::repr)
      .def("get_t0", &BrownianTree::get_t0)
      .def("get_t1", &BrownianTree::get_t1)
      .def("get_w0", &BrownianTree::get_w0)
      .def("get_w1", &BrownianTree::get_w1);
}
