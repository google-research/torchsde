#include <torch/torch.h>

torch::Tensor brownian_bridge(float t, float t0, float t1, torch::Tensor w0,
                              torch::Tensor w1);

std::string format_float(float t, int precision = 3);
