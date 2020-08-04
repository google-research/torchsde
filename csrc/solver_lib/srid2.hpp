#ifndef SRK_DIAGONAL_HPP
#define SRK_DIAGONAL_HPP

#define srid2_STAGES 4

#include <torch/torch.h>

extern double A0[srid2_STAGES][srid2_STAGES];
extern double A1[srid2_STAGES][srid2_STAGES];
extern double B0[srid2_STAGES][srid2_STAGES];
extern double B1[srid2_STAGES][srid2_STAGES];
extern double C0[srid2_STAGES];
extern double C1[srid2_STAGES];

extern double alpha[srid2_STAGES];
extern double beta1[srid2_STAGES];
extern double beta2[srid2_STAGES];
extern double beta3[srid2_STAGES];
extern double beta4[srid2_STAGES];

std::vector<torch::Tensor> srid2_step(
    std::function<std::vector<torch::Tensor>(double, std::vector<torch::Tensor>)> const &f,
    std::function<std::vector<torch::Tensor>(double, std::vector<torch::Tensor>)> const &g,
    double t0,
    double dt,
    double sqrt_dt,
    std::vector<torch::Tensor> const &y0,
    std::vector<torch::Tensor> const &I_k,
    std::vector<torch::Tensor> const &I_kk,
    std::vector<torch::Tensor> const &I_k0,
    std::vector<torch::Tensor> const &I_kkk);
#endif
