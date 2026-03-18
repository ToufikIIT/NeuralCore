#pragma once

#include "optimizer.h"

namespace neuralcore {
namespace optim {

class Adam : public Optimizer {
public:
    Adam(const std::vector<VariablePtr>& params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f);

    void step() override;

private:
    float beta1_, beta2_, eps_, weight_decay_;
    std::vector<Tensor> m_; // first moment
    std::vector<Tensor> v_; // second moment
    int t_ = 0;
    bool initialized_ = false;
};

} // namespace optim
} // namespace neuralcore
