#pragma once

#include "optimizer.h"

namespace neuralcore {
namespace optim {

class RMSProp : public Optimizer {
public:
    RMSProp(const std::vector<VariablePtr>& params, float lr = 0.01f,
            float alpha = 0.99f, float eps = 1e-8f,
            float weight_decay = 0.0f);

    void step() override;

private:
    float alpha_, eps_, weight_decay_;
    std::vector<Tensor> v_; // running average of squared gradients
    bool initialized_ = false;
};

} // namespace optim
} // namespace neuralcore
