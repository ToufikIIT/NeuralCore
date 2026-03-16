#pragma once

#include "optimizer.h"

namespace neuralcore {
namespace optim {

class SGD : public Optimizer {
public:
    SGD(const std::vector<VariablePtr>& params, float lr = 0.01f,
        float momentum = 0.0f, float weight_decay = 0.0f);

    void step() override;

private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocity_;
    bool initialized_ = false;
};

} // namespace optim
} // namespace neuralcore
