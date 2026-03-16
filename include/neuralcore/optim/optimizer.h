#pragma once

#include "../autograd.h"
#include <vector>

namespace neuralcore {
namespace optim {

class Optimizer {
public:
    explicit Optimizer(const std::vector<VariablePtr>& params, float lr = 0.01f)
        : params_(params), lr_(lr) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;

    void zero_grad() {
        for (auto& p : params_) {
            p->zero_grad();
        }
    }

    float learning_rate() const { return lr_; }
    void set_learning_rate(float lr) { lr_ = lr; }

protected:
    std::vector<VariablePtr> params_;
    float lr_;
};

} // namespace optim
} // namespace neuralcore
