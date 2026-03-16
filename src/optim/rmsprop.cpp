#include "neuralcore/optim/rmsprop.h"
#include <cmath>

namespace neuralcore {
namespace optim {

RMSProp::RMSProp(const std::vector<VariablePtr>& params, float lr,
                 float alpha, float eps, float weight_decay)
    : Optimizer(params, lr), alpha_(alpha), eps_(eps),
      weight_decay_(weight_decay) {}

void RMSProp::step() {
    if (!initialized_) {
        v_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
            v_[i] = Tensor::zeros(params_[i]->data.shape());
        }
        initialized_ = true;
    }

    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i]->grad.empty()) continue;

        Tensor grad = params_[i]->grad;

        if (weight_decay_ != 0.0f) {
            grad = grad + params_[i]->data * weight_decay_;
        }

        // v = alpha * v + (1 - alpha) * grad^2
        v_[i] = v_[i] * alpha_ + (grad * grad) * (1.0f - alpha_);

        // param -= lr * grad / (sqrt(v) + eps)
        params_[i]->data -= (grad / (v_[i].sqrt() + eps_)) * lr_;
    }
}

} // namespace optim
} // namespace neuralcore
