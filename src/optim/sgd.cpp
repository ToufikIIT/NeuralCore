#include "neuralcore/optim/sgd.h"

namespace neuralcore {
namespace optim {

SGD::SGD(const std::vector<VariablePtr>& params, float lr,
         float momentum, float weight_decay)
    : Optimizer(params, lr), momentum_(momentum), weight_decay_(weight_decay) {}

void SGD::step() {
    if (!initialized_) {
        velocity_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
            velocity_[i] = Tensor::zeros(params_[i]->data.shape());
        }
        initialized_ = true;
    }

    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i]->grad.empty()) continue;

        Tensor grad = params_[i]->grad;

        // Weight decay
        if (weight_decay_ != 0.0f) {
            grad = grad + params_[i]->data * weight_decay_;
        }

        // Momentum
        if (momentum_ != 0.0f) {
            velocity_[i] = velocity_[i] * momentum_ + grad;
            params_[i]->data -= velocity_[i] * lr_;
        } else {
            params_[i]->data -= grad * lr_;
        }
    }
}

} // namespace optim
} // namespace neuralcore
