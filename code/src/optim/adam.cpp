#include "neuralcore/optim/adam.h"
#include <cmath>

namespace neuralcore {
namespace optim {

Adam::Adam(const std::vector<VariablePtr>& params, float lr,
           float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(params, lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), weight_decay_(weight_decay) {}

void Adam::step() {
    if (!initialized_) {
        m_.resize(params_.size());
        v_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i) {
            m_[i] = Tensor::zeros(params_[i]->data.shape());
            v_[i] = Tensor::zeros(params_[i]->data.shape());
        }
        initialized_ = true;
    }

    t_++;

    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i]->grad.empty()) continue;

        Tensor grad = params_[i]->grad;

        if (weight_decay_ != 0.0f) {
            grad = grad + params_[i]->data * weight_decay_;
        }

        // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
        m_[i] = m_[i] * beta1_ + grad * (1.0f - beta1_);
        // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
        v_[i] = v_[i] * beta2_ + (grad * grad) * (1.0f - beta2_);

        // Bias correction
        float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
        float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
        Tensor m_hat = m_[i] / bc1;
        Tensor v_hat = v_[i] / bc2;

        // Update: param -= lr * m_hat / (sqrt(v_hat) + eps)
        params_[i]->data -= (m_hat / (v_hat.sqrt() + eps_)) * lr_;
    }
}

} // namespace optim
} // namespace neuralcore
