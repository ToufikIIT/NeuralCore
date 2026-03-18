#include "neuralcore/nn/loss.h"
#include <cmath>

namespace neuralcore {
namespace nn {

// ============================================================================
// MSE Loss: mean((pred - target)^2)
// ============================================================================
VariablePtr MSELoss::forward(const VariablePtr& prediction, const VariablePtr& target) {
    auto diff = *prediction - target;
    auto sq = diff->pow(2.0f);
    return sq->mean();
}

// ============================================================================
// BCE Loss: -mean(target * log(pred) + (1 - target) * log(1 - pred))
// ============================================================================
VariablePtr BCELoss::forward(const VariablePtr& prediction, const VariablePtr& target) {
    float eps = 1e-7f;
    auto pred_clamped = prediction->clamp(eps, 1.0f - eps);
    auto log_pred = pred_clamped->log();
    auto one_minus_pred = Variable::create(Tensor::ones(prediction->data.shape()), false);
    one_minus_pred = *one_minus_pred - pred_clamped;
    auto log_one_minus = one_minus_pred->log();

    auto one_minus_target = Variable::create(Tensor::ones(target->data.shape()), false);
    one_minus_target = *one_minus_target - target;

    // target * log(pred) + (1 - target) * log(1 - pred)
    auto term1 = *target * log_pred;
    auto term2 = *one_minus_target * log_one_minus;
    auto loss = *term1 + term2;
    auto neg_loss = -(*loss);
    return neg_loss->mean();
}

// ============================================================================
// Cross Entropy Loss: -sum(target_one_hot * log_softmax(logits)) / batch_size
// Expects logits (raw, unnormalized) and target as class indices (float tensor)
// ============================================================================
VariablePtr CrossEntropyLoss::forward(const VariablePtr& logits, const VariablePtr& target) {
    // Log-softmax along dim 1
    int batch_size = logits->data.shape()[0];
    int num_classes = logits->data.shape()[1];

    // Stable log-softmax
    auto max_val = Variable::create(logits->data.max(1, true), false);
    auto shifted = *logits - max_val;
    auto exp_vals = shifted->exp();
    auto sum_exp = exp_vals->sum(1, true);
    auto log_sum_exp = sum_exp->log();
    auto log_softmax = *shifted - log_sum_exp;

    // Gather the log probabilities at target indices
    Tensor nll_vals({batch_size, 1});
    for (int i = 0; i < batch_size; ++i) {
        int cls = static_cast<int>(target->data.at(i));
        nll_vals.at(i, 0) = log_softmax->data.at(i, cls);
    }

    auto nll = Variable::create(nll_vals, false);
    // This part isn't differentiable through gather, so we build it manually
    // using the full log_softmax and a one-hot encoding
    Tensor one_hot = Tensor::zeros({batch_size, num_classes});
    for (int i = 0; i < batch_size; ++i) {
        int cls = static_cast<int>(target->data.at(i));
        one_hot.at(i, cls) = 1.0f;
    }
    auto one_hot_var = Variable::create(one_hot, false);
    auto selected = *log_softmax * one_hot_var;
    auto loss = selected->sum();
    auto neg_loss = -(*loss);
    return *neg_loss / static_cast<float>(batch_size);
}

} // namespace nn
} // namespace neuralcore
