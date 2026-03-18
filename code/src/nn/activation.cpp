#include "neuralcore/nn/activation.h"

namespace neuralcore {
namespace nn {

VariablePtr ReLU::forward(const VariablePtr& input) {
    return input->relu();
}

VariablePtr Sigmoid::forward(const VariablePtr& input) {
    return input->sigmoid();
}

VariablePtr Tanh::forward(const VariablePtr& input) {
    return input->tanh();
}

VariablePtr Softmax::forward(const VariablePtr& input) {
    int d = dim_;
    if (d < 0) d += input->ndim();

    // Numerically stable softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    // For simplicity, implement along last dim for 2D
    auto max_val = Variable::create(input->data.max(d, true), false);
    auto shifted = *input - max_val;
    auto exp_vals = shifted->exp();
    auto sum_exp = exp_vals->sum(d, true);
    return *exp_vals / sum_exp;
}

} // namespace nn
} // namespace neuralcore
