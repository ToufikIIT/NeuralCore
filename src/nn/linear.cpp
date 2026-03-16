#include "neuralcore/nn/linear.h"

namespace neuralcore {
namespace nn {

Linear::Linear(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    // Xavier/Glorot uniform initialization
    weight = Variable::create(Tensor({out_features, in_features}), true);
    reset_parameters();
    register_parameter(weight);

    if (use_bias_) {
        bias = Variable::create(Tensor::zeros({1, out_features}), true);
        register_parameter(bias);
    }
}

void Linear::reset_parameters() {
    float limit = std::sqrt(6.0f / (in_features_ + out_features_));
    weight->data.uniform_(-limit, limit);
}

VariablePtr Linear::forward(const VariablePtr& input) {
    // input: (batch, in_features), weight: (out_features, in_features)
    // output = input @ weight^T + bias
    auto wt = weight->t();
    auto out = input->matmul(wt);
    if (use_bias_ && bias) {
        out = *out + bias;
    }
    return out;
}

} // namespace nn
} // namespace neuralcore
