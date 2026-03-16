#include "neuralcore/nn/dropout.h"
#include <random>

namespace neuralcore {
namespace nn {

VariablePtr Dropout::forward(const VariablePtr& input) {
    if (!is_training() || p_ == 0.0f) {
        return input;
    }

    // Generate dropout mask
    static thread_local std::mt19937 gen(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - static_cast<double>(p_));

    Tensor mask(input->data.shape());
    float scale = 1.0f / (1.0f - p_);
    for (int i = 0; i < mask.size(); ++i) {
        mask.data()[i] = dist(gen) ? scale : 0.0f;
    }

    auto mask_var = Variable::create(mask, false);
    return *input * mask_var;
}

} // namespace nn
} // namespace neuralcore
