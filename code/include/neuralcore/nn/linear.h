#pragma once

#include "module.h"
#include <cmath>

namespace neuralcore {
namespace nn {

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool use_bias = true);

    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "Linear"; }

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

    VariablePtr weight;
    VariablePtr bias;

private:
    int in_features_;
    int out_features_;
    bool use_bias_;

    void reset_parameters();
};

} // namespace nn
} // namespace neuralcore
