#pragma once

#include "module.h"

namespace neuralcore {
namespace nn {

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5f) : p_(p) {}
    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "Dropout"; }
private:
    float p_;
};

} // namespace nn
} // namespace neuralcore
