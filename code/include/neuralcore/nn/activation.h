#pragma once

#include "module.h"

namespace neuralcore {
namespace nn {

class ReLU : public Module {
public:
    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Module {
public:
    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public Module {
public:
    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "Tanh"; }
};

class Softmax : public Module {
public:
    explicit Softmax(int dim = -1) : dim_(dim) {}
    VariablePtr forward(const VariablePtr& input) override;
    std::string name() const override { return "Softmax"; }
private:
    int dim_;
};

} // namespace nn
} // namespace neuralcore
