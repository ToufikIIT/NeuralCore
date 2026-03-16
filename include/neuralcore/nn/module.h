#pragma once

#include "../autograd.h"
#include <vector>
#include <string>

namespace neuralcore {
namespace nn {

// ============================================================================
// Module: base class for all neural network modules
// ============================================================================
class Module {
public:
    virtual ~Module() = default;
    virtual VariablePtr forward(const VariablePtr& input) = 0;
    virtual std::vector<VariablePtr> parameters();
    virtual std::string name() const { return "Module"; }

    void train() { training_ = true; }
    void eval() { training_ = false; }
    bool is_training() const { return training_; }

    void zero_grad();

protected:
    bool training_ = true;
    std::vector<VariablePtr> params_;
    std::vector<std::shared_ptr<Module>> submodules_;

    void register_parameter(const VariablePtr& param);
    void register_module(const std::shared_ptr<Module>& module);
};

} // namespace nn
} // namespace neuralcore
