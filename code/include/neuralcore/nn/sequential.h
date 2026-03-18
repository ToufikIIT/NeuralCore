#pragma once

#include "module.h"
#include <vector>
#include <initializer_list>

namespace neuralcore {
namespace nn {

class Sequential : public Module {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Module>> modules);

    void add(const std::shared_ptr<Module>& module);

    VariablePtr forward(const VariablePtr& input) override;
    std::vector<VariablePtr> parameters() override;
    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
} // namespace neuralcore
