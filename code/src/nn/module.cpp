#include "neuralcore/nn/module.h"

namespace neuralcore {
namespace nn {

std::vector<VariablePtr> Module::parameters() {
    std::vector<VariablePtr> all_params = params_;
    for (auto& sub : submodules_) {
        auto sub_params = sub->parameters();
        all_params.insert(all_params.end(), sub_params.begin(), sub_params.end());
    }
    return all_params;
}

void Module::zero_grad() {
    for (auto& p : parameters()) {
        p->zero_grad();
    }
}

void Module::register_parameter(const VariablePtr& param) {
    params_.push_back(param);
}

void Module::register_module(const std::shared_ptr<Module>& module) {
    submodules_.push_back(module);
}

} // namespace nn
} // namespace neuralcore
