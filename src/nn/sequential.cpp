#include "neuralcore/nn/sequential.h"

namespace neuralcore {
namespace nn {

Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    for (auto& m : modules) {
        add(m);
    }
}

void Sequential::add(const std::shared_ptr<Module>& module) {
    modules_.push_back(module);
    register_module(module);
}

VariablePtr Sequential::forward(const VariablePtr& input) {
    VariablePtr x = input;
    for (auto& m : modules_) {
        x = m->forward(x);
    }
    return x;
}

std::vector<VariablePtr> Sequential::parameters() {
    std::vector<VariablePtr> all;
    for (auto& m : modules_) {
        auto p = m->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

} // namespace nn
} // namespace neuralcore
