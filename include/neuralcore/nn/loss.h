#pragma once

#include "../autograd.h"

namespace neuralcore {
namespace nn {

// ============================================================================
// Loss functions
// ============================================================================

class MSELoss {
public:
    VariablePtr forward(const VariablePtr& prediction, const VariablePtr& target);
    VariablePtr operator()(const VariablePtr& prediction, const VariablePtr& target) {
        return forward(prediction, target);
    }
};

class BCELoss {
public:
    VariablePtr forward(const VariablePtr& prediction, const VariablePtr& target);
    VariablePtr operator()(const VariablePtr& prediction, const VariablePtr& target) {
        return forward(prediction, target);
    }
};

class CrossEntropyLoss {
public:
    VariablePtr forward(const VariablePtr& logits, const VariablePtr& target);
    VariablePtr operator()(const VariablePtr& logits, const VariablePtr& target) {
        return forward(logits, target);
    }
};

} // namespace nn
} // namespace neuralcore
