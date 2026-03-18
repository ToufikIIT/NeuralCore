#pragma once

#include "autograd.h"
#include "nn/module.h"
#include <string>
#include <vector>
#include <fstream>

namespace neuralcore {

void save_parameters(const std::string& path,
                     const std::vector<VariablePtr>& params);

void load_parameters(const std::string& path,
                     std::vector<VariablePtr>& params);

} // namespace neuralcore
