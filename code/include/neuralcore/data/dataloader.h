#pragma once

#include "dataset.h"
#include <vector>
#include <random>
#include <algorithm>

namespace neuralcore {
namespace data {

class DataLoader {
public:
    DataLoader(const Dataset& dataset, int batch_size = 1,
               bool shuffle = false);

    struct Batch {
        Tensor data;
        Tensor target;
    };

    // Iterator-like interface
    void reset();
    bool has_next() const;
    Batch next();

    int num_batches() const;

private:
    const Dataset& dataset_;
    int batch_size_;
    bool shuffle_;
    std::vector<int> indices_;
    int current_idx_;
};

} // namespace data
} // namespace neuralcore
