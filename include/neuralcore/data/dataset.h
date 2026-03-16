#pragma once

#include "../tensor.h"
#include <vector>
#include <utility>

namespace neuralcore {
namespace data {

class Dataset {
public:
    virtual ~Dataset() = default;
    virtual std::pair<Tensor, Tensor> get(int index) const = 0;
    virtual int size() const = 0;
};

class TensorDataset : public Dataset {
public:
    TensorDataset(const Tensor& data, const Tensor& targets)
        : data_(data), targets_(targets) {}

    std::pair<Tensor, Tensor> get(int index) const override;
    int size() const override { return data_.shape()[0]; }

private:
    Tensor data_;
    Tensor targets_;
};

} // namespace data
} // namespace neuralcore
