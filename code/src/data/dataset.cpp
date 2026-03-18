#include "neuralcore/data/dataset.h"

namespace neuralcore {
namespace data {

std::pair<Tensor, Tensor> TensorDataset::get(int index) const {
    // Extract row at index
    int data_cols = 1;
    for (int i = 1; i < data_.ndim(); ++i) data_cols *= data_.shape()[i];

    int target_cols = 1;
    for (int i = 1; i < targets_.ndim(); ++i) target_cols *= targets_.shape()[i];

    std::vector<float> d(data_cols);
    std::vector<float> t(target_cols);

    for (int j = 0; j < data_cols; ++j) {
        d[j] = data_.data()[index * data_cols + j];
    }
    for (int j = 0; j < target_cols; ++j) {
        t[j] = targets_.data()[index * target_cols + j];
    }

    std::vector<int> d_shape(data_.shape().begin() + 1, data_.shape().end());
    std::vector<int> t_shape(targets_.shape().begin() + 1, targets_.shape().end());
    if (d_shape.empty()) d_shape = {1};
    if (t_shape.empty()) t_shape = {1};

    return {Tensor(d_shape, d), Tensor(t_shape, t)};
}

} // namespace data
} // namespace neuralcore
