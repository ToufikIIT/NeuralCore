#include "neuralcore/data/dataloader.h"

namespace neuralcore {
namespace data {

DataLoader::DataLoader(const Dataset& dataset, int batch_size, bool shuffle)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle),
      current_idx_(0) {
    indices_.resize(dataset_.size());
    for (int i = 0; i < dataset_.size(); ++i) {
        indices_[i] = i;
    }
}

void DataLoader::reset() {
    current_idx_ = 0;
    if (shuffle_) {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::shuffle(indices_.begin(), indices_.end(), gen);
    }
}

bool DataLoader::has_next() const {
    return current_idx_ < static_cast<int>(indices_.size());
}

DataLoader::Batch DataLoader::next() {
    int end = std::min(current_idx_ + batch_size_,
                       static_cast<int>(indices_.size()));
    int actual_batch = end - current_idx_;

    // Get first sample to determine shapes
    auto [first_data, first_target] = dataset_.get(indices_[current_idx_]);

    // Build batch tensors
    std::vector<int> data_shape = {actual_batch};
    for (int s : first_data.shape()) data_shape.push_back(s);

    std::vector<int> target_shape = {actual_batch};
    for (int s : first_target.shape()) target_shape.push_back(s);

    Tensor batch_data(data_shape);
    Tensor batch_target(target_shape);

    int data_stride = first_data.size();
    int target_stride = first_target.size();

    for (int i = 0; i < actual_batch; ++i) {
        auto [d, t] = dataset_.get(indices_[current_idx_ + i]);
        std::memcpy(batch_data.data() + i * data_stride,
                    d.data(), data_stride * sizeof(float));
        std::memcpy(batch_target.data() + i * target_stride,
                    t.data(), target_stride * sizeof(float));
    }

    current_idx_ = end;
    return {batch_data, batch_target};
}

int DataLoader::num_batches() const {
    return (dataset_.size() + batch_size_ - 1) / batch_size_;
}

} // namespace data
} // namespace neuralcore
