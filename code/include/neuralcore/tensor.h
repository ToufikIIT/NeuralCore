#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <random>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <initializer_list>

namespace neuralcore {

// ============================================================================
// Storage: ref-counted contiguous memory block
// ============================================================================
class Storage {
public:
    Storage() : data_(nullptr), size_(0) {}

    explicit Storage(size_t size)
        : data_(new float[size], std::default_delete<float[]>()), size_(size) {}

    Storage(size_t size, float value)
        : data_(new float[size], std::default_delete<float[]>()), size_(size) {
        std::fill(data_.get(), data_.get() + size, value);
    }

    Storage(const float* src, size_t size)
        : data_(new float[size], std::default_delete<float[]>()), size_(size) {
        std::memcpy(data_.get(), src, size * sizeof(float));
    }

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }

    Storage clone() const {
        Storage s(size_);
        if (size_ > 0) {
            std::memcpy(s.data(), data_.get(), size_ * sizeof(float));
        }
        return s;
    }

private:
    std::shared_ptr<float> data_;
    size_t size_;
};

// ============================================================================
// Tensor: N-dimensional array with broadcasting
// ============================================================================
class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, float value);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    Tensor(Storage storage, const std::vector<int>& shape,
           const std::vector<int>& strides, int offset = 0);

    // Properties
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    int size() const { return numel_; }
    int size(int dim) const;
    bool empty() const { return numel_ == 0; }

    float* data() { return storage_.data() + offset_; }
    const float* data() const { return storage_.data() + offset_; }
    Storage& storage() { return storage_; }
    const Storage& storage() const { return storage_; }
    int offset() const { return offset_; }

    // Element access
    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;

    template <typename... Ints>
    float& at(Ints... indices) {
        return operator()(std::vector<int>{static_cast<int>(indices)...});
    }
    template <typename... Ints>
    const float& at(Ints... indices) const {
        return operator()(std::vector<int>{static_cast<int>(indices)...});
    }

    float item() const;

    // Factory functions
    static Tensor zeros(const std::vector<int>& shape);
    static Tensor ones(const std::vector<int>& shape);
    static Tensor full(const std::vector<int>& shape, float value);
    static Tensor rand(const std::vector<int>& shape);
    static Tensor randn(const std::vector<int>& shape);
    static Tensor arange(float start, float end, float step = 1.0f);
    static Tensor eye(int n);
    static Tensor from_vector(const std::vector<float>& data,
                              const std::vector<int>& shape);

    // Copy
    Tensor clone() const;
    Tensor contiguous() const;
    bool is_contiguous() const;

    // Shape manipulation
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor t() const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor flatten(int start_dim = 0, int end_dim = -1) const;
    Tensor expand(const std::vector<int>& new_shape) const;

    // Elementwise binary ops
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // In-place ops
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    Tensor& operator+=(float scalar);
    Tensor& operator-=(float scalar);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);

    // Math functions
    Tensor exp() const;
    Tensor log() const;
    Tensor pow(float exponent) const;
    Tensor sqrt() const;
    Tensor abs() const;
    Tensor clamp(float min_val, float max_val) const;

    // Reduction ops
    Tensor sum(int dim = -1, bool keepdim = false) const;
    Tensor mean(int dim = -1, bool keepdim = false) const;
    Tensor max(int dim = -1, bool keepdim = false) const;
    Tensor min(int dim = -1, bool keepdim = false) const;
    Tensor argmax(int dim = -1) const;

    float sum_all() const;
    float mean_all() const;
    float max_all() const;
    float min_all() const;

    // Matrix ops
    Tensor matmul(const Tensor& other) const;

    // Comparison
    Tensor operator>(float val) const;
    Tensor operator<(float val) const;
    Tensor operator>=(float val) const;
    Tensor operator<=(float val) const;
    Tensor operator==(float val) const;

    // In-place utility
    void fill_(float value);
    void zero_();
    void uniform_(float low = 0.0f, float high = 1.0f);
    void normal_(float mean = 0.0f, float std = 1.0f);

    // Print
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
    std::string to_string() const;

    // Friends for scalar-tensor ops
    friend Tensor operator+(float scalar, const Tensor& t);
    friend Tensor operator-(float scalar, const Tensor& t);
    friend Tensor operator*(float scalar, const Tensor& t);
    friend Tensor operator/(float scalar, const Tensor& t);

private:
    Storage storage_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int offset_;
    int numel_;

    void compute_strides();
    int compute_flat_index(const std::vector<int>& indices) const;
    static std::vector<int> broadcast_shapes(const std::vector<int>& a,
                                              const std::vector<int>& b);
    Tensor binary_op(const Tensor& other,
                     std::function<float(float, float)> op) const;
    Tensor unary_op(std::function<float(float)> op) const;
    Tensor reduce_dim(int dim, bool keepdim,
                      std::function<float(float, float)> op,
                      float init) const;
    static std::vector<int> unravel_index(int flat_idx,
                                           const std::vector<int>& shape);
    static std::mt19937& rng();

    // Get element value using broadcasting logic
    float broadcast_get(const std::vector<int>& idx) const;
};

Tensor operator+(float scalar, const Tensor& t);
Tensor operator-(float scalar, const Tensor& t);
Tensor operator*(float scalar, const Tensor& t);
Tensor operator/(float scalar, const Tensor& t);

} // namespace neuralcore
