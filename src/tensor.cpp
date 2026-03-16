#include "neuralcore/tensor.h"

namespace neuralcore {

// ============================================================================
// Random engine
// ============================================================================
std::mt19937& Tensor::rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

// ============================================================================
// Constructors
// ============================================================================
Tensor::Tensor() : offset_(0), numel_(0) {}

Tensor::Tensor(const std::vector<int>& shape)
    : shape_(shape), offset_(0) {
    compute_strides();
    numel_ = shape_.empty() ? 0 :
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    storage_ = Storage(static_cast<size_t>(numel_), 0.0f);
}

Tensor::Tensor(const std::vector<int>& shape, float value)
    : shape_(shape), offset_(0) {
    compute_strides();
    numel_ = shape_.empty() ? 0 :
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    storage_ = Storage(static_cast<size_t>(numel_), value);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape), offset_(0) {
    compute_strides();
    numel_ = shape_.empty() ? 0 :
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    if (static_cast<int>(data.size()) != numel_) {
        throw std::runtime_error("Data size mismatch with shape");
    }
    storage_ = Storage(data.data(), static_cast<size_t>(numel_));
}

Tensor::Tensor(Storage storage, const std::vector<int>& shape,
               const std::vector<int>& strides, int offset)
    : storage_(storage), shape_(shape), strides_(strides), offset_(offset) {
    numel_ = shape_.empty() ? 0 :
        std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
}

// ============================================================================
// Helpers
// ============================================================================
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

int Tensor::compute_flat_index(const std::vector<int>& indices) const {
    int idx = offset_;
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        idx += indices[i] * strides_[i];
    }
    return idx;
}

std::vector<int> Tensor::unravel_index(int flat_idx,
                                        const std::vector<int>& shape) {
    std::vector<int> indices(shape.size());
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = flat_idx % shape[i];
        flat_idx /= shape[i];
    }
    return indices;
}

std::vector<int> Tensor::broadcast_shapes(const std::vector<int>& a,
                                            const std::vector<int>& b) {
    int ndim = static_cast<int>(std::max(a.size(), b.size()));
    std::vector<int> result(ndim);
    for (int i = 0; i < ndim; ++i) {
        int da = (i < ndim - static_cast<int>(a.size())) ? 1 : a[i - (ndim - a.size())];
        int db = (i < ndim - static_cast<int>(b.size())) ? 1 : b[i - (ndim - b.size())];
        if (da != db && da != 1 && db != 1) {
            throw std::runtime_error("Shapes not broadcastable");
        }
        result[i] = std::max(da, db);
    }
    return result;
}

float Tensor::broadcast_get(const std::vector<int>& idx) const {
    int ndim_diff = static_cast<int>(idx.size()) - ndim();
    int flat = offset_;
    for (int i = 0; i < ndim(); ++i) {
        int dim_idx = idx[i + ndim_diff];
        if (shape_[i] == 1) dim_idx = 0;
        flat += dim_idx * strides_[i];
    }
    return storage_.data()[flat];
}

int Tensor::size(int dim) const {
    if (dim < 0) dim += ndim();
    return shape_[dim];
}

// ============================================================================
// Element access
// ============================================================================
float& Tensor::operator()(const std::vector<int>& indices) {
    return storage_.data()[compute_flat_index(indices)];
}

const float& Tensor::operator()(const std::vector<int>& indices) const {
    return storage_.data()[compute_flat_index(indices)];
}

float Tensor::item() const {
    if (numel_ != 1) {
        throw std::runtime_error("item() requires tensor with exactly 1 element");
    }
    return data()[0];
}

// ============================================================================
// Factory functions
// ============================================================================
Tensor Tensor::zeros(const std::vector<int>& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const std::vector<int>& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::full(const std::vector<int>& shape, float value) {
    return Tensor(shape, value);
}

Tensor Tensor::rand(const std::vector<int>& shape) {
    Tensor t(shape);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < t.numel_; ++i) {
        t.data()[i] = dist(rng());
    }
    return t;
}

Tensor Tensor::randn(const std::vector<int>& shape) {
    Tensor t(shape);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < t.numel_; ++i) {
        t.data()[i] = dist(rng());
    }
    return t;
}

Tensor Tensor::arange(float start, float end, float step) {
    std::vector<float> vals;
    for (float v = start; v < end; v += step) {
        vals.push_back(v);
    }
    return Tensor({static_cast<int>(vals.size())}, vals);
}

Tensor Tensor::eye(int n) {
    Tensor t = zeros({n, n});
    for (int i = 0; i < n; ++i) {
        t.at(i, i) = 1.0f;
    }
    return t;
}

Tensor Tensor::from_vector(const std::vector<float>& data,
                           const std::vector<int>& shape) {
    return Tensor(shape, data);
}

// ============================================================================
// Copy
// ============================================================================
Tensor Tensor::clone() const {
    Tensor t;
    t.shape_ = shape_;
    t.strides_ = strides_;
    t.offset_ = 0;
    t.numel_ = numel_;
    // Make contiguous copy
    Storage s(numel_);
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        s.data()[i] = (*this)(idx);
    }
    t.storage_ = s;
    t.compute_strides();
    return t;
}

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    std::vector<int> expected(shape_.size());
    expected.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        expected[i] = expected[i + 1] * shape_[i + 1];
    }
    return strides_ == expected && offset_ == 0;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone();
}

// ============================================================================
// Shape manipulation
// ============================================================================
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    std::vector<int> resolved = new_shape;
    int neg_idx = -1;
    int known = 1;
    for (int i = 0; i < static_cast<int>(resolved.size()); ++i) {
        if (resolved[i] == -1) {
            if (neg_idx >= 0) throw std::runtime_error("Only one -1 in reshape");
            neg_idx = i;
        } else {
            known *= resolved[i];
        }
    }
    if (neg_idx >= 0) {
        resolved[neg_idx] = numel_ / known;
    }

    int total = std::accumulate(resolved.begin(), resolved.end(), 1,
                                std::multiplies<int>());
    if (total != numel_) {
        throw std::runtime_error("Cannot reshape: size mismatch");
    }

    Tensor c = contiguous();
    c.shape_ = resolved;
    c.compute_strides();
    return c;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    return reshape(new_shape);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += ndim();
    if (dim1 < 0) dim1 += ndim();

    Tensor t;
    t.storage_ = storage_;
    t.shape_ = shape_;
    t.strides_ = strides_;
    t.offset_ = offset_;
    t.numel_ = numel_;

    std::swap(t.shape_[dim0], t.shape_[dim1]);
    std::swap(t.strides_[dim0], t.strides_[dim1]);
    return t;
}

Tensor Tensor::t() const {
    if (ndim() != 2) {
        throw std::runtime_error("t() requires 2D tensor");
    }
    return transpose(0, 1);
}

Tensor Tensor::squeeze(int dim) const {
    std::vector<int> new_shape;
    if (dim == -1) {
        for (int s : shape_) {
            if (s != 1) new_shape.push_back(s);
        }
    } else {
        if (dim < 0) dim += ndim();
        for (int i = 0; i < ndim(); ++i) {
            if (i == dim && shape_[i] == 1) continue;
            new_shape.push_back(shape_[i]);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    return reshape(new_shape);
}

Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);
    return reshape(new_shape);
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    if (start_dim < 0) start_dim += ndim();
    if (end_dim < 0) end_dim += ndim();

    int flat_size = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        flat_size *= shape_[i];
    }

    std::vector<int> new_shape;
    for (int i = 0; i < start_dim; ++i) new_shape.push_back(shape_[i]);
    new_shape.push_back(flat_size);
    for (int i = end_dim + 1; i < ndim(); ++i) new_shape.push_back(shape_[i]);

    return reshape(new_shape);
}

Tensor Tensor::expand(const std::vector<int>& new_shape) const {
    // Expand by setting strides to 0 for broadcasted dims
    int new_ndim = static_cast<int>(new_shape.size());
    int old_ndim = ndim();
    int diff = new_ndim - old_ndim;

    std::vector<int> new_strides(new_ndim, 0);
    for (int i = new_ndim - 1; i >= 0; --i) {
        int old_i = i - diff;
        if (old_i >= 0) {
            if (shape_[old_i] == new_shape[i]) {
                new_strides[i] = strides_[old_i];
            } else if (shape_[old_i] == 1) {
                new_strides[i] = 0; // broadcast
            } else {
                throw std::runtime_error("Cannot expand: incompatible shapes");
            }
        }
    }

    return Tensor(storage_, new_shape, new_strides, offset_);
}

// ============================================================================
// Elementwise operations
// ============================================================================
Tensor Tensor::binary_op(const Tensor& other,
                          std::function<float(float, float)> op) const {
    auto out_shape = broadcast_shapes(shape_, other.shape_);
    int out_numel = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                     std::multiplies<int>());
    Tensor result(out_shape);
    for (int i = 0; i < out_numel; ++i) {
        auto idx = unravel_index(i, out_shape);
        float a = broadcast_get(idx);
        float b = other.broadcast_get(idx);
        result.data()[i] = op(a, b);
    }
    return result;
}

Tensor Tensor::unary_op(std::function<float(float)> op) const {
    Tensor result(shape_);
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        result.data()[i] = op((*this)(idx));
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    return binary_op(other, [](float a, float b) { return a + b; });
}
Tensor Tensor::operator-(const Tensor& other) const {
    return binary_op(other, [](float a, float b) { return a - b; });
}
Tensor Tensor::operator*(const Tensor& other) const {
    return binary_op(other, [](float a, float b) { return a * b; });
}
Tensor Tensor::operator/(const Tensor& other) const {
    return binary_op(other, [](float a, float b) { return a / b; });
}
Tensor Tensor::operator-() const {
    return unary_op([](float a) { return -a; });
}

Tensor Tensor::operator+(float s) const {
    return unary_op([s](float a) { return a + s; });
}
Tensor Tensor::operator-(float s) const {
    return unary_op([s](float a) { return a - s; });
}
Tensor Tensor::operator*(float s) const {
    return unary_op([s](float a) { return a * s; });
}
Tensor Tensor::operator/(float s) const {
    return unary_op([s](float a) { return a / s; });
}

// In-place ops
Tensor& Tensor::operator+=(const Tensor& other) {
    *this = *this + other; return *this;
}
Tensor& Tensor::operator-=(const Tensor& other) {
    *this = *this - other; return *this;
}
Tensor& Tensor::operator*=(const Tensor& other) {
    *this = *this * other; return *this;
}
Tensor& Tensor::operator/=(const Tensor& other) {
    *this = *this / other; return *this;
}
Tensor& Tensor::operator+=(float s) {
    *this = *this + s; return *this;
}
Tensor& Tensor::operator-=(float s) {
    *this = *this - s; return *this;
}
Tensor& Tensor::operator*=(float s) {
    *this = *this * s; return *this;
}
Tensor& Tensor::operator/=(float s) {
    *this = *this / s; return *this;
}

// Math
Tensor Tensor::exp() const {
    return unary_op([](float a) { return std::exp(a); });
}
Tensor Tensor::log() const {
    return unary_op([](float a) { return std::log(a); });
}
Tensor Tensor::pow(float e) const {
    return unary_op([e](float a) { return std::pow(a, e); });
}
Tensor Tensor::sqrt() const {
    return unary_op([](float a) { return std::sqrt(a); });
}
Tensor Tensor::abs() const {
    return unary_op([](float a) { return std::abs(a); });
}
Tensor Tensor::clamp(float lo, float hi) const {
    return unary_op([lo, hi](float a) { return std::max(lo, std::min(hi, a)); });
}

// Free scalar-tensor ops
Tensor operator+(float s, const Tensor& t) { return t + s; }
Tensor operator-(float s, const Tensor& t) { return t.unary_op([s](float a) { return s - a; }); }
Tensor operator*(float s, const Tensor& t) { return t * s; }
Tensor operator/(float s, const Tensor& t) { return t.unary_op([s](float a) { return s / a; }); }

// ============================================================================
// Reductions
// ============================================================================
Tensor Tensor::reduce_dim(int dim, bool keepdim,
                           std::function<float(float, float)> op,
                           float init) const {
    if (dim < 0) dim += ndim();

    std::vector<int> out_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, init);

    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        // Build output index by removing the reduced dim
        std::vector<int> out_idx;
        for (int d = 0; d < ndim(); ++d) {
            if (d == dim) {
                if (keepdim) out_idx.push_back(0);
            } else {
                out_idx.push_back(idx[d]);
            }
        }
        if (out_idx.empty()) out_idx.push_back(0);

        float& r = result(out_idx);
        r = op(r, (*this)(idx));
    }
    return result;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim == -1 && !keepdim) {
        // Global sum
        return Tensor({1}, {sum_all()});
    }
    return reduce_dim(dim, keepdim,
                      [](float a, float b) { return a + b; }, 0.0f);
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim == -1 && !keepdim) {
        return Tensor({1}, {mean_all()});
    }
    if (dim < 0) dim += ndim();
    Tensor s = sum(dim, keepdim);
    return s / static_cast<float>(shape_[dim]);
}

Tensor Tensor::max(int dim, bool keepdim) const {
    if (dim == -1 && !keepdim) {
        return Tensor({1}, {max_all()});
    }
    return reduce_dim(dim, keepdim,
                      [](float a, float b) { return std::max(a, b); },
                      -std::numeric_limits<float>::infinity());
}

Tensor Tensor::min(int dim, bool keepdim) const {
    if (dim == -1 && !keepdim) {
        return Tensor({1}, {min_all()});
    }
    return reduce_dim(dim, keepdim,
                      [](float a, float b) { return std::min(a, b); },
                      std::numeric_limits<float>::infinity());
}

Tensor Tensor::argmax(int dim) const {
    if (dim < 0) dim += ndim();
    // Build output shape
    std::vector<int> out_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i != dim) out_shape.push_back(shape_[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, 0.0f);
    Tensor max_vals(out_shape, -std::numeric_limits<float>::infinity());

    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        std::vector<int> out_idx;
        for (int d = 0; d < ndim(); ++d) {
            if (d != dim) out_idx.push_back(idx[d]);
        }
        if (out_idx.empty()) out_idx.push_back(0);

        float val = (*this)(idx);
        if (val > max_vals(out_idx)) {
            max_vals(out_idx) = val;
            result(out_idx) = static_cast<float>(idx[dim]);
        }
    }
    return result;
}

float Tensor::sum_all() const {
    float s = 0;
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        s += (*this)(idx);
    }
    return s;
}

float Tensor::mean_all() const {
    return sum_all() / static_cast<float>(numel_);
}

float Tensor::max_all() const {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        m = std::max(m, (*this)(idx));
    }
    return m;
}

float Tensor::min_all() const {
    float m = std::numeric_limits<float>::infinity();
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        m = std::min(m, (*this)(idx));
    }
    return m;
}

// ============================================================================
// Matrix multiplication
// ============================================================================
Tensor Tensor::matmul(const Tensor& other) const {
    // Support 2D x 2D
    if (ndim() == 2 && other.ndim() == 2) {
        int M = shape_[0], K = shape_[1], N = other.shape_[1];
        if (K != other.shape_[0]) {
            throw std::runtime_error("matmul: incompatible shapes");
        }
        Tensor result = zeros({M, N});
        Tensor a_c = contiguous();
        Tensor b_c = other.contiguous();
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += a_c.at(i, k) * b_c.at(k, j);
                }
                result.at(i, j) = sum;
            }
        }
        return result;
    }
    // 1D x 1D -> dot product
    if (ndim() == 1 && other.ndim() == 1) {
        if (shape_[0] != other.shape_[0]) {
            throw std::runtime_error("matmul: incompatible shapes");
        }
        float sum = 0;
        for (int i = 0; i < shape_[0]; ++i) {
            sum += at(i) * other.at(i);
        }
        return Tensor({1}, {sum});
    }
    // Batched matmul: (..., M, K) x (..., K, N) -> (..., M, N)
    if (ndim() >= 2 && other.ndim() >= 2) {
        int M = shape_[ndim() - 2], K = shape_[ndim() - 1];
        int N = other.shape_[other.ndim() - 1];
        if (K != other.shape_[other.ndim() - 2]) {
            throw std::runtime_error("matmul: incompatible shapes");
        }
        // Compute batch shape
        std::vector<int> batch_a(shape_.begin(), shape_.end() - 2);
        std::vector<int> batch_b(other.shape_.begin(), other.shape_.end() - 2);
        auto batch_shape = broadcast_shapes(
            batch_a.empty() ? std::vector<int>{1} : batch_a,
            batch_b.empty() ? std::vector<int>{1} : batch_b);
        if (batch_a.empty() && batch_b.empty()) batch_shape.clear();

        std::vector<int> out_shape = batch_shape;
        out_shape.push_back(M);
        out_shape.push_back(N);

        int batch_numel = batch_shape.empty() ? 1 :
            std::accumulate(batch_shape.begin(), batch_shape.end(), 1,
                            std::multiplies<int>());

        Tensor result = zeros(out_shape);
        Tensor a_c = contiguous();
        Tensor b_c = other.contiguous();

        for (int b = 0; b < batch_numel; ++b) {
            auto batch_idx = batch_shape.empty() ?
                std::vector<int>{} : unravel_index(b, batch_shape);

            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k) {
                        std::vector<int> a_idx = batch_idx;
                        a_idx.push_back(i);
                        a_idx.push_back(k);
                        std::vector<int> b_idx = batch_idx;
                        b_idx.push_back(k);
                        b_idx.push_back(j);
                        sum += a_c.broadcast_get(a_idx) * b_c.broadcast_get(b_idx);
                    }
                    std::vector<int> out_idx = batch_idx;
                    out_idx.push_back(i);
                    out_idx.push_back(j);
                    result(out_idx) = sum;
                }
            }
        }
        return result;
    }
    throw std::runtime_error("matmul: unsupported tensor dimensions");
}

// ============================================================================
// Comparisons
// ============================================================================
Tensor Tensor::operator>(float val) const {
    return unary_op([val](float a) { return a > val ? 1.0f : 0.0f; });
}
Tensor Tensor::operator<(float val) const {
    return unary_op([val](float a) { return a < val ? 1.0f : 0.0f; });
}
Tensor Tensor::operator>=(float val) const {
    return unary_op([val](float a) { return a >= val ? 1.0f : 0.0f; });
}
Tensor Tensor::operator<=(float val) const {
    return unary_op([val](float a) { return a <= val ? 1.0f : 0.0f; });
}
Tensor Tensor::operator==(float val) const {
    return unary_op([val](float a) { return a == val ? 1.0f : 0.0f; });
}

// ============================================================================
// In-place utilities
// ============================================================================
void Tensor::fill_(float value) {
    for (int i = 0; i < numel_; ++i) {
        auto idx = unravel_index(i, shape_);
        (*this)(idx) = value;
    }
}

void Tensor::zero_() { fill_(0.0f); }

void Tensor::uniform_(float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    for (int i = 0; i < numel_; ++i) {
        data()[i] = dist(rng());
    }
}

void Tensor::normal_(float mean, float std) {
    std::normal_distribution<float> dist(mean, std);
    for (int i = 0; i < numel_; ++i) {
        data()[i] = dist(rng());
    }
}

// ============================================================================
// Print
// ============================================================================
std::string Tensor::to_string() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (int i = 0; i < ndim(); ++i) {
        if (i > 0) os << ", ";
        os << shape_[i];
    }
    os << "], data=[";

    int max_print = std::min(numel_, 20);
    for (int i = 0; i < max_print; ++i) {
        if (i > 0) os << ", ";
        auto idx = unravel_index(i, shape_);
        os << (*this)(idx);
    }
    if (numel_ > 20) os << ", ...";
    os << "])";
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string();
    return os;
}

} // namespace neuralcore
