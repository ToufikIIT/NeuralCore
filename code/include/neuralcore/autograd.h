#pragma once

#include "tensor.h"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <string>

namespace neuralcore {

class Variable;
using VariablePtr = std::shared_ptr<Variable>;

// ============================================================================
// GradFunction: base class for backward computation graph nodes
// ============================================================================
class GradFunction {
public:
    virtual ~GradFunction() = default;
    virtual std::vector<Tensor> backward(const Tensor& grad_output) = 0;
    std::vector<VariablePtr> inputs;
    std::string name = "GradFunction";
};

using GradFnPtr = std::shared_ptr<GradFunction>;

// ============================================================================
// Variable: wraps a Tensor with autograd support
// ============================================================================
class Variable : public std::enable_shared_from_this<Variable> {
public:
    Tensor data;
    Tensor grad;
    GradFnPtr grad_fn;
    bool requires_grad;

    Variable();
    explicit Variable(const Tensor& data, bool requires_grad = false);
    static VariablePtr create(const Tensor& data, bool requires_grad = false);

    void backward(const Tensor& grad_output);
    void backward();
    void zero_grad();
    VariablePtr detach() const;

    const std::vector<int>& shape() const { return data.shape(); }
    int ndim() const { return data.ndim(); }
    int size() const { return data.size(); }
    int size(int dim) const { return data.size(dim); }

    // Arithmetic
    VariablePtr operator+(const VariablePtr& other);
    VariablePtr operator-(const VariablePtr& other);
    VariablePtr operator*(const VariablePtr& other);
    VariablePtr operator/(const VariablePtr& other);
    VariablePtr operator-();

    VariablePtr operator+(float scalar);
    VariablePtr operator-(float scalar);
    VariablePtr operator*(float scalar);
    VariablePtr operator/(float scalar);

    // Math
    VariablePtr exp();
    VariablePtr log();
    VariablePtr pow(float exponent);
    VariablePtr sqrt();
    VariablePtr relu();
    VariablePtr sigmoid();
    VariablePtr tanh();
    VariablePtr abs();
    VariablePtr clamp(float min_val, float max_val);

    // Reductions
    VariablePtr sum(int dim = -1, bool keepdim = false);
    VariablePtr mean(int dim = -1, bool keepdim = false);

    // Matrix
    VariablePtr matmul(const VariablePtr& other);
    VariablePtr transpose(int dim0, int dim1);
    VariablePtr t();

    // Shape
    VariablePtr reshape(const std::vector<int>& new_shape);
    VariablePtr flatten(int start_dim = 0, int end_dim = -1);
    VariablePtr unsqueeze(int dim);
    VariablePtr squeeze(int dim = -1);
    VariablePtr expand(const std::vector<int>& new_shape);

    friend std::ostream& operator<<(std::ostream& os, const Variable& v);
};

VariablePtr var(const Tensor& data, bool requires_grad = false);

// NoGrad guard
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
    static bool is_enabled();
private:
    static thread_local bool no_grad_active_;
};

} // namespace neuralcore
