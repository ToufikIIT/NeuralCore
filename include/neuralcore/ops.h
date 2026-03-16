#pragma once

#include "autograd.h"

namespace neuralcore {

// Helper: reduce gradient to match target shape (undo broadcasting)
Tensor reduce_grad_to_shape(const Tensor& grad, const std::vector<int>& target_shape);

// --- Arithmetic backward ---
class AddBackward : public GradFunction {
public:
    std::vector<int> shape_a, shape_b;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class SubBackward : public GradFunction {
public:
    std::vector<int> shape_a, shape_b;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class MulBackward : public GradFunction {
public:
    Tensor a_data, b_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class DivBackward : public GradFunction {
public:
    Tensor a_data, b_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class NegBackward : public GradFunction {
public:
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Scalar backward ---
class AddScalarBackward : public GradFunction {
public:
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class MulScalarBackward : public GradFunction {
public:
    float scalar;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class DivScalarBackward : public GradFunction {
public:
    float scalar;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Math backward ---
class ExpBackward : public GradFunction {
public:
    Tensor output_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class LogBackward : public GradFunction {
public:
    Tensor input_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class PowBackward : public GradFunction {
public:
    Tensor input_data;
    float exponent;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class SqrtBackward : public GradFunction {
public:
    Tensor output_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class AbsBackward : public GradFunction {
public:
    Tensor input_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class ClampBackward : public GradFunction {
public:
    Tensor input_data;
    float min_val, max_val;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Activation backward ---
class ReluBackward : public GradFunction {
public:
    Tensor input_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class SigmoidBackward : public GradFunction {
public:
    Tensor output_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class TanhBackward : public GradFunction {
public:
    Tensor output_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Reduction backward ---
class SumBackward : public GradFunction {
public:
    std::vector<int> input_shape;
    int dim;
    bool keepdim;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class MeanBackward : public GradFunction {
public:
    std::vector<int> input_shape;
    int dim;
    bool keepdim;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Matrix backward ---
class MatmulBackward : public GradFunction {
public:
    Tensor a_data, b_data;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class TransposeBackward : public GradFunction {
public:
    int dim0, dim1;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

// --- Shape backward ---
class ReshapeBackward : public GradFunction {
public:
    std::vector<int> original_shape;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class ExpandBackward : public GradFunction {
public:
    std::vector<int> original_shape;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class UnsqueezeBackward : public GradFunction {
public:
    std::vector<int> original_shape;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

class SqueezeBackward : public GradFunction {
public:
    std::vector<int> original_shape;
    std::vector<Tensor> backward(const Tensor& grad_output) override;
};

} // namespace neuralcore
