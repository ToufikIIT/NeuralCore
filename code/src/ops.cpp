#include "neuralcore/ops.h"

namespace neuralcore {

// ============================================================================
// Helper: reduce gradient to match target shape (undo broadcasting)
// ============================================================================
Tensor reduce_grad_to_shape(const Tensor& grad, const std::vector<int>& target_shape) {
    Tensor result = grad;

    // Sum out leading dimensions that were broadcast-added
    while (result.ndim() > static_cast<int>(target_shape.size())) {
        result = result.sum(0, false);
    }

    // Sum along dimensions that are 1 in target but >1 in grad
    for (int i = 0; i < static_cast<int>(target_shape.size()); ++i) {
        if (target_shape[i] == 1 && result.shape()[i] != 1) {
            result = result.sum(i, true);
        }
    }

    return result.reshape(target_shape);
}

// ============================================================================
// Arithmetic backward
// ============================================================================
std::vector<Tensor> AddBackward::backward(const Tensor& grad_output) {
    return {grad_output, grad_output};
}

std::vector<Tensor> SubBackward::backward(const Tensor& grad_output) {
    return {grad_output, -grad_output};
}

std::vector<Tensor> MulBackward::backward(const Tensor& grad_output) {
    return {grad_output * b_data, grad_output * a_data};
}

std::vector<Tensor> DivBackward::backward(const Tensor& grad_output) {
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    Tensor grad_a = grad_output / b_data;
    Tensor grad_b = -grad_output * a_data / (b_data * b_data);
    return {grad_a, grad_b};
}

std::vector<Tensor> NegBackward::backward(const Tensor& grad_output) {
    return {-grad_output};
}

// ============================================================================
// Scalar backward
// ============================================================================
std::vector<Tensor> AddScalarBackward::backward(const Tensor& grad_output) {
    return {grad_output};
}

std::vector<Tensor> MulScalarBackward::backward(const Tensor& grad_output) {
    return {grad_output * scalar};
}

std::vector<Tensor> DivScalarBackward::backward(const Tensor& grad_output) {
    return {grad_output / scalar};
}

// ============================================================================
// Math backward
// ============================================================================
std::vector<Tensor> ExpBackward::backward(const Tensor& grad_output) {
    return {grad_output * output_data};
}

std::vector<Tensor> LogBackward::backward(const Tensor& grad_output) {
    return {grad_output / input_data};
}

std::vector<Tensor> PowBackward::backward(const Tensor& grad_output) {
    // d(x^n)/dx = n * x^(n-1)
    return {grad_output * (input_data.pow(exponent - 1.0f) * exponent)};
}

std::vector<Tensor> SqrtBackward::backward(const Tensor& grad_output) {
    // d(sqrt(x))/dx = 0.5 / sqrt(x)
    return {grad_output / (output_data * 2.0f)};
}

std::vector<Tensor> AbsBackward::backward(const Tensor& grad_output) {
    // d|x|/dx = sign(x)
    Tensor sign = input_data.clamp(-1.0f, 1.0f);
    // Actually compute proper sign
    Tensor result(input_data.shape());
    for (int i = 0; i < input_data.size(); ++i) {
        float v = input_data.data()[i];
        result.data()[i] = v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f);
    }
    return {grad_output * result};
}

std::vector<Tensor> ClampBackward::backward(const Tensor& grad_output) {
    Tensor mask(input_data.shape());
    for (int i = 0; i < input_data.size(); ++i) {
        float v = input_data.data()[i];
        mask.data()[i] = (v >= min_val && v <= max_val) ? 1.0f : 0.0f;
    }
    return {grad_output * mask};
}

// ============================================================================
// Activation backward
// ============================================================================
std::vector<Tensor> ReluBackward::backward(const Tensor& grad_output) {
    Tensor mask(input_data.shape());
    for (int i = 0; i < input_data.size(); ++i) {
        mask.data()[i] = input_data.data()[i] > 0 ? 1.0f : 0.0f;
    }
    return {grad_output * mask};
}

std::vector<Tensor> SigmoidBackward::backward(const Tensor& grad_output) {
    // d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    Tensor ones = Tensor::ones(output_data.shape());
    return {grad_output * output_data * (ones - output_data)};
}

std::vector<Tensor> TanhBackward::backward(const Tensor& grad_output) {
    // d(tanh)/dx = 1 - tanh^2
    Tensor ones = Tensor::ones(output_data.shape());
    return {grad_output * (ones - output_data * output_data)};
}

// ============================================================================
// Reduction backward
// ============================================================================
std::vector<Tensor> SumBackward::backward(const Tensor& grad_output) {
    if (dim == -1) {
        // Global sum: gradient is ones * grad_scalar
        return {Tensor::ones(input_shape) * grad_output.data()[0]};
    }
    // Expand grad along the summed dimension
    Tensor g = grad_output;
    if (!keepdim) {
        g = g.unsqueeze(dim);
    }
    return {g.expand(input_shape).contiguous()};
}

std::vector<Tensor> MeanBackward::backward(const Tensor& grad_output) {
    int n;
    if (dim == -1) {
        n = 1;
        for (int s : input_shape) n *= s;
        return {Tensor::ones(input_shape) * (grad_output.data()[0] / static_cast<float>(n))};
    }
    n = input_shape[dim < 0 ? dim + static_cast<int>(input_shape.size()) : dim];
    Tensor g = grad_output;
    if (!keepdim) {
        g = g.unsqueeze(dim);
    }
    return {(g.expand(input_shape).contiguous()) / static_cast<float>(n)};
}

// ============================================================================
// Matrix backward
// ============================================================================
std::vector<Tensor> MatmulBackward::backward(const Tensor& grad_output) {
    // C = A @ B => dA = dC @ B^T, dB = A^T @ dC
    if (a_data.ndim() == 2 && b_data.ndim() == 2) {
        Tensor grad_a = grad_output.matmul(b_data.t());
        Tensor grad_b = a_data.t().matmul(grad_output);
        return {grad_a, grad_b};
    }
    // 1D dot product
    if (a_data.ndim() == 1 && b_data.ndim() == 1) {
        float g = grad_output.data()[0];
        return {b_data * g, a_data * g};
    }
    // Batched: same logic per batch element
    Tensor grad_a = grad_output.matmul(b_data.transpose(-2, -1));
    Tensor grad_b = a_data.transpose(-2, -1).matmul(grad_output);
    return {grad_a, grad_b};
}

std::vector<Tensor> TransposeBackward::backward(const Tensor& grad_output) {
    return {grad_output.transpose(dim0, dim1).contiguous()};
}

// ============================================================================
// Shape backward
// ============================================================================
std::vector<Tensor> ReshapeBackward::backward(const Tensor& grad_output) {
    return {grad_output.reshape(original_shape)};
}

std::vector<Tensor> ExpandBackward::backward(const Tensor& grad_output) {
    return {reduce_grad_to_shape(grad_output, original_shape)};
}

std::vector<Tensor> UnsqueezeBackward::backward(const Tensor& grad_output) {
    return {grad_output.reshape(original_shape)};
}

std::vector<Tensor> SqueezeBackward::backward(const Tensor& grad_output) {
    return {grad_output.reshape(original_shape)};
}

} // namespace neuralcore
