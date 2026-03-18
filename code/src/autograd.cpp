#include "neuralcore/autograd.h"
#include "neuralcore/ops.h"
#include <queue>
#include <unordered_set>

namespace neuralcore {

// ============================================================================
// NoGradGuard
// ============================================================================
thread_local bool NoGradGuard::no_grad_active_ = false;

NoGradGuard::NoGradGuard() { no_grad_active_ = true; }
NoGradGuard::~NoGradGuard() { no_grad_active_ = false; }
bool NoGradGuard::is_enabled() { return no_grad_active_; }

// ============================================================================
// Variable
// ============================================================================
Variable::Variable() : requires_grad(false) {}

Variable::Variable(const Tensor& data, bool requires_grad)
    : data(data), requires_grad(requires_grad) {}

VariablePtr Variable::create(const Tensor& data, bool requires_grad) {
    return std::make_shared<Variable>(data, requires_grad);
}

VariablePtr var(const Tensor& data, bool requires_grad) {
    return Variable::create(data, requires_grad);
}

void Variable::zero_grad() {
    grad = Tensor::zeros(data.shape());
}

VariablePtr Variable::detach() const {
    return Variable::create(data.clone(), false);
}

// ============================================================================
// Backward pass — topological sort then propagate gradients
// ============================================================================
void Variable::backward() {
    if (data.size() != 1) {
        throw std::runtime_error("backward() requires scalar output");
    }
    backward(Tensor::ones(data.shape()));
}

void Variable::backward(const Tensor& grad_output) {
    if (!requires_grad) return;

    // Initialize gradient
    if (grad.empty()) {
        grad = Tensor::zeros(data.shape());
    }

    // Topological sort
    std::vector<Variable*> topo_order;
    std::unordered_set<Variable*> visited;

    std::function<void(Variable*)> build_topo = [&](Variable* v) {
        if (visited.count(v)) return;
        visited.insert(v);
        if (v->grad_fn) {
            for (auto& input : v->grad_fn->inputs) {
                if (input && input->requires_grad) {
                    build_topo(input.get());
                }
            }
        }
        topo_order.push_back(v);
    };

    build_topo(this);
    std::reverse(topo_order.begin(), topo_order.end());

    // Set gradient for the root
    this->grad = grad_output.clone();

    // Backprop
    for (auto* node : topo_order) {
        if (!node->grad_fn) continue;

        auto grads = node->grad_fn->backward(node->grad);

        for (size_t i = 0; i < node->grad_fn->inputs.size(); ++i) {
            auto& input = node->grad_fn->inputs[i];
            if (!input || !input->requires_grad) continue;
            if (i >= grads.size()) continue;

            Tensor g = grads[i];
            // Reduce gradient if shapes don't match (broadcasting)
            if (g.shape() != input->data.shape()) {
                g = reduce_grad_to_shape(g, input->data.shape());
            }

            if (input->grad.empty()) {
                input->grad = Tensor::zeros(input->data.shape());
            }
            input->grad += g;
        }
    }
}

// ============================================================================
// Arithmetic ops
// ============================================================================
VariablePtr Variable::operator+(const VariablePtr& other) {
    auto result = Variable::create(data + other->data,
        requires_grad || other->requires_grad);
    if (result->requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<AddBackward>();
        fn->name = "AddBackward";
        fn->shape_a = data.shape();
        fn->shape_b = other->data.shape();
        fn->inputs = {shared_from_this(), other};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator-(const VariablePtr& other) {
    auto result = Variable::create(data - other->data,
        requires_grad || other->requires_grad);
    if (result->requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SubBackward>();
        fn->name = "SubBackward";
        fn->shape_a = data.shape();
        fn->shape_b = other->data.shape();
        fn->inputs = {shared_from_this(), other};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator*(const VariablePtr& other) {
    auto result = Variable::create(data * other->data,
        requires_grad || other->requires_grad);
    if (result->requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MulBackward>();
        fn->name = "MulBackward";
        fn->a_data = data.clone();
        fn->b_data = other->data.clone();
        fn->inputs = {shared_from_this(), other};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator/(const VariablePtr& other) {
    auto result = Variable::create(data / other->data,
        requires_grad || other->requires_grad);
    if (result->requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<DivBackward>();
        fn->name = "DivBackward";
        fn->a_data = data.clone();
        fn->b_data = other->data.clone();
        fn->inputs = {shared_from_this(), other};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator-() {
    auto result = Variable::create(-data, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<NegBackward>();
        fn->name = "NegBackward";
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

// Scalar ops
VariablePtr Variable::operator+(float scalar) {
    auto result = Variable::create(data + scalar, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<AddScalarBackward>();
        fn->name = "AddScalarBackward";
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator-(float scalar) {
    return *this + (-scalar);
}

VariablePtr Variable::operator*(float scalar) {
    auto result = Variable::create(data * scalar, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MulScalarBackward>();
        fn->name = "MulScalarBackward";
        fn->scalar = scalar;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::operator/(float scalar) {
    return *this * (1.0f / scalar);
}

// ============================================================================
// Math ops
// ============================================================================
VariablePtr Variable::exp() {
    Tensor out = data.exp();
    auto result = Variable::create(out, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ExpBackward>();
        fn->name = "ExpBackward";
        fn->output_data = out.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::log() {
    auto result = Variable::create(data.log(), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<LogBackward>();
        fn->name = "LogBackward";
        fn->input_data = data.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::pow(float exponent) {
    auto result = Variable::create(data.pow(exponent), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<PowBackward>();
        fn->name = "PowBackward";
        fn->input_data = data.clone();
        fn->exponent = exponent;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::sqrt() {
    Tensor out = data.sqrt();
    auto result = Variable::create(out, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SqrtBackward>();
        fn->name = "SqrtBackward";
        fn->output_data = out.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::relu() {
    Tensor out = data.clamp(0.0f, std::numeric_limits<float>::infinity());
    auto result = Variable::create(out, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReluBackward>();
        fn->name = "ReluBackward";
        fn->input_data = data.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::sigmoid() {
    // sigmoid(x) = 1 / (1 + exp(-x))
    Tensor out = ((-data).exp() + 1.0f);
    // out = 1 / out
    Tensor ones_t = Tensor::ones(data.shape());
    out = ones_t / out;

    auto result = Variable::create(out, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SigmoidBackward>();
        fn->name = "SigmoidBackward";
        fn->output_data = out.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::tanh() {
    // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tensor ex = data.exp();
    Tensor enx = (-data).exp();
    Tensor out = (ex - enx) / (ex + enx);

    auto result = Variable::create(out, requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<TanhBackward>();
        fn->name = "TanhBackward";
        fn->output_data = out.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::abs() {
    auto result = Variable::create(data.abs(), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<AbsBackward>();
        fn->name = "AbsBackward";
        fn->input_data = data.clone();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::clamp(float min_val, float max_val) {
    auto result = Variable::create(data.clamp(min_val, max_val), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ClampBackward>();
        fn->name = "ClampBackward";
        fn->input_data = data.clone();
        fn->min_val = min_val;
        fn->max_val = max_val;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

// ============================================================================
// Reductions
// ============================================================================
VariablePtr Variable::sum(int dim, bool keepdim) {
    auto result = Variable::create(data.sum(dim, keepdim), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SumBackward>();
        fn->name = "SumBackward";
        fn->input_shape = data.shape();
        fn->dim = dim;
        fn->keepdim = keepdim;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::mean(int dim, bool keepdim) {
    auto result = Variable::create(data.mean(dim, keepdim), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MeanBackward>();
        fn->name = "MeanBackward";
        fn->input_shape = data.shape();
        fn->dim = dim;
        fn->keepdim = keepdim;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

// ============================================================================
// Matrix ops
// ============================================================================
VariablePtr Variable::matmul(const VariablePtr& other) {
    auto result = Variable::create(data.matmul(other->data),
        requires_grad || other->requires_grad);
    if (result->requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<MatmulBackward>();
        fn->name = "MatmulBackward";
        fn->a_data = data.clone();
        fn->b_data = other->data.clone();
        fn->inputs = {shared_from_this(), other};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::transpose(int dim0, int dim1) {
    auto result = Variable::create(data.transpose(dim0, dim1), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<TransposeBackward>();
        fn->name = "TransposeBackward";
        fn->dim0 = dim0;
        fn->dim1 = dim1;
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::t() {
    return transpose(0, 1);
}

// ============================================================================
// Shape ops
// ============================================================================
VariablePtr Variable::reshape(const std::vector<int>& new_shape) {
    auto result = Variable::create(data.reshape(new_shape), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReshapeBackward>();
        fn->name = "ReshapeBackward";
        fn->original_shape = data.shape();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::flatten(int start_dim, int end_dim) {
    auto result = Variable::create(data.flatten(start_dim, end_dim), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ReshapeBackward>();
        fn->name = "FlattenBackward";
        fn->original_shape = data.shape();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::unsqueeze(int dim) {
    auto result = Variable::create(data.unsqueeze(dim), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<UnsqueezeBackward>();
        fn->name = "UnsqueezeBackward";
        fn->original_shape = data.shape();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::squeeze(int dim) {
    auto result = Variable::create(data.squeeze(dim), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<SqueezeBackward>();
        fn->name = "SqueezeBackward";
        fn->original_shape = data.shape();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

VariablePtr Variable::expand(const std::vector<int>& new_shape) {
    auto result = Variable::create(data.expand(new_shape), requires_grad);
    if (requires_grad && !NoGradGuard::is_enabled()) {
        auto fn = std::make_shared<ExpandBackward>();
        fn->name = "ExpandBackward";
        fn->original_shape = data.shape();
        fn->inputs = {shared_from_this()};
        result->grad_fn = fn;
    }
    return result;
}

// Print
std::ostream& operator<<(std::ostream& os, const Variable& v) {
    os << "Variable(" << v.data;
    if (v.requires_grad) os << ", requires_grad=true";
    if (v.grad_fn) os << ", grad_fn=" << v.grad_fn->name;
    os << ")";
    return os;
}

} // namespace neuralcore
