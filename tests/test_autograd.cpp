// Test: Autograd engine
#include <iostream>
#include <cmath>
#include "neuralcore/tensor.h"
#include "neuralcore/autograd.h"

using namespace neuralcore;

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL at line " << __LINE__ << ": " \
                  << (a) << " != " << (b) << " (tol=" << (tol) << ")" \
                  << std::endl; return 1; \
    }} while(0)

// Numerical gradient check helper
float numerical_grad(std::function<float(float)> f, float x, float eps = 1e-4f) {
    return (f(x + eps) - f(x - eps)) / (2.0f * eps);
}

int main() {
    std::cout << "=== Autograd Tests ===" << std::endl;

    // Test: simple addition backward
    {
        auto a = Variable::create(Tensor({1}, {3.0f}), true);
        auto b = Variable::create(Tensor({1}, {5.0f}), true);
        auto c = *a + b;
        c->backward();
        ASSERT_NEAR(a->grad.at(0), 1.0f, 1e-5);
        ASSERT_NEAR(b->grad.at(0), 1.0f, 1e-5);
        std::cout << "  [PASS] Addition backward" << std::endl;
    }

    // Test: multiplication backward
    {
        auto a = Variable::create(Tensor({1}, {3.0f}), true);
        auto b = Variable::create(Tensor({1}, {5.0f}), true);
        auto c = *a * b;
        c->backward();
        ASSERT_NEAR(a->grad.at(0), 5.0f, 1e-5);
        ASSERT_NEAR(b->grad.at(0), 3.0f, 1e-5);
        std::cout << "  [PASS] Multiplication backward" << std::endl;
    }

    // Test: chain rule (a * b + c)
    {
        auto a = Variable::create(Tensor({1}, {2.0f}), true);
        auto b = Variable::create(Tensor({1}, {3.0f}), true);
        auto c = Variable::create(Tensor({1}, {4.0f}), true);
        auto d = *(*a * b) + c;
        d->backward();
        ASSERT_NEAR(a->grad.at(0), 3.0f, 1e-5); // d/da = b
        ASSERT_NEAR(b->grad.at(0), 2.0f, 1e-5); // d/db = a
        ASSERT_NEAR(c->grad.at(0), 1.0f, 1e-5); // d/dc = 1
        std::cout << "  [PASS] Chain rule" << std::endl;
    }

    // Test: power backward
    {
        auto x = Variable::create(Tensor({1}, {3.0f}), true);
        auto y = x->pow(2.0f); // y = x^2
        y->backward();
        ASSERT_NEAR(x->grad.at(0), 6.0f, 1e-4); // dy/dx = 2x = 6
        std::cout << "  [PASS] Power backward" << std::endl;
    }

    // Test: exp backward
    {
        auto x = Variable::create(Tensor({1}, {1.0f}), true);
        auto y = x->exp();
        y->backward();
        ASSERT_NEAR(x->grad.at(0), std::exp(1.0f), 1e-4);
        std::cout << "  [PASS] Exp backward" << std::endl;
    }

    // Test: log backward
    {
        auto x = Variable::create(Tensor({1}, {2.0f}), true);
        auto y = x->log();
        y->backward();
        ASSERT_NEAR(x->grad.at(0), 0.5f, 1e-4);
        std::cout << "  [PASS] Log backward" << std::endl;
    }

    // Test: sigmoid backward (numerical check)
    {
        float xv = 0.5f;
        auto x = Variable::create(Tensor({1}, {xv}), true);
        auto y = x->sigmoid();
        y->backward();

        float analytical = x->grad.at(0);
        float numerical = numerical_grad([](float v) {
            float s = 1.0f / (1.0f + std::exp(-v));
            return s;
        }, xv);
        ASSERT_NEAR(analytical, numerical, 1e-3);
        std::cout << "  [PASS] Sigmoid backward (numerical check)" << std::endl;
    }

    // Test: tanh backward (numerical check)
    {
        float xv = 0.7f;
        auto x = Variable::create(Tensor({1}, {xv}), true);
        auto y = x->tanh();
        y->backward();

        float analytical = x->grad.at(0);
        float numerical = numerical_grad([](float v) {
            return std::tanh(v);
        }, xv);
        ASSERT_NEAR(analytical, numerical, 1e-3);
        std::cout << "  [PASS] Tanh backward (numerical check)" << std::endl;
    }

    // Test: relu backward
    {
        auto x = Variable::create(Tensor({3}, {-1.0f, 0.0f, 2.0f}), true);
        auto y = x->relu();
        auto z = y->sum();
        z->backward();
        ASSERT_NEAR(x->grad.at(0), 0.0f, 1e-5);
        ASSERT_NEAR(x->grad.at(1), 0.0f, 1e-5);
        ASSERT_NEAR(x->grad.at(2), 1.0f, 1e-5);
        std::cout << "  [PASS] ReLU backward" << std::endl;
    }

    // Test: matmul backward
    {
        auto a = Variable::create(Tensor({2, 2}, {1, 2, 3, 4}), true);
        auto b = Variable::create(Tensor({2, 2}, {5, 6, 7, 8}), true);
        auto c = a->matmul(b);
        auto loss = c->sum();
        loss->backward();
        // dL/dA = ones @ B^T, dL/dB = A^T @ ones
        // B^T = [[5,7],[6,8]], sum of each row = [12, 14] => a->grad
        ASSERT_NEAR(a->grad.at(0, 0), 11.0f, 1e-4);
        ASSERT_NEAR(a->grad.at(0, 1), 15.0f, 1e-4);
        std::cout << "  [PASS] Matmul backward" << std::endl;
    }

    // Test: mean backward
    {
        auto x = Variable::create(Tensor({4}, {1, 2, 3, 4}), true);
        auto y = x->mean();
        y->backward();
        for (int i = 0; i < 4; ++i) {
            ASSERT_NEAR(x->grad.at(i), 0.25f, 1e-5);
        }
        std::cout << "  [PASS] Mean backward" << std::endl;
    }

    // Test: no_grad
    {
        auto x = Variable::create(Tensor({1}, {5.0f}), true);
        {
            NoGradGuard guard;
            auto y = x->pow(2.0f);
            // y should not have grad_fn
            if (y->grad_fn != nullptr) {
                std::cerr << "FAIL: grad_fn should be null in no_grad mode" << std::endl;
                return 1;
            }
        }
        std::cout << "  [PASS] NoGrad guard" << std::endl;
    }

    std::cout << "\nAll autograd tests passed!" << std::endl;
    return 0;
}
