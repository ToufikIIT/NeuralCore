// Test: Neural network modules
#include <iostream>
#include <cmath>
#include "neuralcore/tensor.h"
#include "neuralcore/autograd.h"
#include "neuralcore/nn/linear.h"
#include "neuralcore/nn/activation.h"
#include "neuralcore/nn/sequential.h"
#include "neuralcore/nn/dropout.h"
#include "neuralcore/nn/loss.h"

using namespace neuralcore;

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL at line " << __LINE__ << ": " \
                  << (a) << " != " << (b) << std::endl; return 1; \
    }} while(0)

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL at line " << __LINE__ << std::endl; return 1; \
    }} while(0)

int main() {
    std::cout << "=== NN Module Tests ===" << std::endl;

    // Test: Linear layer
    {
        nn::Linear layer(3, 2);
        auto input = Variable::create(Tensor({1, 3}, {1, 2, 3}), false);
        auto output = layer.forward(input);
        ASSERT_TRUE(output->data.shape()[0] == 1);
        ASSERT_TRUE(output->data.shape()[1] == 2);
        ASSERT_TRUE(layer.parameters().size() == 2); // weight + bias
        std::cout << "  [PASS] Linear layer" << std::endl;
    }

    // Test: Activations
    {
        nn::ReLU relu;
        auto input = Variable::create(Tensor({3}, {-1.0f, 0.0f, 2.0f}), true);
        auto output = relu.forward(input);
        ASSERT_NEAR(output->data.at(0), 0.0f, 1e-6);
        ASSERT_NEAR(output->data.at(1), 0.0f, 1e-6);
        ASSERT_NEAR(output->data.at(2), 2.0f, 1e-6);

        nn::Sigmoid sig;
        auto out2 = sig.forward(Variable::create(Tensor({1}, {0.0f}), false));
        ASSERT_NEAR(out2->data.at(0), 0.5f, 1e-5);

        std::cout << "  [PASS] Activations" << std::endl;
    }

    // Test: Sequential
    {
        auto model = nn::Sequential({
            std::make_shared<nn::Linear>(4, 8),
            std::make_shared<nn::ReLU>(),
            std::make_shared<nn::Linear>(8, 2)
        });

        auto input = Variable::create(Tensor::rand({2, 4}), false);
        auto output = model.forward(input);
        ASSERT_TRUE(output->data.shape()[0] == 2);
        ASSERT_TRUE(output->data.shape()[1] == 2);
        ASSERT_TRUE(model.parameters().size() == 4); // 2 weights + 2 biases
        std::cout << "  [PASS] Sequential" << std::endl;
    }

    // Test: Dropout
    {
        nn::Dropout dropout(0.5f);
        auto input = Variable::create(Tensor::ones({100}), false);

        // In training mode, some values should be zero
        dropout.train();
        auto out_train = dropout.forward(input);

        // In eval mode, all values should pass through
        dropout.eval();
        auto out_eval = dropout.forward(input);
        ASSERT_NEAR(out_eval->data.sum_all(), 100.0f, 1e-5);

        std::cout << "  [PASS] Dropout" << std::endl;
    }

    // Test: MSE Loss
    {
        nn::MSELoss mse;
        auto pred = Variable::create(Tensor({3}, {1, 2, 3}), true);
        auto target = Variable::create(Tensor({3}, {1, 2, 3}), false);
        auto loss = mse(pred, target);
        ASSERT_NEAR(loss->data.item(), 0.0f, 1e-5);

        auto pred2 = Variable::create(Tensor({3}, {2, 3, 4}), true);
        auto loss2 = mse(pred2, target);
        ASSERT_NEAR(loss2->data.item(), 1.0f, 1e-5); // mean((1)^2) = 1

        std::cout << "  [PASS] MSE Loss" << std::endl;
    }

    // Test: BCE Loss
    {
        nn::BCELoss bce;
        auto pred = Variable::create(Tensor({2}, {0.5f, 0.5f}), true);
        auto target = Variable::create(Tensor({2}, {1.0f, 0.0f}), false);
        auto loss = bce(pred, target);
        // Should be approximately -mean(1*log(0.5) + 0*log(0.5)) = log(2) ≈ 0.693
        ASSERT_NEAR(loss->data.item(), std::log(2.0f), 0.01f);
        std::cout << "  [PASS] BCE Loss" << std::endl;
    }

    // Test: backward through a network
    {
        nn::Linear layer(2, 1);
        nn::MSELoss mse;

        auto input = Variable::create(Tensor({1, 2}, {1, 2}), false);
        auto target = Variable::create(Tensor({1, 1}, {1}), false);

        auto output = layer.forward(input);
        auto loss = mse(output, target);
        loss->backward();

        // Gradients should be non-zero
        bool has_grad = false;
        for (auto& p : layer.parameters()) {
            if (!p->grad.empty() && p->grad.sum_all() != 0.0f) {
                has_grad = true;
                break;
            }
        }
        ASSERT_TRUE(has_grad);
        std::cout << "  [PASS] Backward through network" << std::endl;
    }

    std::cout << "\nAll NN module tests passed!" << std::endl;
    return 0;
}
