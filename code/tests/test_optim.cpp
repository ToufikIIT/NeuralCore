// Test: Optimizers
#include <iostream>
#include <cmath>
#include "neuralcore/tensor.h"
#include "neuralcore/autograd.h"
#include "neuralcore/nn/linear.h"
#include "neuralcore/nn/loss.h"
#include "neuralcore/optim/sgd.h"
#include "neuralcore/optim/adam.h"
#include "neuralcore/optim/rmsprop.h"

using namespace neuralcore;

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL at line " << __LINE__ << std::endl; return 1; \
    }} while(0)

int main() {
    std::cout << "=== Optimizer Tests ===" << std::endl;

    auto run_opt_test = [](const std::string& name, auto& optimizer,
                            nn::Linear& layer, int steps) -> bool {
        nn::MSELoss mse;
        auto input = Variable::create(Tensor({4, 1}, {1, 2, 3, 4}), false);
        auto target = Variable::create(Tensor({4, 1}, {2, 4, 6, 8}), false);

        float initial_loss = 0;
        float final_loss = 0;

        for (int i = 0; i < steps; ++i) {
            auto output = layer.forward(input);
            auto loss = mse(output, target);

            if (i == 0) initial_loss = loss->data.item();
            if (i == steps - 1) final_loss = loss->data.item();

            optimizer.zero_grad();
            loss->backward();
            optimizer.step();
        }

        bool improved = final_loss < initial_loss;
        std::cout << "  " << name << ": " << initial_loss << " -> " << final_loss
                  << (improved ? " [PASS]" : " [FAIL]") << std::endl;
        return improved;
    };

    // Test SGD
    {
        nn::Linear layer(1, 1);
        auto params = layer.parameters();
        optim::SGD opt(params, 0.01f);
        ASSERT_TRUE(run_opt_test("SGD", opt, layer, 100));
    }

    // Test SGD with momentum
    {
        nn::Linear layer(1, 1);
        auto params = layer.parameters();
        optim::SGD opt(params, 0.01f, 0.9f);
        ASSERT_TRUE(run_opt_test("SGD+Momentum", opt, layer, 100));
    }

    // Test Adam
    {
        nn::Linear layer(1, 1);
        auto params = layer.parameters();
        optim::Adam opt(params, 0.01f);
        ASSERT_TRUE(run_opt_test("Adam", opt, layer, 100));
    }

    // Test RMSProp
    {
        nn::Linear layer(1, 1);
        auto params = layer.parameters();
        optim::RMSProp opt(params, 0.01f);
        ASSERT_TRUE(run_opt_test("RMSProp", opt, layer, 100));
    }

    std::cout << "\nAll optimizer tests passed!" << std::endl;
    return 0;
}
