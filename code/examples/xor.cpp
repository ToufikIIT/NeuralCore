// NeuralCore XOR Example
// Trains a 2-layer neural network to learn the XOR function

#include <iostream>
#include "neuralcore/tensor.h"
#include "neuralcore/autograd.h"
#include "neuralcore/nn/linear.h"
#include "neuralcore/nn/activation.h"
#include "neuralcore/nn/sequential.h"
#include "neuralcore/nn/loss.h"
#include "neuralcore/optim/adam.h"

using namespace neuralcore;

int main() {
    std::cout << "=== NeuralCore XOR Example ===" << std::endl;

    // XOR dataset
    Tensor x_data({4, 2}, {0, 0,  0, 1,  1, 0,  1, 1});
    Tensor y_data({4, 1}, {0, 1, 1, 0});

    auto x = Variable::create(x_data, false);
    auto y = Variable::create(y_data, false);

    // Build network: 2 -> 8 -> 1 with ReLU activation
    auto layer1 = std::make_shared<nn::Linear>(2, 8);
    auto relu = std::make_shared<nn::ReLU>();
    auto layer2 = std::make_shared<nn::Linear>(8, 1);
    auto sigmoid = std::make_shared<nn::Sigmoid>();

    nn::Sequential model({layer1, relu, layer2, sigmoid});
    nn::MSELoss criterion;

    // Collect parameters and create optimizer
    auto params = model.parameters();
    optim::Adam optimizer(params, 0.01f);

    // Training loop
    int epochs = 2000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto output = model.forward(x);

        // Compute loss
        auto loss = criterion(output, y);

        // Backward pass
        optimizer.zero_grad();
        loss->backward();

        // Update weights
        optimizer.step();

        if ((epoch + 1) % 200 == 0) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " | Loss: " << loss->data.item() << std::endl;
        }
    }

    // Test
    std::cout << "\n=== Predictions ===" << std::endl;
    auto output = model.forward(x);
    for (int i = 0; i < 4; ++i) {
        float pred = output->data.at(i, 0);
        float target = y_data.at(i, 0);
        std::cout << "Input: (" << x_data.at(i, 0) << ", " << x_data.at(i, 1)
                  << ") => Predicted: " << pred
                  << " | Target: " << target << std::endl;
    }

    // Check if XOR was learned (predictions close to targets)
    bool success = true;
    for (int i = 0; i < 4; ++i) {
        float pred = output->data.at(i, 0);
        float target = y_data.at(i, 0);
        if (std::abs(pred - target) > 0.2f) {
            success = false;
            break;
        }
    }

    std::cout << "\nXOR learned: " << (success ? "YES" : "NO") << std::endl;
    return success ? 0 : 1;
}
