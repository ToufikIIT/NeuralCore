# NeuralCore

A lightweight deep learning framework built from scratch in **C++17** — no external dependencies, just pure C++ and the Standard Library.

NeuralCore provides a PyTorch-inspired API with tensors, automatic differentiation, neural network modules, optimizers, and data utilities. It is designed for educational purposes and small-scale experiments on CPU.

---

## Why NeuralCore?

NeuralCore isn't trying to replace PyTorch or TensorFlow — those frameworks have thousands of engineers, GPU backends, and massive ecosystems. NeuralCore exists for different reasons, and in those areas, it genuinely shines.

### What Makes It Unique

- **Zero dependencies** — PyTorch needs Python, CUDA, cuDNN, MKL, and more. NeuralCore compiles with just a C++ compiler and CMake. Nothing else. That's exceptionally rare for a framework with autograd, optimizers, and a data pipeline.

- **Fully transparent** — Every line of the autograd engine, every optimizer step, every backward pass is code you can read and understand. Production frameworks have millions of lines of opaque code. NeuralCore has ~3,000 lines total.

- **Embeddable** — Because it's a single static library (`.lib` / `.a`) with no external dependencies, it can be dropped into any C++ application — games, robotics, IoT devices, embedded systems — without dragging in Python or a massive runtime.

- **Educationally complete** — Most "build ML from scratch" tutorials stop at a basic neural net with manual backprop. NeuralCore has a full dynamic computation graph, N-dimensional broadcasting, 3 production-grade optimizers, a DataLoader with batching and shuffling, model serialization, and proper module abstractions. It's a real framework, not a toy.

- **Builds in seconds** — No waiting 30+ minutes for a massive dependency tree. Configure and build in under 10 seconds.

### Honest Comparison

| | NeuralCore | PyTorch / TensorFlow |
|---|---|---|
| GPU support | ❌ | ✅ |
| Ecosystem & community | ❌ | ✅ |
| Performance at scale | ❌ | ✅ |
| Pre-trained models | ❌ | ✅ |
| Zero external dependencies | ✅ | ❌ |
| Full code transparency | ✅ | ❌ |
| Embeddable in any C++ app | ✅ | Difficult |
| Build time | ~5 seconds | Minutes to hours |
| Learning & teaching tool | ✅ | ❌ |
| Single-file debuggable | ✅ | ❌ |

### The Real Value

The true differentiator is **understanding**. If someone asks "how does automatic differentiation work?", you don't point to a research paper — you point to your `autograd.cpp`. If someone asks "what does Adam actually do each step?", you open `adam.cpp` and read 30 lines of clear C++.

That level of understanding is worth more than any feature list. Frameworks are tools; understanding is power.

---

## Table of Contents

- [Why NeuralCore?](#why-neuralcore)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building](#building)
  - [Running Tests](#running-tests)
  - [Running Examples](#running-examples)
- [API Reference](#api-reference)
  - [Tensor](#tensor)
  - [Autograd (Variable)](#autograd-variable)
  - [Neural Network Modules](#neural-network-modules)
  - [Optimizers](#optimizers)
  - [Data Utilities](#data-utilities)
  - [Serialization](#serialization)
- [Quick Start Example — XOR](#quick-start-example--xor)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

| Component | Details |
|---|---|
| **Tensor** | N-dimensional array with broadcasting, slicing, reshaping, matrix multiplication, reductions, and elementwise math |
| **Autograd** | Reverse-mode automatic differentiation with a dynamic computation graph (rebuilt each forward pass) |
| **NN Modules** | `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `Dropout`, `Sequential` |
| **Loss Functions** | `MSELoss`, `BCELoss`, `CrossEntropyLoss` |
| **Optimizers** | `SGD` (with momentum), `Adam`, `RMSProp` |
| **Data** | `Dataset` / `TensorDataset`, `DataLoader` with batching & shuffling |
| **Serialization** | Binary save/load for model parameters |

---

## Architecture

```
neuralcore/
├── CMakeLists.txt            # Build configuration
├── include/neuralcore/       # Public headers
│   ├── tensor.h              # Tensor class + Storage
│   ├── autograd.h            # Variable, GradFunction, NoGradGuard
│   ├── ops.h                 # Backward function classes
│   ├── serialization.h       # Save/load parameters
│   ├── nn/
│   │   ├── module.h          # Base Module class
│   │   ├── linear.h          # Fully connected layer
│   │   ├── activation.h      # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── sequential.h      # Sequential container
│   │   ├── dropout.h         # Dropout regularization
│   │   └── loss.h            # MSE, BCE, CrossEntropy
│   ├── optim/
│   │   ├── optimizer.h       # Base Optimizer class
│   │   ├── sgd.h             # SGD + momentum
│   │   ├── adam.h            # Adam optimizer
│   │   └── rmsprop.h         # RMSProp optimizer
│   └── data/
│       ├── dataset.h         # Dataset / TensorDataset
│       └── dataloader.h      # DataLoader with batching
├── src/                      # Implementation files (mirrors include/)
├── tests/                    # Unit tests
│   ├── test_tensor.cpp
│   ├── test_autograd.cpp
│   ├── test_nn.cpp
│   └── test_optim.cpp
└── examples/
    └── xor.cpp               # XOR classification example
```

---

## Getting Started

### Prerequisites

- **C++17** compatible compiler (MSVC 2022, GCC 7+, Clang 5+)
- **CMake** 3.14 or higher

### Building

```bash
# Configure
cmake -S . -B build

# Build
cmake --build build
```

On Windows with Visual Studio:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

Or run individual tests:

```bash
./build/Debug/test_tensor
./build/Debug/test_autograd
./build/Debug/test_nn
./build/Debug/test_optim
```

### Running Examples

```bash
./build/Debug/xor
```

---

## API Reference

### Tensor

The `Tensor` class is the foundation — an N-dimensional array of `float32` values backed by reference-counted `Storage`.

```cpp
#include "neuralcore/tensor.h"
using namespace neuralcore;
```

#### Construction

```cpp
// Empty tensor
Tensor a;

// Zeros of a given shape
Tensor b({3, 4});

// Filled with a constant
Tensor c({3, 4}, 1.0f);

// From explicit data
Tensor d({2, 3}, {1, 2, 3, 4, 5, 6});
```

#### Factory Functions

```cpp
Tensor::zeros({3, 3});         // All zeros
Tensor::ones({2, 4});          // All ones
Tensor::full({2, 2}, 3.14f);   // Filled with 3.14
Tensor::rand({3, 3});          // Uniform [0, 1)
Tensor::randn({3, 3});         // Normal (mean=0, std=1)
Tensor::arange(0, 10, 2);     // [0, 2, 4, 6, 8]
Tensor::eye(4);                // 4x4 identity matrix
```

#### Element Access

```cpp
Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});

float val = t.at(0, 1);       // Row 0, Col 1 → 2.0
t.at(1, 2) = 99.0f;           // Mutate in-place

float scalar = t.item();      // Only valid for scalar tensors
```

#### Shape Manipulation

```cpp
Tensor t({2, 6}, ...);

t.reshape({3, 4});             // New shape, same data
t.view({3, 4});                // Same as reshape (contiguous only)
t.transpose(0, 1);             // Swap dims 0 and 1
t.t();                         // Shorthand for 2D transpose
t.squeeze();                   // Remove size-1 dimensions
t.unsqueeze(0);                // Add dim at position 0
t.flatten();                   // Flatten to 1D
t.expand({4, 3, 3});           // Broadcast-expand to larger shape
```

#### Arithmetic (with broadcasting)

```cpp
Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
Tensor b({3}, {10, 20, 30});

Tensor c = a + b;   // Broadcasting: (2,3) + (3,) → (2,3)
Tensor d = a * 2.0f;
Tensor e = 1.0f - a;
a += b;              // In-place
```

#### Math Functions

```cpp
t.exp();
t.log();
t.pow(2.0f);
t.sqrt();
t.abs();
t.clamp(0.0f, 1.0f);
```

#### Reductions

```cpp
t.sum();                 // Scalar sum (all elements)
t.sum(0);                // Sum along dim 0
t.sum(1, true);          // Sum along dim 1, keep dim
t.mean();
t.max(0);
t.min(1);
t.argmax(1);

// Scalar reductions
float s = t.sum_all();
float m = t.mean_all();
```

#### Matrix Multiplication

```cpp
Tensor a({2, 3}, ...);
Tensor b({3, 4}, ...);
Tensor c = a.matmul(b);  // → shape (2, 4)
```

#### Comparisons

```cpp
Tensor mask = t > 0.5f;
Tensor mask2 = t == 1.0f;
```

#### In-place Utilities

```cpp
t.fill_(0.0f);
t.zero_();
t.uniform_(0.0f, 1.0f);
t.normal_(0.0f, 0.01f);
```

#### Printing

```cpp
std::cout << t << std::endl;
std::string s = t.to_string();
```

---

### Autograd (Variable)

The `Variable` class wraps a `Tensor` and tracks operations for automatic differentiation. When you perform math on Variables, a computation graph is built automatically. Calling `.backward()` traverses this graph in reverse to compute gradients.

```cpp
#include "neuralcore/autograd.h"
using namespace neuralcore;
```

#### Creating Variables

```cpp
auto x = Variable::create(Tensor({2, 2}, {1, 2, 3, 4}), true);  // requires_grad=true
auto y = Variable::create(Tensor({2, 2}, {5, 6, 7, 8}), false);

// Shorthand
auto z = var(Tensor::ones({3, 3}), true);
```

#### Operations

Variables support the same operations as Tensors, but they build a computation graph:

```cpp
auto a = Variable::create(Tensor({2, 2}, {1, 2, 3, 4}), true);
auto b = Variable::create(Tensor({2, 2}, {5, 6, 7, 8}), true);

auto c = a + b;          // AddBackward
auto d = c * a;          // MulBackward
auto e = d->sum();       // SumBackward
auto f = a->matmul(b);   // MatmulBackward
auto g = a->relu();      // ReLUBackward
auto h = a->sigmoid();   // SigmoidBackward
auto i = a->tanh();      // TanhBackward
auto j = a->exp();       // ExpBackward
auto k = a->log();       // LogBackward
```

#### Backward Pass

```cpp
auto x = Variable::create(Tensor({2, 2}, {1, 2, 3, 4}), true);
auto y = (x * x)->sum();   // y = sum(x^2)

y->backward();              // Compute gradients

// x->grad now contains dy/dx = 2*x
std::cout << x->grad << std::endl;  // [2, 4, 6, 8]
```

#### No-Grad Context

Disable gradient tracking for inference or parameter updates:

```cpp
{
    NoGradGuard guard;
    auto out = model.forward(input);
    // No graph is built
}
```

#### Utility

```cpp
x->zero_grad();          // Reset gradient to zero
auto d = x->detach();    // Create a copy without grad tracking
```

---

### Neural Network Modules

All modules inherit from `nn::Module` and implement `forward()`.

```cpp
#include "neuralcore/nn/module.h"
#include "neuralcore/nn/linear.h"
#include "neuralcore/nn/activation.h"
#include "neuralcore/nn/sequential.h"
#include "neuralcore/nn/dropout.h"
#include "neuralcore/nn/loss.h"
using namespace neuralcore;
```

#### Module Base Class

```cpp
class Module {
    virtual VariablePtr forward(const VariablePtr& input) = 0;
    virtual std::vector<VariablePtr> parameters();
    void train();           // Set training mode
    void eval();            // Set evaluation mode
    bool is_training();
    void zero_grad();       // Zero all parameter gradients
};
```

#### Linear Layer

Fully connected layer: `y = x @ W^T + b`, with Xavier/Glorot initialization.

```cpp
auto fc = std::make_shared<nn::Linear>(784, 128);        // in=784, out=128
auto fc_no_bias = std::make_shared<nn::Linear>(128, 10, false);  // no bias

auto output = fc->forward(input);

// Access weights directly
fc->weight;  // VariablePtr, shape (out, in)
fc->bias;    // VariablePtr, shape (out,)
```

#### Activations

```cpp
auto relu    = std::make_shared<nn::ReLU>();
auto sigmoid = std::make_shared<nn::Sigmoid>();
auto tanh    = std::make_shared<nn::Tanh>();
auto softmax = std::make_shared<nn::Softmax>(/*dim=*/ -1);

auto output = relu->forward(input);
```

#### Dropout

Randomly zeros elements with probability `p` during training. Disabled during eval.

```cpp
auto dropout = std::make_shared<nn::Dropout>(0.5f);

model.train();
auto out_train = dropout->forward(input);  // Dropout active

model.eval();
auto out_eval = dropout->forward(input);   // Dropout disabled
```

#### Sequential

Chain modules together:

```cpp
nn::Sequential model({
    std::make_shared<nn::Linear>(2, 8),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Linear>(8, 1),
    std::make_shared<nn::Sigmoid>()
});

auto output = model.forward(input);
auto params = model.parameters();  // All trainable parameters
```

You can also build incrementally:

```cpp
nn::Sequential model;
model.add(std::make_shared<nn::Linear>(2, 8));
model.add(std::make_shared<nn::ReLU>());
```

#### Loss Functions

```cpp
nn::MSELoss mse;
nn::BCELoss bce;
nn::CrossEntropyLoss ce;

// MSE: Mean Squared Error — for regression
auto loss = mse(prediction, target);

// BCE: Binary Cross-Entropy — for binary classification
// (predictions should be passed through sigmoid first)
auto loss = bce(prediction, target);

// CrossEntropy: for multi-class classification
// (logits in, applies log-softmax internally)
auto loss = ce(logits, one_hot_target);
```

All loss functions return a `VariablePtr` that you can call `->backward()` on.

---

### Optimizers

```cpp
#include "neuralcore/optim/sgd.h"
#include "neuralcore/optim/adam.h"
#include "neuralcore/optim/rmsprop.h"
using namespace neuralcore;
```

#### SGD (Stochastic Gradient Descent)

```cpp
auto params = model.parameters();

optim::SGD optimizer(params,
    /*lr=*/ 0.01f,
    /*momentum=*/ 0.9f,
    /*weight_decay=*/ 1e-4f);
```

#### Adam

```cpp
optim::Adam optimizer(params,
    /*lr=*/ 0.001f,
    /*beta1=*/ 0.9f,
    /*beta2=*/ 0.999f,
    /*eps=*/ 1e-8f,
    /*weight_decay=*/ 0.0f);
```

#### RMSProp

```cpp
optim::RMSProp optimizer(params,
    /*lr=*/ 0.01f,
    /*alpha=*/ 0.99f,
    /*eps=*/ 1e-8f,
    /*weight_decay=*/ 0.0f);
```

#### Training Loop Pattern

```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto output = model.forward(input);
    auto loss = criterion(output, target);

    optimizer.zero_grad();   // Clear previous gradients
    loss->backward();        // Compute gradients
    optimizer.step();        // Update parameters
}
```

#### Adjusting Learning Rate

```cpp
optimizer.set_learning_rate(0.001f);
float lr = optimizer.learning_rate();
```

---

### Data Utilities

```cpp
#include "neuralcore/data/dataset.h"
#include "neuralcore/data/dataloader.h"
using namespace neuralcore;
```

#### Dataset

Abstract base class for datasets:

```cpp
class Dataset {
    virtual std::pair<Tensor, Tensor> get(int index) const = 0;
    virtual int size() const = 0;
};
```

#### TensorDataset

Wraps two tensors (data + targets) as a dataset:

```cpp
Tensor x({100, 784}, ...);   // 100 samples, 784 features
Tensor y({100, 10}, ...);    // 100 labels (one-hot)

data::TensorDataset dataset(x, y);

auto [sample, label] = dataset.get(0);  // Get first sample
int n = dataset.size();                 // 100
```

#### DataLoader

Iterates over a dataset in batches, with optional shuffling:

```cpp
data::DataLoader loader(dataset,
    /*batch_size=*/ 32,
    /*shuffle=*/ true);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    loader.reset();
    while (loader.has_next()) {
        auto batch = loader.next();
        // batch.data  → Tensor of shape (batch_size, ...)
        // batch.target → Tensor of shape (batch_size, ...)

        auto x = Variable::create(batch.data, false);
        auto y = Variable::create(batch.target, false);
        // ... forward, loss, backward, step
    }
}

int num_batches = loader.num_batches();
```

---

### Serialization

Save and load model parameters in binary format:

```cpp
#include "neuralcore/serialization.h"
using namespace neuralcore;

auto params = model.parameters();

// Save
save_parameters("model.bin", params);

// Load (params must have same shapes)
load_parameters("model.bin", params);
```

---

## Quick Start Example — XOR

A complete example training a neural network to learn the XOR function:

```cpp
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
    // XOR dataset
    Tensor x_data({4, 2}, {0, 0,  0, 1,  1, 0,  1, 1});
    Tensor y_data({4, 1}, {0, 1, 1, 0});

    auto x = Variable::create(x_data, false);
    auto y = Variable::create(y_data, false);

    // Network: 2 → 8 → 1
    nn::Sequential model({
        std::make_shared<nn::Linear>(2, 8),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(8, 1),
        std::make_shared<nn::Sigmoid>()
    });

    nn::MSELoss criterion;
    optim::Adam optimizer(model.parameters(), 0.01f);

    // Train
    for (int epoch = 0; epoch < 2000; ++epoch) {
        auto output = model.forward(x);
        auto loss = criterion(output, y);

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();

        if ((epoch + 1) % 500 == 0)
            std::cout << "Epoch " << epoch + 1
                      << " Loss: " << loss->data.item() << std::endl;
    }

    // Test
    auto output = model.forward(x);
    for (int i = 0; i < 4; ++i)
        std::cout << x_data.at(i, 0) << " XOR " << x_data.at(i, 1)
                  << " = " << output->data.at(i, 0) << std::endl;

    return 0;
}
```

**Expected output:**

```
Epoch 500 Loss: 0.0312
Epoch 1000 Loss: 0.0021
Epoch 1500 Loss: 0.0008
Epoch 2000 Loss: 0.0004
0 XOR 0 = 0.018
0 XOR 1 = 0.981
1 XOR 0 = 0.979
1 XOR 1 = 0.022
```

---

## Design Decisions

### Why a Dynamic Computation Graph?

NeuralCore uses a **dynamic computation graph** (like PyTorch) rather than a static graph (like TensorFlow 1.x). The graph is rebuilt from scratch on every forward pass. This means:

- **Flexibility** — You can use standard C++ control flow (`if`, `for`, `while`) in your model and the graph adapts automatically.
- **Debugging** — Easier to debug because the graph corresponds directly to your code.
- **Simplicity** — No separate "graph compilation" step.

The tradeoff is that static graphs can be optimized more aggressively, but for an educational framework, flexibility and clarity win.

### Why float32 Only?

Supporting a single data type (`float`) keeps the code simple and avoids template complexity. `float32` is the standard precision for neural network training and is sufficient for all common use cases.

### Why Reference-Counted Storage?

`Storage` uses `std::shared_ptr` internally so that operations like `transpose()`, `reshape()`, and `expand()` can create **views** of the same underlying data without copying. This is both memory-efficient and matches how real frameworks work.

### Why Row-Major (C-Contiguous)?

Row-major layout matches C/C++ array conventions and makes integration with other C++ code natural. The `strides_` array enables non-contiguous views (e.g., after `transpose()`).

### Why Xavier/Glorot Initialization?

Linear layers use Xavier uniform initialization: `limit = sqrt(6 / (fan_in + fan_out))`. This keeps the variance of activations roughly constant across layers, preventing vanishing or exploding gradients during the initial forward pass.

### Why No External Dependencies?

The framework is built using only the C++17 Standard Library. This means:

- Zero setup friction — just CMake and a compiler
- Maximum portability across platforms
- Full understanding of every component (no black boxes)

---

## Project Structure

| Directory | Contents |
|---|---|
| `include/neuralcore/` | All public headers — this is the API surface |
| `src/` | Implementation files (one `.cpp` per header) |
| `tests/` | Unit tests for tensor, autograd, nn, optimizers |
| `examples/` | Complete training examples |
| `build/` | CMake build output (generated) |

### Build Targets

| Target | Type | Description |
|---|---|---|
| `neuralcore` | Static library | Core library (`.lib` / `.a`) |
| `test_tensor` | Executable | Tensor unit tests |
| `test_autograd` | Executable | Autograd unit tests (includes numerical gradient checks) |
| `test_nn` | Executable | Neural network module tests |
| `test_optim` | Executable | Optimizer tests |
| `xor` | Executable | XOR training example |

---

## License

This project is provided as-is for educational purposes.
