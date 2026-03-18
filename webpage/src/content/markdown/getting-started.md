# Getting Started

NeuralCore is a C++ deep learning framework focused on clarity, speed, and control.

## What You Will Build

By the end of this guide, you will:

- Configure and compile the framework
- Run the built-in tests
- Train a simple regression model
- Understand the minimal train loop structure

## Project Layout

Typical folders in this repository:

| Folder | Purpose |
| --- | --- |
| `include/` | Public headers and APIs |
| `src/` | Core implementation (tensor, autograd, nn, optim) |
| `tests/` | Unit and behavior tests |
| `examples/` | Runnable example programs |

## Install

Clone the repository and build in Debug mode.

```bash
cmake -S . -B build
cmake --build build --config Debug
```

If you use Visual Studio generators, target binaries are emitted in `build/Debug`.

## Validate the Build

Run tests before writing your own model code.

```bash
ctest --test-dir build -C Debug --output-on-failure
```

This gives you confidence that tensor math, autograd, and optimizer behavior are healthy.

## Your First Model

Create tensors, define a simple MLP, and optimize with Adam.

```cpp
#include <neuralcore/neuralcore.hpp>
using namespace nc;

int main() {
  Tensor x = Tensor::randn({64, 2});
  Tensor y = Tensor::randn({64, 1});
  nn::Sequential model({ nn::Linear(2, 32), nn::ReLU(), nn::Linear(32, 1) });
  optim::Adam optimizer(model.parameters(), 1e-3);

  for (int step = 0; step < 1000; ++step) {
    Tensor pred = model.forward(x);
    Tensor loss = nn::mse_loss(pred, y);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
  }
}
```

## Why This Loop Works

Each iteration follows the same four-step pattern:

1. Forward pass computes predictions.
2. Loss computes objective value.
3. Backward computes gradients for each parameter.
4. Optimizer applies gradient-based updates.

This pattern scales from toy examples to large models with validation and checkpointing.

## Common Setup Mistakes

- Building and running tests from different output folders
- Forgetting to call `optimizer.zero_grad()` every step
- Mixing tensors with incompatible shapes in loss functions
- Treating this debug build as a performance benchmark

## Suggested Next Reading

- Tensor Fundamentals for shape and indexing behavior
- Autograd Deep Dive for gradient internals
- Training at Scale for robust production loops

