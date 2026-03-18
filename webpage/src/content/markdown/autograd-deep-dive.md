# Autograd Deep Dive

Autograd in NeuralCore tracks tensor operations in a dynamic computation graph.

## Why Dynamic Graphs

Dynamic graph execution makes model logic easier to write and debug:

- Graph structure follows runtime control flow
- You can inspect intermediates during normal execution
- Error localization is often more straightforward

## Graph Construction

When gradient tracking is enabled, each operation records its parents and backward rule.

### Internal Perspective

Every differentiable op contributes:

- Output tensor value
- Backward closure/function
- References to parent nodes

## Backward Pass

Calling `loss.backward()` traverses the graph in reverse topological order.

During traversal, upstream gradients are transformed by local Jacobian logic and propagated to parent tensors.

### Gradient Accumulation

Gradients are accumulated in parameter buffers, so reset before each optimization step.

```cpp
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

## Retained Graphs and Memory

Retaining graph references can increase memory use. Avoid storing large computation histories unless required.

## Numerical Stability

Watch for unstable regions in:

- Exponentials and logarithms
- Division by small magnitudes
- Piecewise operations near boundaries

Gradient clipping and careful loss scaling can help.

## Inspecting Gradients

Use parameter iteration to inspect gradient magnitudes and detect exploding updates.

```cpp
for (auto& p : model.parameters()) {
  std::cout << p.grad().abs().mean().item<float>() << std::endl;
}
```

### Interpreting Values

- Near-zero gradients across many layers may indicate vanishing gradients.
- Very large gradients suggest instability or bad learning rates.
- Sudden spikes often correlate with data outliers or bad batches.

## Common Pitfalls

- Forgetting `zero_grad()` each step.
- Performing in-place ops that break graph history.
- Mixing detached tensors into differentiable paths.
- Calling backward on non-scalar outputs without upstream gradients.
- Reusing stale tensors from prior iterations unintentionally.
