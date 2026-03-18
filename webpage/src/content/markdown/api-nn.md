# NN API Reference

The `nn` namespace defines modules, layers, activation functions, and losses.

## Core Abstractions

### `Module`

Base class for trainable components. Exposes `forward`, `parameters`, and state handling.

### `Sequential`

Ordered module container for fast prototyping.

```cpp
nn::Sequential model({
  nn::Linear(2, 64),
  nn::ReLU(),
  nn::Linear(64, 1)
});
```

## Layers

- `Linear(in, out)`
- `Dropout(p)`
- `BatchNorm(...)` (when available in your build)

## Activations

Common activations include `ReLU`, `Tanh`, and `Sigmoid`.

## Losses

- `mse_loss(pred, target)`
- `binary_cross_entropy(pred, target)`
- `cross_entropy(logits, labels)`

## Best Practices

- Keep forward methods side-effect free.
- Isolate preprocessing outside modules.
- Use explicit module names for readability.
