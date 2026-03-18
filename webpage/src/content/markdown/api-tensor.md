# Tensor API Reference

The `Tensor` module is the foundation of NeuralCore. It defines storage, shape semantics, and vectorized math behavior.

## Constructors

Use static constructors for most workflows:

| Function | Description | Example |
| --- | --- | --- |
| `Tensor::zeros(shape)` | Allocates a zero-initialized tensor | `Tensor::zeros({32, 128})` |
| `Tensor::ones(shape)` | Allocates a one-initialized tensor | `Tensor::ones({1, 10})` |
| `Tensor::randn(shape)` | Samples from normal distribution | `Tensor::randn({64, 64})` |

## Shape Operations

### `reshape`

Returns a view-like tensor with compatible element count.

### `transpose`

Swaps dimensions for matrix-style operators and batch-major transforms.

## Indexing and Slicing

Use index helpers to slice rows, columns, and spans.

```cpp
Tensor firstRow = x.index({0});
Tensor leftHalf = x.index({Slice(), Slice(0, x.size(1) / 2)});
```

## Arithmetic and Reductions

Elementwise arithmetic is broadcast-aware. Reduction ops include `sum`, `mean`, `max`, and `argmax`.

## Best Practices

- Validate shapes at module boundaries.
- Keep batch dimension explicit.
- Prefer vectorized operations over scalar loops.
