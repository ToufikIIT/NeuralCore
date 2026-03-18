# Tensor Fundamentals

NeuralCore tensors are strongly shape-aware and support familiar math primitives.

## Mental Model

Think of a tensor as:

- A contiguous or view-based block of numeric data
- A shape descriptor that defines dimension boundaries
- A set of operations that preserve or transform shape semantics

## Creating Tensors

Use random, zeros, and shape constructors.

```cpp
Tensor a = Tensor::zeros({32, 128});
Tensor b = Tensor::randn({32, 128});
Tensor c = Tensor::ones({32, 128});
```

For reproducible experiments, seed your random generator before calling stochastic constructors.

## Shape and Broadcast Rules

Elementwise operations follow broadcasting semantics from trailing dimensions.

### Example

`[64, 1] + [1, 32] -> [64, 32]`

### Rules of Thumb

- Compare dimensions from right to left.
- Dimensions are compatible when equal or one side is `1`.
- Missing leading dimensions are treated as `1`.

When in doubt, print shapes before operations to avoid silent logical errors.

## Indexing and Slicing

Use row and column slicing to inspect minibatches and feature subsets.

```cpp
Tensor row = x.index({0});
Tensor firstTwoCols = x.index({Slice(), Slice(0, 2)});
```

Indexing is often the first place bugs show up. Keep a small debug batch for shape inspection.

### Performance Note

Prefer slicing and vectorized operations over per-element loops.

## Reductions

Common reductions include sum, mean, max, and norm.

```cpp
Tensor logits = model.forward(x);
Tensor avg = logits.mean();
Tensor energy = logits.pow(2).sum();
```

## Practical Debug Checklist

- Verify batch dimension remains stable across pipeline stages.
- Confirm reduction axes are what you intended.
- Check for unintended broadcasting in loss computations.
- Inspect min/max statistics before and after normalization.

## Recommended Conventions

- Use explicit names like `batchSize`, `featureDim`, and `numClasses`.
- Keep tensor shape comments near non-obvious transforms.
- Centralize shape-changing logic when possible.
