# Optim API Reference

The `optim` namespace contains optimization algorithms and parameter-group configuration.

## Optimizer Lifecycle

1. Construct optimizer with model parameters.
2. Call `zero_grad()` before each backward pass.
3. Call `step()` after gradients are available.

## Built-in Optimizers

| Optimizer | Typical Use | Notes |
| --- | --- | --- |
| `SGD` | Baseline and large-batch training | Supports momentum |
| `Adam` | Default choice for many workloads | Adaptive moments |
| `RMSprop` | Non-stationary settings | Exponential avg of squared gradients |

## Parameter Groups

Create separate parameter groups when layers need distinct learning rates or weight decay.

## Example

```cpp
optim::Adam optimizer(model.parameters(), 1e-3);

for (int step = 0; step < maxSteps; ++step) {
  optimizer.zero_grad();
  Tensor loss = criterion(model.forward(x), y);
  loss.backward();
  optimizer.step();
}
```

## Tuning Checklist

- Start with conservative learning rates.
- Track moving-average loss, not only per-step noise.
- Use gradient clipping when unstable.
