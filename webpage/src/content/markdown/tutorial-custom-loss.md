# Tutorial: Custom Loss Functions

Custom losses let you encode task-specific behavior beyond standard objectives.

## Use Case

Penalize underestimation more heavily than overestimation.

## Define the Loss

```cpp
Tensor asymmetric_loss(const Tensor& pred, const Tensor& target) {
  Tensor err = pred - target;
  Tensor over = err.clamp_min(0.0f);
  Tensor under = (-err).clamp_min(0.0f);
  return (0.7f * over + 1.3f * under).mean();
}
```

## Integrate into Training

Replace the standard criterion call with your custom function.

## Gradient Validation

Validate gradients with:

- Numerical checks on small tensors
- Monitoring gradient magnitude per layer

## Stability Tips

- Keep custom loss smooth when possible.
- Avoid non-differentiable branches unless deliberate.
