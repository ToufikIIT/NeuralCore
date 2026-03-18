# Autograd API Reference

Autograd records operation graphs and executes reverse-mode differentiation.

## Gradient Control

### `requires_grad`

Enable gradient tracking for trainable tensors.

### `detach`

Creates a tensor that shares values but does not keep graph connectivity.

## Backpropagation APIs

### `backward`

Computes gradients from scalar losses or from provided upstream gradients.

```cpp
Tensor loss = criterion(pred, target);
loss.backward();
```

### `grad`

Accesses accumulated gradient buffer of a parameter tensor.

## Graph Lifetime

The graph is rebuilt each forward pass by default. Keep references minimal and avoid accidental retention.

## Debugging Tips

- Check for `NaN` in gradients after `backward`.
- Monitor gradient norms across layers.
- Ensure detached tensors are intentional.
