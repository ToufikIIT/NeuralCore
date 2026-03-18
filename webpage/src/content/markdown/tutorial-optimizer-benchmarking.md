# Tutorial: Optimizer Benchmarking

Compare optimizer behavior under fixed architecture and dataset conditions.

## Experiment Setup

- Same model weights initialization
- Same train/validation split
- Same number of updates

## Configurations

| Optimizer | Learning Rate | Momentum/Beta |
| --- | --- | --- |
| SGD | `1e-2` | momentum `0.9` |
| Adam | `1e-3` | betas `(0.9, 0.999)` |
| RMSprop | `1e-3` | alpha `0.99` |

## Metrics to Track

- Training loss
- Validation loss
- Validation accuracy
- Time per epoch

## Analysis

Inspect not only final metric values but also convergence speed and variance.

## Reporting Template

Summarize each optimizer with:

- Best validation score
- Epoch at best score
- Stability notes
