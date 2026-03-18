# Tutorial: Binary Classification with MLP

This tutorial walks through a full binary classification pipeline.

## Goal

Train an MLP that predicts class labels from 2D features.

## Step 1: Build the Model

```cpp
nn::Sequential model({
  nn::Linear(2, 32),
  nn::ReLU(),
  nn::Linear(32, 1),
  nn::Sigmoid()
});
```

## Step 2: Define Training Components

Use BCE loss with Adam optimizer.

## Step 3: Train Loop

For each epoch:

1. Forward pass
2. Compute loss
3. Backward pass
4. Optimizer step

## Step 4: Validation

Track validation loss and accuracy every epoch.

## Step 5: Save Best Model

Checkpoint when validation improves.
