# SciREX Training

Simple, modular training utilities for Flax NNX models.

## Key Features

- **Simple API**: Minimal boilerplate for common training patterns.
- **JIT Compilation**: Automatic JIT compilation of train/eval steps.
- **Progress Tracking**: Built-in `loguru` logging for easy monitoring.
- **Flexible**: Compatible with any NNX model and custom loss functions.
- **Automatic History**: Built-in tracking of training and evaluation metrics.

## Quick Start: The Trainer Class

The `Trainer` class is the recommended way to train models in SciREX. It automates the training loop, evaluation, and history tracking.

```python
import jax.numpy as jnp
from flax import nnx as nn
import optax
from scirex.training import Trainer

# 1. Define your loss function
# Requirement: MUST return a tuple of (loss, logits)
def loss_fn(model, batch, rngs):
    x, y = batch
    logits = model(x)
    loss = jnp.mean((logits - y) ** 2)
    return loss, logits  # Always return both!

# 2. Setup metrics for evaluation
eval_metrics = nn.MultiMetric(
    loss=nn.metrics.Average("loss"),
    accuracy=nn.metrics.Accuracy(),
)

# 3. Create the trainer
trainer = Trainer(
    model=model,
    optimizer=nn.Optimizer(model, optax.adam(1e-3)),
    loss_fn=loss_fn,
    eval_metrics=eval_metrics,
)

# 4. Train and visualize results
trainer.train(train_loader, n_epochs=10, eval_loader=val_loader)
trainer.visualize()
```

### Important: Loss Function Contract
The `Trainer` class internal logic uses `nn.value_and_grad(loss_fn, has_aux=True)`. This implies that your `loss_fn` **must** return a tuple: `(loss_value, auxiliary_data)`. By convention in SciREX, `auxiliary_data` should be the model's `logits`, which are then used by the `eval_metrics` during evaluation.

## API Reference

### Trainer Class
`Trainer(model, optimizer, loss_fn, eval_metrics=None, gradient_clip=None, rngs=None)`

- `train(train_loader, n_epochs, eval_loader=None, eval_freq=1)`: Executes the training loop and returns the `history` dictionary.
- `eval(eval_loader)`: Performs a single evaluation pass with progress monitoring.
- `visualize(figsize=(12, 4), save_path=None)`: Plots training loss and evaluation metrics.
- `history`: Dictionary containing tracked metrics (e.g., `train_loss`, `eval_accuracy`, `total_training_time`).

## File Structure

```
.
├── trainer.py    # Core Trainer class for managing the optimization loop and progress tracking.
├── utils.py      # Utility functions for training and data processing.
├── __init__.py   # Package initialization and component exports.
└── README.md
```

## Examples
- [Text Classification](../../examples/simple/text_classification.py): Sentiment analysis on IMDB.
- [Machine Translation](../../examples/simple/machine_translation.py): Spanish to English translation.
- [Diffusion Toy](../../examples/simple/diffusion_toy.py): 2D point cloud generation.
