# SciREX Data Module

The `scirex.data` module provides utilities for creating data loaders using the [Grain](https://github.com/google/grain) library.

## Features

- **Simple API**: Create dataloaders with a single function call
- **Flexible**: Support for labeled data, custom transformations, and more
- **Efficient**: Parallel data loading with configurable workers
- **Compatible**: Works with JAX arrays, NumPy arrays, and custom data sources

## Quick Start

### Basic Usage

```python
import jax.numpy as jnp
from scirex.data import create_dataloader

# Create some data
data = jnp.ones((1000, 28, 28, 1))

# Create dataloader
loader = create_dataloader(data, batch_size=32)

# Use in training
for batch in loader:
    print(batch.shape)  # (32, 28, 28, 1)
```

### With Labels

```python
data = jnp.ones((1000, 28, 28, 1))
labels = jnp.arange(1000)

loader = create_dataloader(data, batch_size=32, labels=labels)

for batch_data, batch_labels in loader:
    print(batch_data.shape, batch_labels.shape)  # (32, 28, 28, 1) (32,)
```

### With Custom Transformations

```python
import grain.python as grain
from scirex.data import create_dataloader, ArrayTransform

class NormalizeTransform(grain.MapTransform):
    def map(self, x):
        return (x - x.mean()) / x.std()

loader = create_dataloader(
    data,
    batch_size=32,
    custom_operations=[
        ArrayTransform(data),
        NormalizeTransform(),
    ]
)
```

## API Reference

### `create_dataloader`

```python
def create_dataloader(
    data: Any,
    batch_size: int,
    labels: Optional[Any] = None,
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 2,
    worker_buffer_size: int = 2,
    drop_remainder: bool = True,
    custom_operations: Optional[list[grain.Transformation]] = None,
) -> grain.DataLoader
```

Create a Grain DataLoader for array data.

**Parameters:**
- `data`: The data array to load (JAX or NumPy array)
- `batch_size`: Number of samples per batch
- `labels`: Optional labels array for supervised learning
- `shuffle`: Whether to shuffle the data (default: True)
- `seed`: Random seed for shuffling (default: 42)
- `worker_count`: Number of worker processes (default: 2)
- `worker_buffer_size`: Batches to buffer per worker (default: 2)
- `drop_remainder`: Drop incomplete last batch (default: True)
- `custom_operations`: Custom Grain transformations (default: None)

**Returns:**
- `grain.DataLoader`: Configured dataloader

### `ArrayTransform`

A Grain `MapTransform` that converts indices to array elements.

```python
class ArrayTransform(grain.MapTransform):
    def __init__(self, data: Any)
```

### `LabeledTransform`

A Grain `MapTransform` that returns (data, label) tuples.

```python
class LabeledTransform(grain.MapTransform):
    def __init__(self, data: Any, labels: Any)
```

### `ArrayDataSource`

A simple Grain `RandomAccessDataSource` wrapper for arrays.

```python
class ArrayDataSource(grain.RandomAccessDataSource):
    def __init__(self, data: Any)
```

## Examples

See the following examples for usage:
- [digit_generation.py](../../examples/simple/digit_generation.py) - Simple array loading
- [toy_diffusion.ipynb](../../examples/simple/toy_diffusion.ipynb) - Custom transformations

## Notes

- The dataloader uses `num_epochs=1` in the sampler, meaning it iterates through the data once per call
- For multi-epoch training, call the dataloader in a loop or recreate it for each epoch
- Custom operations are applied before batching
- Worker processes enable parallel data loading for better performance
