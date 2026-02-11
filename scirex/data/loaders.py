# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""Data loader utilities using Grain."""

from typing import Any, Optional

import grain.python as grain  # type: ignore[import-untyped]


class ArrayDataSource(grain.RandomAccessDataSource):  # type: ignore[no-any-unimported]
    """A simple data source for JAX arrays.

    Args:
        data: The data array to wrap. Can be a JAX array or numpy array.
    """

    def __init__(self, data: Any):
        self._data = data

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]

    def __len__(self) -> int:
        return len(self._data)


class ArrayTransform(grain.MapTransform):  # type: ignore[no-any-unimported]
    """Transform that converts indices to array elements.

    This is useful when using range-based data sources with Grain.

    Args:
        data: The data array to index into.
    """

    def __init__(self, data: Any):
        self._data = data

    def map(self, idx: int) -> Any:
        return self._data[idx]

    def __call__(self, ds: Any) -> Any:
        return ds.map(self)


class LabeledTransform(grain.MapTransform):  # type: ignore[no-any-unimported]
    """Transform that returns (data, label) tuples.

    Args:
        data: The data array.
        labels: The labels array.
    """

    def __init__(self, data: Any, labels: Any):
        self._data = data
        self._labels = labels

    def map(self, idx: int) -> tuple[Any, Any]:
        return self._data[idx], self._labels[idx]

    def __call__(self, ds: Any) -> Any:
        return ds.map(self)


def create_dataloader(  # type: ignore[no-any-unimported]
    data: Any,
    batch_size: int,
    labels: Optional[Any] = None,
    shuffle: bool = True,
    seed: int = 42,
    worker_count: int = 2,
    worker_buffer_size: int = 2,
    drop_remainder: bool = True,
    custom_operations: Optional[list[grain.Transformation]] = None,
) -> grain.DataLoader:
    """Create a Grain DataLoader for array data.

    Args:
        data: The data array to load. Can be a JAX array or numpy array.
        batch_size: Number of samples per batch.
        labels: Optional labels array for supervised learning. If provided,
            the dataloader will yield (data, label) tuples.
        shuffle: Whether to shuffle the data. Defaults to True.
        seed: Random seed for shuffling. Defaults to 42.
        worker_count: Number of worker processes for parallel loading. Defaults to 2.
        worker_buffer_size: Number of batches to buffer per worker. Defaults to 2.
        drop_remainder: Whether to drop the last incomplete batch. Defaults to True.
        custom_operations: Optional list of custom Grain transformations to apply
            before batching. If None, uses default ArrayTransform or LabeledTransform.

    Returns:
        A Grain DataLoader configured with the specified parameters.

    Example:
        Basic usage:
        >>> import jax.numpy as jnp
        >>> from scirex.data import create_dataloader
        >>> data = jnp.ones((100, 10))
        >>> loader = create_dataloader(data, batch_size=32)
        >>> for batch in loader:
        ...     print(batch.shape)  # (32, 10)

        With labels:
        >>> data = jnp.ones((100, 10))
        >>> labels = jnp.arange(100)
        >>> loader = create_dataloader(data, batch_size=32, labels=labels)
        >>> for batch_data, batch_labels in loader:
        ...     print(batch_data.shape, batch_labels.shape)  # (32, 10) (32,)

        With custom transformations:
        >>> class CustomTransform(grain.MapTransform):
        ...     def map(self, x):
        ...         return x * 2
        >>> loader = create_dataloader(
        ...     data, batch_size=32,
        ...     custom_operations=[CustomTransform()]
        ... )
    """
    sampler = grain.IndexSampler(
        len(data),
        shuffle=shuffle,
        seed=seed,
        shard_options=grain.NoSharding(),
        num_epochs=1,  # Iterate over the dataset for one epoch
    )

    # Build operations list
    if custom_operations is not None:
        operations = custom_operations
    elif labels is not None:
        operations = [LabeledTransform(data, labels)]
    else:
        operations = [ArrayTransform(data)]

    # Add batching
    operations.append(grain.Batch(batch_size, drop_remainder=drop_remainder))

    dataloader = grain.DataLoader(
        data_source=range(len(data)),
        sampler=sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        operations=operations,
    )

    return dataloader
