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

"""Tests for data loaders."""

import jax.numpy as jnp
import numpy as np

from scirex.data import ArrayTransform, LabeledTransform, create_dataloader


def test_create_dataloader_basic():
    """Test basic dataloader creation."""
    data = jnp.ones((100, 10))
    loader = create_dataloader(data, batch_size=32, worker_count=0)

    batches = list(loader)
    assert len(batches) == 3  # 100 // 32 = 3 (with drop_remainder=True)
    assert batches[0].shape == (32, 10)


def test_create_dataloader_with_labels():
    """Test dataloader with labels."""
    data = jnp.ones((100, 10))
    labels = jnp.arange(100)
    loader = create_dataloader(data, batch_size=32, labels=labels, worker_count=0)

    batches = list(loader)
    assert len(batches) == 3

    batch_data, batch_labels = batches[0]
    assert batch_data.shape == (32, 10)
    assert batch_labels.shape == (32,)


def test_create_dataloader_no_shuffle():
    """Test dataloader without shuffling."""
    data = jnp.arange(100).reshape(100, 1)
    loader = create_dataloader(data, batch_size=10, shuffle=False, worker_count=0)

    batches = list(loader)
    first_batch = batches[0]

    # Without shuffling, first batch should be [0, 1, 2, ..., 9]
    expected = jnp.arange(10).reshape(10, 1)
    assert jnp.allclose(first_batch, expected)


def test_create_dataloader_with_shuffle():
    """Test dataloader with shuffling."""
    data = jnp.arange(100).reshape(100, 1)
    loader = create_dataloader(data, batch_size=10, shuffle=True, seed=42, worker_count=0)

    batches = list(loader)
    first_batch = batches[0]

    # With shuffling, first batch should NOT be [0, 1, 2, ..., 9]
    expected = jnp.arange(10).reshape(10, 1)
    assert not jnp.allclose(first_batch, expected)


def test_create_dataloader_drop_remainder():
    """Test drop_remainder parameter."""
    data = jnp.ones((105, 10))

    # With drop_remainder=True (default)
    loader = create_dataloader(data, batch_size=32, drop_remainder=True, worker_count=0)
    batches = list(loader)
    assert len(batches) == 3  # 105 // 32 = 3

    # With drop_remainder=False
    loader = create_dataloader(data, batch_size=32, drop_remainder=False, worker_count=0)
    batches = list(loader)
    assert len(batches) == 4  # ceil(105 / 32) = 4
    assert batches[-1].shape[0] == 9  # Last batch has 9 samples


def test_array_transform():
    """Test ArrayTransform."""
    data = jnp.arange(10)
    transform = ArrayTransform(data)

    assert transform.map(0) == 0
    assert transform.map(5) == 5
    assert transform.map(9) == 9


def test_labeled_transform():
    """Test LabeledTransform."""
    data = jnp.ones((10, 5))
    labels = jnp.arange(10)
    transform = LabeledTransform(data, labels)

    sample_data, sample_label = transform.map(3)
    assert sample_data.shape == (5,)
    assert sample_label == 3


def test_create_dataloader_with_numpy():
    """Test dataloader with numpy arrays."""
    data = np.ones((100, 10))
    loader = create_dataloader(data, batch_size=32, worker_count=0)

    batches = list(loader)
    assert len(batches) == 3
    assert batches[0].shape == (32, 10)


def test_create_dataloader_custom_operations():
    """Test dataloader with custom operations."""
    import grain.python as grain

    class DoubleTransform(grain.MapTransform):
        def map(self, x):
            return x * 2

    data = jnp.ones((100, 10))
    loader = create_dataloader(
        data, batch_size=32, custom_operations=[ArrayTransform(data), DoubleTransform()], worker_count=0
    )

    batches = list(loader)
    # Values should be doubled
    assert jnp.allclose(batches[0], jnp.ones((32, 10)) * 2)
