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

"""Tests for UNet architecture."""

import jax.numpy as jnp
from flax import nnx as nn

from scirex.transformers import UNet


def test_unet_forward():
    """Test UNet forward pass."""
    batch_size, height, width = 2, 8, 8
    in_channels, out_channels, features = 1, 1, 16

    rng = nn.Rngs(0)
    model = UNet(in_channels, out_channels, features, rngs=rng)

    x = jnp.ones((batch_size, height, width, in_channels))
    t = jnp.array([0, 10])

    output = model(x, t)

    assert output.shape == (batch_size, height, width, out_channels)
    assert not jnp.isnan(output).any()


def test_unet_time_embedding():
    """Test UNet with different timesteps."""
    batch_size, height, width = 2, 8, 8
    in_channels, out_channels, features = 1, 1, 16

    rng = nn.Rngs(0)
    model = UNet(in_channels, out_channels, features, rngs=rng)

    x = jnp.ones((batch_size, height, width, in_channels))
    t1 = jnp.array([0, 0])
    t2 = jnp.array([50, 100])

    out1 = model(x, t1)
    out2 = model(x, t2)

    # Different timesteps should produce different outputs
    assert not jnp.allclose(out1, out2)


def test_unet_shape_preservation():
    """Test that U-Net preserves spatial dimensions."""
    batch_size, height, width = 4, 16, 16
    in_channels, out_channels, features = 3, 3, 32

    rng = nn.Rngs(0)
    model = UNet(in_channels, out_channels, features, rngs=rng)

    x = jnp.ones((batch_size, height, width, in_channels))
    t = jnp.array([10, 20, 30, 40])

    output = model(x, t)

    # Output should have same spatial dimensions
    assert output.shape[0] == batch_size
    assert output.shape[1] == height
    assert output.shape[2] == width
    assert output.shape[3] == out_channels


def test_unet_different_features():
    """Test UNet with different feature sizes."""
    batch_size, height, width = 2, 8, 8
    in_channels, out_channels = 1, 1

    rng1 = nn.Rngs(0)
    model_small = UNet(in_channels, out_channels, features=8, rngs=rng1)

    rng2 = nn.Rngs(1)
    model_large = UNet(in_channels, out_channels, features=32, rngs=rng2)

    x = jnp.ones((batch_size, height, width, in_channels))
    t = jnp.array([0, 10])

    out_small = model_small(x, t)
    out_large = model_large(x, t)

    assert out_small.shape == out_large.shape
