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

"""Tests for positional embeddings."""

import jax.numpy as jnp
from flax import nnx as nn

from scirex.transformers import PositionalEmbedding, sinusoidal_positional_encoding


def test_positional_embedding():
    """Test learned positional embedding."""
    batch_size, seq_len, d_model = 2, 10, 64
    context_size = 20

    rng = nn.Rngs(0)
    pos_emb = PositionalEmbedding(context_size, d_model, rngs=rng)

    x = jnp.ones((batch_size, seq_len, d_model))
    output = pos_emb(x)

    assert output.shape == (1, seq_len, d_model)
    assert not jnp.isnan(output).any()


def test_sinusoidal_encoding():
    """Test sinusoidal positional encoding."""
    batch_size, dim = 4, 128
    timesteps = jnp.array([0, 10, 50, 100])

    encoding = sinusoidal_positional_encoding(timesteps, dim)

    assert encoding.shape == (batch_size, dim)
    assert not jnp.isnan(encoding).any()
    # Check range is reasonable
    assert jnp.all(jnp.abs(encoding) <= 1.5)


def test_sinusoidal_encoding_different_timesteps():
    """Test that different timesteps produce different encodings."""
    dim = 64
    t1 = jnp.array([0])
    t2 = jnp.array([100])

    enc1 = sinusoidal_positional_encoding(t1, dim)
    enc2 = sinusoidal_positional_encoding(t2, dim)

    # Encodings should be different
    assert not jnp.allclose(enc1, enc2)


def test_positional_embedding_compute_mask():
    """Test mask computation."""
    context_size, d_model = 20, 64
    batch_size, seq_len = 2, 10

    rng = nn.Rngs(0)
    pos_emb = PositionalEmbedding(context_size, d_model, rngs=rng)

    # Test with non-zero input
    x = jnp.ones((batch_size, seq_len))
    mask = pos_emb.compute_mask(x, mask=True)
    assert mask is not None
    assert mask.shape == x.shape

    # Test without mask
    mask_none = pos_emb.compute_mask(x, mask=None)
    assert mask_none is None
