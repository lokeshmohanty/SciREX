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

"""Tests for transformer blocks."""

import jax.numpy as jnp
from flax import nnx as nn

from scirex.transformers import DecoderBlock, EncoderBlock


def test_encoder_block_forward():
    """Test EncoderBlock forward pass."""
    batch_size, seq_len, d_model = 2, 10, 64
    d_hidden, n_heads = 128, 4

    rng = nn.Rngs(0)
    block = EncoderBlock(d_model, d_hidden, n_heads, rngs=rng)

    x = jnp.ones((batch_size, seq_len, d_model))
    output = block(x)

    assert output.shape == (batch_size, seq_len, d_model)
    assert not jnp.isnan(output).any()


def test_encoder_block_with_mask():
    """Test EncoderBlock with attention mask."""
    batch_size, seq_len, d_model = 2, 10, 64
    d_hidden, n_heads = 128, 4

    rng = nn.Rngs(0)
    block = EncoderBlock(d_model, d_hidden, n_heads, rngs=rng)

    x = jnp.ones((batch_size, seq_len, d_model))
    mask = jnp.ones((batch_size, seq_len))
    output = block(x, mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)


def test_decoder_block_forward():
    """Test DecoderBlock forward pass."""
    batch_size, seq_len, d_model = 2, 10, 64
    d_hidden, n_heads = 128, 4

    rng = nn.Rngs(0)
    block = DecoderBlock(d_model, d_hidden, n_heads, rngs=rng)

    x = jnp.ones((batch_size, seq_len, d_model))
    y = jnp.ones((batch_size, seq_len, d_model))
    output = block(x, y)

    assert output.shape == (batch_size, seq_len, d_model)
    assert not jnp.isnan(output).any()


def test_decoder_block_causal_mask():
    """Test DecoderBlock causal mask generation."""
    d_model, d_hidden, n_heads = 64, 128, 4
    context_size = 10

    rng = nn.Rngs(0)
    block = DecoderBlock(d_model, d_hidden, n_heads, rngs=rng)

    mask = block.get_causal_attention_mask(context_size)

    assert mask.shape == (1, 1, context_size, context_size)
    # Check mask is lower triangular
    for i in range(context_size):
        for j in range(context_size):
            if i >= j:
                assert mask[0, 0, i, j] == 1
            else:
                assert mask[0, 0, i, j] == 0


def test_decoder_block_with_mask():
    """Test DecoderBlock with custom mask."""
    batch_size, seq_len, d_model = 2, 10, 64
    d_hidden, n_heads = 128, 4

    rng = nn.Rngs(0)
    block = DecoderBlock(d_model, d_hidden, n_heads, rngs=rng)

    x = jnp.ones((batch_size, seq_len, d_model))
    y = jnp.ones((batch_size, seq_len, d_model))
    mask = jnp.ones((batch_size, seq_len))
    output = block(x, y, mask=mask)

    assert output.shape == (batch_size, seq_len, d_model)
