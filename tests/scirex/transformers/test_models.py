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

"""Tests for complete transformer models."""

import jax.numpy as jnp
from flax import nnx as nn

from scirex.transformers import EncoderDecoderModel, EncoderModel


def test_encoder_model():
    """Test EncoderModel forward pass."""
    batch_size, seq_len = 2, 10
    context_size, vocab_size = 20, 1000
    d_model, d_hidden, n_heads = 64, 128, 4
    dropout_rate = 0.1

    rng = nn.Rngs(0)
    model = EncoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)

    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    logits = model(x, deterministic=True)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert not jnp.isnan(logits).any()


def test_encoder_model_with_mask():
    """Test EncoderModel with attention mask."""
    batch_size, seq_len = 2, 10
    context_size, vocab_size = 20, 1000
    d_model, d_hidden, n_heads = 64, 128, 4
    dropout_rate = 0.1

    rng = nn.Rngs(0)
    model = EncoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)

    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    mask = jnp.ones((batch_size, seq_len))
    logits = model(x, mask=mask, deterministic=True)

    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_encoder_decoder_model():
    """Test EncoderDecoderModel forward pass."""
    batch_size, src_len, tgt_len = 2, 10, 8
    context_size, vocab_size = 20, 1000
    d_model, d_hidden, n_heads = 64, 128, 4
    dropout_rate = 0.1

    rng = nn.Rngs(0)
    model = EncoderDecoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)

    x = jnp.ones((batch_size, src_len), dtype=jnp.int32)
    y = jnp.ones((batch_size, tgt_len), dtype=jnp.int32)
    logits = model(x, y, deterministic=True)

    assert logits.shape == (batch_size, tgt_len, vocab_size)
    assert not jnp.isnan(logits).any()


def test_encoder_decoder_model_with_mask():
    """Test EncoderDecoderModel with mask."""
    batch_size, src_len, tgt_len = 2, 10, 8
    context_size, vocab_size = 20, 1000
    d_model, d_hidden, n_heads = 64, 128, 4
    dropout_rate = 0.1

    rng = nn.Rngs(0)
    model = EncoderDecoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)

    x = jnp.ones((batch_size, src_len), dtype=jnp.int32)
    y = jnp.ones((batch_size, tgt_len), dtype=jnp.int32)
    mask = jnp.ones((batch_size, src_len))
    logits = model(x, y, mask=mask, deterministic=True)

    assert logits.shape == (batch_size, tgt_len, vocab_size)


def test_encoder_model_dropout():
    """Test EncoderModel dropout behavior."""
    batch_size, seq_len = 2, 10
    context_size, vocab_size = 20, 1000
    d_model, d_hidden, n_heads = 64, 128, 4
    dropout_rate = 0.5

    rng = nn.Rngs(0)
    model = EncoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)

    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # With dropout (training)
    logits_train = model(x, deterministic=False)

    # Without dropout (eval)
    logits_eval = model(x, deterministic=True)

    # Outputs should be different due to dropout
    assert not jnp.allclose(logits_train, logits_eval)
