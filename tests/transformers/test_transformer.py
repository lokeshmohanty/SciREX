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


"""
Module: test_transformer.py

This module contains functional tests for the Transformer architecture implementation.

Key Features:
    - Tests EncoderModel forward pass
    - Tests EncoderDecoderModel forward pass
    - Verifies output shapes and numerical stability

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

"""

import jax
import jax.numpy as jnp
from flax import nnx as nn

from scirex.transformers.transformer import EncoderDecoderModel, EncoderModel


def test_encoder_model_forward():
    key = jax.random.PRNGKey(0)
    rngs = nn.Rngs(0)

    batch_size = 2
    context_size = 10
    vocab_size = 100
    d_model = 16
    d_hidden = 32
    n_heads = 2
    dropout_rate = 0.1

    model = EncoderModel(
        context_size=context_size,
        vocab_size=vocab_size,
        d_model=d_model,
        d_hidden=d_hidden,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    x = jax.random.randint(key, (batch_size, context_size), 0, vocab_size)
    output = model(x, deterministic=True)

    assert output.shape == (batch_size, context_size, vocab_size)
    assert jnp.all(jnp.isfinite(output))


def test_encoder_decoder_model_forward():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    rngs = nn.Rngs(0)

    batch_size = 2
    context_size = 10
    vocab_size = 100
    d_model = 16
    d_hidden = 32
    n_heads = 2
    dropout_rate = 0.1

    model = EncoderDecoderModel(
        context_size=context_size,
        vocab_size=vocab_size,
        d_model=d_model,
        d_hidden=d_hidden,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    x = jax.random.randint(k1, (batch_size, context_size), 0, vocab_size)
    y = jax.random.randint(k2, (batch_size, context_size), 0, vocab_size)

    output = model(x, y, deterministic=True)

    assert output.shape == (batch_size, context_size, vocab_size)
    assert jnp.all(jnp.isfinite(output))
