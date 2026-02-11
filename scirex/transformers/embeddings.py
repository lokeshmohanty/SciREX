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
Positional embeddings for Transformer models.

This module contains various positional embedding implementations.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx as nn

Array = jax.Array


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding module.

    References:
        - Gehring et al., "Convolutional Sequence to Sequence Learning", ICML 2017.
    """

    def __init__(self, context_size: int, d_model: int, rngs: nn.Rngs):
        """Initializes the PositionalEmbedding module.

        Args:
            context_size: Maximum sequence length.
            d_model: Dimension of the embeddings.
            rngs: Random number generators.
        """
        self.pos_emb = nn.Embed(context_size, d_model, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        """
        Adds positional embeddings to the input.

        Args:
            x: Input array of shape (batch_size, seq_len, d_model).

        Returns:
            Positional embeddings of shape (1, seq_len, d_model).
        """
        pos = jnp.arange(0, x.shape[1])[None, :]
        return self.pos_emb(pos)

    def compute_mask(self, x: Array, mask: Optional[Array] = None) -> Array | None:
        """Computes a boolean mask identifying non-padding (non-zero) elements.

        Args:
            x: Input token indices of shape (batch_size, seq_len).
            mask: If not None, a mask is computed.

        Returns:
            A boolean array of shape (batch_size, seq_len) if mask is provided, else None.
        """
        return jnp.not_equal(x, 0) if mask is not None else None


def sinusoidal_positional_encoding(timesteps: Array, dim: int) -> Array:
    r"""
    Apply sinusoidal positional encoding for time embedding.

    The encoding is defined as:
    $PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
    $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$

    Useful for diffusion models and other temporal architectures.

    Args:
        timesteps: Time embedding, representing the timestep.
        dim: The dimension of the output positional encoding.

    Returns:
        Sinusoidal positional embedding per timestep.
    """
    half_dim = dim // 2
    # Compute the logarithmic scaling factor for sinusoidal frequencies
    emb = jnp.log(10000.0) / (half_dim - 1)
    # Generate a range of sinusoidal frequencies
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    # Create the positional encoding by multiplying time embeddings
    emb = timesteps[:, None] * emb[None, :]
    # Concatenate sine and cosine components for richer representation
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb
