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
Transformer building blocks.

This module contains EncoderBlock and DecoderBlock components
for building Transformer architectures.

The core mechanism is Scaled Dot-Product Attention:
$\text{Attention}(Q, K, V) = \text{softmax}\\left(\frac{QK^T}{\\sqrt{d_k}}\right)V$

References:
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx as nn

Array = jax.Array


class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Attributes:
        d_model: The number of expected features in the input.
        d_hidden: The internal dimension of the feedforward network.
        n_heads: The number of heads in the multiheadattention models.
        attn: The multi-head attention module.
        ffn: The feedforward network module.
        norm1: Layer normalization for the attention output.
        norm2: Layer normalization for the feedforward output.
    """

    def __init__(self, d_model: int, d_hidden: int, n_heads: int, rngs: nn.Rngs):
        """Initializes the EncoderBlock.

        Args:
            d_model: The number of features in the input.
            d_hidden: The dimension of the hidden layer in the FFN.
            n_heads: The number of attention heads.
            rngs: Random number generators for weight initialization.
        """
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_heads = n_heads

        self.attn = nn.MultiHeadAttention(n_heads, d_model, decode=False, rngs=rngs)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden, rngs=rngs),
            nn.relu,
            nn.Linear(d_hidden, d_model, rngs=rngs),
        )

        self.norm1 = nn.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nn.LayerNorm(d_model, rngs=rngs)

    def __call__(self, x: Array, mask: Optional[Array] = None) -> Array:
        """
        Forward pass of the EncoderBlock.

        Args:
            x: Input array of shape (batch_size, seq_len, d_model).
            mask: Optional mask array of shape (batch_size, seq_len).

        Returns:
            Output array of shape (batch_size, seq_len, d_model).
        """
        # Reshape mask to (batch, 1, 1, seq_len) for proper broadcasting
        padding_mask = mask[:, None, None, :] if mask is not None else None
        x = self.norm1(x + self.attn(x, x, x, mask=padding_mask, decode=False))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block.

    Attributes:
        d_model: The number of expected features in the input.
        d_hidden: The internal dimension of the feedforward network.
        n_heads: The number of heads in the multiheadattention models.
        self_attn: The self-attention module.
        cross_attn: The cross-attention module.
        ffn: The feedforward network module.
        norm1: Layer normalization for the self-attention output.
        norm2: Layer normalization for the cross-attention output.
        norm3: Layer normalization for the feedforward output.
    """

    def __init__(self, d_model: int, d_hidden: int, n_heads: int, rngs: nn.Rngs):
        """Initializes the DecoderBlock.

        Args:
            d_model: The number of features in the input.
            d_hidden: The dimension of the hidden layer in the FFN.
            n_heads: The number of attention heads.
            rngs: Random number generators for weight initialization.
        """
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.self_attn = nn.MultiHeadAttention(n_heads, d_model, decode=False, rngs=rngs)
        self.cross_attn = nn.MultiHeadAttention(n_heads, d_model, decode=False, rngs=rngs)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden, rngs=rngs),
            nn.relu,
            nn.Linear(d_hidden, d_model, rngs=rngs),
        )
        self.norm1 = nn.LayerNorm(d_model, rngs=rngs)
        self.norm2 = nn.LayerNorm(d_model, rngs=rngs)
        self.norm3 = nn.LayerNorm(d_model, rngs=rngs)

    def get_causal_attention_mask(self, context_size: int) -> Array:
        """
        Creates a causal mask for the attention mechanism.

        Args:
            context_size: The size of the context window.

        Returns:
            A causal mask of shape (1, 1, context_size, context_size).
        """
        i = jnp.arange(context_size)[:, None]
        j = jnp.arange(context_size)
        mask = (i >= j).astype(jnp.int32)
        mask = jnp.reshape(mask, (1, 1, context_size, context_size))
        return mask

    def __call__(self, x: Array, y: Array, mask: Optional[Array] = None) -> Array:
        """
        Forward pass of the DecoderBlock.

        Args:
            x: Input array (target sequence) of shape (batch_size, tgt_seq_len, d_model).
            y: Input array (source sequence from encoder) of shape (batch_size, src_seq_len, d_model).
            mask: Optional mask array of shape (batch_size, src_seq_len) for masking encoder outputs in cross-attention.

        Returns:
            Output array of shape (batch_size, tgt_seq_len, d_model).
        """
        # Causal mask for self-attention on target sequence
        causal_mask = self.get_causal_attention_mask(x.shape[1])

        # Source mask for cross-attention (attending to encoder outputs)
        # Reshape mask to (batch, 1, 1, src_seq_len) for proper broadcasting
        cross_attn_mask = mask[:, None, None, :] if mask is not None else None

        x = self.norm1(x + self.self_attn(x, x, x, mask=causal_mask))
        x = self.norm2(x + self.cross_attn(x, y, y, mask=cross_attn_mask))
        x = self.norm3(x + self.ffn(x))
        return x
