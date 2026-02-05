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
Module: transformer.py

This module implements Transformer architectures using Flax NNX.

Key Classes:
    EncoderBlock: Transformer Encoder Block
    DecoderBlock: Transformer Decoder Block
    PositionalEmbedding: Standard Positional Embedding
    EncoderModel: Transformer Encoder-only Model
    EncoderDecoderModel: Transformer Encoder-Decoder Model

Key Features:
    - Built on top of JAX and Flax NNX
    - Supports both Encoder and Encoder-Decoder architectures
    - includes Multi-Head Attention and Position-wise Feedforward Networks
    - Configurable dropout and layer normalization

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

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
            mask: Optional mask array.

        Returns:
            Output array of shape (batch_size, seq_len, d_model).
        """
        padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32) if mask else None
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
            x: Input array (target sequence) of shape (batch_size, seq_len, d_model).
            y: Input array (source sequence from encoder) of shape (batch_size, seq_len, d_model).
            mask: Optional mask array.

        Returns:
            Output array of shape (batch_size, seq_len, d_model).
        """
        causal_mask = self.get_causal_attention_mask(x.shape[1])
        if mask is not None:
            padding_mask = jnp.expand_dims(mask, axis=1).astype(jnp.int32)
            padding_mask = jnp.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        x = self.norm1(x + self.self_attn(x, x, x, mask=causal_mask))
        x = self.norm2(x + self.cross_attn(x, y, y, mask=padding_mask))
        x = self.norm3(x + self.ffn(x))
        return x


class PositionalEmbedding(nn.Module):
    """
    Positional Encoding module.

    Attributes:
        pos_emb: Embedding layer for position indices.
    """

    def __init__(self, context_size: int, d_model: int, rngs: nn.Rngs):
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
        """
        Computes the mask for the input.

        Args:
            x: Input array.
            mask: Optional mask array.

        Returns:
            Boolean mask array identifying non-zero elements if mask is provided, else None.
        """
        return jnp.not_equal(x, 0) if mask else None


class EncoderModel(nn.Module):
    """
    Transformer Encoder Model.

    Attributes:
        emb: Embedding function (token + positional).
        encoder: Encoder block.
        dropout: Dropout layer.
        head: Linear head for output projection.
    """

    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        dropout_rate: float,
        rngs: nn.Rngs,
    ):
        token_emb = nn.Embed(vocab_size, d_model, rngs=rngs)
        pos_emb = PositionalEmbedding(context_size, d_model, rngs=rngs)
        self.emb = lambda x: token_emb(x) + pos_emb(x)
        self.encoder = EncoderBlock(d_model, d_hidden, n_heads, rngs=rngs)
        self.dropout = nn.Dropout(dropout_rate, rngs=rngs)
        self.head = nn.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, x: Array, mask: Optional[Array] = None, deterministic: bool = False) -> Array:
        """
        Forward pass of the EncoderModel.

        Args:
            x: Input token indices of shape (batch_size, seq_len).
            mask: Optional mask array.
            deterministic: If True, dropout is disabled.

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        x = self.emb(x)
        x = self.encoder(x, mask=mask)
        # per nn.Dropout - disable (deterministic=True) for eval, keep (False) for training
        x = self.dropout(x, deterministic=deterministic)

        return self.head(x)


class EncoderDecoderModel(nn.Module):
    """
    Transformer Encoder-Decoder Model.

    Attributes:
        emb: Embedding function (token + positional).
        encoder: Encoder block.
        decoder: Decoder block.
        dropout: Dropout layer.
        head: Linear head for output projection.
    """

    def __init__(
        self,
        context_size: int,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        dropout_rate: float,
        rngs: nn.Rngs,
    ):
        token_emb = nn.Embed(vocab_size, d_model, rngs=rngs)
        pos_emb = PositionalEmbedding(context_size, d_model, rngs=rngs)
        self.emb = lambda x: token_emb(x) + pos_emb(x)
        self.encoder = EncoderBlock(d_model, d_hidden, n_heads, rngs=rngs)
        self.decoder = DecoderBlock(d_model, d_hidden, n_heads, rngs=rngs)
        self.dropout = nn.Dropout(dropout_rate, rngs=rngs)
        self.head = nn.Linear(d_model, vocab_size, rngs=rngs)

    def __call__(self, x: Array, y: Array, mask: Optional[Array] = None, deterministic: bool = False) -> Array:
        """
        Forward pass of the EncoderDecoderModel.

        Args:
            x: Source token indices of shape (batch_size, source_seq_len).
            y: Target token indices of shape (batch_size, target_seq_len).
            mask: Optional mask array.
            deterministic: If True, dropout is disabled.

        Returns:
            Logits of shape (batch_size, target_seq_len, vocab_size).
        """
        x, y = self.emb(x), self.emb(y)
        x = self.encoder(x, mask=mask)
        y = self.decoder(y, x, mask=mask)
        # per nn.Dropout - disable (deterministic=True) for eval, keep (False) for training
        y = self.dropout(y, deterministic=deterministic)

        return self.head(y)
