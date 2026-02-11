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
Complete Transformer models.

This module contains end-to-end Transformer architectures based on the
original Transformer design.

References:
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from typing import Optional

import jax
from flax import nnx as nn

from .blocks import DecoderBlock, EncoderBlock
from .embeddings import PositionalEmbedding

Array = jax.Array


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
        """Initializes the Transformer Encoder Model.

        Args:
            context_size: Maximum sequence length (context window).
            vocab_size: Size of the vocabulary.
            d_model: Dimension of the token embeddings and model layers.
            d_hidden: Dimension of the feed-forward hidden layers.
            n_heads: Number of attention heads.
            dropout_rate: Probability of dropout during training.
            rngs: Random number generators for weights and dropout.
        """
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
            mask: Optional attention mask array of shape (batch_size, 1, 1, seq_len)
                  or (batch_size, 1, seq_len, seq_len).
            deterministic: If True, dropout is disabled (used during evaluation).

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
        """Initializes the Transformer Encoder-Decoder Model.

        Args:
            context_size: Maximum sequence length (context window).
            vocab_size: Size of the vocabulary.
            d_model: Dimension of the token embeddings and model layers.
            d_hidden: Dimension of the feed-forward hidden layers.
            n_heads: Number of attention heads.
            dropout_rate: Probability of dropout during training.
            rngs: Random number generators for weights and dropout.
        """
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
