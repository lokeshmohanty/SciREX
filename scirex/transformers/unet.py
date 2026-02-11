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
U-Net architecture for diffusion models.

The U-Net architecture is characterized by its symmetric encoder-decoder
structure and skip connections that preserve spatial information.

References:
    - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015.
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020 (for time conditioning).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx as nn

from .embeddings import sinusoidal_positional_encoding

Array = jax.Array


class UNet(nn.Module):
    """
    U-Net architecture with time embedding for diffusion models.

    Features:
        - Encoder-decoder architecture with skip connections
        - Time embedding via sinusoidal positional encoding
        - Multi-head self-attention at multiple scales
        - Residual blocks with layer normalization

    Attributes:
        features: Base number of features (multiplied at each level).
        time_emb_dim: Dimension of time embeddings.
    """

    def __init__(self, in_channels: int, out_channels: int, features: int, time_emb_dim: int = 128, *, rngs: nn.Rngs):
        """Initializes the U-Net architecture with time conditioning.

        Args:
            in_channels: Number of input channels (e.g., 1 for grayscale or 3 for RGB).
            out_channels: Number of output channels (typically matching input for reconstruction).
            features: Base feature dimension, which is doubled at each downsampling step.
            time_emb_dim: Dimension of the sinusoidal time embedding (default: 128).
            rngs: Random number generators for weight initialization.
        """
        self.features = features

        # Time embedding layers for diffusion timestep conditioning
        self.time_mlp_1 = nn.Linear(in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs)
        self.time_mlp_2 = nn.Linear(in_features=time_emb_dim, out_features=time_emb_dim, rngs=rngs)

        # Time projection layers for different scales
        self.time_proj1 = nn.Linear(in_features=time_emb_dim, out_features=features, rngs=rngs)
        self.time_proj2 = nn.Linear(in_features=time_emb_dim, out_features=features * 2, rngs=rngs)
        self.time_proj3 = nn.Linear(in_features=time_emb_dim, out_features=features * 4, rngs=rngs)
        self.time_proj4 = nn.Linear(in_features=time_emb_dim, out_features=features * 8, rngs=rngs)

        # Encoder path
        self.down_conv1 = self._create_residual_block(in_channels, features, rngs)
        self.down_conv2 = self._create_residual_block(features, features * 2, rngs)
        self.down_conv3 = self._create_residual_block(features * 2, features * 4, rngs)
        self.down_conv4 = self._create_residual_block(features * 4, features * 8, rngs)

        # Multi-head self-attention blocks
        self.attention1 = self._create_attention_block(features * 4, rngs)
        self.attention2 = self._create_attention_block(features * 8, rngs)

        # Bridge connecting encoder and decoder
        self.bridge_down = self._create_residual_block(features * 8, features * 16, rngs)
        self.bridge_attention = self._create_attention_block(features * 16, rngs)
        self.bridge_up = self._create_residual_block(features * 16, features * 16, rngs)

        # Decoder path with skip connections
        self.up_conv4 = self._create_residual_block(features * 24, features * 8, rngs)
        self.up_conv3 = self._create_residual_block(features * 12, features * 4, rngs)
        self.up_conv2 = self._create_residual_block(features * 6, features * 2, rngs)
        self.up_conv1 = self._create_residual_block(features * 3, features, rngs)

        # Output layers
        self.final_norm = nn.LayerNorm(features, rngs=rngs)
        self.final_conv = nn.Conv(
            in_features=features,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )

    def _create_attention_block(self, channels: int, rngs: nn.Rngs) -> Callable:
        """
        Create a self-attention block with learned query, key, value projections.

        Args:
            channels: The number of channels in the input feature maps.
            rngs: Random number generators.

        Returns:
            A function representing a forward pass through the attention block.
        """
        query_proj = nn.Linear(in_features=channels, out_features=channels, rngs=rngs)
        key_proj = nn.Linear(in_features=channels, out_features=channels, rngs=rngs)
        value_proj = nn.Linear(in_features=channels, out_features=channels, rngs=rngs)

        def forward(x: Array) -> Array:
            """
            Apply self-attention to the input.

            Args:
                x: Input tensor with shape [batch, height, width, channels].

            Returns:
                Output tensor after applying self-attention.
            """
            # Shape: batch, height, width, channels
            B, H, W, C = x.shape
            scale = jnp.sqrt(C).astype(x.dtype)

            # Project the input into query, key, value projections
            q = query_proj(x)
            k = key_proj(x)
            v = value_proj(x)

            # Reshape for attention computation
            q = q.reshape(B, H * W, C)
            k = k.reshape(B, H * W, C)
            v = v.reshape(B, H * W, C)

            # Compute scaled dot-product attention
            attention = jnp.einsum("bic,bjc->bij", q, k) / scale
            attention = jax.nn.softmax(attention, axis=-1)

            # Output tensor
            out = jnp.einsum("bij,bjc->bic", attention, v)
            out = out.reshape(B, H, W, C)

            return x + out  # ResNet-style residual connection

        return forward

    def _create_residual_block(self, in_channels: int, out_channels: int, rngs: nn.Rngs) -> Callable:
        """
        Create a residual block with two convolutions and normalization.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            rngs: Random number generators.

        Returns:
            A function representing the forward pass through the residual block.
        """
        # Convolutional layers with layer normalization
        conv1 = nn.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        norm1 = nn.LayerNorm(out_channels, rngs=rngs)
        conv2 = nn.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        norm2 = nn.LayerNorm(out_channels, rngs=rngs)

        # Projection shortcut if dimensions change
        shortcut = nn.Conv(
            in_features=in_channels, out_features=out_channels, kernel_size=(1, 1), strides=(1, 1), rngs=rngs
        )

        def forward(x: Array) -> Array:
            """Forward pass through the residual block."""
            identity = shortcut(x)

            x = conv1(x)
            x = norm1(x)
            x = nn.gelu(x)

            x = conv2(x)
            x = norm2(x)
            x = nn.gelu(x)

            return x + identity

        return forward

    def _downsample(self, x: Array) -> Array:
        """Downsample the input feature map with max pooling."""
        return jnp.asarray(nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME"))

    def _upsample(self, x: Array, target_size: int) -> Array:
        """Upsample the input feature map using nearest neighbor interpolation."""
        return jnp.asarray(jax.image.resize(x, (x.shape[0], target_size, target_size, x.shape[3]), method="nearest"))

    def __call__(self, x: Array, t: Array) -> Array:
        """
        Perform forward pass through the U-Net using time embeddings.

        Args:
            x: Input tensor of shape [batch, height, width, channels].
            t: Timestep tensor for diffusion process.

        Returns:
            Output tensor of same shape as input.
        """
        # Time embedding and projection
        t_emb = sinusoidal_positional_encoding(t, 128)
        t_emb = self.time_mlp_1(t_emb)
        t_emb = nn.gelu(t_emb)
        t_emb = self.time_mlp_2(t_emb)

        # Project time embeddings for each scale
        t_emb1 = self.time_proj1(t_emb)[:, None, None, :]
        t_emb2 = self.time_proj2(t_emb)[:, None, None, :]
        t_emb3 = self.time_proj3(t_emb)[:, None, None, :]
        t_emb4 = self.time_proj4(t_emb)[:, None, None, :]

        # Encoder path with time injection
        d1 = self.down_conv1(x)
        t_emb1 = jnp.broadcast_to(t_emb1, d1.shape)
        d1 = d1 + t_emb1

        d2 = self.down_conv2(self._downsample(d1))
        t_emb2 = jnp.broadcast_to(t_emb2, d2.shape)
        d2 = d2 + t_emb2

        d3 = self.down_conv3(self._downsample(d2))
        d3 = self.attention1(d3)
        t_emb3 = jnp.broadcast_to(t_emb3, d3.shape)
        d3 = d3 + t_emb3

        d4 = self.down_conv4(self._downsample(d3))
        d4 = self.attention2(d4)
        t_emb4 = jnp.broadcast_to(t_emb4, d4.shape)
        d4 = d4 + t_emb4

        # Bridge
        b = self._downsample(d4)
        b = self.bridge_down(b)
        b = self.bridge_attention(b)
        b = self.bridge_up(b)

        # Decoder path with skip connections
        u4 = self.up_conv4(jnp.concatenate([self._upsample(b, d4.shape[1]), d4], axis=-1))
        u3 = self.up_conv3(jnp.concatenate([self._upsample(u4, d3.shape[1]), d3], axis=-1))
        u2 = self.up_conv2(jnp.concatenate([self._upsample(u3, d2.shape[1]), d2], axis=-1))
        u1 = self.up_conv1(jnp.concatenate([self._upsample(u2, d1.shape[1]), d1], axis=-1))

        # Final layers
        x = self.final_norm(u1)
        x = nn.gelu(x)
        return self.final_conv(x)
