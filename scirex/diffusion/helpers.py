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
Module: helpers.py

This module implements helper classes and models for diffusion, ported from
smalldiffusion to JAX/Flax (nnx).

Classes:
    ModelMixin: Base mixin with common utility methods.
    SigmaEmbedderSinCos: Sinusoidal embeddings for sigma.
    TimeInputMLP: Simple MLP taking time/sigma as input.
    CondEmbedderLabel: Embeddings for conditional labels.
    ConditionalMLP: MLP with time and class conditioning.
    IdealDenoiser: Exact denoiser using the full dataset.
"""

from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx as nn

from scirex.diffusion.forward import get_sigma_embeds

_CFG_ERROR_MSG = "Model must have 'cond_embed.null_cond' to use internal CFG generation."

Array = jax.Array


class ModelMixin:
    """Mixin class providing common diffusion methods."""

    def __call__(self, x: Array, sigma: Array, cond: Optional[Array] = None) -> Array:
        """Forward pass. Should be implemented by subclasses."""
        raise NotImplementedError

    def rand_input(self, rng: nn.Rngs, batchsize: int) -> Array:
        """Generate random input noise."""
        if not hasattr(self, "input_dims"):
            raise AttributeError('Model must have "input_dims" attribute!')  # noqa: TRY003
        return jax.random.normal(rng(), (batchsize, *self.input_dims))

    def get_loss(
        self,
        x0: Array,
        sigma: Array,
        eps: Array,
        cond: Optional[Array] = None,
        loss_fn: Any = None,
    ) -> Array:
        """Compute loss for epsilon prediction."""
        if loss_fn is None:
            loss_fn = lambda a, b: jnp.mean((a - b) ** 2)

        # Predict eps
        eps_pred = self(x0 + sigma.reshape(sigma.shape + (1,) * (x0.ndim - 1)) * eps, sigma, cond=cond)
        loss = loss_fn(eps, eps_pred)
        # Ensure result is cast to Array, as loss_fn might return Any/scalar
        return jnp.asarray(loss)

    def predict_eps(self, x: Array, sigma: Array, cond: Optional[Array] = None) -> Array:
        """Predict noise (epsilon)."""
        return self(x, sigma, cond=cond)

    def predict_eps_cfg(self, x: Array, sigma: Array, cond: Array, cfg_scale: float) -> Array:
        """Predict noise with Classifier-Free Guidance."""
        if cond is None or cfg_scale == 0:
            return self.predict_eps(x, sigma, cond=cond)

        # Assuming null_cond is available on the model (e.g., from CondEmbedderLabel)
        # However, purely functional approach might require passing uncond explicitly.
        # Here we follow the structure where the model handles it if possible,
        # or we expect 'cond' to already contain the formatting or we do concatenation.

        # Ideally, we should pass 'uncond' explicitly or have the model manage it.
        # For this mixin, we'll assume the standard CFG pattern:
        # eps_cond = model(x, sigma, cond)
        # eps_uncond = model(x, sigma, uncond) -> requires uncond to be known.

        # Since smalldiffusion's `predict_eps_cfg` generates `uncond` internally using `self.cond_embed.null_cond`,
        # we will attempt to replicate that if `cond_embed` exists.

        if not hasattr(self, "cond_embed") or not hasattr(self.cond_embed, "null_cond"):
            raise AttributeError(_CFG_ERROR_MSG)

        # Create uncond batch matching cond shape but filled with null_cond
        # smalldiffusion: uncond = torch.full_like(cond, self.cond_embed.null_cond)
        uncond = jnp.full_like(cond, self.cond_embed.null_cond)

        # We can either run twice or concatenate. Concatenation is usually more efficient if batch norm isn't an issue.
        # JAX/Flax often favors vmap or simple concatenation.

        # x_double = jnp.concatenate([x, x], axis=0)
        # sigma_double = jnp.concatenate([sigma, sigma], axis=0)
        # cond_double = jnp.concatenate([cond, uncond], axis=0)

        # eps_combined = self(x_double, sigma_double, cond=cond_double)
        # eps_cond, eps_uncond = jnp.split(eps_combined, 2, axis=0)

        # Using separate calls for clarity and to avoid batch size doubling issues if complex
        eps_cond_val = self(x, sigma, cond)
        eps_uncond_val = self(x, sigma, uncond)

        return eps_uncond_val + cfg_scale * (eps_cond_val - eps_uncond_val)


class SigmaEmbedderSinCos(nn.Module):
    """Sinusoidal embedding for sigma values."""

    def __init__(
        self,
        hidden_size: int,
        scaling_factor: float = 0.5,
        log_scale: bool = True,
        *,
        rngs: nn.Rngs,
    ):
        self.scaling_factor = scaling_factor
        self.log_scale = log_scale

        self.linear1 = nn.Linear(2, hidden_size, rngs=rngs)
        self.linear2 = nn.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, sigma: Array) -> Array:
        # sigma: (B,)
        # get_sigma_embeds expects (B,) and returns (B, 2)
        # Note: scirex's get_sigma_embeds does log scaling internally

        # In smalldiffusion:
        # if log_scale: sigma = log(sigma)
        # s = sigma * scaling_factor
        # cat(sin(s), cos(s))

        # In scirex.diffusion.forward.get_sigma_embeds:
        # sigma = log(sigma) * scaling_factor
        # stack(sin, cos)

        # So scirex's function matches smalldiffusion's logic.

        sig_embed = get_sigma_embeds(sigma, self.scaling_factor)  # (B, 2)

        x = self.linear1(sig_embed)
        x = nn.silu(x)
        x = self.linear2(x)
        return x


class CondEmbedderLabel(nn.Module):
    """Embedding for class labels."""

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        dropout_prob: float = 0.1,
        *,
        rngs: nn.Rngs,
    ):
        self.num_classes = num_classes
        self.null_cond = num_classes  # Use num_classes as the null token
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size

        self.embeddings = nn.Embed(num_classes + 1, hidden_size, rngs=rngs)

    def __call__(self, labels: Array, training: bool = False, rngs: Optional[nn.Rngs] = None) -> Array:
        # labels: (B,)
        if training and self.dropout_prob > 0:
            if rngs is None:
                raise ValueError("rngs must be provided for dropout during training")  # noqa: TRY003

            # Generate dropout mask
            mask = jax.random.bernoulli(rngs(), p=self.dropout_prob, shape=labels.shape)
            # Replace dropped labels with null_cond
            labels = jnp.where(mask, self.null_cond, labels)

        return self.embeddings(labels)


class TimeInputMLP(nn.Module, ModelMixin):
    """Simple MLP taking x and sigma as input."""

    sigma_dim = 2

    def __init__(
        self,
        dim: int = 2,
        output_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (16, 128, 256, 128, 16),
        *,
        rngs: nn.Rngs,
    ):
        self.input_dims = (dim,)
        output_dim = output_dim or dim

        self.input_dims = (dim,)
        output_dim = output_dim or dim

        layers: list[Any] = []
        in_dim = dim + self.sigma_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, rngs=rngs))
            layers.append(nn.gelu)  # Using standard GELU function
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim, rngs=rngs))

        self.layers = layers

    def __call__(self, x: Array, sigma: Array, cond: Optional[Array] = None) -> Array:
        # x: (B, dim)
        # sigma: (B,)

        # Get sinusoidal embeddings for sigma directly or use SigmaEmbedder?
        # smalldiffusion's TimeInputMLP uses `get_sigma_embeds` directly, NOT SigmaEmbedderSinCos.
        # It concatenates (B, 2) directly.

        sigma_embeds = get_sigma_embeds(sigma)  # (B, 2)

        # Concatenate x and sigma embeddings
        x = jnp.concatenate([x, sigma_embeds], axis=-1)

        for layer in self.layers:
            x = layer(x)

        return x


class ConditionalMLP(TimeInputMLP):
    """MLP with class conditioning."""

    def __init__(
        self,
        dim: int = 2,
        hidden_dims: Sequence[int] = (16, 128, 256, 128, 16),
        cond_dim: int = 4,
        num_classes: int = 10,
        dropout_prob: float = 0.1,
        *,
        rngs: nn.Rngs,
    ):
        # Initialize internal TimeInputMLP structure but with adapted dimensions
        # In smalldiffusion, ConditionalMLP inherits TimeInputMLP and calls super().__init__
        # with dim = dim + cond_dim.

        # We need to initialize the parent with the adjusted input dimension
        # so the layers are created correctly.
        super().__init__(dim=dim + cond_dim, output_dim=dim, hidden_dims=hidden_dims, rngs=rngs)

        # Reset input_dims to original dim because that's what the user sees
        self.input_dims = (dim,)
        self.real_dim = dim

        self.cond_embed = CondEmbedderLabel(
            hidden_size=cond_dim, num_classes=num_classes, dropout_prob=dropout_prob, rngs=rngs
        )

    def __call__(
        self,
        x: Array,
        sigma: Array,
        cond: Optional[Array] = None,
        training: bool = False,
        rngs: Optional[nn.Rngs] = None,
    ) -> Array:
        if cond is None:
            raise ValueError("ConditionalMLP requires 'cond' argument.")  # noqa: TRY003

        # Embed condition
        cond_embeds = self.cond_embed(cond, training=training, rngs=rngs)  # (B, cond_dim)

        # Concatenate x and cond BEFORE passing to parent logic (which adds sigma)
        # But parent's __call__ expects just 'x'.
        # Parent __call__:
        #   sigma_embeds = get_sigma_embeds(sigma)
        #   nn_input = concatenate([x, sigma_embeds])
        #   return net(nn_input)

        # If we pass x_cat = [x, cond_embeds] to parent:
        #   nn_input = [x, cond_embeds, sigma_embeds]
        # This matches smalldiffusion: cat([x, sigma_embeds, cond_embeds]) mostly?
        # smalldiffusion ConditionalMLP order:
        #   nn_input = torch.cat([x, sigma_embeds, cond_embeds], dim=1)

        # Our TimeInputMLP (parent) does:
        #   x = jnp.concatenate([x, sigma_embeds], axis=-1)

        # So if we pass x = [x, cond_embeds], we get [x, cond_embeds, sigma_embeds].
        # The ordering of weights will depend on modification, but logically it contains same info.

        x_combined = jnp.concatenate([x, cond_embeds], axis=-1)

        return super().__call__(x_combined, sigma)


class IdealDenoiser(nn.Module, ModelMixin):
    """Ideal denoiser using the full dataset."""

    def __init__(self, dataset: Array):
        # dataset: (N, D) - full dataset in memory
        self.data = dataset
        self.input_dims = dataset.shape[1:]

    def __call__(self, x: Array, sigma: Array, cond: Optional[Array] = None) -> Array:
        # x: (B, D)
        # sigma: (B,)

        # smalldiffusion logic:
        # sq_diffs = ||x - x0||^2
        # weights = softmax(-sq_diffs / 2 / sigma^2)
        # eps = sum(weights * data)
        # return (x - eps) / sigma

        # Efficient JAX implementation:
        # We need pairwise distances between X (batch) and Data (all).
        # x: (B, D), Data: (N, D)

        # dists: (B, N)
        # expand dims for broadcasting
        # x: (B, 1, D)
        # data: (1, N, D)

        x_expanded = jnp.expand_dims(x, 1)  # (B, 1, D)
        data_expanded = jnp.expand_dims(self.data, 0)  # (1, N, D)

        # Squared Euclidean distance
        # (x - y)^2 = x^2 + y^2 - 2xy
        # But direct substraction broadcast is safer for numerics if memory allows

        diffs = x_expanded - data_expanded  # (B, N, D)
        sq_diffs = jnp.sum(diffs**2, axis=-1)  # (B, N)

        # sigma: (B,)
        sigmas_reshaped = jnp.expand_dims(sigma, 1)  # (B, 1)

        # Weights
        # -sq_diffs / (2 * sigma^2)
        logits = -sq_diffs / (2 * sigmas_reshaped**2)

        weights = jax.nn.softmax(logits, axis=1)  # (B, N)

        # eps_pred = weights @ data -> weighted average of data points?
        # Wait, smalldiffusion logic:
        # eps derived from x0_hat?
        # smalldiffusion:
        #   weights = softmax(...)
        #   eps = einsum(weights, data) -> this is effectively x0_hat, the denoised estimate?
        #   weights is (db, xb) -> (Dataset, Batch).
        #   einsum 'ij,i...->j...' -> sum over dataset index.
        #   So yes, it computes x0_hat = E[x0 | x_t].
        #   Then returns (x - x0_hat) / sigma.

        x0_hat = jnp.einsum("bn,nd->bd", weights, self.data)

        return (x - x0_hat) / sigma.reshape(sigma.shape + (1,) * (x.ndim - 1))
