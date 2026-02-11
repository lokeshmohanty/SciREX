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

"""Tests for diffusion helper modules."""

import jax
import jax.numpy as jnp
from flax import nnx as nn

from scirex.diffusion.helpers import ConditionalMLP, IdealDenoiser, SigmaEmbedderSinCos, TimeInputMLP


def test_time_input_mlp():
    rngs = nn.Rngs(0)
    mlp = TimeInputMLP(dim=2, hidden_dims=(16, 16), rngs=rngs)

    x = jax.random.normal(rngs(), (4, 2))
    sigma = jnp.array([0.1, 0.5, 1.0, 2.0])

    out = mlp(x, sigma)
    assert out.shape == (4, 2)


def test_conditional_mlp():
    rngs = nn.Rngs(0)
    mlp = ConditionalMLP(dim=2, hidden_dims=(16, 16), cond_dim=4, num_classes=5, rngs=rngs)

    x = jax.random.normal(rngs(), (4, 2))
    sigma = jnp.array([0.1, 0.5, 1.0, 2.0])
    cond = jnp.array([0, 1, 2, 4])

    out = mlp(x, sigma, cond=cond)
    assert out.shape == (4, 2)

    # Test internal CFG generation logic via mixin if applicable
    cond_embed_linear = mlp.cond_embed
    assert cond_embed_linear.null_cond == 5

    out_cfg = mlp.predict_eps_cfg(x, sigma, cond, cfg_scale=1.5)
    assert out_cfg.shape == (4, 2)


def test_sigma_embedder():
    rngs = nn.Rngs(0)
    embedder = SigmaEmbedderSinCos(hidden_size=16, rngs=rngs)

    sigma = jnp.array([0.1, 10.0])
    emb = embedder(sigma)
    assert emb.shape == (2, 16)


def test_ideal_denoiser():
    dataset = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])

    denoiser = IdealDenoiser(dataset)

    # Test point near [1.0, 1.0] with small noise
    x = jnp.array([[1.1, 0.9]])
    sigma = jnp.array([0.1])

    # The ideal denoiser should predict epsilon such that x - sigma*eps ~ [1.0, 1.0]
    # eps_pred = (x - x0_hat) / sigma
    # x0_hat = x - sigma * eps_pred

    eps_pred = denoiser(x, sigma)
    x0_hat = x - sigma * eps_pred

    # Check if close to [1.0, 1.0]
    assert jnp.allclose(x0_hat, jnp.array([[1.0, 1.0]]), atol=0.2)
