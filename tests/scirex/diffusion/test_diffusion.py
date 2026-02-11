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

"""Tests for diffusion module."""

import jax.numpy as jnp
from flax import nnx as nn

from scirex.diffusion import sample_ddim, sample_ddpm


class MockModel(nn.Module):
    def __init__(self, rngs):
        pass

    def __call__(self, x, sigma, cond=None):
        # Mock output: just return something shaped like x
        # eps prediction
        out = jnp.zeros_like(x)
        if cond is not None:
            # Broadcast cond if needed or just add
            # Assume cond is compatible or ignore content
            pass
        return out


def test_sample_ddpm():
    rngs = nn.Rngs(0)
    model = MockModel(rngs)
    sigmas = jnp.array([10.0, 1.0, 0.1, 0.01])
    samples = sample_ddpm(model, sigmas, batchsize=2, shape=(1,), rng=rngs)
    assert samples.shape == (2, 1)
    assert not jnp.isnan(samples).any()


def test_sample_ddim():
    rngs = nn.Rngs(0)
    model = MockModel(rngs)
    sigmas = jnp.array([10.0, 1.0, 0.1, 0.01])
    samples = sample_ddim(model, sigmas, batchsize=2, shape=(1,), rng=rngs)
    assert samples.shape == (2, 1)
    assert not jnp.isnan(samples).any()


def test_sample_cfg_implicit_uncond():
    """Test that CFG runs (doesn't crash) when uncond is None but guidance_scale is provided."""
    rngs = nn.Rngs(0)
    model = MockModel(rngs)
    sigmas = jnp.array([10.0, 1.0, 0.1, 0.01])
    # cond shape (2, 1), x shape (2, 1)
    cond = jnp.ones((2, 1))

    # This should trigger the code path: if guidance_scale is not None and cond is not None:
    # calls cfg(..., uncond=None)
    # MockModel handles cond=None (implicitly uncond) by doing nothing specific, but it shouldn't crash.
    samples = sample_ddim(model, sigmas, batchsize=2, shape=(1,), cond=cond, uncond=None, guidance_scale=5.0, rng=rngs)

    assert samples.shape == (2, 1)
    assert not jnp.isnan(samples).any()
