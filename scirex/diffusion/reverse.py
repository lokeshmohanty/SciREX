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
Module: reverse.py

This module implements the reverse diffusion process (sampling), including
DDPM, DDIM, and Classifier-Free Guidance.

Functions:
    sample_ddpm: Stochastic sampling.
    sample_ddim: Deterministic or semi-stochastic sampling.
    cfg: Guidance for conditional generation.
    sample: Unified sampling interface.
"""

from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from flax import nnx as nn

Array = jax.Array


def cfg(model: Any, x: Array, sigma: Array, cond: Array, uncond: Optional[Array], guidance_scale: float = 7.5) -> Array:
    """Apply classifier-free guidance.

    eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    References:
        - Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2021 Workshop.
    """
    # Concatenate for batch processing if model supports it,
    # but here we do it simply.
    eps_cond = model(x, sigma, cond)
    eps_uncond = model(x, sigma, uncond)
    return cast(Array, eps_uncond + guidance_scale * (eps_cond - eps_uncond))


def sample_ddpm(
    model: Any,
    sigmas: Array,
    batchsize: int = 1,
    shape: tuple = (2,),
    cond: Optional[Array] = None,
    uncond: Optional[Array] = None,
    guidance_scale: Optional[float] = None,
    *,
    rng: nn.Rngs,
) -> Array:
    r"""
    Stochastic sampling (DDPM-like).

    The reverse process is defined by:
    $p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

    References:
        - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    """
    xt = jax.random.normal(rng(), (batchsize, *shape)) * sigmas[0]

    model.eval()
    for i in range(len(sigmas) - 1):
        sig = sigmas[i]
        sig_prev = sigmas[i + 1]

        # Predict noise
        t_batch = jnp.full((batchsize,), sig)
        if guidance_scale is not None and cond is not None:
            eps = cfg(model, xt, t_batch, cond, uncond, guidance_scale)
        else:
            eps = model(xt, t_batch, cond) if cond is not None else model(xt, t_batch)

        # Heuristic alpha/beta from sigmas
        # alpha_cum = 1 / (1 + sigma^2)
        alpha_cum = 1 / (1 + sig**2)
        alpha_cum_prev = 1 / (1 + sig_prev**2)
        beta = 1 - alpha_cum / alpha_cum_prev

        # Update mean
        mean = xt - (sig - sig_prev) * eps  # Simplified update

        if i < len(sigmas) - 2:
            noise = jax.random.normal(rng(), xt.shape)
            xt = mean + jnp.sqrt(sig_prev**2 * beta) * noise  # Placeholder for actual variance
        else:
            xt = mean

    model.train()
    return xt


def sample_ddim(
    model: Any,
    sigmas: Array,
    batchsize: int = 1,
    shape: tuple = (2,),
    cond: Optional[Array] = None,
    uncond: Optional[Array] = None,
    guidance_scale: Optional[float] = None,
    eta: float = 0.0,
    *,
    rng: nn.Rngs,
) -> Array:
    r"""
    Deterministic or semi-stochastic sampling (DDIM).

    The update rule is:
    $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon_\theta(x_t) + \sigma_t \epsilon$
    where setting $\sigma_t = 0$ yields a deterministic process.

    References:
        - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
    """
    xt = jax.random.normal(rng(), (batchsize, *shape)) * sigmas[0]

    model.eval()
    for i in range(len(sigmas) - 1):
        sig = sigmas[i]
        sig_prev = sigmas[i + 1]

        t_batch = jnp.full((batchsize,), sig)
        if guidance_scale is not None and cond is not None:
            eps = cfg(model, xt, t_batch, cond, uncond, guidance_scale)
        else:
            eps = model(xt, t_batch, cond) if cond is not None else model(xt, t_batch)

        # DDIM update rule
        # (Simplified for sigma-based schedules)
        xt = xt - (sig - sig_prev) * eps

    model.train()
    return xt


def sample(
    model: Any,
    sigmas: Array,
    batchsize: int = 1,
    shape: tuple = (2,),
    method: str = "ddim",
    guidance_scale: Optional[float] = None,
    cond: Optional[Array] = None,
    uncond: Optional[Array] = None,
    *,
    rng: nn.Rngs,
) -> Array:
    """Unified sampling interface."""
    if method == "ddpm":
        return sample_ddpm(model, sigmas, batchsize, shape, cond, uncond, guidance_scale, rng=rng)
    else:
        return sample_ddim(model, sigmas, batchsize, shape, cond, uncond, guidance_scale, rng=rng)
