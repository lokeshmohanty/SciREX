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

r"""
Module: forward.py

This module implements the forward diffusion process, including noise schedules
and sigma embeddings.

The forward process is defined by:
$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$
where $\bar{\alpha}_t$ is the cumulative product of $1 - \beta_s$ for $s=1 \dots t$.

Key Classes:
    Schedule: Base class for noise schedules
    ScheduleLogLinear, ScheduleDDPM, ScheduleLDM, ScheduleCosine, ScheduleSigmoid
    TimeInputMLP: Simple MLP for toy diffusion examples.

Functions:
    get_sigma_embeds: Sinusoidal embeddings for sigma values.
    diffusion_loss: Generate a loss function for the Trainer.

References:
    - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
    - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021.
    - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx as nn

Array = jax.Array


def sigmas_from_betas(betas: Array) -> Array:
    """Convert beta schedule to sigma values."""
    alpha_cumprod = jnp.cumprod(1.0 - betas)
    return jnp.sqrt(1.0 / alpha_cumprod - 1.0)


class Schedule:
    """Base class for diffusion noise schedules parameterized by sigma."""

    def __init__(self, sigmas: Array):
        self.sigmas = sigmas

    def __getitem__(self, i: Any) -> Array:
        return self.sigmas[i]

    def __len__(self) -> int:
        return len(self.sigmas)

    def sample_sigmas(self, steps: int) -> Array:
        """Generates a decreasing sigma schedule for sampling."""
        N = len(self)
        indices = jnp.round(N * (1 - jnp.arange(steps) / steps)).astype(jnp.int32) - 1
        indices = jnp.concatenate([indices, jnp.array([0])])
        return self.sigmas[indices]

    def sample_batch(self, batch_size: int, rng: nn.Rngs) -> Array:
        """Sample random sigma values for a batch during training."""
        indices = jax.random.randint(rng(), (batch_size,), 0, len(self))
        return self.sigmas[indices]


class ScheduleLogLinear(Schedule):
    """Log-linear sigma schedule."""

    def __init__(self, N: int, sigma_min: float = 0.02, sigma_max: float = 10.0):
        log_min = jnp.log10(sigma_min)
        log_max = jnp.log10(sigma_max)
        sigmas = 10.0 ** jnp.linspace(log_min, log_max, N)
        super().__init__(sigmas)


class ScheduleDDPM(Schedule):
    """DDPM beta schedule converted to sigma values."""

    def __init__(self, N: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        betas = jnp.linspace(beta_start, beta_end, N)
        sigmas = sigmas_from_betas(betas)
        super().__init__(sigmas)


class ScheduleLDM(Schedule):
    """LDM (Latent Diffusion Model) beta schedule."""

    def __init__(self, N: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        betas = jnp.linspace(beta_start**0.5, beta_end**0.5, N) ** 2
        sigmas = sigmas_from_betas(betas)
        super().__init__(sigmas)


class ScheduleSigmoid(Schedule):
    """Sigmoid beta schedule."""

    def __init__(self, N: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        t = jnp.linspace(-6, 6, N)
        betas = 1.0 / (1.0 + jnp.exp(-t)) * (beta_end - beta_start) + beta_start
        sigmas = sigmas_from_betas(betas)
        super().__init__(sigmas)


class ScheduleCosine(Schedule):
    r"""
    Cosine beta schedule.

    The $\bar{\alpha}_t$ values follow:
    $\bar{\alpha}_t = \frac{f(t)}{f(0)}$ where $f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$

    References:
        - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021.
    """

    def __init__(self, N: int = 1000, max_beta: float = 0.999):
        def alpha_bar(t: Array) -> Array:
            return jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2

        ts = jnp.arange(N) / N
        alpha_bars = alpha_bar(ts)
        alpha_bars_next = jnp.concatenate([alpha_bars[1:], jnp.array([0.0])])
        betas = 1.0 - alpha_bars_next / alpha_bars
        betas = jnp.clip(betas, 0, max_beta)
        sigmas = sigmas_from_betas(betas)
        super().__init__(sigmas)


def get_sigma_embeds(sigma: Array, scaling_factor: float = 0.5) -> Array:
    """Get sinusoidal embeddings for sigma values."""
    sigma = jnp.log(sigma) * scaling_factor
    return jnp.stack([jnp.sin(sigma), jnp.cos(sigma)], axis=-1)


def diffusion_loss(schedule: Schedule) -> Callable:
    """Generate a loss function for the Trainer."""

    def loss_fn(model: Any, batch: Any, rngs: nn.Rngs) -> tuple[Array, Array]:
        x0, cond = batch if isinstance(batch, (tuple, list)) else (batch, None)

        batch_size = x0.shape[0]
        sigma = schedule.sample_batch(batch_size, rngs)
        sigma_reshaped = sigma.reshape((batch_size,) + (1,) * (x0.ndim - 1))
        eps = jax.random.normal(rngs(), x0.shape)

        xt = x0 + sigma_reshaped * eps
        eps_pred = model(xt, sigma, cond) if cond is not None else model(xt, sigma)

        return jnp.mean((eps_pred - eps) ** 2), eps_pred

    return loss_fn
