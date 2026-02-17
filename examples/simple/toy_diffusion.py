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
Toy diffusion example to generate 2D spiral data.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx as nn

from scirex.data import create_dataloader
from scirex.diffusion import ScheduleCosine, diffusion_loss, sample
from scirex.diffusion.helpers import TimeInputMLP
from scirex.training import Trainer

# Set up plotting style
plt.style.use("seaborn-v0_8-muted")


def create_spiral_data(tmin=0, tmax=5 * jnp.pi, n_points: int = 100) -> jnp.ndarray:
    """Create a 2D spiral dataset."""
    t = jnp.linspace(tmin, tmax, n_points)
    x = t * jnp.cos(t) / tmax
    y = t * jnp.sin(t) / tmax
    data = jnp.stack([x, y], axis=1)
    # Normalize to [-1, 1]
    data = data / jnp.abs(data).max()
    return data


def main():
    # 1. Data Generation
    print("Generating data...")
    train_data_2d = create_spiral_data(n_points=2000)

    # Save data plot
    plt.figure(figsize=(6, 6))
    plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], alpha=0.6, s=10)
    plt.title("2D Spiral Training Data")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.savefig("spiral_data.png")
    print("Saved training data plot to 'spiral_data.png'")
    plt.close()

    # 2. Model Architecture
    rngs = nn.Rngs(0)
    model_2d = TimeInputMLP(dim=2, output_dim=2, hidden_dims=(128, 128, 128), rngs=rngs)

    # Create trainer
    schedule = ScheduleCosine(N=1000)
    optimizer = nn.Optimizer(model_2d, optax.adam(1e-3))

    trainer = Trainer(
        model=model_2d,
        optimizer=optimizer,
        loss_fn=diffusion_loss(schedule),
        rngs=rngs,
    )

    # 3. Training
    print("Starting training...")
    train_loader = create_dataloader(
        np.array(train_data_2d),
        batch_size=128,
        shuffle=True,
        seed=0,
        worker_count=0,  # Run in main process to avoid potential errors
    )

    trainer.train(train_loader=train_loader, n_epochs=100)
    print("Training complete.")

    # 4. Sampling
    print("Sampling from model...")
    sigmas = schedule.sample_sigmas(20)  # 20 steps

    # DDIM Sampling
    print("Running DDIM sampling...")
    sampled_data_ddim = sample(
        model_2d,
        sigmas,
        batchsize=1000,
        shape=(2,),
        method="ddim",
        rng=nn.Rngs(42).sample,
    )

    # Generalized Sampling (Momentum)
    print("Running Generalized sampling (Momentum, gamma=2.0)...")
    sampled_data_gen = sample(
        model_2d,
        sigmas,
        batchsize=1000,
        shape=(2,),
        method="generalized",
        gamma=2.0,
        rng=nn.Rngs(43).sample,
    )

    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot DDIM
    plt.subplot(1, 2, 1)
    plt.scatter(
        sampled_data_ddim[:, 0],
        sampled_data_ddim[:, 1],
        alpha=0.6,
        s=10,
        label="Sampled (DDIM)",
    )
    plt.scatter(
        train_data_2d[:, 0],
        train_data_2d[:, 1],
        alpha=0.1,
        s=10,
        label="Real",
    )
    plt.title("DDIM Generation")
    plt.legend()
    plt.axis("equal")

    # Plot Generalized
    plt.subplot(1, 2, 2)
    plt.scatter(
        sampled_data_gen[:, 0],
        sampled_data_gen[:, 1],
        alpha=0.6,
        s=10,
        label="Sampled (Gamma=2)",
        c="orange",
    )
    plt.scatter(
        train_data_2d[:, 0],
        train_data_2d[:, 1],
        alpha=0.1,
        s=10,
        label="Real",
    )
    plt.title("Generalized Generation (Momentum)")
    plt.legend()
    plt.axis("equal")

    plt.tight_layout()
    plt.savefig("spiral_generated_comparison.png")
    print("Saved generated data plot to 'spiral_generated_comparison.png'")
    plt.close()


if __name__ == "__main__":
    main()
