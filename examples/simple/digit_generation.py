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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx as nn

# Load and preprocess the `digits` dataset.
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from scirex.data import create_dataloader
from scirex.diffusion import ScheduleCosine, diffusion_loss, sample
from scirex.training import Trainer
from scirex.transformers import UNet  # Import from module instead of defining here

digits = load_digits()
images = digits.images.astype("float32") / 16.0  # Normalize to [0, 1]
images = images[:, :, :, None]  # Add channel dimension: (n_samples, 8, 8, 1)

# Split data
images_train, images_test = train_test_split(images, test_size=0.2, random_state=42)

# Convert to JAX arrays
images_train = jnp.array(images_train)
images_test = jnp.array(images_test)


def plot_samples(samples: jax.Array, num_samples: int = 10, title: str = "Generated Samples"):
    """Plot a grid of generated samples."""
    _fig, axes = plt.subplots(2, num_samples, figsize=(8, 2))
    for i in range(num_samples):
        axes[0, i].imshow(samples[i, :, :, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(samples[i + num_samples, :, :, 0], cmap="gray")
        axes[1, i].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Model Training on Digits Dataset (Simplified)")
    print("=" * 60)

    # Hyperparameters
    in_channels = 1
    out_channels = 1
    features = 64
    num_steps = 1000
    n_epochs = 500
    learning_rate = 1e-4
    batch_size = 64
    seed = 42

    # Initialize model
    print("\nInitializing U-Net model...")
    rngs = nn.Rngs(seed)
    model = UNet(in_channels, out_channels, features, rngs=rngs)
    print(f"Model created with {features} features")

    # Create diffusion schedule
    print(f"Creating diffusion schedule with {num_steps} steps...")
    schedule = ScheduleCosine(N=num_steps)

    # Create optimizer with warmup + cosine decay
    warmup_steps = 100
    total_steps = n_epochs

    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps),
            optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=total_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )

    optimizer = nn.Optimizer(model, optax.adam(schedule_fn))
    print("Optimizer created with warmup + cosine decay")

    # Create trainer
    print("\nCreating Trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=diffusion_loss(schedule),
        gradient_clip=0.3,
        rngs=rngs,
    )
    print("Trainer initialized")

    # Create dataloader
    train_loader = create_dataloader(images_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    trainer.train(train_loader=train_loader, n_epochs=n_epochs)

    # Plot training history
    print("\nPlotting training history...")
    trainer.visualize()

    # Generate samples using DDPM (standard)
    print("\nGenerating samples with DDPM...")
    model.eval()
    sigmas = schedule.sample_sigmas(steps=num_steps)
    samples_ddpm = sample(model, sigmas, batchsize=20, shape=(8, 8, 1), method="ddpm", rng=rngs)
    plot_samples(samples_ddpm, num_samples=10, title="DDPM Samples (1000 steps)")

    # Generate samples using DDIM (fast)
    print("\nGenerating samples with DDIM (50 steps)...")
    sigmas_fast = schedule.sample_sigmas(steps=50)
    samples_ddim = sample(model, sigmas_fast, batchsize=20, shape=(8, 8, 1), method="ddim", rng=rngs)
    plot_samples(samples_ddim, num_samples=10, title="DDIM Samples (50 steps)")

    print("\n" + "=" * 60)
    print("Training and sampling complete!")
    print("=" * 60)
