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
Module: trainer.py

Simple, modular training interface for NNX models.

Key Features:
    - Clean Trainer API for standard training workflows
    - Automatic history tracking
    - Evaluation support with MultiMetric
    - Visualization capabilities
    - Gradient clipping support

Classes:
    Trainer: Primary training interface

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

"""

import time
from collections.abc import Iterator
from typing import Any, Callable, Optional, TypeAlias, cast

import jax
import jax.errors
import jax.numpy as jnp
from flax import nnx as nn
from loguru import logger
from matplotlib import pyplot as plt

from .utils import moving_average

Array = jax.Array
LossFn: TypeAlias = Callable[..., Any]


class Trainer:
    """Primary training interface for NNX models.

    Provides a complete training workflow with automatic history tracking,
    evaluation, and visualization capabilities.

    Example:
        >>> from scirex.training import Trainer
        >>>
        >>> def loss_fn(model, batch):
        ...     x, y = batch
        ...     logits = model(x)
        ...     loss = jnp.mean((logits - y) ** 2)
        ...     return loss, logits  # Always return (loss, logits)
        >>>
        >>> eval_metrics = nn.MultiMetric(loss=nn.metrics.Average("loss"))
        >>> trainer = Trainer(model, optimizer, loss_fn, eval_metrics)
        >>> trainer.train(train_loader, n_epochs=10, eval_loader=eval_loader)
        >>> trainer.visualize()
    """

    def __init__(
        self,
        model: Any,
        optimizer: nn.Optimizer,
        loss_fn: LossFn,
        eval_metrics: Optional[nn.MultiMetric] = None,
        gradient_clip: Optional[float] = None,
        rngs: Optional[nn.Rngs] = None,
    ):
        """Initialize the Trainer.

        Attributes:
            model: The model instance to be trained.
            optimizer: The Flax NNX optimizer instance (containing model parameters).
            loss_fn: A callable `f(model, batch)` that returns a tuple `(loss, logits)`.
                The `logits` (auxiliary data) are required for evaluation metrics.
            eval_metrics: Optional `nn.MultiMetric` instance. If provided, the trainer
                will automatically track and log these metrics during evaluation.
            gradient_clip: Optional gradient clipping threshold for stability.
        """
        self.model = model
        self.optimizer = optimizer
        self.eval_metrics = eval_metrics
        self.gradient_clip = gradient_clip
        self.rngs = rngs

        # Create JIT-compiled train step (loss_fn always returns logits as aux)
        self.train_step_fn = self._create_train_step(loss_fn)

        # Auto-derive eval_fn from loss_fn
        self.eval_step_fn = self._create_eval_step(loss_fn)

        # Initialize history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "epoch_time": [],
            "total_training_time": [],
        }
        if eval_metrics is not None:
            for metric_name in eval_metrics.compute():
                self.history[f"eval_{metric_name}"] = []

    def _create_train_step(self, loss_fn: LossFn) -> Callable:
        """Create JIT-compiled training step derived from loss_fn."""
        import inspect

        sig = inspect.signature(loss_fn)
        has_rngs = len(sig.parameters) >= 3

        @nn.jit
        def train_step(
            model: nn.Module,
            optimizer: nn.Optimizer,
            batch: Any,
            rngs: Optional[nn.Rngs],
        ) -> Array:
            def internal_loss_fn(m: nn.Module, b: Any, r: Optional[nn.Rngs]) -> tuple[Array, Any]:
                res = loss_fn(m, b, r) if has_rngs else loss_fn(m, b)

                if isinstance(res, (tuple, list)) and len(res) == 2:
                    loss, aux = res
                    return cast(Array, loss), aux
                return cast(Array, res), None

            grad_fn = nn.value_and_grad(internal_loss_fn, has_aux=True)
            (loss_val, _aux), grads = grad_fn(model, batch, rngs)

            if self.gradient_clip is not None:
                clip_val = self.gradient_clip
                grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_val, clip_val), grads)

            optimizer.update(grads)
            return cast(Array, loss_val)

        return train_step

    def _extract_labels(self, batch: Any) -> Optional[Array]:
        """Extract labels from batch - handle both tuple and dict formats."""
        if isinstance(batch, dict):
            for key in ["labels", "label", "y", "target_output"]:
                if key in batch:
                    return cast(Array, batch[key])
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return cast(Array, batch[1])  # Assume (x, y) format
        return None

    def _create_eval_step(self, loss_fn: LossFn) -> Callable:
        """Create JIT-compiled evaluation step derived from loss_fn."""
        import inspect

        sig = inspect.signature(loss_fn)
        has_rngs = len(sig.parameters) >= 3

        @nn.jit
        def eval_step(model: nn.Module, batch: Any, rngs: Optional[nn.Rngs]) -> dict:
            res = loss_fn(model, batch, rngs) if has_rngs else loss_fn(model, batch)

            if isinstance(res, (tuple, list)) and len(res) == 2:
                loss, logits = res
            else:
                loss, logits = res, None

            labels = self._extract_labels(batch)

            result = {"loss": loss, "logits": logits}
            if labels is not None:
                result["labels"] = labels
            return result

        return eval_step

    def train(
        self,
        train_loader: Iterator,
        n_epochs: int,
        eval_loader: Optional[Iterator] = None,
        eval_freq: int = 1,
    ) -> None:
        """Executes the training loop.

        Args:
            train_loader: An iterator yielding batches of training data.
            n_epochs: Total number of epochs to train for.
            eval_loader: Optional iterator yielding batches of evaluation data.
            eval_freq: How often (in epochs) to run evaluation and log progress. Defaults to 1.
        """
        self.model.train()

        total_start_time = time.time()
        logger.info(f"Starting training for {n_epochs} epochs")

        accumulated_time = 0.0

        for epoch in range(n_epochs):
            start_time = time.time()
            # Training epoch
            epoch_losses = []
            for batch in train_loader:
                loss = self.train_step_fn(self.model, self.optimizer, batch, self.rngs)
                epoch_losses.append(float(loss))

            epoch_time = time.time() - start_time
            accumulated_time += epoch_time

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            self.history["train_loss"].append(avg_loss)
            self.history["epoch_time"].append(epoch_time)
            self.history["total_training_time"].append(time.time() - total_start_time)

            log_msg = f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {avg_loss:.4f} | Time: {accumulated_time:.2f}s"

            # Evaluation epoch (optional)
            if eval_loader is not None and (epoch + 1) % eval_freq == 0:
                eval_metrics = self.eval(eval_loader)
                for key, val in eval_metrics.items():
                    if key.startswith("eval_"):
                        metric_name = key.replace("eval_", "")
                        log_msg += f" | {metric_name}: {val:.4f}"

            if (epoch + 1) % eval_freq == 0 or (epoch + 1) == n_epochs:
                logger.info(log_msg)
                accumulated_time = 0.0

    def eval(self, eval_loader: Iterator) -> dict:
        """Evaluate the model.

        Args:
            eval_loader: Evaluation data loader/iterator.

        Returns:
            Dictionary of evaluation metrics with 'eval_' prefix.
        """
        self.model.eval()
        self._reset_metrics()

        # Run evaluation loop and gather results
        all_metrics = []
        eval_start_time = time.time()
        num_batches = 0

        # Determine if eval_loader has length
        total_batches = None
        if hasattr(eval_loader, "__len__"):
            from collections.abc import Sized

            total_batches = len(cast(Sized, eval_loader))

        logger.info(f"Starting evaluation on {total_batches if total_batches else 'unknown'} batches")

        for batch in eval_loader:
            metrics = self.eval_step_fn(self.model, batch, self.rngs)
            if self.eval_metrics is not None:
                self.eval_metrics.update(**metrics)
            else:
                all_metrics.append(metrics)
            num_batches += 1

        eval_dict = self._process_eval_results(all_metrics, num_batches, eval_start_time)
        self._update_history(eval_dict)

        return eval_dict

    def _reset_metrics(self) -> None:
        """Reset evaluation metrics if they exist."""
        if self.eval_metrics is not None:
            self.eval_metrics.reset()

    def _process_eval_results(self, all_metrics: list[dict], num_batches: int, eval_start_time: float) -> dict:
        """Process results into eval_dict."""
        inference_time = (time.time() - eval_start_time) / num_batches if num_batches > 0 else 0.0
        eval_dict = {"inference_time": inference_time}

        if self.eval_metrics is not None:
            computed_metrics = self.eval_metrics.compute()
            for metric_name, metric_value in computed_metrics.items():
                eval_dict[f"eval_{metric_name}"] = float(metric_value)
        elif all_metrics:
            for key in all_metrics[0]:
                if key != "logits":  # Don't average logits
                    avg_val = sum(m[key] for m in all_metrics) / len(all_metrics)
                    eval_dict[f"eval_{key}"] = float(avg_val)

        return eval_dict

    def _update_history(self, eval_dict: dict) -> None:
        """Update history with evaluation results."""
        for key, val in eval_dict.items():
            hist_key = key if key.startswith("eval_") else f"eval_{key}"
            if hist_key not in self.history:
                self.history[hist_key] = []
            self.history[hist_key].append(val)

    def visualize(self, figsize: tuple = (12, 4), save_path: Optional[str] = None) -> "Trainer":
        """Visualize training history.

        Creates plots for training loss and all evaluation metrics.

        Args:
            figsize: Figure size (width, height).
            save_path: Optional path to save the figure.

        Returns:
            Self (for method chaining).
        """

        # Determine number of subplots needed
        eval_metrics = [k for k in self.history if k.startswith("eval_") and k != "eval_inference_time"]
        num_plots = 1 + len(eval_metrics)  # training loss + eval metrics

        if num_plots == 1:
            # Only training loss
            _fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            axes = [ax]
        else:
            # Multiple plots
            _fig, axes = plt.subplots(1, num_plots, figsize=figsize)
            if num_plots == 1:
                axes = [axes]

        # Plot training loss
        train_loss = self.history["train_loss"]
        axes[0].plot(train_loss, label="Train Loss", alpha=0.3)

        # Add moving average
        if len(train_loss) > 10:
            # Adjust span based on length, typically 10% or fixed 100/1000
            span = min(len(train_loss) // 10 + 1, 100)
            smoothed_loss = moving_average(train_loss, span=span)
            axes[0].plot(smoothed_loss, label=f"Moving Avg (span={span})", color="red", linewidth=2)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot evaluation metrics
        for idx, metric_key in enumerate(eval_metrics, start=1):
            metric_name = metric_key.replace("eval_", "").replace("_", " ").title()
            axes[idx].plot(self.history[metric_key], label=metric_name, color="orange")
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_title(f"Evaluation {metric_name}")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")

        plt.show()

        return self
