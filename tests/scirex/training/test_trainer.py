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

"""Tests for Trainer class."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx as nn

from scirex.training import Trainer, moving_average


def test_trainer_init():
    """Test Trainer initialization."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    eval_metrics = nn.MultiMetric(loss=nn.metrics.Average("loss"))

    trainer = Trainer(model, optimizer, loss_fn, eval_metrics)

    assert trainer.model is model
    assert trainer.optimizer is optimizer
    assert "train_loss" in trainer.history
    assert "eval_loss" in trainer.history


def test_trainer_history_tracking():
    """Test that Trainer tracks history correctly."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    eval_metrics = nn.MultiMetric(
        loss=nn.metrics.Average("loss"),
    )

    trainer = Trainer(model, optimizer, loss_fn, eval_metrics)

    # Check history initialized
    assert "train_loss" in trainer.history
    assert "eval_loss" in trainer.history
    assert len(trainer.history["train_loss"]) == 0


def test_trainer_eval():
    """Test Trainer eval method."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    eval_metrics = nn.MultiMetric(loss=nn.metrics.Average("loss"))

    trainer = Trainer(model, optimizer, loss_fn, eval_metrics)

    # Create simple eval loader
    eval_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]

    metrics = trainer.eval(eval_data)

    assert "eval_loss" in metrics
    assert len(trainer.history["eval_loss"]) == 1


def test_trainer_train():
    """Test Trainer train method."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    eval_metrics = nn.MultiMetric(loss=nn.metrics.Average("loss"))
    trainer = Trainer(model, optimizer, loss_fn, eval_metrics)

    # Simple train loader
    train_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))] * 2
    eval_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]

    trainer.train(train_data, n_epochs=2, eval_loader=eval_data)

    assert len(trainer.history["train_loss"]) == 2
    assert len(trainer.history["eval_loss"]) == 2


def test_trainer_visualize(tmp_path):
    """Test Trainer visualize method."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    trainer = Trainer(model, optimizer, loss_fn)
    trainer.history["train_loss"] = [0.5, 0.4, 0.3]

    # Test saving to file
    save_path = tmp_path / "test_plot.png"
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    trainer.visualize(save_path=str(save_path))

    assert save_path.exists()


def test_moving_average():
    """Test moving_average utility function."""

    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Test with small span
    smoothed = moving_average(data, span=2)
    assert len(smoothed) == 5
    # Growing window for first element: avg([1]) = 1.0
    # Window 2 for others: avg([1, 2])=1.5, avg([2, 3])=2.5, avg([3, 4])=3.5, avg([4, 5])=4.5
    expected = [1.0, 1.5, 2.5, 3.5, 4.5]
    assert jnp.allclose(jnp.array(smoothed), jnp.array(expected))

    # Test with large span
    smoothed = moving_average(data, span=10)
    # Span is capped at len(data) = 5
    # Growing window: [1/1, (1+2)/2, (1+2+3)/3, (1+2+3+4)/4, (1+2+3+4+5)/5]
    expected = [1.0, 1.5, 2.0, 2.5, 3.0]
    assert jnp.allclose(jnp.array(smoothed), jnp.array(expected))

    # Test empty data
    assert len(moving_average([], span=10)) == 0


def test_trainer_logging_frequency():
    """Test that logging occurs only at eval_freq intervals."""
    from unittest.mock import patch

    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    trainer = Trainer(model, optimizer, loss_fn)

    # Simple train loader
    train_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]

    # Mock logger in trainer module
    with patch("scirex.training.trainer.logger") as mock_logger:
        # Train for 10 epochs with eval_freq=2
        # Should log at epoch 2, 4, 6, 8, 10 (5 times) + start msg = 6 times
        trainer.train(train_data, n_epochs=10, eval_freq=2)

        # Verify call count
        # 1 start message + 5 epoch messages
        assert mock_logger.info.call_count == 6

        # Verify content of last call (epoch 10)
        args, _ = mock_logger.info.call_args
        assert "Epoch 10/10" in args[0]

        mock_logger.reset_mock()

        # Train for 10 epochs with eval_freq=10
        # Should log at epoch 10 (1 time) + start msg = 2 times
        trainer.train(train_data, n_epochs=10, eval_freq=10)
        assert mock_logger.info.call_count == 2


def test_trainer_default_eval_metrics():
    """Test Trainer with default (None) eval_metrics."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    trainer = Trainer(model, optimizer, loss_fn)

    assert isinstance(trainer.eval_metrics, nn.MultiMetric)
    assert len(trainer.eval_metrics.compute()) == 0

    eval_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]
    metrics = trainer.eval(eval_data)

    # Should still track loss as a default if eval_metrics is empty
    assert "eval_loss" in metrics
    assert "eval_loss" in trainer.history
    assert len(trainer.history["eval_loss"]) == 1


def test_trainer_two_arg_loss_fn():
    """Test Trainer with a 2-argument loss_fn (no RNGs)."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    # 2-arg signature (no rngs)
    def loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        return jnp.mean((logits - y) ** 2), logits

    trainer = Trainer(model, optimizer, loss_fn)
    assert not trainer._use_rngs

    train_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]
    eval_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]

    trainer.train(train_data, n_epochs=1, eval_loader=eval_data)

    assert len(trainer.history["train_loss"]) == 1
    assert "eval_loss" in trainer.history


def test_trainer_three_arg_loss_fn():
    """Test Trainer with a 3-argument loss_fn using RNGs."""
    model = nn.Linear(10, 2, rngs=nn.Rngs(0))
    optimizer = nn.Optimizer(model, optax.adam(0.01))

    # 3-arg signature (using rngs)
    def loss_fn(model, batch, rngs):
        x, y = batch
        # Use RNG to add some noise during loss calculation (dummy usage)
        noise = jax.random.normal(rngs(), x.shape)
        logits = model(x + 0.01 * noise)
        return jnp.mean((logits - y) ** 2), logits

    trainer = Trainer(model, optimizer, loss_fn, rngs=nn.Rngs(0))
    assert trainer._use_rngs

    train_data = [(jnp.ones((4, 10)), jnp.ones((4, 2)))]
    trainer.train(train_data, n_epochs=1)

    assert len(trainer.history["train_loss"]) == 1
