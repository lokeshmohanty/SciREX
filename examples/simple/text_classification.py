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
Module: text_classification.py

This module implements a Text Classification example using the Transformer architecture.

Key Features:
    - Uses EncoderModel from scirex.transformers
    - Classifies IMDB movie reviews (Sentiment Analysis)
    - Implements custom data loading and preprocessing with Grain
    - Includes training and evaluation loops

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

"""

import io
import json
import textwrap
import typing

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import requests
from flax import nnx as nn

from scirex.training import Trainer
from scirex.transformers import EncoderModel

n_epochs = 10  # Number of epochs during training.
learning_rate = 0.0001  # The learning rate.
momentum = 0.9  # Momentum for Adam.
bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


def prepare_imdb_dataset(num_words: int, index_from: int, oov_char: int = 2) -> tuple:
    """Download and preprocess the IMDB dataset from TensorFlow Datasets.

    Args:
        num_words (int): The maximum number of words to keep in the vocabulary.
        index_from (int): The starting index for word indices.
        oov_char (int): The character to use for out-of-vocabulary words. Defaults to 2.

    Returns:
        A tuple containing the training and test sets with labels.
    """
    response = requests.get("https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz", timeout=5)
    response.raise_for_status()

    # The training and test sets.
    with np.load(io.BytesIO(response.content), allow_pickle=True) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test, y_test = f["x_test"], f["y_test"]

    # Shuffle the training and test sets.
    rng = np.random.RandomState(113)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    y_test = y_test[indices]

    # Adjust word indices to start from the specified index.
    x_train = [[w + index_from for w in x] for x in x_train]
    x_test = [[w + index_from for w in x] for x in x_test]

    # Combine training and test sets, then truncates/pads sequences.
    xs = x_train + x_test
    labels = np.concatenate([y_train, y_test])
    xs = [[w if w < num_words else oov_char for w in x] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
    x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]

    return (x_train, y_train), (x_test, y_test)


def pad_sequences(arrs: typing.Iterable, max_len: int) -> np.ndarray:
    """Pad array sequences to a fixed length.

    Args:
        arrs (typing.Iterable): A list of arrays.
        max_len (int): The desired maximum length.

    Returns:
        A NumPy array of padded sequences.
    """
    # Ensure that each sample is the same length
    result = []
    for arr in arrs:
        arr_len = len(arr)
        if arr_len < max_len:
            padded_arr = np.pad(arr, (max_len - arr_len, 0), "constant", constant_values=0)
        else:
            padded_arr = np.array(arr[arr_len - max_len :])
        result.append(padded_arr)

    return np.asarray(result)


def preprocess(x_train, x_test, y_train, y_test):
    # Set the batch sizes for the dataset.
    seed = 12
    train_batch_size = 128
    test_batch_size = 2 * train_batch_size
    n_batches = len(x_train) // train_batch_size

    # Implement a custom data source for Grain to handle the IMDB dataset.
    class DataSource(grain.RandomAccessDataSource):
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def __getitem__(self, idx):
            return {"encoded_indices": self._x[idx], "label": self._y[idx]}

        def __len__(self):
            return len(self._x)

    # Instantiate the training and test set data sources.
    train_source = DataSource(x_train, y_train)
    test_source = DataSource(x_test, y_test)

    # Define `grain.IndexSampler`s for training and testing data.
    train_sampler = grain.IndexSampler(
        len(train_source),
        shuffle=True,
        seed=seed,
        shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup.
        num_epochs=1,  # Iterate over the dataset for one epoch.
    )
    test_sampler = grain.IndexSampler(
        len(test_source),
        shuffle=False,
        seed=seed,
        shard_options=grain.NoSharding(),  # No sharding since this is a single-device setup.
        num_epochs=1,  # Iterate over the dataset for one epoch.
    )

    # Create `grain.DataLoader`s for training and test sets.
    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,  # A `grain.IndexSampler` determining how to access the data.
        worker_count=4,  # The number of child processes launched to parallelize the transformations.
        worker_buffer_size=2,  # The number count of output batches to produce in advance per worker.
        operations=[
            grain.Batch(train_batch_size, drop_remainder=True),
        ],
    )
    test_loader = grain.DataLoader(
        data_source=test_source,
        sampler=test_sampler,  # A `grain.IndexSampler` to determine how to access the data.
        worker_count=4,  # The number of child processes launched to parallelize the transformations.
        worker_buffer_size=2,  # The number count of output batches to produce in advance per worker.
        operations=[
            grain.Batch(test_batch_size),
        ],
    )
    return train_loader, test_loader, n_batches


def loss_fn(model: nn.Module, batch: dict[str, jax.Array]):
    """Loss function for training. Always returns (loss, logits)."""
    batch_tokens = jnp.array(batch["encoded_indices"])
    labels = jnp.array(batch["label"], dtype=jnp.int32)
    logits = model(batch_tokens)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits  # Always return (loss, logits)


def show_reviews(indices: list[int]) -> None:
    label_to_str = lambda x: "positive" if x else "negative"
    for idx in indices:
        x = x_test[idx][x_test[idx] != 0]
        y = y_test[idx]
        y_pred = model(x_test[idx][None, :]).argmax()
        review = ""
        for w_x in x:
            idx = w_x - index_from if w_x >= index_from else w_x
            review += f"{word_map[idx]} "

        print("Review:")
        for line in textwrap.wrap(review):
            print(line)
        print("Predicted sentiment: ", label_to_str(y_pred))
        print("Actual sentiment: ", label_to_str(y), "\n")


if __name__ == "__main__":
    index_from = 3  # Ensures that 0 encodes the padding token.
    vocab_size = 20000  # Considers only the top 20,000 words.
    context_size = 200  # Limits each review to the first 200 words.

    # Instantiate the training and test sets.
    (x_train, y_train), (x_test, y_test) = prepare_imdb_dataset(num_words=vocab_size, index_from=index_from)
    print(len(x_train), "Training sequences")
    print(len(x_test), "Validation sequences")

    # Pad array sequences to a fixed length.
    x_train = pad_sequences(x_train, max_len=context_size)
    x_test = pad_sequences(x_test, max_len=context_size)

    train_loader, test_loader, n_batches = preprocess(x_train, x_test, y_train, y_test)

    d_model = 32  # The embedding size for each token.
    n_heads = 2  # The number of attention heads.
    d_hidden = 32  # The hidden layer size in the feed-forward network inside the transformer.
    rng = nn.Rngs(0)

    model = EncoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate=0.1, rngs=rng)
    model.head = nn.Sequential(
        lambda x: jnp.mean(x, axis=(1,)),  # global average pooling
        nn.Dropout(0.1, rngs=rng),
        nn.Linear(d_model, 20, rngs=rng),
        nn.relu,
        nn.Dropout(0.1, rngs=rng),
        nn.Linear(20, 2, rngs=rng),
        jax.nn.softmax,
    )

    optimizer = nn.Optimizer(model, optax.adam(learning_rate, momentum))

    eval_metrics = nn.MultiMetric(
        loss=nn.metrics.Average("loss"),
        accuracy=nn.metrics.Accuracy(),
    )

    # Create trainer with automatic JIT compilation and history tracking
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,  # Always returns (loss, logits)
        eval_metrics=eval_metrics,  # eval_fn auto-derived!
    )

    # Train the model
    print("\nTraining...")
    trainer.train(
        train_loader=train_loader,
        n_epochs=n_epochs,
        eval_loader=test_loader,
        eval_freq=1,
    )

    # Visualize training history
    trainer.visualize()

    # Print final metrics
    print(f"\nFinal train loss: {trainer.history['train_loss'][-1]:.4f}")
    print(f"Final eval accuracy: {trainer.history['eval_accuracy'][-1]:.4f}")

    # Show some predictions
    print("\nSample predictions:")
    response = requests.get(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json", timeout=5
    )
    response.raise_for_status()
    word_map = {v: k for k, v in json.loads(response.content).items()}
    show_reviews([0, 500, 600, 1000, 1800, 2000])
