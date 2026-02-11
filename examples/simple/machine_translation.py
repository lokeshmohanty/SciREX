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
Module: machine_translation.py

This module implements a Machine Translation example using the Transformer architecture.

Key Features:
    - Uses EncoderDecoderModel from scirex.transformers
    - Translates Spanish to English
    - Implements custom data loading and preprocessing with Grain
    - Includes training and evaluation loops

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

"""

import argparse
import pathlib
import random
import re
import string
import tempfile
import zipfile

import grain.python as grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import requests
import tiktoken
from flax import nnx as nn

from scirex.training import Trainer
from scirex.transformers import EncoderDecoderModel

## Hyperparameters
rng = nn.Rngs(0)
d_model = 256
d_hidden = 2048
n_heads = 8
dropout_rate = 0.5
context_size = 20
learning_rate = 1.5e-3
n_epochs = 10

bar_format = "{desc}[{n_fmt}/{total_fmt}]{postfix} [{elapsed}<{remaining}]"


def get_data():
    url = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        zip_file_path = temp_path / "spa-eng.zip"

        response = requests.get(url, timeout=5)
        zip_file_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        text_file = temp_path / "spa-eng" / "spa.txt"

        with open(text_file) as f:
            lines = f.read().split("\n")[:-1]
        text_pairs = []
        for line in lines:
            eng, spa = line.split("\t")
            spa = "[start] " + spa + " [end]"
            text_pairs.append((eng, spa))

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples :]

    print(f"{len(text_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

    return text_pairs, train_pairs, val_pairs, test_pairs


def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    lowercase = input_string.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", lowercase)


def tokenize_and_pad(text, tokenizer, max_length):
    tokens = tokenizer.encode(text)[:max_length]
    padded = tokens + [0] * (max_length - len(tokens)) if len(tokens) < max_length else tokens
    return padded


def preprocess(train_paris, val_paris, test_paris):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def format_dataset(eng, spa, tokenizer, context_size):
        """Standardizes, tokenizes and pads the input/target sequences.

        Args:
            eng: Spanish input string.
            spa: English target string.
            tokenizer: tiktoken encoder.
            context_size: Fixed length for padding.

        Returns:
            Dictionary with encoder_inputs, decoder_inputs, and target_output.
        """
        eng = custom_standardization(eng)
        spa = custom_standardization(spa)
        eng = tokenize_and_pad(eng, tokenizer, context_size)
        spa = tokenize_and_pad(spa, tokenizer, context_size)
        return {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:-1],  # Shifted right for teacher forcing
            "target_output": spa[1:],  # Shifted left for prediction target
        }

    context_size = 20

    train_data = [format_dataset(eng, spa, tokenizer, context_size) for eng, spa in train_pairs]
    val_data = [format_dataset(eng, spa, tokenizer, context_size) for eng, spa in val_pairs]
    # test_data = [format_dataset(eng, spa, tokenizer, context_size) for eng, spa in test_pairs]
    print(train_data[135])

    batch_size = 512  # set here for the loader and model train later on
    n_batches = len(train_data) // batch_size

    train_sampler = grain.IndexSampler(
        len(train_data),
        shuffle=True,
        seed=12,  # Seed for reproducibility
        shard_options=grain.NoSharding(),  # No sharding since it's a single-device setup
        num_epochs=1,  # Iterate over the dataset for one epoch
    )
    val_sampler = grain.IndexSampler(
        len(val_data),
        shuffle=False,
        seed=12,
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    class CustomPreprocessing(grain.MapTransform):
        def __init__(self):
            pass

        def map(self, data):
            return {
                "encoder_inputs": np.array(data["encoder_inputs"]),
                "decoder_inputs": np.array(data["decoder_inputs"]),
                "target_output": np.array(data["target_output"]),
            }

    train_loader = grain.DataLoader(
        data_source=train_data,
        sampler=train_sampler,  # Sampler to determine how to access the data
        worker_count=4,  # Number of child processes launched to parallelize the transformations
        worker_buffer_size=2,  # Count of output batches to produce in advance per worker
        operations=[
            CustomPreprocessing(),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
    )
    val_loader = grain.DataLoader(
        data_source=val_data,
        sampler=val_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=[
            CustomPreprocessing(),
            grain.Batch(batch_size=batch_size),
        ],
    )
    return train_loader, val_loader, n_batches, tokenizer


def decode_sequence(input_sentence):

    input_sentence = custom_standardization(input_sentence)
    tokenized_input_sentence = tokenize_and_pad(input_sentence, tokenizer, context_size)

    decoded_sentence = "[start"
    for i in range(context_size):
        tokenized_target_sentence = tokenize_and_pad(decoded_sentence, tokenizer, context_size)[:-1]
        predictions = model(jnp.array([tokenized_input_sentence]), jnp.array([tokenized_target_sentence]))

        sampled_token_index = np.argmax(predictions[0, i, :]).item(0)
        sampled_token = tokenizer.decode([sampled_token_index])
        decoded_sentence += "" + sampled_token

        if decoded_sentence[-5:] == "[end]":
            break
    return decoded_sentence


def visualize(test_pairs, history):
    _fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].set_title("Loss value on eval set")
    axs[0].plot(history["eval_loss"])
    axs[1].set_title("Accuracy on eval set")
    axs[1].plot(history["eval_accuracy"])

    test_eng_texts = [pair[0] for pair in test_pairs]
    test_result_pairs = []
    for _ in range(10):
        input_sentence = random.choice(test_eng_texts)
        translated = decode_sequence(input_sentence)

        test_result_pairs.append(f"[Input]: {input_sentence} [Translation]: {translated}")

    for i in test_result_pairs:
        print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Translation Example with Profiling")
    parser.add_argument("--profile-compute", type=str, help="Directory to save compute traces")
    parser.add_argument("--profile-memory", type=str, help="Filename to save memory profile")
    args = parser.parse_args()

    text_paris, train_pairs, val_pairs, test_pairs = get_data()
    train_loader, val_loader, n_batches, tokenizer = preprocess(train_pairs, val_pairs, test_pairs)
    vocab_size = tokenizer.n_vocab

    eval_metrics = nn.MultiMetric(
        loss=nn.metrics.Average("loss"),
        accuracy=nn.metrics.Accuracy(),
    )
    train_metrics_history = {
        "train_loss": [],
    }
    eval_metrics_history = {
        "test_loss": [],
        "test_accuracy": [],
    }

    model = EncoderDecoderModel(context_size, vocab_size, d_model, d_hidden, n_heads, dropout_rate, rngs=rng)
    optimizer = nn.Optimizer(model, optax.adamw(learning_rate))

    def loss_fn(model, batch):
        """Loss function for training."""
        encoder_inputs = jnp.array(batch["encoder_inputs"])
        decoder_inputs = jnp.array(batch["decoder_inputs"])
        target_output = jnp.array(batch["target_output"])
        logits = model(encoder_inputs, decoder_inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=target_output)
        return jnp.mean(loss), logits

    # Create trainer (uses scirex.training.Trainer)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_metrics=eval_metrics,
    )

    print(f"\nTraining for {n_epochs} epochs...")
    if args.profile_compute:
        print(f"Profiling compute trace to {args.profile_compute}...")
        with jax.profiler.trace(args.profile_compute):
            trainer.train(
                train_loader=train_loader,
                n_epochs=n_epochs,
                eval_loader=val_loader,
                eval_freq=1,
            )
            # Ensure everything is finished before stopping trace
            jax.block_until_ready(trainer.model)
    else:
        trainer.train(
            train_loader=train_loader,
            n_epochs=n_epochs,
            eval_loader=val_loader,
            eval_freq=1,
        )

    if args.profile_memory:
        print(f"Saving memory profile to {args.profile_memory}...")
        # Ensure model is ready before profiling memory
        jax.block_until_ready(trainer.model)
        jax.profiler.save_device_memory_profile(args.profile_memory)

    # Print timing summary
    avg_epoch_time = sum(trainer.history["epoch_time"]) / len(trainer.history["epoch_time"])
    print(f"\nAverage time per epoch: {avg_epoch_time:.2f}s")
    if trainer.history.get("eval_inference_time"):
        avg_inference_time = sum(trainer.history["eval_inference_time"]) / len(trainer.history["eval_inference_time"])
        print(f"Average inference time per batch: {avg_inference_time * 1000:.2f}ms")

    plt.plot(trainer.history["train_loss"], label="Loss value during the training")
    plt.yscale("log")
    plt.legend()

    visualize(test_pairs, trainer.history)
