# Examples: Simple Tasks

This directory contains simple, end-to-end examples demonstrating core SciREX modules.

## File Structure

```
.
├── toy_diffusion.ipynb     # toy diffusion
├── digit_generation.py     # Handwritten digits generation using VAE.
├── machine_translation.py  # Spanish to English translation.
├── text_classification.py  # Sentiment analysis on IMDB.
└── README.md               # This documentation file.
```

## Available Examples

- **[Toy Diffusion](./toy_diffusion.ipynb)**: Toy diffusion problem (Adapted from [Chenyang](https://www.chenyang.co/diffusion.html))
- **[Digits Generator](./digit_generation.py)**: A Variational Autoencoder (VAE) for generating handwritten digits. (Adapted from jaxaistack docs)
- **[Machine Translation](./machine_translation.py)**: Spanish to English translation using the `EncoderDecoderModel`. (Adapted from jaxaistack docs)
- **[Text Classification](./text_classification.py)**: Sentiment analysis on the IMDB dataset using the `EncoderModel`. (Adapted from jaxaistack docs)

## How to Run

Most examples can be run directly using `uv`:

```bash
uv run python examples/simple/<example_name>.py
```

*Note: Some examples may require downloading external datasets on first run (e.g., IMDB, Spanish-English pairs).*
