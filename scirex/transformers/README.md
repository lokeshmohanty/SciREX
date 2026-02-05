# SciREX Transformers

This module implements Transformer architectures using Flax NNX.

## Key Features

- **EncoderBlock**: Transformer Encoder Block
- **DecoderBlock**: Transformer Decoder Block
- **PositionalEmbedding**: Standard Positional Embedding
- **EncoderModel**: Transformer Encoder-only Model
- **EncoderDecoderModel**: Transformer Encoder-Decoder Model

## Examples

Check out the following examples in `examples/simple/` to see how to use these models:

- [Machine Translation](../../examples/simple/machine_translation.py): Translates Spanish to English using `EncoderDecoderModel`.
- [Text Classification](../../examples/simple/text_classification.py): Classifies IMDB movie reviews using `EncoderModel`.
