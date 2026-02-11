# SciREX Transformers

High-performance Transformer architectures and components implemented using Flax NNX.

## Key Features

- **Modular Components**: Reusable blocks like `EncoderBlock`, `DecoderBlock`, `MultiHeadAttention`, and `FeedForward`.
- **Pre-configured Models**: Out-of-the-box support for `EncoderModel` (classification/embedding) and `EncoderDecoderModel` (translation/seq2seq).
- **Vision Transformers**: Core components for vision tasks, including `UNet` with time-conditioning for diffusion.
- **Sinusoidal Embeddings**: Built-in support for positional and time-based embeddings.

## Scientific Background

SciREX Transformers are based on the implementation of the classic Transformer architecture and specialized variants like U-Net for diffusion.

### Attention Mechanism
The core of the Transformer is Scaled Dot-Product Attention:
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### U-Net Architecture
For image-based diffusion, SciREX uses a U-Net with time conditioning $t$:
$\epsilon_\theta(x_t, t) = \text{UNet}(x_t, \text{Embed}(t))$

### References
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
- Ronneberger, O., et al. (2015). [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). MICCAI.
- Dosovitskiy, A., et al. (2021). [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). ICLR.

## Quick Start: EncoderModel

```python
from flax import nnx as nn
from scirex.transformers import EncoderModel

# Create an Encoder model for sequence classification
model = EncoderModel(
    context_size=128,
    vocab_size=10000,
    d_model=256,
    d_hidden=512,
    n_heads=8,
    dropout_rate=0.1,
    rngs=nn.Rngs(0)
)

# Forward pass
logits = model(tokens, deterministic=True)
```

## Quick Start: EncoderDecoderModel

```python
from scirex.transformers import EncoderDecoderModel

# Create a sequence-to-sequence model
model = EncoderDecoderModel(
    context_size=64,
    vocab_size=5000,
    d_model=128,
    d_hidden=256,
    n_heads=4,
    dropout_rate=0.2,
    rngs=nn.Rngs(42)
)

# Forward pass with source and target sequences
logits = model(source_tokens, target_tokens, deterministic=False)
```

## API Reference

### Models
- `EncoderModel`: Transformer encoder with a linear head for projection.
- `EncoderDecoderModel`: Classic encoder-decoder structure for translation tasks.
- `UNet`: A specialized U-Net implementation designed for image-based diffusion models.

### Layers and Blocks
- `EncoderBlock`: A single transformer encoder layer.
- `DecoderBlock`: A single transformer decoder layer with cross-attention.
- `PositionalEmbedding`: Learnable or fixed positional information injection.

## File Structure

```
.
├── blocks.py         # Modular Transformer components including Attention mechanisms and Feed-Forward layers.
├── embeddings.py     # Positional and patch embedding implementations for various Transformer architectures.
├── models.py         # High-level Transformer model definitions and configurations.
├── unet.py           # U-Net architecture implementation.
└── __init__.py       # Package initialization and component exports.
```

## Examples
- [Machine Translation](../../examples/simple/machine_translation.py): Translates Spanish to English using `EncoderDecoderModel`.
- [Text Classification](../../examples/simple/text_classification.py): Classifies IMDB movie reviews using `EncoderModel`.
