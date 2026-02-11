# SciREX: Scientific Research and Engineering eXcellence

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://scirex.org/docs)
[![PyPI version](https://badge.fury.io/py/scirex.svg)](https://badge.fury.io/py/scirex)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2024.xxxxx)

</div>

SciREX is an open-source scientific AI and machine learning framework designed for researchers and engineers. Jointly developed by Zenteiq Aitech Innovations Private Limited and the AiREX (AI for Research and Engineering eXcellence) Lab at Indian Institute of Science, Bangalore, SciREX bridges the gap between theoretical research and practical implementation while maintaining mathematical rigor and computational efficiency.

## Key Features

- **Research-First Design**: Built specifically for scientific computing and research workflows
- **Mathematical Foundations**: Strong emphasis on mathematical correctness and theoretical foundations
- **Hardware Optimization**: Efficient implementation with GPU acceleration support
- **Reproducible Research**: Built-in experiment tracking and result reproduction capabilities
- **Scientific Visualization**: Publication-ready plotting and visualization tools
- **Industrial Integration**: Enterprise-ready solutions backed by Zenteiq's industrial expertise

## Quick Start

Get started with SciREX by installing it via `pip`:

```bash
pip install scirex
```

### Simple Training Example
```python
import jax.numpy as jnp
from flax import nnx as nn
from scirex.training import Trainer

# Define a simple model and loss
model = nn.Linear(10, 2, rngs=nn.Rngs(0))
def loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return jnp.mean((logits - y) ** 2), logits

# Initialize Trainer
trainer = Trainer(model=model, optimizer=nn.Optimizer(model, optax.adam(1e-3)), loss_fn=loss_fn)

# Train!
trainer.train(batch_iterator, num_epochs=5)
```

## Core Capabilities

- **[Diffusion Module](scirex/diffusion/README.md)**: Implementation of noise schedules and diffusion samplers.
- **[Training Module](scirex/training/README.md)**: Utilities for managing optimization loops and model training.
- **[Transformers Module](scirex/transformers/README.md)**: Modular Transformer and U-Net architecture implementations.
- **[Experimental Module](scirex/experimental/README.md)**: Research-focused implementations and experimental features.
- **[Examples](examples/README.md)**: Comprehensive end-to-end examples and tutorials.

## Documentation

Visit our [documentation](https://scirex.org/docs/) for:
- Getting Started Guide
- Tutorials and Examples
- Contribution Guidelines

## License

Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and AiREX Lab, Indian Institute of Science, Bangalore.
All rights reserved.

## Software License

SciREX is licensed under the Apache License, Version 2.0 (the "License"). You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

## Intellectual Property

### Copyright Holders
- <a href="https://zenteiq.ai/" target="_blank">Zenteiq Aitech Innovations Private Limited</a>
- <a href="https://airexlab.cds.iisc.ac.in/" target="_blank">The AiREX Lab at IISc Bangalore</a>

### Components and Libraries
- The core SciREX framework and its original components are copyright of the above holders
- Third-party libraries and dependencies are subject to their respective licenses
- Mathematical algorithms and scientific methods implemented may be subject to their own patents or licenses

## Contributing

We welcome contributions from both the research and industrial communities! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Community

- <a href="https://discord.gg/NWcCPx22Hq/" target="_blank">Discord</a>

## Official Partners

- [**ARTPARK**](https://artpark.in) (AI & Robotics Technology Park) at IISc
- In discussion with NVIDIA and other technology companies

## Acknowledgments

SciREX is developed and maintained through the collaborative efforts of Zenteiq Aitech Innovations and the AiREX Lab at IISc Bangalore. We thank all contributors from both industry and academia for their valuable input and support in advancing scientific computing.
