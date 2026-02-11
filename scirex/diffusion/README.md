# SciREX Diffusion

Denoising Diffusion Probabilistic Models (DDPM) and utilities for scientific generative modeling.

## Key Features

- **Standard Noise Schedules**: Pre-configured schedules like LogLinear, DDPM, LDM, Cosine, and Sigmoid.
- **Advanced Sampling**: Support for DDPM (stochastic), DDIM (deterministic), and accelerated samplers.
- **Scientific Generative AI**: Designed to be integrated into scientific workflows for data generation and uncertainty quantification.
- **Flax NNX Native**: Leverages stateful objects and JIT-friendly patterns.

## Quick Start: Noise Schedules and Sampling

```python
from flax import nnx as nn
import optax
import jax.random as jr
from scirex.diffusion import ScheduleLogLinear, TimeInputMLP, train_step, sample

# 1. Initialize Schedule and Model
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10.0)
model = TimeInputMLP(hidden_dims=(128, 128), output_dim=2, rngs=nn.Rngs(0))

# 2. Setup Training State
optimizer = nn.Optimizer(model, optax.adam(1e-3))
rng = nn.Rngs(42)

# 3. Simple Training Step
# Note: Usually performed inside a loop with Trainer or DiffusionTrainer
loss = train_step(model, optimizer, batch, schedule, rng)

# 4. Generate Samples
# sigmas define the denoising path
sampling_sigmas = schedule.sample_sigmas(steps=20)
generated_samples = sample(model, sampling_sigmas, batchsize=500, rng=rng)
```

## API Reference

### Noise Schedules
Schedules define the amount of noise added to data at each diffusion step.

- `ScheduleLogLinear`: Standard log-linear sigma spacing.
- `ScheduleDDPM`: Traditional linear beta schedule.
- `ScheduleCosine`: Improved cosine schedule for better performance at smaller timesteps.
- `ScheduleLDM`: Beta schedule typically used in Latent Diffusion Models.
- `ScheduleSigmoid`: Sigmoid-based noise progression.

### Core Functions
- `train_step(...)`: Performs a single DDPM training step (noise prediction).
- `sample(model, sigmas, batchsize=1, shape=(2,), method="ddim", ...)`: High-level sampling function that supports both DDPM (stochastic) and DDIM (deterministic) sampling.
- `classifier_free_guidance(...)`: Implementation of classifier-free guidance for conditional generation.

## File Structure

```
.
├── forward.py    # Implementation of noise schedules, embeddings, and forward process.
├── reverse.py    # Implementation of sampling methods (DDPM, DDIM, CFG).
├── __init__.py   # Package initialization and component exports.
└── README.md
```

## Documentation and Research
For deeper theoretical background on the implemented schedules and sampling logic, please refer to our [documentation](https://scirex.org/docs/diffusion).
