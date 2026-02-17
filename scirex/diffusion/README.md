# SciREX Diffusion

Denoising Diffusion Probabilistic Models (DDPM) and utilities for scientific generative modeling.

## Key Features

- **Standard Noise Schedules**: Pre-configured schedules like LogLinear, DDPM, LDM, Cosine, and Sigmoid.
- **Advanced Sampling**: Support for DDPM (stochastic), DDIM (deterministic), and accelerated samplers.
- **Scientific Generative AI**: Designed to be integrated into scientific workflows for data generation and uncertainty quantification.
- **Flax NNX Native**: Leverages stateful objects and JIT-friendly patterns.

## Scientific Background

SciREX implements Denoising Diffusion Probabilistic Models (DDPM), which model data generation as a reverse diffusion process.

### Forward Process
The forward process adds Gaussian noise to the data $x_0$ over $T$ steps:
$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$
$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$

### Reverse Process (Sampling)
The model learns to predict the noise $\epsilon_\theta(x_t, t)$, which is used to recover $x_0$:
$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$

### Optimization Perspective & Generalized Sampling
Recent work interprets the denoiser as an approximate projection onto the data manifold. A generalized sampler can be derived:
$x_{t-1} = x_t - (\sigma_t - \sigma_{t-1}) \epsilon_{av} + \eta w_t$
where $\epsilon_{av} = \gamma \epsilon_t + (1-\gamma) \epsilon_{t-1}$.

Parameters:
- $\gamma$ (gamma): Momentum term. $\gamma=1$ is standard, $\gamma > 1$ accelerates convergence.
- $\mu$ (mu): Controls noise injection. $\mu=0$ is deterministic (DDIM-like), $\mu=0.5$ mimics DDPM.

### References
- Ho, J., Jain, A., & Abbeel, P. (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS.
- Song, J., Meng, C., & Ermon, S. (2021). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502). ICLR.
- Ho, J., & Salimans, T. (2022). [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). NeurIPS Workshop.
- Permenter, F., & Yuan, C. (2024). [Interpreting and Improving Diffusion Models from an Optimization Perspective](https://arxiv.org/abs/2306.04848). ICML.

## Quick Start: Noise Schedules and Sampling

```python
from flax import nnx as nn
import optax
from scirex.diffusion import ScheduleLogLinear, diffusion_loss, sample

# 1. Initialize Schedule and Model
schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10.0)

# Define a simple MLP for noise prediction
class SimpleMLP(nn.Module):
    def __init__(self, rngs: nn.Rngs):
        self.mlp = nn.Sequential(
            nn.Linear(4, 128, rngs=rngs), nn.relu,
            nn.Linear(128, 2, rngs=rngs)
        )
    def __call__(self, x, sigma):
        # Concatenate x and sigma embedding (simplified)
        return self.mlp(x)

model = SimpleMLP(rngs=nn.Rngs(0))

# 2. Setup Training State
optimizer = nn.Optimizer(model, optax.adam(1e-3))
rng = nn.Rngs(42)

# 3. Setup Loss Function
loss_fn = diffusion_loss(schedule)

# 4. Simple Training Step
# Note: Usually performed inside a loop with Trainer or DiffusionTrainer
(loss, _), grads = nn.value_and_grad(loss_fn)(model, batch, rng)
optimizer.update(grads)

# 5. Generate Samples
# sigmas define the denoising path
sampling_sigmas = schedule.sample_sigmas(steps=20)

# Standard DDIM sampling
samples_ddim = sample(
    model, sampling_sigmas, batchsize=500, rng=rng, method="ddim"
)

# Improved sampling with momentum (Gradient Estimation)
samples_improved = sample(
    model, sampling_sigmas, batchsize=500, rng=rng,
    method="generalized", gamma=2.0
)
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
- `diffusion_loss(schedule)`: Returns a loss function for the diffusion process.
- `sample(...)`: Unified sampling interface supporting "ddpm", "ddim", and "generalized" methods.
    - `sample(..., method="generalized", gamma=2.0)`: Uses the improved gradient estimation sampler.
- `cfg(...)`: Implementation of classifier-free guidance for conditional generation.

## File Structure

```
.
├── forward.py    # Implementation of noise schedules, embeddings, and forward process.
├── reverse.py    # Implementation of sampling methods (DDPM, DDIM, CFG, Generalized).
├── __init__.py   # Package initialization and component exports.
└── README.md
```

## Documentation and Research
For deeper theoretical background on the implemented schedules and sampling logic, please refer to our [documentation](https://scirex.org/docs/diffusion).
