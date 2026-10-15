# UrbanCool-JAX

**Differentiable fluid–thermal Lattice Boltzmann engine for inverse urban microclimate design — in pure JAX.**

[![JAX](https://img.shields.io/badge/JAX-0.4%2B-4285F4)](https://github.com/jax-ml/jax)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Physics](https://img.shields.io/badge/Physics-D2Q9%20%2B%20D2Q5%20thermal%20LBM-FF6F00)]()

Instead of running one static CFD case at a time, UrbanCool-JAX differentiates through the whole coupled flow + heat simulation. State a pedestrian-comfort objective, and Adam automatically designs where tree canopy should go.

![Optimized canopy and temperature field](docs/assets/comparison.png)

## What it does

- **Coupled thermal LBM** — D2Q9 momentum + D2Q5 temperature lattices, Boussinesq buoyancy via Guo forcing, solar heating and evaporative canopy cooling.
- **Smagorinsky LES** — local eddy viscosity from non-equilibrium moments, no finite-difference stencils, `sqrt(· + 1e-12)` everywhere so gradients never NaN at rest.
- **Differentiable tree canopies** — raw parameters → 3×3 smoothing → sigmoid (SIMP) density `γ ∈ [0, 1]`, acting as porous Brinkman–Forchheimer drag + latent cooling.
- **Cheap backprop through time** — non-differentiable burn-in (`stop_gradient`) + `jax.checkpoint`-wrapped `lax.scan` windows: O(√N) memory instead of O(N).

![Domain masks](docs/assets/domain.png)

## Results (80×40 canyon, 30 Adam steps)

| Metric | Before | After |
|---|---|---|
| Comfort loss | 0.0103 | **0.0035** (−66%) |
| Pedestrian temp | 0.100 | **0.071** (−29%) |
| Canopy fraction | 0.018 | 0.037 (budget 0.35) |

![Optimization history](docs/assets/optimization_history.png)

## Install

```bash
git clone https://github.com/Lulzx/urbancool-jax.git
cd urbancool-jax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import jax.numpy as jnp
from urbancool.grid import create_urban_canyon_domain
from urbancool.solver import ThermalLBMSolver, SimConfig
from urbancool.optimization import ComfortObjective, optimize_canopy_layout

config = SimConfig(u_inlet=0.06, nu_0=0.01, alpha_0=0.01)
domain = create_urban_canyon_domain(80, 40)
solver = ThermalLBMSolver(config, domain)
objective = ComfortObjective(t_target=0.15, u_max=0.10, canopy_budget=0.35)

theta, history = optimize_canopy_layout(
    solver=solver, objective=objective, theta_init=-jnp.ones((80, 40)),
    burn_in_steps=200, outer_steps=10, inner_steps=15,
    opt_steps=30, learning_rate=0.1)
print(history["loss"][0], "->", history["loss"][-1])
```

## Demo & tests

```bash
python scripts/run_inverse_design.py   # writes artifacts/optimization_summary.json + PNG
python -m pytest -q                    # 6 tests: mass conservation, thermal, FD-vs-grad (x64)
```

Gradient quality is verified, not assumed: analytical `jax.grad` matches central finite differences with cosine similarity > 0.999 and relative error < 1e-3 (measured 4.7e-05).

## Layout

```text
urbancool/   lattice.py · grid.py · canopy.py · physics.py · solver.py · optimization.py
scripts/     run_inverse_design.py
tests/       test_invariants.py · test_thermal.py · test_gradients.py
docs/assets/ figures used above
```

Full mathematical specification: [spec.md](spec.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).
