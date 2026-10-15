"""x64 finite-difference vs jax.grad verification on a compact canyon."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from urbancool.grid import create_urban_canyon_domain
from urbancool.optimization import ComfortObjective, loss_fn
from urbancool.physics import SimConfig

KW = dict(burn_in_steps=20, outer_steps=4, inner_steps=5)


@pytest.fixture(scope="module")
def grad_check():
    domain = create_urban_canyon_domain(32, 20)
    # Rebuild masks in float64 (domain built before x64 was enabled is f64
    # already when x64 is on at import; cast defensively).
    domain = domain._replace(
        building_mask=domain.building_mask.astype(jnp.float64),
        solar_mask=domain.solar_mask.astype(jnp.float64),
        plantable_mask=domain.plantable_mask.astype(jnp.float64),
        pedestrian_mask=domain.pedestrian_mask.astype(jnp.float64))
    config = SimConfig()
    theta0 = jnp.zeros((32, 20), dtype=jnp.float64)

    def loss(th):
        return loss_fn(th, config, domain, **KW)

    grads = jax.grad(loss)(theta0)
    coords = [(int(x), int(y)) for x, y in np.argwhere(
        np.asarray(domain.plantable_mask) > 0.5)]
    # Six well-spread active coordinates inside the plantable mask.
    picks = [coords[i] for i in np.linspace(0, len(coords) - 1, 6).astype(int)]
    eps = 1e-5
    fd, ad = [], []
    for (i, j) in picks:
        e = jnp.zeros_like(theta0).at[i, j].set(1.0)
        fd.append(float((loss(theta0 + eps * e) - loss(theta0 - eps * e)) / (2 * eps)))
        ad.append(float(grads[i, j]))
    return grads, domain, np.array(fd), np.array(ad)


def test_grad_no_nan_and_nonzero(grad_check):
    grads, domain, _, _ = grad_check
    assert not bool(jnp.any(jnp.isnan(grads)))
    assert float(jnp.sum(jnp.abs(grads) * domain.plantable_mask)) > 0.0


def test_grad_matches_finite_differences(grad_check):
    _, _, fd, ad = grad_check
    cos = float(np.dot(fd, ad) / (np.linalg.norm(fd) * np.linalg.norm(ad)))
    rel = np.abs(fd - ad) / np.maximum(np.abs(fd), 1e-12)
    print(f"\nFD={fd}\nAD={ad}\ncos={cos:.8f} max_rel={rel.max():.3e}")
    assert cos > 0.999, cos
    assert float(rel.max()) < 1e-3, rel


def test_physics_active_gradient_matches_finite_differences():
    """FD check where the gradient flows through the LBM rollout itself.

    At the ``grad_check`` point above (theta=0, default objective) the
    canopy-budget penalty dominates the gradient, so that gate mostly
    verifies the smoothing/sigmoid chain. Here the budget and
    binarisation weights are zero, the temperature target is 0 so the
    thermal term is active, and burn-in is 0 so finite differences see
    exactly the un-truncated function that ``jax.grad`` differentiates
    (with burn-in, AD deliberately returns the ``stop_gradient``-
    truncated gradient while FD would include burn-in sensitivity).
    """
    domain = create_urban_canyon_domain(32, 20)
    domain = domain._replace(
        building_mask=domain.building_mask.astype(jnp.float64),
        solar_mask=domain.solar_mask.astype(jnp.float64),
        plantable_mask=domain.plantable_mask.astype(jnp.float64),
        pedestrian_mask=domain.pedestrian_mask.astype(jnp.float64))
    config = SimConfig()
    objective = ComfortObjective(t_target=0.0, w_wind=0.0,
                                 w_budget=0.0, w_bin=0.0)
    kw = dict(burn_in_steps=0, outer_steps=6, inner_steps=10)
    theta0 = jnp.zeros((32, 20), dtype=jnp.float64)

    def loss(th):
        return objective(th, config, domain, **kw)[0]

    grads = jax.grad(loss)(theta0)
    assert not bool(jnp.any(jnp.isnan(grads)))
    # The through-physics gradient must be materially non-zero, and the
    # temperature term must actually be engaged at this point.
    assert float(jnp.max(jnp.abs(grads))) > 1e-5
    coords = [(int(x), int(y)) for x, y in np.argwhere(
        np.asarray(domain.plantable_mask) > 0.5)]
    picks = [coords[i] for i in np.linspace(0, len(coords) - 1, 5).astype(int)]
    eps = 1e-5
    fd, ad = [], []
    for (i, j) in picks:
        e = jnp.zeros_like(theta0).at[i, j].set(1.0)
        fd.append(float((loss(theta0 + eps * e) - loss(theta0 - eps * e)) / (2 * eps)))
        ad.append(float(grads[i, j]))
    fd, ad = np.array(fd), np.array(ad)
    cos = float(np.dot(fd, ad) / (np.linalg.norm(fd) * np.linalg.norm(ad)))
    rel = np.abs(fd - ad) / np.maximum(np.abs(fd), 1e-12)
    print(f"\nphysics FD={fd}\nphysics AD={ad}\ncos={cos:.8f} max_rel={rel.max():.3e}")
    assert cos > 0.999, cos
    assert float(rel.max()) < 1e-3, rel
