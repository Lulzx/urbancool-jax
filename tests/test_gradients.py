"""x64 finite-difference vs jax.grad verification on a compact canyon."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from urbancool.grid import create_urban_canyon_domain
from urbancool.optimization import loss_fn
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
