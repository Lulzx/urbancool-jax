"""Solar heating and canopy cooling directional physics checks."""
import jax.numpy as jnp

from urbancool.grid import create_urban_canyon_domain
from urbancool.physics import SimConfig
from urbancool.solver import rollout_from_gamma


def _ped_means(out, domain):
    ped = domain.pedestrian_mask
    n = jnp.maximum(jnp.sum(ped), 1.0)
    return float(jnp.sum(ped * out["T_mean"]) / n), float(jnp.sum(ped * out["speed_mean"]) / n)


def test_solar_heating_and_canopy_cooling():
    domain = create_urban_canyon_domain(60, 30)
    config = SimConfig()
    # 300 steps total per run via the checkpointed rollout engine.
    kw = dict(burn_in_steps=150, outer_steps=10, inner_steps=15)
    no_trees = rollout_from_gamma(jnp.zeros((60, 30)), config, domain, **kw)
    trees = rollout_from_gamma(0.8 * domain.plantable_mask, config, domain, **kw)
    t0, u0 = _ped_means(no_trees, domain)
    t1, u1 = _ped_means(trees, domain)
    assert t0 > 0.01, t0
    assert t1 < t0, (t1, t0)
    assert u1 < u0, (u1, u0)
