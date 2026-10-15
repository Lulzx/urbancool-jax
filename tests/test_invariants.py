"""Mass conservation (periodic) and channel-flow sanity checks."""
import jax
import jax.numpy as jnp

from urbancool.grid import create_urban_canyon_domain
from urbancool.physics import SimConfig, compute_f_eq, lbm_step
from urbancool.solver import initial_state


def test_periodic_mass_conservation_200_steps():
    domain = create_urban_canyon_domain(40, 24)
    config = SimConfig()
    gamma = jnp.zeros((40, 24))
    state = initial_state(config, domain)
    mass0 = jnp.sum(state[0])

    def body(carry, _):
        return lbm_step(carry, gamma, config, domain, periodic=True), None

    final, _ = jax.lax.scan(body, state, None, length=200)
    mass1 = jnp.sum(final[0])
    assert jnp.all(jnp.isfinite(final[0])) and jnp.all(jnp.isfinite(final[1]))
    assert float(mass1) > 0.0
    assert abs(float(mass1 - mass0)) / abs(float(mass0)) < 1e-5


def test_equilibrium_moments():
    rho = jnp.ones((8, 6)); ux = jnp.full((8, 6), 0.05); uy = jnp.zeros((8, 6))
    f_eq = compute_f_eq(rho, ux, uy)
    assert jnp.allclose(jnp.sum(f_eq, -1), rho, atol=1e-6)
    from urbancool.lattice import E_D2Q9
    mom_x = jnp.sum(f_eq * E_D2Q9[:, 0].astype(f_eq.dtype), -1)
    assert jnp.allclose(mom_x, rho * ux, atol=1e-6)


def test_channel_flow_stays_finite_and_positive_density():
    domain = create_urban_canyon_domain(40, 24)
    config = SimConfig()
    gamma = jnp.zeros((40, 24))
    state = initial_state(config, domain)

    def body(carry, _):
        return lbm_step(carry, gamma, config, domain), None

    final, _ = jax.lax.scan(body, state, None, length=100)
    rho = jnp.sum(final[0], axis=-1)
    assert jnp.all(jnp.isfinite(final[0]))
    assert float(jnp.min(rho)) > 0.5
