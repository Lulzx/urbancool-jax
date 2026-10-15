"""Burn-in + checkpointed lax.scan rollout engine (TBPTT, O(sqrt N) memory)."""
import jax
import jax.numpy as jnp

from urbancool.canopy import project_canopy_density
from urbancool.physics import SimConfig, compute_f_eq, compute_macroscopic, lbm_step

__all__ = ["SimConfig", "rollout_simulation", "rollout_from_gamma",
           "initial_state", "ThermalLBMSolver"]


def initial_state(config, domain):
    nx, ny = domain.building_mask.shape
    ux0 = config.u_inlet * (1.0 - domain.building_mask)
    f0 = compute_f_eq(jnp.ones((nx, ny)), ux0, jnp.zeros((nx, ny)))
    g0 = jnp.zeros((nx, ny, 5), dtype=f0.dtype)
    return f0, g0


def _macro_fields(state, gamma, config, domain):
    _, _, u_x, u_y = compute_macroscopic(state[0], state[1], gamma, config, domain)
    T = jnp.sum(state[1], axis=-1)
    speed = jnp.sqrt(u_x ** 2 + u_y ** 2 + 1e-12)
    return T, speed


def rollout_from_gamma(gamma, config, domain, burn_in_steps=100,
                       outer_steps=10, inner_steps=10):
    """Roll out with an explicit canopy field; returns time-mean fields."""
    state = initial_state(config, domain)

    def burn_step(carry, _):
        return lbm_step(carry, gamma, config, domain), None

    if burn_in_steps and int(burn_in_steps) > 0:
        state, _ = jax.lax.scan(burn_step, state, None, length=int(burn_in_steps))
    state = jax.lax.stop_gradient(state)

    def accum_step(carry, _):
        st, t_sum, s_sum = carry
        st = lbm_step(st, gamma, config, domain)
        T, speed = _macro_fields(st, gamma, config, domain)
        return (st, t_sum + T, s_sum + speed), None

    @jax.checkpoint
    def inner_block(carry):
        carry, _ = jax.lax.scan(accum_step, carry, None, length=int(inner_steps))
        return carry

    def outer_step(carry, _):
        return inner_block(carry), None

    total = max(int(outer_steps) * int(inner_steps), 1)
    init = (state, jnp.zeros_like(gamma), jnp.zeros_like(gamma))
    if int(outer_steps) > 0 and int(inner_steps) > 0:
        (state, t_sum, s_sum), _ = jax.lax.scan(
            outer_step, init, None, length=int(outer_steps))
        T_mean, speed_mean = t_sum / total, s_sum / total
    else:  # degenerate window: report instantaneous fields
        T_mean, speed_mean = _macro_fields(state, gamma, config, domain)
    return {"T_mean": T_mean, "speed_mean": speed_mean, "gamma": gamma,
            "f": state[0], "g": state[1], "state": state}


def rollout_simulation(theta_raw, config, domain, burn_in_steps=100,
                       outer_steps=10, inner_steps=10):
    gamma = project_canopy_density(theta_raw, domain.plantable_mask, config.beta)
    return rollout_from_gamma(gamma, config, domain, burn_in_steps,
                              outer_steps, inner_steps)


class ThermalLBMSolver:
    """Thin OO wrapper (README API) around the functional rollout."""

    def __init__(self, config, domain):
        self.config = config
        self.domain = domain

    def rollout(self, theta_raw, burn_in_steps=100, outer_steps=10, inner_steps=10):
        return rollout_simulation(theta_raw, self.config, self.domain,
                                  burn_in_steps, outer_steps, inner_steps)
