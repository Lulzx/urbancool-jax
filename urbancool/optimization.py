"""Pedestrian comfort objective, value_and_grad wrapper, Optax Adam loop."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from urbancool.solver import rollout_simulation


def _softplus_tau(z, tau=20.0):
    return jax.nn.softplus(tau * z) / tau


class ComfortObjective(NamedTuple):
    t_target: float = 0.15
    u_max: float = 0.10
    canopy_budget: float = 0.35
    w_temp: float = 10.0
    w_wind: float = 2.0
    w_budget: float = 15.0
    w_bin: float = 0.1

    def compute(self, T_mean, speed_mean, gamma, domain):
        ped = domain.pedestrian_mask
        ped_n = jnp.maximum(jnp.sum(ped), 1.0)
        plant_n = jnp.maximum(jnp.sum(domain.plantable_mask), 1.0)
        ped_temp = jnp.sum(ped * T_mean) / ped_n
        ped_wind = jnp.sum(ped * speed_mean) / ped_n
        ped_wind_max = jnp.max(jnp.where(ped > 0, speed_mean, 0.0))
        canopy_frac = jnp.sum(gamma) / plant_n
        loss_temp = jnp.sum(ped * _softplus_tau(T_mean - self.t_target) ** 2) / ped_n
        loss_wind = jnp.sum(ped * _softplus_tau(speed_mean - self.u_max) ** 2) / ped_n
        loss_budget = _softplus_tau(canopy_frac - self.canopy_budget) ** 2
        loss_bin = jnp.sum(domain.plantable_mask * gamma * (1.0 - gamma)) / plant_n
        loss = (self.w_temp * loss_temp + self.w_wind * loss_wind
                + self.w_budget * loss_budget + self.w_bin * loss_bin)
        aux = {"loss": loss, "ped_temp": ped_temp, "ped_wind": ped_wind,
               "ped_wind_max": ped_wind_max, "canopy_frac": canopy_frac}
        return loss, aux

    def __call__(self, theta_raw, config, domain, burn_in_steps=100,
                 outer_steps=10, inner_steps=10):
        out = rollout_simulation(theta_raw, config, domain, burn_in_steps,
                                 outer_steps, inner_steps)
        return self.compute(out["T_mean"], out["speed_mean"], out["gamma"], domain)


# Module-level default used by loss_fn.
DEFAULT_OBJECTIVE = ComfortObjective()


def loss_fn(theta_raw, config, domain, burn_in_steps=100, outer_steps=10,
            inner_steps=10, objective=DEFAULT_OBJECTIVE):
    loss, _ = objective(theta_raw, config, domain, burn_in_steps,
                        outer_steps, inner_steps)
    return loss


def value_and_grad_fn(theta_raw, config, domain, burn_in_steps=100,
                      outer_steps=10, inner_steps=10, objective=DEFAULT_OBJECTIVE):
    """Return ((loss, aux), grads) for one theta field."""
    def _loss(th):
        return objective(th, config, domain, burn_in_steps, outer_steps, inner_steps)
    return jax.value_and_grad(_loss, has_aux=True)(theta_raw)


def optimize_canopy_layout(solver=None, objective=DEFAULT_OBJECTIVE, theta_init=None,
                           config=None, domain=None, burn_in_steps=100,
                           outer_steps=10, inner_steps=10, opt_steps=30,
                           learning_rate=0.1, grad_steps=None, beta_schedule=None):
    """Adam loop. Accepts either (solver, objective) or (config, domain).

    ``beta_schedule`` optionally ramps the sigmoid sharpness ``beta``
    (continuation method): ``opt_steps`` is split as evenly as possible
    across the schedule values, e.g. ``(2, 4, 8, 16)`` pushes grey canopy
    values toward crisp binary placements in later stages. ``None`` keeps
    the fixed ``config.beta``.
    """
    if solver is not None:
        config, domain = solver.config, solver.domain
    if grad_steps is not None:  # README-style alias: split grad window ~ sqrt
        inner = max(int(round(grad_steps ** 0.5)), 1)
        outer_steps, inner_steps = inner, inner
    theta = jnp.zeros_like(domain.plantable_mask) if theta_init is None else theta_init
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(theta)
    history = {"loss": [], "mean_temp": [], "ped_temp": [], "ped_wind": [],
               "ped_wind_max": [], "canopy_frac": [], "beta": []}

    betas = [float(config.beta)] if beta_schedule is None else [float(b) for b in beta_schedule]
    base, extra = divmod(int(opt_steps), len(betas))
    stage_steps = [base + (1 if i < extra else 0) for i in range(len(betas))]

    for beta, n_steps in zip(betas, stage_steps):
        stage_config = config._replace(beta=beta)

        @jax.jit
        def step(theta, opt_state, _cfg=stage_config):
            (loss, aux), grads = value_and_grad_fn(
                theta, _cfg, domain, burn_in_steps, outer_steps, inner_steps, objective)
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            return optax.apply_updates(theta, updates), opt_state, loss, aux

        for _ in range(n_steps):
            theta, opt_state, loss, aux = step(theta, opt_state)
            history["loss"].append(float(loss))
            history["mean_temp"].append(float(aux["ped_temp"]))
            history["ped_temp"].append(float(aux["ped_temp"]))
            history["ped_wind"].append(float(aux["ped_wind"]))
            history["ped_wind_max"].append(float(aux["ped_wind_max"]))
            history["canopy_frac"].append(float(aux["canopy_frac"]))
            history["beta"].append(beta)
    return theta, history
