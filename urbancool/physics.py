"""Coupled D2Q9/D2Q5 thermal LBM single step: LES, Guo forcing, canopy drag."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from urbancool.lattice import (E_D2Q9, W_D2Q9, OPP_D2Q9, E_D2Q5, W_D2Q5,
                               OPP_D2Q5, REFL_Y_D2Q9, REFL_Y_D2Q5)

EPS = 1e-12


class SimConfig(NamedTuple):
    u_inlet: float = 0.06
    nu_0: float = 0.01
    alpha_0: float = 0.01
    c_smag: float = 0.12
    g_buoy: float = 0.0005
    q_solar: float = 0.002
    lambda_cool: float = 0.015
    t_wet: float = 0.0
    alpha_drag: float = 5.0
    beta: float = 4.0


def compute_f_eq(rho, u_x, u_y):
    rho = jnp.asarray(rho); u_x = jnp.asarray(u_x); u_y = jnp.asarray(u_y)
    # Broadcast scalars / fields to a common (nx, ny) shape.
    rho, u_x, u_y = jnp.broadcast_arrays(rho, u_x, u_y)
    ex = E_D2Q9[:, 0].astype(rho.dtype); ey = E_D2Q9[:, 1].astype(rho.dtype)
    cu = ex * u_x[..., None] + ey * u_y[..., None]
    usq = u_x ** 2 + u_y ** 2
    return W_D2Q9.astype(rho.dtype) * rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * usq[..., None])


def compute_g_eq(T, u_x, u_y):
    T = jnp.asarray(T); u_x = jnp.asarray(u_x); u_y = jnp.asarray(u_y)
    T, u_x, u_y = jnp.broadcast_arrays(T, u_x, u_y)
    cx = E_D2Q5[:, 0].astype(T.dtype); cy = E_D2Q5[:, 1].astype(T.dtype)
    cu = cx * u_x[..., None] + cy * u_y[..., None]
    return W_D2Q5.astype(T.dtype) * T[..., None] * (1.0 + 3.0 * cu)


def compute_macroscopic(f, g, gamma, config, domain):
    """Return (rho, T, u_x, u_y) with buoyancy, canopy drag, Mach limiter."""
    rho = jnp.sum(f, axis=-1)
    T = jnp.sum(g, axis=-1)
    fluid_mask = 1.0 - domain.building_mask
    F_x = jnp.zeros_like(rho)
    F_y = rho * config.g_buoy * T * fluid_mask
    ex = E_D2Q9[:, 0].astype(f.dtype); ey = E_D2Q9[:, 1].astype(f.dtype)
    u_x_star = (jnp.sum(f * ex, axis=-1) + 0.5 * F_x) / rho
    u_y_star = (jnp.sum(f * ey, axis=-1) + 0.5 * F_y) / rho
    speed_star = jnp.sqrt(u_x_star ** 2 + u_y_star ** 2 + EPS)
    drag_denom = 1.0 + config.alpha_drag * gamma * (0.1 + speed_star)
    u_x = fluid_mask * u_x_star / drag_denom
    u_y = fluid_mask * u_y_star / drag_denom
    speed = jnp.sqrt(u_x ** 2 + u_y ** 2 + EPS)
    scale = jnp.minimum(1.0, 0.20 / speed)
    return rho, T, u_x * scale, u_y * scale


def lbm_step(state, gamma, config, domain, periodic=False):
    """One pure-functional coupled LBM step. ``state = (f, g)``."""
    f, g = state
    nx, ny = f.shape[0], f.shape[1]
    rho, T, u_x, u_y = compute_macroscopic(f, g, gamma, config, domain)
    fluid_mask = 1.0 - domain.building_mask

    f_eq = compute_f_eq(rho, u_x, u_y)
    g_eq = compute_g_eq(T, u_x, u_y)

    # Smagorinsky LES from non-equilibrium momentum flux.
    f_neq = f - f_eq
    ex = E_D2Q9[:, 0].astype(f.dtype); ey = E_D2Q9[:, 1].astype(f.dtype)
    pi_xx = jnp.sum(f_neq * (ex * ex), axis=-1)
    pi_xy = jnp.sum(f_neq * (ex * ey), axis=-1)
    pi_yy = jnp.sum(f_neq * (ey * ey), axis=-1)
    pi_norm = jnp.sqrt(pi_xx ** 2 + 2.0 * pi_xy ** 2 + pi_yy ** 2 + EPS)
    tau_0 = 3.0 * config.nu_0 + 0.5
    tau_eff = 0.5 * (tau_0 + jnp.sqrt(
        tau_0 ** 2 + 18.0 * jnp.sqrt(2.0) * (config.c_smag ** 2) * pi_norm / rho))
    nu_t = (tau_eff - tau_0) / 3.0
    tau_T = 3.0 * (config.alpha_0 + nu_t / 0.85) + 0.5
    tau_T = jnp.maximum(tau_T, 0.51)

    # Guo buoyancy forcing (F_x = 0, F_y as in compute_macroscopic).
    F_y = rho * config.g_buoy * T * fluid_mask
    cu = ex * u_x[..., None] + ey * u_y[..., None]
    # [3(e_i - u) + 9 (e_i.u) e_i] . F  with F = (0, F_y)
    guo_bracket_y = 3.0 * (ey - u_y[..., None]) + 9.0 * cu * ey
    S = (1.0 - 1.0 / (2.0 * tau_eff))[..., None] * W_D2Q9.astype(f.dtype) * (
        guo_bracket_y * F_y[..., None])

    q_net = (config.q_solar * domain.solar_mask
             - config.lambda_cool * gamma
             * jax.nn.softplus(5.0 * (T - config.t_wet)) / 5.0) * fluid_mask

    f_col = f - (f - f_eq) / tau_eff[..., None] + S
    g_col = g - (g - g_eq) / tau_T[..., None] + W_D2Q5.astype(g.dtype) * q_net[..., None]

    # Streaming (trace-time unroll over populations; time loop is lax.scan).
    shifts9 = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
               (1, 1), (-1, 1), (-1, -1), (1, -1)]
    shifts5 = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    f_s = jnp.stack([jnp.roll(f_col[..., i], shift=s, axis=(0, 1))
                     for i, s in enumerate(shifts9)], axis=-1)
    g_s = jnp.stack([jnp.roll(g_col[..., i], shift=s, axis=(0, 1))
                     for i, s in enumerate(shifts5)], axis=-1)

    iy = jnp.arange(ny)[None, :]
    ix = jnp.arange(nx)[:, None]
    # Building & ground bounce-back.
    obstacle = ((domain.building_mask > 0.5) | (iy == 0)).astype(f.dtype)
    # Full-way on-site bounce-back: permuting the streamed populations at a
    # solid node conserves mass exactly (unlike overwriting with f_col).
    f_new = jnp.where(obstacle[..., None], f_s[:, :, OPP_D2Q9], f_s)
    g_new = jnp.where(obstacle[..., None].astype(g.dtype), g_s[:, :, OPP_D2Q5], g_s)

    if not periodic:
        # Top sky: specular free-slip reflection.
        top = (iy == ny - 1).astype(f.dtype)
        f_new = jnp.where(top[..., None], f_new[:, :, REFL_Y_D2Q9], f_new)
        g_new = jnp.where(top[..., None].astype(g.dtype), g_new[:, :, REFL_Y_D2Q5], g_new)
        # West inlet: equilibrium.
        f_in = compute_f_eq(jnp.ones_like(rho), config.u_inlet * jnp.ones_like(rho),
                            jnp.zeros_like(rho))
        g_in = compute_g_eq(jnp.zeros_like(rho), config.u_inlet * jnp.ones_like(rho),
                            jnp.zeros_like(rho))
        inlet = (ix == 0).astype(f.dtype)
        f_new = jnp.where(inlet[..., None], f_in, f_new)
        g_new = jnp.where(inlet[..., None].astype(g.dtype), g_in, g_new)
        # East outlet: zero gradient (copy of x = nx-2 column).
        outlet = (ix == nx - 1).astype(f.dtype)
        f_new = jnp.where(outlet[..., None], jnp.roll(f_new, 1, axis=0), f_new)
        g_new = jnp.where(outlet[..., None].astype(g.dtype), jnp.roll(g_new, 1, axis=0), g_new)
    return f_new, g_new
