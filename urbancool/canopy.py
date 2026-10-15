"""Differentiable porous tree-canopy parameterisation.

Raw unconstrained parameters ``theta_raw`` are spatially smoothed with a
3x3 tent kernel and projected through a sigmoid (SIMP-style) so the
physical canopy density ``gamma`` lies in [0, 1] on the plantable mask.
"""
import jax
import jax.numpy as jnp
import jax.scipy.signal

_KERNEL = jnp.array([[1.0, 2.0, 1.0],
                     [2.0, 4.0, 2.0],
                     [1.0, 2.0, 1.0]]) / 16.0


def project_canopy_density(theta_raw, plantable_mask, beta=4.0):
    """Smooth, sigmoid-project and mask ``theta_raw`` into ``gamma`` in [0, 1]."""
    theta = jnp.asarray(theta_raw)
    kernel = _KERNEL.astype(theta.dtype)
    # jax.scipy.signal.convolve2d only supports boundary="fill"; symmetric
    # ("symm") extension is applied explicitly with jnp.pad first.
    padded = jnp.pad(theta, 1, mode="symmetric")
    theta_smoothed = jax.scipy.signal.convolve2d(
        padded, kernel, mode="valid", boundary="fill")
    return plantable_mask * jax.nn.sigmoid(beta * theta_smoothed)
