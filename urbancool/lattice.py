"""D2Q9 (momentum) and D2Q5 (thermal) lattice constants."""
import jax.numpy as jnp

# Rows: [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
E_D2Q9 = jnp.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
     [1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=jnp.int32)
W_D2Q9 = jnp.array(
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])
OPP_D2Q9 = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)
CX_D2Q9 = E_D2Q9[:, 0]
CY_D2Q9 = E_D2Q9[:, 1]

E_D2Q5 = jnp.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)
W_D2Q5 = jnp.array([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
OPP_D2Q5 = jnp.array([0, 3, 4, 1, 2], dtype=jnp.int32)
CX_D2Q5 = E_D2Q5[:, 0]
CY_D2Q5 = E_D2Q5[:, 1]

# Specular (free-slip) reflection at a horizontal wall: flips the y velocity.
REFL_Y_D2Q9 = jnp.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=jnp.int32)
REFL_Y_D2Q5 = jnp.array([0, 1, 4, 3, 2], dtype=jnp.int32)

CS2 = 1.0 / 3.0
