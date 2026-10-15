"""Urban canyon domain construction: buildings and evaluation/design masks.

All masks have shape ``(nx, ny)`` and are indexed ``[x, y]`` with ``y = 0``
the street level. Masks are float arrays (1.0 inside, 0.0 outside) so they
compose branchlessly with ``jnp.where`` / multiplication in the physics.
"""
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class UrbanDomain(NamedTuple):
    building_mask: jnp.ndarray    # 1.0 inside solid buildings
    solar_mask: jnp.ndarray       # 1.0 on solar-heated surfaces
    plantable_mask: jnp.ndarray   # 1.0 where tree canopy may be placed
    pedestrian_mask: jnp.ndarray  # 1.0 in pedestrian evaluation zones

    @property
    def nx(self) -> int:
        return int(self.building_mask.shape[0])

    @property
    def ny(self) -> int:
        return int(self.building_mask.shape[1])


def _resolve_nx_ny(nx, ny):
    # Accept either ints or a config object carrying .nx/.ny (README form).
    if hasattr(nx, "nx"):
        cfg = nx
        return int(cfg.nx), int(getattr(cfg, "ny", ny))
    return int(nx), int(ny)


def create_urban_canyon_domain(nx=120, ny=60) -> UrbanDomain:
    """Build a canyon with 3 buildings of varying height on the street.

    Buildings sit on ``y = 0``; solar strips cover the exposed street and
    the fluid cells adjacent to building walls/roofs; plantable zones are
    the open street/courtyard band (disjoint from buildings); pedestrian
    zones are the low-level walkways between and downstream of buildings.
    """
    nx, ny = _resolve_nx_ny(nx, ny)
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]

    # Building footprints as fractions of nx, heights as fractions of ny.
    specs = [(0.16, 0.30, 0.30), (0.42, 0.54, 0.45), (0.68, 0.80, 0.24)]
    building = np.zeros((nx, ny), dtype=np.float64)
    for fx0, fx1, fh in specs:
        x0, x1 = int(fx0 * nx), max(int(fx1 * nx), int(fx0 * nx) + 2)
        h = max(int(fh * ny), 3)
        building[x0:x1, 0:h] = 1.0

    fluid = 1.0 - building
    ground = (y == 0).astype(np.float64) * fluid  # exposed street cells

    # Solar: exposed street layer just above ground + fluid cells adjacent
    # (4-neighbourhood) to any building cell (roofs / windward / leeward walls).
    adj = np.zeros_like(building)
    adj[1:, :] += building[:-1, :]
    adj[:-1, :] += building[1:, :]
    adj[:, 1:] += building[:, :-1]
    adj[:, :-1] += building[:, 1:]
    street_solar = ((y >= 1) & (y <= 2)).astype(np.float64) * fluid
    solar = np.clip(street_solar + (adj > 0).astype(np.float64) * fluid, 0.0, 1.0)
    solar = solar * (1.0 - ground)

    # Plantable: open low band (street + courtyards), strictly disjoint
    # from buildings, excluding the inlet/outlet columns.
    band_h = max(4, ny // 3)
    plantable = ((y >= 1) & (y < band_h)).astype(np.float64) * fluid
    plantable[:2, :] = 0.0
    plantable[-2:, :] = 0.0

    # Pedestrian: walkways at y in [1, 6] (scaled), fluid only.
    ped_h = max(3, min(7, ny // 8 + 2))
    pedestrian = ((y >= 1) & (y < ped_h)).astype(np.float64) * fluid
    pedestrian[:2, :] = 0.0
    pedestrian[-2:, :] = 0.0

    return UrbanDomain(
        building_mask=jnp.asarray(building),
        solar_mask=jnp.asarray(solar),
        plantable_mask=jnp.asarray(plantable),
        pedestrian_mask=jnp.asarray(pedestrian),
    )
