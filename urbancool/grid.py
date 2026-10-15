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


def masks_from_building_mask(building_mask, plan_view=False) -> UrbanDomain:
    """Derive solar / plantable / pedestrian masks from a building mask.

    This is the single mask-derivation path shared by the synthetic canyon
    and by externally rasterised footprints (see ``urbancool.osm``).

    Vertical cross-section mode (default): ``y`` is height, buildings sit
    on ``y = 0``. Plantable and pedestrian cells must be *ground-rooted*:
    no building cell anywhere in their column at or below them, so canopy
    cannot float above a roof.

    Plan-view mode (``plan_view=True``): ``y`` is the second horizontal
    (northing) axis, as for rasterised map footprints. Plantable and
    pedestrian zones are then all open interior space, and solar heating
    is applied to open cells adjacent to buildings (sunlit pavement).
    Use with ``SimConfig(g_buoy=0.0)`` since buoyancy acts along ``+y``.
    """
    building = (np.asarray(building_mask, dtype=np.float64) > 0.5).astype(np.float64)
    nx, ny = building.shape
    y = np.arange(ny)[None, :]

    fluid = 1.0 - building

    # Solar: exposed street layer + fluid cells adjacent (4-neighbourhood)
    # to any building cell (roofs / windward / leeward walls).
    adj = np.zeros_like(building)
    adj[1:, :] += building[:-1, :]
    adj[:-1, :] += building[1:, :]
    adj[:, 1:] += building[:, :-1]
    adj[:, :-1] += building[:, 1:]
    wall_solar = (adj > 0).astype(np.float64) * fluid
    if plan_view:
        solar = wall_solar
        interior = np.ones_like(building)
        interior[:2, :] = 0.0
        interior[-2:, :] = 0.0
        interior[:, :1] = 0.0
        interior[:, -1:] = 0.0
        plantable = interior * fluid
        pedestrian = interior * fluid
    else:
        ground = (y == 0).astype(np.float64) * fluid  # exposed street cells
        street_solar = ((y >= 1) & (y <= 2)).astype(np.float64) * fluid
        solar = np.clip(street_solar + wall_solar, 0.0, 1.0) * (1.0 - ground)

        # Ground-rooted: no building cell in the column at or below y.
        rooted = (np.cumsum(building, axis=1) == 0).astype(np.float64)
        band_h = max(4, ny // 3)
        plantable = ((y >= 1) & (y < band_h)).astype(np.float64) * fluid * rooted
        plantable[:2, :] = 0.0
        plantable[-2:, :] = 0.0

        # Pedestrian: street-level walkways at y in [1, 6] (scaled).
        ped_h = max(3, min(7, ny // 8 + 2))
        pedestrian = ((y >= 1) & (y < ped_h)).astype(np.float64) * fluid * rooted
        pedestrian[:2, :] = 0.0
        pedestrian[-2:, :] = 0.0

    return UrbanDomain(
        building_mask=jnp.asarray(building),
        solar_mask=jnp.asarray(solar),
        plantable_mask=jnp.asarray(plantable),
        pedestrian_mask=jnp.asarray(pedestrian),
    )


def create_urban_canyon_domain(nx=120, ny=60) -> UrbanDomain:
    """Build a canyon with 3 buildings of varying height on the street."""
    nx, ny = _resolve_nx_ny(nx, ny)
    # Building footprints as fractions of nx, heights as fractions of ny.
    specs = [(0.16, 0.30, 0.30), (0.42, 0.54, 0.45), (0.68, 0.80, 0.24)]
    building = np.zeros((nx, ny), dtype=np.float64)
    for fx0, fx1, fh in specs:
        x0, x1 = int(fx0 * nx), max(int(fx1 * nx), int(fx0 * nx) + 2)
        h = max(int(fh * ny), 3)
        building[x0:x1, 0:h] = 1.0
    return masks_from_building_mask(building)
