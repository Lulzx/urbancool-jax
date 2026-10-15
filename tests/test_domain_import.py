"""Ground-rooted plantable zones and offline footprint rasterisation."""
import jax.numpy as jnp
import numpy as np
import pytest

from urbancool.grid import create_urban_canyon_domain
from urbancool.osm import domain_from_polygons, rasterize_polygons


def test_plantable_is_ground_rooted_not_on_roofs():
    for nx, ny in [(120, 60), (80, 40), (32, 20)]:
        domain = create_urban_canyon_domain(nx, ny)
        building = np.asarray(domain.building_mask)
        plantable = np.asarray(domain.plantable_mask)
        pedestrian = np.asarray(domain.pedestrian_mask)
        # No building cell anywhere in the column at or below the cell.
        blocked = np.cumsum(building, axis=1) > 0
        assert not np.any(plantable * blocked), (nx, ny)
        assert not np.any(pedestrian * blocked), (nx, ny)
        assert not np.any(plantable * building)
        assert float(np.sum(plantable)) > 0


def test_rasterize_axis_aligned_square():
    square = [(0.26, 0.26), (0.74, 0.26), (0.74, 0.74), (0.26, 0.74)]
    mask = rasterize_polygons([square], nx=10, ny=10, bounds=(0, 0, 1, 1))
    assert mask.shape == (10, 10)
    assert mask[5, 5] == 1.0          # centre inside
    assert mask[0, 0] == 0.0 and mask[9, 9] == 0.0
    assert mask.sum() == 16           # centres 0.35..0.65 in both axes -> 4x4


def test_domain_from_polygons_plan_view_masks_are_sane():
    square = [(30, 20), (60, 20), (60, 40), (30, 40)]
    domain = domain_from_polygons([square], nx=40, ny=24, bounds=(0, 0, 120, 60))
    building = np.asarray(domain.building_mask)
    for other in (domain.solar_mask, domain.plantable_mask,
                  domain.pedestrian_mask):
        arr = np.asarray(other)
        assert not np.any(arr * building)          # disjoint from buildings
        assert arr.sum() > 0                        # non-empty zones
    assert building.sum() > 0
    # A short rollout on the imported domain must stay finite (plan-view
    # runs use g_buoy=0 since buoyancy acts along +y = northing here).
    from urbancool.physics import SimConfig
    from urbancool.solver import rollout_from_gamma
    out = rollout_from_gamma(jnp.zeros((40, 24)), SimConfig(g_buoy=0.0),
                             domain, burn_in_steps=10, outer_steps=2,
                             inner_steps=5)
    assert bool(jnp.all(jnp.isfinite(out["T_mean"])))
    assert "u_x_mean" in out and "u_y_mean" in out


def test_rasterize_rejects_empty_input():
    with pytest.raises(ValueError):
        rasterize_polygons([], nx=8, ny=8)
    with pytest.raises(ValueError):
        rasterize_polygons([[(0, 0), (1, 0), (1, 0)]], nx=8, ny=8,
                           bounds=(0, 0, 0, 1))
