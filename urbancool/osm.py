"""Real-world domain import: rasterise building footprints into a domain.

The solver is a 2D model. Map footprints are *plan view* (top-down), so a
domain built here treats the ``y`` axis as northing rather than height:
run it with ``SimConfig(g_buoy=0.0)`` (buoyancy acts along ``+y``) and
interpret ``T`` as a plan-view thermal plume approximation.

``osmnx`` is an optional dependency (``pip install "urbancool-jax[osm]"``)
and is only imported inside :func:`domain_from_osm`. The rasteriser itself
needs only numpy + matplotlib and is unit-tested offline.
"""
import numpy as np

from urbancool.grid import UrbanDomain, masks_from_building_mask

__all__ = ["UrbanDomain", "rasterize_polygons", "domain_from_polygons",
           "domain_from_osm"]


def rasterize_polygons(polygons, nx, ny, bounds=None):
    """Rasterise polygons (sequences of ``(x, y)`` vertices) to a mask.

    Returns a float array of shape ``(nx, ny)`` with 1.0 where a cell
    centre falls inside any polygon. ``bounds`` is
    ``(xmin, ymin, xmax, ymax)``; when omitted it is the polygons' extent.
    """
    from matplotlib.path import Path

    polys = [np.asarray(p, dtype=np.float64) for p in polygons if len(p) >= 3]
    if not polys:
        raise ValueError("no polygons with >= 3 vertices to rasterise")
    if bounds is None:
        all_pts = np.concatenate(polys)
        bounds = (all_pts[:, 0].min(), all_pts[:, 1].min(),
                  all_pts[:, 0].max(), all_pts[:, 1].max())
    xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"degenerate bounds {bounds}")
    xs = xmin + (np.arange(nx) + 0.5) * (xmax - xmin) / nx
    ys = ymin + (np.arange(ny) + 0.5) * (ymax - ymin) / ny
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel()], axis=1)
    mask = np.zeros(nx * ny, dtype=bool)
    for poly in polys:
        mask |= Path(poly).contains_points(centres)
    return mask.reshape(nx, ny).astype(np.float64)


def domain_from_polygons(polygons, nx=120, ny=60, bounds=None) -> UrbanDomain:
    """Build a plan-view :class:`UrbanDomain` from footprint polygons."""
    building = rasterize_polygons(polygons, nx, ny, bounds)
    if building.sum() == 0:
        raise ValueError("polygons rasterised to an empty building mask; "
                         "check bounds / resolution")
    return masks_from_building_mask(building, plan_view=True)


def _polygon_exteriors(geometry):
    """Yield exterior coordinate rings from shapely (Multi)Polygons."""
    geoms = getattr(geometry, "geoms", [geometry])
    for geom in geoms:
        exterior = getattr(geom, "exterior", None)
        if exterior is not None:
            yield np.asarray(exterior.coords, dtype=np.float64)


def domain_from_osm(lat, lon, dist=300.0, nx=120, ny=60) -> UrbanDomain:
    """Build a plan-view domain from OpenStreetMap buildings near a point.

    Fetches building footprints within ``dist`` metres of ``(lat, lon)``
    via osmnx, projects them to a local metre CRS, and rasterises them.
    Requires network access and the optional ``osmnx`` dependency.
    """
    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "domain_from_osm requires osmnx: pip install \"urbancool-jax[osm]\""
        ) from exc
    gdf = ox.features_from_point((float(lat), float(lon)),
                                 tags={"building": True}, dist=float(dist))
    if gdf is None or len(gdf) == 0:
        raise ValueError(f"no OSM buildings within {dist} m of ({lat}, {lon})")
    gdf = ox.project_gdf(gdf)
    polygons = []
    for geometry in gdf.geometry:
        polygons.extend(_polygon_exteriors(geometry))
    if not polygons:
        raise ValueError("OSM returned buildings but no polygon footprints")
    xmin, ymin, xmax, ymax = (float(v) for v in gdf.total_bounds)
    return domain_from_polygons(polygons, nx=nx, ny=ny,
                                 bounds=(xmin, ymin, xmax, ymax))
