"""UrbanCool-JAX: differentiable coupled fluid-thermal LBM + inverse design."""
from urbancool.lattice import E_D2Q9, W_D2Q9, OPP_D2Q9, E_D2Q5, W_D2Q5, OPP_D2Q5
from urbancool.grid import UrbanDomain, create_urban_canyon_domain
from urbancool.canopy import project_canopy_density

__version__ = "0.1.0"
__all__ = [
    "E_D2Q9", "W_D2Q9", "OPP_D2Q9", "E_D2Q5", "W_D2Q5", "OPP_D2Q5",
    "UrbanDomain", "create_urban_canyon_domain", "project_canopy_density",
]
