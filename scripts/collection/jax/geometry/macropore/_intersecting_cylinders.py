import pnmlib.models as mods
import models.jax as models

intersecting_cylinders = {
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.intersecting_cylinders,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.intersecting_cylinders,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "pore.volume": {
        "model": models.pore_volume.cylinder,
        "pore_diameter": "pore.diameter",
        "pore_thickness": "pore.thickness",
    },
    "throat.volume": {
        "model": mods.misc.constant,
        "value": 1e-32,
    }
}