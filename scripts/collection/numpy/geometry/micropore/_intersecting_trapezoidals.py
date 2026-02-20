import models.numpy as models

intersecting_trapezoidals = {
    "throat.diameter": {
        "model": models.throat_diameter.intersecting_trapezoidals,
        "pore_diameter": "pore.diameter",
    },
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.intersecting_trapezoidals,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
        "throat_thickness": "throat.thickness",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.no_flow,
    },
    "pore.volume": {
        "model": models.pore_volume.intersecting_trapezoidals,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
        "throat_thickness": "throat.thickness",
    },
}
