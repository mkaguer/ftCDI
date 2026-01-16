import openpnm.models as mods
import models

cylinders_and_trapezoidals = {
    "throat.max_size": {
        "model": mods.misc.from_neighbor_pores,
        "prop": "pore.diameter",
        "mode": "min",
    },
    "throat.diameter": {
        "model": mods.misc.scaled,
        "factor": 1.0,
        "prop": "throat.max_size",
    },
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.cylinders_and_trapezoidals,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.no_flow,
    },
}
