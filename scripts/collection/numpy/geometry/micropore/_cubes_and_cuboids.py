import openpnm.models as mods
import models.numpy as models

cubes_and_cuboids = {
    "throat.max_size": {
        "model": mods.misc._neighbor_lookups.from_neighbor_pores,
        "prop": "pore.diameter",
        "mode": "max",
    },
    "throat.diameter": {
        "model": mods.misc._basic_math.scaled,
        "factor": 1.0,
        "prop": "throat.max_size",
    },
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.cubes_and_cuboids,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.cubes_and_cuboids,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.length": {
        "model": mods.geometry.throat_length.cubes_and_cuboids,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.volume": {
        "model": mods.geometry.throat_volume.cuboid,
        "throat_diameter": "throat.diameter",
        "throat_length": "throat.length",
    },
    "pore.volume": {
        "model": mods.geometry.pore_volume.cube,
        "pore_diameter": "pore.diameter",
    },
}