import models.jax as models
import pnmlib.models as mods

spheres_and_cylinders = {
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.spheres_and_cylinders,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.spheres_and_cylinders,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.length": {
        "model": models.throat_length.spheres_and_cylinders,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.total_volume": {
        "model": mods.geometry.throat_volume.cylinder,
        "throat_diameter": "throat.diameter",
        "throat_length": "throat.length",
    },
    "throat.lens_volume": {
        "model": models.throat_volume.lens,
        "throat_diameter": "throat.diameter",
        "pore_diameter": "pore.diameter",
    },
    "throat.volume": {
        "model": mods.misc.difference,
        "props": ["throat.total_volume", "throat.lens_volume"],
    },
    "pore.volume": {
        "model": mods.geometry.pore_volume.sphere,
        "pore_diameter": "pore.diameter",
    },
}
