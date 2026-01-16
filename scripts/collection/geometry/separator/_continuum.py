import openpnm.models as mods
import models

continuum = {
    "throat.diffusive_size_factors": {
        "model": models.diffusive_size_factors.continuum,
        "throat_diameter": "throat.diameter",
    },
    "throat.hydraulic_size_factors": {
        "model": models.hydraulic_size_factors.continuum,
        "throat_diameter": "throat.diameter",
    },
    "throat.length": {
        "model": models.throat_length.continuum,
        "pore_diameter": "pore.diameter",
        "throat_diameter": "throat.diameter",
    },
    "throat.volume": {
        "model": models.throat_volume.continuum,
        "throat_diameter": "throat.diameter",
        "throat_length": "throat.length",
    },
    "pore.volume": {
        "model": mods.misc.constant,
        "value": 1e-32,
    },
}

