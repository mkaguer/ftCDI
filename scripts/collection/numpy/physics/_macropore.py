import openpnm.models as mods
import models.numpy as models

macropore = {
    "throat.diffusive_conductance": {
        "model": mods.physics.diffusive_conductance.generic_diffusive,
        "pore_diffusivity": "pore.diffusivity",
        "throat_diffusivity": "throat.diffusivity",
        "size_factors": "throat.diffusive_size_factors",
    },
    "throat.hydraulic_conductance": {
        "model": mods.misc.constant,
        "value": 2.45436926e-66,
    },
    "throat.mass_conductance": {
        "model": models.misc.reshape,
        "prop": "throat.diffusive_conductance",
    },
    "pore.conductivity": {
        "model": models.misc.conductivity,
        "concentration": "pore.concentration_old",
        "diffusivity": "pore.diffusivity",
        "temperature": "pore.temperature",
    },
    "throat.conductivity": {
        "model": models.misc.conductivity,
        "concentration": "throat.concentration_old",
        "diffusivity": "throat.diffusivity",
        "temperature": "throat.temperature",
    },
    "throat.ionic_conductance": {
        "model": mods.physics.diffusive_conductance.generic_diffusive,
        "pore_diffusivity": "pore.conductivity",
        "throat_diffusivity": "throat.conductivity",
        "size_factors": "throat.diffusive_size_factors",
    },
}
