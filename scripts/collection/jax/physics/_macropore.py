import pnmlib.models as mods
import models.jax as models

macropore = {
    "throat.mass_conductance": {
        "model": mods.physics.diffusive_conductance.generic_diffusive,
        "pore_diffusivity": "pore.diffusivity",
        "throat_diffusivity": "throat.diffusivity",
        "size_factors": "throat.diffusive_size_factors",
    },
    "throat.hydraulic_conductance": {
        "model": mods.misc.constant,
        "value": 2.45436926e-66,
    },
    "pore.conductivity": {
        "model": mods.misc.conductivity,
        "concentration": "pore.concentration_old",
        "diffusivity": "pore.diffusivity",
        "temperature": "pore.temperature",
    },
    "throat.conductivity": {
        "model": mods.misc.conductivity,
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
